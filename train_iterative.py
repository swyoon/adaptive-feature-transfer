#!/usr/bin/env python3
"""
Iterative training script that performs the following steps repeatedly:
1. Train student model using AFT knowledge distillation
2. Generate synthetic data using the trained model
3. Extract teacher features for both original and synthetic data
4. Repeat the process with the updated synthetic dataset

This version runs everything in a single process to maintain optimizer/scheduler state.
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
import time
import torch
import numpy as np
import random
from tqdm import tqdm
from PIL import Image
import yaml
import torchvision
import wandb

# Import your existing modules
from train import train
from data import get_dataset, get_loader, split_train, get_out_dim, FeatureDataset, SyntheticDataset, ConcatFeatureDataset
import utils as u
import models
from models import Concat, LinearModel, ProductLinearModel
from prior import UniformPrior, get_prior, get_btune_prior
from diffusion.edm.model import EDM
from diffusion.edm.fkd.fkd_rewards import DiversityModel
from generate_edm_fk_steering import AFTModule, do_aft_score
from schedulers import WarmupStableDecayScheduler


class IterativeTrainer:
    def __init__(self, args):
        self.args = args
        self.current_synthetic_dir = None
        self.current_synthetic_features = None
        
        # Track all synthetic data directories and features for cumulative training
        self.all_synthetic_dirs = []
        self.all_synthetic_features = []
        
        # Initialize model and optimizer states that persist across iterations
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.prior = None
        
        # Track global steps across all iterations for continuous wandb logging
        self.global_steps = 0
        
        # Initialize feature extraction model (reused across iterations)
        self.feature_model = None
        self.feature_get_transform = None
        self.feature_tokenizer = None
        self.feature_input_collate_fn = None
        
        # Initialize generation models (reused across iterations to save memory)
        self.aft_module = None
        self.edm_generator = None
        
        # Initialize wandb if enabled
        self.wandb_run = None
        if args.use_wandb:
            self.initialize_wandb()
        
        # Initialize directories
        self.create_directory(args.base_output_dir)
        
    def create_directory(self, path):
        """Create directory if it doesn't exist."""
        Path(path).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {path}")
    
    def initialize_wandb(self):
        """Initialize wandb for logging."""
        config = {
            'model_class': self.args.model_class,
            'pretrained_model': self.args.pretrained_model,
            'dataset': self.args.dataset,
            'method': self.args.method,
            'num_iterations': self.args.num_iterations,
            'steps_per_iteration': self.args.steps,
            'lr': self.args.lr,
            'prior_lr': self.args.prior_lr,
            'batch_size': self.args.batch_size,
            'optimizer': self.args.optimizer,
            'wd': self.args.wd,
            'prec': self.args.prec,
            'train_frac': self.args.train_frac,
            'seed': self.args.seed,
            'use_all_synthetic': self.args.use_all_synthetic,
            'num_target_images': self.args.num_target_images,
            'aft_score': self.args.aft_score,
        }
        
        # Set project name and run name
        project_name = self.args.wandb_project or f"aft-iterative-{self.args.dataset}"
        run_name = self.args.wandb_name or f"{self.args.model_class}_{self.args.dataset}_{self.args.num_iterations}iter"
        
        self.wandb_run = wandb.init(
            project=project_name,
            name=run_name,
            config=config,
            tags=[self.args.dataset, self.args.method, 'iterative'],
            notes=f"Iterative AFT training with {self.args.num_iterations} iterations"
        )
        
        print(f"Initialized wandb logging - Project: {project_name}, Run: {run_name}")
    
    def initialize_feature_extraction_model(self):
        """Initialize the feature extraction model once and reuse it."""
        if self.feature_model is None:
            print("Initializing feature extraction model...")
            self.feature_model, self.feature_get_transform, self.feature_tokenizer, self.feature_input_collate_fn = models.create_model(
                self.args.pretrained_model, out_dim=0, pretrained=True, extract_features=True
            )
            self.feature_model.eval()
            print(f"Feature extraction model ({self.args.pretrained_model}) initialized")
    
    def initialize_generation_models(self, model_ckpt, prior_ckpt):
        """Initialize the generation models once and reuse them."""
        print("Initializing/updating generation models...")
        
        # Always recreate AFT module with new checkpoints (it's relatively lightweight)
        # This ensures we always have the latest trained model and prior
        if self.aft_module is not None:
            del self.aft_module  # Clean up old AFT module
            torch.cuda.empty_cache()
        
        self.aft_module = AFTModule(
            model=self.args.model_class,
            model_out_dim=get_out_dim(self.args.dataset),
            pretrained_model=self.args.pretrained_model,
            prior_prec=self.args.prec,
            model_ckpt=model_ckpt,
            prior_ckpt=prior_ckpt,
        ).to('cuda')
        print("AFT module created with updated checkpoints")
        
        # Create EDM generator only once (it's memory-intensive and doesn't change)
        if self.edm_generator is None:
            DEVICE = 'cuda'
            config = f"""
                network_pkl: {self.args.edm_ckpt}
                batch_size: 1
                dtype: float16
                num_steps: 60
                S_churn: 40
            """
            config = yaml.safe_load(config)
            
            self.edm_generator = EDM(**config)
            self.edm_generator.to(DEVICE)
            self.edm_generator.eval()
            print("EDM generator initialized and moved to GPU")
        else:
            # Move existing EDM generator to GPU
            self.edm_generator = self.edm_generator.to('cuda')
            print("EDM generator moved to GPU")
    
    def cleanup_generation_models(self):
        """Move generation models to CPU to free GPU memory."""
        if self.aft_module is not None:
            self.aft_module = self.aft_module.cpu()
            print("AFT module moved to CPU")
        
        if self.edm_generator is not None:
            self.edm_generator = self.edm_generator.cpu()
            print("EDM generator moved to CPU")
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        print("GPU memory cache cleared")
    
    def initialize_training_components(self, iteration=0):
        """Initialize or update training components."""
        print("Initializing training components...")
        
        # Set seed
        u.set_seed(self.args.seed)
        
        # Get dataset info
        out_dim = get_out_dim(self.args.dataset)
        
        # Create model if first iteration
        if self.model is None:
            self.model, get_transform, tokenizer, input_collate_fn = models.create_model(
                self.args.model_class, out_dim=out_dim, pretrained=True
            )
            self.get_transform = get_transform
            self.tokenizer = tokenizer
            self.input_collate_fn = input_collate_fn
            
            # Move model to GPU
            self.model = self.model.cuda()
            print("Model moved to GPU")
        
        # Get datasets
        train_ds, test_ds = get_dataset(
            self.args.dataset, self.get_transform, self.tokenizer, 
            no_augment=True, cache=False
        ) # NOTE: no_augment?
        
        val_frac = 0.1 if self.args.use_val else 0
        train_ds, val_ds = split_train(train_ds, self.args.train_frac, val_frac)
        train_indices = train_ds.indices
        val_indices = val_ds.indices
        
        raw_train_ds, raw_test_ds = get_dataset(
            self.args.dataset, self.get_transform, self.tokenizer, 
            no_augment=True, cache=False
        )
        
        # Handle pretrained models and features
        if self.args.pretrained_models not in [None, "none"]:
            pretrained_models = self.args.pretrained_models.split(",")
            pretrained_models = [model.strip() for model in pretrained_models]
            
            if self.args.original_feature_path is not None:
                feature_paths = [self.args.original_feature_path.replace(pretrained_models[0], model) for model in pretrained_models]
            else:
                feature_paths = [f"features/{model}_{self.args.dataset}.pt" for model in pretrained_models]
            
            # Create feature datasets
            train_ds = FeatureDataset(raw_train_ds, feature_paths, indices=train_indices, split="train")
            val_ds = FeatureDataset(raw_train_ds, feature_paths, indices=val_indices, split="train")
            test_ds = FeatureDataset(raw_test_ds, feature_paths, indices=None, split="test")
        else:
            # Dummy feature dataset
            train_ds = FeatureDataset(raw_train_ds, None, indices=train_indices, split="train")
            val_ds = FeatureDataset(raw_train_ds, None, indices=val_indices, split="train")
            test_ds = FeatureDataset(raw_test_ds, None, indices=None, split="test")
        
        # Add synthetic dataset if available
        if self.args.use_all_synthetic and len(self.all_synthetic_dirs) > 0:
            print(f"Adding ALL synthetic datasets from {len(self.all_synthetic_dirs)} iterations...")
            synthetic_datasets = []
            
            for i, (syn_dir, syn_features) in enumerate(zip(self.all_synthetic_dirs, self.all_synthetic_features)):
                print(f"  - Iteration {i+1}: {syn_dir}")
                synthetic_raw_ds = SyntheticDataset(
                    syn_dir, self.args.class_file, None,
                    transform=self.get_transform(train=True)
                )
                
                if self.args.pretrained_models not in [None, "none"] and syn_features is not None:
                    synthetic_ds = FeatureDataset(
                        synthetic_raw_ds, [syn_features], 
                        indices=None, split="train"
                    )
                else:
                    synthetic_ds = FeatureDataset(synthetic_raw_ds, None, indices=None, split="train")
                
                synthetic_datasets.append(synthetic_ds)
            
            # Combine all synthetic datasets
            if len(synthetic_datasets) > 0:
                combined_synthetic_ds = synthetic_datasets[0]
                for syn_ds in synthetic_datasets[1:]:
                    combined_synthetic_ds = ConcatFeatureDataset(combined_synthetic_ds, syn_ds)
                
                train_ds = ConcatFeatureDataset(train_ds, combined_synthetic_ds)
                total_synthetic_size = sum([len(ds) for ds in synthetic_datasets])
                print(f"Combined training set size: {len(train_ds)} (original: {len(train_ds.datasets[0])}, synthetic: {total_synthetic_size})")
                
        elif self.current_synthetic_dir is not None:
            print(f"Adding CURRENT synthetic dataset from: {self.current_synthetic_dir}")
            synthetic_raw_ds = SyntheticDataset(
                self.current_synthetic_dir, self.args.class_file, None,
                transform=self.get_transform(train=True)
            )
            
            if self.args.pretrained_models not in [None, "none"] and self.current_synthetic_features is not None:
                synthetic_ds = FeatureDataset(
                    synthetic_raw_ds, [self.current_synthetic_features], 
                    indices=None, split="train"
                )
            else:
                synthetic_ds = FeatureDataset(synthetic_raw_ds, None, indices=None, split="train")
            
            train_ds = ConcatFeatureDataset(train_ds, synthetic_ds)
            print(f"Combined training set size: {len(train_ds)} (original: {len(train_ds.datasets[0])}, synthetic: {sum(train_ds.lengths[1:])})")
        
        # Create data loaders
        num_workers = 0
        if self.args.use_val:
            loaders = [
                get_loader(ds, self.args.batch_size, num_workers=num_workers, 
                          shuffle=(ds is train_ds), input_collate_fn=self.input_collate_fn)
                for ds in [train_ds, val_ds]
            ]
        else:
            loaders = [
                get_loader(ds, self.args.batch_size, num_workers=num_workers,
                          shuffle=(ds is train_ds), input_collate_fn=self.input_collate_fn)
                for ds in [train_ds, test_ds]
            ]
        
        # Initialize prior if first iteration
        if self.prior is None:
            if self.args.method == "aft":
                self.prior = get_prior(
                    self.model.feat_dim, train_ds, self.args.prec, 
                    learn_scales=False, tensor_product=(self.args.dataset == "snli-ve"),
                    prior_type="kernel"
                )
            else:
                self.prior = UniformPrior()
            
            # Move prior to GPU if it's a PyTorch module
            if isinstance(self.prior, torch.nn.Module):
                self.prior = self.prior.cuda()
                print("Prior moved to GPU")
        
        return loaders, train_ds
    
    def train_iteration(self, iteration, loaders):
        """Train the model for one iteration."""
        print(f"\n[ITERATION {iteration + 1}] Training student model...")
        print(f"Starting from global step: {self.global_steps}")
        
        DEVICE="cuda"
        # Move model to GPU for training
        self.model = self.model.to(DEVICE)
        if isinstance(self.prior, torch.nn.Module):
            self.prior = self.prior.to(DEVICE)
        print("Model and prior moved to GPU for training")
        
        # Initialize optimizer if first iteration or if it doesn't exist
        if self.optimizer is None:
            # Collect model parameters
            model_params = [p for p in self.model.parameters()]
            param_groups = [{'params': model_params, 'lr': self.args.lr, 'weight_decay': self.args.wd}]
            
            # Add prior parameters if they exist
            if hasattr(self.prior, "parameters") and list(self.prior.parameters()):
                prior_params = [p for p in self.prior.parameters()]
                param_groups.append({'params': prior_params, 'lr': self.args.prior_lr, 'weight_decay': 0})
            
            # Create single optimizer with multiple parameter groups
            if self.args.optimizer == "sgd":
                self.optimizer = torch.optim.SGD(param_groups, momentum=0.9)
            else:
                self.optimizer = torch.optim.Adam(param_groups)
            
            if self.args.scheduler == 'warmup_stable_decay':
                self.scheduler = WarmupStableDecayScheduler(
                    optimizer=self.optimizer,
                    warmup_steps=self.args.warmup_steps,
                    stable_steps=self.args.stable_steps,
                    decay_steps=self.args.decay_steps,
                )
            else:  # cosine_annealing
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=total_steps
                )
        
        # Training hyperparameters
        hypers = {
            "lr": self.args.lr,
            "steps": self.args.steps,
            "eval_steps": self.args.eval_steps,
            "wd": self.args.wd,
            "prior_lr": self.args.prior_lr,
            "prior_pretrain_steps": 0,
            "prior_freq": self.args.prior_freq,
            "step_offset": self.global_steps,  # Pass current global step count
        }
        
        hypers["optimizer"] = self.optimizer
        hypers["scheduler"] = self.scheduler
        # Train with existing optimizer
        last_acc, best_acc = train(
            loaders, self.model, None, self.prior, self.wandb_run, hypers,
        )
        
        # Update global step counter after training
        self.global_steps += self.args.steps
        print(f"Updated global steps to: {self.global_steps}")
        
        # Move model to CPU after training to free GPU memory
        self.model = self.model.cpu()
        if isinstance(self.prior, torch.nn.Module):
            self.prior = self.prior.cpu()
        torch.cuda.empty_cache()
        print("Model and prior moved to CPU, GPU cache cleared")
        
        return last_acc, best_acc
    
    def generate_synthetic_data(self, iteration, ckpt_dir, synthetic_dir):
        """Generate synthetic data using the trained model."""
        print(f"\n[ITERATION {iteration + 1}] Generating synthetic data...")
        
        model_ckpt = os.path.join(ckpt_dir, "model.pt")
        prior_ckpt = os.path.join(ckpt_dir, "prior.pt")
        
        # Initialize or reuse generation models
        self.initialize_generation_models(model_ckpt, prior_ckpt)
        
        # Load class names
        with open(self.args.class_file, "r") as f:
            class_names = [line.strip() for line in f.readlines()]
        
        # Generate synthetic data
        self._generate_with_edm(synthetic_dir, class_names)
        
        # Move generation models to CPU after generation
        self.cleanup_generation_models()
        
        # Log GPU memory usage after generation
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            print(f"GPU memory after generation - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB")
    
    def _generate_with_edm(self, save_dir, class_names):
        """Generate synthetic data using EDM + FK steering."""
        DEVICE = 'cuda'

        # Initialize feature pools
        init_feature_pool_model = torch.empty((0, 2048)).to(DEVICE)
        init_feature_pool_pretrained = torch.empty((0, 1536)).to(DEVICE)
        init_target_pool = torch.empty((0,), dtype=torch.int64).to(DEVICE)
        
        if self.args.use_downstream:
            train_dataset = get_dataset("flowers", lambda train: lambda x: torchvision.transforms.ToTensor()(x), None, no_augment=True, cache=False)[0]
            train_loader = get_loader(train_dataset, 1, num_workers=4, shuffle=False, input_collate_fn=self.aft_module.input_collate_fn)

            for data in tqdm(train_loader, desc="Loading downstream features"):
                if isinstance(data, (tuple, list)):
                    inputs, labels = data
                else:
                    inputs = data
                
                inputs = inputs.to(DEVICE)
                feature_model = self.aft_module.get_model_feature(inputs)
                feature_pretrained = self.aft_module.get_pretrained_feature(inputs)
                
                init_feature_pool_model = torch.cat([init_feature_pool_model, feature_model], dim=0)
                init_feature_pool_pretrained = torch.cat([init_feature_pool_pretrained, feature_pretrained], dim=0)
                init_target_pool = torch.cat([init_target_pool, labels.to(DEVICE)], dim=0)
        
        FKD_ARGS = {
            "potential_type": "diff",
            "lmbda": 1.0,
            "num_particles": 4,
            "adaptive_resampling": True,
            "resample_frequency": 5,
            "resampling_t_start": 0,
            "resampling_t_end": 60,
            "time_steps": 60,
            "latent_to_decode_fn": lambda x: torch.clamp(x, -1, 1) * 0.5 + 0.5,
            "use_smc": True if not self.args.no_steering else False,
            "output_dir": "./outputs/generated/fkd_results",
            "print_rewards": False,
            "visualize_intermediate": False,
            "visualize_x0": False,
        }

        reward_fn = do_aft_score

        reward_fn_args = {
            "feature_pool_model": init_feature_pool_model,
            "feature_pool_pretrained": init_feature_pool_pretrained,
            "target_pool": init_target_pool,
            "aft_module": self.aft_module,
            "score": self.args.aft_score,
        }
        
        # # Set seeds
        # torch.manual_seed(self.args.seed)
        # np.random.seed(self.args.seed)
        # random.seed(self.args.seed)
        
        image_ind = 0
        for _ in tqdm(range(int(np.ceil(self.args.num_target_images / self.edm_generator.batch_size))), 
                     desc="Generating synthetic images"):
            with torch.autocast(device_type=next(iter(self.edm_generator.parameters())).device.type, dtype=torch.float16):
                result = self.edm_generator.sample_fk_steering(fkd_args=FKD_ARGS, reward_fn=reward_fn, reward_fn_args=reward_fn_args)
            
            images = result[0]
            labels = result[1].cpu().numpy().tolist()
            
            images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            
            for image_np, label in zip(images_np, labels):
                image_path = os.path.join(save_dir, class_names[label], f"{image_ind:06d}.png")
                os.makedirs(os.path.dirname(image_path), exist_ok=True)
                Image.fromarray(image_np, "RGB").save(image_path)
                image_ind += 1
            
            # Update feature pools
            images = torch.clamp(images, -1., 1.) * 0.5 + 0.5
            features_model = self.aft_module.get_model_feature(images)
            features_pretrained = self.aft_module.get_pretrained_feature(images)
            
            reward_fn_args["feature_pool_model"] = torch.cat([reward_fn_args["feature_pool_model"], features_model], dim=0)
            reward_fn_args["feature_pool_pretrained"] = torch.cat([reward_fn_args["feature_pool_pretrained"], features_pretrained], dim=0)
    
    def extract_synthetic_features(self, iteration, synthetic_dir, features_dir):
        """Extract teacher features for synthetic data."""
        print(f"\n[ITERATION {iteration + 1}] Extracting synthetic data features...")
        
        synthetic_features_path = os.path.join(features_dir, f"{self.args.pretrained_model}_synthetic_iter{iteration + 1}.pt")
        
        # Use shared feature extraction model
        self.initialize_feature_extraction_model()
        
        # Move feature model to GPU for extraction
        self.feature_model = self.feature_model.cuda()
        print("Feature extraction model moved to GPU")
        
        # Create synthetic dataset
        synthetic_ds = SyntheticDataset(
            synthetic_dir, self.args.class_file, None,
            transform=self.feature_get_transform(train=True)
        )
        
        synthetic_loader = get_loader(
            synthetic_ds, self.args.batch_size, num_workers=0, 
            shuffle=False, input_collate_fn=self.feature_input_collate_fn
        )
        
        # Extract features
        features = []
        with torch.no_grad():
            for batch in tqdm(synthetic_loader, desc="Extracting features"):
                if isinstance(batch, tuple) or isinstance(batch, list):
                    inputs = batch[0]
                else:
                    inputs = batch
                    if 'labels' in inputs:
                        inputs.pop('labels')
                
                if hasattr(inputs, 'items'):
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                else:
                    inputs = inputs.cuda()
                
                feat = self.feature_model(inputs).detach().cpu()
                features.append(feat)
        
        # Move feature model to CPU after extraction
        self.feature_model = self.feature_model.cpu()
        torch.cuda.empty_cache()
        print("Feature extraction model moved to CPU, GPU cache cleared")
        
        features = torch.cat(features, dim=0)
        print(f'Synthetic features shape: {features.size()}')
        
        # # Log feature extraction metrics to wandb
        # if self.wandb_run:
        #     self.wandb_run.log({
        #         f'synthetic_features_iter_{iteration + 1}_count': len(features),
        #         f'synthetic_features_iter_{iteration + 1}_dim': features.size(1),
        #     }, step=self.global_steps)
        
        feature_dict = {'train': features}
        os.makedirs(os.path.dirname(synthetic_features_path), exist_ok=True)
        torch.save(feature_dict, synthetic_features_path)
        print(f'Saved synthetic features to {synthetic_features_path}')
        
        return synthetic_features_path
    
    def extract_original_features(self, features_dir):
        """Extract teacher features for original data (first iteration only)."""
        print("Extracting original data features...")
        
        original_features_path = os.path.join(features_dir, f"{self.args.pretrained_model}_{self.args.dataset}_original.pt")
        
        # Use shared feature extraction model  
        self.initialize_feature_extraction_model()
        
        # Move feature model to GPU for extraction
        self.feature_model = self.feature_model.cuda()
        print("Feature extraction model moved to GPU")
        
        # Get datasets
        train_test_dataset = get_dataset(self.args.dataset, self.feature_get_transform, self.feature_tokenizer, no_augment=True)
        train_test_loaders = [
            get_loader(ds, self.args.batch_size, num_workers=0, shuffle=False, input_collate_fn=self.feature_input_collate_fn)
            for ds in train_test_dataset
        ]
        train_loader, test_loader = train_test_loaders
        
        # Extract train features
        train_features = []
        with torch.no_grad():
            for batch in tqdm(train_loader, desc="Extracting train features"):
                if isinstance(batch, tuple) or isinstance(batch, list):
                    inputs = batch[0]
                else:
                    inputs = batch
                    if 'labels' in inputs:
                        inputs.pop('labels')
                
                if hasattr(inputs, 'items'):
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                else:
                    inputs = inputs.cuda()
                
                feat = self.feature_model(inputs).detach().cpu()
                train_features.append(feat)
        
        # Extract test features
        test_features = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Extracting test features"):
                if isinstance(batch, tuple) or isinstance(batch, list):
                    inputs = batch[0]
                else:
                    inputs = batch
                    if 'labels' in inputs:
                        inputs.pop('labels')
                
                if hasattr(inputs, 'items'):
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                else:
                    inputs = inputs.cuda()
                
                feat = self.feature_model(inputs).detach().cpu()
                test_features.append(feat)
        
        # Move feature model to CPU after extraction
        self.feature_model = self.feature_model.cpu()
        torch.cuda.empty_cache()
        print("Feature extraction model moved to CPU, GPU cache cleared")
        
        train_features = torch.cat(train_features, dim=0)
        test_features = torch.cat(test_features, dim=0)
        
        print(f'Original features shape - Train: {train_features.size()}, Test: {test_features.size()}')
        
        # # Log original feature extraction metrics to wandb
        # if self.wandb_run:
        #     self.wandb_run.log({
        #         'original_train_features_count': len(train_features),
        #         'original_test_features_count': len(test_features),
        #         'original_features_dim': train_features.size(1),
        #     }, step=0)  # Log at step 0 since this happens before any training
        
        feature_dict = {'train': train_features, 'test': test_features}
        os.makedirs(os.path.dirname(original_features_path), exist_ok=True)
        torch.save(feature_dict, original_features_path)
        print(f'Saved original features to {original_features_path}')
        
        return original_features_path


    def run_iterations(self):
        """Run the complete iterative training process."""
        for iteration in range(self.args.num_iterations):
            print(f"\n{'#'*80}")
            print(f"STARTING ITERATION {iteration + 1}/{self.args.num_iterations}")
            print(f"{'#'*80}")
            
            # Define paths for current iteration
            iter_dir = os.path.join(self.args.base_output_dir, f"iteration_{iteration + 1}")
            ckpt_dir = os.path.join(iter_dir, "checkpoints")
            synthetic_dir = os.path.join(iter_dir, "synthetic_images")
            features_dir = os.path.join(iter_dir, "features")
            
            self.create_directory(iter_dir)
            self.create_directory(ckpt_dir)
            self.create_directory(synthetic_dir)
            self.create_directory(features_dir)
            
            # Step 1: Extract features for original data (first iteration only)
            if iteration == 0 and self.args.original_feature_path is None:
                original_features_path = self.extract_original_features(features_dir)
                self.args.original_feature_path = original_features_path
            
            # Step 2: Extract features for synthetic data from previous iteration (if available)
            if self.current_synthetic_dir is not None:
                print(f"\n[ITERATION {iteration + 1}] Step 1: Extracting features from previous synthetic data...")
                synthetic_features_path = self.extract_synthetic_features(iteration-1, self.current_synthetic_dir, features_dir)
                self.current_synthetic_features = synthetic_features_path
            
            if hasattr(self, 'current_synthetic_features') and self.current_synthetic_features:
                self.all_synthetic_features.append(self.current_synthetic_features)

            # Step 3: Initialize/update training components
            loaders, train_ds = self.initialize_training_components(iteration)
            
            # Step 4: Train student model
            last_acc, best_acc = self.train_iteration(iteration, loaders)
            
            # Log iteration results to wandb
            if self.wandb_run:
                self.wandb_run.log({
                    'iteration': iteration + 1,
                    'iteration_last_acc': last_acc,
                    'iteration_best_acc': best_acc,
                    'total_synthetic_dirs': len(self.all_synthetic_dirs),
                    'training_set_size': len(train_ds) if 'train_ds' in locals() else 0,
                    'global_steps': self.global_steps,
                }, step=self.global_steps)
            
            # Step 5: Save checkpoints
            print(f"Saving checkpoints to {ckpt_dir}")
            model_path = os.path.join(ckpt_dir, "model.pt")
            prior_path = os.path.join(ckpt_dir, "prior.pt")
            
            torch.save(self.model.state_dict(), model_path)
            if hasattr(self.prior, "state_dict"):
                torch.save(self.prior.state_dict(), prior_path)
            
            # # Save checkpoints as wandb artifacts
            # if self.wandb_run and self.args.wandb_save_checkpoints:
            #     artifact = wandb.Artifact(
            #         name=f"model_iter_{iteration + 1}",
            #         type="model",
            #         description=f"Model checkpoint from iteration {iteration + 1}"
            #     )
            #     artifact.add_file(model_path)
            #     if hasattr(self.prior, "state_dict"):
            #         artifact.add_file(prior_path)
            #     self.wandb_run.log_artifact(artifact)
            
            # Save results
            with open(os.path.join(ckpt_dir, "results.txt"), "w") as f:
                f.write(f"Last test acc: {last_acc:.3f}\n")
                f.write(f"Best test acc: {best_acc:.3f}\n")
            
            # Step 6: Generate synthetic data for next iteration
            self.generate_synthetic_data(iteration, ckpt_dir, synthetic_dir)
            
            # Update synthetic directory for next iteration
            self.current_synthetic_dir = synthetic_dir
            
            # Add to accumulated synthetic data lists for cumulative training option
            self.all_synthetic_dirs.append(synthetic_dir)
            
            # Log iteration completion
            iteration_log = os.path.join(iter_dir, "iteration_log.txt")
            with open(iteration_log, 'w') as f:
                f.write(f"Iteration {iteration + 1} completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Model checkpoint: {os.path.join(ckpt_dir, 'model.pt')}\n")
                f.write(f"Prior checkpoint: {os.path.join(ckpt_dir, 'prior.pt')}\n")
                f.write(f"Synthetic images: {synthetic_dir}\n")
                if hasattr(self, 'current_synthetic_features') and self.current_synthetic_features:
                    f.write(f"Synthetic features: {self.current_synthetic_features}\n")
                f.write(f"Last accuracy: {last_acc:.3f}\n")
                f.write(f"Best accuracy: {best_acc:.3f}\n")
                if self.args.original_feature_path:
                    f.write(f"Original features: {self.args.original_feature_path}\n")
            
            print(f"\n[ITERATION {iteration + 1}] Completed successfully!")
            print(f"Last accuracy: {last_acc:.3f}, Best accuracy: {best_acc:.3f}")
            print(f"Checkpoints saved to: {ckpt_dir}")
            print(f"Synthetic images saved to: {synthetic_dir}")
            if hasattr(self, 'current_synthetic_features') and self.current_synthetic_features:
                print(f"Synthetic features: {self.current_synthetic_features}")
        
        # Log final summary to wandb and finish the run
        if self.wandb_run:
            self.wandb_run.log({
                'total_iterations_completed': self.args.num_iterations,
                'final_last_acc': last_acc,
                'final_best_acc': best_acc,
                'total_synthetic_datasets': len(self.all_synthetic_dirs),
                'final_global_steps': self.global_steps,
            }, step=self.global_steps)
            
            # Finish the wandb run
            self.wandb_run.finish()
            print("Wandb run finished")
        
        print(f"\n{'#'*80}")
        print(f"ALL {self.args.num_iterations} ITERATIONS COMPLETED SUCCESSFULLY!")
        print(f"Results saved in: {self.args.base_output_dir}")
        if self.args.use_wandb:
            print(f"Wandb logging completed")
        print(f"{'#'*80}")


def main():
    parser = argparse.ArgumentParser(description="Iterative AFT training with synthetic data generation")
    
    # Training parameters
    parser.add_argument('--model_class', type=str, default='resnet50.a1_in1k', help='Student model class')
    parser.add_argument('--pretrained_model', type=str, default='vit_giant_patch14_dinov2.lvd142m', help='Teacher model class')
    parser.add_argument('--pretrained_models', type=str, default='vit_giant_patch14_dinov2.lvd142m', help='Teacher model class (alias)')
    parser.add_argument('--dataset', type=str, default='flowers', help='Dataset name')
    parser.add_argument('--train_frac', type=float, default=1.0, help='Fraction of training data to use')
    parser.add_argument('--use_val', action='store_true', help='Use validation set')
    parser.add_argument('--method', type=str, default='aft', help='Training method')
    parser.add_argument('--steps', type=int, default=5000, help='Training steps per iteration')
    parser.add_argument('--eval_steps', type=int, default=50, help='Evaluation frequency') # TODO: 1000
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--prec', type=int, default=10, help='Prior precision')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--optimizer', type=str, default='sgd', help='Optimizer type')
    parser.add_argument('--wd', type=float, default=0, help='Weight decay')
    parser.add_argument('--prior_lr', type=float, default=1e-2, help='Prior learning rate')
    parser.add_argument('--prior_freq', type=int, default=1, help='Prior update frequency')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    
    # Generation parameters
    parser.add_argument('--edm_ckpt', type=str, required=True, help='Path to EDM checkpoint')
    parser.add_argument('--aft_score', type=str, default='total', help='AFT score type')
    parser.add_argument('--num_target_images', type=int, default=3000, help='Number of synthetic images to generate')
    parser.add_argument('--use_downstream', action='store_true', help='Use downstream data for feature pool initialization')
    parser.add_argument('--no_steering', action='store_true', help='Disable FK steering (use unconditional generation)')

    # Iteration parameters
    parser.add_argument('--num_iterations', type=int, default=3, help='Number of iterations to perform')
    parser.add_argument('--base_output_dir', type=str, default='./iterative_training', help='Base output directory')
    parser.add_argument('--use_all_synthetic', action='store_true', help='Use all accumulated synthetic data from previous iterations (default: use only current synthetic data)')
    
    # File paths
    parser.add_argument('--class_file', type=str, default='./classes/flowers.txt', help='Path to class file')
    parser.add_argument('--original_feature_path', type=str, help='Path to original dataset features')
    
    # Wandb logging parameters
    parser.add_argument('--use_wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--wandb_project', type=str, help='Wandb project name (default: aft-iterative-{dataset})')
    parser.add_argument('--wandb_name', type=str, help='Wandb run name (default: {model}_{dataset}_{iterations}iter)')
    parser.add_argument('--wandb_save_checkpoints', action='store_true', help='Save model checkpoints as wandb artifacts')
        
    # Scheduler arguments
    parser.add_argument('--scheduler', type=str, default='cosine_annealing', choices=['cosine_annealing', 'warmup_stable_decay'],
                      help='Type of learning rate scheduler')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='Number of warmup steps for warmup_stable_decay scheduler')
    parser.add_argument('--stable_steps', type=int, default=5000, help='Number of stable steps for warmup_stable_decay scheduler')
    parser.add_argument('--decay_steps', type=int, default=1000, help='Number of decay steps for warmup_stable_decay scheduler')

    args = parser.parse_args()
    
    # Set pretrained_models if not provided but pretrained_model is
    if hasattr(args, 'pretrained_model') and not hasattr(args, 'pretrained_models'):
        args.pretrained_models = args.pretrained_model
    elif hasattr(args, 'pretrained_models') and not hasattr(args, 'pretrained_model'):
        args.pretrained_model = args.pretrained_models
    
    print("Starting iterative AFT training...")
    print(f"Configuration:")
    print(f"  Model: {args.model_class}")
    print(f"  Teacher: {args.pretrained_model}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Iterations: {args.num_iterations}")
    print(f"  Steps per iteration: {args.steps}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Output directory: {args.base_output_dir}")
    print(f"  Synthetic data mode: {'All accumulated' if args.use_all_synthetic else 'Current only'}")
    print(f"  Wandb logging: {'Yes' if args.use_wandb else 'No'}")
    if args.use_wandb:
        project_name = args.wandb_project or f"aft-iterative-{args.dataset}"
        run_name = args.wandb_name or f"{args.model_class}_{args.dataset}_{args.num_iterations}iter"
        print(f"    Project: {project_name}")
        print(f"    Run name: {run_name}")
    
    # Create and run iterative trainer
    trainer = IterativeTrainer(args)
    trainer.run_iterations()


if __name__ == "__main__":
    main()
