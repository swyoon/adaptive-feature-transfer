import os
import torch
import yaml
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import random
import torchvision

from diffusion.edm.model import EDM

from models import create_model
from prior import get_prior
from data import get_dataset, get_loader

class DummyFeatureDataset:
    feat_dims = [1536] # NOTE hard coded for vit_giant_patch14_dinov2.lvd142m
    num_features = 1536
    def __len__(self):
        return 1

class AFTModule(torch.nn.Module):
    def __init__(self, model, model_out_dim, pretrained_model, prior_prec, model_ckpt, prior_ckpt, **kwargs):
        super().__init__()
        self.model, get_transform, tokenizer, input_collate_fn = create_model(
        model, out_dim=model_out_dim, pretrained=False
        )
        model_state = torch.load(model_ckpt, map_location="cpu")
        self.model.load_state_dict(model_state, strict=False)
        self.model.eval()

        self.transform = get_transform(train=False)
        

        self.pretrained_model, get_transform_pretrained, tokenizer_pretrained, input_collate_fn_pretrained = create_model(pretrained_model, out_dim=0, pretrained=True, extract_features=True)
        self.pretrained_model.eval()

        self.transform_pretrained = get_transform_pretrained(train=False)

        ds = DummyFeatureDataset()
        train_ds = DummyFeatureDataset()

        self.prior = get_prior(
            self.model.feat_dim,
            ds,
            prior_prec,
            learn_scales=False,
            tensor_product=False, # NOTE: should be true if experimenting with dataset == "snli-ve",
            prior_type="kernel",
        )
        prior_state = torch.load(prior_ckpt, map_location="cpu")
        self.prior.load_state_dict(prior_state, strict=True)

    def get_ce_loss(self, x, y, reduction="mean"):
        x = self.transform(x)
        outputs = self.model(x)
        loss = torch.nn.functional.cross_entropy(outputs, y, reduction=reduction)
        return loss

    def get_aft_loss_batch(self, x):
        outputs, feats = self.model(self.transform(x), return_feat=True)

        feats_pretrained = self.pretrained_model(self.transform_pretrained(x))

        prior_loss = -self.prior.prec * self.prior.log_prob(feats, feats_pretrained)
        return prior_loss

    def get_total_loss_batch(self, x, y):
        outputs, feats = self.model(self.transform(x), return_feat=True)

        feats_pretrained = self.pretrained_model(self.transform_pretrained(x))
        
        ce_loss = torch.nn.functional.cross_entropy(outputs, y)

        prior_loss = - self.prior.prec * self.prior.log_prob(feats, feats_pretrained)
        return ce_loss + prior_loss

    def get_aft_loss(self, x, feature_pool_model, feature_pool_pretrained):
        outputs, feats = self.model(self.transform(x), return_feat=True)

        feats_pretrained = self.pretrained_model(self.transform_pretrained(x))

        new_feature_pool_model = torch.cat([feature_pool_model, feats], dim=0)
        new_feature_pool_pretrained = torch.cat([feature_pool_pretrained, feats_pretrained], dim=0)

        prior_loss = - self.prior.prec * self.prior.log_prob(new_feature_pool_model, new_feature_pool_pretrained)
        return prior_loss

    def get_total_loss(self, x, y, feature_pool_model, feature_pool_pretrained):
        outputs, feats = self.model(self.transform(x), return_feat=True)

        feats_pretrained = self.pretrained_model(self.transform_pretrained(x))
        
        new_feature_pool_model = torch.cat([feature_pool_model, feats], dim=0)
        new_feature_pool_pretrained = torch.cat([feature_pool_pretrained, feats_pretrained], dim=0)

        ce_loss = torch.nn.functional.cross_entropy(outputs, y)

        prior_loss = - self.prior.prec * self.prior.log_prob(new_feature_pool_model, new_feature_pool_pretrained)
        return ce_loss + prior_loss

    def _diversity(self, feature, feature_pool, method):
        if len(feature_pool) == 0:
            return 0.0
        if method == "cos_min":
            cos_sims = torch.nn.functional.cosine_similarity(feature_pool, feature.unsqueeze(0), dim=1)
            diversity_score = (1-cos_sims).min().item()
        elif method == "cos_mean":
            cos_sims = torch.nn.functional.cosine_similarity(feature_pool, feature.unsqueeze(0), dim=1)
            diversity_score = (1-cos_sims).mean().item()
        elif method == "l2_min":
            l2_dists = torch.norm(feature_pool - feature.unsqueeze(0), dim=1)
            diversity_score = l2_dists.min().item()
        elif method == "l2_mean":
            l2_dists = torch.norm(feature_pool - feature.unsqueeze(0), dim=1)
            diversity_score = l2_dists.mean().item()
        else:
            raise NotImplementedError(f"Unknown method {method}")
        return diversity_score

    def get_pretrained_diversity(self, x, feature_pool_pretrained, method):
        features = self.get_pretrained_feature(x)
        
        scaled_feature_pool_pretrained = self.prior(feature_pool_pretrained)
        scaled_features = self.prior(features)

        diversity_scores = []
        for scaled_feature in scaled_features:
            diversity_score = self._diversity(scaled_feature, scaled_feature_pool_pretrained, method)
            diversity_scores.append(diversity_score)      

        diversity_scores = torch.tensor(diversity_scores, device=x.device)  
        
        return diversity_scores


    def get_model_diversity(self, x, feature_pool_model, method):
        features = self.get_model_feature(x)
        
        diversity_scores = []
        for feature in features:
            diversity_score = self._diversity(feature, feature_pool_model, method)
            diversity_scores.append(diversity_score)

        diversity_scores = torch.tensor(diversity_scores, device=x.device)

        return diversity_scores

    def get_pretrained_intra_diversity(self, x, target, feature_pool_pretrained, target_pool, method):
        features = self.get_pretrained_feature(x)
        
        scaled_feature_pool_pretrained = self.prior(feature_pool_pretrained)
        scaled_features = self.prior(features)

        diversity_scores = []
        for scaled_feature, tgt in zip(scaled_features, target):
            tgt_indices = (target_pool == tgt).nonzero(as_tuple=True)[0]
            diversity_score = self._diversity(scaled_feature, scaled_feature_pool_pretrained[tgt_indices], method)
            diversity_scores.append(diversity_score)

        diversity_scores = torch.tensor(diversity_scores, device=x.device)
        
        return diversity_scores
               

    def get_model_intra_diversity(self, x, target, feature_pool_model, target_pool, method):
        features = self.get_model_feature(x)
        
        diversity_scores = []
        for feature, tgt in zip(features, target):
            tgt_indices = (target_pool == tgt).nonzero(as_tuple=True)[0]
            diversity_score = self._diversity(feature, feature_pool_model[tgt_indices], method)
            diversity_scores.append(diversity_score)

        diversity_scores = torch.tensor(diversity_scores, device=x.device)

        return diversity_scores

    def get_pretrained_inter_intra_diversity(self, x, target, feature_pool_pretrained, target_pool, method):
        features = self.get_pretrained_feature(x)

        scaled_feature_pool_pretrained = self.prior(feature_pool_pretrained)
        scaled_features = self.prior(features) 

        diversity_scores = []
        for scaled_feature, tgt in zip(scaled_features, target):
            tgt_indices = (target_pool == tgt).nonzero(as_tuple=True)[0]
            non_tgt_indices = (target_pool != tgt).nonzero(as_tuple=True)[0]

            intra_diversity_score = self._diversity(scaled_feature, scaled_feature_pool_pretrained[tgt_indices], method)
            inter_diversity_score = self._diversity(scaled_feature, scaled_feature_pool_pretrained[non_tgt_indices], method)

            diversity_scores.append(inter_diversity_score / (intra_diversity_score + 1e-8)) # avoid division by zero
        diversity_scores = torch.tensor(diversity_scores, device=x.device)
        
        return diversity_scores

    def get_model_inter_intra_diversity(self, x, target, feature_pool_model, target_pool, method):
        features = self.get_model_feature(x)
        diversity_scores = []
        for feature, tgt in zip(features, target):
            tgt_indices = (target_pool == tgt).nonzero(as_tuple=True)[0]
            non_tgt_indices = (target_pool != tgt).nonzero(as_tuple=True)[0]

            intra_diversity_score = self._diversity(feature, feature_pool_model[tgt_indices], method)
            inter_diversity_score = self._diversity(feature, feature_pool_model[non_tgt_indices], method)

            diversity_scores.append(inter_diversity_score / (intra_diversity_score + 1e-8)) # avoid division by zero
        diversity_scores = torch.tensor(diversity_scores, device=x.device)
        return diversity_scores


    def aft_score_times_pretrained_diversity(self, x, target, feature_pool_model, feature_pool_pretrained, target_pool, method):
        outputs, feats = self.model(self.transform(x), return_feat=True)

        feats_pretrained = self.pretrained_model(self.transform_pretrained(x))
        
        ce_loss = torch.nn.functional.cross_entropy(outputs, target)

        new_feature_pool_model = torch.cat([feature_pool_model, feats], dim=0)
        new_feature_pool_pretrained = torch.cat([feature_pool_pretrained, feats_pretrained], dim=0)
        prior_loss = - self.prior.prec * self.prior.log_prob(new_feature_pool_model, new_feature_pool_pretrained)
        
        loss = ce_loss + prior_loss

        scaled_feature_pool_pretrained = self.prior(feature_pool_pretrained)
        scaled_features = self.prior(feats_pretrained)

        
        diversity_score = self._diversity(scaled_features[0], scaled_feature_pool_pretrained, method)

        return loss * diversity_score


    def aft_score_times_model_diversity(self, x, target, feature_pool_model, feature_pool_pretrained, target_pool, method):
        outputs, feats = self.model(self.transform(x), return_feat=True)

        feats_pretrained = self.pretrained_model(self.transform_pretrained(x))
        
        ce_loss = torch.nn.functional.cross_entropy(outputs, target)

        new_feature_pool_model = torch.cat([feature_pool_model, feats], dim=0)
        new_feature_pool_pretrained = torch.cat([feature_pool_pretrained, feats_pretrained], dim=0)
        prior_loss = - self.prior.prec * self.prior.log_prob(new_feature_pool_model, new_feature_pool_pretrained)
        
        loss = ce_loss + prior_loss

        diversity_score = self._diversity(feats[0], feature_pool_model, method)

        return loss * diversity_score


    def aft_score_times_pretrained_intra_diversity(self, x, target, feature_pool_model, feature_pool_pretrained, target_pool, method):
        outputs, feats = self.model(self.transform(x), return_feat=True)

        feats_pretrained = self.pretrained_model(self.transform_pretrained(x))
        
        ce_loss = torch.nn.functional.cross_entropy(outputs, target)

        new_feature_pool_model = torch.cat([feature_pool_model, feats], dim=0)
        new_feature_pool_pretrained = torch.cat([feature_pool_pretrained, feats_pretrained], dim=0)
        prior_loss = - self.prior.prec * self.prior.log_prob(new_feature_pool_model, new_feature_pool_pretrained)
        
        loss = ce_loss + prior_loss

        scaled_feature_pool_pretrained = self.prior(feature_pool_pretrained)
        scaled_features = self.prior(feats_pretrained)

        tgt_indices = (target_pool == target[0]).nonzero(as_tuple=True)[0]
        diversity_score = self._diversity(scaled_features[0], scaled_feature_pool_pretrained[tgt_indices], method)

        return loss * diversity_score

    def aft_score_times_model_intra_diversity(self, x, target, feature_pool_model, feature_pool_pretrained, target_pool, method):
        outputs, feats = self.model(self.transform(x), return_feat=True)

        feats_pretrained = self.pretrained_model(self.transform_pretrained(x))
        
        ce_loss = torch.nn.functional.cross_entropy(outputs, target)

        new_feature_pool_model = torch.cat([feature_pool_model, feats], dim=0)
        new_feature_pool_pretrained = torch.cat([feature_pool_pretrained, feats_pretrained], dim=0)
        prior_loss = - self.prior.prec * self.prior.log_prob(new_feature_pool_model, new_feature_pool_pretrained)
        
        loss = ce_loss + prior_loss

        tgt_indices = (target_pool == target[0]).nonzero(as_tuple=True)[0]
        diversity_score = self._diversity(feats[0], feature_pool_model[tgt_indices], method)

        return loss * diversity_score

    def aft_score_times_pretrained_inter_intra_diversity(self, x, target, feature_pool_model, feature_pool_pretrained, target_pool, method):
        outputs, feats = self.model(self.transform(x), return_feat=True)

        feats_pretrained = self.pretrained_model(self.transform_pretrained(x))
        
        ce_loss = torch.nn.functional.cross_entropy(outputs, target)

        new_feature_pool_model = torch.cat([feature_pool_model, feats], dim=0)
        new_feature_pool_pretrained = torch.cat([feature_pool_pretrained, feats_pretrained], dim=0)
        prior_loss = - self.prior.prec * self.prior.log_prob(new_feature_pool_model, new_feature_pool_pretrained)
        
        loss = ce_loss + prior_loss

        scaled_feature_pool_pretrained = self.prior(feature_pool_pretrained)
        scaled_features = self.prior(feats_pretrained)

        tgt_indices = (target_pool == target[0]).nonzero(as_tuple=True)[0]
        non_tgt_indices = (target_pool != target[0]).nonzero(as_tuple=True)[0]

        intra_diversity_score = self._diversity(scaled_features[0], scaled_feature_pool_pretrained[tgt_indices], method)
        inter_diversity_score = self._diversity(scaled_features[0], scaled_feature_pool_pretrained[non_tgt_indices], method)

        diversity_score = inter_diversity_score / (intra_diversity_score + 1e-8) # avoid division by zero

        return loss * diversity_score

    def aft_score_times_model_inter_intra_diversity(self, x, target, feature_pool_model, feature_pool_pretrained, target_pool, method):
        outputs, feats = self.model(self.transform(x), return_feat=True)

        feats_pretrained = self.pretrained_model(self.transform_pretrained(x))
        
        ce_loss = torch.nn.functional.cross_entropy(outputs, target)

        new_feature_pool_model = torch.cat([feature_pool_model, feats], dim=0)
        new_feature_pool_pretrained = torch.cat([feature_pool_pretrained, feats_pretrained], dim=0)
        prior_loss = - self.prior.prec * self.prior.log_prob(new_feature_pool_model, new_feature_pool_pretrained)
        
        loss = ce_loss + prior_loss

        tgt_indices = (target_pool == target[0]).nonzero(as_tuple=True)[0]
        non_tgt_indices = (target_pool != target[0]).nonzero(as_tuple=True)[0]

        intra_diversity_score = self._diversity(feats[0], feature_pool_model[tgt_indices], method)
        inter_diversity_score = self._diversity(feats[0], feature_pool_model[non_tgt_indices], method)

        diversity_score = inter_diversity_score / (intra_diversity_score + 1e-8) # avoid division by zero

        return loss * diversity_score



    def get_model_feature(self, x):
        _, feats = self.model(self.transform(x), return_feat=True)
        return feats

    def get_pretrained_feature(self, x):
        feats = self.pretrained_model(self.transform_pretrained(x))
        return feats



def do_aft_score(images, aft_module, score, targets=None, feature_pool_model=None, feature_pool_pretrained=None, target_pool=None,**kwargs):
    if score == "ce":
        rewards = aft_module.get_ce_loss(images, targets, reduction="none")
    elif score == "aft":
        rewards = []
        for image in images:
            reward = aft_module.get_aft_loss(image.unsqueeze(0), feature_pool_model, feature_pool_pretrained)
            rewards.append(reward)
        rewards = torch.stack(rewards)
    elif score == "total":
        rewards = []
        for image, target in zip(images, targets):
            reward = aft_module.get_total_loss(image.unsqueeze(0), target.unsqueeze(0), feature_pool_model, feature_pool_pretrained)
            rewards.append(reward)
        rewards = torch.stack(rewards)
    elif score == "pretrained_diversity_cos_min":
        rewards = aft_module.get_pretrained_diversity(images, feature_pool_pretrained, method="cos_min")
    elif score == "pretrained_diversity_l2_min":
        rewards = aft_module.get_pretrained_diversity(images, feature_pool_pretrained, method="l2_min")
    elif score == "model_diversity_cos_min":
        rewards = aft_module.get_model_diversity(images, feature_pool_model, method="cos_min")
    elif score == "model_diversity_l2_min":
        rewards = aft_module.get_model_diversity(images, feature_pool_model, method="l2_min")
    elif score == "pretrained_intra_diversity_cos_min":
        rewards = aft_module.get_pretrained_intra_diversity(images, targets, feature_pool_pretrained, target_pool, method="cos_min")
    elif score == "pretrained_intra_diversity_l2_min":
        rewards = aft_module.get_pretrained_intra_diversity(images, targets, feature_pool_pretrained, target_pool, method="l2_min")
    elif score == "model_intra_diversity_cos_min":
        rewards = aft_module.get_model_intra_diversity(images, targets, feature_pool_model, target_pool, method="cos_min")
    elif score == "model_intra_diversity_l2_min":
        rewards = aft_module.get_model_intra_diversity(images, targets, feature_pool_model, target_pool, method="l2_min")
    elif score == "pretrained_inter_intra_diversity_cos_min":
        rewards = aft_module.get_pretrained_inter_intra_diversity(images, targets, feature_pool_pretrained, target_pool, method="cos_min")
    elif score == "pretrained_inter_intra_diversity_l2_min":
        rewards = aft_module.get_pretrained_inter_intra_diversity(images, targets, feature_pool_pretrained, target_pool, method="l2_min")    
    elif score == "model_inter_intra_diversity_cos_min":
        rewards = aft_module.get_model_inter_intra_diversity(images, targets, feature_pool_model, target_pool, method="cos_min")
    elif score == "model_inter_intra_diversity_l2_min":
        rewards = aft_module.get_model_inter_intra_diversity(images, targets, feature_pool_model, target_pool, method="l2_min")
    elif score == "aft_times_pretrained_diversity_cos_min":
        rewards = []
        for image, target in zip(images, targets):
            reward = aft_module.aft_score_times_pretrained_diversity(image.unsqueeze(0), target.unsqueeze(0), feature_pool_model, feature_pool_pretrained, target_pool, method="cos_min")
            rewards.append(reward)
        rewards = torch.tensor(rewards)
    elif score == "aft_times_pretrained_diversity_l2_min":
        rewards = []
        for image, target in zip(images, targets):
            reward = aft_module.aft_score_times_pretrained_diversity(image.unsqueeze(0), target.unsqueeze(0), feature_pool_model, feature_pool_pretrained, target_pool, method="l2_min")
            rewards.append(reward)
        rewards = torch.tensor(rewards)
    elif score == "aft_times_model_diversity_cos_min":
        rewards = []
        for image, target in zip(images, targets):
            reward = aft_module.aft_score_times_model_diversity(image.unsqueeze(0), target.unsqueeze(0), feature_pool_model, feature_pool_pretrained, target_pool, method="cos_min")
            rewards.append(reward)
        rewards = torch.tensor(rewards)
    elif score == "aft_times_model_diversity_l2_min":
        rewards = []
        for image, target in zip(images, targets):
            reward = aft_module.aft_score_times_model_diversity(image.unsqueeze(0), target.unsqueeze(0), feature_pool_model, feature_pool_pretrained, target_pool, method="l2_min")
            rewards.append(reward)
        rewards = torch.tensor(rewards)
    elif score == "aft_times_pretrained_intra_diversity_cos_min":
        rewards = []
        for image, target in zip(images, targets):
            reward = aft_module.aft_score_times_pretrained_intra_diversity(image.unsqueeze(0), target.unsqueeze(0), feature_pool_model, feature_pool_pretrained, target_pool, method="cos_min")
            rewards.append(reward)
        rewards = torch.tensor(rewards)
    elif score == "aft_times_pretrained_intra_diversity_l2_min":
        rewards = []
        for image, target in zip(images, targets):
            reward = aft_module.aft_score_times_pretrained_intra_diversity(image.unsqueeze(0), target.unsqueeze(0), feature_pool_model, feature_pool_pretrained, target_pool, method="l2_min")
            rewards.append(reward)
        rewards = torch.tensor(rewards)
    elif score == "aft_times_model_intra_diversity_cos_min":
        rewards = []
        for image, target in zip(images, targets):
            reward = aft_module.aft_score_times_model_intra_diversity(image.unsqueeze(0), target.unsqueeze(0), feature_pool_model, feature_pool_pretrained, target_pool, method="cos_min")
            rewards.append(reward)
    elif score == "aft_times_model_intra_diversity_l2_min":
        rewards = []
        for image, target in zip(images, targets):
            reward = aft_module.aft_score_times_model_intra_diversity(image.unsqueeze(0), target.unsqueeze(0), feature_pool_model, feature_pool_pretrained, target_pool, method="l2_min")
            rewards.append(reward)
        rewards = torch.tensor(rewards)
    elif score == "aft_times_pretrained_inter_intra_diversity_cos_min":
        rewards = []
        for image, target in zip(images, targets):
            reward = aft_module.aft_score_times_pretrained_inter_intra_diversity(image.unsqueeze(0), target.unsqueeze(0), feature_pool_model, feature_pool_pretrained, target_pool, method="cos_min")
            rewards.append(reward)
        rewards = torch.tensor(rewards)
    elif score == "aft_times_pretrained_inter_intra_diversity_l2_min":
        rewards = []
        for image, target in zip(images, targets):
            reward = aft_module.aft_score_times_pretrained_inter_intra_diversity(image.unsqueeze(0), target.unsqueeze(0), feature_pool_model, feature_pool_pretrained, target_pool, method="l2_min")
            rewards.append(reward)
        rewards = torch.tensor(rewards)
    elif score == "aft_times_model_inter_intra_diversity_cos_min":
        rewards = []
        for image, target in zip(images, targets):
            reward = aft_module.aft_score_times_model_inter_intra_diversity(image.unsqueeze(0), target.unsqueeze(0), feature_pool_model, feature_pool_pretrained, target_pool, method="cos_min")
            rewards.append(reward)
        rewards = torch.tensor(rewards)
    elif score == "aft_times_model_inter_intra_diversity_l2_min":
        rewards = []
        for image, target in zip(images, targets):
            reward = aft_module.aft_score_times_model_inter_intra_diversity(image.unsqueeze(0), target.unsqueeze(0), feature_pool_model, feature_pool_pretrained, target_pool, method="l2_min")
            rewards.append(reward)
        rewards = torch.tensor(rewards)
    else:
        raise ValueError(f"Unknown score: {score}")
    return rewards

def main(seed, edm_ckpt, aft_module, aft_score, num_target_images, save_dir, class_names, use_downstream, no_steering, gradient_scale=0.03):
    DEVICE = 'cuda'
    config = f"""
        network_pkl: {edm_ckpt}
        batch_size: 1
        dtype: float32
        S_churn: 40
        """

    config = yaml.safe_load(config)

    generator = EDM(**config)

    generator.to(DEVICE)
    generator.eval()

    init_feature_pool_model = torch.empty((0, 2048)).to(DEVICE) # random features for demo; replace with real features
    init_feature_pool_pretrained = torch.empty((0, 1536)).to(DEVICE) # random features for demo; replace with real features
    init_target_pool = torch.empty((0,), dtype=torch.int64).to(DEVICE)

    if use_downstream:
        train_dataset = get_dataset("flowers", lambda train: lambda x: torchvision.transforms.ToTensor()(x), None, no_augment=True)[0]
        train_loader = get_loader(train_dataset, 1, num_workers=4, shuffle=False, input_collate_fn=None)

        for data in tqdm(train_loader):
            if isinstance(data, (tuple, list)):
                inputs, labels = data
            else:
                inputs = data

            inputs = inputs.to(DEVICE)

            feature_model = aft_module.get_model_feature(inputs)
            feature_pretrained = aft_module.get_pretrained_feature(inputs)

            init_feature_pool_model = torch.cat([init_feature_pool_model, feature_model], dim=0)
            init_feature_pool_pretrained = torch.cat([init_feature_pool_pretrained, feature_pretrained], dim=0)
            init_target_pool = torch.cat([init_target_pool, labels.to(DEVICE)], dim=0)


    aft_args = {
        "aft_module": aft_module,
        "score": aft_score,
        "feature_pool_model": init_feature_pool_model.detach(),
        "feature_pool_pretrained": init_feature_pool_pretrained.detach(),
        "target_pool": init_target_pool.detach(),
    }


    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    image_ind = 0
    for _ in tqdm(range(int(np.ceil(num_target_images / config['batch_size'])))):
        with torch.autocast(device_type=next(iter(generator.parameters())).device.type, dtype=torch.float32):

            result = generator.sample_gradient_guidance(config['batch_size'], do_aft_score if not no_steering else None, aft_args, guide_every=5, guidance_scale=gradient_scale)

        images = result[0]
        labels = result[1].cpu().numpy().tolist()

        images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()

        for image_np, label in zip(images_np, labels):
            image_path = os.path.join(save_dir, class_names[label], f"{image_ind:06d}.png")
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            Image.fromarray(image_np, "RGB").save(image_path)
            image_ind += 1


        features_model = aft_module.get_model_feature(images)
        features_pretrained = aft_module.get_pretrained_feature(images)

        aft_args["feature_pool_model"] = torch.cat([aft_args["feature_pool_model"], features_model], dim=0).detach()
        aft_args["feature_pool_pretrained"] = torch.cat([aft_args["feature_pool_pretrained"], features_pretrained], dim=0).detach()


if __name__ == "__main__":
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--edm_ckpt', type=str, required=True, help='Path to EDM checkpoint')
    parser.add_argument('--model_ckpt', type=str, required=True, help='Path to AFT model checkpoint')
    parser.add_argument('--prior_ckpt', type=str, required=True, help='Path to AFT prior checkpoint')
    parser.add_argument('--aft_score', type=str, default='total', help='AFT score to use')
    parser.add_argument('--num_target_images', type=int, default=3000, help='Number of target images to generate')
    parser.add_argument('--use_downstream', type=str2bool, default=False, help='Whether to use downstream data for feature pool initialization')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save generated images')
    parser.add_argument('--no_steering', action='store_true', help='If set, do not use steering (for ablation)')
    parser.add_argument('--gradient_scale', type=float, default=0.03, help='Guidance scale for gradient steering')
    args = parser.parse_args()

    print("generating data with edm guidance steering...")

    # seed = 0
    # edm_ckpt = "/NFS/workspaces/tg.ahn/Collab/edm/training-runs-flowers102/00001-flowers102-64x64-cond-ddpmpp-edm-gpus1-batch32-fp32/network-snapshot-008132.pkl"
    # model_ckpt = "/NFS/workspaces/tg.ahn/Collab/adaptive-feature-transfer/ckpts/aft/flowers/resnet50.a1_in1k_vit_giant_patch14_dinov2.lvd142m_lr1e-3_seed0/model.pt"
    # prior_ckpt = "/NFS/workspaces/tg.ahn/Collab/adaptive-feature-transfer/ckpts/aft/flowers/resnet50.a1_in1k_vit_giant_patch14_dinov2.lvd142m_lr1e-3_seed0/prior.pt"

    seed = args.seed
    edm_ckpt = args.edm_ckpt
    model_ckpt = args.model_ckpt
    prior_ckpt = args.prior_ckpt

    aft_module = AFTModule(
        model="resnet50.a1_in1k",
        model_out_dim=102,
        pretrained_model="vit_giant_patch14_dinov2.lvd142m",
        prior_prec=10,
        model_ckpt=model_ckpt,
        prior_ckpt=prior_ckpt,
    ).to('cuda')

    # aft_score = "total"
    # use_downstream = True
    # num_target_images = 3000

    aft_score = args.aft_score
    use_downstream = args.use_downstream
    num_target_images = args.num_target_images
    save_dir = args.save_dir
    
    class_file = "./classes/flowers.txt"
    with open(class_file, "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    main(seed, edm_ckpt, aft_module, aft_score, num_target_images, save_dir, class_names, use_downstream, args.no_steering, args.gradient_scale)