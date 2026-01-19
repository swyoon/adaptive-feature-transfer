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
from diffusion.edm.fkd.fkd_rewards import DiversityModel

from models import create_model
from prior import get_prior, UniformPrior
from data import get_dataset, get_loader, get_out_dim

class DummyFeatureDataset:
    feat_dims = [1536] # NOTE hard coded for vit_giant_patch14_dinov2.lvd142m
    num_features = 1536
    def __len__(self):
        return 1

class AFTModule(torch.nn.Module):
    def __init__(self, model, model_out_dim, pretrained_model, prior_prec, model_ckpt, prior_ckpt, method="aft", **kwargs):
        super().__init__()
        self.model, get_transform, tokenizer, input_collate_fn = create_model(
        model, out_dim=model_out_dim, pretrained=False
        )
        model_state = torch.load(model_ckpt, map_location="cpu")
        self.model.load_state_dict(model_state, strict=False)
        self.model.eval()

        self.transform = get_transform(train=False)

        self.get_transform = get_transform
        self.tokenizer = tokenizer
        self.input_collate_fn = input_collate_fn
        

        self.pretrained_model, get_transform_pretrained, tokenizer_pretrained, input_collate_fn_pretrained = create_model(pretrained_model, out_dim=0, pretrained=True, extract_features=True)
        self.pretrained_model.eval()

        self.transform_pretrained = get_transform_pretrained(train=False)

        ds = DummyFeatureDataset()
        train_ds = DummyFeatureDataset()
        
        if method == "aft":
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
        else:
            self.prior = UniformPrior()
    
    def entropy(self, x):
        x = self.transform(x)
        with torch.no_grad():
            outputs = self.model(x)
        prob = torch.nn.functional.softmax(outputs, dim=-1)
        entropy = torch.nn.functional.cross_entropy(outputs, prob, reduction='none')
        return entropy

    def get_ce_loss(self, x, y, reduction="mean"):
        x = self.transform(x)
        with torch.no_grad():
            outputs = self.model(x)
        loss = torch.nn.functional.cross_entropy(outputs, y, reduction=reduction)
        return loss

    def get_aft_loss_batch(self, x):
        with torch.no_grad():
            outputs, feats = self.model(self.transform(x), return_feat=True)

            feats_pretrained = self.pretrained_model(self.transform_pretrained(x))

        prior_loss = -self.prior.prec * self.prior.log_prob(feats, feats_pretrained)
        return prior_loss

    def get_total_loss_batch(self, x, y):
        with torch.no_grad():
            outputs, feats = self.model(self.transform(x), return_feat=True)

            feats_pretrained = self.pretrained_model(self.transform_pretrained(x))
        
        ce_loss = torch.nn.functional.cross_entropy(outputs, y)

        prior_loss = - self.prior.prec * self.prior.log_prob(feats, feats_pretrained)
        return ce_loss + prior_loss

    def get_aft_loss(self, x, feature_pool_model, feature_pool_pretrained):
        with torch.no_grad():
            outputs, feats = self.model(self.transform(x), return_feat=True)

            feats_pretrained = self.pretrained_model(self.transform_pretrained(x))

        new_feature_pool_model = torch.cat([feature_pool_model, feats], dim=0)
        new_feature_pool_pretrained = torch.cat([feature_pool_pretrained, feats_pretrained], dim=0)

        prior_loss = - self.prior.prec * self.prior.log_prob(new_feature_pool_model, new_feature_pool_pretrained)
        return prior_loss

    def get_total_loss(self, x, y, feature_pool_model, feature_pool_pretrained):
        with torch.no_grad():
            outputs, feats = self.model(self.transform(x), return_feat=True)

            feats_pretrained = self.pretrained_model(self.transform_pretrained(x))
        
        new_feature_pool_model = torch.cat([feature_pool_model, feats], dim=0)
        new_feature_pool_pretrained = torch.cat([feature_pool_pretrained, feats_pretrained], dim=0)

        ce_loss = torch.nn.functional.cross_entropy(outputs, y)

        prior_loss = - self.prior.prec * self.prior.log_prob(new_feature_pool_model, new_feature_pool_pretrained)
        return ce_loss + prior_loss

    def get_total_loss_batch(self, x, y, feature_pool_model, feature_pool_pretrained):
        with torch.no_grad():
            outputs, feats = self.model(self.transform(x), return_feat=True)

            feats_pretrained = self.pretrained_model(self.transform_pretrained(x))
        
        ce_loss = torch.nn.functional.cross_entropy(outputs, y, reduction="none")

        prior_loss_ = []
        for ind in range(x.shape[0]):
            new_feature_pool_model = torch.cat([feature_pool_model, feats[ind:ind+1]], dim=0)
            new_feature_pool_pretrained = torch.cat([feature_pool_pretrained, feats_pretrained[ind:ind+1]], dim=0)

            prior_loss = - self.prior.prec * self.prior.log_prob(new_feature_pool_model, new_feature_pool_pretrained)
            prior_loss_.append(prior_loss)
        prior_loss = torch.stack(prior_loss_)
        
        return ce_loss + prior_loss



    def get_model_feature(self, x):
        with torch.no_grad():
            _, feats = self.model(self.transform(x), return_feat=True)
        return feats

    def get_pretrained_feature(self, x):
        with torch.no_grad():
            feats = self.pretrained_model(self.transform_pretrained(x))
        return feats
    

def do_aft_score(images, aft_module, score, targets=None, feature_pool_model=None, feature_pool_pretrained=None, target_pool=None,**kwargs):
    if score == "entropy":
        rewards = aft_module.entropy(images)
    elif score == "ce":
        rewards = aft_module.get_ce_loss(images, targets, reduction="none")
    elif score == "aft":
        rewards = []
        for image in images:
            reward = aft_module.get_aft_loss(image.unsqueeze(0), feature_pool_model, feature_pool_pretrained)
            rewards.append(reward)
        rewards = torch.stack(rewards)
    elif score == "total":
        rewards = aft_module.get_total_loss_batch(images, targets, feature_pool_model, feature_pool_pretrained)
    else:
        raise ValueError(f"Unknown score: {score}")
    return rewards

def main(seed, dataset, edm_ckpt, aft_module, aft_score, num_target_images, save_dir, class_names, use_downstream, args):
    num_steps = args.num_steps
    lmda = args.lmda
    resample_frequency = args.resample_frequency
    no_steering = args.no_steering
    aft_batch_size = args.aft_batch_size
    DEVICE = 'cuda'
    config = f"""
        network_pkl: {edm_ckpt}
        batch_size: 1
        dtype: float16
        num_steps: {num_steps}
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
        train_dataset = get_dataset(dataset, lambda train: lambda x: torchvision.transforms.ToTensor()(x), None, no_augment=True)[0]
        train_loader = get_loader(train_dataset, 1, num_workers=4, shuffle=False, input_collate_fn=aft_module.input_collate_fn)

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

    FKD_ARGS = {
        "potential_type": "diff",
        "lmbda": lmda,
        "num_particles": 4,
        "adaptive_resampling": True,
        "resample_frequency": resample_frequency,
        "resampling_t_start": 0,
        "resampling_t_end": num_steps,
        "time_steps": num_steps, # set as same as resampling_t_end
        "latent_to_decode_fn": lambda x: torch.clamp(x, -1, 1) * 0.5 + 0.5,  # identity for EDM (already image-space)
        "use_smc": True if not no_steering else False,
        "output_dir": "./outputs/generated/fkd_results", # modify
        "print_rewards": False, # print rewards during sampling
        "visualize_intermediate": False, # save results during sampling in output_dir
        "visualize_x0": False, # save x0 prediction during sampling in output_dir
    }

    # define reward_score
    reward_fn = do_aft_score

    reward_fn_args = {
        "feature_pool_model_all": init_feature_pool_model,
        "feature_pool_pretrained_all": init_feature_pool_pretrained,
        "target_pool_all": init_target_pool,
        "score": aft_score,
        "aft_module": aft_module,
    }
    # reward_fn =
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    image_ind = 0
    for _ in tqdm(range(int(np.ceil(num_target_images / config['batch_size'])))):
        # Resample a subset of feature pool for aft loss computation
        if aft_score in ["aft", "total"]:
            randidx = np.random.choice(len(reward_fn_args["feature_pool_model_all"]), size=aft_batch_size, replace=False)
            reward_fn_args["feature_pool_model"] = reward_fn_args["feature_pool_model_all"][randidx]
            reward_fn_args["feature_pool_pretrained"] = reward_fn_args["feature_pool_pretrained_all"][randidx]
            reward_fn_args["target_pool"] = reward_fn_args["target_pool_all"][randidx]
        with torch.autocast(device_type=next(iter(generator.parameters())).device.type, dtype=torch.float16):
            result = generator.sample_fk_steering(fkd_args=FKD_ARGS, reward_fn=reward_fn, reward_fn_args=reward_fn_args)

        images = result[0]
        labels = result[1].cpu().numpy().tolist()

        images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()

        for image_np, label in zip(images_np, labels):
            image_path = os.path.join(save_dir, class_names[label], f"{image_ind:06d}.png")
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            Image.fromarray(image_np, "RGB").save(image_path)
            image_ind += 1

        images = torch.clamp(images, -1., 1.) * 0.5 + 0.5
        features_model = aft_module.get_model_feature(images)
        features_pretrained = aft_module.get_pretrained_feature(images)

        reward_fn_args["feature_pool_model_all"] = torch.cat([reward_fn_args["feature_pool_model_all"], features_model], dim=0)
        reward_fn_args["feature_pool_pretrained_all"] = torch.cat([reward_fn_args["feature_pool_pretrained_all"], features_pretrained], dim=0)
        reward_fn_args["target_pool_all"] = torch.cat([reward_fn_args["target_pool_all"], torch.tensor(labels).to(DEVICE)], dim=0)


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
    parser.add_argument('--dataset', type=str, default='flowers', help='Dataset name')
    parser.add_argument('--edm_ckpt', type=str, required=True, help='Path to EDM checkpoint')
    parser.add_argument('--model_ckpt', type=str, required=True, help='Path to AFT model checkpoint')
    parser.add_argument('--prior_ckpt', type=str, required=True, help='Path to AFT prior checkpoint')
    parser.add_argument('--prec', type=int, default=10, help='hyperparaneter beta for AFT')
    parser.add_argument('--aft_score', type=str, default='total', help='AFT score to use')
    parser.add_argument('--num_target_images', type=int, default=3000, help='Number of target images to generate')
    parser.add_argument('--use_downstream', type=str2bool, default=False, help='Whether to use downstream data for feature pool initialization')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save generated images')
    parser.add_argument('--num_steps', type=int, default=18, help='Number of EDM sampling steps')
    parser.add_argument('--lmda', type=float, default=10.0, help='Lambda value for FKD steering')
    parser.add_argument('--resample_frequency', type=int, default=1, help='Resampling frequency for FKD steering')
    parser.add_argument('--no_steering', action='store_true', help='If set, do not use FKD steering (for ablation)')
    parser.add_argument('--aft_batch_size', type=int, default=128, help='Batch size for AFT loss computation')
    args = parser.parse_args()

    print("generating data with edm fk steering...")

    # seed = 0
    # edm_ckpt = "/NFS/workspaces/tg.ahn/Collab/edm/training-runs-flowers102/00001-flowers102-64x64-cond-ddpmpp-edm-gpus1-batch32-fp32/network-snapshot-008132.pkl"
    # model_ckpt = "/NFS/workspaces/tg.ahn/Collab/adaptive-feature-transfer/ckpts/aft/flowers/resnet50.a1_in1k_vit_giant_patch14_dinov2.lvd142m_lr1e-3_seed0/model.pt"
    # prior_ckpt = "/NFS/workspaces/tg.ahn/Collab/adaptive-feature-transfer/ckpts/aft/flowers/resnet50.a1_in1k_vit_giant_patch14_dinov2.lvd142m_lr1e-3_seed0/prior.pt"

    seed = args.seed
    dataset = args.dataset
    edm_ckpt = args.edm_ckpt
    model_ckpt = args.model_ckpt
    prior_ckpt = args.prior_ckpt

    aft_module = AFTModule(
        model="resnet50.a1_in1k",
        model_out_dim=get_out_dim(args.dataset),
        pretrained_model="vit_giant_patch14_dinov2.lvd142m",
        prior_prec=args.prec,
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
    
    class_file = f"./classes/{args.dataset}.txt"
    with open(class_file, "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    main(seed, dataset, edm_ckpt, aft_module, aft_score, num_target_images, save_dir, class_names, use_downstream, args)