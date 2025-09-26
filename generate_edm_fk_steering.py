import os
import torch
import yaml
import numpy as np
from PIL import Image
from tqdm import tqdm
import random

from diffusion.edm.model import EDM
from diffusion.edm.fkd.fkd_rewards import DiversityModel


def main(seed, edm_ckpt, scale_ckpt, num_target_images, save_dir, class_names):
    DEVICE = 'cuda'
    config = f"""
        network_pkl: {edm_ckpt}
        batch_size: 1
        dtype: float16
        """

    config = yaml.safe_load(config)

    generator = EDM(**config)

    generator.to(DEVICE)
    generator.eval()


    FKD_ARGS = {
        "potential_type": "diff",
        "lmbda": 1.0,
        "num_particles": 4,
        "adaptive_resampling": True,
        "resample_frequency": 5,
        "resampling_t_start": 0,
        "resampling_t_end": 60,
        "time_steps": 60, # set as same as resampling_t_end
        "latent_to_decode_fn": lambda x: x,  # identity for EDM (already image-space)
        "get_reward_fn": "Diversity",  # "ClassifierLoss" will be defined later
        "cls_model": None,  # it is required when using "ClassifierLoss"
        "use_smc": True,
        "output_dir": "./outputs/generated/fkd_results", # modify
        "print_rewards": False, # print rewards during sampling
        "visualize_intermediate": False, # save results during sampling in output_dir
        "visualzie_x0": False, # save x0 prediction during sampling in output_dir
        "feature_pool": torch.empty((0, 1536)).to(DEVICE), # random features for demo; replace with real features
        "model_class": "vit_giant_patch14_dinov2.lvd142m",
        "num_features": 1536,
        "diag": True,
        "tensor_product": False,
        "scale_ckpt": scale_ckpt,
    }

    model = DiversityModel(**FKD_ARGS).to(DEVICE)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    image_ind = 0
    for _ in tqdm(range(int(np.ceil(num_target_images / config['batch_size'])))):
        with torch.autocast(device_type=next(iter(generator.parameters())).device.type, dtype=torch.float16):
            result = generator.sample_fk_steering(fkd_args=FKD_ARGS)

        images = result[0]
        labels = result[1].cpu().numpy().tolist()

        images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()

        for image_np, label in zip(images_np, labels):
            image_path = os.path.join(save_dir, class_names[label], f"{image_ind:06d}.png")
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            Image.fromarray(image_np, "RGB").save(image_path)
            image_ind += 1


        with torch.no_grad():
            features = model.get_scaled_feature(result[0])

        FKD_ARGS["feature_pool"] = torch.cat([FKD_ARGS["feature_pool"], features], dim=0)

if __name__ == "__main__":
    seed = 0
    edm_ckpt = "/NFS/workspaces/tg.ahn/Collab/edm/training-runs-flowers102/00001-flowers102-64x64-cond-ddpmpp-edm-gpus1-batch32-fp32/network-snapshot-008132.pkl"
    scale_ckpt = "/NFS/workspaces/tg.ahn/Collab/adaptive-feature-transfer/ckpts/aft/flowers/resnet50.a1_in1k_vit_giant_patch14_dinov2.lvd142m_lr1e-3_seed0/prior.pt"
    num_target_images = 3000
    save_dir = f"./outputdir5/edm_div_cos_min_steering/flowers-{seed}"
    class_file = "./classes/flowers.txt"
    with open(class_file, "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    main(seed, edm_ckpt, scale_ckpt, num_target_images, save_dir, class_names)