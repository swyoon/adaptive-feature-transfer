import os
import torch
import yaml
import numpy as np
from PIL import Image
from tqdm import tqdm
import random

from diffusion.edm.model import EDM
from diffusion.edm.fkd.fkd_rewards import DiversityModel


def main(seed, edm_ckpt, num_target_images, save_dir, class_names):
    DEVICE = 'cuda'
    config = f"""
        network_pkl: {edm_ckpt}
        batch_size: 4
        dtype: float16
        num_steps: 60
        """

    config = yaml.safe_load(config)

    generator = EDM(**config)

    generator.to(DEVICE)
    generator.eval()

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    image_ind = 0
    for _ in tqdm(range(int(np.ceil(num_target_images / config['batch_size'])))):
        with torch.autocast(device_type=next(iter(generator.parameters())).device.type, dtype=torch.float16):
            result = generator.sample()

        images = result[0]
        labels = result[1].cpu().numpy().tolist()

        images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()

        for image_np, label in zip(images_np, labels):
            image_path = os.path.join(save_dir, class_names[label], f"{image_ind:06d}.png")
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            Image.fromarray(image_np, "RGB").save(image_path)
            image_ind += 1


if __name__ == "__main__":
    seed = 0
    edm_ckpt = "/NFS/workspaces/tg.ahn/Collab/edm/training-runs-flowers102/00001-flowers102-64x64-cond-ddpmpp-edm-gpus1-batch32-fp32/network-snapshot-008132.pkl"
    num_target_images = 3000
    save_dir = f"./outputdir5/edm/flowers-{seed}"
    class_file = "./classes/flowers.txt"
    with open(class_file, "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    main(seed, edm_ckpt, num_target_images, save_dir, class_names)