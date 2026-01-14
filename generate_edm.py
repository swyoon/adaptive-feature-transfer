import os
import torch
import yaml
import numpy as np
from PIL import Image
from tqdm import tqdm
import random
import argparse

from diffusion.edm.model import EDM


def main(seed, edm_ckpt, num_target_images, save_dir, class_names, num_steps, batch_size):
    DEVICE = 'cuda'
    config = f"""
        network_pkl: {edm_ckpt}
        batch_size: {batch_size}
        dtype: float16
        num_steps: {num_steps}
        S_churn: 40
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
            if image_ind >= num_target_images:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--dataset', type=str, default='flowers', help='Dataset name')
    parser.add_argument('--edm_ckpt', type=str, required=True, help='Path to EDM checkpoint')
    parser.add_argument('--num_target_images', type=int, default=3000, help='Number of target images to generate')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save generated images')
    parser.add_argument('--num_steps', type=int, default=18, help='Number of EDM sampling steps')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for EDM sampling')
    args = parser.parse_args()

    print("generating data with edm...")

    seed = args.seed
    dataset = args.dataset
    edm_ckpt = args.edm_ckpt
    num_target_images = args.num_target_images
    save_dir = args.save_dir
    num_steps = args.num_steps
    batch_size = args.batch_size

    class_file = f"./classes/{args.dataset}.txt"
    with open(class_file, "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    main(seed, edm_ckpt, num_target_images, save_dir, class_names, num_steps, batch_size)