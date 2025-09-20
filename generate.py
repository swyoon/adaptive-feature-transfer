import os
from diffusers import StableDiffusionXLPipeline
import torch
from PIL import Image

def generate_sdxl_images(
    dataset: str,
    samples_per_class: int,
    width: int,
    height: int,
    output_dir: str,
    device: str = "cuda"
):
    # Read class names
    class_file = f"classes/{dataset}.txt"
    with open(class_file, "r") as f:
        class_names = [line.strip() for line in f if line.strip()]

    # Load SDXL pipeline with a more photorealistic model
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True
    ).to(device)
    
    # Enable memory efficient attention
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()

    # Generate images for each class
    for class_name in class_names:
        class_dir = os.path.join(output_dir, dataset, class_name)
        os.makedirs(class_dir, exist_ok=True)
        for idx in range(samples_per_class):
            prompt = f"a photo of a {class_name}"
            image = pipe(
                prompt, 
                width=1024, 
                height=1024,
                guidance_scale=10.0,  # Even higher guidance for more prompt adherence
                num_inference_steps=50,  # More steps for better quality
                clip_skip=1  # Use last CLIP layer for more realistic results
            ).images[0]
            # resize to desired dimensions
            image = image.resize((width, height), Image.LANCZOS)
            image_path = os.path.join(class_dir, f"{idx:06d}.png")
            image.save(image_path)
            print(f"Saved: {image_path}")

if __name__ == "__main__":
    # Example usage
    generate_sdxl_images(
        dataset="pets",                # or "flowers"
        samples_per_class=5,           # number of images per class
        width=512,                     # image width
        height=512,                    # image height
        output_dir="outputdir"         # directory to save images
    )