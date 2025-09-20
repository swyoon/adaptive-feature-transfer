import os
from diffusers import StableDiffusionXLPipeline
import torch
from PIL import Image

def generate_sdxl_images(
    dataset: str,
    num_samples: int,
    width: int,
    height: int,
    output_dir: str,
    device: str = "cuda"
):

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
    class_dir = os.path.join(output_dir, dataset)
    os.makedirs(class_dir, exist_ok=True)
    for idx in range(num_samples):    
        prompt = f"a photo of a {dataset}"

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
        dataset="pet",                # or "flowers"
        num_samples=3000,           # number of images per class
        width=512,                     # image width
        height=512,                    # image height
        output_dir="outputdir3"         # directory to save images
    )

    # generate_sdxl_images(
    #     dataset="flowers",                # or "flowers"
    #     samples_per_class=30,           # number of images per class
    #     width=512,                     # image width
    #     height=512,                    # image height
    #     output_dir="outputdir2"         # directory to save images
    # )