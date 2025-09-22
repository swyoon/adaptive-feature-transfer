import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, DDPMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from torchvision import datasets, transforms
from peft import LoraConfig, get_peft_model, TaskType
import argparse
import os
from tqdm import tqdm
import wandb
from PIL import Image
import numpy as np

class Flowers102Dataset(torch.utils.data.Dataset):
    def __init__(self, split='train', size=1024):
        self.dataset = datasets.Flowers102(
            root='./data', 
            split=split, 
            download=True, 
            transform=transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
        )
        
        # Simple flower class names for conditioning
        self.class_names = [
            'pink primrose',
            'hard-leaved pocket orchid',
            'canterbury bells',
            'sweet pea',
            'english marigold',
            'tiger lily',
            'moon orchid',
            'bird of paradise',
            'monkshood',
            'globe thistle',
            'snapdragon',
            "colt's foot",
            'king protea',
            'spear thistle',
            'yellow iris',
            'globe-flower',
            'purple coneflower',
            'peruvian lily',
            'balloon flower',
            'giant white arum lily',
            'fire lily',
            'pincushion flower',
            'fritillary',
            'red ginger',
            'grape hyacinth',
            'corn poppy',
            'prince of wales feathers',
            'stemless gentian',
            'artichoke',
            'sweet william',
            'carnation',
            'garden phlox',
            'love in the mist',
            'mexican aster',
            'alpine sea holly',
            'ruby-lipped cattleya',
            'cape flower',
            'great masterwort',
            'siam tulip',
            'lenten rose',
            'barbeton daisy',
            'daffodil',
            'sword lily',
            'poinsettia',
            'bolero deep blue',
            'wallflower',
            'marigold',
            'buttercup',
            'oxeye daisy',
            'common dandelion',
            'petunia',
            'wild pansy',
            'primula',
            'sunflower',
            'pelargonium',
            'bishop of llandaff',
            'gaura',
            'geranium',
            'orange dahlia',
            'pink-yellow dahlia?',
            'cautleya spicata',
            'japanese anemone',
            'black-eyed susan',
            'silverbush',
            'californian poppy',
            'osteospermum',
            'spring crocus',
            'bearded iris',
            'windflower',
            'tree poppy',
            'gazania',
            'azalea',
            'water lily',
            'rose',
            'thorn apple',
            'morning glory',
            'passion flower',
            'lotus',
            'toad lily',
            'anthurium',
            'frangipani',
            'clematis',
            'hibiscus',
            'columbine',
            'desert-rose',
            'tree mallow',
            'magnolia',
            'cyclamen',
            'watercress',
            'canna lily',
            'hippeastrum',
            'bee balm',
            'ball moss',
            'foxglove',
            'bougainvillea',
            'camellia',
            'mallow',
            'mexican petunia',
            'bromelia',
            'blanket flower',
            'trumpet creeper',
            'blackberry lily'
        ]
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        prompt = f"a photo of a {self.class_names[label]}"
        return {
            'pixel_values': image.to(torch.float16),  # Ensure float16 dtype
            'input_ids': prompt
        }

def setup_models(model_id="stabilityai/stable-diffusion-xl-base-1.0"):
    """Setup SDXL models with LoRA adapters"""
    
    # Load models
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=torch.float16)
    text_encoder_1 = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.float16)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(model_id, subfolder="text_encoder_2", torch_dtype=torch.float16)
    tokenizer_1 = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer_2")
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float16)
    
    # Setup LoRA for UNet
    unet_lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=0.1,
    )
    
    unet = get_peft_model(unet, unet_lora_config)
    
    return unet, text_encoder_1, text_encoder_2, tokenizer_1, tokenizer_2, noise_scheduler, vae

def encode_prompts(prompts, text_encoder_1, text_encoder_2, tokenizer_1, tokenizer_2, device):
    """Encode prompts using both CLIP text encoders for SDXL"""
    
    # Tokenize with first tokenizer
    text_inputs_1 = tokenizer_1(
        prompts,
        padding="max_length",
        max_length=tokenizer_1.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    
    # Tokenize with second tokenizer
    text_inputs_2 = tokenizer_2(
        prompts,
        padding="max_length",
        max_length=tokenizer_2.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    
    # Encode with both text encoders
    with torch.no_grad():
        # Text encoder 1 (CLIP ViT-L/14)
        prompt_embeds_1 = text_encoder_1(text_inputs_1.input_ids.to(device), output_hidden_states=True)
        prompt_embeds_1 = prompt_embeds_1.hidden_states[-2]  # Use penultimate layer
        
        # Text encoder 2 (CLIP ViT-G/14) 
        prompt_embeds_2 = text_encoder_2(text_inputs_2.input_ids.to(device), output_hidden_states=True)
        pooled_prompt_embeds = prompt_embeds_2[0]  # Pooled embeddings for micro-conditioning
        prompt_embeds_2 = prompt_embeds_2.hidden_states[-2]  # Use penultimate layer
    
    # Concatenate the embeddings from both encoders
    prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)
    
    # For SDXL, we also need to handle the pooled embeddings for micro-conditioning
    # Create time_ids (original size, crops coords, target size) - using default values
    batch_size = len(prompts)
    original_size = (1024, 1024)
    crops_coords_top_left = (0, 0)
    target_size = (1024, 1024)
    
    add_time_ids = torch.tensor([original_size + crops_coords_top_left + target_size]).repeat(batch_size, 1).to(device, dtype=torch.float16)
    
    # Concatenate pooled embeddings with time embeddings for micro-conditioning
    added_cond_kwargs = {
        "text_embeds": pooled_prompt_embeds,
        "time_ids": add_time_ids
    }
    
    return prompt_embeds, added_cond_kwargs

def train_step(batch, unet, text_encoder_1, text_encoder_2, tokenizer_1, tokenizer_2, noise_scheduler, vae, device):
    """Single training step"""
    
    pixel_values = batch['pixel_values'].to(device, dtype=torch.float16)
    prompts = batch['input_ids']
    
    # Encode images to latents
    with torch.no_grad():
        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
    
    # Encode prompts
    encoder_hidden_states, added_cond_kwargs = encode_prompts(prompts, text_encoder_1, text_encoder_2, tokenizer_1, tokenizer_2, device)
    
    # Sample noise and timesteps
    noise = torch.randn_like(latents, dtype=torch.float16)
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device)
    timesteps = timesteps.long()
    
    # Add noise to latents
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
    
    # Predict noise with added conditioning for SDXL
    model_pred = unet(
        noisy_latents, 
        timesteps, 
        encoder_hidden_states.to(torch.float16),
        added_cond_kwargs={k: v.to(torch.float16) for k, v in added_cond_kwargs.items()}
    ).sample
    
    # Calculate loss
    loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
    
    return loss

def main(args):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup wandb
    if args.use_wandb:
        wandb.init(project="sdxl-flowers102-finetune", config=args)
    
    # Load dataset
    train_dataset = Flowers102Dataset(split='train', size=args.resolution)
    val_dataset = Flowers102Dataset(split='val', size=args.resolution)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Setup models
    unet, text_encoder_1, text_encoder_2, tokenizer_1, tokenizer_2, noise_scheduler, vae = setup_models(args.model_id)
    
    # Move models to device
    unet = unet.to(device)
    text_encoder_1 = text_encoder_1.to(device)
    text_encoder_2 = text_encoder_2.to(device)
    vae = vae.to(device)
    
    # Set models to correct mode
    text_encoder_1.eval()
    text_encoder_2.eval()
    vae.eval()
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Training loop
    global_step = 0
    unet.train()
    
    for epoch in range(args.num_epochs):
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            loss = train_step(batch, unet, text_encoder_1, text_encoder_2, tokenizer_1, tokenizer_2, noise_scheduler, vae, device)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
            optimizer.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
            
            # Log to wandb
            if args.use_wandb:
                wandb.log({"train_loss": loss.item(), "global_step": global_step})
            
            # Save checkpoint
            if global_step % args.save_steps == 0:
                checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                os.makedirs(checkpoint_path, exist_ok=True)
                unet.save_pretrained(checkpoint_path)
        
        print(f"Epoch {epoch+1} completed. Average loss: {epoch_loss/len(train_loader):.4f}")
        
        # Validation
        if args.run_validation:
            unet.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    loss = train_step(batch, unet, text_encoder_1, text_encoder_2, tokenizer_1, tokenizer_2, noise_scheduler, vae, device)
                    val_loss += loss.item()
            
            val_loss_avg = val_loss / len(val_loader)
            print(f"Validation loss: {val_loss_avg:.4f}")
            
            if args.use_wandb:
                wandb.log({"val_loss": val_loss_avg, "epoch": epoch})
            
            unet.train()
    
    # Save final model
    final_path = os.path.join(args.output_dir, "final_model")
    os.makedirs(final_path, exist_ok=True)
    unet.save_pretrained(final_path)
    
    print("Training completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune SDXL on Flowers102 dataset")
    parser.add_argument("--model_id", default="stabilityai/stable-diffusion-xl-base-1.0", help="Model ID")
    parser.add_argument("--output_dir", default="./outputs", help="Output directory")
    parser.add_argument("--resolution", type=int, default=1024, help="Image resolution")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--run_validation", action="store_true", help="Run validation")
    parser.add_argument("--use_wandb", action="store_true", help="Use wandb for logging")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)
