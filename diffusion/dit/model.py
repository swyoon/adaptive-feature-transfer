import torch
from .models import DiT_models
from .diffusion import create_diffusion
from diffusers.models import AutoencoderKL


class DiT(torch.nn.Module):
    def __init__(
        self,
        model="DiT-XL/2",
        ckpt_path=None,
        image_size=256,
        num_classes=1000,
        vae="ema",
        num_steps=50,
        seed=0,
        batch_size=16,
    ):
        super().__init__()
 

        # torch.set_grad_enabled(False)

        # Setup DDP:
        torch.manual_seed(seed)
        self.batch_size = batch_size

        # Load model:
        self.latent_size = image_size // 8
        self.num_classes = num_classes
        self.model = DiT_models[model](
            input_size=self.latent_size,
            num_classes=self.num_classes
        )

        if ckpt_path is not None:
            state_dict = torch.load(ckpt_path, map_location=lambda storage, loc: storage, weights_only=False)
            if "ema" in state_dict:  # supports checkpoints from train.py
                state_dict = state_dict["ema"]
            self.model.load_state_dict(state_dict)
        self.model.eval()  # important!

        self.num_steps = num_steps
        self.diffusion = create_diffusion(str(self.num_steps))
        self.vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{vae}")

    def sample(self, latents=None, class_labels=None):
        return self._sample(self.batch_size, latents, class_labels)

    def _sample(self, batch_size, latents=None, class_labels=None):
        device = next(iter(self.model.parameters())).device
        # dtype = dtype_mapping[self.dtype]

        if latents is None:
            latents = torch.randn(batch_size, self.model.in_channels, self.latent_size, self.latent_size, device=device)
    
        if class_labels is not None:
            y = class_labels.nonzero(as_tuple=True)[1].to(device)
        else:
            y = torch.randint(0, self.num_classes, (batch_size,), device=device)
    
        # Setup classifier-free guidance:
        model_kwargs = dict(y=y)
        sample_fn = self.model.forward

        # Sample images:
        samples = self.diffusion.p_sample_loop(
            sample_fn, latents.shape, latents, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
        )
        
        samples = self.vae.decode(samples / 0.18215).sample

        return samples.to(torch.float32), y
