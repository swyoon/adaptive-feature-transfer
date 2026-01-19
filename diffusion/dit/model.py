import warnings

from typing import Optional, Union
from copy import deepcopy

import torch
import numpy as np

from .models import DiT_models
from .diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from ..edm.fkd import FKD, FKD_ARGS_DEFAULTS

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


    def sample_fk_steering(
        self,
        latents=None,
        class_label:  Optional[Union[int, list]] = None,
        reward_fn=None,
        reward_fn_args: dict = {},
        fkd_args: dict = FKD_ARGS_DEFAULTS,
    ):
        device = next(iter(self.model.parameters())).device
        # dtype = dtype_mapping[self.dtype]

        fkd_args = deepcopy(fkd_args)
        self.batch_size = fkd_args["num_particles"]
        if fkd_args["resampling_t_start"] == 0:
            # raise NotImplementedError("resampling_t_start==0 is not implemented")
            warnings.warn("fkd_args['resampling_t_start']=0 is not implemented, setting to 1")
            fkd_args["resampling_t_start"] = 1
        if fkd_args["resampling_t_end"] is None:
            fkd_args["resampling_t_end"] = self.num_steps

        fkd_args["time_steps"] = self.num_steps

        if latents is None:
            latents = torch.randn(self.batch_size, self.model.in_channels, self.latent_size, self.latent_size, device=device)
    

        if class_label is None:
            class_label = torch.randint(self.num_classes, (), device="cpu").item()
        if isinstance(class_label, int):
            y = torch.full((self.batch_size,), class_label, device=device, dtype=torch.long)
        

        reward_fn_args["targets"] = y
        # Set up for FK-steering
        reward_fn_args["prompts"] = class_label

        if fkd_args is not None and fkd_args.get("use_smc", False):
            latent_to_decode_fn_ = fkd_args.get("latent_to_decode_fn", None)
            def latent_to_decode_fn(latents):
                samples = self.vae.decode(latents / 0.18215).sample

                if latent_to_decode_fn_ is not None:
                    samples = latent_to_decode_fn_(samples)
                return samples
            fkd_args["latent_to_decode_fn"] = latent_to_decode_fn

            fkd = FKD(
                reward_fn=reward_fn,
                **fkd_args,
            )
        else:
            fkd = None

    
        # Setup classifier-free guidance:
        model_kwargs = dict(y=y)
        sample_fn = self.model.forward

        # Sample images:
        samples = self.diffusion.p_sample_loop_fkd(
            sample_fn, latents.shape, latents, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device, fkd=fkd, reward_fn_args=reward_fn_args
        )
        
        samples = self.vae.decode(samples / 0.18215).sample

        if fkd is not None:
            population_images = latent_to_decode_fn_(samples)
            rewards = fkd.reward_fn(population_images, **reward_fn_args).cpu().tolist()
            max_index = np.argmax(rewards)
        else:
            rewards = [0] * self.batch_size
            max_index = 0  # if not using FKD, just return the first sample

        return (
            samples[max_index].unsqueeze(0).to(torch.float32),
            y[max_index].unsqueeze(0),
            rewards[max_index],
        )
