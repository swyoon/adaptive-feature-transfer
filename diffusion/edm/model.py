import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from typing import Optional, Union

from . import dnnlib
from .fkd import FKD, FKD_ARGS_DEFAULTS

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

dtype_mapping = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
}


class EDM(nn.Module):
    def __init__(
        self,
        network_pkl,
        seed=0,
        batch_size=256,
        num_steps=18,
        sigma_min=0.002,
        sigma_max=80,
        rho=7,
        S_churn=0,
        S_min=0,
        S_max=float("inf"),
        S_noise=1,
        dtype="float32",
    ):
        super().__init__()
        torch.manual_seed(seed)

        # Load network.
        print(f'Loading network from "{network_pkl}"...')
        with dnnlib.util.open_url(network_pkl) as f:
            net = pickle.load(f)["ema"]

        self.net = net
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.S_churn = S_churn
        self.S_min = S_min
        self.S_max = S_max
        self.S_noise = S_noise
        self.dtype = dtype

        if self.dtype == "float16":
            self.net.use_fp16 = True

    def sample(self, latents=None, class_labels=None):
        return self._sample(self.batch_size, latents, class_labels)

    def _sample(self, batch_size, latents=None, class_labels=None):
        device = next(iter(self.net.parameters())).device
        dtype = dtype_mapping[self.dtype]

        if latents is None:
            latents = torch.randn(
                [batch_size, self.net.img_channels, self.net.img_resolution, self.net.img_resolution],
                device=device,
            )
        if class_labels is None:
            if self.net.label_dim:
                class_labels = torch.eye(self.net.label_dim, device=device)[
                    torch.randint(self.net.label_dim, size=[batch_size], device=device)
                ]

        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(self.sigma_min, self.net.sigma_min)
        sigma_max = min(self.sigma_max, self.net.sigma_max)

        # Time step discretization.
        step_indices = torch.arange(self.num_steps, dtype=dtype, device=device)
        t_steps = (
            sigma_max ** (1 / self.rho)
            + step_indices
            / (self.num_steps - 1)
            * (sigma_min ** (1 / self.rho) - sigma_max ** (1 / self.rho))
        ) ** self.rho
        t_steps = torch.cat([self.net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])  # t_N = 0
        # Main sampling loop.
        x_next = latents.to(dtype) * t_steps[0]
        for i, (t_cur, t_next) in list(enumerate(zip(t_steps[:-1], t_steps[1:]))):  # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = (
                min(self.S_churn / self.num_steps, np.sqrt(2) - 1)
                if self.S_min <= t_cur <= self.S_max
                else 0
            )
            t_hat = self.net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * self.S_noise * torch.randn_like(x_cur)

            # Euler step.
            with torch.autocast(device_type=device.type, dtype=dtype):
                denoised = self.net(x_hat, t_hat, class_labels).to(dtype)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < self.num_steps - 1:
                with torch.autocast(device_type=device.type, dtype=dtype):
                    denoised = self.net(x_next, t_next, class_labels).to(dtype)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next.to(torch.float32), class_labels.nonzero(as_tuple=True)[1]

    def sample_fk_steering(
        self,
        latents=None,
        class_label:  Optional[Union[int, list]] = None,
        reward_fn=None,
        reward_fn_args: dict = {},
        fkd_args: dict = FKD_ARGS_DEFAULTS,
    ):
        device = next(iter(self.net.parameters())).device
        dtype = dtype_mapping[self.dtype]

        output_dir = fkd_args.get("output_dir", "./fkd_results")
        # set EDM parameters to fkd_args
        self.batch_size = fkd_args["num_particles"]
        if "resampling_t_end" in fkd_args:
            assert fkd_args["resampling_t_end"] == self.num_steps, "resampling_t_end in fkd_args must equal to self.num_steps"
        fkd_args["resampling_t_end"] = self.num_steps

        assert self.S_churn > 0 and self.S_noise > 0, "For FKD steering, S_churn and S_noise must be greater than 0."

        if latents is None:
            latents = torch.randn(
                [
                    1,
                    self.net.img_channels,
                    self.net.img_resolution,
                    self.net.img_resolution,
                ],
                device=device,
            )
            latents = latents.repeat(self.batch_size, 1, 1, 1)
        if class_label is None:
            if self.net.label_dim:
                class_label = torch.randint(self.net.label_dim, (), device="cpu").item()
        if isinstance(class_label, int):
            if self.net.label_dim:
                class_labels = torch.eye(self.net.label_dim, device=device)[
                    torch.tensor([class_label] * self.batch_size, device=device)
                ]

        reward_fn_args["targets"] = torch.tensor([class_label] * self.batch_size, device=device) # TODO: add conditioning
            
        # Set up for FK-steering

        if fkd_args is not None and fkd_args.get("use_smc", False):
            self.fkd = FKD(
                reward_fn=reward_fn,
                **fkd_args,
            )
        else:
            self.fkd = None

        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(self.sigma_min, self.net.sigma_min)
        sigma_max = min(self.sigma_max, self.net.sigma_max)

        # Time step discretization.
        step_indices = torch.arange(self.num_steps, dtype=dtype, device=device)
        t_steps = (
            sigma_max ** (1 / self.rho)
            + step_indices
            / (self.num_steps - 1)
            * (sigma_min ** (1 / self.rho) - sigma_max ** (1 / self.rho))
        ) ** self.rho
        t_steps = torch.cat([self.net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])  # t_N = 0
        # Main sampling loop.
        x_next = latents.to(dtype) * t_steps[0]

        for i, (t_cur, t_next) in list(enumerate(zip(t_steps[:-1], t_steps[1:]))):  # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = (
                min(self.S_churn / self.num_steps, np.sqrt(2) - 1)
                if self.S_min <= t_cur <= self.S_max
                else 0
            )
            t_hat = self.net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * self.S_noise * torch.randn_like(x_cur)

            # Euler step.
            with torch.autocast(device_type=device.type, dtype=dtype):
                denoised = self.net(x_hat, t_hat, class_labels).to(dtype)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < self.num_steps - 1:
                with torch.autocast(device_type=device.type, dtype=dtype):
                    denoised = self.net(x_next, t_next, class_labels).to(dtype)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
                x0_for_reward = denoised
            else:
                x0_for_reward = denoised

            # FK steering
            if self.fkd is not None:
                # latents( current population) and x0 predictions are required
                # in EDM, "latents" == image-sapce sample x; x0_preds == x0 (clean)
                x_next, current_pop_images = self.fkd.resample(
                    sampling_idx=i,
                    latents=x_next,
                    x0_preds=x0_for_reward,
                    reward_fn_args=reward_fn_args
                )
                # visualize and save it in the ouput_dir
                if fkd_args["print_rewards"] and current_pop_images is not None:
                    print(f"For class{class_label} Step {i}, rewards: ", self.fkd.population_rs.tolist())
                if fkd_args["visualize_intermediate"] and current_pop_images is not None:
                    imgs = self._to_pil_from_tensor_batch(current_pop_images)
                    for k, img in enumerate(imgs):
                        # imgs are ordered by rewards (high to low)
                        os.makedirs(f"{output_dir}/candidates/", exist_ok=True)
                        img.save(
                            f"{output_dir}/candidates/class{class_label}_step{i}_{k}_r{self.fkd.population_rs.tolist()[k]:.0f}.png"
                        )
                if fkd_args["visualzie_x0"]:
                    for k, img in enumerate(x0_for_reward):
                        img = self._to_pil_from_tensor_batch(img.unsqueeze(0))
                        os.makedirs(f"{output_dir}/x0", exist_ok=True)
                        img[0].save(f"{output_dir}/x0/class{class_label}_step{i}_{k}.png")

        if self.fkd is not None:
            rewards = self.fkd.population_rs.tolist()
            max_index = np.argmax(rewards)
        else:
            rewards = [0] * self.batch_size
            max_index = 0  # if not using FKD, just return the first sample

        return (
            x_next[max_index].unsqueeze(0).to(torch.float32),
            class_labels.nonzero(as_tuple=True)[1][max_index].unsqueeze(0),
            rewards[max_index],
        )

    def sample_gradient_guidance(
        self,
        batch_size,
        func=None,
        func_kwargs={},
        guidance_scale=1.0,
        guide_every=1,
        latents=None,
        class_labels=None,
        return_dict=False,
    ):
        """
        func: a function that takes a tensor and returns a tensor
        latents: (B, C, H, W)
        reference: https://github.com/AlexMaOLS/EluCD/blob/0b79a4ecca7d0ed2590d63c21363a9f4a487415c/EDM/generate.py#L317
        related paper: https://openreview.net/forum?id=9DXXMXnIGm
        """
        device = next(iter(self.net.parameters())).device
        dtype = dtype_mapping[self.dtype]
        if latents is None:
            latents = torch.randn(
                [batch_size, self.net.img_channels, self.net.img_resolution, self.net.img_resolution],
                device=device,
            )
        if class_labels is None:
            if self.net.label_dim:
                class_labels = torch.eye(self.net.label_dim, device=device)[
                    torch.randint(self.net.label_dim, size=[batch_size], device=device)
                ]

        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(self.sigma_min, self.net.sigma_min)
        sigma_max = min(self.sigma_max, self.net.sigma_max)

        # Time step discretization.
        step_indices = torch.arange(self.num_steps, dtype=dtype, device=device)
        t_steps = (
            sigma_max ** (1 / self.rho)
            + step_indices
            / (self.num_steps - 1)
            * (sigma_min ** (1 / self.rho) - sigma_max ** (1 / self.rho))
        ) ** self.rho
        t_steps = torch.cat([self.net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])  # t_N = 0

        # Main sampling loop.
        x_next = latents.to(dtype) * t_steps[0]
        l_sample = [x_next.detach().cpu()]
        for i, (t_cur, t_next) in list(enumerate(zip(t_steps[:-1], t_steps[1:]))):  # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = (
                min(self.S_churn / self.num_steps, np.sqrt(2) - 1)
                if self.S_min <= t_cur <= self.S_max
                else 0
            )
            t_hat = self.net.round_sigma(t_cur + gamma * t_cur)

            x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * self.S_noise * torch.randn_like(x_cur)

            # gradient guidance
            if func is not None and i % guide_every == 0:
                func_kwargs["targets"] = class_labels.nonzero(as_tuple=True)[1].detach()

                x_hat = x_hat.detach().requires_grad_(True)

                with torch.enable_grad():
                    with torch.autocast(device_type=device.type, dtype=dtype):
                        denoised = self.net(x_hat, t_hat, class_labels).to(dtype)

                    x0 = denoised.float()
                    x0 = torch.clamp(denoised, -1, 1) * 0.5 + 0.5
                    f = func(x0, **func_kwargs)

                    grads = torch.autograd.grad(f.sum(), x_hat, retain_graph=True)[0]

                    x_hat_norm = ((x_hat**2).sum((1, 2, 3), keepdim=True)) ** 0.5
                    grads_norm = ((grads**2).sum((1, 2, 3), keepdim=True)) ** 0.5

                    grads = grads / (grads_norm + 1e-8) * x_hat_norm * guidance_scale
                grads = grads.detach()

            else:
                grads = 0 

            x_hat = x_hat + grads
            
            # Euler step.
            with torch.autocast(device_type=device.type, dtype=dtype):
                denoised = self.net(x_hat, t_hat, class_labels).to(dtype)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < self.num_steps - 1:
                with torch.autocast(device_type=device.type, dtype=dtype):
                    denoised = self.net(x_next, t_next, class_labels).to(dtype)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
            l_sample.append(x_next.detach().cpu())

        if return_dict:
            return {
                "sample": x_next.to(torch.float32),
                "l_sample": l_sample,
                "class_labels": class_labels.nonzero(as_tuple=True)[1],
            }
        else:
            return x_next.to(torch.float32), class_labels.nonzero(as_tuple=True)[1]

    def sample_softlabel(self, batch_size, class_label1, class_label2, weight1, weight2, latents=None):
        """
        class_label1: (B, C) one-hot vector
        class_label2: (B, C) one-hot vector
        weight1: (B,)
        weight2: (B,)
        latents: (B, C, H, W)
        """
        device = next(iter(self.net.parameters())).device
        dtype = dtype_mapping[self.dtype]

        weight1 = weight1.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        weight2 = weight2.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        if latents is None:
            latents = torch.randn(
                [batch_size, self.net.img_channels, self.net.img_resolution, self.net.img_resolution],
                device=device,
            )
        # if class_labels is None:
        #     if self.net.label_dim:
        #         class_labels = torch.eye(self.net.label_dim, device=device)[
        #             torch.randint(self.net.label_dim, size=[batch_size], device=device)
        #         ]

        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(self.sigma_min, self.net.sigma_min)
        sigma_max = min(self.sigma_max, self.net.sigma_max)

        # Time step discretization.
        step_indices = torch.arange(self.num_steps, dtype=dtype, device=device)
        t_steps = (
            sigma_max ** (1 / self.rho)
            + step_indices
            / (self.num_steps - 1)
            * (sigma_min ** (1 / self.rho) - sigma_max ** (1 / self.rho))
        ) ** self.rho
        t_steps = torch.cat([self.net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])  # t_N = 0
        # Main sampling loop.
        x_next = latents.to(dtype) * t_steps[0]
        for i, (t_cur, t_next) in list(enumerate(zip(t_steps[:-1], t_steps[1:]))):  # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = (
                min(self.S_churn / self.num_steps, np.sqrt(2) - 1)
                if self.S_min <= t_cur <= self.S_max
                else 0
            )
            t_hat = self.net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * self.S_noise * torch.randn_like(x_cur)

            # Euler step.
            with torch.autocast(device_type=device.type, dtype=dtype):
                denoised_1 = self.net(x_hat, t_hat, class_label1).to(dtype)
                denoised_2 = self.net(x_hat, t_hat, class_label2).to(dtype)
                denoised = denoised_1 * weight1 + denoised_2 * weight2
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < self.num_steps - 1:
                with torch.autocast(device_type=device.type, dtype=dtype):
                    denoised_1 = self.net(x_next, t_next, class_label1).to(dtype)
                    denoised_2 = self.net(x_next, t_next, class_label2).to(dtype)
                    denoised = denoised_1 * weight1 + denoised_2 * weight2
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next.to(torch.float32)  # , class_labels.nonzero(as_tuple=True)[1]

    def sample_softlabel_deprecated(self, batch_size, class_labels=None, latents=None):
        """
        An attemp to directly interpolate the class label vectors.
        class_labels: (B, C)
        """
        device = next(iter(self.net.parameters())).device
        dtype = dtype_mapping[self.dtype]

        if latents is None:
            latents = torch.randn(
                [batch_size, self.net.img_channels, self.net.img_resolution, self.net.img_resolution],
                device=device,
            )
        if class_labels is None:
            if self.net.label_dim:
                class_labels = torch.eye(self.net.label_dim, device=device)[
                    torch.randint(self.net.label_dim, size=[batch_size], device=device)
                ]

        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(self.sigma_min, self.net.sigma_min)
        sigma_max = min(self.sigma_max, self.net.sigma_max)

        # Time step discretization.
        step_indices = torch.arange(self.num_steps, dtype=dtype, device=device)
        t_steps = (
            sigma_max ** (1 / self.rho)
            + step_indices
            / (self.num_steps - 1)
            * (sigma_min ** (1 / self.rho) - sigma_max ** (1 / self.rho))
        ) ** self.rho
        t_steps = torch.cat([self.net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])  # t_N = 0
        # Main sampling loop.
        x_next = latents.to(dtype) * t_steps[0]
        for i, (t_cur, t_next) in list(enumerate(zip(t_steps[:-1], t_steps[1:]))):  # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = (
                min(self.S_churn / self.num_steps, np.sqrt(2) - 1)
                if self.S_min <= t_cur <= self.S_max
                else 0
            )
            t_hat = self.net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * self.S_noise * torch.randn_like(x_cur)

            # Euler step.
            with torch.autocast(device_type=device.type, dtype=dtype):
                denoised = self.net(x_hat, t_hat, class_labels).to(dtype)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < self.num_steps - 1:
                with torch.autocast(device_type=device.type, dtype=dtype):
                    denoised = self.net(x_next, t_next, class_labels).to(dtype)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next.to(torch.float32), class_labels.nonzero(as_tuple=True)[1]

    def sample_mixup(self, batch_size, class_label1, class_label2, weight1, weight2, latents=None):
        samples1 = self._sample(batch_size, latents, class_label1)[0]
        samples2 = self._sample(batch_size, latents, class_label2)[0]

        samples = samples1 * weight1[:, None, None, None] + samples2 * weight2[:, None, None, None]
        return samples

    def _to_pil_from_tensor_batch(self, x):
        """
        It maps x, samples from EDM, to [0, 255] for visualization.
        x: (B, C, H, W), image-space float tensor.
        """
        x = x.detach().float().clamp(-1, 1)
        x = (x * 0.5) + 0.5  # [-1,1] -> [0,1]
        x = (x * 255).clamp(0, 255).to(torch.uint8)
        x = x.permute(0, 2, 3, 1).cpu().numpy()  # (B,H,W,C)
        return [Image.fromarray(arr) for arr in x]