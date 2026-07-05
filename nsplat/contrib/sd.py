#!/usr/bin/env python3
"""
 )   ___
(__/_____)     /) /) ,                 /)
  /       _   // //    _   __  _  __  (/
 /       (_(_(/_(/__(_(_/_/ (_(_(_/_)_/ )_
(______)             .-/       .-/
                    (_/       (_/

Stable diffusion and score distillation sampling utils
"""

from diffusers import (
    DiffusionPipeline,
    StableDiffusionControlNetPipeline,
    AutoPipelineForText2Image,
    StableDiffusionPipeline,
    ControlNetModel,
    DPMSolverMultistepScheduler,
    DDIMScheduler,
    DDIMInverseScheduler,
    DEISMultistepScheduler,
    StableDiffusionControlNetImg2ImgPipeline,
)
import torch.nn as nn
import torch
from torch.cuda.amp import custom_bwd, custom_fwd
from diffusers.utils.import_utils import is_xformers_available
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
from packaging import version
from PIL import Image
from .sd_step import *


def parse_version(v):
    return version.parse(v)


import random
import warnings
from contextlib import contextmanager

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Simple no-op perf timer to replace util.perf_timer
@contextmanager
def perf_timer(name, verbose=False):
    """No-op performance timer context manager."""
    yield


cfg = lambda: None
cfg.batch_size = 1
cfg.use_xformers = False
cfg.del_text_encoders = True
cfg.enable_memory_efficient_attention = True
cfg.enable_sequential_cpu_offload = False
cfg.enable_attention_slicing = False
cfg.enable_channels_last_format = False
cfg.weighting_strategy = "sds"
cfg.t_power = 0.5  # 2.0
cfg.noise_free = True  # Enables noise free score distillation (Katzir et al.,2023
cfg.show_perf_time = False

# Additional parameters for ISM
guidance_opt = lambda: None
guidance_opt.xs_delta_t = 100
guidance_opt.xs_inv_steps = 3
guidance_opt.xs_eta = 0.0
guidance_opt.denoise_guidance_scale = 1.5
guidance_opt.delta_t = 150
guidance_opt.delta_t_start = 80
guidance_opt.max_t_range = 0.98
guidance_opt.min_t_range = 0.02
guidance_opt.warmup_end_t = 0.5  # 5 #0.61


def to_batch(x, rgb, batch_size=1):
    if isinstance(x, Image.Image):
        x = torch.tensor(np.array(x) / 255, device=device)
        rgb = True
    elif isinstance(x, np.ndarray):
        x = torch.tensor(x, device=device)
    if rgb:
        if len(x.shape) == 3:
            x = x[:, :, :, np.newaxis]
        x = x.permute((3, 2, 0, 1))  # to NCHW
    else:
        if len(x.shape) == 2:
            x = x[np.newaxis, np.newaxis, :, :]
        x = x.repeat(1, 3, 1, 1)
    x = x.repeat(batch_size, 1, 1, 1)
    return x


# From https://github.com/fudan-zvg/PGC-3D/blob/14b96533a2711d421be2d3bbe4f35ae13d435e7d/guidance/guidance_utils.py
def w_star(t, m1=800, m2=500, s1=300, s2=100):
    # max time 1000
    r = np.ones_like(t) * 1.0
    r[t > m1] = np.exp(-((t[t > m1] - m1) ** 2) / (2 * s1 * s1))
    r[t < m2] = np.exp(-((t[t < m2] - m2) ** 2) / (2 * s2 * s2))
    return r


def precompute_prior(T=1000, min_t=200, max_t=800, **kwargs):
    ts = np.arange(T)
    prior = w_star(ts, **kwargs)[min_t:max_t]
    prior = prior / prior.sum()
    prior = prior[::-1].cumsum()[::-1]
    return prior, min_t


def time_prioritize(step_ratio, time_prior, min_t=200):
    return np.abs(time_prior - step_ratio).argmin() + min_t


class StableDiffusion(nn.Module):
    _global_pipe = None

    def __init__(
        self,
        caption,
        version="1.5",
        timesteps=None,
        rgb=False,
        size=(512, 512),
        controlnet="",
        guidance_scale=7.5,
        conditioning_scale=0.4,
        guess_mode=True,
        guess_mode_reweight=False,
        grad_method="sds",  # 'ism'
        time_schedule="dreamtime",  # 'dtc'
        batch_size=1,
        t_range=[0.4, 0.98],
        seed=61232,
        num_hifa_denoise_steps=4,
        # negative_prompt="oversaturated color, tiling, lowres, bad quality, worst_quality, messy",
        negative_prompt="unrealistic, blurry, low quality, out of focus, ugly, low contrast, dull, dark, low-resolution, gloomy",
        null_prompt=None,
        ip_adapter="",
        ip_adapter_scale=1.0,
        multistep_scheduler=False,
    ):
        super(StableDiffusion, self).__init__()
        self.time_schedule = time_schedule
        self.t_range = t_range
        self.grad_method = grad_method
        self.device = device
        self.rgb = rgb
        self.guidance_scale = guidance_scale
        self.conditioning_scale = conditioning_scale
        self.size = size
        self.guess_mode = guess_mode
        self.guess_mode_reweight = guess_mode_reweight
        self.caption = caption
        self.negative_prompt = negative_prompt
        self.null_prompt = null_prompt
        self.num_hifa_denoise_steps = num_hifa_denoise_steps
        self.batch_size = batch_size

        if controlnet:
            print("Forcing SD version to 1.5 with controlnet")
            version = "1.5"

        if version == "2.1":
            self.model_key = "stabilityai/stable-diffusion-2-1-base"
        elif version == "2.0":
            self.model_key = "stabilityai/stable-diffusion-2-base"
        elif version == "1.5":
            self.model_key = "runwayml/stable-diffusion-v1-5"
        else:
            raise ValueError(
                f"Stable-diffusion version {self.sd_version} not supported."
            )

        self.weights_dtype = torch.float16

        self.seed = seed
        self.generator = torch.Generator(device=device).manual_seed(self.seed)

        # Pipeline
        self.init_pipe(controlnet, ip_adapter)

        if multistep_scheduler:
            self.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config, torch_dtype=self.weights_dtype
            )  # Faster scheduler
        else:
            # # DDIM scheduling
            self.scheduler = DDIMScheduler.from_config(
                self.pipe.scheduler.config, torch_dtype=self.weights_dtype
            )

        print("Scheduler:")
        print(self.scheduler)
        if timesteps is not None:
            self.scheduler.set_timesteps(timesteps)
            self.num_timesteps = timesteps
        else:
            self.num_timesteps = self.scheduler.config.num_train_timesteps
            self.scheduler.set_timesteps(self.num_timesteps)
        # Timesteps for ISM
        self.timesteps = torch.flip(self.scheduler.timesteps, dims=(0,)).to(device)

        # Time prioro for dreamtime schedule
        self.time_prior, _ = precompute_prior(
            # min_t=int(self.t_range[0]*self.num_timesteps),
            max_t=int(self.t_range[1] * self.num_timesteps),
            m2=t_range[0] * self.num_timesteps,
        )

        self.ind_t = 0

        if cfg.enable_memory_efficient_attention:
            if parse_version(torch.__version__) >= parse_version("2"):
                print("PyTorch2.0 uses memory efficient attention by default.")
            elif not is_xformers_available():
                print(
                    "xformers is not available, memory efficient attention is not enabled."
                )
            else:
                self.pipe.enable_xformers_memory_efficient_attention()

        if cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()

        if cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)

        if cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)

        # Create model
        self.vae = self.pipe.vae
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)

        if cfg.use_xformers:
            self.pipe.enable_xformers_memory_efficient_attention()

        self.text_embeddings = self.embed_text(
            caption, negative_prompt=negative_prompt, null_prompt=null_prompt
        )

        if ip_adapter:
            self.pipe.set_ip_adapter_scale(ip_adapter_scale)

        self.unet = self.pipe.unet
        self.added_cond_kwargs = None
        self.t_saved = 0

    def release(self):
        del self.pipe
        self.pipe = None
        if self.controlnet is not None:
            del self.controlnet
            self.controlnet = None

    @property
    def has_controlnet(self):
        return self.controlnet is not None

    def init_pipe(self, controlnet_model, ip_adapter):
        if controlnet_model:
            controlnet = ControlNetModel.from_pretrained(
                controlnet_model,
                torch_dtype=self.weights_dtype,
            ).to(device)
            self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                self.model_key,
                torch_dtype=self.weights_dtype,
                # variant="fp16",
                controlnet=controlnet,
                safety_checker=None,
            )
            self.controlnet = self.pipe.controlnet  # .eval()
        else:
            self.pipe = AutoPipelineForText2Image.from_pretrained(
                self.model_key, torch_dtype=self.weights_dtype
            )  # , variant="fp16")
            self.controlnet = None
        self.pipe = self.pipe.to(self.device)

        if ip_adapter:
            if "xl" in ip_adapter:
                self.pipe.load_ip_adapter(
                    "h94/IP-Adapter", subfolder="sdxl_models", weight_name=ip_adapter
                )
            else:
                self.pipe.load_ip_adapter(
                    "h94/IP-Adapter", subfolder="models", weight_name=ip_adapter
                )

    def set_prompt(self, prompt, negative_prompt=None):
        if negative_prompt is None:
            negative_prompt = self.negative_prompt
        else:
            self.negative_prompt = negative_prompt
        self.caption = prompt
        self.text_embeddings = self.embed_text(
            prompt, negative_prompt=negative_prompt, null_prompt=self.null_prompt
        )

    def set_seed(self, seed):
        self.seed = seed
        self.generator = torch.Generator(device=device).manual_seed(self.seed)

    def set_ip_adapter_scale(self, scale):
        self.pipe.set_ip_adapter_scale(scale)

    def generate_image(
        self,
        image_cond=None,
        conditioning_scale=None,
        guidance_scale=None,
        guess_mode=None,
        num_inference_steps=20,
        seed=None,
        ip_adapter_image=None,
    ):
        if conditioning_scale is None:
            conditioning_scale = self.conditioning_scale
        if guidance_scale is None:
            guidance_scale = self.guidance_scale
        if seed is None:
            generator = self.generator
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        if guess_mode is None:
            guess_mode = self.guess_mode

        if self.controlnet is not None:
            image_cond = to_batch(image_cond, False)
            img = self.pipe(
                self.caption,
                generator=generator,
                image=image_cond,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                ip_adapter_image=ip_adapter_image,
                controlnet_conditioning_scale=conditioning_scale,
                guess_mode=guess_mode,
                negative_prompt=self.negative_prompt,
            ).images[0]
        else:
            img = self.pipe(
                self.caption,
                generator=generator,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=conditioning_scale,
                ip_adapter_image=ip_adapter_image,
                guess_mode=guess_mode,
                negative_prompt=self.negative_prompt,
            ).images[0]
        # Reset seed
        self.generator = torch.Generator(device=device).manual_seed(self.seed)
        return img

    def encode_images(self, imgs):
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents

    def decode_latent(self, latent):
        latent = latent.to(self.weights_dtype)
        x = self.vae.decode(latent / 0.18215).sample  # .to(self.config.dtype)
        x = (x / 2 + 0.5).clamp(0, 1)
        return x

    def add_noise_with_cfg(
        self,
        latents,
        noise,
        ind_t,
        ind_prev_t,
        text_embeddings=None,
        cfg=1.0,
        delta_t=1,
        inv_steps=1,
        is_noisy_latent=False,
        eta=0.0,
    ):
        # From https://github.com/EnVision-Research/LucidDreamer/blob/main/guidance/sd_utils.py
        batch_size = latents.shape[0]
        text_embeddings = text_embeddings.to(self.weights_dtype)
        if cfg <= 1.0:
            uncond_text_embedding = text_embeddings.reshape(
                2, -1, text_embeddings.shape[-2], text_embeddings.shape[-1]
            )[1]

        unet = self.unet

        if is_noisy_latent:
            prev_noisy_lat = latents
        else:
            prev_noisy_lat = self.scheduler.add_noise(
                latents, noise, self.timesteps[ind_prev_t]
            )

        cur_ind_t = ind_prev_t
        cur_noisy_lat = prev_noisy_lat

        pred_scores = []

        for i in range(inv_steps):
            # pred noise
            cur_noisy_lat_ = self.scheduler.scale_model_input(
                cur_noisy_lat, self.timesteps[cur_ind_t]
            ).to(self.weights_dtype)

            if cfg > 1.0:
                latent_model_input = torch.cat([cur_noisy_lat_, cur_noisy_lat_])
                timestep_model_input = (
                    self.timesteps[cur_ind_t]
                    .reshape(1, 1)
                    .repeat(latent_model_input.shape[0], 1)
                    .reshape(-1)
                )
                unet_output = unet(
                    latent_model_input,
                    timestep_model_input,
                    encoder_hidden_states=text_embeddings,
                    added_cond_kwargs=self.added_cond_kwargs,
                ).sample

                uncond, cond = torch.chunk(unet_output, chunks=2)
                unet_output = cond + cfg * (
                    uncond - cond
                )  # reverse cfg to enhance the distillation
            else:
                timestep_model_input = (
                    self.timesteps[cur_ind_t]
                    .reshape(1, 1)
                    .repeat(cur_noisy_lat_.shape[0], 1)
                    .reshape(-1)
                )
                unet_output = unet(
                    cur_noisy_lat_,
                    timestep_model_input,
                    encoder_hidden_states=uncond_text_embedding,
                    added_cond_kwargs=self.added_cond_kwargs,
                ).sample

            pred_scores.append((cur_ind_t, unet_output))

            next_ind_t = min(cur_ind_t + delta_t, ind_t)
            cur_t, next_t = self.timesteps[cur_ind_t], self.timesteps[next_ind_t]
            delta_t_ = (
                next_t - cur_t
                if isinstance(self.scheduler, DDIMScheduler)
                else next_ind_t - cur_ind_t
            )
            cur_noisy_lat = ddim_step(
                self.scheduler, unet_output, cur_t, cur_noisy_lat, -delta_t_, eta
            ).prev_sample
            cur_ind_t = next_ind_t

            del unet_output
            torch.cuda.empty_cache()

            if cur_ind_t == ind_t:
                break

        return prev_noisy_lat, cur_noisy_lat, pred_scores[::-1]

    def compute_grad_sds(
        self, latents, image_cond, step=None, num_steps=None, ip_adapter_image=None
    ):

        if step is not None and num_steps is not None:
            step_ratio = step / num_steps
            warmup_steps = int(num_steps * guidance_opt.warmup_end_t)  # /2.5)
            warm_up_rate = 1.0 - min(step / warmup_steps, 1)
        else:
            step_ratio = None

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        min_step = max(int(self.t_range[0] * self.num_timesteps), 1)
        max_step = min(
            int(self.t_range[1] * self.num_timesteps), self.num_timesteps - 1
        )
        warmup_end_t = (
            self.t_range[0]
            + (self.t_range[1] - self.t_range[0]) * guidance_opt.warmup_end_t
        )
        warmup_end_step = int(warmup_end_t * self.num_timesteps)
        warmup_step = max_step - warmup_end_step

        if step_ratio is None or self.time_schedule == "random":
            t = torch.randint(
                min_step,
                max_step + 1,
                (latents.shape[0],),
                dtype=torch.long,
                device=self.device,
                generator=self.generator,
            )
        else:
            if self.time_schedule == "ism":
                delta_t = 50
                delta_t_start = 150
                current_delta_t = guidance_opt.delta_t
                ind_t = torch.randint(
                    min_step,
                    warmup_end_step + int(warmup_step * warm_up_rate),
                    (1,),
                    dtype=torch.long,
                    generator=self.generator,
                    device=device,
                )[0]
                ind_prev_t = max(ind_t - current_delta_t, torch.ones_like(ind_t) * 0)
                prev_t = self.timesteps[ind_prev_t]
            elif self.time_schedule == "dtc":
                # Based on
                # Zhang, Li & Zhang et al. (2024) HumanRef: Single Image to 3D Human Generation via Reference-Guided Diffusion
                # Seems to work well
                l = 1000 / num_steps
                delta = 2.0 / num_steps
                tmid = guidance_opt.max_t_range - (
                    guidance_opt.max_t_range - self.t_range[0]
                ) * np.log(1 + (np.floor(step / l) * l) / num_steps)
                mint = tmid - delta
                maxt = tmid + delta
                ind_t = torch.randint(
                    max(int(mint * self.num_timesteps), 1),
                    min(int(maxt * self.num_timesteps), self.num_timesteps),
                    (1,),
                    dtype=torch.long,
                    generator=self.generator,
                    device=self.device,
                )[0]
            elif self.time_schedule == "dreamtime":
                t = time_prioritize(step_ratio, self.time_prior)
                ind_t = torch.tensor(t, dtype=torch.long, device=self.device)
            elif self.time_schedule == "pow":
                t = int(
                    np.clip(
                        (
                            self.t_range[1]
                            - (self.t_range[1] - self.t_range[0])
                            * (step_ratio**cfg.t_power)
                        )
                        * self.num_timesteps,
                        0,
                        self.num_timesteps - 1,
                    )
                )
                ind_t = torch.tensor(t, dtype=torch.long, device=self.device)
            else:  # Linear
                progress = step / num_steps
                t_lin = int(max_step - (max_step - min_step) * progress)
                # Add small random noise (e.g., ±2% of total step range)
                current_delta_t = guidance_opt.delta_t
                random_noise = torch.randint(
                    low=-current_delta_t,
                    high=current_delta_t,
                    size=(1,),
                    dtype=torch.long,
                    device=device,
                    generator=self.generator,
                )[0]

                # Final timestep with noise
                ind_t = torch.clamp(t_lin + random_noise, min_step, max_step)
                ind_prev_t = max(ind_t - current_delta_t, torch.ones_like(ind_t) * 0)

            self.ind_t = ind_t.detach().cpu()

            t = self.timesteps[ind_t]
            # print("t", self.timesteps[ind_t])
            self.t_saved = t.detach().cpu()

        with torch.no_grad():
            if ip_adapter_image is not None and self.added_cond_kwargs is None:
                consider_cfg = False
                image_embeds = self.pipe.prepare_ip_adapter_image_embeds(
                    ip_adapter_image, None, device, 1, consider_cfg  # Do cfg
                )
                if consider_cfg:
                    image_embeds[0] = torch.cat(
                        [image_embeds[0], torch.zeros_like(image_embeds[0][:1])]
                    )
                else:
                    image_embeds[0] = torch.cat(
                        [image_embeds[0]] + [torch.zeros_like(image_embeds[0][:1])] * 2
                    )
                self.added_cond_kwargs = {"image_embeds": image_embeds}

            tt = t.reshape(1, 1).repeat(latents.shape[0], 1).reshape(-1)

            # add noise
            noise = torch.normal(
                torch.zeros_like(latents),
                torch.ones_like(latents),
                generator=self.generator,
            )

            if self.grad_method != "ism":
                target = noise
                latents_noisy = self.scheduler.add_noise(latents, noise, tt)
            else:
                # Step 1: sample x_s with larger steps
                inverse_text_embeddings = self.text_embeddings[1:]
                xs_delta_t = (
                    guidance_opt.xs_delta_t
                    if guidance_opt.xs_delta_t is not None
                    else current_delta_t
                )
                xs_inv_steps = (
                    guidance_opt.xs_inv_steps
                    if guidance_opt.xs_inv_steps is not None
                    else int(np.ceil(ind_prev_t / xs_delta_t))
                )
                starting_ind = max(
                    ind_prev_t - xs_delta_t * xs_inv_steps, torch.ones_like(ind_t) * 0
                )

                _, prev_latents_noisy, pred_scores_xs = self.add_noise_with_cfg(
                    latents,
                    noise,
                    ind_prev_t,
                    starting_ind,
                    inverse_text_embeddings,
                    guidance_opt.denoise_guidance_scale,
                    xs_delta_t,
                    xs_inv_steps,
                    eta=guidance_opt.xs_eta,
                )
                # Step 2: sample x_t
                _, latents_noisy, pred_scores_xt = self.add_noise_with_cfg(
                    prev_latents_noisy,
                    noise,
                    ind_t,
                    ind_prev_t,
                    inverse_text_embeddings,
                    guidance_opt.denoise_guidance_scale,
                    current_delta_t,
                    1,
                    is_noisy_latent=True,
                )

                pred_scores = pred_scores_xt + pred_scores_xs
                target = pred_scores[0][1]

            # pred noise
            # noise_pred_text, noise_pred_uncond (negative), noise_pred_null (inverse)
            if self.has_controlnet:
                # When using "guess mode" the controlnet implementation runs only on the conditional branch
                # And not on uncond and null.
                # See https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/controlnet/pipeline_controlnet.py
                # Guess mode discussed in "TexFusion"
                # https://openaccess.thecvf.com/content/ICCV2023/papers/Cao_TexFusion_Synthesizing_3D_Textures_with_Text-Guided_Image_Diffusion_Models_ICCV_2023_paper.pdf
                # In the actual controlnet paper "guess mode" is referred to as "CFG Resolution Weighting"
                # it turns out that for image generation, not doing so results in an excessive guidance
                # Maybe this explains why we need lower controlnet strength to get decent results (without guess mode in controlnet impl)
                # Indeed setting guess_mode=True in controlnet call and increasing conditioning_scale to 1 seems to work well.
                consider_uncond = False
                if self.guess_mode:
                    if consider_uncond:
                        # Deprecate ?
                        ctrl_text_embeddings = self.text_embeddings[
                            :2
                        ]  # .chunk(3)[0:2]
                        latent_model_input = torch.cat([latents_noisy] * 2).to(
                            self.weights_dtype
                        )
                        t_input = torch.cat([tt] * 2, dim=0)
                    else:
                        ctrl_text_embeddings = self.text_embeddings[
                            :1
                        ]  # .chunk(3)[0:2]
                        latent_model_input = torch.cat([latents_noisy] * 1).to(
                            self.weights_dtype
                        )
                        t_input = torch.cat([tt] * 1, dim=0)
                else:
                    ctrl_text_embeddings = self.text_embeddings[:2]  # .chunk(3)[0:2]
                    latent_model_input = torch.cat([latents_noisy] * 2).to(
                        self.weights_dtype
                    )
                    t_input = torch.cat([tt] * 2, dim=0)

                with perf_timer("Controlnet"):
                    down_block_residuals, mid_block_residual = self.controlnet(
                        latent_model_input,
                        t_input,
                        ctrl_text_embeddings,
                        controlnet_cond=image_cond,
                        conditioning_scale=self.conditioning_scale,
                        guess_mode=self.guess_mode and self.guess_mode_reweight,
                        return_dict=False,
                    )

                if self.guess_mode:
                    with perf_timer("Controlnet residuals"):
                        if consider_uncond:
                            # Pad zeros for  null
                            down_block_residuals = [
                                torch.cat([d] + [torch.zeros_like(d)[:1]] * 1)
                                for d in down_block_residuals
                            ]
                            mid_block_residual = torch.cat(
                                [mid_block_residual]
                                + [torch.zeros_like(mid_block_residual)[:1]] * 1
                            )
                        else:
                            # Pad zeros for uncond and null
                            down_block_residuals = [
                                torch.cat([d] + [torch.zeros_like(d)] * 2)
                                for d in down_block_residuals
                            ]
                            mid_block_residual = torch.cat(
                                [mid_block_residual]
                                + [torch.zeros_like(mid_block_residual)] * 2
                            )
                else:
                    down_block_residuals = [
                        torch.cat([d] + [torch.zeros_like(d)[:1]] * 1)
                        for d in down_block_residuals
                    ]
                    mid_block_residual = torch.cat(
                        [mid_block_residual]
                        + [torch.zeros_like(mid_block_residual)[:1]] * 1
                    )
            else:
                down_block_residuals, mid_block_residual = None, None

            if self.grad_method == "hifa":
                self.scheduler.set_timesteps(self.num_hifa_denoise_steps)
                step = (
                    self.scheduler.config.num_train_timesteps
                    // self.num_hifa_denoise_steps
                )
                schedule = range(t, -1, -step)
                print("hifa sched", list(schedule))
                latents_denoised = latents_noisy
                for td in schedule:
                    latent_model_input = torch.cat([latents_denoised] * 3).to(
                        self.weights_dtype
                    )
                    t_input = torch.ones_like(tt) * td
                    noise_pred = self.unet(
                        latent_model_input,
                        t_input,
                        encoder_hidden_states=self.text_embeddings,
                        down_block_additional_residuals=down_block_residuals,
                        mid_block_additional_residual=mid_block_residual,
                        added_cond_kwargs=self.added_cond_kwargs,
                    ).sample

                    # retrieve noise for different conditions
                    (
                        noise_pred_text,
                        noise_pred_uncond,
                        noise_pred_null,
                    ) = noise_pred.chunk(3)
                    delta_noise_preds = noise_pred_text - noise_pred_uncond
                    noise_pred = (
                        noise_pred_uncond + self.guidance_scale * delta_noise_preds
                    )
                    latents_denoised = self.scheduler.step(
                        noise_pred, td, latents_denoised, eta=1.0
                    )["prev_sample"]
                self.scheduler.set_timesteps(self.num_timesteps)
            else:
                latent_model_input = torch.cat([latents_noisy] * 3).to(
                    self.weights_dtype
                )
                t_input = torch.cat([tt] * 3, dim=0)

                noise_pred = self.unet(
                    latent_model_input,
                    t_input,
                    encoder_hidden_states=self.text_embeddings,
                    down_block_additional_residuals=down_block_residuals,
                    mid_block_additional_residual=mid_block_residual,
                    added_cond_kwargs=self.added_cond_kwargs,
                ).sample
                # retrieve noise for different conditions
                noise_pred_text, noise_pred_uncond, noise_pred_null = noise_pred.chunk(
                    3
                )

        if self.grad_method == "hifa":
            return latents_denoised, t
        else:
            if cfg.weighting_strategy == "sds":
                w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
            elif cfg.weighting_strategy == "uniform":
                w = 1
            elif cfg.weighting_strategy == "fantasia3d":
                w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1)
            else:
                raise ValueError(
                    f"Unknown weighting strategy: {cfg.weighting_strategy}"
                )

            if self.guidance_scale > 0.0 and self.grad_method != "csd":
                if not cfg.noise_free and self.grad_method != "ism":
                    # This is the usual SDS
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )
                    grad = w * (noise_pred - target)
                else:
                    # Noise free score distillation according to Katzir et al. (2023)
                    # allows to use a standard guidance scale
                    delta_c = self.guidance_scale * (noise_pred_text - noise_pred_null)
                    mask = (t < 200).int().view(self.batch_size, 1, 1, 1)
                    delta_d = mask * noise_pred_null + (1 - mask) * (
                        noise_pred_null - noise_pred_uncond
                    )
                    grad = w * (delta_c + delta_d)
            else:
                grad = w * (noise_pred_text - noise_pred_uncond)  # CSD
        return grad, t

    def embed_text(self, prompt, negative_prompt=None, null_prompt=None):
        # TODO look into "clip skip" here:
        # diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_prompt
        if negative_prompt is None:
            negative_prompt = [""]
        if null_prompt is None:
            null_prompt = [""]
        if type(prompt) == str:
            prompt = [prompt]
        if type(negative_prompt) == str:
            negative_prompt = [negative_prompt]
        if type(null_prompt) == str:
            null_prompt = [null_prompt]

        embeddings = []
        for text in [prompt, negative_prompt, null_prompt]:
            # Tokenize text and get embeddings
            text_input = self.pipe.tokenizer(
                text,
                padding="max_length",
                max_length=self.pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )

            with torch.no_grad():
                text_embeddings = self.pipe.text_encoder(
                    text_input.input_ids.to(self.device)
                )[0]
            embeddings.append(text_embeddings)

        res = torch.cat(embeddings)
        return res

    def drop_nans(self, grads):
        assert torch.isfinite(grads).all()
        return torch.nan_to_num(grads.detach().float(), 0.0, 0.0, 0.0)

    def grads(
        self,
        latent_z,
        image_cond=None,
        step=None,
        num_steps=None,
        ip_adapter_image=None,
        **sds_kwargs,
    ):
        grad, t = self.compute_grad_sds(
            latent_z,
            image_cond,
            step=step,
            num_steps=num_steps,
            ip_adapter_image=ip_adapter_image,
        )
        grad = torch.nan_to_num(grad)
        return grad, t


class SDSLoss(StableDiffusion):
    def __init__(
        self, caption, augment=0, input_size=(512, 512), size=(512, 512), **kwargs
    ):
        super(SDSLoss, self).__init__(
            caption, size=size, batch_size=augment if augment else 1, **kwargs
        )

        self.uncrop_size = input_size
        self.augment = augment

        self.grad_norm = torch.tensor(0.0)
        # NB augmentation seems to be very harmful for controlnet (at least with the params below!)
        self.augment_trans = transforms.Compose(
            [
                # transforms.ColorJitter(0.1),
                transforms.RandomPerspective(distortion_scale=0.1, fill=1.0),
                transforms.RandomResizedCrop(size, scale=(0.9, 1.0), ratio=(1.0, 1.0)),
                transforms.GaussianBlur(
                    kernel_size=21, sigma=(0.01, 2.0)
                ),  # Example: 5x5 kernel
            ]
        )

    def forward(
        self,
        img,
        image_cond=None,
        step=None,
        num_steps=None,
        ip_adapter_image=None,
        grad_scale=0.01,
    ):
        img = to_batch(img, self.rgb)
        interp_mode = "bilinear"
        # interp_mode = 'nearest'
        if interp_mode == "bilinear":
            align_corners = False
        else:
            align_corners = None  # False
        if image_cond is not None:
            image_cond = to_batch(image_cond, False)
            if (
                self.size[0] != image_cond.shape[-2]
                or self.size[1] != image_cond.shape[-1]
            ):
                image_cond = torch.nn.functional.interpolate(
                    image_cond,
                    mode=interp_mode,
                    size=self.size,
                    align_corners=align_corners,
                )
            image_cond = image_cond.to(self.weights_dtype)
        if self.uncrop_size[0] != img.shape[-2] or self.uncrop_size[1] != img.shape[-1]:
            img = torch.nn.functional.interpolate(
                img,
                mode=interp_mode,
                size=self.uncrop_size,
                align_corners=align_corners,
            )

        img = img.to(self.weights_dtype)

        if self.augment > 0:
            x_aug = []
            for _ in range(self.augment):
                x_aug.append(self.augment_trans(img))
            x_aug = torch.cat(x_aug, dim=0)
            image_cond = torch.cat([image_cond] * self.augment, dim=0)
        else:
            x_aug = img

        with perf_timer("Encoding"):
            latent_z = self.encode_images(x_aug)

        step_ratio = None
        if step is not None and num_steps is not None:
            step_ratio = min(1.0, step / num_steps)

        grad_z, t = self.grads(
            latent_z,
            image_cond,
            step=step,
            num_steps=num_steps,
            ip_adapter_image=ip_adapter_image,
        )
        if self.grad_method == "hifa":
            w = 0.05
            size_scale = 1.0 / 512**2
            error = latent_z - grad_z.detach()
            if True:  # self.opt.clip_grad:
                thres = (
                    10 / 0.05 * (1 - self.alphas[t]) ** 0.5
                )  # threshold scales with noise strength. Want to clip grad at 10 at max timestep
                error_clamped = error.clamp(-thres, thres)
                if not torch.allclose(error, error_clamped):
                    warnings.warn(
                        "grad clip just happened. Might be a gradient explosion"
                    )
                error = error_clamped
            sds_loss = torch.mean(error**2)  # /2 * 2 * size_scale
            target_recon = self.decode_latent(grad_z)
            # target_recon = torch.mean(recon_image, axis=1)
            self.target_recon = (
                target_recon[0].permute(1, 2, 0).detach().cpu().numpy() * 255
            ).astype(np.uint8)
            error_grayscale = target_recon - img  # torch.mean(img, axis=1)
            img_loss = torch.mean(error_grayscale**2)
            print("zloss", sds_loss.item(), "imgloss", img_loss.item())
            return (sds_loss + img_loss * 0.01) * grad_scale
        else:
            target_recon = self.decode_latent(grad_z)
            self.target_recon = (
                target_recon[0].permute(1, 2, 0).detach().cpu().numpy() * 255
            ).astype(np.uint8)
            # print(self.target_recon.shape)

        torch.nn.utils.clip_grad_norm_(grad_z, 100.0)
        grad_z = grad_z * grad_scale
        self.grad_norm = torch.norm(grad_z)
        targets = (latent_z - grad_z).detach()
        loss = (
            0.5
            * F.mse_loss(latent_z.float(), targets, reduction="sum")
            / latent_z.shape[0]
        )
        return loss
