"""
Loss functions for image comparison
Mostly adapted from https://github.com/colormotor/calligraph
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from nsplat import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## CONFIG ##
batch_size = 1
clip_models = {}
## END CONFIG ##


def to_batch(x, rgb: bool):
    if isinstance(x, Image.Image):
        if not rgb:
            x = x.convert("L")
        x = torch.tensor(np.array(x) / 255.0, device=device)
    elif isinstance(x, np.ndarray):
        x = torch.tensor(x, device=device)
    elif isinstance(x, torch.Tensor):
        x = x.to(device)
    else:
        raise TypeError(f"Unsupported input type: {type(x)}")

    if rgb:
        if len(x.shape) == 3:
            x = x[:, :, :, np.newaxis]
        x = x.permute((3, 2, 0, 1))  # to NCHW
    else:
        if len(x.shape) > 2:
            x = torch.mean(x, -1)
        if len(x.shape) == 2:
            x = x[np.newaxis, np.newaxis, :, :]
        x = x.repeat(1, 3, 1, 1)
    x = x.repeat(batch_size, 1, 1, 1)
    return x


class MultiscaleMSELoss(torch.nn.Module):
    """Multiscale MSE loss for images, adapted from PyDiffvg examples"""

    def __init__(self, sigma: float = 1, rgb: bool = True, debug: bool = False):
        super().__init__()
        self.rgb = rgb
        self.blur = transforms.GaussianBlur(
            kernel_size=int(np.ceil(4 * sigma)) + 1, sigma=(sigma, sigma)
        )
        self.debug = debug

    def forward(
        self, im, target, mult: float = 1, scale_factor: float = 0.5, num_levels=None
    ):
        im = to_batch(im, self.rgb)
        target = to_batch(target, self.rgb).to(im.dtype)
        _, _, h, _ = im.shape

        if num_levels is None:
            num_levels = max(int(np.ceil(np.log2(h))) - 2, 1)

        sz = im.shape[-1]
        losses = []
        w = 1.0
        ims = []
        targets = []
        wsum = 0.0
        for _ in range(num_levels):
            loss = F.mse_loss(im, target)
            losses.append(loss * w)
            wsum += w
            if self.debug:
                ims.append(
                    F.interpolate(
                        im,
                        scale_factor=sz / im.shape[-1],
                        align_corners=True,
                        mode="bicubic",
                    )
                    * w
                )
                targets.append(
                    F.interpolate(
                        target,
                        scale_factor=sz / target.shape[-1],
                        align_corners=True,
                        mode="bicubic",
                    )
                    * w
                )
            w = w * mult

            im = F.interpolate(self.blur(im), scale_factor=scale_factor, mode="nearest")
            target = F.interpolate(
                self.blur(target), scale_factor=scale_factor, mode="nearest"
            )

        if self.debug and ims and targets:
            self.blur_im = (torch.stack(ims).sum(dim=0) / wsum)[0, 0, :, :]
            self.blur_target = (torch.stack(targets).sum(dim=0) / wsum)[0, 0, :, :]

        losses = torch.stack(losses)
        return losses.sum()


class CLIPPatchLoss(torch.nn.Module):
    def __init__(
        self,
        text_prompts: Iterable[str] | None = None,
        image_prompts: Iterable[Image.Image] | None = None,
        negative_prompts: Iterable[str] | None = None,
        use_negative: bool = True,
        model: str = "ViT-B-32",
        crop_scale=(0.6, 0.9),
        distortion_scale: float = 0.0,
        thresh: float = 0.5,
        blur_sigma: float = 0.0,
        min_size=None,
        cut_scale: float = 0.25,
        num_batches: int = 1,
        n_cuts: int = 16,
        rgb: bool = True,
        clipag: bool = False,
    ):
        super().__init__()

        if text_prompts is None:
            text_prompts = []
        if image_prompts is None:
            image_prompts = []
        if negative_prompts is None:
            negative_prompts = ["A badly drawn sketch.", "Many ugly, messy drawings."]

        self.rgb = rgb
        if clipag:
            model = "CLIPAG"

        self.clip_model, _, tokenizer, self.clip_model_input_size = load_clip_model(
            model
        )
        self.clip_model: Any = self.clip_model
        tokenizer = tokenizer
        if not use_negative:
            negative_prompts = []
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(
                    size=self.clip_model_input_size, max_size=None, antialias=False
                ),
                transforms.CenterCrop(
                    size=(self.clip_model_input_size, self.clip_model_input_size)
                ),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        self.clip_model.to(device)
        self.clip_model.eval()

        target_embeds: list[torch.Tensor] = []
        negative_embeds: list[torch.Tensor] = []
        image_prompt_tensors = [
            to_batch(img, rgb).to(torch.float32) for img in image_prompts
        ]
        with torch.no_grad():
            for text_prompt in text_prompts:
                tokenized_text = tokenizer([text_prompt]).to(device)
                target_embeds.append(self.clip_model.encode_text(tokenized_text))
            for image_prompt in image_prompt_tensors:
                image_embed = self.clip_model.encode_image(
                    self.preprocess(image_prompt)
                )
                target_embeds.append(image_embed)
            for text_prompt in negative_prompts:
                tokenized_text = tokenizer([text_prompt]).to(device)
                negative_embeds.append(self.clip_model.encode_text(tokenized_text))

        self.num_positive = len(target_embeds)
        self.num_negative = len(negative_embeds)

        self.target_embeds = (
            torch.cat(target_embeds) if target_embeds else torch.empty(0, device=device)
        )
        if negative_embeds:
            self.negative_embeds = negative_embeds
        else:
            self.negative_embeds = None
        self.n_cuts = n_cuts
        self.num_batches = num_batches
        self.cut_scale = cut_scale
        self.thresh = thresh

        augment_list = []
        if distortion_scale > 0:
            augment_list.append(
                transforms.RandomPerspective(
                    fill=1, p=1.0, distortion_scale=distortion_scale
                )
            )
        if blur_sigma > 0.0:
            augment_list.append(
                transforms.GaussianBlur(
                    kernel_size=5, sigma=(blur_sigma * 0.01, blur_sigma)
                )
            )
        self.augment_compose = transforms.Compose(augment_list)
        if min_size is None:
            self.min_size = self.clip_model_input_size
        else:
            self.min_size = min_size

    def forward(self, input, *args):
        input = to_batch(input, self.rgb)

        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.min_size)

        cuts_per_batch = self.n_cuts // self.num_batches
        remaining_cuts = self.n_cuts % self.num_batches
        loss = 0
        cutout_count = 0

        for batch_idx in range(self.num_batches):
            n_cuts_this_batch = cuts_per_batch + (
                1 if batch_idx < remaining_cuts else 0
            )
            cutouts = []

            for _ in range(n_cuts_this_batch):
                size = int(
                    torch.rand([]) * self.cut_scale * (max_size - min_size) + min_size
                )
                offsetx = torch.randint(0, sideX - size + 1, ())
                offsety = torch.randint(0, sideY - size + 1, ())
                cutout = input[:, :, offsety : offsety + size, offsetx : offsetx + size]
                cutout = torch.nn.functional.adaptive_avg_pool2d(
                    cutout, self.clip_model_input_size
                )
                cutout = self.augment_compose(cutout)
                cutouts.append(cutout)

            cutouts_batch = torch.cat(cutouts, dim=0)

            if batch_idx == 0:
                self.test_cutout = cutouts_batch[0].detach().cpu().numpy()[0, :, :]

            input_embeds = self.clip_model.encode_image(self.preprocess(cutouts_batch))

            for n in range(n_cuts_this_batch):
                patch_loss = torch.cosine_similarity(
                    self.target_embeds, input_embeds[n : n + 1], dim=1
                )
                if self.thresh > 0 and patch_loss < self.thresh:
                    patch_loss = 0
                loss -= patch_loss

                if self.negative_embeds is not None:
                    div = 1 / self.num_negative
                    for feat in self.negative_embeds:
                        loss += (
                            torch.cosine_similarity(
                                feat, input_embeds[n : n + 1], dim=1
                            )
                            * div
                        )

                cutout_count += 1

            del cutouts_batch, input_embeds
            torch.cuda.empty_cache()

        return loss / cutout_count


def _get_preprocess_input_size(preprocess: Any) -> int:
    transforms_list = getattr(preprocess, "transforms", None)
    if transforms_list and hasattr(transforms_list[0], "size"):
        size = transforms_list[0].size
        return int(size[0] if isinstance(size, (tuple, list)) else size)
    size = getattr(preprocess, "size", None)
    if isinstance(size, (tuple, list)):
        return int(size[0])
    if isinstance(size, int):
        return size
    return 224


def load_clip_model(model_name: str):
    import open_clip

    if model_name in clip_models:
        print(model_name, "already loaded")
        return clip_models[model_name]

    if model_name == "CLIPAG":
        print("Downlading CLIPAG")
        url = "https://zenodo.org/records/10446026/files/CLIPAG_ViTB32.pt?download=1"
        path = "./CLIPAG_ViTB32.pt"
        utils.download_file_once(url, path)
        pretrained = path
        model_name = "ViT-B-32"
    else:
        pretrained_map = {
            "ViT-H/14-quickgelu": "dfn5b",
            "ViT-B-32": "laion2b_s34b_b79k",
            "ViT-B-16-SigLIP-384": "webli",
            "ViT-L-16-SigLIP-256": "webli",
            "ViT-L-16-SigLIP-384": "webli",
            "ViT-SO400M-14-SigLIP-384": "webli",
            "ViT-SO400M-14-SigLIP": "webli",
            "ViT-SO400M/14": "webli",
            "ViT-L-14": "laion2b_s32b_b82k",
            "ViT-L-14-quickgelu": "metaclip_fullcc",
            "ViT-g-14": "laion2b_s34b_b88k",
            "ViT-B-16": "datacomp_xl_s13b_b90k",
            "EVA02-L-14": "merged2b_s4b_b131k",
            "ViT-H-14-CLIPA": "datacomp1b",
            "ViT-H-14-378-quickgelu": "dfn5b",
            "ViT-L-14-CLIPA-336": "datacomp1b",
            "ViT-H-14-quickgelu": "metaclip_fullcc",
            "ViT-L-14-CLIPA": "datacomp1b",
            "ViT-B-32-256": "datacomp_s34b_b86k",
        }
        pretrained = pretrained_map[model_name]

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        precision="amp",
        weights_only=False,
        device=device,
    )

    input_size = _get_preprocess_input_size(preprocess)
    print("Input size is ", input_size)
    tokenizer = open_clip.get_tokenizer(model_name)
    clip_models[model_name] = (model, preprocess, tokenizer, input_size)
    return model, preprocess, tokenizer, input_size
