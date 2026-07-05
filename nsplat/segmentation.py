"""
Adapted from https://github.com/colormotor/calligraph
"""

import torch
from transformers import (
    AutoProcessor,
    AutoImageProcessor,
    AutoModelForUniversalSegmentation,
)
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

device = "cuda" if torch.cuda.is_available() else "cpu"


def xdog(im, k=10, gamma=0.98, phi=200, eps=-0.1, sigma=0.8, binarize=True):
    # From https://github.com/yael-vinker/CLIPasso/blob/main/models/painter_params.py
    from scipy.ndimage.filters import gaussian_filter
    from skimage.color import rgb2gray
    from skimage.filters import threshold_otsu

    if type(im) != np.ndarray:
        im = np.array(im)
    if len(im.shape) > 2 and im.shape[2] == 3:
        im = rgb2gray(im)
    imf1 = gaussian_filter(im, sigma)
    imf2 = gaussian_filter(im, sigma * k)
    imdiff = imf1 - gamma * imf2
    imdiff = (imdiff < eps) * 1.0 + (imdiff >= eps) * (1.0 + np.tanh(phi * imdiff))
    imdiff -= imdiff.min()
    imdiff /= imdiff.max()
    if binarize:
        th = threshold_otsu(imdiff)
        imdiff = imdiff >= th
    imdiff = imdiff.astype("float32")
    return imdiff


def ood_saliency(image):
    processor = AutoProcessor.from_pretrained("shi-labs/oneformer_coco_swin_large")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForUniversalSegmentation.from_pretrained(
        "shi-labs/oneformer_coco_swin_large"
    )

    model.to(device)

    features = {}

    panoptic_inputs = processor(
        images=[image], task_inputs=["panoptic"], return_tensors="pt"
    )

    panoptic_inputs.to(device)

    with torch.no_grad():
        outputs = model(**panoptic_inputs)

    del model
    del processor
    del panoptic_inputs

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Finished model inference. Starting postprocessing")

    class_queries_logits = (
        outputs.class_queries_logits
    )  # [batch_size, num_queries, num_classes+1]
    masks_queries_logits = (
        outputs.masks_queries_logits
    )  # [batch_size, num_queries, height, width]

    mask_probs = (
        masks_queries_logits.sigmoid()
    )  # [batch_size, num_queries, height, width]

    test = torch.nn.functional.interpolate(
        mask_probs[0].unsqueeze(0),
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )[0]

    mean_logits = test.mean(axis=0).cpu()

    low, high = np.percentile(mean_logits, [1, 99])

    clipped_mean_logits = np.clip(mean_logits, low, high)

    normalized_clipped_mean_logits = (
        clipped_mean_logits - clipped_mean_logits.min()
    ) / (clipped_mean_logits.max() - clipped_mean_logits.min())

    normalized_clipped_mean_logits = (1 - normalized_clipped_mean_logits) * 255

    normalized_clipped_mean_logits = normalized_clipped_mean_logits.numpy().astype(
        np.uint8
    )

    return normalized_clipped_mean_logits, None


def clip_saliency(image, prompt="", model="ViT-B/32", layers=[]):
    # https://arxiv.org/abs/2304.05653
    from skimage.transform import resize
    from .contrib.CLIPExplain import clip

    model, preprocess = clip.load(model, device=device, jit=False)
    text = clip.tokenize([prompt]).to(device)
    img = preprocess(image).unsqueeze(0).to(device)
    fn = interpret
    if prompt:
        fn = interpret2
    res = resize(fn(img, text, model, layers), (image.height, image.width))
    del model
    del preprocess
    return res


def interpret(image, texts, model, layers=[], start_layer=-1):
    images = image.repeat(1, 1, 1, 1)
    res = model.encode_image(images)  # , mode="saliency")
    model.zero_grad()
    image_attn_blocks = list(
        dict(model.visual.transformer.resblocks.named_children()).values()
    )
    # print(image_attn_blocks)
    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    R = torch.eye(
        num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype
    ).to(device)
    R = R.unsqueeze(0).expand(1, num_tokens, num_tokens)
    cams = []  # there are 12 attention blocks
    if start_layer == -1:
        # calculate index of last layer
        start_layer = len(image_attn_blocks) - 1
    print("n blocks:", len(image_attn_blocks))
    print("start layer", start_layer)
    if not layers:
        layers = range(len(image_attn_blocks))
    for i in layers:  # , blk in enumerate(image_attn_blocks):
        # if i < start_layer:
        #    continue

        blk = image_attn_blocks[i]
        cam = blk.attn_probs.detach()  # attn_probs shape is 12, 50, 50
        # print('shp', cam.shape)
        # each patch is 7x7 so we have 49 pixels + 1 for positional encoding
        cam = cam.reshape(1, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0)
        cam = cam.clamp(min=0).mean(dim=1)  # mean of the 12 something
        cams.append(cam)
        R = R + torch.bmm(cam, R)
    cams_avg = torch.cat(cams)  # 12, 50, 50
    cams_avg = cams_avg[:, 0, 1:]  # 12, 1, 49
    image_relevance = cams_avg.mean(dim=0).unsqueeze(0)
    image_relevance = image_relevance.reshape(1, 1, 7, 7)
    image_relevance = torch.nn.functional.interpolate(
        image_relevance, size=224, mode="bicubic"
    )
    image_relevance = (
        image_relevance.reshape(224, 224).data.cpu().numpy().astype(np.float32)
    )
    image_relevance = (image_relevance - image_relevance.min()) / (
        image_relevance.max() - image_relevance.min()
    )
    return image_relevance


def interpret2(image, texts, model, layers=[]):  # start_layer=-1):
    batch_size = texts.shape[0]
    images = image.repeat(batch_size, 1, 1, 1)
    # image_features = model.encode_image(images, mode='saliency')
    logits_per_image, logits_per_text = model(images, texts)  # , mode='saliency')
    # logits_per_image = image_features/image_features.norm(dim=-1, keepdim=True)
    probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
    index = [i for i in range(batch_size)]
    one_hot = np.zeros(
        (logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32
    )
    one_hot[torch.arange(logits_per_image.shape[0]), index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * logits_per_image)
    model.zero_grad()

    image_attn_blocks = list(
        dict(model.visual.transformer.resblocks.named_children()).values()
    )

    # if start_layer == -1:
    #   # calculate index of last layer
    #   start_layer = len(image_attn_blocks) - 1

    if not layers:
        layers = range(len(image_attn_blocks))

    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    R = torch.eye(
        num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype
    ).to(device)
    R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    cams = []
    for i in layers:  # , blk in enumerate(image_attn_blocks):
        # if i < start_layer:
        #  continue
        blk = image_attn_blocks[i]
        grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[
            0
        ].detach()
        cam = blk.attn_probs.detach()
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0).mean(dim=1)
        cams.append(cam)
        R = R + torch.bmm(cam, R)
    # print('R', R.shape)
    image_relevance = R[:, 0, 1:]

    # cams_avg = torch.cat(cams) # 12, 50, 50
    # print('cams_avg', cams_avg.shape)
    # cams_avg = cams_avg[:, 0, 1:] # 12, 1, 49
    # image_relevance = cams_avg.mean(dim=0).unsqueeze(0)

    dim = int(image_relevance.numel() ** 0.5)
    # print('dim is', dim)
    image_relevance = image_relevance.reshape(1, 1, dim, dim)
    image_relevance = torch.nn.functional.interpolate(
        image_relevance, size=224, mode="bilinear"
    )
    image_relevance = image_relevance.reshape(224, 224).cuda().data.cpu().numpy()
    image_relevance = (image_relevance - image_relevance.min()) / (
        image_relevance.max() - image_relevance.min()
    )
    return image_relevance
