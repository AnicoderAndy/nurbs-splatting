"""
Area filling with NURBS splatting.
This is a port of demo_area_fill_01.py from calligraph to nsplat API.

Directly run this script or tune the config parameters below.
"""

import time

import numpy as np
import torch
import os
from typing import Literal
from PIL import Image

from nsplat import image_losses, plut, stroke_init
from nsplat.nurbs import NURBSSplat
from nsplat.spline_losses import make_bbox_loss, make_deriv_loss
from nsplat.utils import make_cosine_lr_scheduler

##### BEGIN CONFIG #####
size = 256
padding = 15
alpha = 1
image_alpha = 0.5  # density of coverage
point_density = 0.015
closed = False

target_char = "2"
font_path = "data/Calistoga-Regular.ttf"

order = 6
density = 10
seed = 133

stroke_width = 1.0
minw, maxw = 0.0, 3.5

lr_pos = 2.0
lr_width = 0.3
lr_weights = 0.01
lr_interval = 0.0
num_opt_steps = 250
log_every = 1
save_every = 1
output_dir = "output_area_fill"

## For MSE loss
mse_w = 20.0
mse_mul = 1.0

## For derivative smoothing loss
smooth_order = 3
smooth_w = 30.0
deriv_approx_method = "analytical"

## For bounding-box loss
bbox_pad = 10.0
bbox_w = 10.0

## For style loss
style_w = 10.0  # 0 will disable style loss
style_img_path = "data/style_imgs/zcal10.jpg"
distortion_scale = 0.3
patch_size = 128
n_cuts = 64

##### END CONFIG #####

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(seed)
np.random.seed(seed)
os.makedirs(output_dir, exist_ok=True)

optimize_params = tuple()
if lr_pos > 0.0:
    optimize_params += ("key_points",)
if lr_width > 0.0:
    optimize_params += ("stroke_width",)
if lr_weights > 0.0:
    optimize_params += ("weights",)
if lr_interval > 0.0:
    optimize_params += ("knot_interval",)
print(f"Optimizing parameters: {optimize_params}")

input_img = plut.font_to_image(target_char, (size, size), padding, font_path=font_path)
img = np.array(input_img) / 255.0

h, w = img.shape
print(f"Image size: {w}x{h}")

target_img = 1 - (1.0 - img) * (alpha * image_alpha)
Image.fromarray((target_img * 255).astype(np.uint8)).save(
    os.path.join(output_dir, "target.png")
)
background_image = np.ones((h, w))

# Initialize path
num_points = int(np.sum(1 - img) * point_density)
path, _ = stroke_init.init_path_tsp(1 - img, num_points, closed=closed)

widths = np.full((path.shape[0], 1), stroke_width, dtype=np.float32)
key_points = np.concatenate([path.astype(np.float32), widths], axis=1)
key_points_t = torch.tensor(key_points, device=device)
weights_t = torch.ones(key_points.shape[0], device=device)

# Create NURBS splat object
nurbs = NURBSSplat(
    key_points_t,
    weights_t,
    order,
    width=w,
    height=h,
    density=density,
    color=torch.tensor([0.0, 0.0, 0.0], device=device),
    closed=closed,
)

optimizer = torch.optim.Adam(
    nurbs.get_params(
        # optimize=("key_points", "stroke_width"),
        optimize=("key_points", "stroke_width", "weights", "knot_interval"),
        lrs={
            "key_points": lr_pos,
            "stroke_width": lr_width,
            "weights": lr_weights,
            "knot_interval": lr_interval,
        },
    )
)
scheduler = make_cosine_lr_scheduler(optimizer, num_opt_steps, lr_min_scale=0.1)

target_t = torch.tensor(target_img, device=device, dtype=torch.float32)
mse_loss_fn = image_losses.MultiscaleMSELoss(rgb=False)
style_img = Image.open(style_img_path).convert("L").resize((512, 512))
style_loss_fn = image_losses.CLIPPatchLoss(
    rgb=False,
    image_prompts=[style_img],
    model="CLIPAG",
    min_size=patch_size,
    cut_scale=0.0 if distortion_scale > 0.0 else 0.35,
    distortion_scale=distortion_scale,
    blur_sigma=0.0,
    thresh=0.0,
    n_cuts=n_cuts,
    use_negative=False,
)

# Derivative order needed for scene.splat()
_splat_deriv_order = smooth_order if smooth_w > 0 and order > 4 else 0

smooth_loss_fn = make_deriv_loss(
    smooth_order,
    ref_size=size,
    approx_method=deriv_approx_method,
)
bbox_loss_fn = make_bbox_loss(((0.0, 0.0), (float(w), float(h))), pad=bbox_pad)

startup, _ = nurbs.splat()
startup = startup.detach().cpu().numpy()[..., 0]
Image.fromarray((startup * 255).astype(np.uint8)).save(
    os.path.join(output_dir, "startup.png")
)

style_loss = torch.tensor(0.0, device=device)
start_time = time.perf_counter()
for step in range(num_opt_steps):
    optimizer.zero_grad()
    out, pts = nurbs.splat(max_deriv_order=_splat_deriv_order)
    im = out[..., 0]

    mse_loss = mse_loss_fn(im, target_t, mult=mse_mul)

    smooth_loss = smooth_loss_fn([nurbs])
    loss = mse_w * mse_loss + smooth_w * smooth_loss

    if style_w > 0.0:
        style_loss = style_loss_fn(im)
        loss = loss + style_w * style_loss

    bbox_loss = bbox_loss_fn([nurbs])
    loss = loss + bbox_w * bbox_loss

    loss.backward()
    optimizer.step()
    scheduler.step()

    with torch.no_grad():
        nurbs._stroke_width.clamp_(minw, maxw)
        nurbs._key_points[:, 0].clamp_(0, nurbs.W)
        nurbs._key_points[:, 1].clamp_(0, nurbs.H)
        nurbs._weights.clamp_(0.01, 10.0)
        nurbs._knot_interval.clamp_(0.0, 2.0)

    if step % log_every == 0 or step == num_opt_steps - 1:
        log_msg = (
            f"step {step:04d} | loss {loss.item():.6f} | mse {mse_loss.item():.6f}"
            f" | style {style_loss.item():.6f} | smooth {smooth_loss.item():.6f}"
        )
        if bbox_w > 0:
            log_msg += f" | bbox {bbox_loss.item():.6f}"
        print(log_msg)  # type: ignore

    if step % save_every == 0 or step == num_opt_steps - 1:
        render_np = im.detach().cpu().numpy()
        Image.fromarray((render_np * 255).astype(np.uint8)).save(
            os.path.join(output_dir, f"step_{step:04d}.png")
        )

time_elapsed = time.perf_counter() - start_time
print(f"Optimization finished in {time_elapsed:.2f} seconds.")

final_out, _ = nurbs.splat()
final_render = final_out.detach().cpu().numpy()[..., 0]
Image.fromarray((final_render * 255).astype(np.uint8)).save(
    os.path.join(output_dir, "final.png")
)
print("Weights max/min/mean:")
weights_max = torch.max(nurbs._weights).item()
weights_min = torch.min(nurbs._weights).item()
weights_mean = torch.mean(nurbs._weights).item()
print(weights_max, weights_min, weights_mean)

print("Knot interval max/min/mean:")
knot_interval_max = torch.max(nurbs._knot_interval).item()
knot_interval_min = torch.min(nurbs._knot_interval).item()
knot_interval_mean = torch.mean(nurbs._knot_interval).item()
print(knot_interval_max, knot_interval_min, knot_interval_mean)
