"""
Neural Image Abstraction Using NURBS Splats with Score Distillation Sampling
This is a port of demo_sds_strokes_01.py from calligraph to nsplat API.

Directly run this script or tune the config parameters below.
"""

import os
import time
import numpy as np
import torch
import shutil
from pathlib import Path
from PIL import Image
from skimage import feature

from nsplat.nurbs import NURBSSplat
from nsplat.scene import NURBSScene
from nsplat import image_losses, spline_losses, stroke_init
from nsplat.contrib import sd
from nsplat.utils import make_cosine_lr_scheduler

##### BEGIN CONFIG #####
# I/O settings
input_file = "data/beethoven.jpg"
style_path = "data/style_imgs/zcal2.jpg"
output_path = "./output/"
save_every = 1

# Spline settings
order = 6
minw, maxw = 0.5, 7.0
closed = False
density_for_nsplat = 5
# Width annealing
startup_width = 1.0
width_anneal_start = 0.0
width_anneal_end = 0.75

# Initialization settings
point_density = 0.002
seed = 333
multiplicity = 3
noise = 0.0

# Optimization settings
optimized_params = ("key_points", "stroke_width", "weights")
lr_pos = 3.0
lr_width = 0.4
lr_weights = 0.01
lr_interval = 0.0
num_opt_steps = 300

# Smoothing loss
smooth_order = 3
smooth_w = 200.0

# MSE loss
mse_w = 0.0
mse_mul = 1.0

# Bbox loss
bbox_w = 10.0
bbox_pad = 1.0

# CLIP/Style settings
style_w = 300.0
clip_model = "CLIPAG"
distortion_scale = 0.3
patch_size = 128

# SDS settings
use_sds = True
sds_w = 1.0
cond_scale = 0.7
guess_mode = True
ip_adapter = True
ip_scale = 0.9
cfg_scale = 7.5
t_min, t_max = 0.5, 0.98
grad_method = "ism"
prompt = "A black and white ink drawing"
canny_sigma = 1.0
##### END CONFIG #####

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

torch.manual_seed(seed)
np.random.seed(seed)

# Ensure output directory exists
os.makedirs(output_path, exist_ok=True)
shutil.copy(Path(__file__), os.path.join(output_path, "src_code.py"))
print(f"Output path: {output_path}")
print(f"Saved source code to {os.path.join(output_path, 'src_code.py')}")

# Load input image
size = 512
input_img = Image.open(input_file).convert("L").resize((size, size))
input_img.save(os.path.join(output_path, "input.png"))
print(f"Saved input image to {os.path.join(output_path, 'input.png')}")

# Load style image
style_img = Image.open(style_path).convert("L").resize((size, size))
style_img.save(os.path.join(output_path, "style.png"))
print(f"Saved style image to {os.path.join(output_path, 'style.png')}")

img = np.array(input_img) / 255.0
h, w = img.shape

# Create canny edge image for conditioning
cond_img = feature.canny(img, canny_sigma)
cond_img_pil = Image.fromarray((cond_img * 255).astype(np.uint8)).convert("RGB")
cond_img_pil.save(os.path.join(output_path, "cond.png"))
print(f"Saved conditioning image to {os.path.join(output_path, 'cond.png')}")

# Target and background
target_img = img
background_image = np.ones_like(target_img)

##############################################
# Initialization paths using TSP


def add_multiplicity(Q, noise=0.0):
    Q = np.kron(Q, np.ones((multiplicity, 1)))
    return Q + np.random.uniform(-noise, noise, Q.shape)


num_points = int(np.sum(1 - img) * point_density)
print(f"Initializing {num_points} points using TSP...")

P, density_map = stroke_init.init_path_tsp(
    input_img,
    num_points,
    saliency_type="ood",
    closed=closed,
)

P = add_multiplicity(P, noise=noise)

# Add stroke widths
widths = np.full((P.shape[0], 1), startup_width, dtype=np.float32)
Pw = np.concatenate([P.astype(np.float32), widths], axis=1)
startup_paths = [Pw]

# Save density map
density_img = Image.fromarray((density_map * 255).astype(np.uint8))
density_img.save(os.path.join(output_path, "density.png"))

##############################################
# Create NURBSScene

background = torch.ones(3, device=device, dtype=dtype)
scene = NURBSScene(width=w, height=h, background=background)
scene.to(device)

for Pw in startup_paths:
    if Pw.shape[0] < order:
        print(f"Skipping: only {Pw.shape[0]} points, need at least {order}")
        continue

    key_points_t = torch.tensor(Pw, device=device, dtype=dtype)
    weights_t = torch.ones(Pw.shape[0], device=device, dtype=dtype)

    nurbs = NURBSSplat(
        key_points_t,
        weights_t,
        order=order,
        width=w,
        height=h,
        density=density_for_nsplat,
        color=torch.tensor([0.0, 0.0, 0.0], device=device),
        closed=closed,
    )
    scene.add_nurbs(nurbs)

print(f"Created {len(scene)} NURBS spline(s) for optimization.")

# Save startup image
startup_out, _ = scene.splat(max_deriv_order=0)
startup_np = startup_out.detach().cpu().numpy()[..., 0]
startup_img = Image.fromarray((startup_np * 255).astype(np.uint8))
startup_img.save(os.path.join(output_path, "startup.png"))
im = startup_out[..., 0]
print(f"Saved startup image to {os.path.join(output_path, 'startup.png')}")

##############################################
# Optimizer

optimizer = torch.optim.Adam(
    scene.get_params(
        optimize=optimized_params,
        lrs={
            "key_points": lr_pos,
            "stroke_width": lr_width,
            "weights": lr_weights,
            "knot_interval": lr_interval,
        },
    )
)

scheduler = make_cosine_lr_scheduler(optimizer, num_opt_steps, lr_min_scale=0.1)

##############################################
# Loss functions

target_t = torch.tensor(target_img, device=device, dtype=dtype)
mse_loss_fn = image_losses.MultiscaleMSELoss(rgb=False)

# Derivative order needed for scene.splat()
_splat_deriv_order = smooth_order if smooth_w > 0 and order > 4 else 0

smooth_loss_fn = spline_losses.make_deriv_loss(
    smooth_order,
    ref_size=float(w),
    approx_method="analytical",
)

bbox_loss_fn = spline_losses.make_bbox_loss(
    ((0.0, 0.0), (float(w), float(h))), pad=bbox_pad
)

# Style loss
style_loss_fn = image_losses.CLIPPatchLoss(
    rgb=False,
    image_prompts=[style_img],
    model=clip_model,
    min_size=patch_size,
    cut_scale=0.0 if distortion_scale > 0.0 else 0.35,
    distortion_scale=distortion_scale,
    blur_sigma=0.0,
    thresh=0.0,
    n_cuts=64,
    num_batches=1,
    use_negative=False,
)

# SDS loss
if use_sds:
    print("Initializing SDS loss...")
    sds_loss_obj = sd.SDSLoss(
        prompt,
        augment=0,
        rgb=False,
        controlnet="lllyasviel/sd-controlnet-canny",
        seed=seed,
        t_range=(t_min, t_max),
        guidance_scale=cfg_scale,
        conditioning_scale=cond_scale,
        num_hifa_denoise_steps=4,
        ip_adapter="ip-adapter-plus_sd15.bin" if ip_adapter else "",
        ip_adapter_scale=ip_scale,
        time_schedule="ism",
        grad_method=grad_method,
        guess_mode=guess_mode,
    )
else:
    sds_loss_obj = None

##############################################
# Optimization loop

print("Starting optimization...")
time_count = 0.0

for step in range(num_opt_steps):
    perf_t = time.perf_counter()

    optimizer.zero_grad()

    # Render
    out_img, _ = scene.splat(max_deriv_order=_splat_deriv_order)
    im = out_img[..., 0]  # Take first channel (grayscale)

    # Compute losses
    loss = torch.tensor(0.0, device=device)

    # MSE loss
    if mse_w > 0:
        mse_loss = mse_loss_fn(im, target_t, mult=mse_mul)
        loss = loss + mse_w * mse_loss
    else:
        mse_loss = torch.tensor(0.0)

    # Smoothing loss
    if smooth_w > 0 and order > 4:
        smooth_loss = smooth_loss_fn(scene.nurbs_list)
        loss = loss + smooth_w * smooth_loss
    else:
        smooth_loss = torch.tensor(0.0)

    # Bbox loss
    if bbox_w > 0:
        bbox_loss = bbox_loss_fn(scene.nurbs_list)
        loss = loss + bbox_w * bbox_loss
    else:
        bbox_loss = torch.tensor(0.0)

    # Style loss
    if style_w > 0:
        style_loss = style_loss_fn(im)
        loss = loss + style_w * style_loss
    else:
        style_loss = torch.tensor(0.0)

    # SDS loss
    if sds_loss_obj is not None and sds_w > 0:
        sds_loss = sds_loss_obj(
            im,
            cond_img_pil,
            step,
            num_opt_steps,
            grad_scale=0.01 if grad_method == "ism" else 0.1,
            ip_adapter_image=input_img,
        )
        loss = loss + sds_w * sds_loss
    else:
        sds_loss = torch.tensor(0.0)

    # Backward and step
    loss.backward()
    optimizer.step()
    scheduler.step()

    # Width annealing
    current_minw = minw
    if width_anneal_start > 0 and step > num_opt_steps * width_anneal_start:
        start = num_opt_steps * width_anneal_start
        end = num_opt_steps * width_anneal_end
        t = np.clip((step - start) / (end - start), 0.0, 1.0)
        current_minw = minw - t * minw

    # Constrain parameters
    with torch.no_grad():
        for nurbs in scene.nurbs_list:
            nurbs._stroke_width.data.clamp_(current_minw, maxw)
            # _key_points are in pixel coords: clamp to image bounds
            nurbs._key_points.data[:, 0].clamp_(0, nurbs.W)
            nurbs._key_points.data[:, 1].clamp_(0, nurbs.H)
            nurbs._weights.data.clamp_(0.01, 10.0)
            nurbs._knot_interval.data.clamp_(0.0, 2.0)

    elapsed = time.perf_counter() - perf_t
    time_count += elapsed

    # Logging and saving
    must_save = step % save_every == save_every - 1 or step == num_opt_steps - 1

    if must_save:
        im_np = im.detach().cpu().numpy()
        step_img = Image.fromarray((im_np * 255).astype(np.uint8))
        step_img.save(os.path.join(output_path, f"step_{step:04d}.png"))

        log_msg = f"step {step:04d} | loss {loss.item():.4f}"
        log_msg += f" | mse {mse_loss.item():.4f}"
        log_msg += f" | smooth {smooth_loss.item():.4f}"
        log_msg += f" | bbox {bbox_loss.item():.4f}"
        log_msg += f" | style {style_loss.item():.4f}"
        if sds_loss_obj is not None:
            log_msg += f" | sds {sds_loss.item():.4f} (t={sds_loss_obj.t_saved})"
        log_msg += f" | time {time_count:.2f}s"
        print(log_msg)

# Save final output
im_np = im.detach().cpu().numpy()
final_img = Image.fromarray((im_np * 255).astype(np.uint8))
final_img.save(os.path.join(output_path, "output.png"))

print(f"\nOptimization complete. Total time: {time_count:.2f}s")
print(f"Final output saved to: {os.path.join(output_path, 'output.png')}")

# Print statistics
all_weights = torch.cat([nurbs._weights for nurbs in scene.nurbs_list])
all_knot_intervals = torch.cat([nurbs._knot_interval for nurbs in scene.nurbs_list])
all_stroke_widths = torch.cat([nurbs._stroke_width for nurbs in scene.nurbs_list])

print(f"\nTotal NURBS splines: {len(scene)}")
print("Weights max/min/mean:")
print(
    f"  {all_weights.max().item():.4f} / {all_weights.min().item():.4f} / {all_weights.mean().item():.4f}"
)
print("Knot intervals max/min/mean:")
print(
    f"  {all_knot_intervals.max().item():.4f} / {all_knot_intervals.min().item():.4f} / {all_knot_intervals.mean().item():.4f}"
)
print("Stroke widths max/min/mean:")
print(
    f"  {all_stroke_widths.max().item():.4f} / {all_stroke_widths.min().item():.4f} / {all_stroke_widths.mean().item():.4f}"
)
