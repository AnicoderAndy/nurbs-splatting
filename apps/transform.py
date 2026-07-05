"""
Optimize a filled closed NURBS curve to match a target image.

Initializes keypoints as a CCW circle and uses MSE loss to fit.
Saves step images to an output folder instead of using a GUI.

Usage:
    python transform.py --target target.jpg
"""

import argparse
import pathlib
import math
import time

import numpy as np
import torch
from PIL import Image

from nsplat.nurbs import NURBSSplat
from nsplat.scene import NURBSScene
from nsplat.utils import make_cosine_lr_scheduler
from nsplat import spline_losses

# ──────────────────────────── CLI ────────────────────────────
parser = argparse.ArgumentParser(
    description="Fit a red filled NURBS curve to a target image."
)
parser.add_argument(
    "--target", type=str, default="data/target.jpg", help="Path to target image."
)
parser.add_argument(
    "--outdir", type=str, default="output_transform", help="Directory for step images."
)
parser.add_argument(
    "--num_kp", type=int, default=24, help="Number of keypoints on the initial circle."
)
parser.add_argument(
    "--degree", type=int, default=5, help="NURBS degree (order = degree + 1)."
)
parser.add_argument(
    "--density", type=int, default=10, help="Density for NURBS sampling."
)
parser.add_argument(
    "--fill_boundary_samples",
    type=int,
    default=128,
    help="Boundary samples for winding number.",
)
parser.add_argument(
    "--stroke_width", type=float, default=2.0, help="Initial stroke width."
)
parser.add_argument(
    "--lr_pos", type=float, default=2.0, help="Learning rate for key points."
)
parser.add_argument(
    "--lr_weights", type=float, default=0.005, help="Learning rate for NURBS weights."
)
parser.add_argument(
    "--lr_interval", type=float, default=0.005, help="Learning rate for knot intervals."
)
parser.add_argument("--steps", type=int, default=300, help="Optimisation steps.")
parser.add_argument(
    "--save_every", type=int, default=1, help="Save an image every N steps."
)
parser.add_argument(
    "--radius", type=float, default=0.1, help="Initial circle radius in NDC."
)
parser.add_argument(
    "--lr_color", type=float, default=0.01, help="Learning rate for color."
)
parser.add_argument(
    "--smooth_order", type=int, default=3, help="Derivative order for smoothing loss."
)
parser.add_argument(
    "--smooth_w", type=float, default=15.0, help="Weight for smoothing loss."
)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# ──────────────────────────── Setup ──────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(args.seed)
np.random.seed(args.seed)

outdir = pathlib.Path(args.outdir)
outdir.mkdir(parents=True, exist_ok=True)

# Load target image (RGB, float32 in [0, 1])
target_pil = Image.open(args.target).convert("RGB")
target_np = np.array(target_pil).astype(np.float32) / 255.0
H, W = target_np.shape[:2]
target_t = torch.tensor(target_np, device=device)  # (H, W, 3)

# ──────────────── Circle keypoints (CCW) ─────────────────────
num_kp = args.num_kp
angles = torch.linspace(0, 2 * math.pi, num_kp + 1, device=device)[
    :-1
]  # exclude duplicate
# CCW: x = cos(θ), y = -sin(θ)  (image y-axis points down)
cx, cy = 0.0, 0.0  # centre in NDC
kp_x = cx + args.radius * torch.cos(angles)
kp_y = cy + args.radius * torch.sin(angles)
kp_xy = torch.stack([kp_x, kp_y], dim=1)  # (num_kp, 2)
sw = torch.full((num_kp, 1), args.stroke_width, device=device)
key_points = torch.cat([kp_xy, sw], dim=1)  # (num_kp, 3)

order = args.degree + 1
weights = torch.ones(num_kp, device=device)

# ───────────────── Create filled NURBS ───────────────────────
nurbs = NURBSSplat(
    key_points,
    weights,
    order,
    width=W,
    height=H,
    density=args.density,
    color=torch.tensor([0.0, 0.0, 1.0], device=device),
    closed=True,
    filled=True,
    fill_boundary_samples=args.fill_boundary_samples,
    init_with_ndc=True,
)

scene = NURBSScene(width=W, height=H)
scene.to(device)
scene.add_nurbs(nurbs)

# ──────────────── Smoothing loss ─────────────────────────────
_splat_deriv_order = args.smooth_order if args.smooth_w > 0 and order > 4 else 0

smooth_loss_fn = spline_losses.make_deriv_loss(
    args.smooth_order,
    ref_size=float(W),
    approx_method="analytical",
)

# ──────────────────── Optimizer ──────────────────────────────
optimizer = torch.optim.Adam(
    scene.get_params(
        optimize=("key_points", "weights", "knot_interval", "color"),
        lrs={
            "key_points": args.lr_pos,
            "weights": args.lr_weights,
            "knot_interval": args.lr_interval,
            "color": args.lr_color,
        },
    )
)
scheduler = make_cosine_lr_scheduler(optimizer, args.steps, lr_min_scale=0.1)


# ──────────────────── Helper ─────────────────────────────────
def save_image(tensor_hw3: torch.Tensor, path: str):
    """Save a (H, W, 3) float tensor as an image."""
    img = (tensor_hw3.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)


# Save target for reference
save_image(target_t, str(outdir / "target.png"))

# ──────────────── Optimisation loop ──────────────────────────
print(f"Target: {args.target}  ({W}x{H})")
print(f"Device: {device}")
print(f"Steps : {args.steps}, saving every {args.save_every}")
print("Optimization starts.")

start_time = time.perf_counter()
smooth_loss = torch.tensor(0.0, device=device)  # for logging if not computed
for step in range(args.steps):
    optimizer.zero_grad()
    out_img, _ = scene.splat(max_deriv_order=_splat_deriv_order)  # (H, W, 3)
    mse = torch.nn.functional.mse_loss(out_img, target_t)
    loss = mse

    # Smoothing loss
    if args.smooth_w > 0 and order > 4:
        smooth_loss = smooth_loss_fn(scene.nurbs_list)
        loss = loss + args.smooth_w * smooth_loss

    loss.backward()
    optimizer.step()
    scheduler.step()

    # Clamp parameters to valid ranges
    with torch.no_grad():
        for n in scene.nurbs_list:
            # _key_points are in pixel coords: clamp to image bounds
            n._key_points[:, 0].clamp_(0, n.W)
            n._key_points[:, 1].clamp_(0, n.H)
            n._weights.clamp_(0.01, 10.0)
            n._knot_interval.clamp_(0.0, 2.0)
            n._color.clamp_(0.0, 1.0)

    if step % args.save_every == 0 or step == args.steps - 1:
        elapsed = time.perf_counter() - start_time
        print(
            f"step {step:04d} | loss {loss.item():.6f} | mse {mse.item():.6f} |"
            f" smooth {smooth_loss.item():.6f} | time {elapsed:.1f}s"
        )
        save_image(out_img, str(outdir / f"step_{step:04d}.png"))

elapsed_total = time.perf_counter() - start_time
print(f"Optimization completed in {elapsed_total:.1f}s.")
print("Results saved to", outdir)
