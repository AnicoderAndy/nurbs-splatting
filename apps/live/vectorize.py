"""
LIVE-style Layer-wise Image Vectorization using NURBS Splatting.

Replaces DiffVG with the NURBS Splatting differentiable renderer
and uses closed filled NURBS curves instead of cubic Bézier paths.

Based on: Ma et al., "Towards Layer-wise Image Vectorization", CVPR 2022.
Backend:  NURBS Splatting (NURBSSplat / NURBSScene)

Usage:
    python vectorize.py --target path/to/image.png [OPTIONS]

Example:
    python vectorize.py --target data/demo.png --max_paths 16 --num_iter 500
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import time
import os
import os.path as osp
import random
import warnings

warnings.filterwarnings("ignore")

import cv2
import numpy as np
import numpy.random as npr
import torch
from PIL import Image
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from nsplat.nurbs import NURBSSplat
from nsplat.scene import NURBSScene
from nsplat import spline_losses

from nsplat.contrib.live.utils import (
    check_and_create_dir,
    get_path_schedule,
    linear_decay_lrlambda_f,
    naive_coord_init,
    random_coord_init,
    sparse_coord_init,
)
from nsplat.contrib.live.xing_loss import xing_loss


# ──────────────────────────── CLI ────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="LIVE-style image vectorization with NURBS Splatting."
    )
    p.add_argument("--target", type=str, required=True, help="Target image path.")
    p.add_argument(
        "--outdir", type=str, default="output_live", help="Output directory."
    )
    p.add_argument("--seed", type=int, default=0, help="Random seed.")

    # Path schedule
    p.add_argument(
        "--schedule_type",
        type=str,
        default="repeat",
        choices=["repeat", "list", "exp"],
        help="Path schedule type.",
    )
    p.add_argument(
        "--max_paths",
        type=int,
        default=10,
        help="Number of rounds (repeat) or total paths (exp).",
    )
    p.add_argument(
        "--schedule_each", type=int, default=1, help="Paths per round (repeat mode)."
    )
    p.add_argument("--exp_base", type=int, default=2, help="Base for exp schedule.")
    p.add_argument(
        "--max_path_per_iter",
        type=int,
        default=16,
        help="Cap per round for exp schedule.",
    )

    # NURBS parameters
    p.add_argument("--num_kp", type=int, default=12, help="Key points per NURBS curve.")
    p.add_argument("--degree", type=int, default=3, help="NURBS degree.")
    p.add_argument("--density", type=float, default=10, help="Sampling density.")
    p.add_argument(
        "--fill_boundary_samples",
        type=int,
        default=128,
        help="Boundary samples for fill.",
    )
    p.add_argument(
        "--fill_grid_step",
        type=float,
        default=1.0,
        help="Fill grid step (higher = faster but coarser). Default to 1.",
    )
    p.add_argument(
        "--init_radius",
        type=float,
        default=16.0,
        help="Initial circle radius (pixels).",
    )
    p.add_argument(
        "--stroke_width", type=float, default=1.0, help="Initial stroke width."
    )

    # Optimisation
    p.add_argument(
        "--num_iter", type=int, default=300, help="Inner iterations per round."
    )
    p.add_argument(
        "--lr_point",
        type=float,
        default=2.0,
        help="LR for key points. Set any of the following LRs to 0 to freeze that parameter group.",
    )
    p.add_argument("--lr_color", type=float, default=0.01, help="LR for color.")
    p.add_argument(
        "--lr_weights", type=float, default=0.01, help="LR for NURBS weights."
    )
    p.add_argument("--lr_knot", type=float, default=0.01, help="LR for knot intervals.")
    p.add_argument("--lr_opacity", type=float, default=0.01, help="LR for opacity.")
    p.add_argument("--lr_bg", type=float, default=0.01, help="LR for background color.")
    p.add_argument("--decay_ratio", type=float, default=0.4, help="LR decay ratio.")

    # Losses
    p.add_argument(
        "--use_distance_weighted_loss",
        action="store_true",
        default=True,
        help="Use UDF-weighted loss.",
    )
    p.add_argument(
        "--no_distance_weighted_loss",
        dest="use_distance_weighted_loss",
        action="store_false",
    )
    p.add_argument(
        "--xing_loss_weight", type=float, default=0.01, help="Xing loss weight."
    )
    p.add_argument(
        "--smooth_order",
        type=int,
        default=3,
        help="Derivative order for smoothing loss (default: 3).",
    )
    p.add_argument(
        "--smooth_w",
        type=float,
        default=3.0,
        help="Weight for smoothing loss. 0 disables it. Note: requires degree > smooth_order. Default off due to default degree=3.",
    )
    p.add_argument(
        "--use_l1_loss",
        action="store_true",
        default=False,
        help="Use L1 loss instead of MSE.",
    )

    # Initialisation
    p.add_argument(
        "--coord_init",
        type=str,
        default="sparse",
        choices=["sparse", "naive", "random"],
        help="Coordinate initialization strategy.",
    )

    # Training
    p.add_argument(
        "--trainable_bg",
        action="store_true",
        default=False,
        help="Make background color trainable.",
    )
    p.add_argument(
        "--trainable_record",
        action="store_true",
        default=True,
        help="Keep optimizing previously added curves.",
    )
    p.add_argument(
        "--no_trainable_record", dest="trainable_record", action="store_false"
    )

    # Saving
    p.add_argument(
        "--save_every", type=int, default=50, help="Save image every N inner iters."
    )
    p.add_argument(
        "--save_video",
        action="store_true",
        default=False,
        help="Save per-iteration PNG frames.",
    )

    # Step annealing
    p.add_argument(
        "--step_annealing",
        action="store_true",
        default=True,
        help="Enable cosine annealing of fill_grid_step within each round.",
    )
    p.add_argument(
        "--no_step_annealing",
        dest="step_annealing",
        action="store_false",
    )
    p.add_argument(
        "--step_anneal_begin",
        type=float,
        default=0.1,
        help="Proportion of round at which annealing begins (default: 0.1).",
    )
    p.add_argument(
        "--step_anneal_end",
        type=float,
        default=0.8,
        help="Proportion of round at which annealing ends (default: 0.8).",
    )
    p.add_argument(
        "--step_anneal_begin_val",
        type=float,
        default=4.0,
        help="fill_grid_step at annealing begin. 0 means use args.fill_grid_step.",
    )
    p.add_argument(
        "--step_anneal_end_val",
        type=float,
        default=1.0,
        help="fill_grid_step at annealing end (default: 1.0 = finest).",
    )
    return p.parse_args()


# ───────────────── NURBS circle init ──────────────────────────


def get_nurbs_circle_keypoints(
    center: list[float],
    radius: float,
    num_kp: int,
    stroke_width: float,
    device: torch.device,
) -> torch.Tensor:
    """Create key points for a closed NURBS circle.

    Args:
        center: ``[x, y]`` in pixel coordinates.
        radius: Circle radius in pixels.
        num_kp: Number of key points.
        stroke_width: Initial stroke width.
        device: Torch device.

    Returns:
        Key-point tensor of shape ``(num_kp, 3)``  (x, y, stroke_width).
    """
    angles = torch.linspace(0, 2 * math.pi, num_kp + 1, device=device)[:-1]
    kp_x = center[0] + radius * torch.cos(angles)
    kp_y = center[1] + radius * torch.sin(angles)
    sw = torch.full((num_kp,), stroke_width, device=device)
    return torch.stack([kp_x, kp_y, sw], dim=1)


# ──────────── Initialise new NURBS shapes ─────────────────────


def init_nurbs_shapes(
    num_paths: int,
    num_kp: int,
    degree: int,
    canvas_size: tuple[int, int],
    pos_init_method,
    init_radius: float,
    stroke_width: float,
    density: float,
    fill_boundary_samples: int,
    fill_grid_step: int,
    device: torch.device,
    gt: torch.Tensor | None = None,
) -> tuple[list[NURBSSplat], list[torch.Tensor], list[torch.Tensor]]:
    """Create *num_paths* new closed filled NURBS shapes.

    Returns:
        nurbs_list: List of new NURBSSplat objects.
        point_vars: List of key-point tensors (to be optimised).
        color_vars: List of color tensors (to be optimised).
    """
    H, W = canvas_size
    order = degree + 1
    nurbs_list: list[NURBSSplat] = []
    point_vars: list[torch.Tensor] = []
    color_vars: list[torch.Tensor] = []

    for _ in range(num_paths):
        center = pos_init_method()  # [x_pixel, y_pixel]

        kp = get_nurbs_circle_keypoints(
            center, init_radius, num_kp, stroke_width, device
        )

        # Determine initial color from GT
        if gt is not None:
            wref = max(0, min(int(center[0]), W - 1))
            href = max(0, min(int(center[1]), H - 1))
            color_init = gt[0, :, href, wref].clone().detach()  # (3,)
        else:
            color_init = torch.tensor(
                npr.uniform(size=[3]), dtype=torch.float32, device=device
            )

        weights = torch.ones(num_kp, device=device)

        nurbs = NURBSSplat(
            key_points=kp,
            weights=weights,
            order=order,
            width=W,
            height=H,
            density=density,
            color=color_init,
            closed=True,
            filled=True,
            fill_boundary_samples=fill_boundary_samples,
            fill_grid_step=fill_grid_step,
        )

        nurbs_list.append(nurbs)
        point_vars.append(nurbs._key_points)
        color_vars.append(nurbs._color)

    return nurbs_list, point_vars, color_vars


# ────────── GPU-based UDF weight from boundary points ─────────


def compute_udf_weight(
    nurbs_list: list[NURBSSplat],
    H: int,
    W: int,
    device: torch.device,
    truncate: float = 10.0,
) -> torch.Tensor:
    """Compute an unsigned-distance-field based loss weight on GPU.

    Uses the boundary sample points already stored in each NURBSSplat
    (populated by the preceding ``scene.splat()`` call) to compute the
    minimum distance from every pixel to the nearest boundary *segment*
    of the closed polyline.  The result is inverted and normalised to
    [0, 1] so pixels *near* the boundary receive high weight — matching
    the semantics of the original ``render_binary_mask`` + ``get_sdf``
    pipeline, but entirely on GPU without creating a temporary scene or
    transferring data to CPU.

    Returns:
        torch.Tensor: Weight map of shape ``(H, W)`` on *device*.
    """
    # Collect boundary polylines (pixel coords) from all new curves.
    boundaries: list[torch.Tensor] = []
    for nurbs in nurbs_list:
        sp = getattr(nurbs, "sample_points", None)
        if sp is not None and sp.shape[0] >= 2:
            boundaries.append(sp[:, :2].detach())

    if len(boundaries) == 0:
        return torch.zeros(H, W, device=device)

    # Build pixel grid at integer positions (matching rasteriser & skfmm).
    gy = torch.arange(H, device=device, dtype=torch.float32)
    gx = torch.arange(W, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")
    query = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)  # (Q, 2)
    Q = query.shape[0]

    min_dist_sq = torch.full((Q,), float("inf"), device=device)

    # For each curve, compute min distance from all pixels to boundary
    # *segments* (closed polyline), NOT just to the sample points.
    CHUNK_Q = 16384  # chunk over query pixels to limit GPU memory
    for bp in boundaries:
        # bp: (N, 2) — closed polyline boundary
        a = bp  # (N, 2)
        b = torch.roll(bp, shifts=-1, dims=0)  # (N, 2)
        ab = b - a  # (N, 2)
        ab_dot_ab = (ab * ab).sum(dim=-1)  # (N,)

        for qi in range(0, Q, CHUNK_Q):
            q = query[qi : qi + CHUNK_Q]  # (C, 2)
            aq = q.unsqueeze(1) - a.unsqueeze(0)  # (C, N, 2)
            aq_dot_ab = (aq * ab.unsqueeze(0)).sum(-1)  # (C, N)
            t = (aq_dot_ab / ab_dot_ab.unsqueeze(0).clamp(min=1e-12)).clamp(0.0, 1.0)
            closest = a.unsqueeze(0) + t.unsqueeze(-1) * ab.unsqueeze(0)  # (C, N, 2)
            diff = q.unsqueeze(1) - closest  # (C, N, 2)
            dsq = (diff * diff).sum(dim=-1)  # (C, N)
            chunk_min, _ = dsq.min(dim=1)  # (C,)
            min_dist_sq[qi : qi + CHUNK_Q] = torch.min(
                min_dist_sq[qi : qi + CHUNK_Q], chunk_min
            )

    dist = torch.sqrt(min_dist_sq + 1e-12).reshape(H, W)

    # Truncate, invert, normalise to [0, 1]  (same as get_sdf … normalize="to1")
    dist = dist.clamp(max=truncate)
    sd = dist.max() - dist  # invert: near boundary → high weight
    m = sd.max()
    if m > 0:
        sd = sd / m

    return sd


# ──────────────── Save helper ─────────────────────────────────


def save_image(tensor_hw3: torch.Tensor, path: str):
    """Save a (H, W, 3) float tensor as a PNG image."""
    check_and_create_dir(path)
    img = (tensor_hw3.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)


# ═══════════════════════════  MAIN  ═══════════════════════════


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimize_parameters = []
    if args.lr_point > 0:
        optimize_parameters.append("key_points")
    if args.lr_weights > 0:
        optimize_parameters.append("weights")
    if args.lr_knot > 0:
        optimize_parameters.append("knot_interval")
    if args.lr_color > 0:
        optimize_parameters.append("color")
    if args.lr_opacity > 0:
        optimize_parameters.append("opacity")

    # Seed
    random.seed(args.seed)
    npr.seed(args.seed)
    torch.manual_seed(args.seed)

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # ──────────── Load target ─────────────────────────────────
    gt_pil = Image.open(args.target).convert("RGB")
    gt_np = np.array(gt_pil).astype(np.float32) / 255.0
    H, W = gt_np.shape[:2]
    gt = (
        torch.tensor(gt_np, device=device).permute(2, 0, 1).unsqueeze(0)
    )  # (1, 3, H, W)
    print(f"Target: {args.target}  ({W}x{H}), device={device}")

    save_image(torch.tensor(gt_np, device=device), osp.join(outdir, "target.png"))

    # ──────────── Path schedule ───────────────────────────────
    schedule_kwargs = {"type": args.schedule_type}
    if args.schedule_type == "repeat":
        schedule_kwargs["max_path"] = args.max_paths
        schedule_kwargs["schedule_each"] = args.schedule_each
    elif args.schedule_type == "exp":
        schedule_kwargs["base"] = args.exp_base
        schedule_kwargs["max_path"] = args.max_paths
        schedule_kwargs["max_path_per_iter"] = args.max_path_per_iter
    elif args.schedule_type == "list":
        schedule_kwargs["schedule"] = [1] * args.max_paths  # default list
    path_schedule = get_path_schedule(**schedule_kwargs)
    print(f"Path schedule: {path_schedule}  (total {sum(path_schedule)} curves)")

    # ──────────── Scene ───────────────────────────────────────
    if args.trainable_bg:
        para_bg = torch.tensor([1.0, 1.0, 1.0], device=device, requires_grad=True)
    else:
        para_bg = torch.tensor([1.0, 1.0, 1.0], device=device, requires_grad=False)

    scene = NURBSScene(width=W, height=H, background=para_bg.detach())
    scene.to(device)

    # ──────────── Coordinate initialiser ──────────────────────
    bg_img = para_bg.detach().view(1, -1, 1, 1).expand(1, 3, H, W)
    if args.coord_init == "sparse":
        pos_init_method = sparse_coord_init(bg_img, gt)
    elif args.coord_init == "naive":
        pos_init_method = naive_coord_init(bg_img, gt)
    else:
        pos_init_method = random_coord_init([H, W])

    # ──────────── Smoothing loss ──────────────────────────────
    order = args.degree + 1
    _splat_deriv_order = args.smooth_order if args.smooth_w > 0 and order > 4 else 0
    print(_splat_deriv_order)
    smooth_loss_fn = spline_losses.make_deriv_loss(
        args.smooth_order,
        ref_size=float(W),
        approx_method="analytical",
        device=str(device),
    )

    # ──────────── Training ────────────────────────────────────
    lrlambda_f = linear_decay_lrlambda_f(args.num_iter, args.decay_ratio)
    optim_scheduler_dict: dict[int, tuple] = {}

    loss_weight = None
    loss_weight_keep: torch.Tensor | int = 0
    all_nurbs: list[NURBSSplat] = []  # track all curves for rendering
    pathn_record: list[int] = []
    accumulated_pathn = 0

    # ──────────── Step annealing helper ─────────────────────
    step_anneal_begin_val = (
        args.step_anneal_begin_val
        if args.step_anneal_begin_val > 0
        else args.fill_grid_step
    )
    step_anneal_end_val = args.step_anneal_end_val

    def _annealed_grid_step(t: int, num_iter: int) -> float:
        """Cosine annealing of fill_grid_step within a round.

        Returns *begin_val* before the annealing window, *end_val* after,
        and a cosine interpolation in between.
        """
        if not args.step_annealing:
            return args.fill_grid_step
        frac = t / max(num_iter - 1, 1)
        if frac <= args.step_anneal_begin:
            return step_anneal_begin_val
        if frac >= args.step_anneal_end:
            return step_anneal_end_val
        # Cosine interpolation in [begin, end]
        phase = (frac - args.step_anneal_begin) / (
            args.step_anneal_end - args.step_anneal_begin
        )
        cos_val = 0.5 * (1.0 + math.cos(math.pi * (1.0 - phase)))
        return (
            step_anneal_begin_val
            + (step_anneal_end_val - step_anneal_begin_val) * cos_val
        )

    print("Optimization starts.")
    start_time = time.perf_counter()

    for path_idx, pathn in enumerate(path_schedule):
        print(f"\n=> Round {path_idx}: adding {pathn} NURBS curve(s) ...")
        pathn_record.append(pathn)
        accumulated_pathn += pathn
        round_str = f"round{path_idx:03d}-paths{accumulated_pathn}"

        # ---- Initialise new NURBS ----
        new_nurbs, point_vars, color_vars = init_nurbs_shapes(
            num_paths=pathn,
            num_kp=args.num_kp,
            degree=args.degree,
            canvas_size=(H, W),
            pos_init_method=pos_init_method,
            init_radius=args.init_radius,
            stroke_width=args.stroke_width,
            density=args.density,
            fill_boundary_samples=args.fill_boundary_samples,
            fill_grid_step=args.fill_grid_step,
            device=device,
            gt=gt,
        )

        # Add to scene
        for nn_ in new_nurbs:
            scene.add_nurbs(nn_)
        all_nurbs.extend(new_nurbs)

        # ---- Optimiser for new curves ----
        para: list[dict] = []
        for nn_ in new_nurbs:
            para.extend(
                nn_.get_params(
                    optimize=optimize_parameters,
                    lrs={
                        "key_points": args.lr_point,
                        "weights": args.lr_weights,
                        "knot_interval": args.lr_knot,
                        "color": args.lr_color,
                        "opacity": args.lr_opacity,
                    },
                )
            )
        if args.trainable_bg and path_idx == 0:
            para.append({"params": para_bg, "lr": args.lr_bg})

        optim = torch.optim.Adam(para)
        if args.trainable_record:
            scheduler = LambdaLR(optim, lr_lambda=lrlambda_f, last_epoch=-1)
        else:
            scheduler = LambdaLR(optim, lr_lambda=lrlambda_f, last_epoch=args.num_iter)
        optim_scheduler_dict[path_idx] = (optim, scheduler)

        # ---- Inner optimisation loop ----
        t_range = tqdm(range(args.num_iter), desc=f"Round {path_idx}")
        for t in t_range:
            # ---- Step annealing ----
            current_step = _annealed_grid_step(t, args.num_iter)
            scene.set_fill_grid_step(current_step)

            # Zero gradients for ALL active optimisers
            for _, (opt, _) in optim_scheduler_dict.items():
                opt.zero_grad()

            # ---- Forward: render ----
            scene.background = para_bg.detach() if not args.trainable_bg else para_bg
            out_img, _ = scene.splat(
                max_deriv_order=_splat_deriv_order,
            )  # (H, W, 3)

            x = out_img.unsqueeze(0).permute(0, 3, 1, 2)  # (1, 3, H, W)

            # ---- Compute loss ----
            if args.use_l1_loss:
                loss = torch.abs(x - gt)
            else:
                loss = (x - gt) ** 2

            # ---- UDF weighting (pure GPU, no temp scene) ----
            if args.use_distance_weighted_loss:
                with torch.no_grad():
                    lw = compute_udf_weight(new_nurbs, H, W, device)
                    if isinstance(loss_weight_keep, torch.Tensor):
                        lw = lw + loss_weight_keep
                    lw = lw.clamp(0, 1)
                loss_weight = lw

            if loss_weight is not None:
                loss = (loss.sum(1) * loss_weight).mean()
            else:
                loss = loss.sum(1).mean()

            # ---- Smoothing loss ----
            smooth_loss = torch.tensor(0.0, device=device)
            if args.smooth_w > 0 and order > 4:
                smooth_loss = smooth_loss_fn(scene.nurbs_list)
                loss = loss + args.smooth_w * smooth_loss

            # ---- Xing loss on new curves ----
            if args.xing_loss_weight > 0:
                kp_xy_list = [
                    (
                        nn_._key_points[:, :2]
                        if nn_._key_points.shape[1] > 2
                        else nn_._key_points
                    )
                    for nn_ in new_nurbs
                ]
                # Only apply if we have actual key-point params
                if kp_xy_list:
                    loss_xing = xing_loss(kp_xy_list) * args.xing_loss_weight
                    loss = loss + loss_xing

            t_range.set_postfix({"loss": f"{loss.item():.6f}"})
            loss.backward()

            # ---- Step all active optimisers ----
            for _, (opt, sched) in optim_scheduler_dict.items():
                opt.step()
                sched.step()

            # ---- Clamp parameters ----
            with torch.no_grad():
                for nn_ in all_nurbs:
                    nn_._key_points[:, 0].clamp_(0, W)
                    nn_._key_points[:, 1].clamp_(0, H)
                    nn_._weights.clamp_(0.01, 10.0)
                    nn_._knot_interval.clamp_(0.1, 2.0)
                    nn_._color.clamp_(0.0, 1.0)
                    nn_._opacity.clamp_(0.0, 1.0)
                if args.trainable_bg:
                    para_bg.data.clamp_(0.0, 1.0)

            # ---- Save intermediate ----
            if args.save_video or (args.save_every > 0 and t % args.save_every == 0):
                fname = osp.join(outdir, "video", f"{round_str}-iter{t:04d}.png")
                save_image(out_img, fname)

        # ---- End of round bookkeeping ----
        if args.use_distance_weighted_loss and loss_weight is not None:
            loss_weight_keep = loss_weight.detach().clone()

        if not args.trainable_record:
            # Freeze previous parameters
            optim_scheduler_dict = {}

        # Save round output
        with torch.no_grad():
            out_img_final, _ = scene.splat()
        save_image(out_img_final, osp.join(outdir, f"round_{path_idx:03d}.png"))
        # Re-init coordinate selector from current error
        with torch.no_grad():
            x_current = out_img_final.unsqueeze(0).permute(0, 3, 1, 2)
        if args.coord_init == "sparse":
            pos_init_method = sparse_coord_init(x_current, gt)
        elif args.coord_init == "naive":
            pos_init_method = naive_coord_init(x_current, gt)
        else:
            pos_init_method = random_coord_init([H, W])

        # Log round summary
        elapsed = time.perf_counter() - start_time
        print(f"Round {path_idx} completed. Time elapsed: {elapsed:.1f}s.")

    end_time = time.perf_counter()
    total_time = end_time - start_time
    print(
        f"\nDone. Total {sum(path_schedule)} NURBS curves added.\n"
        f"Total time: {total_time:.1f}s."
    )
    # ──────────── Final output ────────────────────────────────
    with torch.no_grad():
        final_img, _ = scene.splat()
    save_image(final_img, osp.join(outdir, "final.png"))

    json_path = osp.join(outdir, "final.json")
    scene.export_to_json(json_path)
    print(f"Saved scene JSON to '{json_path}'")

    print(f"\nDone. {sum(path_schedule)} NURBS curves. Final loss: {loss.item():.6f}")
    print(f"Results saved to {outdir}/")

    # ──────────── Log timing to CSV ───────────────────────────
    csv_path = osp.join(osp.dirname(osp.abspath(outdir)), "timings.csv")
    target_name = osp.splitext(osp.basename(args.target))[0]
    write_header = not osp.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["target", "time_seconds", "num_curves", "final_loss"])
        writer.writerow(
            [target_name, f"{total_time:.2f}", sum(path_schedule), f"{loss.item():.6f}"]
        )
    print(f"Timing appended to {csv_path}")


if __name__ == "__main__":
    main()
