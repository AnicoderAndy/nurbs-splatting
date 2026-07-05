"""
The core NURBS Splat class.
Author: Jingye Qiu
"""

import json
import math

import torch
import torch.nn as nn
import numpy as np
from typing import Literal, Optional, Tuple, Union, overload
from gsplat.project_gaussians_2d_scale_rot import project_gaussians_2d_scale_rot
from gsplat.rasterize import rasterize_gaussians
import logging

_gramian_cache = {}


class NURBSSplat(nn.Module):
    def __init__(
        self,
        key_points: torch.Tensor,
        weights: torch.Tensor,
        order: int,
        width: int = 1024,
        height: int = 1024,
        density: float = 10,
        color: torch.Tensor = torch.tensor([0.0, 0.0, 0.0]),
        closed: bool = False,
        filled: bool = False,
        fill_boundary_samples: int = 100,
        fill_grid_step: float = 1.0,
        scale_ratio: float = 2.0,
        init_with_ndc: bool = False,
        opacity: float = 1.0,
        endpoint_repeat_times: int = 2,
        **kwargs,
    ):
        """
        Initialize a NURBS splat with key points, weights, and degree.

        Args:
            key_points (torch.Tensor): Control points of shape (M, 3). Given in pixel coordinates.
                The third dimension is the stroke width for each key point.
            weights (torch.Tensor): Weights for each control point of shape (M,).
            order (int): Order of the NURBS. Note that order(k) = degree(p) + 1.
            color (torch.Tensor): Color of the splat.
        """
        self.W = width
        self.H = height
        self._filled = filled
        self._fill_boundary_samples = fill_boundary_samples
        self._fill_grid_step = fill_grid_step
        if not closed and filled:
            logging.warning("Forced closed=True when filled=True.")
            closed = True
        self._closed = closed
        if key_points.shape[0] < 2:
            raise ValueError("At least two key points are required.")
        if key_points.shape[1] < 3:
            raise ValueError(
                "key_points must have shape (M, 3) with stroke width in the third dimension."
            )
        if not closed and key_points.shape[0] < order:
            raise ValueError(
                f"Open curves require at least {order} key points (order={order}), "
                f"but only {key_points.shape[0]} were given."
            )
        super(NURBSSplat, self).__init__()

        # Store key points in pixel coordinates for better precision
        if not init_with_ndc:
            self._key_points = key_points[:, :2]
        else:
            self._key_points = self.ndc_to_pixel(key_points[:, :2])

        self._stroke_width = key_points[:, 2]
        self._order = order
        self._control_points = self._compute_ctrls()
        self._weights = weights

        # Initialize knot intervals
        self._knot_interval = self._init_knot_interval(kwargs.get("knot_vec", None))
        self._knot_vector = self._generate_knots()

        # Compute number of spans based on curve type
        n_kp = key_points.shape[0]
        k = order
        p = k - 1
        if closed:
            # For closed curves: n_cp = n_kp + p, num_spans = n_kp
            self._num_spans = n_kp
        else:
            # For open curves: control points = key points directly
            self._num_spans = n_kp - k + 1

        self._density = density
        self._scale_ratio = scale_ratio
        self._color = color
        self._opacity = torch.tensor(
            opacity, dtype=torch.float32, device=key_points.device
        )
        self.BLOCK_W = 16
        self.BLOCK_H = 16
        self._endpoint_repeat_times: int = endpoint_repeat_times
        self.sample_points: torch.Tensor = self._sample_points(compute_derivative=False)

    @property
    def fill_grid_step(self) -> float:
        """Current fill grid step size."""
        return self._fill_grid_step

    @fill_grid_step.setter
    def fill_grid_step(self, value: float) -> None:
        """Set the fill grid step size.

        Args:
            value (float): New grid step. Must be >= 1.
        """
        # if value < 1:
        # raise ValueError(f"fill_grid_step must be >= 1, got {value}")
        self._fill_grid_step = value

    @property
    def density(self) -> float:
        """Current sampling density."""
        return self._density

    @density.setter
    def density(self, value: float) -> None:
        """Set the sampling density.

        Args:
            value (float): New density. Must be > 0.
        """
        if value <= 0:
            raise ValueError(f"density must be > 0, got {value}")
        self._density = value

    def get_params(
        self,
        optimize=("key_points", "stroke_width", "weights"),
        lrs=None,
        default_lr: float = 0.01,
    ) -> list:
        """Return optimizer parameter groups.

        Args:
            optimize (Iterable[str] | str): Parameters to optimize. Supported values:
                "key_points", "stroke_width", "weights", "knot_interval", "color".
            lrs (dict | None): Optional per-parameter learning rates
            default_lr (float): Default learning rate if not specified in `lrs`.

        Returns:
            list: List of parameter group dicts in the form
                [{"params": tensor, "lr": float}, ...].
        """
        if isinstance(optimize, str):
            optimize = (optimize,)

        aliases = {
            "key_point": "key_points",
            "keypoints": "key_points",
            "keypoint": "key_points",
            "stroke": "stroke_width",
            "width": "stroke_width",
            "weight": "weights",
            "knot": "knot_interval",
            "knots": "knot_interval",
            "knot_intervals": "knot_interval",
            "colors": "color",
            "colour": "color",
            "alpha": "opacity",
        }

        normalized = []
        for name in optimize:
            if not isinstance(name, str):
                continue
            key = aliases.get(name, name)
            normalized.append(key)

        lrs = lrs or {}
        param_groups = []

        # Ensure only requested tensors are leaf Parameters.
        def _ensure_param(attr_name: str) -> nn.Parameter:
            tensor = getattr(self, attr_name)
            if isinstance(tensor, nn.Parameter):
                return tensor
            param = nn.Parameter(tensor.clone().detach())
            setattr(self, attr_name, param)
            return param

        if "key_points" in normalized:
            self._key_points = _ensure_param("_key_points")
            param_groups.append(
                {
                    "params": self._key_points,
                    "lr": float(lrs.get("key_points", default_lr)),
                }
            )
        if "stroke_width" in normalized:
            self._stroke_width = _ensure_param("_stroke_width")
            param_groups.append(
                {
                    "params": self._stroke_width,
                    "lr": float(lrs.get("stroke_width", default_lr)),
                }
            )
        if "weights" in normalized:
            self._weights = _ensure_param("_weights")
            param_groups.append(
                {"params": self._weights, "lr": float(lrs.get("weights", default_lr))}
            )
        if "knot_interval" in normalized:
            self._knot_interval = _ensure_param("_knot_interval")
            param_groups.append(
                {
                    "params": self._knot_interval,
                    "lr": float(lrs.get("knot_interval", default_lr)),
                }
            )
        if "color" in normalized:
            self._color = _ensure_param("_color")
            param_groups.append(
                {
                    "params": self._color,
                    "lr": float(lrs.get("color", default_lr)),
                }
            )
        if "opacity" in normalized:
            self._opacity = _ensure_param("_opacity")
            param_groups.append(
                {
                    "params": self._opacity,
                    "lr": float(lrs.get("opacity", default_lr)),
                }
            )

        return param_groups

    def sample_gaussians(
        self,
        factor: int = 1,
        max_deriv_order: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample gaussians along the spline.

        Algorithm:
          1. Approximate the arc length using the 1st derivative.
          2. Number of samples = density * arc_length.
          3. Uniformly sample along the curve; convert positions to NDC.
             Optionally compute and store higher-order derivatives.
          4. scale_x = scale_y = stroke_width / scale_ratio  (in pixels).

        Args:
            factor (int): Resolution scaling factor.
            max_deriv_order (int): If > 0, compute analytical derivatives up to
                this order and store them in ``self.sample_derivs``.
        """
        self._control_points = self._compute_ctrls()
        self._knot_vector = self._generate_knots()

        knots = self._knot_vector
        k = self._order

        # Step 1: Approximate arc length using the analytical 1st derivative.
        #   arc_length ≈ ∫ ||C'(u)||₂ du  (trapezoidal rule)
        # n_length_samples = max(self.W, self.H) * 3
        n_length_samples = int(max(self.W, self.H))
        length_samples = self._sample_points(
            factor=1,
            compute_derivative=False,
            num_samples=n_length_samples,
        )  # pixel coords
        u_min = knots[k - 1]
        u_max = knots[-k]

        # use numerical derivative for length approximation
        length_d1 = (
            torch.diff(length_samples, dim=0) / (u_max - u_min) * (n_length_samples - 1)
        )
        h_len = ((u_max - u_min) / (n_length_samples - 1)).detach()
        arc_length_pixel = torch.sum(
            torch.sqrt(torch.sum(length_d1[:, :2] ** 2, dim=1)) * h_len
        ).detach()

        if self._filled:
            eff_edriv = max(max_deriv_order, 1)
            samples_res = self._sample_points(
                factor=1,
                num_samples=max(int(arc_length_pixel * 0.5), 2),
                max_deriv_order=eff_edriv,
            )
            samples = samples_res[0]  # (N, 3) pixel coords of sampled points
            self.sample_derivs = list(samples_res[1:])  # store derivatives if computed
            self.sample_points = samples  # keep sample_points up-to-date
            boundary_px = samples[:, :2]
            fill_xy, fill_scales, fill_rotations, fill_opacity = (
                self._compute_fill_gaussians(boundary_px, factor)
            )
            colors = self._color.view(1, 3).repeat(fill_xy.shape[0], 1)
            fill_opacity = fill_opacity * self._opacity
            return (fill_xy, fill_scales, fill_rotations, colors, fill_opacity)

        # Step 2: Number of samples = density * arc_length
        num_samples = max(int(self._density * arc_length_pixel.item() * factor), 2)

        # Step 3: Sample curve points (pixel coords) and convert to NDC.
        #   Request analytical derivatives too
        eff_deriv = max_deriv_order if max_deriv_order > 0 else 0
        sample_result = self._sample_points(
            factor,
            num_samples=num_samples,
            max_deriv_order=eff_deriv,
        )

        # we really did not design a clean API for returning optional derivatives, so here we:
        self.sample_points = (
            sample_result[0] if eff_deriv > 0 else sample_result
        )  # (N, 3)
        # Store analytical derivatives (list of tensors, each (N, 3))
        self.sample_derivs: list[torch.Tensor] = (
            list(sample_result[1:]) if eff_deriv > 0 else []
        )

        xyz = self.sample_points  # (N, 3) pixel
        xy = self.pixel_to_ndc(xyz[:, :2])  # (N, 2) NDC for gaussians
        widths = xyz[:, 2]  # pixel stroke widths

        # Step 4: scale_x = scale_y = stroke_width / scale_ratio (pixels)
        rotations = torch.zeros((xy.shape[0], 1), device=xy.device)
        scales = self._compute_scale(xy.view(-1, 2), widths)

        endpoint_repeat = int(self._density * self._endpoint_repeat_times)
        xy = torch.cat(
            [
                xy[:1].repeat(endpoint_repeat, 1),
                xy,
                xy[-1:].repeat(endpoint_repeat, 1),
            ],
            dim=0,
        )
        widths = torch.cat(
            [
                widths[:1].repeat(endpoint_repeat),
                widths,
                widths[-1:].repeat(endpoint_repeat),
            ],
            dim=0,
        )
        rotations = torch.cat(
            [
                rotations[:1].repeat(endpoint_repeat, 1),
                rotations,
                rotations[-1:].repeat(endpoint_repeat, 1),
            ],
            dim=0,
        )
        scales = torch.cat(
            [
                scales[:1].repeat(endpoint_repeat, 1),
                scales,
                scales[-1:].repeat(endpoint_repeat, 1),
            ],
            dim=0,
        )

        opacity = torch.ones((xy.shape[0], 1), device=xy.device) * self._opacity
        colors = self._color.view(1, 3).repeat(xy.shape[0], 1)

        return (xy, scales, rotations, colors, opacity)

    def splat(
        self, factor=1, max_deriv_order: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Render the NURBS into an image.
        Args:
            factor (int, optional): Scaling factor for the output dimensions. Defaults to 1.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the output image tensor and
                the sampled points tensor.
        """
        final_h = int(self.H * factor)
        final_w = int(self.W * factor)

        xy, scales, rotations, colors, opacity = self.sample_gaussians(
            factor, max_deriv_order=max_deriv_order
        )

        self.tile_bounds = (
            (final_w + self.BLOCK_W - 1) // self.BLOCK_W,
            (final_h + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        )

        self.xys, depths_, self.radii, conics, num_tiles_hit = (
            project_gaussians_2d_scale_rot(
                xy, scales, rotations, final_h, final_w, self.tile_bounds
            )
        )

        out_img = rasterize_gaussians(
            self.xys,
            depths_,
            self.radii,
            conics,
            num_tiles_hit,  # type: ignore
            colors,
            opacity,
            final_h,
            final_w,
            self.BLOCK_H,
            self.BLOCK_W,
            return_alpha=False,
        )
        return out_img, self.sample_points

    def approx_grad_energy(
        self,
        der: int = 1,
        sw_weight: float = 0.1,
        normalize: bool = False,
        normalize_size: float | None = None,
    ) -> torch.Tensor:
        """Approximate gradient energy using a discrete Gram matrix.

        Energy = $Q^T G Q$, where Q = control points * corresponding weights.

        Args:
            der (int): Derivative order for the finite-difference operator.
            drop_sw (bool): Whether to drop the stroke width dimension.
            normalize (bool): Whether to normalize by curve length.
            normalize_size (float | None): Optional precomputed length.

        Returns:
            torch.Tensor: Scalar energy value.
        """
        cp = self._compute_ctrls()
        cp = cp.clone()
        cp[:, 2] = cp[:, 2] * sw_weight
        weights = self._expand_weights(self._weights)
        Q = cp * weights.unsqueeze(1)

        n = Q.shape[0]
        dim = Q.shape[1]
        cache_key = (n, der, dim, Q.device, Q.dtype)
        if cache_key in _gramian_cache:
            G = _gramian_cache[cache_key]
        else:
            D = torch.diff(torch.eye(n, device=Q.device, dtype=Q.dtype), n=der, dim=0)
            Gd = D.T @ D
            I = torch.eye(dim, device=Q.device, dtype=Q.dtype)
            G = torch.kron(Gd, I)
            _gramian_cache[cache_key] = G

        if normalize or normalize_size is not None:
            if normalize_size is None:
                P = self.sample_points[:, :2]
                diffs = torch.diff(P, dim=0)
                l = torch.sum(torch.sqrt(torch.sum(diffs**2, dim=1))).clamp_min(1e-8)
            else:
                l = torch.tensor(normalize_size, device=Q.device, dtype=Q.dtype)
        else:
            l = torch.tensor(1.0, device=Q.device, dtype=Q.dtype)

        Qhat = Q.reshape(-1, 1) / l
        return (Qhat.T @ G @ Qhat).squeeze()

    def pixel_to_ndc(self, pts: torch.Tensor) -> torch.Tensor:
        """
        Convert pixel coordinates to normalized device coordinates (NDC) in the range [-1, 1].

        Args:
            pts (torch.Tensor): Pixel coordinates of shape (N, 2).

        Returns:
            torch.Tensor: NDC coordinates of shape (N, 2).
        """
        ndc_coords = (pts / torch.tensor([self.W, self.H], device=pts.device)) * 2 - 1
        return ndc_coords

    def ndc_to_pixel(self, pts: torch.Tensor) -> torch.Tensor:
        """
        Convert normalized device coordinates (NDC) in the range [-1, 1] to pixel coordinates.

        Args:
            pts (torch.Tensor): NDC coordinates of shape (N, 2).
        Returns:
            torch.Tensor: Pixel coordinates of shape (N, 2).
        """
        pixel_coords = (pts + 1) * torch.tensor([self.W, self.H], device=pts.device) / 2
        return pixel_coords

    def approx_length(self, num_samples: int = 1000) -> torch.Tensor:
        """Approximate curve length by sampling points and summing distances.

        Args:
            num_samples (int): Number of sample points to use for approximation.
        Returns:
            torch.Tensor: Approximate length of the curve.
        """
        samples = self._sample_points(factor=1, num_samples=num_samples)
        diffs = torch.diff(samples[:, :2], dim=0)
        length = torch.sum(torch.sqrt(torch.sum(diffs**2, dim=1)))
        return length

    def _downsample_boundary(self, xy: torch.Tensor, n: int) -> torch.Tensor:
        """Uniformly downsample boundary points to *n* samples.

        Uses linear interpolation so the result stays differentiable.

        Args:
            xy (torch.Tensor): Sampled curve points of shape (M, 2).
            n (int): Target number of boundary samples.

        Returns:
            torch.Tensor: Downsampled boundary of shape (n, 2).
        """
        M = xy.shape[0]
        if M <= n:
            return xy
        # Interpolation indices in [0, M-1]
        idx_float = torch.linspace(0, M - 1, n, device=xy.device)
        idx_lo = idx_float.long().clamp(max=M - 2)
        frac = (idx_float - idx_lo.float()).unsqueeze(1)  # (n, 1)
        return xy[idx_lo] * (1 - frac) + xy[idx_lo + 1] * frac

    def _compute_fill_gaussians(
        self, xy: torch.Tensor, factor: int = 1, use_downsample: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute interior fill gaussians for a closed curve.

        All geometric computations (bounding box, grid, SDF) are performed
        in **pixel** coordinates for better numerical precision.  The
        final gaussian positions are converted to NDC before returning.

        Steps:
            1. Bounding box from *xy* (pixel coords).
            2. Grid of query points at pixel resolution inside the bbox.
            3. Downsample *xy* to ``_fill_boundary_samples`` points.
            4. SDF-based interior mask.
            5. Convert positions to NDC and return gaussian parameters.

        Args:
            xy (torch.Tensor): Sampled curve positions in **pixel**
                coordinates, shape (N, 2).
            factor (int): Resolution scaling factor.

        Returns:
            Tuple of (positions_ndc, scales, rotations, opacities).
        """
        device = xy.device

        final_w = int(self.W * factor)
        final_h = int(self.H * factor)

        # Guard against empty boundary (degenerate curve)
        if xy.shape[0] == 0:
            empty_xy = torch.zeros((0, 2), device=device)
            empty_scales = torch.zeros((0, 2), device=device)
            empty_rot = torch.zeros((0, 1), device=device)
            empty_opacity = torch.zeros((0, 1), device=device)
            return empty_xy, empty_scales, empty_rot, empty_opacity

        # 1. Bounding box in pixel coords with padding
        bbox_min = xy.min(dim=0).values  # (2,)
        bbox_max = xy.max(dim=0).values  # (2,)

        bbox_extent = (bbox_max - bbox_min).detach()  # (2,)
        pad = bbox_extent.clamp(min=2.0) * 0.5  # at least 1 px padding
        bbox_min_padded = bbox_min - pad
        bbox_max_padded = bbox_max + pad
        # Clip to pixel canvas [0, W] x [0, H]
        bbox_min_padded = bbox_min_padded.clamp(min=0.0)
        bbox_max_padded[0] = bbox_max_padded[0].clamp(max=float(final_w))
        bbox_max_padded[1] = bbox_max_padded[1].clamp(max=float(final_h))

        # 2. Number of grid points: use _fill_grid_step to reduce density
        padded_w = (
            (bbox_max_padded[0] - bbox_min_padded[0]).detach().clamp(min=1).item()
        )
        padded_h = (
            (bbox_max_padded[1] - bbox_min_padded[1]).detach().clamp(min=1).item()
        )
        min_extent = min(padded_w, padded_h)

        # Limit step so grid dimensions stay >= 20 (empirically safe threshold)
        min_grid_dim = 20
        max_safe_step = max(1, int(min_extent / min_grid_dim))
        step = max(1, min(self._fill_grid_step, max_safe_step))

        n_x = max(min_grid_dim, int(padded_w / step))
        n_y = max(min_grid_dim, int(padded_h / step))

        # 3. Uniform grid in pixel coords inside padded bbox (coarse grid)
        gx = torch.linspace(
            bbox_min_padded[0].item(),
            bbox_max_padded[0].item(),
            n_x,
            device=device,
        )
        gy = torch.linspace(
            bbox_min_padded[1].item(),
            bbox_max_padded[1].item(),
            n_y,
            device=device,
        )
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")
        query = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)  # (Q, 2)

        # 4. Downsample boundary (still in pixel coords)
        if use_downsample:
            boundary = self._downsample_boundary(xy, self._fill_boundary_samples)
        else:
            boundary = xy

        # 5. Signed-distance-field (SDF) based interior mask.
        #    SDF is already in pixel units since both boundary and query
        #    are in pixel coordinates.
        sdf_px = self._signed_distance(boundary, query)  # (Q,) in pixels

        # Pixel-space steepness adjusted for grid step:
        #   For step=1: 5.0 → ~1 px gradient band
        #   For step>1: reduce steepness so boundary remains smooth
        sdf_steepness_px = 5.0 / step
        interior_mask = torch.sigmoid(
            sdf_steepness_px * sdf_px
        )  # ~1 inside, ~0 outside

        # 6. Gaussian scale for fill coverage (pixel units).
        #    Scale up by grid step to ensure coverage with fewer gaussians
        fill_sigma = (1.5 * step) / self._scale_ratio
        fill_scales = torch.full((query.shape[0], 2), fill_sigma, device=device)
        fill_rotations = torch.zeros((query.shape[0], 1), device=device)
        fill_opacity = interior_mask.unsqueeze(1)  # (Q, 1)

        # 7. Convert query positions from pixel coords to NDC.
        query_ndc = self.pixel_to_ndc(query)

        return query_ndc, fill_scales, fill_rotations, fill_opacity

    def winding_number(
        self, boundary: torch.Tensor, query: torch.Tensor, chunk_size: int = 8192
    ) -> torch.Tensor:
        """Compute winding number for query points around a CCW boundary.

        Uses chunking over query points to avoid OOM on large shapes.

        Args:
            boundary (torch.Tensor): Boundary points of shape (N, 2), CCW ordered.
            query (torch.Tensor): Query points of shape (Q, 2).
            chunk_size (int): Max query points per chunk to limit memory.

        Returns:
            torch.Tensor: Winding number of shape (Q,).
        """
        if boundary.ndim != 2 or boundary.shape[1] != 2:
            raise ValueError("boundary must have shape (N, 2)")
        if query.ndim != 2 or query.shape[1] != 2:
            raise ValueError("query must have shape (Q, 2)")

        Q = query.shape[0]
        device = query.device

        # If small enough, compute directly without chunking
        if Q <= chunk_size:
            p_i = boundary
            p_j = torch.roll(boundary, shifts=-1, dims=0)
            v_i = p_i.unsqueeze(0) - query.unsqueeze(1)
            v_j = p_j.unsqueeze(0) - query.unsqueeze(1)
            cross = v_i[..., 0] * v_j[..., 1] - v_i[..., 1] * v_j[..., 0]
            dot = (v_i * v_j).sum(dim=-1)
            angles = torch.atan2(cross, dot)
            return angles.sum(dim=1) / (2.0 * torch.pi)

        # Chunked computation for large queries
        p_i = boundary
        p_j = torch.roll(boundary, shifts=-1, dims=0)
        wn = torch.empty(Q, device=device)

        for qi in range(0, Q, chunk_size):
            q = query[qi : qi + chunk_size]
            v_i = p_i.unsqueeze(0) - q.unsqueeze(1)
            v_j = p_j.unsqueeze(0) - q.unsqueeze(1)
            cross = v_i[..., 0] * v_j[..., 1] - v_i[..., 1] * v_j[..., 0]
            dot = (v_i * v_j).sum(dim=-1)
            angles = torch.atan2(cross, dot)
            wn[qi : qi + chunk_size] = angles.sum(dim=1) / (2.0 * torch.pi)

        return wn

    def _signed_distance(
        self, boundary: torch.Tensor, query: torch.Tensor, chunk_size: int = 8192
    ) -> torch.Tensor:
        """Compute signed distance from query points to the boundary.

        Positive inside, negative outside.  The unsigned distance is
        the minimum distance from each query point to the nearest
        boundary *segment* (not just vertices).  The sign is determined
        by the winding number (detached — no gradient flows through it).

        Uses chunking over query points to avoid OOM on large shapes.

        Args:
            boundary (torch.Tensor): Boundary points (N, 2), forming a
                closed polyline (last → first edge is implicit).
            query (torch.Tensor): Query points (Q, 2).
            chunk_size (int): Max query points per chunk to limit memory.

        Returns:
            torch.Tensor: Signed distances of shape (Q,).
        """
        Q = query.shape[0]
        device = query.device

        # Precompute segment endpoints and vectors
        a = boundary  # (N, 2)
        b = torch.roll(boundary, shifts=-1, dims=0)  # (N, 2)
        ab = b - a  # (N, 2)
        ab_dot_ab = (ab * ab).sum(dim=-1)  # (N,)

        # Compute sign from winding number (detached, no gradient)
        with torch.no_grad():
            wn = self.winding_number(boundary, query, chunk_size=chunk_size)
            sign = 2.0 * (wn > 0.5).float() - 1.0  # +1 inside, -1 outside

        # Small queries: compute directly
        if Q <= chunk_size:
            aq = query.unsqueeze(1) - a.unsqueeze(0)  # (Q, N, 2)
            aq_dot_ab = (aq * ab.unsqueeze(0)).sum(dim=-1)  # (Q, N)
            t = (aq_dot_ab / ab_dot_ab.unsqueeze(0).clamp(min=1e-12)).clamp(0.0, 1.0)
            closest = a.unsqueeze(0) + t.unsqueeze(-1) * ab.unsqueeze(0)
            diff = query.unsqueeze(1) - closest
            dist_sq = (diff * diff).sum(dim=-1)
            min_dist_sq, _ = dist_sq.min(dim=1)
            unsigned_dist = torch.sqrt(min_dist_sq + 1e-12)
            return sign * unsigned_dist

        # Chunked computation for large queries
        unsigned_dist = torch.empty(Q, device=device)

        for qi in range(0, Q, chunk_size):
            q = query[qi : qi + chunk_size]  # (C, 2)
            aq = q.unsqueeze(1) - a.unsqueeze(0)  # (C, N, 2)
            aq_dot_ab = (aq * ab.unsqueeze(0)).sum(dim=-1)  # (C, N)
            t = (aq_dot_ab / ab_dot_ab.unsqueeze(0).clamp(min=1e-12)).clamp(0.0, 1.0)
            closest = a.unsqueeze(0) + t.unsqueeze(-1) * ab.unsqueeze(0)  # (C, N, 2)
            diff = q.unsqueeze(1) - closest
            dist_sq = (diff * diff).sum(dim=-1)
            chunk_min, _ = dist_sq.min(dim=1)
            unsigned_dist[qi : qi + chunk_size] = torch.sqrt(chunk_min + 1e-12)

        return sign * unsigned_dist

    def _expand_weights(self, weights: torch.Tensor) -> torch.Tensor:
        """Expand weights to match the number of control points.

        For closed curves, wraps boundary weights. For open curves,
        weights are used directly (one per key point / control point).

        Args:
            weights (torch.Tensor): Weights tensor.

        Returns:
            torch.Tensor: Expanded weights tensor.
        """
        p = self._order - 1
        if self._closed:
            half_offset = p // 2
            offset_ramainder = p - half_offset
            weights = torch.cat(
                [weights[-(offset_ramainder):], weights, weights[:half_offset]], dim=0
            )
            return weights

        # Open curves: key points = control points, no expansion needed
        return weights

    def _effective_weights(self, weights: torch.Tensor) -> torch.Tensor:
        """Stabilize weights by enforcing positivity and normalizing scale.

        This prevents extreme weights from causing self-crossing artifacts.
        """
        w = torch.nn.functional.softplus(weights)
        mean = w.mean().clamp_min(1e-6)
        return w / mean

    def _compute_ctrls(self) -> torch.Tensor:
        """Compute control points given key-points and order of NURBS.

        For closed curves, wraps boundary points. For open curves,
        key points are used as control points directly.

        Returns:
            torch.Tensor: control points of shape (n_cp, 3)
        """
        kp = self._key_points
        sw = self._stroke_width
        k = self._order
        closed = self._closed
        p = k - 1
        if closed:
            half_offset = p // 2
            offset_ramainder = p - half_offset
            P = torch.vstack([kp[-(offset_ramainder):, :], kp, kp[:half_offset, :]])
            S = torch.cat([sw[-(offset_ramainder):], sw, sw[:half_offset]], dim=0)
        else:
            # Open curves: use key points directly as control points
            P = kp
            S = sw
        return torch.cat([P, S.unsqueeze(1)], dim=1)

    def _init_knot_interval(self, knot_vec: torch.Tensor | None = None) -> torch.Tensor:
        """Initialize knot intervals from a knot vector or with ones.

        Args:
            knot_vec (torch.Tensor, optional): User-specified knot vector.
                If provided, intervals are computed as differences.
                If None, intervals are initialized to ones.

        Returns:
            torch.Tensor: Knot intervals of shape (m-1,) where m is the number of knots.
        """
        k = self._order
        p = k - 1
        n_cp = self._control_points.shape[0]
        m = n_cp + k  # Total number of knots
        device = self._key_points.device

        if knot_vec is not None:
            # Compute intervals from provided knot vector
            knot_vec = knot_vec.to(device)
            if knot_vec.shape[0] != m:
                raise ValueError(
                    f"knot_vec must have {m} elements, got {knot_vec.shape[0]}"
                )
            all_intervals = knot_vec[1:] - knot_vec[:-1]
            if (all_intervals < 0).any():
                raise ValueError(
                    "knot_vec must be non-decreasing (all intervals must be non-negative)"
                )
            if not self._closed:
                # For open curves, store only the interior intervals
                # (the boundary zero-padding is applied in _generate_knots).
                intervals = all_intervals[p:-p] if p > 0 else all_intervals
            else:
                # For closed curves, extract the n_kp core intervals;
                # periodicity is enforced structurally in _generate_knots.
                n_kp = self._key_points.shape[0]
                intervals = all_intervals[p : p + n_kp] if p > 0 else all_intervals
        elif not self._closed:
            # Open curves: clamped knot vector.
            # Boundary zero-intervals (for knot multiplicity p+1) are added
            # structurally in _generate_knots and are NOT part of the
            # optimizable _knot_interval.  Only the interior intervals are
            # stored here so the optimizer cannot corrupt the clamping.
            n_interior = n_cp - k + 1  # = num_spans
            intervals = torch.ones(n_interior, device=device)
        else:
            # Closed curves: store only n_kp core intervals;
            # periodicity is enforced structurally in _generate_knots.
            n_kp = self._key_points.shape[0]
            intervals = torch.ones(n_kp, device=device)

        return intervals

    def _generate_knots(self) -> torch.Tensor:
        """Compute knots from knot intervals.

        For open curves the stored ``_knot_interval`` contains only the
        interior intervals; this method pads with ``p`` zeros on each side
        to form the full clamped knot vector.  For closed curves the stored
        ``_knot_interval`` contains only the ``n_kp`` core intervals;
        periodicity is enforced by prepending the last ``p`` and appending
        the first ``p`` core intervals.

        Returns:
            torch.Tensor: Knot vector of shape (m,) where m = n_cp + k.
        """
        p = self._order - 1
        device = self._knot_interval.device

        if not self._closed:
            # Open curve: _knot_interval stores only interior intervals.
            # Pad with structural zeros for clamped multiplicity.
            zeros = torch.zeros(p, device=device, dtype=self._knot_interval.dtype)
            full_intervals = torch.cat([zeros, self._knot_interval, zeros])
        else:
            # Closed curve: _knot_interval stores the n_kp core intervals.
            # Enforce periodicity by wrapping: prepend the last p and
            # append the first p intervals from the core.
            core = self._knot_interval
            if p > 0:
                full_intervals = torch.cat([core[-p:], core, core[:p]])
            else:
                full_intervals = core

        # First knot is -p
        first_knot = torch.tensor([-p], device=device, dtype=self._knot_interval.dtype)

        # Subsequent knots are cumulative sum of intervals starting from -p
        knots = torch.cat(
            [first_knot, first_knot + torch.cumsum(full_intervals, dim=0)]
        )

        return knots

    def _evaluate_basis(self, u: torch.Tensor) -> torch.Tensor:
        """
        Evaluate basis functions N_{i,p}(u) using Cox-De Boor recursion.

        Args:
            u (torch.Tensor): Sample locations.

        Returns:
            torch.Tensor: Basis functions of shape (M, N_cp)
        """
        knots = self._knot_vector
        device = u.device
        k = self._order
        p = k - 1

        # Number of control points - use actual count from control points tensor
        n_cp = self._control_points.shape[0]

        # Level 0: N_{i,0}
        # Vectorized Level 0
        knots_starts = knots[:-1].unsqueeze(0)  # (1, num_intervals)
        knots_ends = knots[1:].unsqueeze(0)

        # N_{i,0}(u) = 1 if t_i <= u < t_{i+1}
        N = (u.unsqueeze(1) >= knots_starts) & (u.unsqueeze(1) < knots_ends)

        # Handle the last point (u_max) which is excluded by strict inequality
        u_max = knots[-k]
        is_last = u == u_max
        if is_last.any():
            N[is_last, :] = 0
            N[is_last, n_cp - 1] = 1

        N = N.float()

        # Recursion
        for d in range(1, p + 1):
            current_count = N.shape[1]
            next_count = current_count - 1
            idx_i = torch.arange(next_count, device=device)

            # Term 1
            t_i = knots[idx_i]
            t_id = knots[idx_i + d]
            numer1 = u.unsqueeze(1) - t_i.unsqueeze(0)
            denom1 = (t_id - t_i).unsqueeze(0)
            term1 = (numer1 / (denom1 + 1e-8)) * N[:, idx_i]
            term1 = torch.where(denom1 == 0, torch.zeros_like(term1), term1)

            # Term 2
            t_id1 = knots[idx_i + d + 1]
            t_i1 = knots[idx_i + 1]
            numer2 = t_id1.unsqueeze(0) - u.unsqueeze(1)
            denom2 = (t_id1 - t_i1).unsqueeze(0)
            term2 = (numer2 / (denom2 + 1e-8)) * N[:, idx_i + 1]
            term2 = torch.where(denom2 == 0, torch.zeros_like(term2), term2)

            N = term1 + term2

        return N

    # For correctness comparison
    def _sample_points_legacy(self, factor=1) -> torch.Tensor:
        """Sample points along the NURBS splat.
        Returns:
            torch.Tensor: Sampled points of shape `(NS, 3)` where NS is the total number of sampled points.
        """
        # Recompute control points to ensure gradients flow back to key_points
        # k = self._order
        # first_point = self._key_points[0].unsqueeze(0).repeat(k - 1, 1)
        # last_point = self._key_points[-1].unsqueeze(0).repeat(k - 1, 1)
        # control_points = torch.cat([first_point, self._key_points, last_point], dim=0)
        cp = self._control_points
        k = self._order
        # Stabilize and expand weights
        # weights = self._effective_weights(self._weights)
        weights = self._expand_weights(self._weights)

        knots = self._knot_vector
        device = cp.device

        # Valid domain
        u_min = knots[k - 1]
        u_max = knots[-k]

        # Determine sampling parameters
        num_spans = self._num_spans
        total_samples = int(self._density * num_spans * factor)
        u = torch.linspace(u_min, u_max, total_samples, device=device)

        # Evaluate Basis Functions
        N = self._evaluate_basis(u)

        # Rational Evaluation
        # N is (M, n_cp)
        # Weights is (n_cp,)
        # Control Points is (n_cp, 3)

        denom = torch.matmul(N, weights.unsqueeze(1))  # (M, 1)
        weighted_points = cp * weights.unsqueeze(1)  # (n_cp, 3)
        numer = torch.matmul(N, weighted_points)  # (M, 3)

        curve_points = numer / (denom + 1e-8)

        return curve_points

    @overload
    def _sample_points(
        self,
        factor: int,
        compute_derivative: Literal[True],
        coords_system: Literal["ndc", "pixel"] = "pixel",
        num_samples: int = 1000,
        user_supplied_u: Optional[torch.Tensor] = None,
        max_deriv_order: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...

    @overload
    def _sample_points(
        self,
        factor: int = 1,
        compute_derivative: Literal[False] = False,
        coords_system: Literal["ndc", "pixel"] = "pixel",
        num_samples: int = 1000,
        user_supplied_u: Optional[torch.Tensor] = None,
        max_deriv_order: int = 0,
    ) -> torch.Tensor: ...

    def _sample_points(
        self,
        factor: int = 1,
        compute_derivative: bool = False,
        coords_system: Literal["ndc", "pixel"] = "pixel",
        num_samples: int = 1000,
        user_supplied_u: Optional[torch.Tensor] = None,
        max_deriv_order: int = 0,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Span-local NURBS sampling.

        This version exploits the local support property of B-splines,
        evaluating only the p+1 non-zero basis functions per sample point.

        Args:
            factor (int, optional): Scaling factor for sampling density. Defaults to 1.
            compute_derivative (bool): If True, also return the first
                derivative dC/du of shape (M, 3).  Kept for backward compat;
                equivalent to ``max_deriv_order=1``.
            coords_system (Literal["ndc", "pixel"]): Coordinate system for output points.
            num_samples (int): Total number of sample points to generate along the curve.
                Invalid if *user_supplied_u* is provided.
            user_supplied_u (torch.Tensor, optional): If provided, use these parameter values
                directly instead of generating them uniformly. Should be of shape (M,).
            max_deriv_order (int): Maximum derivative order to return.
                When > 0, overrides *compute_derivative*.  The return value
                becomes ``(curve_points, C'(u), C''(u), ..., C^(max)(u))``.

        Returns:
            torch.Tensor: Sampled points of shape (M, 3).
            If *compute_derivative* is True, returns ``(curve_points, curve_deriv)``.
            If *max_deriv_order* > 0, returns
                ``(curve_points, deriv_1, ..., deriv_max_deriv_order)``.
        """
        cp = self._control_points  # (n_cp, 3)
        weights = self._expand_weights(self._weights)  # (n_cp,)
        knots = self._knot_vector
        k = self._order
        p = k - 1
        device = cp.device

        # Resolve the effective derivative order
        eff_deriv = max(max_deriv_order, 1 if compute_derivative else 0)

        # valid parameter domain
        u_min = knots[k - 1]
        u_max = knots[-k]

        num_spans = self._num_spans
        total_samples = num_samples

        if user_supplied_u is not None:
            u = user_supplied_u.to(device)
        else:
            u = torch.linspace(u_min, u_max, total_samples, device=device)

        # find knot span for each u
        # span: index i s.t. u in [t_i, t_{i+1})
        span = torch.searchsorted(knots, u, right=True) - 1
        n_cp = cp.shape[0]
        span = span.clamp(min=p, max=n_cp - 1)  # (M,)

        basis_result = self._evaluate_basis_local(u, span, max_deriv_order=eff_deriv)
        if eff_deriv >= 1:
            N = basis_result[0]  # (M, p+1)
            basis_derivs = basis_result[1:]  # tuple of dN_1 .. dN_eff_deriv
        else:
            N = basis_result

        # indices: (M, p+1)
        idx = span.unsqueeze(1) - torch.arange(p, -1, -1, device=device)

        cp_local = cp[idx]  # (M, p+1, 3)
        w_local = weights[idx]  # (M, p+1)

        # rational evaluation: C(u) = A(u) / W(u)
        wN = N * w_local  # (M, p+1)

        denom = wN.sum(dim=1, keepdim=True)  # W(u)  (M, 1) # type: ignore
        numer = (wN.unsqueeze(-1) * cp_local).sum(dim=1)  # A(u)  (M, 3) # type: ignore

        curve_points = numer / (denom + 1e-8)

        if eff_deriv >= 1:
            # ---------------------------------------------------------
            # Rational curve derivatives via the Leibniz product rule.
            #   W * C = A  ⟹  C^(k) = (A^(k) - Σ_{j=1..k} C(k,j) W^(j) C^(k-j)) / W
            # ---------------------------------------------------------

            # Pre-compute A^(m) and W^(m) for m = 1 .. eff_deriv
            A_derivs: list[torch.Tensor] = []
            W_derivs: list[torch.Tensor] = []
            for m in range(eff_deriv):
                dN_m = basis_derivs[m]  # (m+1)-th derivative of basis (M, p+1)
                dwN_m = dN_m * w_local  # (M, p+1)
                A_derivs.append(
                    (dwN_m.unsqueeze(-1) * cp_local).sum(dim=1)
                )  # A^(m+1) (M, 3)
                W_derivs.append(dwN_m.sum(dim=1, keepdim=True))  # W^(m+1) (M, 1)

            # Build C^(1), C^(2), ..., C^(eff_deriv) iteratively
            C_list: list[torch.Tensor] = [curve_points]  # C^(0)
            curve_derivs: list[torch.Tensor] = []
            for m in range(1, eff_deriv + 1):
                A_m = A_derivs[m - 1]  # A^(m)
                correction = torch.zeros_like(A_m)
                for j in range(1, m + 1):
                    correction = correction + (
                        math.comb(m, j) * W_derivs[j - 1] * C_list[m - j]
                    )
                C_m = (A_m - correction) / (denom + 1e-8)
                C_list.append(C_m)
                curve_derivs.append(C_m)

            # Coordinate system transformation (linear → same scale on all orders)
            if coords_system == "ndc":
                xy_ndc = self.pixel_to_ndc(curve_points[:, :2])
                curve_points = torch.cat([xy_ndc, curve_points[:, 2:]], dim=1)
                scale_factor = torch.tensor([2.0 / self.W, 2.0 / self.H], device=device)
                for i in range(len(curve_derivs)):
                    curve_derivs[i] = torch.cat(
                        [curve_derivs[i][:, :2] * scale_factor, curve_derivs[i][:, 2:]],
                        dim=1,
                    )

            if max_deriv_order > 0:
                return (curve_points,) + tuple(curve_derivs)
            else:
                # Backward compatible: compute_derivative=True → (points, d1)
                return curve_points, curve_derivs[0]

        if coords_system == "ndc":
            xy_ndc = self.pixel_to_ndc(curve_points[:, :2])
            curve_points = torch.cat([xy_ndc, curve_points[:, 2:]], dim=1)

        return curve_points

    def _evaluate_basis_local(
        self,
        u: torch.Tensor,
        span: torch.Tensor,
        compute_derivative: bool = False,
        max_deriv_order: int = 0,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Evaluate local B-spline basis functions N_{i-p,...,i}(u).

        This implements the standard triangular table algorithm for B-spline
        basis function evaluation, computing only the p+1 non-zero basis
        functions at each parameter value.

        Uses out-of-place operations to preserve autograd graph.

        Supports higher-order derivatives via the recursive formula:
            dN_{i,d}^{(k)} = d * (dN_{i,d-1}^{(k-1)} / (t_{i+d}-t_i)
                                 - dN_{i+1,d-1}^{(k-1)} / (t_{i+d+1}-t_{i+1}))

        Args:
            u (torch.Tensor): Parameter values of shape (M,).
            span (torch.Tensor): Knot span indices of shape (M,).
            compute_derivative (bool): If True, also return the 1st basis
                derivative of shape (M, p+1).  Equivalent to
                ``max_deriv_order=1``; kept for backward compatibility.
            max_deriv_order (int): Maximum derivative order to compute (0, 1, 2
                or 3).  Overrides *compute_derivative* when > 0.  Requires
                spline degree >= *max_deriv_order*.

        Returns:
            torch.Tensor: Basis function values of shape (M, p+1), where
                N[:, j] corresponds to N_{i-p+j, p}(u).
            If derivatives are requested, returns a tuple
                ``(N, dN_1, dN_2, ..., dN_max_deriv_order)``.
            For the legacy ``compute_derivative=True`` path (when
                *max_deriv_order* is 0) returns ``(N, dN)``.
        """
        knots = self._knot_vector
        device = u.device
        k = self._order
        p = k - 1
        M = u.shape[0]

        # Resolve the effective derivative order
        eff_max_deriv = max(max_deriv_order, 1 if compute_derivative else 0)
        eff_max_deriv = min(eff_max_deriv, p)  # cannot exceed spline degree

        # Initialize N as a list of columns to avoid in-place operations
        # N[j] corresponds to N_{i-p+j, d}(u) at current degree d
        N_cols = [torch.ones(M, device=device)]  # N_{i,0} = 1
        for _ in range(p):
            N_cols.append(torch.zeros(M, device=device))

        # Precompute left and right differences for all degrees
        # left[d] = u - knots[span + 1 - d]
        # right[d] = knots[span + d] - u
        left_list = [torch.zeros(M, device=device)]  # placeholder for d=0
        right_list = [torch.zeros(M, device=device)]  # placeholder for d=0
        for d in range(1, p + 1):
            left_list.append(u - knots[span + 1 - d])
            right_list.append(knots[span + d] - u)

        # Dictionary to store intermediate degree basis values for derivative
        # computation.  Keys are integers (degree), values are (M, deg+1) tensors.
        saved_bases: dict[int, torch.Tensor] = {}

        for d in range(1, p + 1):
            # Save degree (d-1) basis if needed for derivative computation.
            # We need degrees p-eff_max_deriv .. p-1.
            if eff_max_deriv > 0 and (p - eff_max_deriv) <= (d - 1) < p:
                saved_bases[d - 1] = torch.stack(N_cols[:d], dim=1)

            saved = torch.zeros(M, device=device)
            new_cols = []

            for j in range(d):
                denom = right_list[j + 1] + left_list[d - j]
                # B-spline convention: 0/0 = 0.  Use torch.where to avoid
                # NaN gradients when the denominator is zero (clamped knots).
                safe_denom = torch.where(
                    denom.abs() < 1e-10,
                    torch.ones_like(denom),
                    denom,
                )
                zero_mask = denom.abs() < 1e-10
                temp = torch.where(
                    zero_mask, torch.zeros_like(N_cols[j]), N_cols[j] / safe_denom
                )
                new_val = saved + right_list[j + 1] * temp
                new_cols.append(new_val)
                saved = left_list[d - j] * temp

            new_cols.append(saved)

            # Pad remaining columns with zeros if needed
            for j in range(d + 1, p + 1):
                new_cols.append(torch.zeros(M, device=device))

            N_cols = new_cols

        # Stack columns into (M, p+1) tensor
        N = torch.stack(N_cols, dim=1)

        if eff_max_deriv > 0:
            # ----------------------------------------------------------
            # Compute basis derivatives up to order eff_max_deriv via
            # the recursive B-spline derivative formula.
            # ----------------------------------------------------------
            def _deriv_step(vals_dm1: torch.Tensor, deg_d: int) -> torch.Tensor:
                """Apply one derivative recursion step.

                Given values at degree ``deg_d - 1`` with shape ``(M, deg_d)``,
                returns the derivative-step result at degree ``deg_d`` with
                shape ``(M, deg_d + 1)``.

                Uses the formula:
                    result[j] = deg_d * (vals[j-1]/left_denom[j]
                                        - vals[j]/right_denom[j])
                """
                zeros_pad = torch.zeros(M, 1, device=device)
                padded = torch.cat(
                    [zeros_pad, vals_dm1, zeros_pad], dim=1
                )  # (M, deg_d + 2)

                j_idx = torch.arange(deg_d + 1, device=device)

                l_hi = span.unsqueeze(1) + j_idx.unsqueeze(0)
                l_lo = span.unsqueeze(1) - deg_d + j_idx.unsqueeze(0)
                r_hi = l_hi + 1
                r_lo = l_lo + 1

                l_denom = knots[l_hi] - knots[l_lo]
                r_denom = knots[r_hi] - knots[r_lo]

                l_val = padded[:, : deg_d + 1]
                r_val = padded[:, 1 : deg_d + 2]

                l_zero = l_denom.abs() < 1e-10
                r_zero = r_denom.abs() < 1e-10
                safe_l = torch.where(l_zero, torch.ones_like(l_denom), l_denom)
                safe_r = torch.where(r_zero, torch.ones_like(r_denom), r_denom)

                t1 = torch.where(l_zero, torch.zeros_like(l_val), l_val / safe_l)
                t2 = torch.where(r_zero, torch.zeros_like(r_val), r_val / safe_r)

                return float(deg_d) * (t1 - t2)

            # Derivative table: dt[(degree, order)] = (M, degree+1) tensor.
            # Seed with the saved 0th-derivative (basis value) tables.
            dt: dict[tuple[int, int], torch.Tensor] = {}
            for d_val, basis in saved_bases.items():
                dt[(d_val, 0)] = basis

            result_derivs: list[torch.Tensor] = []
            for kk in range(1, eff_max_deriv + 1):
                for dd in range(p - eff_max_deriv + kk, p + 1):
                    dt[(dd, kk)] = _deriv_step(dt[(dd - 1, kk - 1)], dd)
                result_derivs.append(dt[(p, kk)])

            if max_deriv_order > 0:
                return (N,) + tuple(result_derivs)
            else:
                # Backward compatible: compute_derivative=True → (N, dN)
                return N, result_derivs[0]

        return N

    def _compute_rotations(
        self,
        points: torch.Tensor,
        derivatives: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the rotation angle at each sampled point along each curve.

        Args:
            points (torch.Tensor): Tensor of shape (N, 2) representing
                sampled points along the NURBS curve.
            derivatives (torch.Tensor | None): Analytic tangent vectors of
                shape (N, 2).  When provided, rotations are computed from
                these directly instead of finite-difference approximation.

        Returns:
            torch.Tensor: Tensor of shape (N,) representing rotation angles
                in radians at each sampled point.
        """
        if derivatives is not None:
            # Analytic path: use tangent dC/du directly
            # Points and derivatives are already in pixel coords.
            diffs = derivatives.clone()
            theta = torch.atan2(-diffs[:, 1], diffs[:, 0])  # (N,)
            return theta

        # Fallback: finite-difference approximation
        # Points are in pixel coords, so diffs are directly in pixels.
        diffs = points[2:] - points[:-2]  # (N-2, 2)

        theta = torch.atan2(-diffs[:, 1], diffs[:, 0])  # (N-2,)

        theta_first = theta[:1]
        theta_last = theta[-1:]
        rotations = torch.cat([theta_first, theta, theta_last], dim=0)  # (N,)
        return rotations

    def _compute_scale(
        self, points: torch.Tensor, widths: torch.Tensor
    ) -> torch.Tensor:
        """Compute the scale (radii) at each sampled point along each curve

        Args:
            points (torch.Tensor): Tensor of shape (N, 2) representing sampled points along the NURBS curve.
        Returns:
            torch.Tensor: Tensor of shape (N, 2) representing scale at each sampled point.
        """
        pts = points.detach()
        # pts = points
        n = pts.shape[0]

        rho_x = self._scale_ratio
        sigma_x = widths / rho_x
        sigma_y = widths / rho_x

        scaling = torch.stack([sigma_x, sigma_y], dim=1)  # (N, 2)
        return scaling

    def export_to_json(self, path: Optional[str] = None) -> dict:
        """Export this NURBSSplat to a JSON-serializable dictionary.

        Args:
            path (str, optional): If provided, write JSON to this file path.

        Returns:
            dict: A dictionary containing all parameters needed to reconstruct
                this NURBSSplat.
        """
        # Combine _key_points (pixel) and _stroke_width into (M, 3)
        kp = self._key_points.detach().cpu()
        sw = self._stroke_width.detach().cpu()
        key_points = torch.cat([kp, sw.unsqueeze(1)], dim=1)  # (M, 3)

        data = {
            "type": "NURBSSplat",
            "key_points": key_points.tolist(),
            "weights": self._weights.detach().cpu().tolist(),
            "order": self._order,
            "width": self.W,
            "height": self.H,
            "density": self._density,
            "color": self._color.detach().cpu().tolist(),
            "closed": self._closed,
            "filled": self._filled,
            "fill_boundary_samples": self._fill_boundary_samples,
            "fill_grid_step": self._fill_grid_step,
            "scale_ratio": self._scale_ratio,
            "opacity": self._opacity.detach().cpu().item(),
            "knot_interval": self._knot_interval.detach().cpu().tolist(),
        }

        if path is not None:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)

        return data

    @classmethod
    def load_from_json(
        cls, path_or_dict: Union[str, dict], device: str = "cpu"
    ) -> "NURBSSplat":
        """Create a NURBSSplat from a JSON file or dictionary.

        Args:
            path_or_dict (str | dict): Path to a JSON file, or a dictionary
                previously returned by :meth:`export_to_json`.
            device (str): Device to place tensors on.

        Returns:
            NURBSSplat: The reconstructed NURBS splat.
        """
        if isinstance(path_or_dict, str):
            with open(path_or_dict, "r") as f:
                data = json.load(f)
        else:
            data = path_or_dict

        key_points = torch.tensor(
            data["key_points"], dtype=torch.float32, device=device
        )
        weights = torch.tensor(data["weights"], dtype=torch.float32, device=device)
        color = torch.tensor(data["color"], dtype=torch.float32, device=device)

        nurbs = cls(
            key_points=key_points,
            weights=weights,
            order=data["order"],
            width=data["width"],
            height=data["height"],
            density=data["density"],
            color=color,
            closed=data["closed"],
            filled=data["filled"],
            fill_boundary_samples=data["fill_boundary_samples"],
            fill_grid_step=data["fill_grid_step"],
            scale_ratio=data["scale_ratio"],
            opacity=data["opacity"],
            init_with_ndc=False,
        )

        # Restore knot intervals directly to preserve exact values
        knot_interval = torch.tensor(
            data["knot_interval"], dtype=torch.float32, device=device
        )
        if isinstance(nurbs._knot_interval, nn.Parameter):
            nurbs._knot_interval = nn.Parameter(knot_interval)
        else:
            nurbs._knot_interval = knot_interval
        nurbs._knot_vector = nurbs._generate_knots()
        nurbs.sample_points = nurbs._sample_points(compute_derivative=False)

        return nurbs


def main():
    # DEBUG
    import matplotlib.pyplot as plt

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Define picture space
    W, H = 512, 512

    key_points = torch.tensor(
        [
            [0, -0.5, 1],
            [0.5, -0.25, 1],
            [0.5, 0.25, 1],
            [0, 0.5, 1],
            [-0.5, 0.25, 1],
            [-0.5, -0.25, 1],
        ],
        device=device,
    )
    knotvec = torch.tensor(
        [0, 1, 2, 3, 4, 4, 4, 4, 5, 5, 10], device=device, dtype=torch.float32
    )

    weights = torch.ones(key_points.shape[0], device=device)
    order = 3

    nurbs_splat = NURBSSplat(
        key_points,
        weights,
        order,
        width=W,
        height=H,
        color=torch.tensor([0.45, 0.35, 0.7], device=device),
        init_with_ndc=True,
        density=15,
        closed=True,
        filled=True,
        knot_vec=knotvec,
    )

    # Call splat() function to get the image
    out_img, pts = nurbs_splat.splat(max_deriv_order=1)

    # (1) save jpg
    img_np = (out_img.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    plt.imsave("output.jpg", img_np)
    print(pts.shape)

    # (2) plot with matplotlib
    # sample_points are already in pixel coords
    pts_pixel = pts[:, :2]
    pts_np = pts_pixel.detach().cpu().numpy()

    plt.figure()
    plt.imshow(img_np)
    plt.scatter(pts_np[:, 0], pts_np[:, 1], c="r", s=1)
    kp = key_points.cpu()
    kp = nurbs_splat.ndc_to_pixel(kp[:, :2]).cpu().numpy()
    plt.scatter(kp[:, 0], kp[:, 1], c="b", s=10)
    for i, (xi, yi) in enumerate(zip(kp[:, 0], kp[:, 1])):
        plt.text(xi, yi, str(i), fontsize=4, color="white")
    plt.savefig("output_with_points.jpg", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
