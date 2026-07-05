"""
NURBSScene: A scene class for differentiable rendering of multiple NURBS curves.
Author: Jingye Qiu
"""

import json
import torch
import torch.nn as nn
import numpy as np
from collections import deque
from typing import List, Tuple, Optional, Dict, Iterable, Union
from gsplat.project_gaussians_2d_scale_rot import project_gaussians_2d_scale_rot
from gsplat.rasterize import rasterize_gaussians

from nsplat.nurbs import NURBSSplat


class NURBSScene(nn.Module):
    """A scene containing multiple NURBS curves for differentiable rendering.

    This class manages a collection of NURBSSplat objects and provides
    unified APIs for rendering, optimization, and manipulation.

    Attributes:
        width (float): Width of the rendering canvas in pixels.
        height (float): Height of the rendering canvas in pixels.
        nurbs_list (List[NURBSSplat]): List of NURBSSplat objects in the scene.
        background (torch.Tensor): Background color for rendering.
    """

    nurbs_list: deque[NURBSSplat]
    background: torch.Tensor

    def __init__(
        self,
        width: float = 256.0,
        height: float = 256.0,
        background: Optional[torch.Tensor] = None,
    ):
        """Initialize a NURBSScene.

        Args:
            width (float): Width of the rendering canvas in pixels.
            height (float): Height of the rendering canvas in pixels.
            background (torch.Tensor, optional): Background color (RGB). Defaults to white.
        """
        super(NURBSScene, self).__init__()
        self.W = width
        self.H = height
        self.nurbs_list: deque[NURBSSplat] = deque()
        self.BLOCK_W = 16
        self.BLOCK_H = 16

        if background is None:
            self.register_buffer("background", torch.ones(3))
        else:
            self.register_buffer("background", background)

    def add_nurbs(self, nurbs: NURBSSplat) -> int:
        """Add a single NURBS splat to the front of the scene.

        Args:
            nurbs (NURBSSplat): The NURBS splat to add.

        Returns:
            int: Index of the added NURBS in the scene (always 0).
        """
        # Ensure the NURBS has matching dimensions
        if nurbs.W != self.W or nurbs.H != self.H:
            raise ValueError(
                f"NURBS dimensions ({nurbs.W}, {nurbs.H}) do not match "
                f"scene dimensions ({self.W}, {self.H})"
            )
        self.nurbs_list.appendleft(nurbs)
        return 0

    def add_nurbs_list(self, nurbs_list: List[NURBSSplat]) -> List[int]:
        """Add multiple NURBS splats to the front of the scene.

        The relative order of the input list is preserved, i.e. the first
        element ends up at index 0.

        Args:
            nurbs_list (List[NURBSSplat]): List of NURBS splats to add.

        Returns:
            List[int]: Indices of the added NURBS in the scene.
        """
        # Prepend in reverse so the original order is preserved at the front
        for nurbs in reversed(nurbs_list):
            self.add_nurbs(nurbs)
        return list(range(len(nurbs_list)))

    def remove_nurbs(self, index: int) -> NURBSSplat:
        """Remove a NURBS splat from the scene by index.

        Args:
            index (int): Index of the NURBS to remove.

        Returns:
            NURBSSplat: The removed NURBS splat.
        """
        if index < 0 or index >= len(self.nurbs_list):
            raise IndexError(
                f"NURBS index {index} out of range [0, {len(self.nurbs_list)})"
            )
        nurbs = self.nurbs_list[index]
        del self.nurbs_list[index]
        return nurbs

    def get_nurbs(self, index: int) -> NURBSSplat:
        """Get a NURBS splat by index.

        Args:
            index (int): Index of the NURBS to retrieve.

        Returns:
            NURBSSplat: The NURBS splat at the given index.
        """
        return self.nurbs_list[index]

    def __len__(self) -> int:
        """Return the number of NURBS in the scene."""
        return len(self.nurbs_list)

    def __iter__(self):
        """Iterate over NURBS splats in the scene."""
        return iter(self.nurbs_list)

    def set_fill_grid_step(
        self,
        step: Union[float, List[float]],
    ) -> None:
        """Set the fill grid step for splines in the scene.

        Args:
            step (float | List[float]): A single value applied to every spline,
                or a per-spline list whose length must match the number of
                splines in the scene.

        Raises:
            ValueError: If a list is provided with the wrong length.
        """
        if isinstance(step, (list, tuple)):
            if len(step) != len(self.nurbs_list):
                raise ValueError(
                    f"Length of step list ({len(step)}) must match the number "
                    f"of splines ({len(self.nurbs_list)})"
                )
            for i, (nurbs, s) in enumerate(zip(self.nurbs_list, step)):
                nurbs.fill_grid_step = s
        else:
            for nurbs in self.nurbs_list:
                nurbs.fill_grid_step = step

    def set_density(
        self,
        density: Union[float, List[float]],
    ) -> None:
        """Set the sampling density for splines in the scene.

        Args:
            density (float | List[float]): A single value applied to every spline,
                or a per-spline list whose length must match the number of
                splines in the scene.

        Raises:
            ValueError: If a list is provided with the wrong length.
        """
        if isinstance(density, (list, tuple)):
            if len(density) != len(self.nurbs_list):
                raise ValueError(
                    f"Length of density list ({len(density)}) must match the number "
                    f"of splines ({len(self.nurbs_list)})"
                )
            for nurbs, d in zip(self.nurbs_list, density):
                nurbs.density = d
        else:
            for nurbs in self.nurbs_list:
                nurbs.density = density

    def get_params(
        self,
        optimize: Union[Iterable[str], str] = (
            "key_points",
            "stroke_width",
            "weights",
            "knot_interval",
        ),
        lrs: Optional[Dict[str, float]] = None,
        default_lr: float = 0.01,
        per_nurbs_lrs: Optional[List[Dict[str, float]]] = None,
    ) -> List[Dict]:
        """Return optimizer parameter groups for all NURBS in the scene.

        This method collects parameters from all NURBS splats and returns them
        in a format suitable for PyTorch optimizers.

        Args:
            optimize (Iterable[str] | str): Parameters to optimize. Supported values:
                "key_points", "stroke_width", "weights", "knot_interval", "color", "opacity".
            lrs (dict | None): Global per-parameter learning rates, e.g.
                "key_points": 0.001, "stroke_width": 0.005, "weights": 0.02, "knot_interval": 0.001, "color": 0.01, "opacity": 0.01}.
            default_lr (float): Default learning rate if not specified in `lrs`.
            per_nurbs_lrs (List[Dict[str, float]] | None): Optional per-NURBS learning
                rates. If provided, must have the same length as the number of NURBS
                in the scene.

        Returns:
            list: List of parameter group dicts in the form
                [{"params": tensor, "lr": float}, ...].
        """
        if len(self.nurbs_list) == 0:
            return []

        all_param_groups = []

        for i, nurbs in enumerate(self.nurbs_list):
            # Determine learning rates for this NURBS
            if per_nurbs_lrs is not None and i < len(per_nurbs_lrs):
                nurbs_lrs = per_nurbs_lrs[i]
            else:
                nurbs_lrs = lrs

            param_groups = nurbs.get_params(
                optimize=optimize,
                lrs=nurbs_lrs,
                default_lr=default_lr,
            )
            all_param_groups.extend(param_groups)

        return all_param_groups

    def splat(
        self, factor: int = 1, max_deriv_order: int = 0
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Render all NURBS curves in the scene.

        This method samples gaussians from all NURBS curves and rasterizes them
        together into a single image.

        Args:
            factor (int, optional): Scaling factor for the output dimensions. Defaults to 1.
            max_deriv_order (int, optional): If > 0, each stroke will compute
                and store analytical derivatives up to this order during
                sampling (accessible via ``stroke.sample_derivs``).

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: A tuple containing:
                - out_img (torch.Tensor): The rendered image of shape (H*factor, W*factor, 3).
                - sample_points_list (List[torch.Tensor]): List of sampled points for each NURBS.
        """
        if len(self.nurbs_list) == 0:
            # Return empty image
            final_h = int(self.H * factor)
            final_w = int(self.W * factor)
            device = self.background.device
            empty_img = (
                self.background.view(1, 1, 3).expand(final_h, final_w, 3).clone()
            )
            return empty_img, []

        final_h = int(self.H * factor)
        final_w = int(self.W * factor)

        # Collect all gaussian parameters from all NURBS
        all_xys = []
        all_scales = []
        all_rotations = []
        all_colors = []
        all_opacities = []
        sample_points_list: List[torch.Tensor] = []

        for nurbs in self.nurbs_list:
            xy, scales, rotations, colors, opacity = nurbs.sample_gaussians(
                factor, max_deriv_order
            )

            all_xys.append(xy)
            all_scales.append(scales)
            all_rotations.append(rotations)
            all_colors.append(colors)
            all_opacities.append(opacity)
            sample_points_list.append(nurbs.sample_points)

        # Concatenate all gaussian parameters
        xys = torch.cat(all_xys, dim=0)
        scales = torch.cat(all_scales, dim=0)
        rotations = torch.cat(all_rotations, dim=0)
        colors = torch.cat(all_colors, dim=0)
        opacities = torch.cat(all_opacities, dim=0)

        # Compute tile bounds
        tile_bounds = (
            (final_w + self.BLOCK_W - 1) // self.BLOCK_W,
            (final_h + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        )

        # Project gaussians
        xys_proj, depths, radii, conics, num_tiles_hit = project_gaussians_2d_scale_rot(
            xys, scales, rotations, final_h, final_w, tile_bounds
        )

        # Rasterize
        out_img = rasterize_gaussians(
            xys_proj,
            depths,
            radii,
            conics,
            num_tiles_hit,  # type: ignore
            colors,
            opacities,
            final_h,
            final_w,
            self.BLOCK_H,
            self.BLOCK_W,
            background=self.background,
            return_alpha=False,
        )

        return out_img, sample_points_list

    def total_approx_grad_energy(
        self,
        der: int = 1,
        sw_weight: float = 0.1,
        normalize: bool = False,
    ) -> torch.Tensor:
        """Compute total approximate gradient energy for all NURBS in the scene.

        Args:
            der (int): Derivative order for the finite-difference operator.
            sw_weight (float): Weight for stroke width in the energy computation.
            normalize (bool): Whether to normalize by curve length.

        Returns:
            torch.Tensor: Total scalar energy value.
        """
        if len(self.nurbs_list) == 0:
            return torch.tensor(0.0)

        total_energy = torch.tensor(0.0, device=self.nurbs_list[0]._key_points.device)
        for nurbs in self.nurbs_list:
            energy = nurbs.approx_grad_energy(
                der=der,
                sw_weight=sw_weight,
                normalize=normalize,
            )
            total_energy = total_energy + energy

        return total_energy

    def forward(self, factor: int = 1) -> torch.Tensor:
        """Forward pass for nn.Module compatibility.

        Args:
            factor (int, optional): Scaling factor for the output dimensions. Defaults to 1.

        Returns:
            torch.Tensor: The rendered image.
        """
        out_img, _ = self.splat(factor)
        return out_img

    def export_to_json(self, path: Optional[str] = None) -> dict:
        """Export the scene to a JSON-serializable dictionary.

        Args:
            path (str, optional): If provided, write JSON to this file path.

        Returns:
            dict: A dictionary containing all parameters needed to reconstruct
                this NURBSScene, including all contained NURBSSplat objects.
        """
        data = {
            "type": "NURBSScene",
            "width": self.W,
            "height": self.H,
            "background": self.background.detach().cpu().tolist(),
            "nurbs_list": [nurbs.export_to_json() for nurbs in self.nurbs_list],
        }

        if path is not None:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)

        return data

    @classmethod
    def load_from_json(
        cls, path_or_dict: Union[str, dict], device: str = "cpu"
    ) -> "NURBSScene":
        """Create a NURBSScene from a JSON file or dictionary.

        Args:
            path_or_dict (str | dict): Path to a JSON file, or a dictionary
                previously returned by :meth:`export_to_json`.
            device (str): Device to place tensors on.

        Returns:
            NURBSScene: The reconstructed scene with all NURBS splats.
        """
        if isinstance(path_or_dict, str):
            with open(path_or_dict, "r") as f:
                data = json.load(f)
        else:
            data = path_or_dict

        background = torch.tensor(
            data["background"], dtype=torch.float32, device=device
        )
        scene = cls(
            width=data["width"],
            height=data["height"],
            background=background,
        )

        nurbs_dicts = data.get("nurbs_list", [])
        nurbs_objects = [
            NURBSSplat.load_from_json(nd, device=device) for nd in nurbs_dicts
        ]
        scene.add_nurbs_list(nurbs_objects)

        return scene
