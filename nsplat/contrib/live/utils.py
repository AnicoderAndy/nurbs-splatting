"""
Utility functions for LIVE vectorization.

Adapted from LIVE (Ma et al., CVPR 2022).
"""

import os
import os.path as osp
import copy
import math

import cv2
import numpy as np
import numpy.random as npr
import torch


# ─────────────────── Experiment helpers ──────────────────────


def get_experiment_id(debug: bool = False) -> int:
    if debug:
        return 999999999999
    import time

    time.sleep(0.5)
    return int(time.time() * 100)


def check_and_create_dir(path: str):
    pathdir = osp.split(path)[0]
    if not osp.isdir(pathdir):
        os.makedirs(pathdir, exist_ok=True)


def edict_2_dict(x):
    if isinstance(x, dict):
        return {k: edict_2_dict(v) for k, v in x.items()}
    elif isinstance(x, list):
        return [edict_2_dict(i) for i in x]
    return x


# ─────────────────── Path schedule ───────────────────────────


def get_path_schedule(type: str, **kwargs) -> list:
    """Return a list of integers specifying how many paths to add per round.

    Supported types:
        - ``repeat``: Add ``schedule_each`` paths for ``max_path`` rounds.
        - ``list``:   Explicit schedule as a Python list.
        - ``exp``:    Exponential growth capped at ``max_path_per_iter``.
    """
    if type == "repeat":
        return [kwargs["schedule_each"]] * kwargs["max_path"]
    elif type == "list":
        return kwargs["schedule"]
    elif type == "exp":
        base = kwargs["base"]
        max_path = kwargs["max_path"]
        max_path_per_iter = kwargs["max_path_per_iter"]
        schedule = []
        cnt = 0
        while sum(schedule) < max_path:
            proposed = min(max_path - sum(schedule), base**cnt, max_path_per_iter)
            cnt += 1
            schedule.append(proposed)
        return schedule
    else:
        raise ValueError(f"Unknown schedule type: {type}")


# ───────────────── Coordinate initializers ───────────────────


class random_coord_init:
    """Place new paths at uniformly-random canvas positions."""

    def __init__(self, canvas_size):
        self.canvas_size = canvas_size  # (H, W)

    def __call__(self):
        h, w = self.canvas_size
        return [npr.uniform(0, 1) * w, npr.uniform(0, 1) * h]


class naive_coord_init:
    """Place new paths at the pixel with maximum squared error."""

    def __init__(self, pred, gt, format="[bs x c x 2D]", replace_sampling=True):
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(gt, torch.Tensor):
            gt = gt.detach().cpu().numpy()

        if format == "[bs x c x 2D]":
            self.map = ((pred[0] - gt[0]) ** 2).sum(0)
        else:
            raise ValueError(f"Unsupported format: {format}")
        self.replace_sampling = replace_sampling

    def __call__(self):
        coord = np.where(self.map == self.map.max())
        coord_h, coord_w = coord[0][0], coord[1][0]
        if self.replace_sampling:
            self.map[coord_h, coord_w] = -1
        return [coord_w, coord_h]


class sparse_coord_init:
    """Component-wise initialization: greedily select the largest error region.

    Steps:
        1. Compute per-pixel squared error.
        2. Quantize into histogram bins.
        3. On each call, select the bin with the most pixels → run connected
           component analysis → return centroid of largest component.
    """

    def __init__(
        self,
        pred,
        gt,
        format="[bs x c x 2D]",
        quantile_interval=200,
        nodiff_thres=0.1,
    ):
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(gt, torch.Tensor):
            gt = gt.detach().cpu().numpy()

        if format == "[bs x c x 2D]":
            self.map = ((pred[0] - gt[0]) ** 2).sum(0)
            self.reference_gt = copy.deepcopy(np.transpose(gt[0], (1, 2, 0)))
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Zero out tiny errors to avoid dead-loops
        self.map[self.map < nodiff_thres] = 0

        quantile_interval = np.linspace(0.0, 1.0, quantile_interval)
        quantized_interval = np.quantile(self.map, quantile_interval)
        quantized_interval = np.unique(quantized_interval)
        quantized_interval = sorted(quantized_interval[1:-1])
        self.map = np.digitize(self.map, quantized_interval, right=False)
        self.map = np.clip(self.map, 0, 255).astype(np.uint8)

        self.idcnt = {}
        for idi in sorted(np.unique(self.map)):
            self.idcnt[idi] = int((self.map == idi).sum())
        # Remove the "no-error" bin
        self.idcnt.pop(min(self.idcnt.keys()), None)

    def __call__(self):
        if len(self.idcnt) == 0:
            h, w = self.map.shape
            return [npr.uniform(0, 1) * w, npr.uniform(0, 1) * h]

        target_id = max(self.idcnt, key=self.idcnt.get)
        _, component, cstats, ccenter = cv2.connectedComponentsWithStats(
            (self.map == target_id).astype(np.uint8), connectivity=4
        )
        csize = [ci[-1] for ci in cstats[1:]]
        target_cid = csize.index(max(csize)) + 1
        center = ccenter[target_cid][::-1]
        coord = np.stack(np.where(component == target_cid)).T
        dist = np.linalg.norm(coord - center, axis=1)
        target_coord_id = np.argmin(dist)
        coord_h, coord_w = coord[target_coord_id]

        # Remove sampled component
        self.idcnt[target_id] -= max(csize)
        if self.idcnt[target_id] <= 0:
            self.idcnt.pop(target_id, None)
        self.map[component == target_cid] = 0

        return [coord_w, coord_h]


# ──────────────── SDF-based loss weighting ───────────────────


def get_sdf(phi: np.ndarray, method: str = "skfmm", **kwargs) -> np.ndarray:
    """Compute unsigned distance field from a binary mask.

    The result is inverted and normalised so pixels near the boundary
    receive *high* weight while far-away pixels receive *low* weight.

    Args:
        phi: 2-D float array, values in [0, 1] where >0.5 is inside.
        method: Only ``'skfmm'`` is currently supported.

    Keyword Args:
        flip_negative (bool): Take absolute value of signed distance (default True).
        truncate (float): Truncation range (default 10).
        zero2max (bool): Invert so max distance → 0 weight (default True).
        normalize (str): ``'sum'`` or ``'to1'`` (default ``'sum'``).
    """
    if method != "skfmm":
        raise ValueError(f"Unsupported SDF method: {method}")
    import skfmm

    phi = (phi - 0.5) * 2
    if phi.max() <= 0 or phi.min() >= 0:
        return np.zeros(phi.shape, dtype=np.float32)

    sd = skfmm.distance(phi, dx=1)

    flip_negative = kwargs.get("flip_negative", True)
    if flip_negative:
        sd = np.abs(sd)

    truncate = kwargs.get("truncate", 10)
    sd = np.clip(sd, -truncate, truncate)

    zero2max = kwargs.get("zero2max", True)
    if zero2max and flip_negative:
        sd = sd.max() - sd
    elif zero2max:
        raise ValueError("zero2max requires flip_negative=True")

    normalize = kwargs.get("normalize", "sum")
    if normalize == "sum":
        s = sd.sum()
        if s > 0:
            sd /= s
    elif normalize == "to1":
        m = sd.max()
        if m > 0:
            sd /= m

    return sd.astype(np.float32)


# ─────────────── LR schedule ────────────────────────────────


class linear_decay_lrlambda_f:
    """Linear interpolation between successive exponential decay steps."""

    def __init__(self, decay_every: int, decay_ratio: float):
        self.decay_every = decay_every
        self.decay_ratio = decay_ratio

    def __call__(self, n: int) -> float:
        decay_time = n // self.decay_every
        decay_step = n % self.decay_every
        lr_s = self.decay_ratio**decay_time
        lr_e = self.decay_ratio ** (decay_time + 1)
        r = decay_step / self.decay_every
        return lr_s * (1 - r) + lr_e * r
