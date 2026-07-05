from . import segmentation, tsp_art
from easydict import EasyDict as edict
import numpy as np
from PIL import Image
from typing import Literal


def softmax(x, tau=0.2):
    e_x = np.exp(x / tau)
    return e_x / e_x.sum()


def init_path_tsp(
    img: np.ndarray | Image.Image,
    n: int,
    nb_iter: int = 30,
    saliency_type: Literal["intensity", "ood", "clip"] = "intensity",
    mask: np.ndarray | None = None,
    closed: bool = False,
    minutes_limit: float = 1 / 30,
    **kwargs,
):
    # Saliency
    if type(img) == np.ndarray:
        img = Image.fromarray((img * 255).astype(np.uint8))
    if saliency_type == "ood":
        sal = segmentation.ood_saliency(img.convert("RGB"))[0]
    elif saliency_type == "clip":
        sal = segmentation.clip_saliency(img.convert("RGB"))
    else:
        sal = np.array(img.convert("L")) / 255

    density_map = ((sal - sal.min()) / (sal.max() - sal.min())) ** 2  # (sal*(1-img))
    if mask is not None:
        density_map *= mask
    # density_map = density_map*(1-img)
    points = tsp_art.weighted_voronoi_sampling(density_map, n, nb_iter=nb_iter)
    # Find top left point
    n = len(points)
    P = [tuple(p) for p in points]
    # sorted according to x,y
    I = sorted(list(range(n)), key=lambda i: P[i])
    points = np.array([points[i] for i in I])
    # TSP func assumes first and last points are fixed if cycle is True and end_to_end is True
    I = tsp_art.heuristic_solve(
        points,
        time_limit_minutes=minutes_limit,
        cycle=closed,
        end_to_end=not closed,
        **kwargs,
    )  # , logging=True, verbose=True)
    if I is None:
        print("Warning: TSP solver failed to find a solution, using unsolved order")
        P = points
    else:
        P = points[I]
    return P, density_map
