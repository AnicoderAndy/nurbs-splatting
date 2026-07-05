from typing import Literal
import torch
import math


def first_deriv_energy(
    samples: torch.Tensor, domain_length: float = 1.0
) -> torch.Tensor:
    """Compute 1st derivative energy using finite differences.

    Approximates ∫ |f'(u)|² du using forward differences:
        f'(u_i) ≈ (p[i+1] - p[i]) / h

    Energy = Σ |p[i+1] - p[i]|² / h  (Riemann sum of squared derivative)
    Normalized by 1 / (umax - umin).

    Args:
        samples (torch.Tensor): Sampled points with shape (N, D).
        domain_length (float): Parameter domain length (umax - umin).

    Returns:
        torch.Tensor: Scalar energy value.
    """
    N = samples.shape[0]
    if N < 2:
        return torch.tensor(0.0, device=samples.device, dtype=samples.dtype)

    h = domain_length / (N - 1)

    # Forward difference: d[i] = p[i+1] - p[i]
    d = samples[1:] - samples[:-1]  # (N-1, D)

    # Energy = Σ |d|² / h, normalized by 1/domain_length
    energy = torch.sum(d**2) / h
    return energy / domain_length


def second_deriv_energy(
    samples: torch.Tensor, domain_length: float = 1.0
) -> torch.Tensor:
    """Compute 2nd derivative energy using finite differences.

    Approximates ∫ |f''(u)|² du using central differences:
        f''(u_i) ≈ (p[i+2] - 2*p[i+1] + p[i]) / h²

    Energy = Σ |p[i+2] - 2*p[i+1] + p[i]|² / h³  (Riemann sum of squared 2nd derivative)
    Normalized by 1 / (umax - umin).

    Args:
        samples (torch.Tensor): Sampled points with shape (N, D).
        domain_length (float): Parameter domain length (umax - umin).

    Returns:
        torch.Tensor: Scalar energy value.
    """
    N = samples.shape[0]
    if N < 3:
        return torch.tensor(0.0, device=samples.device, dtype=samples.dtype)

    h = domain_length / (N - 1)

    # Second difference: d2[i] = p[i+2] - 2*p[i+1] + p[i]
    d2 = samples[2:] - 2.0 * samples[1:-1] + samples[:-2]  # (N-2, D)

    # Energy = Σ |d2|² / h³, normalized by 1/domain_length
    energy = torch.sum(d2**2) / (h**3)
    return energy / domain_length


def third_deriv_energy(
    samples: torch.Tensor, domain_length: float = 1.0
) -> torch.Tensor:
    """Compute 3rd derivative energy using finite differences.

    Approximates ∫ |f'''(u)|² du using forward differences:
        f'''(u_i) ≈ (p[i+3] - 3*p[i+2] + 3*p[i+1] - p[i]) / h³

    Energy = Σ |p[i+3] - 3*p[i+2] + 3*p[i+1] - p[i]|² / h⁵  (Riemann sum of squared 3rd derivative)
    Normalized by 1 / (umax - umin).

    Args:
        samples (torch.Tensor): Sampled points with shape (N, D).
        domain_length (float): Parameter domain length (umax - umin).

    Returns:
        torch.Tensor: Scalar energy value.
    """
    N = samples.shape[0]
    if N < 4:
        return torch.tensor(0.0, device=samples.device, dtype=samples.dtype)

    h = domain_length / (N - 1)

    # Third difference: d3[i] = p[i+3] - 3*p[i+2] + 3*p[i+1] - p[i]
    d3 = (
        samples[3:] - 3.0 * samples[2:-1] + 3.0 * samples[1:-2] - samples[:-3]
    )  # (N-3, D)

    # Energy = Σ |d3|² / h⁵, normalized by 1/domain_length
    energy = torch.sum(d3**2) / (h**5)
    return energy / domain_length


def energy_from_samples(
    samples: torch.Tensor,
    der: int,
    domain_length: float = 1.0,
) -> torch.Tensor:
    """Compute derivative energy from sampled points using finite differences.

    Dispatches to explicit implementations for derivative orders 1, 2, and 3.
    Energy is normalized by 1 / (umax - umin).

    Args:
        samples (torch.Tensor): Sampled points with shape (N, D).
        der (int): Derivative order (1, 2, or 3).
        domain_length (float): Parameter domain length (umax - umin).

    Returns:
        torch.Tensor: Scalar energy value.
    """
    if samples.dim() != 2:
        raise ValueError("samples must be a 2D tensor of shape (N, D).")
    if der < 1 or der > 3:
        raise ValueError("der must be 1, 2, or 3.")

    if der == 1:
        return first_deriv_energy(samples, domain_length)
    elif der == 2:
        return second_deriv_energy(samples, domain_length)
    else:  # der == 3
        return third_deriv_energy(samples, domain_length)


def energy_from_analytical_deriv(
    deriv_values: torch.Tensor,
    domain_length: float = 1.0,
) -> torch.Tensor:
    """Compute ∫ |C^(k)(u)|² du / domain_length using quadrature.

    Unlike :func:`energy_from_samples` which approximates derivatives via
    finite differences on sampled points, this function takes the *exact*
    analytical derivative values (evaluated at uniformly-spaced parameter
    values) and integrates them directly.

    Args:
        deriv_values (torch.Tensor): (N, D) the k-th derivative C^(k)(u_i)
            evaluated at N uniformly-spaced parameter values.
        domain_length (float): Parameter domain length (u_max - u_min).

    Returns:
        torch.Tensor: Scalar energy value.
    """
    N = deriv_values.shape[0]
    if N < 2:
        return torch.tensor(0.0, device=deriv_values.device, dtype=deriv_values.dtype)
    h = domain_length / (N - 1)
    energy = torch.sum(deriv_values**2) * h / domain_length
    return energy


def make_deriv_loss(
    deriv: int,
    ref_size: float = 1.0,
    approx_method: Literal["gram", "sample", "analytical"] = "analytical",
    device: str = "cuda",
):
    """Create a smoothing loss on spline derivatives.

    Args:
        deriv (int): Derivative order to penalize.
        ref_size (float): Reference length for normalization.
        dimensionless (bool): Whether to use dimensionless scaling.
        log (bool): Whether to apply a log transform to the energy.
        approx_method (Literal["gram", "sample", "analytical"]):
            "gram" uses NURBS Gramian approximation which seems to be wrong;
            "sample" uses finite differences on sampled points;
            "analytical" uses pre-computed analytical derivatives
            stored in ``stroke.sample_derivs``
            (requires ``sample_gaussians(max_deriv_order=deriv)``).
            "gram" seems to give incorrect results.

    Returns:
        Callable: A loss function that accepts an iterable of strokes.
    """

    def _domain_length(stroke) -> torch.Tensor:
        if hasattr(stroke, "_knot_vector") and hasattr(stroke, "_order"):
            knots = stroke._knot_vector
            k = stroke._order
            u_min = knots[k - 1]
            u_max = knots[-k]
            return (u_max - u_min).abs().clamp_min(1e-8)
        return torch.tensor(
            1.0, device=stroke.sample_points.device, dtype=stroke.sample_points.dtype
        )

    def _energy(stroke, der: int, domain_len: float = 1.0) -> torch.Tensor:
        if approx_method == "gram":
            return stroke.approx_grad_energy(der=der, normalize_size=ref_size)
        if approx_method == "analytical":
            if not hasattr(stroke, "sample_derivs") or len(stroke.sample_derivs) < der:
                raise RuntimeError(
                    f"Analytical derivatives of order {der} not available on stroke. "
                    f"Call sample_gaussians(max_deriv_order={der}) first."
                )
            deriv_vals = stroke.sample_derivs[der - 1]  # (N, 3)
            combined = deriv_vals / ref_size
            return energy_from_analytical_deriv(combined, domain_length=domain_len)
        if approx_method == "sample":
            combined = stroke.sample_points / ref_size
            return energy_from_samples(combined, der, domain_length=domain_len)
        raise ValueError(f"Unknown approx_method: {approx_method}")

    def deriv_loss(paths) -> torch.Tensor:
        jloss = torch.tensor(0.0, device=device)
        c = 0

        for stroke in paths:
            if hasattr(stroke, "approx_grad_energy") and hasattr(
                stroke, "sample_points"
            ):
                T = _domain_length(stroke)
                d = _energy(stroke, deriv, domain_len=T.item())
                jloss += d
                c += 1

        if c == 0:
            return torch.tensor(0.0)
        return jloss / c

    return deriv_loss


def make_bbox_loss(box, pad: float = 5.0, device: str = "cuda"):
    """Create a soft bounding-box loss for points or paths.

    Args:
        box (tuple[tuple[float, float], tuple[float, float]]):
            Bounding box as ((x_min, y_min), (x_max, y_max)).
        pad (float): Padding applied inside the box.

    Returns:
        Callable: A loss function that penalizes points outside the box.
    """
    (x_min, y_min), (x_max, y_max) = box
    x_min = x_min + pad
    x_max = x_max - pad
    y_min = y_min + pad
    y_max = y_max - pad

    func = torch.nn.functional.softplus

    def _points_from_path(path) -> torch.Tensor:
        if hasattr(path, "sample_points"):
            pts = path.sample_points[:, :2]
            return pts
        raise ValueError("Unsupported path type for bbox loss.")

    def bbox_loss(paths):
        loss = torch.tensor(0.0, device=device)
        n = 0

        for path in paths:
            points = _points_from_path(path)
            if points.numel() == 0:
                continue
            x = points[:, 0]
            y = points[:, 1]
            x_min_t = torch.tensor(x_min, device=points.device, dtype=points.dtype)
            x_max_t = torch.tensor(x_max, device=points.device, dtype=points.dtype)
            y_min_t = torch.tensor(y_min, device=points.device, dtype=points.dtype)
            y_max_t = torch.tensor(y_max, device=points.device, dtype=points.dtype)
            x_lo = func(x_min_t - x)
            x_hi = func(x - x_max_t)
            y_lo = func(y_min_t - y)
            y_hi = func(y - y_max_t)
            loss += torch.sum((x_lo + x_hi + y_lo + y_hi))
            n += len(x)

        if n == 0:
            return torch.tensor(0.0)
        return loss / n

    return bbox_loss


# This version is designed for geometric derivatives
# But does not work well.
# def energy_from_samples(
#     samples: torch.Tensor,
#     der: int,
#     eps: float = 1e-8,
#     reduction: str = "mean",
# ) -> torch.Tensor:
#     """
#     Compute k-th order geometric derivative energy from sampled curve points.

#     Args:
#         samples (Tensor): (N, D) sampled points along the curve (approximately arc-length spaced).
#         order (int): Derivative order k >= 1.
#         eps (float): Numerical stability constant.
#         reduction (str): "mean" or "sum".

#     Returns:
#         Tensor: scalar energy.
#     """
#     if samples.ndim != 2:
#         raise ValueError("samples must have shape (N, D)")
#     if der < 1:
#         raise ValueError("order must be >= 1")
#     if samples.shape[0] <= der:
#         return torch.zeros((), device=samples.device, dtype=samples.dtype)

#     # ---- estimate arc-length step h ----
#     diffs = samples[1:] - samples[:-1]  # (N-1, D)
#     ds = torch.norm(diffs, dim=1).clamp_min(eps)  # (N-1,)
#     h = ds.mean()

#     # ---- k-th order finite differences ----
#     coeffs = torch.tensor(
#         [(-1) ** (der - j) * math.comb(der, j) for j in range(der + 1)],
#         device=samples.device,
#         dtype=samples.dtype,
#     )  # (k+1,)

#     # build Δ^k x
#     diffs_k = 0.0
#     for j in range(der + 1):
#         diffs_k = diffs_k + coeffs[j] * samples[j : samples.shape[0] - der + j]

#     # ---- energy ----
#     energy_density = torch.sum(diffs_k**2, dim=1)
#     energy = energy_density.sum() / (h ** (2 * der - 1))

#     if reduction == "mean":
#         energy = energy / diffs_k.shape[0]

#     return energy
