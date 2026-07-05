"""
Self-Crossing (Xing) loss for NURBS control points.

Adapted from LIVE (Ma et al., CVPR 2022) for use with NURBS key points.

The original loss penalises cubic Bezier segments that self-cross.
For NURBS, we apply a similar penalty to consecutive triplets of
key-point segments along the curve.
"""

import torch


def compute_sine_theta(s1: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
    """Compute the sine of the angle between two 2-D segments.

    Each segment is represented as ``[[x0, y0], [x1, y1]]``.
    The sign indicates the winding direction (positive = CCW).
    """
    v1 = s1[1, :] - s1[0, :]
    v2 = s2[1, :] - s2[0, :]
    sine = (v1[0] * v2[1] - v1[1] * v2[0]) / (torch.norm(v1) * torch.norm(v2) + 1e-8)
    return sine


def xing_loss(point_list: list[torch.Tensor], scale: float = 1e-3) -> torch.Tensor:
    """Self-crossing loss for a list of NURBS key-point tensors.

    For **each** key-point set (shape ``(N, 2)``), we form consecutive
    segments and penalise triplets whose third segment reverses the
    winding direction set by the first two.

    This is a direct adaptation of LIVE's ``xing_loss`` but works on any
    ordered 2-D point sequence (not just cubic Bézier control points).

    Args:
        point_list: List of tensors, each of shape ``(N, 2)`` containing
            ordered key-point positions (xy only, no stroke width).
        scale: Scalar multiplier for the loss.

    Returns:
        Scalar loss tensor.
    """
    loss = torch.tensor(0.0, device=point_list[0].device)
    for x in point_list:
        N = x.shape[0]
        if N < 4:
            continue
        # Form segments from consecutive points, closing the curve
        x_closed = torch.cat([x, x[0:1]], dim=0)  # (N+1, 2)
        segments = torch.stack(
            [x_closed[:-1], x_closed[1:]], dim=1
        )  # (N, 2, 2)  [start/end, xy]

        seg_loss = torch.tensor(0.0, device=x.device)
        # Process consecutive triplets of segments
        num_triplets = N // 3
        if num_triplets == 0:
            # Fallback: just use consecutive pairs
            num_triplets = max(N - 2, 1)
            for i in range(min(N - 2, num_triplets)):
                cs1 = segments[i]
                cs2 = segments[i + 1]
                cs3 = segments[i + 2]
                direct = (compute_sine_theta(cs1, cs2) >= 0).float()
                opst = 1.0 - direct
                sina = compute_sine_theta(cs1, cs3)
                seg_loss = (
                    seg_loss + direct * torch.relu(-sina) + opst * torch.relu(sina)
                )
            seg_loss = seg_loss / num_triplets
        else:
            for i in range(num_triplets):
                cs1 = segments[i * 3]
                cs2 = segments[i * 3 + 1]
                cs3 = segments[min(i * 3 + 2, N - 1)]
                direct = (compute_sine_theta(cs1, cs2) >= 0).float()
                opst = 1.0 - direct
                sina = compute_sine_theta(cs1, cs3)
                seg_loss = (
                    seg_loss + direct * torch.relu(-sina) + opst * torch.relu(sina)
                )
            seg_loss = seg_loss / num_triplets

        loss = loss + seg_loss * scale

    return loss / max(len(point_list), 1)
