"""Probabilistic ensemble scores: CRPS and rank histogram."""

from __future__ import annotations

import torch

from .. import conventions as cv
from .deterministic import expand_mask, masked_spatial_mean


def crps_ensemble(fc: torch.Tensor, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Ensemble CRPS (paper Eq. 8, not the "fair" version), averaged over valid pixels.

    Per pixel ``CRPS = (1/K) Σ_k |x_k - y| - (1/(2K²)) Σ_k Σ_j |x_k - x_j|``, with the
    double sum evaluated from the sorted members as
    ``(1/K²) Σ_i (2i - K - 1) x_(i)``, i = 1..K, so no K x K tensor is built.

    Parameters
    ----------
    fc : torch.Tensor
        Forecast ``(B, T, K, H, W)``.
    obs : torch.Tensor
        Observation ``(B, T, H, W)``.
    mask : torch.Tensor
        Boolean ``(B, H, W)`` or ``(B, T, H, W)``, True = valid pixel.

    Returns
    -------
    torch.Tensor
        ``(B, T)``, NaN for cases without valid pixels.
    """
    mask = expand_mask(mask, obs)
    k = fc.shape[2]
    skill = (fc - obs[:, :, None]).abs().mean(dim=2)
    sorted_fc = torch.sort(fc, dim=2).values
    i = torch.arange(1, k + 1, device=fc.device, dtype=fc.dtype)
    weights = (2 * i - k - 1).reshape(1, 1, k, 1, 1)
    spread = (weights * sorted_fc).sum(dim=2) / k**2
    return masked_spatial_mean(skill - spread, mask)


def rank_histogram(
    fc: torch.Tensor,
    obs: torch.Tensor,
    mask: torch.Tensor,
    zero_threshold: float = cv.RANK_ZERO_THRESHOLD_MM_H,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Relative frequency of the observation's rank among the K members, per case.

    Values below ``zero_threshold`` are set to 0 in forecast and observation. The rank
    is ``1 + #{x_k < y} + floor(U (1 + #{x_k == y}))`` with ``U ~ Uniform[0, 1)``,
    so ties are broken at random.

    Parameters
    ----------
    fc : torch.Tensor
        Forecast ``(B, T, K, H, W)``.
    obs : torch.Tensor
        Observation ``(B, T, H, W)``.
    mask : torch.Tensor
        Boolean ``(B, H, W)`` or ``(B, T, H, W)``, True = valid pixel.
    zero_threshold : float
        Rain rate (mm/h) below which values count as zero.
    generator : torch.Generator, optional
        Random generator on the device of ``fc``.

    Returns
    -------
    torch.Tensor
        ``(B, T, K + 1)`` summing to 1 per case (NaN for cases without valid pixels).
    """
    mask = expand_mask(mask, obs)
    b, t, k = fc.shape[:3]
    zero = torch.zeros((), dtype=fc.dtype, device=fc.device)
    fc = torch.where(fc < zero_threshold, zero, fc)
    obs = torch.where(obs < zero_threshold, zero, obs)[:, :, None]
    n_below = (fc < obs).sum(dim=2)
    n_equal = (fc == obs).sum(dim=2)
    u = torch.rand(n_below.shape, generator=generator, device=fc.device, dtype=fc.dtype)
    rank0 = n_below + torch.floor(u * (1 + n_equal)).long()  # 0-based rank in 0..K
    rank0 = rank0.clamp(0, k)  # guards against NaN inputs at invalid pixels
    case = torch.arange(b * t, device=fc.device).reshape(b, t, 1, 1)
    flat_idx = (case * (k + 1) + rank0).reshape(-1)
    counts = torch.zeros(b * t * (k + 1), dtype=fc.dtype, device=fc.device)
    counts.scatter_add_(0, flat_idx, mask.reshape(-1).to(fc.dtype))
    counts = counts.reshape(b, t, k + 1)
    return counts / counts.sum(dim=-1, keepdim=True)
