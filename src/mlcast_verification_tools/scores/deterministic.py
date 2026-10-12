"""Deterministic scores of the ensemble mean."""

from __future__ import annotations

import torch


def expand_mask(mask: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    """
    Broadcast a validity mask to the observation shape ``(B, T, H, W)``.

    Parameters
    ----------
    mask : torch.Tensor
        Boolean ``(B, H, W)`` (same for all lead times) or ``(B, T, H, W)``.
    obs : torch.Tensor
        Observation ``(B, T, H, W)``.

    Returns
    -------
    torch.Tensor
        Boolean ``(B, T, H, W)``.
    """
    mask = mask.to(device=obs.device, dtype=torch.bool)
    if mask.ndim == 3:
        mask = mask[:, None]
    return mask.expand(obs.shape)


def masked_spatial_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of ``values (B, T, H, W)`` over valid pixels (NaN where none are valid)."""
    zero = torch.zeros((), dtype=values.dtype, device=values.device)
    total = torch.where(mask, values, zero).sum(dim=(-2, -1))
    return total / mask.sum(dim=(-2, -1)).to(values.dtype)


def mae(fc: torch.Tensor, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Mean absolute error of the ensemble mean (paper Eq. 21).

    Parameters
    ----------
    fc : torch.Tensor
        Forecast ``(B, T, M, H, W)``.
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
    return masked_spatial_mean((fc.mean(dim=2) - obs).abs(), mask)
