"""Device-agnostic torch kernels computing per-case verification scores.

All kernels take ``fc (B, T, M, H, W)`` and ``obs (B, T, H, W)`` in mm/h and a boolean
validity ``mask (B, H, W)`` or ``(B, T, H, W)`` and return per-case scores ``(B, T, ...)``.
"""

from __future__ import annotations

from .deterministic import mae
from .probabilistic import crps_ensemble, rank_histogram
from .spectral import ensemble_radial_psd, psd_2d, psd_wavelengths, radial_psd

__all__ = [
    "crps_ensemble",
    "ensemble_radial_psd",
    "mae",
    "psd_2d",
    "psd_wavelengths",
    "radial_psd",
    "rank_histogram",
]
