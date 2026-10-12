"""Radially integrated power spectral density (IRENE paper, Eqs. 16-17).

Normalisation
-------------
For a field ``f`` on an ``Ny x Nx`` grid (``Ny = H``, ``Nx = W``) with spacing
``dx = dy`` (km), the 2-D Hann window ``w = hann(Ny) ⊗ hann(Nx)`` (``periodic=False``)
is applied and the 2-D PSD is

    P(kx, ky) = |FFT2(f w)|² dx dy / (Nx Ny <w²>),

in (field units)² km². By Parseval's theorem, with ``dkx = 1/(Nx dx)`` and
``dky = 1/(Ny dy)``,

    Σ_{kx,ky} P dkx dky = <(f w)²> / <w²>,

i.e. the window-weighted mean square of the field. For zero-mean white noise of
variance σ², E[P] = σ² dx dy at every wavenumber.

Radial binning
--------------
Wavenumbers are expressed in units of ``Δk = 1/(N dx)`` with ``N = min(H, W)``, i.e.
``kx = fftfreq(Nx) * N`` and ``ky = fftfreq(Ny) * N`` (dimensionless), and each
(kx, ky) is assigned to the integer bin ``k = round(sqrt(kx² + ky²))``. Bins
``k = 1 .. N // 2`` are kept (``k = 0`` and the corners beyond ``N // 2`` are dropped).
The radial PSD is the discretisation of ``k ∮ P dθ``:

    PSD(k) = Σ_{annulus k} P dkx dky / Δk,

in (field units)² km, so that ``Σ_k PSD(k) Δk`` equals the Parseval sum restricted
to the kept annuli. White noise therefore gives ``PSD(k) ≈ 2π k σ² dx / N`` (∝ k).
The wavelength of bin ``k`` is ``N dx / k``.
"""

from __future__ import annotations

import functools

import torch

from .deterministic import expand_mask


def psd_wavelengths(height: int, width: int, dx_km: float) -> torch.Tensor:
    """
    Wavelengths (km) of the radial PSD bins, ``N dx / k`` for ``k = 1 .. N // 2``.

    Parameters
    ----------
    height, width : int
        Grid size; ``N = min(height, width)``.
    dx_km : float
        Grid spacing in km.

    Returns
    -------
    torch.Tensor
        float64 tensor of shape ``(N // 2,)``, descending.
    """
    n = min(height, width)
    k = torch.arange(1, n // 2 + 1, dtype=torch.float64)
    return n * dx_km / k


@functools.lru_cache(maxsize=32)
def _spectral_setup(
    height: int, width: int, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Hann window ``(H, W)``, flat bin index ``(H*W,)`` and number of bins for a grid."""
    n = min(height, width)
    n_bins = n // 2
    win = torch.outer(
        torch.hann_window(height, periodic=False, dtype=dtype, device=device),
        torch.hann_window(width, periodic=False, dtype=dtype, device=device),
    )
    ky = torch.fft.fftfreq(height, device=device, dtype=torch.float64) * n
    kx = torch.fft.fftfreq(width, device=device, dtype=torch.float64) * n
    k = torch.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    k_bin = torch.round(k).long()
    # bins 1..n_bins -> index 0..n_bins-1; everything else -> dump bin n_bins
    idx = torch.where((k_bin >= 1) & (k_bin <= n_bins), k_bin - 1, n_bins)
    return win, idx.reshape(-1), n_bins


def psd_2d(field: torch.Tensor, dx_km: float, mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    Windowed 2-D PSD ``P = |FFT2(f w)|² dx dy / (Nx Ny <w²>)`` (paper Eq. 16).

    Parameters
    ----------
    field : torch.Tensor
        Real field of shape ``(..., H, W)``.
    dx_km : float
        Grid spacing in km (``dx = dy``).
    mask : torch.Tensor, optional
        Boolean, broadcastable to ``field``; False pixels are set to 0 before windowing.

    Returns
    -------
    torch.Tensor
        Shape ``(..., H, W)`` in FFT order, units (field units)² km².
    """
    height, width = field.shape[-2:]
    win, _, _ = _spectral_setup(height, width, field.device, field.dtype)
    if mask is not None:
        field = torch.where(mask, field, torch.zeros((), dtype=field.dtype, device=field.device))
    spec = torch.fft.fft2(field * win)
    norm = dx_km * dx_km / (height * width * (win**2).mean())
    return spec.abs() ** 2 * norm


def radial_psd(field: torch.Tensor, dx_km: float, mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    Radially integrated PSD ``k ∮ P dθ`` (paper Eq. 17), see module docstring.

    Parameters
    ----------
    field : torch.Tensor
        Real field of shape ``(..., H, W)``.
    dx_km : float
        Grid spacing in km.
    mask : torch.Tensor, optional
        Boolean, broadcastable to ``field``; False pixels are set to 0 before windowing.

    Returns
    -------
    torch.Tensor
        Shape ``(..., N // 2)`` with ``N = min(H, W)``, ordered by increasing wavenumber
        (matching :func:`psd_wavelengths`), units (field units)² km.
    """
    height, width = field.shape[-2:]
    _, idx, n_bins = _spectral_setup(height, width, field.device, field.dtype)
    p = psd_2d(field, dx_km, mask)
    lead_shape = p.shape[:-2]
    p = p.reshape(-1, height * width)
    out = torch.zeros(p.shape[0], n_bins + 1, dtype=p.dtype, device=p.device)
    out.scatter_add_(1, idx.expand(p.shape[0], -1), p)
    n = min(height, width)
    dk2 = 1.0 / (width * dx_km * height * dx_km)
    delta_k = 1.0 / (n * dx_km)
    out = out[:, :n_bins] * (dk2 / delta_k)
    return out.reshape(*lead_shape, n_bins)


def ensemble_radial_psd(
    fc: torch.Tensor, obs: torch.Tensor, mask: torch.Tensor, dx_km: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Radial PSD of the forecast (mean over members) and of the observation.

    Parameters
    ----------
    fc : torch.Tensor
        Forecast ``(B, T, M, H, W)``.
    obs : torch.Tensor
        Observation ``(B, T, H, W)``.
    mask : torch.Tensor
        Boolean ``(B, H, W)`` or ``(B, T, H, W)``, True = valid.
    dx_km : float
        Grid spacing in km.

    Returns
    -------
    psd_fc, psd_obs : torch.Tensor
        Each ``(B, T, n_bins)``.
    """
    mask = expand_mask(mask, obs)
    psd_fc = radial_psd(fc, dx_km, mask[:, :, None]).mean(dim=2)
    psd_obs = radial_psd(obs, dx_km, mask)
    return psd_fc, psd_obs
