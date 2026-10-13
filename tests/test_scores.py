from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from mlcast_verification_tools import testing
from mlcast_verification_tools.scores import (
    crps_ensemble,
    ensemble_radial_psd,
    mae,
    psd_2d,
    psd_wavelengths,
    radial_psd,
    rank_histogram,
)


def _random_case(b=2, t=3, k=5, h=8, w=10, seed=0, dtype=torch.float64):
    gen = torch.Generator().manual_seed(seed)
    fc = torch.rand(b, t, k, h, w, generator=gen, dtype=dtype) * 5
    obs = torch.rand(b, t, h, w, generator=gen, dtype=dtype) * 5
    mask = torch.rand(b, t, h, w, generator=gen) > 0.3
    return fc, obs, mask


# --- MAE / CRPS ---------------------------------------------------------------------


def test_mae_hand_computed():
    fc = torch.tensor([1.0, 3.0, 2.0, 6.0]).reshape(1, 1, 2, 1, 2)  # members (1,3),(2,6)
    obs = torch.tensor([0.0, 10.0]).reshape(1, 1, 1, 2)
    # ensemble mean (1.5, 4.5); abs errors (1.5, 5.5)
    assert mae(fc, obs, torch.ones(1, 1, 2, dtype=torch.bool)).item() == pytest.approx(3.5)
    mask = torch.tensor([True, False]).reshape(1, 1, 2)
    assert mae(fc, obs, mask).item() == pytest.approx(1.5)


def test_mae_ignores_nan_at_invalid_pixels_and_nan_without_valid_pixels():
    fc, obs, mask = _random_case()
    obs = obs.clone()
    obs[~mask] = float("nan")
    out = mae(fc, obs, mask)
    assert torch.isfinite(out).all()
    out = mae(fc, obs, torch.zeros_like(mask))
    assert torch.isnan(out).all()


def test_crps_matches_brute_force():
    fc, obs, mask = _random_case(k=7)
    k = fc.shape[2]
    term1 = (fc - obs[:, :, None]).abs().mean(dim=2)
    term2 = (fc[:, :, :, None] - fc[:, :, None, :]).abs().sum(dim=(2, 3)) / (2 * k**2)
    per_pixel = term1 - term2
    expected = (per_pixel * mask).sum(dim=(-2, -1)) / mask.sum(dim=(-2, -1))
    torch.testing.assert_close(crps_ensemble(fc, obs, mask), expected)


def test_crps_single_member_equals_mae():
    fc, obs, mask = _random_case(k=1)
    torch.testing.assert_close(crps_ensemble(fc, obs, mask), mae(fc, obs, mask))


def test_mask_3d_broadcasts_over_lead_time():
    fc, obs, _ = _random_case()
    mask3 = torch.rand(fc.shape[0], *fc.shape[-2:]) > 0.5
    mask4 = mask3[:, None].expand(obs.shape)
    torch.testing.assert_close(crps_ensemble(fc, obs, mask3), crps_ensemble(fc, obs, mask4))


# --- rank histogram -----------------------------------------------------------------


def test_rank_histogram_calibrated_is_flat():
    k = 9
    gen = torch.Generator().manual_seed(1)
    fc = torch.randn(2, 2, k, 64, 64, generator=gen, dtype=torch.float64).exp()
    obs = torch.randn(2, 2, 64, 64, generator=gen, dtype=torch.float64).exp()
    mask = torch.ones(2, 2, 64, 64, dtype=torch.bool)
    hist = rank_histogram(fc, obs, mask, zero_threshold=0.0, generator=gen)
    assert hist.shape == (2, 2, k + 1)
    torch.testing.assert_close(hist.sum(-1), torch.ones(2, 2, dtype=torch.float64))
    rmsd = ((hist - 1 / (k + 1)) ** 2).mean(-1).sqrt()
    assert (rmsd < 0.01).all()


def test_rank_histogram_ties_broken_randomly():
    k = 4
    fc = torch.zeros(1, 1, k, 100, 100)
    obs = torch.full((1, 1, 100, 100), 0.01)  # below threshold -> zero -> all tied
    mask = torch.ones(1, 100, 100, dtype=torch.bool)
    hist = rank_histogram(fc, obs, mask, generator=torch.Generator().manual_seed(0))
    torch.testing.assert_close(hist, torch.full((1, 1, k + 1), 1 / (k + 1)), atol=0.02, rtol=0)


def test_rank_histogram_threshold_any_units():
    """Sub-threshold values tie below all others, also for negative values (e.g. dBZ)."""
    fc, obs, mask = _random_case(k=5)
    gen = torch.Generator().manual_seed(2)
    expected = rank_histogram(fc, obs, mask, zero_threshold=0.5, generator=gen)
    # shifting all values and the threshold by the same amount does not change ranks
    gen = torch.Generator().manual_seed(2)
    shifted = rank_histogram(fc - 40, obs - 40, mask, zero_threshold=-39.5, generator=gen)
    torch.testing.assert_close(shifted, expected)


def test_rank_histogram_obs_above_all_members():
    fc, obs, mask = _random_case(k=5)
    hist = rank_histogram(fc, obs + 10, mask)
    expected = torch.zeros_like(hist)
    expected[..., -1] = 1
    torch.testing.assert_close(hist, expected)


def test_rank_histogram_nan_without_valid_pixels():
    fc, obs, mask = _random_case()
    mask = mask.clone()
    mask[0] = False
    hist = rank_histogram(fc, obs, mask)
    assert torch.isnan(hist[0]).all()
    assert torch.isfinite(hist[1]).all()


def test_rank_histogram_reproducible_with_generator():
    fc = torch.zeros(1, 1, 3, 16, 16)
    obs = torch.zeros(1, 1, 16, 16)
    mask = torch.ones(1, 16, 16, dtype=torch.bool)
    h1 = rank_histogram(fc, obs, mask, generator=torch.Generator().manual_seed(3))
    h2 = rank_histogram(fc, obs, mask, generator=torch.Generator().manual_seed(3))
    torch.testing.assert_close(h1, h2)


# --- spectra ------------------------------------------------------------------------


def test_wavelengths_match_testing_convention():
    ds = testing.make_synthetic_scores(domain_size=64, dx_km=2.0, n_init_times=2, n_lead_times=2)
    np.testing.assert_allclose(psd_wavelengths(64, 64, 2.0).numpy(), ds["wavelength"].values)
    assert radial_psd(torch.zeros(64, 64), 2.0).shape == (32,)


def test_wavelengths_non_square_use_smaller_side():
    wl = psd_wavelengths(48, 80, 1.0)
    assert wl.shape == (24,)
    assert wl[0].item() == pytest.approx(48.0)
    assert radial_psd(torch.zeros(3, 48, 80), 1.0).shape == (3, 24)


@pytest.mark.parametrize("shape", [(32, 32), (24, 40)])
def test_psd_parseval(shape):
    """Σ P dkx dky = <(f w)²> / <w²> (see module docstring)."""
    dx = 2.5
    h, w = shape
    gen = torch.Generator().manual_seed(0)
    field = torch.randn(3, h, w, generator=gen, dtype=torch.float64) + 1.0
    win = torch.outer(
        torch.hann_window(h, periodic=False, dtype=torch.float64),
        torch.hann_window(w, periodic=False, dtype=torch.float64),
    )
    p = psd_2d(field, dx)
    lhs = p.sum(dim=(-2, -1)) / (w * dx * h * dx)
    rhs = ((field * win) ** 2).mean(dim=(-2, -1)) / (win**2).mean()
    torch.testing.assert_close(lhs, rhs)


def test_radial_psd_sums_annuli():
    """Σ_k PSD(k) Δk equals Σ P dkx dky over the kept annuli 1 <= |k| <= N/2."""
    dx, n = 1.5, 32
    field = torch.randn(n, n, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    p = psd_2d(field, dx)
    k = torch.fft.fftfreq(n, dtype=torch.float64) * n
    kr = torch.round(torch.sqrt(k[:, None] ** 2 + k[None, :] ** 2))
    keep = (kr >= 1) & (kr <= n // 2)
    expected = (p * keep).sum() / (n * dx) ** 2
    assert radial_psd(field, dx).sum().item() / (n * dx) == pytest.approx(expected.item())


def test_white_noise_flat_2d_psd_and_linear_radial_psd():
    """White noise of variance σ² has E[P] = σ² dx²; then PSD(k) ≈ 2π k σ² dx / N."""
    n, dx, sigma = 64, 2.0, 3.0
    gen = torch.Generator().manual_seed(0)
    field = sigma * torch.randn(400, n, n, generator=gen, dtype=torch.float64)
    p_mean = psd_2d(field, dx).mean(0)
    # flat: away from k=0 (where the window leaks) every wavenumber has the same power
    rel = p_mean[4:-4, 4:-4] / (sigma**2 * dx**2)
    assert rel.mean().item() == pytest.approx(1.0, abs=0.02)
    assert rel.std().item() < 0.1

    radial = radial_psd(field, dx).mean(0)
    # exact expectation: (#pixels in annulus) σ² dx² dkx dky / Δk = count σ² dx / N
    kf = torch.fft.fftfreq(n, dtype=torch.float64) * n
    kr = torch.round(torch.sqrt(kf[:, None] ** 2 + kf[None, :] ** 2)).long().flatten()
    count = torch.bincount(kr, minlength=n)[1 : n // 2 + 1].double()
    ratio = radial[2:] / (count[2:] * sigma**2 * dx / n)
    assert (ratio - 1).abs().max().item() < 0.05
    # and the annulus count is ≈ 2πk, so the radial PSD grows linearly with k
    k = torch.arange(1, n // 2 + 1, dtype=torch.float64)
    linear = radial[4:] / (2 * math.pi * k[4:] * sigma**2 * dx / n)
    assert linear.mean().item() == pytest.approx(1.0, abs=0.05)


@pytest.mark.parametrize("shape,k0", [((64, 64), 8), ((64, 96), 5)])
def test_sinusoid_peaks_at_its_wavelength(shape, k0):
    h, w = shape
    dx = 1.0
    n = min(h, w)
    wavelength = n * dx / k0
    x = torch.arange(w, dtype=torch.float64) * dx
    field = torch.sin(2 * math.pi * x / wavelength).expand(h, w)
    psd = radial_psd(field, dx)
    wl = psd_wavelengths(h, w, dx)
    assert wl[psd.argmax()].item() == pytest.approx(wavelength)


def test_psd_mask_zeroes_invalid_pixels():
    field = torch.randn(32, 32, dtype=torch.float64)
    mask = torch.rand(32, 32) > 0.2
    masked = torch.where(mask, field, torch.full_like(field, float("nan")))
    torch.testing.assert_close(
        radial_psd(masked, 1.0, mask), radial_psd(torch.where(mask, field, 0.0), 1.0)
    )


def test_ensemble_radial_psd_shapes_and_member_mean():
    fc, obs, mask = _random_case(k=3, h=16, w=16)
    psd_fc, psd_obs = ensemble_radial_psd(fc, obs, mask, 1.0)
    assert psd_fc.shape == psd_obs.shape == (2, 3, 8)
    torch.testing.assert_close(psd_fc, radial_psd(fc, 1.0, mask[:, :, None]).mean(2))
    torch.testing.assert_close(psd_obs, radial_psd(obs, 1.0, mask))


# --- device equality -----------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cpu_cuda_equal():
    fc, obs, mask = _random_case(k=6, h=32, w=32)
    for fn in (mae, crps_ensemble):
        torch.testing.assert_close(fn(fc, obs, mask), fn(fc.cuda(), obs.cuda(), mask.cuda()).cpu())
    for cpu, gpu in zip(
        ensemble_radial_psd(fc, obs, mask, 1.0),
        ensemble_radial_psd(fc.cuda(), obs.cuda(), mask.cuda(), 1.0),
        strict=True,
    ):
        torch.testing.assert_close(cpu, gpu.cpu())
    # continuous data -> no ties -> rank histogram independent of the random draws
    hist_cpu = rank_histogram(fc, obs, mask, zero_threshold=0.0)
    hist_gpu = rank_histogram(fc.cuda(), obs.cuda(), mask.cuda(), zero_threshold=0.0)
    torch.testing.assert_close(hist_cpu, hist_gpu.cpu())
