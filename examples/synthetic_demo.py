"""
End-to-end demo on synthetic data: forecasts -> per-case scores -> spec file -> plots.

    uv run --extra cpu python examples/synthetic_demo.py --out demo_output

The "truth" is a stochastic rain field that advects and slowly decorrelates. Two
synthetic ensemble forecasts are verified against it:

- `calibrated`: members evolve with the same stochastic process as the truth, so they
  are statistically indistinguishable from it (flat rank histogram, realistic spectrum);
- `smoothed`: fewer members, too little stochastic evolution and increasing spatial
  smoothing with lead time (U-shaped rank histogram, missing small-scale power), like a
  blurry deterministic-ish nowcast.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from mlcast_verification_tools.io import validate_scores, write_scores
from mlcast_verification_tools.offline import compute_scores
from mlcast_verification_tools.plots import plot_all

GRID = 128
DX_KM = 2.0
TIME_STEP = pd.Timedelta("10min")
N_LEAD_TIMES = 12
INIT_SPACING_STEPS = 36  # one forecast every 6 h
N_INIT_TIMES = 40
RHO = 0.97  # step-to-step correlation of the latent field
SPECTRAL_SLOPE = 3.0  # latent field power ~ k^-slope


def _spectral_filter(slope: float = SPECTRAL_SLOPE) -> np.ndarray:
    k = np.hypot(*np.meshgrid(np.fft.fftfreq(GRID), np.fft.fftfreq(GRID), indexing="ij"))
    k[0, 0] = np.inf
    return k ** (-slope / 2)


FILTER = _spectral_filter()


def _noise(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    """Gaussian random fields with a power-law spectrum and unit variance."""
    white = rng.standard_normal((*shape, GRID, GRID))
    field = np.fft.ifft2(np.fft.fft2(white) * FILTER).real
    return field / field.std(axis=(-2, -1), keepdims=True)


def _step(latent: np.ndarray, rng: np.random.Generator, noise_scale: float = 1.0) -> np.ndarray:
    """Advect one pixel to the east and decorrelate a little."""
    innovation = np.sqrt(1 - RHO**2) * noise_scale * _noise(rng, latent.shape[:-2])
    return RHO * np.roll(latent, 1, axis=-1) + innovation


def _to_rain(latent: np.ndarray) -> np.ndarray:
    """Latent Gaussian field -> intermittent rain rate in mm/h."""
    return np.maximum(np.exp(1.2 * latent) - 1.5, 0.0).astype(np.float32)


def _smooth(field: np.ndarray, length_px: float) -> np.ndarray:
    if length_px <= 0:
        return field
    k2 = sum(f**2 for f in np.meshgrid(np.fft.fftfreq(GRID), np.fft.fftfreq(GRID), indexing="ij"))
    gauss = np.exp(-2 * (np.pi * length_px) ** 2 * k2)
    return np.fft.ifft2(np.fft.fft2(field) * gauss).real


SOURCES = [  # name, members, stochastic noise scale, smoothing length per lead time (px)
    ("calibrated", 8, 1.0, 0.0),
    ("smoothed", 4, 0.3, 0.6),
]


def make_synthetic_data(seed: int = 0) -> tuple[xr.Dataset, xr.Dataset]:
    """
    Make the truth and two ensemble forecasts in the layouts of `offline.compute_scores`.

    Returns
    -------
    ds_truth : xr.Dataset
        `tp` with dims (time, y, x) in mm/h.
    ds_forecasts : xr.Dataset
        `tp` with dims (forecast_source, init_time, lead_time, member, y, x) in mm/h;
        the source with fewer members is NaN-padded along `member`.
    """
    rng = np.random.default_rng(seed)
    n_times = N_INIT_TIMES * INIT_SPACING_STEPS + N_LEAD_TIMES + 1
    times = pd.date_range("2025-06-01", periods=n_times, freq=TIME_STEP)
    x = (np.arange(GRID) * DX_KM * 1000).astype(float)  # metres, as in projected radar data
    y = x.copy()

    latent = np.empty((n_times, GRID, GRID))
    latent[0] = _noise(rng, ())
    for t in range(1, n_times):
        latent[t] = _step(latent[t - 1], rng)
    truth = _to_rain(latent)
    truth[:, :10, :20] = np.nan  # a corner outside radar coverage
    ds_truth = xr.Dataset(
        {"tp": (("time", "y", "x"), truth, {"units": "mm h-1"})},
        coords={"time": times, "y": y, "x": x},
    )

    init_idx = np.arange(N_INIT_TIMES) * INIT_SPACING_STEPS
    lead_times = pd.timedelta_range(TIME_STEP, periods=N_LEAD_TIMES, freq=TIME_STEP)
    max_members = max(n for _, n, _, _ in SOURCES)
    fc = np.full(
        (len(SOURCES), N_INIT_TIMES, N_LEAD_TIMES, max_members, GRID, GRID),
        np.nan,
        dtype=np.float32,
    )
    for s, (_, n_members, noise_scale, smoothing_px) in enumerate(SOURCES):
        for i, t0 in enumerate(init_idx):
            state = np.repeat(latent[t0][None], n_members, axis=0)
            for lt in range(N_LEAD_TIMES):
                state = _step(state, rng, noise_scale)
                fc[s, i, lt, :n_members] = _to_rain(_smooth(state, smoothing_px * (lt + 1)))
    ds_forecasts = xr.Dataset(
        {
            "tp": (
                ("forecast_source", "init_time", "lead_time", "member", "y", "x"),
                fc,
                {"units": "mm h-1"},
            )
        },
        coords={
            "forecast_source": [name for name, *_ in SOURCES],
            "init_time": times[init_idx],
            "lead_time": lead_times,
            "member": np.arange(max_members),
            "y": y,
            "x": x,
        },
    )
    return ds_truth, ds_forecasts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--out", type=Path, default=Path("demo_output"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("Making synthetic observations and forecasts ...")
    ds_truth, ds_forecasts = make_synthetic_data(args.seed)

    print("Computing per-case scores ...")
    scores = compute_scores(ds_forecasts, ds_truth, batch_size=4, seed=args.seed)

    path = write_scores(
        scores,
        args.out / "scores.zarr",
        title="Synthetic precipitation nowcast scores (demo)",
        forecasts="synthetic ensembles 'calibrated' (8 members) and 'smoothed' (4 members)",
        truth="synthetic advected stochastic rain field",
    )
    print(f"Wrote {path}")

    report = validate_scores(path)
    report.console_print()

    figures = plot_all(xr.open_zarr(path), args.out)
    print("Wrote figures:", *map(str, figures), sep="\n  ")


if __name__ == "__main__":
    main()
