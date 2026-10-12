"""Synthetic score datasets in the package's spec layout, for tests and examples.

The values are made up but have plausible shapes (errors grow with lead time, rank
histograms are U-shaped, spectra lose small-scale power with lead time), so plots
made from them look like real verification plots. Only numpy/xarray are needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from . import conventions as cv


def make_synthetic_scores(
    ensemble_sizes: dict[str, int] | None = None,
    n_init_times: int = 40,
    n_lead_times: int = 12,
    lead_time_step: str = "5min",
    domain_size: int = 256,
    dx_km: float = 1.0,
    seed: int = 0,
) -> xr.Dataset:
    """
    Make a score dataset (without global attributes) in the spec layout.

    Parameters
    ----------
    ensemble_sizes : dict of str to int, optional
        Number of members per forecast source. Defaults to two sources with
        different ensemble sizes, so the `rank` dim is NaN-padded for one of them.
    n_init_times, n_lead_times : int
        Number of forecast cases and lead times.
    lead_time_step : str
        pandas timedelta string of the lead time spacing (first lead time = one step).
    domain_size, dx_km : int, float
        Size (pixels) and grid spacing of the square domain, which set the
        `wavelength` coordinate: wavelength = domain_size * dx_km / k, k = 1..domain_size//2.
    seed : int
        Random seed.

    Returns
    -------
    xr.Dataset
        Variables `mae.tp`, `crps.tp`, `rank_histogram.tp`, `psd.tp`, `psd_obs.tp`.
    """
    if ensemble_sizes is None:
        ensemble_sizes = {"model_a": 10, "model_b": 4}
    rng = np.random.default_rng(seed)

    sources = list(ensemble_sizes)
    init_time = pd.date_range("2025-01-01", periods=n_init_times, freq="1D")
    step = pd.Timedelta(lead_time_step)
    lead_time = pd.timedelta_range(step, periods=n_lead_times, freq=step)
    k_max = max(ensemble_sizes.values())
    rank = np.arange(1, k_max + 2)
    k = np.arange(1, domain_size // 2 + 1)
    wavelength = domain_size * dx_km / k

    n_s, n_i, n_l = len(sources), n_init_times, n_lead_times
    lt_frac = np.linspace(0, 1, n_l)

    # case-to-case variability (some days are rainier)
    case_scale = rng.lognormal(0.0, 0.4, size=(1, n_i, 1))
    source_skill = np.linspace(1.0, 1.3, n_s)[:, None, None]
    mae = 0.1 + 0.4 * lt_frac[None, None, :] * source_skill * case_scale
    crps = 0.6 * mae * (1 + 0.05 * rng.standard_normal((n_s, n_i, n_l)))

    # rank histograms: U-shape that grows with lead time, plus noise, normalised
    rank_hist = np.full((n_s, n_i, n_l, k_max + 1), np.nan)
    for i_s, src in enumerate(sources):
        n_ranks = ensemble_sizes[src] + 1
        x = np.linspace(-1, 1, n_ranks)
        u_depth = (0.2 + 0.8 * lt_frac) * (0.5 + 0.5 * i_s)
        shape = 1 + u_depth[:, None] * x[None, :] ** 2
        noisy = shape[None] * rng.lognormal(0, 0.1, size=(n_i, n_l, n_ranks))
        rank_hist[i_s, :, :, :n_ranks] = noisy / noisy.sum(axis=-1, keepdims=True)

    # spectra: power law in wavelength; forecasts lose small-scale power with lead time
    base = wavelength ** (5 / 3)
    psd_obs_case = (
        base[None, None, :]
        * case_scale[..., None]
        * rng.lognormal(0, 0.1, size=(1, n_i, n_l, len(k)))
    )
    psd_obs = np.broadcast_to(psd_obs_case, (n_s, n_i, n_l, len(k))).copy()
    cutoff_km = 2.0 + 30.0 * lt_frac[None, None, :, None] * source_skill[..., None]
    damping = 1.0 / (1.0 + (cutoff_km / wavelength[None, None, None, :]) ** 2)
    psd = psd_obs * damping

    case_dims = cv.CASE_DIMS
    return xr.Dataset(
        {
            cv.var_name(cv.MAE): (case_dims, mae, cv.METRIC_ATTRS[cv.MAE]),
            cv.var_name(cv.CRPS): (case_dims, crps, cv.METRIC_ATTRS[cv.CRPS]),
            cv.var_name(cv.RANK_HISTOGRAM): (
                (*case_dims, cv.RANK),
                rank_hist,
                cv.METRIC_ATTRS[cv.RANK_HISTOGRAM],
            ),
            cv.var_name(cv.PSD): ((*case_dims, cv.WAVELENGTH), psd, cv.METRIC_ATTRS[cv.PSD]),
            cv.var_name(cv.PSD_OBS): (
                (*case_dims, cv.WAVELENGTH),
                psd_obs,
                cv.METRIC_ATTRS[cv.PSD_OBS],
            ),
        },
        coords={
            cv.FORECAST_SOURCE: np.array(sources, dtype=str),
            cv.INIT_TIME: init_time.as_unit("ns").values,
            cv.LEAD_TIME: lead_time.as_unit("ns").values,
            cv.RANK: (cv.RANK, rank, cv.COORD_ATTRS[cv.RANK]),
            cv.WAVELENGTH: (cv.WAVELENGTH, wavelength, cv.COORD_ATTRS[cv.WAVELENGTH]),
        },
    )
