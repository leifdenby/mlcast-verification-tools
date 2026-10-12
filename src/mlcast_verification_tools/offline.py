"""Score forecasts stored as xarray DataArrays against gridded observations."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
import xarray as xr

from . import conventions as cv
from .accumulator import ScoreAccumulator, merge_sources

FC_DIMS = (cv.INIT_TIME, cv.LEAD_TIME, "member", "y", "x")
OBS_DIMS = ("time", "y", "x")


def infer_dx_km(da: xr.DataArray) -> float:
    """
    Grid spacing in km from the `x` coordinate.

    The spacing is taken to be in metres if it exceeds 100, otherwise in km.

    Parameters
    ----------
    da : xr.DataArray
        Array with a numeric `x` coordinate.

    Returns
    -------
    float
        Grid spacing in km.
    """
    if "x" not in da.coords or da.sizes["x"] < 2:
        raise ValueError("cannot infer dx_km: need an `x` coordinate with at least 2 points")
    dx = float(np.median(np.abs(np.diff(da["x"].values.astype(float)))))
    return dx / 1000.0 if dx > 100 else dx


def compute_scores(
    forecasts: Mapping[str, xr.DataArray],
    obs: xr.DataArray,
    *,
    dx_km: float | None = None,
    device: str | None = None,
    batch_size: int = 1,
    seed: int = 0,
) -> xr.Dataset:
    """
    Compute per-case scores of several forecast sources.

    Parameters
    ----------
    forecasts : mapping of str to xr.DataArray
        Forecasts per source, dims ``(init_time, lead_time, member, y, x)`` in mm/h.
    obs : xr.DataArray
        Observations, dims ``(time, y, x)`` in mm/h on the forecast grid; NaN = missing.
        Each case is verified against ``obs`` at valid time ``init_time + lead_time``
        (all-NaN if that time is missing).
    dx_km : float, optional
        Grid spacing in km. If None, inferred from the spacing of the `x` coordinate
        of ``obs``: metres if the spacing is larger than 100, else km.
    device : str, optional
        Torch device; defaults to "cuda" if available, else "cpu".
    batch_size : int
        Number of init times scored at once.
    seed : int
        Seed for the random tie-breaking in the rank histogram.

    Returns
    -------
    xr.Dataset
        Scores of all sources in the spec layout (see :func:`.accumulator.merge_sources`).

    Notes
    -----
    A pixel is valid for a case (per lead time) where the observation is finite at the
    valid time and the forecast is finite for all members.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dx_km is None:
        dx_km = infer_dx_km(obs)
    obs = obs.transpose(*OBS_DIMS)

    datasets = []
    for source, fc_da in forecasts.items():
        fc_da = fc_da.transpose(*FC_DIMS)
        if fc_da.shape[-2:] != obs.shape[-2:]:
            raise ValueError(
                f"grid of {source!r} {fc_da.shape[-2:]} differs from obs grid {obs.shape[-2:]}"
            )
        lead_times = fc_da[cv.LEAD_TIME].values
        acc = ScoreAccumulator(source, lead_times, dx_km, seed=seed)
        n_init = fc_da.sizes[cv.INIT_TIME]
        for start in range(0, n_init, batch_size):
            batch = fc_da.isel({cv.INIT_TIME: slice(start, start + batch_size)})
            init_times = batch[cv.INIT_TIME].values
            fc_np = np.asarray(batch.values, dtype=np.float32)
            valid_times = (init_times[:, None] + lead_times[None, :]).ravel()
            obs_np = np.asarray(obs.reindex(time=valid_times).values, dtype=np.float32)
            obs_np = obs_np.reshape(len(init_times), len(lead_times), *obs_np.shape[-2:])
            mask_np = np.isfinite(obs_np) & np.isfinite(fc_np).all(axis=2)

            acc.update(
                torch.from_numpy(fc_np).to(device),
                torch.from_numpy(obs_np).to(device),
                torch.from_numpy(mask_np).to(device),
                init_times,
            )
        datasets.append(acc.to_dataset())
    return merge_sources(datasets)
