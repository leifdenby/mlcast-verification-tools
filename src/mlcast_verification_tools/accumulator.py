"""Collect per-case scores batch by batch and turn them into a spec-layout dataset."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
import xarray as xr

from . import conventions as cv
from .scores import crps_ensemble, ensemble_radial_psd, mae, psd_wavelengths, rank_histogram


class ScoreAccumulator:
    """
    Accumulate per-case scores of one forecast source.

    Parameters
    ----------
    forecast_source : str
        Name of the forecast source (value of the `forecast_source` coordinate).
    lead_times : array-like of timedelta
        Lead times, numpy ``timedelta64`` array or pandas ``TimedeltaIndex``; must have
        the length of the T dim of the forecasts passed to :meth:`update`.
    dx_km : float
        Grid spacing in km (for the radial PSD).
    zero_threshold : float
        Rain rate (mm/h) below which values count as zero in the rank histogram.
    seed : int
        Seed of the random generator used to break ties in the rank histogram.
    compute_psd : bool
        Whether to compute the radial power spectra.
    """

    def __init__(
        self,
        forecast_source: str,
        lead_times,
        dx_km: float,
        *,
        zero_threshold: float = cv.RANK_ZERO_THRESHOLD_MM_H,
        seed: int = 0,
        compute_psd: bool = True,
    ) -> None:
        self.forecast_source = str(forecast_source)
        self.lead_times = pd.to_timedelta(np.asarray(lead_times)).values.astype("timedelta64[ns]")
        self.dx_km = float(dx_km)
        self.zero_threshold = zero_threshold
        self.seed = seed
        self.compute_psd = compute_psd
        self._generators: dict[torch.device, torch.Generator] = {}
        self._init_times: list[np.ndarray] = []
        self._results: dict[str, list[np.ndarray]] = {
            cv.MAE: [],
            cv.CRPS: [],
            cv.RANK_HISTOGRAM: [],
            cv.PSD: [],
            cv.PSD_OBS: [],
        }
        self._n_members: int | None = None
        self._grid_shape: tuple[int, int] | None = None

    def _generator(self, device: torch.device) -> torch.Generator:
        if device not in self._generators:
            gen = torch.Generator(device=device)
            gen.manual_seed(self.seed)
            self._generators[device] = gen
        return self._generators[device]

    def update(self, fc: torch.Tensor, obs: torch.Tensor, mask: torch.Tensor, init_times) -> None:
        """
        Score a batch of forecast cases.

        Parameters
        ----------
        fc : torch.Tensor
            Forecast ``(B, T, M, H, W)`` in mm/h, on any device.
        obs : torch.Tensor
            Observation ``(B, T, H, W)`` in mm/h.
        mask : torch.Tensor
            Boolean ``(B, H, W)`` or ``(B, T, H, W)``, True = valid pixel.
        init_times : sequence of datetime-like
            Initialisation times of the B cases.
        """
        b, t, m, h, w = fc.shape
        init_times = pd.to_datetime(list(init_times)).values.astype("datetime64[ns]")
        if len(init_times) != b:
            raise ValueError(f"got {len(init_times)} init_times for a batch of {b} cases")
        if t != len(self.lead_times):
            raise ValueError(f"forecast has {t} lead times, expected {len(self.lead_times)}")
        if self._n_members is None:
            self._n_members, self._grid_shape = m, (h, w)
        elif self._n_members != m or self._grid_shape != (h, w):
            raise ValueError(
                f"inconsistent ensemble size/grid: got {m} members on {(h, w)}, "
                f"expected {self._n_members} members on {self._grid_shape}"
            )

        obs = obs.to(fc.device)
        mask = mask.to(fc.device)
        with torch.no_grad():
            out = {
                cv.MAE: mae(fc, obs, mask),
                cv.CRPS: crps_ensemble(fc, obs, mask),
                cv.RANK_HISTOGRAM: rank_histogram(
                    fc, obs, mask, self.zero_threshold, generator=self._generator(fc.device)
                ),
            }
            if self.compute_psd:
                out[cv.PSD], out[cv.PSD_OBS] = ensemble_radial_psd(fc, obs, mask, self.dx_km)
        for metric, value in out.items():
            self._results[metric].append(value.detach().cpu().double().numpy())
        self._init_times.append(init_times)

    def to_dataset(self) -> xr.Dataset:
        """
        Scores collected so far in the spec layout (see :mod:`.testing`).

        Returns
        -------
        xr.Dataset
            Dims ``(forecast_source, init_time, lead_time[, rank | wavelength])`` with
            ``forecast_source`` of length 1 and ``init_time`` sorted ascending.
        """
        if not self._init_times:
            raise ValueError("no forecast cases have been added")
        init_time = np.concatenate(self._init_times)
        if len(np.unique(init_time)) != len(init_time):
            raise ValueError(f"duplicate init_time in scores of {self.forecast_source!r}")
        order = np.argsort(init_time, kind="stable")

        def stacked(metric: str) -> np.ndarray:
            return np.concatenate(self._results[metric])[order][None].astype(np.float64)

        case_dims = cv.CASE_DIMS
        data_vars = {
            cv.var_name(cv.MAE): (case_dims, stacked(cv.MAE), cv.METRIC_ATTRS[cv.MAE]),
            cv.var_name(cv.CRPS): (case_dims, stacked(cv.CRPS), cv.METRIC_ATTRS[cv.CRPS]),
            cv.var_name(cv.RANK_HISTOGRAM): (
                (*case_dims, cv.RANK),
                stacked(cv.RANK_HISTOGRAM),
                cv.METRIC_ATTRS[cv.RANK_HISTOGRAM],
            ),
        }
        coords = {
            cv.FORECAST_SOURCE: np.array([self.forecast_source], dtype=str),
            cv.INIT_TIME: init_time[order],
            cv.LEAD_TIME: self.lead_times,
            cv.RANK: (cv.RANK, np.arange(1, self._n_members + 2), cv.COORD_ATTRS[cv.RANK]),
        }
        if self.compute_psd:
            for metric in (cv.PSD, cv.PSD_OBS):
                data_vars[cv.var_name(metric)] = (
                    (*case_dims, cv.WAVELENGTH),
                    stacked(metric),
                    cv.METRIC_ATTRS[metric],
                )
            wavelength = psd_wavelengths(*self._grid_shape, self.dx_km).numpy()
            coords[cv.WAVELENGTH] = (cv.WAVELENGTH, wavelength, cv.COORD_ATTRS[cv.WAVELENGTH])
        return xr.Dataset(data_vars, coords=coords)


def merge_sources(datasets: Sequence[xr.Dataset]) -> xr.Dataset:
    """
    Concatenate score datasets of different forecast sources.

    Uses an outer join: `rank` is NaN-padded to the largest ensemble size, and scores
    are NaN where init times, lead times or wavelengths of the sources do not overlap.

    Parameters
    ----------
    datasets : sequence of xr.Dataset
        Datasets in the spec layout, e.g. from :meth:`ScoreAccumulator.to_dataset`.

    Returns
    -------
    xr.Dataset
        Merged dataset with variable and coordinate attributes kept.
    """
    if not datasets:
        raise ValueError("no datasets to merge")
    sources = np.concatenate([ds[cv.FORECAST_SOURCE].values for ds in datasets])
    if len(np.unique(sources)) != len(sources):
        raise ValueError(f"duplicate forecast_source among datasets: {list(sources)}")
    merged = xr.concat(
        list(datasets),
        dim=cv.FORECAST_SOURCE,
        join="outer",
        data_vars="all",
        coords="different",
        compat="equals",
        combine_attrs="override",
        fill_value=np.nan,
    )
    merged = merged.assign_coords({cv.FORECAST_SOURCE: sources.astype(str)})
    if cv.RANK in merged.coords:
        merged = merged.assign_coords({cv.RANK: merged[cv.RANK].values.astype(int)})
    for name in merged.data_vars:
        merged[name].attrs = dict(next(d[name].attrs for d in datasets if name in d))
    for coord, attrs in cv.COORD_ATTRS.items():
        if coord in merged.coords:
            merged[coord].attrs = dict(attrs)
    return merged.transpose(*cv.CASE_DIMS, ...)
