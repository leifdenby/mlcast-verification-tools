"""Collect per-case scores batch by batch and turn them into a spec-layout dataset."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
import xarray as xr

from . import conventions as cv
from .scores import crps_ensemble, ensemble_radial_psd, mae, psd_wavelengths, rank_histogram
from .scores.deterministic import expand_mask


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
    variable : str
        Short name of the verified variable, used in the score names
        ``{metric}.{variable}``, e.g. ``"tp"`` or ``"dbz"``.
    units : str
        Units of the forecasts and observations (scores are computed in these units,
        without conversion), e.g. ``"mm h-1"`` or ``"dBZ"``.
    zero_threshold : float
        Value (in `units`) below which values count as zero in the rank histogram; see
        :func:`.units.default_zero_threshold`.
    seed : int
        Seed of the random generator used to break ties in the rank histogram.
    compute_psd : bool
        Whether to compute the radial power spectra.
    pool_by_init_time : bool
        If False (default) every case passed to :meth:`update` is kept and each
        init time may occur only once. If True, cases sharing an init time (e.g. many
        crops of one forecast, within and across batches) are pooled online into one
        entry per init time: MAE and CRPS are weighted by the number of valid pixels
        per lead time (giving the pixel mean over all pooled cases), the rank histogram
        is the pooled (pixel-count weighted) relative frequency, and the PSDs are the
        plain mean over the cases with valid pixels. Cases (and lead times) without
        valid pixels are ignored; scores are NaN where an init time has none.
    """

    def __init__(
        self,
        forecast_source: str,
        lead_times,
        dx_km: float,
        *,
        variable: str = cv.VARIABLE,
        units: str = "mm h-1",
        zero_threshold: float = cv.RANK_ZERO_THRESHOLD_MM_H,
        seed: int = 0,
        compute_psd: bool = True,
        pool_by_init_time: bool = False,
    ) -> None:
        self.forecast_source = str(forecast_source)
        self.lead_times = pd.to_timedelta(np.asarray(lead_times)).values.astype("timedelta64[ns]")
        self.dx_km = float(dx_km)
        self.variable = str(variable)
        self.units = str(units)
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
        self.pool_by_init_time = pool_by_init_time
        # pooling state: slot index per init time and running sums of shape (slot, T, ...)
        self._slots: dict[np.datetime64, int] = {}
        self._sums: dict[str, np.ndarray] = {}

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
            Forecast ``(B, T, M, H, W)`` in `units`, on any device.
        obs : torch.Tensor
            Observation ``(B, T, H, W)`` in `units`.
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
            if self.pool_by_init_time:
                n_valid = expand_mask(mask, obs).sum(dim=(-2, -1))
        out = {metric: value.detach().cpu().double().numpy() for metric, value in out.items()}
        if self.pool_by_init_time:
            self._pool(init_times, out, n_valid.cpu().double().numpy())
            return
        for metric, value in out.items():
            self._results[metric].append(value)
        self._init_times.append(init_times)

    def _pool(self, init_times: np.ndarray, out: dict[str, np.ndarray], n: np.ndarray) -> None:
        """Add the per-case scores `out` (B, T[, ...]) with valid-pixel counts `n` (B, T)."""
        unique, inverse = np.unique(init_times, return_inverse=True)
        for it in unique:
            self._slots.setdefault(it, len(self._slots))
        slot = np.array([self._slots[it] for it in unique])[inverse.reshape(-1)]
        valid = n > 0
        increments = {
            "n": n,
            cv.MAE: np.where(valid, out[cv.MAE] * n, 0.0),
            cv.CRPS: np.where(valid, out[cv.CRPS] * n, 0.0),
            cv.RANK_HISTOGRAM: np.where(
                valid[..., None], out[cv.RANK_HISTOGRAM] * n[..., None], 0.0
            ),
        }
        if self.compute_psd:
            increments["n_psd"] = valid.astype(np.float64)
            for metric in (cv.PSD, cv.PSD_OBS):
                increments[metric] = np.where(valid[..., None], out[metric], 0.0)
        n_slots = len(self._slots)
        for key, value in increments.items():
            sums = self._sums.get(key, np.zeros((0, *value.shape[1:])))
            if len(sums) < n_slots:  # grow geometrically to keep appends amortised O(1)
                grown = np.zeros((max(n_slots, 2 * len(sums)), *value.shape[1:]))
                grown[: len(sums)] = sums
                sums = self._sums[key] = grown
            np.add.at(sums, slot, value)

    def _pooled_results(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Init times and per-init-time scores (N, T[, ...]) from the pooled sums."""
        n_slots = len(self._slots)
        init_time = np.array(list(self._slots), dtype="datetime64[ns]")
        sums = {key: value[:n_slots] for key, value in self._sums.items()}

        def ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
            return np.divide(num, den, out=np.full(num.shape, np.nan), where=den > 0)

        n = sums["n"]
        results = {
            cv.MAE: ratio(sums[cv.MAE], n),
            cv.CRPS: ratio(sums[cv.CRPS], n),
            cv.RANK_HISTOGRAM: ratio(sums[cv.RANK_HISTOGRAM], n[..., None]),
        }
        if self.compute_psd:
            for metric in (cv.PSD, cv.PSD_OBS):
                results[metric] = ratio(sums[metric], sums["n_psd"][..., None])
        return init_time, results

    def to_dataset(self) -> xr.Dataset:
        """
        Scores collected so far in the spec layout (see :mod:`.testing`).

        Returns
        -------
        xr.Dataset
            Dims ``(forecast_source, init_time, lead_time[, rank | wavelength])`` with
            ``forecast_source`` of length 1 and ``init_time`` sorted ascending (one
            entry per init time when pooling by init time).
        """
        if self.pool_by_init_time:
            if not self._slots:
                raise ValueError("no forecast cases have been added")
            init_time, results = self._pooled_results()
        else:
            if not self._init_times:
                raise ValueError("no forecast cases have been added")
            init_time = np.concatenate(self._init_times)
            if len(np.unique(init_time)) != len(init_time):
                raise ValueError(f"duplicate init_time in scores of {self.forecast_source!r}")
            results = {
                metric: np.concatenate(values) for metric, values in self._results.items() if values
            }
        order = np.argsort(init_time, kind="stable")

        def stacked(metric: str) -> np.ndarray:
            return results[metric][order][None].astype(np.float64)

        case_dims = cv.CASE_DIMS
        attrs = cv.metric_attrs(self.units)

        def name(metric: str) -> str:
            return cv.var_name(metric, self.variable)

        data_vars = {
            name(cv.MAE): (case_dims, stacked(cv.MAE), attrs[cv.MAE]),
            name(cv.CRPS): (case_dims, stacked(cv.CRPS), attrs[cv.CRPS]),
            name(cv.RANK_HISTOGRAM): (
                (*case_dims, cv.RANK),
                stacked(cv.RANK_HISTOGRAM),
                attrs[cv.RANK_HISTOGRAM],
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
                data_vars[name(metric)] = (
                    (*case_dims, cv.WAVELENGTH),
                    stacked(metric),
                    attrs[metric],
                )
            wavelength = psd_wavelengths(*self._grid_shape, self.dx_km).numpy()
            coords[cv.WAVELENGTH] = (cv.WAVELENGTH, wavelength, cv.COORD_ATTRS[cv.WAVELENGTH])
        return xr.Dataset(data_vars, coords=coords)


def merge_sources(datasets: Sequence[xr.Dataset]) -> xr.Dataset:
    """
    Concatenate score datasets of different forecast sources.

    Uses an outer join: `rank` is NaN-padded to the largest ensemble size, and scores
    are NaN where init times, lead times or wavelengths of the sources do not overlap.
    Scores of the same variable must have the same units in every dataset.

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
    for name in {name for ds in datasets for name in ds.data_vars}:
        units = {ds[name].attrs.get("units") for ds in datasets if name in ds}
        if len(units) > 1:
            raise ValueError(
                f"{name!r} has different units in the datasets: {sorted(map(str, units))}"
            )
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
