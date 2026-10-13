"""
Score forecasts stored as an xarray Dataset against a gridded truth Dataset.

Input layouts
-------------
``ds_forecasts``
    An ``xr.Dataset`` with the data variable ``variable`` (default ``"tp"``) of dims
    ``(forecast_source, init_time, lead_time, member, y, x)``
    (:data:`.conventions.FORECAST_DIMS`, in any order):

    - ``init_time`` is datetime-like, ``lead_time`` timedelta-like;
    - ``forecast_source`` (string labels) is optional: data of a single source may
      instead carry a scalar ``forecast_source`` coordinate, or the name is given with
      the ``forecast_source=`` argument of :func:`compute_scores`;
    - sources with fewer members are NaN-padded along ``member``; members that are
      all-NaN for a source (checked on its first batch of init times with data) are
      dropped before scoring, so each source is scored with its own ensemble size;
    - init times at which a source is all-NaN (e.g. padding after concatenating sources
      with different init times) are skipped for that source, so its scores there are
      NaN after merging;
    - the variable (or the dataset) must have a ``units`` attribute, e.g. ``"mm h-1"``,
      ``"kg m-2 s-1"`` or ``"dBZ"``.

``ds_truth``
    An ``xr.Dataset`` with the same data variable of dims ``(time, y, x)``
    (:data:`.conventions.TRUTH_DIMS`) on the forecast grid, with the same ``units``
    attribute as the forecasts; NaN marks invalid pixels. ``time`` must be unique; valid times
    ``init_time + lead_time`` missing from ``time`` are treated as all-NaN.

Both datasets should keep their ``x``/``y`` projection coordinates (metres or km): the
grid spacing ``dx_km`` is inferred from ``x`` (:func:`infer_dx_km`), and when both
datasets have them the coordinates must agree. Data is loaded one batch of init times
at a time, so dask- or Zarr-backed datasets work without loading everything.

Scores are computed in the units of the data, without conversion, and named
``{metric}.{variable}``.
"""

from __future__ import annotations

import numpy as np
import torch
import xarray as xr

from . import conventions as cv
from .accumulator import ScoreAccumulator, merge_sources
from .units import default_zero_threshold


def infer_dx_km(da: xr.DataArray | xr.Dataset) -> float:
    """
    Grid spacing in km from the `x` coordinate.

    The spacing is taken to be in metres if it exceeds 100, otherwise in km.

    Parameters
    ----------
    da : xr.DataArray or xr.Dataset
        Array or dataset with a numeric `x` coordinate.

    Returns
    -------
    float
        Grid spacing in km.
    """
    if cv.X not in da.coords or da.sizes.get(cv.X, 0) < 2:
        raise ValueError("cannot infer dx_km: need an `x` coordinate with at least 2 points")
    dx = float(np.median(np.abs(np.diff(da[cv.X].values.astype(float)))))
    return dx / 1000.0 if dx > 100 else dx


def _get_variable(ds: xr.Dataset, variable: str, what: str) -> tuple[xr.DataArray, str]:
    """Data variable `variable` of `ds` and its units."""
    if not isinstance(ds, xr.Dataset):
        raise TypeError(f"{what} must be an xr.Dataset, got {type(ds).__name__}")
    if variable not in ds.data_vars:
        raise ValueError(f"{what} has no data variable {variable!r} (has {list(ds.data_vars)})")
    da = ds[variable]
    units = da.attrs.get("units", ds.attrs.get("units"))
    if units is None:
        raise ValueError(f"{what}[{variable!r}] has no `units` attribute")
    return da, str(units)


def _same_units(a: str, b: str) -> bool:
    return a.strip().lower().replace(" ", "") == b.strip().lower().replace(" ", "")


def _split_sources(fc: xr.DataArray, forecast_source: str | None) -> list[tuple[str, xr.DataArray]]:
    """Split forecasts into (name, DataArray without forecast_source dim) per source."""
    if cv.FORECAST_SOURCE in fc.dims:
        if forecast_source is not None:
            raise ValueError(
                "`forecast_source=` is only for forecasts without a `forecast_source` dim"
            )
        names = [str(s) for s in fc[cv.FORECAST_SOURCE].values]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate forecast_source labels: {names}")
        return [(name, fc.isel({cv.FORECAST_SOURCE: i})) for i, name in enumerate(names)]
    if forecast_source is None:
        if cv.FORECAST_SOURCE not in fc.coords:
            raise ValueError(
                "forecasts need a `forecast_source` dim, a scalar `forecast_source` "
                "coordinate or the `forecast_source=` argument"
            )
        forecast_source = str(fc[cv.FORECAST_SOURCE].values)
    return [(str(forecast_source), fc)]


def _check_grid(fc: xr.DataArray, truth: xr.DataArray) -> None:
    for dim in (cv.Y, cv.X):
        if fc.sizes[dim] != truth.sizes[dim]:
            raise ValueError(
                f"forecast grid {fc.shape[-2:]} differs from truth grid {truth.shape[-2:]}"
            )
        if dim in fc.coords and dim in truth.coords:
            a = np.asarray(fc[dim].values, dtype=float)
            b = np.asarray(truth[dim].values, dtype=float)
            if not np.allclose(a, b):
                raise ValueError(f"`{dim}` coordinates of forecasts and truth differ")


def compute_scores(
    ds_forecasts: xr.Dataset,
    ds_truth: xr.Dataset,
    *,
    variable: str = cv.VARIABLE,
    zero_threshold: float | None = None,
    forecast_source: str | None = None,
    dx_km: float | None = None,
    device: str | None = None,
    batch_size: int = 1,
    seed: int = 0,
) -> xr.Dataset:
    """
    Compute per-case scores of one or more forecast sources.

    See the module docstring for the layouts of ``ds_forecasts`` and ``ds_truth``.

    Parameters
    ----------
    ds_forecasts : xr.Dataset
        Forecasts, variable ``variable`` with dims
        ``([forecast_source,] init_time, lead_time, member, y, x)`` and a ``units`` attr.
    ds_truth : xr.Dataset
        Truth, variable ``variable`` with dims ``(time, y, x)`` on the forecast grid and
        the same ``units`` attr as the forecasts; NaN = invalid. Each case is verified
        against the truth at valid time ``init_time + lead_time`` (all-NaN if that time
        is missing).
    variable : str
        Name of the data variable in both datasets, also used in the score names
        ``{metric}.{variable}``.
    zero_threshold : float, optional
        Value (in the data units) below which values count as zero in the rank
        histogram. Default: :func:`.units.default_zero_threshold` of the units.
    forecast_source : str, optional
        Name of the source if ``ds_forecasts`` has no ``forecast_source`` dim; overrides
        a scalar ``forecast_source`` coordinate.
    dx_km : float, optional
        Grid spacing in km. If None, inferred from the `x` coordinate of ``ds_truth``
        (or of ``ds_forecasts``): metres if the spacing is larger than 100, else km.
    device : str, optional
        Torch device; defaults to "cuda" if available, else "cpu".
    batch_size : int
        Number of init times loaded and scored at once.
    seed : int
        Seed for the random tie-breaking in the rank histogram.

    Returns
    -------
    xr.Dataset
        Scores of all sources in the spec layout (see :func:`.accumulator.merge_sources`),
        in the units of the data.

    Notes
    -----
    A pixel is valid for a case (per lead time) where the truth is finite at the valid
    time and the forecast is finite for all (non-padding) members.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    fc_all, units = _get_variable(ds_forecasts, variable, "ds_forecasts")
    truth, truth_units = _get_variable(ds_truth, variable, "ds_truth")
    if not _same_units(units, truth_units):
        raise ValueError(
            f"forecasts and truth have different units ({units!r}, {truth_units!r}); "
            "scores are computed without unit conversion"
        )
    if zero_threshold is None:
        zero_threshold = default_zero_threshold(units)

    if set(truth.dims) != set(cv.TRUTH_DIMS):
        raise ValueError(f"ds_truth[{variable!r}] must have dims {cv.TRUTH_DIMS}, got {truth.dims}")
    truth = truth.transpose(*cv.TRUTH_DIMS)
    if dx_km is None:
        dx_km = infer_dx_km(ds_truth if cv.X in ds_truth.coords else ds_forecasts)

    fc_dims = cv.FORECAST_DIMS[1:]
    datasets = []
    for source, fc in _split_sources(fc_all, forecast_source):
        if set(fc.dims) != set(fc_dims):
            raise ValueError(
                f"forecasts of {source!r} must have dims {cv.FORECAST_DIMS} "
                f"(forecast_source optional), got {fc.dims}"
            )
        fc = fc.transpose(*fc_dims)
        _check_grid(fc, truth)
        datasets.append(
            _score_source(
                fc,
                truth,
                ScoreAccumulator(
                    source,
                    fc[cv.LEAD_TIME].values,
                    dx_km,
                    variable=variable,
                    units=units,
                    zero_threshold=zero_threshold,
                    seed=seed,
                ),
                device,
                batch_size,
            )
        )
    return merge_sources(datasets)


def _score_source(
    fc: xr.DataArray,
    truth: xr.DataArray,
    acc: ScoreAccumulator,
    device: str,
    batch_size: int,
) -> xr.Dataset:
    """Scores of one source, fc ``(init_time, lead_time, member, y, x)``."""
    source = acc.forecast_source
    lead_times = fc[cv.LEAD_TIME].values
    members = None  # indices of the non-padding members, from the first batch with data
    for start in range(0, fc.sizes[cv.INIT_TIME], batch_size):
        batch = fc.isel({cv.INIT_TIME: slice(start, start + batch_size)})
        fc_np = np.asarray(batch.values, dtype=np.float32)
        finite = np.isfinite(fc_np)
        has_data = finite.any(axis=(1, 2, 3, 4))  # all-NaN cases: padding, skipped
        if not has_data.any():
            continue
        init_times = batch[cv.INIT_TIME].values[has_data]
        fc_np, finite = fc_np[has_data], finite[has_data]
        finite_members = finite.any(axis=(0, 1, 3, 4))
        if members is None:
            members = np.flatnonzero(finite_members)
        elif finite_members[np.setdiff1d(np.arange(len(finite_members)), members)].any():
            raise ValueError(
                f"members of {source!r} that are all-NaN in the first batch of init times "
                "(taken as padding) have values in later init times"
            )
        if len(members) < fc_np.shape[2]:
            fc_np = fc_np[:, :, members]

        valid_times = (init_times[:, None] + lead_times[None, :]).ravel()
        obs_np = np.asarray(truth.reindex({cv.TIME: valid_times}).values, dtype=np.float32)
        obs_np = obs_np.reshape(len(init_times), len(lead_times), *obs_np.shape[-2:])
        mask_np = np.isfinite(obs_np) & np.isfinite(fc_np).all(axis=2)

        acc.update(
            torch.from_numpy(fc_np).to(device),
            torch.from_numpy(obs_np).to(device),
            torch.from_numpy(mask_np).to(device),
            init_times,
        )
    if members is None:
        raise ValueError(f"forecasts of {source!r} are all-NaN")
    return acc.to_dataset()
