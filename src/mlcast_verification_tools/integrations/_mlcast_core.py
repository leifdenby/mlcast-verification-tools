"""Torch/mlcast implementation of :mod:`mlcast_verification_tools.integrations.mlcast`.

Kept separate so that importing the public module (done by the CLI for every
sub-command) does not import torch or mlcast. Import from the public module.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import datetime
import math
import os
import pathlib
import time
import warnings
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import pandas as pd
import torch
import xarray as xr
from loguru import logger
from mlcast.data.normalization import DEFAULT_SCALING, ReflectivityScaling
from torch.utils.data import DataLoader, Dataset

from .. import conventions as cv
from ..accumulator import ScoreAccumulator
from ..units import default_zero_threshold, mm_per_hour_factor  # noqa: F401 - re-exported
from .mlcast import PRECISIONS, SUPPORTED_STANDARD_NAMES

if TYPE_CHECKING:  # pragma: no cover
    import fiddle as fdl
    from mlcast.nowcasting_module import NowcastLightningModule

# training-time path of a TensorBoard run is <log_dir>/checkpoints/<name>.ckpt with the
# Fiddle config written to <log_dir>/config.yaml (mlcast.config.orchestrator)
_CONFIG_YAML_NAME = "config.yaml"


# ---------------------------------------------------------------------------
# Units and (de)normalisation
# ---------------------------------------------------------------------------


_REFLECTIVITY = "equivalent_reflectivity_factor"

_LONG_NAMES = {
    "rainfall_rate": "rainfall rate",
    "rainfall_flux": "rainfall flux",
    _REFLECTIVITY: "equivalent reflectivity factor",
}


def _check_standard_name(standard_name: str) -> None:
    if standard_name not in SUPPORTED_STANDARD_NAMES:
        raise ValueError(
            f"standard_name {standard_name!r} is not supported, expected one of "
            f"{SUPPORTED_STANDARD_NAMES}"
        )


def denormalize_torch(
    x: torch.Tensor,
    standard_name: str = "rainfall_rate",
    scaling: ReflectivityScaling | None = None,
) -> torch.Tensor:
    """
    Torch port of mlcast's de-normalisation (normalised -> units of the source variable).

    Maps the normalised range ``[norm_min, norm_max]`` linearly onto
    ``[dbz_floor, dbz_ceiling]`` dBZ, as ``ReflectivityScaling.denormalize_reflectivity``.
    For reflectivities that is the result. For rain rates the inverse Marshall-Palmer
    relation ``R = 10 ** ((dBZ - 10 log10(a)) / (10 b))`` follows; values at or below
    ``dbz_floor`` become exactly 0, as in
    ``ReflectivityScaling.normalized_to_rainfall_rate``. NaN propagates. Runs on the
    device of `x`.

    Parameters
    ----------
    x : torch.Tensor
        Normalised values (floating point), any shape and device.
    standard_name : {"rainfall_rate", "rainfall_flux", "equivalent_reflectivity_factor"}
        CF standard name of the variable the model was trained on. Both rain rates use
        the same maths (mlcast reuses the mm/h relation for ``rainfall_flux``), so the
        result is in the units of the source variable.
    scaling : mlcast.data.normalization.ReflectivityScaling, optional
        Scaling constants; defaults to mlcast's ``DEFAULT_SCALING``.

    Returns
    -------
    torch.Tensor
        Values in the units of the source variable (dBZ for reflectivities), same
        shape/dtype/device as `x`.
    """
    _check_standard_name(standard_name)
    if not torch.is_floating_point(x):
        raise TypeError(f"expected a floating point tensor, got {x.dtype}")
    s = scaling or DEFAULT_SCALING
    dbz = s.dbz_floor + (x - s.norm_min) * (
        (s.dbz_ceiling - s.dbz_floor) / (s.norm_max - s.norm_min)
    )
    if standard_name == _REFLECTIVITY:
        return dbz
    rate = torch.pow(10.0, (dbz - 10.0 * math.log10(s.a)) / (10.0 * s.b))
    return torch.where(
        dbz <= s.dbz_floor, torch.zeros((), dtype=rate.dtype, device=rate.device), rate
    )


def _truth_values(values, standard_name: str, scaling: ReflectivityScaling):
    """
    Observed values as scored: reflectivities clipped to ``[dbz_floor, dbz_ceiling]``.

    The model can only predict that range (mlcast clips its inputs to it too), so
    values outside it, e.g. a "no echo" value far below the floor, are clipped. Rain
    rates are returned unchanged. NaN is kept. Works on numpy, dask and xarray arrays.
    """
    if standard_name == _REFLECTIVITY:
        return values.clip(scaling.dbz_floor, scaling.dbz_ceiling)
    return values


def _variable_attrs(standard_name: str, units: str) -> dict[str, str]:
    """Attributes of a forecast/truth variable in the units of the source."""
    return {
        "units": units,
        "long_name": _LONG_NAMES.get(standard_name, standard_name.replace("_", " ")),
        "standard_name": standard_name,
    }


def _normalize_numpy(
    values: np.ndarray, standard_name: str, scaling: ReflectivityScaling
) -> np.ndarray:
    """mlcast's forward normalisation, without its per-call ``rainfall_flux`` warning."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Assuming we can use the same function")
        return scaling.normalization_registry[standard_name](values)


# ---------------------------------------------------------------------------
# Run configs, test splits and checkpoints
# ---------------------------------------------------------------------------


def find_config_yaml(checkpoint: str | os.PathLike) -> pathlib.Path | None:
    """
    Find the Fiddle ``config.yaml`` of the mlcast run a checkpoint belongs to.

    mlcast's TensorBoard layout is ``<run>/checkpoints/<name>.ckpt`` with the config
    written to ``<run>/config.yaml``; a ``config.yaml`` next to the checkpoint is
    also accepted.

    Parameters
    ----------
    checkpoint : str or os.PathLike
        Lightning ``.ckpt`` file.

    Returns
    -------
    pathlib.Path or None
        The config file, or None if there is none in either place.
    """
    checkpoint = pathlib.Path(checkpoint)
    for directory in (checkpoint.parent, checkpoint.parent.parent):
        candidate = directory / _CONFIG_YAML_NAME
        if candidate.is_file():
            return candidate
    return None


def _is_url(path: str) -> bool:
    return "://" in path


def _resolve_path(path: str | os.PathLike, root: pathlib.Path) -> str:
    """Resolve a relative local path against `root`; URLs and absolute paths are kept."""
    text = os.fspath(path)
    if _is_url(text) or pathlib.Path(text).is_absolute():
        return text
    return str((root / text).resolve())


def _tuplify_splits(splits: Any) -> Any:
    """``(start, end)`` split ranges come back from YAML as lists; mlcast wants tuples."""
    if not isinstance(splits, dict):
        return splits
    return {
        coord: (
            {
                name: tuple(value) if isinstance(value, list) else value
                for name, value in coord_splits.items()
            }
            if isinstance(coord_splits, dict)
            else coord_splits
        )
        for coord, coord_splits in splits.items()
    }


def load_run_config(
    config_yaml: str | os.PathLike,
    *,
    data_root: str | os.PathLike | None = None,
    zarr_path: str | os.PathLike | None = None,
    index_path: str | os.PathLike | None = None,
    storage_options: dict[str, Any] | None = None,
) -> fdl.Config:
    """
    Load the Fiddle config of an mlcast training run, ready to rebuild its data and model.

    Uses mlcast's ``mlcast.config.load_yaml_config``, then:

    - turns ``(start, end)`` split ranges, which YAML stores as lists, back into the
      tuples mlcast's split validation expects;
    - applies the given overrides of the dataset factory's ``zarr_path``,
      ``index_path`` and ``storage_options``;
    - resolves relative ``zarr_path``/``index_path`` (mlcast's defaults are relative,
      e.g. ``./data/radar.zarr``) against `data_root`.

    .. warning:: Loading imports every module named in the YAML, so only load
       configs you trust.

    Parameters
    ----------
    config_yaml : str or os.PathLike
        The run's ``config.yaml``, as written by mlcast at the start of training.
    data_root : str or os.PathLike, optional
        Directory that relative data paths are resolved against, typically the
        directory training was started from. Default: the current directory.
    zarr_path, index_path : str or os.PathLike, optional
        Replace the source-data Zarr / sampling index of the config (e.g. when the
        data has moved since training).
    storage_options : dict, optional
        Replace the ``storage_options`` used to open the Zarr.

    Returns
    -------
    fdl.Config
        The experiment config (not built), with ``pl_module``, ``data`` and ``trainer``.
    """
    from mlcast.config import load_yaml_config

    cfg = load_yaml_config(config_yaml)
    try:
        data = cfg.data
        factory = data.dataset_factory
    except AttributeError as exc:
        raise ValueError(
            f"{config_yaml} is not an mlcast experiment config (no data.dataset_factory)"
        ) from exc
    data.splits = _tuplify_splits(data.splits)
    _apply_data_overrides(
        factory,
        data_root=data_root,
        zarr_path=zarr_path,
        index_path=index_path,
        storage_options=storage_options,
    )
    return cfg


def _apply_data_overrides(
    factory: Any,
    *,
    data_root: str | os.PathLike | None = None,
    zarr_path: str | os.PathLike | None = None,
    index_path: str | os.PathLike | None = None,
    storage_options: dict[str, Any] | None = None,
) -> None:
    """Override and resolve the data paths of a dataset factory config (in place)."""
    if zarr_path is not None:
        factory.zarr_path = os.fspath(zarr_path)
    if index_path is not None:
        factory.index_path = os.fspath(index_path)
    if storage_options is not None:
        factory.storage_options = storage_options
    root = pathlib.Path(data_root) if data_root is not None else pathlib.Path.cwd()
    for name in ("zarr_path", "index_path"):
        value = getattr(factory, name, None)
        if value is not None:
            setattr(factory, name, _resolve_path(value, root))


def _as_run_config(config: fdl.Config | str | os.PathLike) -> fdl.Config:
    if isinstance(config, str | os.PathLike):
        return load_run_config(config)
    return config


def _resolve_config(
    config: fdl.Config | str | os.PathLike,
    *,
    data_root: str | os.PathLike | None = None,
    zarr_path: str | os.PathLike | None = None,
    index_path: str | os.PathLike | None = None,
    storage_options: dict[str, Any] | None = None,
) -> tuple[fdl.Config, str]:
    """
    The run config (loaded if a path) with the data overrides applied, and a label.

    A loaded config is copied before the overrides are applied, so the caller's
    config is never modified. The label is the path of the ``config.yaml``, or
    ``"the given run config"`` for a loaded config.
    """
    overrides = {
        "data_root": data_root,
        "zarr_path": zarr_path,
        "index_path": index_path,
        "storage_options": storage_options,
    }
    if config is None:
        raise ValueError(
            "the run's config (an fdl.Config or the path of its config.yaml) is needed"
        )
    if isinstance(config, str | os.PathLike):
        return load_run_config(config, **overrides), os.fspath(config)
    try:
        config.data.dataset_factory  # noqa: B018 - checks the structure
    except AttributeError as exc:
        raise ValueError("not an mlcast experiment config (no data.dataset_factory)") from exc
    if any(value is not None for value in overrides.values()):
        config = copy.deepcopy(config)
        _apply_data_overrides(config.data.dataset_factory, **overrides)
    return config, "the given run config"


def _single_standard_name(config: fdl.Config) -> str:
    standard_names = list(config.data.dataset_factory.standard_names)
    if len(standard_names) != 1:
        raise ValueError(
            f"only single-variable runs are supported, the config has {standard_names}"
        )
    return standard_names[0]


def split_init_times(
    config: fdl.Config | str | os.PathLike, split: str = "test"
) -> pd.DatetimeIndex:
    """
    Init times of the crops mlcast's sampler selects for a split of a training run.

    Builds the run's data module (only the Zarr metadata and the sampling index are
    read), lets mlcast set up the split, and maps each selected crop to an init time.
    A crop is the start ``t`` of a ``time_depth``-step window, from which mlcast
    takes ``input_steps + forecast_steps`` frames at a random offset; here the offset
    is 0, so the inputs are frames ``t .. t + input_steps - 1`` and the init time is
    the time of frame ``t + input_steps - 1``. Crops at the same time but other
    positions give the same init time; the verification forecasts the full domain.

    Parameters
    ----------
    config : fdl.Config or str or os.PathLike
        Run config from :func:`load_run_config`, or the path of its ``config.yaml``.
    split : {"test", "val"}
        Which split.

    Returns
    -------
    pd.DatetimeIndex
        Sorted, unique init times.
    """
    import fiddle as fdl

    stages = {"test": "test", "val": "validate"}
    if split not in stages:
        raise ValueError(f"split must be one of {sorted(stages)}, got {split!r}")
    config = _as_run_config(config)
    dm = fdl.build(config.data)
    dm.setup(stages[split])
    dataset = getattr(dm, f"{split}_dataset", None)
    if dataset is None:
        raise ValueError(f"the run config defines no {split!r} split")
    candidates = getattr(dataset, "candidates", None)
    if candidates is None:
        raise ValueError(
            f"{type(dataset).__name__} has no sampling index (candidates); only "
            "SourceDataIndexedDataset runs define a reproducible test split"
        )
    # mlcast rebases `t` onto the split's time slice; `_time_index_slice` is private
    # (mlcast.data.source_data_datasets.SourceDataIndexedDataset), so its use is kept here
    time_slice = getattr(dataset, "_time_index_slice", None)
    offset = 0 if time_slice is None or time_slice.start is None else int(time_slice.start)
    t_abs = candidates["t"].to_numpy(dtype=np.int64) + offset
    t_init = np.unique(t_abs + int(dataset.input_steps) - 1)

    ds = xr.open_zarr(dataset._zarr_path, storage_options=dataset.storage_options)
    times = pd.DatetimeIndex(ds.indexes[dataset.t_dim])
    if len(t_init) and t_init[-1] >= len(times):
        raise ValueError("sampling index refers to times beyond the source Zarr")
    init_times = pd.DatetimeIndex(times[t_init]).unique().sort_values()
    logger.info(
        "{} split: {} sampled crops at {} unique init times",
        split,
        len(candidates),
        len(init_times),
    )
    return init_times


def load_mlcast_model(
    checkpoint_path: str | os.PathLike,
    *,
    config: fdl.Config | str | os.PathLike | None,
    network: torch.nn.Module | None = None,
    device: str | torch.device = "cpu",
) -> NowcastLightningModule:
    """
    Load a trained ``NowcastLightningModule`` from a Lightning checkpoint.

    The network architecture is not stored in mlcast checkpoints, so it is rebuilt
    from ``pl_module.network`` of the run config `config`, unless `network` is given.
    The config is never looked up implicitly; :func:`find_config_yaml` locates the
    ``config.yaml`` of a run if needed.

    Parameters
    ----------
    checkpoint_path : str or os.PathLike
        Lightning ``.ckpt`` file.
    config : fdl.Config or str or os.PathLike or None
        The run config (:func:`load_run_config`) or the path of its ``config.yaml``.
        May only be None if `network` is given.
    network : torch.nn.Module, optional
        Network with the architecture used in training (weights are overwritten);
        takes precedence over `config`.
    device : str or torch.device
        Device to load the weights to.

    Returns
    -------
    NowcastLightningModule
        Model in eval mode on `device`.
    """
    import fiddle as fdl
    from mlcast.config import load_yaml_config
    from mlcast.nowcasting_module import NowcastLightningModule

    checkpoint_path = pathlib.Path(checkpoint_path)
    if network is None:
        if config is None:
            raise ValueError(
                f"cannot restore {checkpoint_path}: the network is not stored in mlcast "
                "checkpoints. Pass `config=` (the run's config) or `network=`."
            )
        if isinstance(config, str | os.PathLike):
            config = load_yaml_config(config)
        try:
            network_cfg = config.pl_module.network
        except AttributeError as exc:
            raise ValueError("no pl_module.network in the run config") from exc
        network = fdl.build(network_cfg)

    model = NowcastLightningModule.load_from_checkpoint(
        checkpoint_path,
        network=network,
        map_location=torch.device(device),
        strict=True,
        weights_only=False,
    )
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Loaded {} ({:.1f}M parameters) from {}",
        type(network).__name__,
        n_params / 1e6,
        checkpoint_path,
    )
    return model.to(device).eval()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _to_timedelta(value) -> pd.Timedelta:
    """Accept ``"PT6H"`` (ISO 8601), ``"6h"`` (pandas), timedeltas and offsets."""
    if isinstance(value, str):
        try:
            return pd.Timedelta(value)
        except ValueError:
            return pd.Timedelta(pd.tseries.frequencies.to_offset(value))
    return pd.Timedelta(value)


def init_times_from_range(start, end, freq) -> np.ndarray:
    """
    Regularly spaced initialisation times, both ends inclusive.

    Parameters
    ----------
    start, end : str or datetime-like
        First and last init time, e.g. ``"2023-07-01T00:00"``.
    freq : str or timedelta-like
        Spacing, ISO 8601 duration (``"PT6H"``) or pandas string (``"6h"``).

    Returns
    -------
    np.ndarray
        ``datetime64[ns]`` init times.
    """
    return pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq=_to_timedelta(freq)).values


def _spatial_dims(da: xr.DataArray, t_dim: str) -> tuple[str, str]:
    try:
        axes = da.cf.axes
    except Exception:  # noqa: BLE001 - cf_xarray accessor not available / failing
        axes = {}
    y_dim = axes.get("Y", [None])[0]
    x_dim = axes.get("X", [None])[0]
    rest = [d for d in da.dims if d != t_dim]
    if y_dim not in da.dims or x_dim not in da.dims:
        # same fallback as mlcast (dims named y/x), else the last two dims
        y_dim, x_dim = ("y", "x") if {"y", "x"} <= set(rest) else tuple(rest[-2:])
    return y_dim, x_dim


def _grid_spacing_km(coord: xr.DataArray) -> float:
    values = np.asarray(coord.values, dtype=float)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError(
            f"cannot infer the grid spacing from coordinate {coord.name!r}; pass dx_km"
        )
    spacing = float(np.median(np.abs(np.diff(values))))
    units = str(coord.attrs.get("units", "")).strip().lower()
    if units in ("km", "kilometre", "kilometer", "kilometres", "kilometers"):
        return spacing
    if units not in ("m", "metre", "meter", "metres", "meters"):
        warnings.warn(
            f"coordinate {coord.name!r} has units {units!r}, assuming metres for the grid spacing",
            stacklevel=3,
        )
    return spacing / 1000.0


class VerificationDataset(Dataset):
    """
    Forecast cases on a fixed domain of an mlcast source-data Zarr.

    Each case is identified by its init time, the valid time of the last input
    frame; its targets are the following `forecast_steps` frames. Unlike mlcast's
    training datasets there is no random cropping, offset or augmentation, so a
    case is uniquely identified by its init time.

    Inputs are normalised exactly as in mlcast (``ReflectivityScaling`` forward
    conversion of the raw stored values, NaN -> ``norm_min``), and the validity
    mask is mlcast's: finite (after normalisation) at every input and target step.
    Targets are in the native units of the source variable, reflectivities clipped to
    ``[dbz_floor, dbz_ceiling]`` (the range the model can predict).

    Parameters
    ----------
    source : xr.Dataset or str or os.PathLike
        Source data, an open dataset or a path/URL of the Zarr store.
    init_times : sequence of datetime-like
        Requested init times (duplicates are not allowed).
    input_steps, forecast_steps : int
        Number of input frames and of forecast lead times.
    variable : str, optional
        Name of the data variable. Default: the variable with `standard_name`, or the
        single variable with a supported standard name.
    region : dict, optional
        Integer index slices ``{"y": slice, "x": slice}`` (keys are the logical axes,
        mapped to the dataset's Y/X dims) selecting a fixed sub-domain.
    standard_name : str, optional
        CF standard name used to look up the variable and select the scaling;
        defaults to the variable's ``standard_name`` attribute.
    time_step : str or timedelta-like, optional
        Spacing of the frames; default: the most common spacing of the time coordinate.
    on_missing : {"skip", "raise"}
        What to do with requested init times for which a needed frame is absent from
        the time coordinate (or listed in a ``missing_times`` coordinate): skip them
        with a warning (default), or raise. NaN-filled frames are not skipped, they
        are masked out.
    scaling : mlcast.data.normalization.ReflectivityScaling, optional
        Scaling constants, default mlcast's ``DEFAULT_SCALING``.
    zero_threshold : float, optional
        Value (in the source units) below which values count as zero in the rank
        histogram. Default: ``scaling.dbz_floor`` for reflectivities, 0.037 mm/h in the
        source units for rain rates (:func:`.units.default_zero_threshold`).
    dx_km : float, optional
        Grid spacing in km; default from the X coordinate spacing (metres -> km).
    storage_options : dict, optional
        Passed to ``xr.open_zarr`` when `source` is a path/URL.

    Notes
    -----
    Items are dicts with ``input`` (T_in, 1, H, W) float32 normalised, ``target``
    (T_out, H, W) float32 in the source units with NaN kept, ``mask`` (H, W) bool and
    ``init_time`` (int64 ns since the epoch, so it collates into a tensor).
    """

    def __init__(
        self,
        source: xr.Dataset | str | os.PathLike,
        init_times: Sequence,
        input_steps: int,
        forecast_steps: int,
        *,
        variable: str | None = None,
        region: dict[str, slice] | None = None,
        standard_name: str | None = None,
        time_step=None,
        on_missing: str = "skip",
        scaling: ReflectivityScaling | None = None,
        zero_threshold: float | None = None,
        dx_km: float | None = None,
        storage_options: dict[str, Any] | None = None,
    ) -> None:
        if input_steps < 1 or forecast_steps < 1:
            raise ValueError("input_steps and forecast_steps must be at least 1")
        if on_missing not in ("skip", "raise"):
            raise ValueError(f"on_missing must be 'skip' or 'raise', got {on_missing!r}")
        if isinstance(source, xr.Dataset):
            self._path, self._ds = None, source
        else:
            self._path, self._ds = str(source), None
        self.storage_options = storage_options
        self.input_steps = int(input_steps)
        self.forecast_steps = int(forecast_steps)
        self.scaling = scaling or DEFAULT_SCALING
        self.region = dict(region or {})
        if set(self.region) - {"y", "x"}:
            raise ValueError(f"region keys must be 'y' and/or 'x', got {sorted(self.region)}")

        ds = self.ds
        self.variable, self.standard_name = self._resolve_variable(ds, variable, standard_name)
        da = ds[self.variable]
        self.t_dim = self._time_dim(da)
        self.y_dim, self.x_dim = _spatial_dims(da, self.t_dim)
        units = da.attrs.get("units")
        if units is None:
            raise ValueError(f"variable {self.variable!r} has no `units` attribute")
        self.units = str(units)
        # scores are in the native units and named after the kind of variable (tp, dbz)
        self.score_variable = cv.SHORT_NAMES[self.standard_name]
        if zero_threshold is not None:
            self.zero_threshold = float(zero_threshold)
        elif self.standard_name == _REFLECTIVITY:
            self.zero_threshold = float(self.scaling.dbz_floor)
        else:
            self.zero_threshold = default_zero_threshold(self.units)

        times = pd.DatetimeIndex(ds.indexes[self.t_dim])
        if time_step is None:
            diffs = np.diff(times.values)
            if len(diffs) == 0:
                raise ValueError("cannot infer time_step from a single time; pass time_step")
            values, counts = np.unique(diffs, return_counts=True)
            self.time_step = pd.Timedelta(values[np.argmax(counts)])
        else:
            self.time_step = _to_timedelta(time_step)

        if dx_km is None:
            x_coord = ds[self.x_dim] if self.x_dim in ds.coords else None
            if x_coord is None:
                raise ValueError(f"no coordinate for dim {self.x_dim!r}; pass dx_km")
            dx_km = _grid_spacing_km(x_coord)
            if self.y_dim in ds.coords:
                dy_km = _grid_spacing_km(ds[self.y_dim])
                if not np.isclose(dx_km, dy_km, rtol=0.01):
                    warnings.warn(
                        f"grid spacing differs in x ({dx_km} km) and y ({dy_km} km); "
                        "the radial PSD uses the x spacing",
                        stacklevel=2,
                    )
        self.dx_km = float(dx_km)

        self._index_cases(times, init_times, on_missing, ds)
        if self._path is not None:
            # reopen lazily in each DataLoader worker (as mlcast does)
            self._ds = None

    @classmethod
    def from_mlcast_config(
        cls,
        config: fdl.Config | str | os.PathLike,
        split: str = "test",
        *,
        forecast_steps: int | None = None,
        max_cases: int | None = None,
        region: dict[str, slice] | None = None,
        **kwargs: Any,
    ) -> VerificationDataset:
        """
        Cases of a training run's test (or validation) split, over the full domain.

        The source Zarr, its ``storage_options``, the standard name (first of
        ``standard_names``), ``input_steps`` and ``forecast_steps`` come from the
        run's ``data.dataset_factory``; the init times are those of the crops mlcast
        selects for the split (:func:`split_init_times`).

        Parameters
        ----------
        config : fdl.Config or str or os.PathLike
            Run config from :func:`load_run_config`, or the path of its ``config.yaml``
            (relative data paths then resolve against the current directory).
        split : {"test", "val"}
            Which split.
        forecast_steps : int, optional
            Number of lead times; default the training value. A ConvGRU can be
            unrolled further than it was trained for.
        max_cases : int, optional
            Keep at most this many init times, evenly spread over the split
            (deterministic), for quick runs.
        region : dict, optional
            Fixed sub-domain, see the class parameters. Default: full domain.
        **kwargs
            Passed to the constructor (``on_missing``, ``zero_threshold``, ...).

        Returns
        -------
        VerificationDataset
        """
        config = _as_run_config(config)
        factory = config.data.dataset_factory
        standard_name = _single_standard_name(config)
        init_times = split_init_times(config, split)
        if max_cases is not None:
            if max_cases < 1:
                raise ValueError("max_cases must be at least 1")
            if max_cases < len(init_times):
                pick = np.unique(np.linspace(0, len(init_times) - 1, max_cases).round().astype(int))
                logger.info(
                    "Subsampling evenly to {} of the {} init times (max_cases)",
                    len(pick),
                    len(init_times),
                )
                init_times = init_times[pick]
        return cls(
            factory.zarr_path,
            init_times,
            int(factory.input_steps),
            int(forecast_steps if forecast_steps is not None else factory.forecast_steps),
            standard_name=standard_name,
            region=region,
            storage_options=getattr(factory, "storage_options", None),
            **kwargs,
        )

    # -- setup helpers -------------------------------------------------------

    @staticmethod
    def _resolve_variable(ds, variable, standard_name) -> tuple[str, str]:
        if variable is None:
            candidates = [
                name
                for name, var in ds.data_vars.items()
                if var.attrs.get("standard_name")
                in ((standard_name,) if standard_name else SUPPORTED_STANDARD_NAMES)
            ]
            if len(candidates) != 1:
                raise ValueError(
                    f"found {len(candidates)} data variables with standard_name "
                    f"{standard_name or SUPPORTED_STANDARD_NAMES} ({candidates}); pass `variable`"
                )
            variable = str(candidates[0])
        elif variable not in ds.data_vars:
            raise KeyError(f"no data variable {variable!r} in the source dataset")
        standard_name = standard_name or ds[variable].attrs.get("standard_name")
        if standard_name is None:
            raise ValueError(
                f"variable {variable!r} has no standard_name attribute; pass standard_name"
            )
        _check_standard_name(standard_name)
        return variable, standard_name

    @staticmethod
    def _time_dim(da: xr.DataArray) -> str:
        if "time" in da.dims:
            return "time"
        try:
            return da.cf["time"].dims[0]
        except Exception as exc:
            raise ValueError(f"cannot find the time dimension of {da.name!r}") from exc

    def _index_cases(self, times: pd.DatetimeIndex, init_times, on_missing: str, ds) -> None:
        requested = pd.to_datetime(list(np.atleast_1d(np.asarray(init_times)))).values.astype(
            "datetime64[ns]"
        )
        if len(np.unique(requested)) != len(requested):
            raise ValueError("init_times contains duplicates")
        offsets = (
            np.arange(-(self.input_steps - 1), self.forecast_steps + 1)
            * self.time_step.to_timedelta64()
        )
        needed = requested[:, None] + offsets[None, :]
        indices = times.get_indexer(needed.ravel()).reshape(needed.shape)
        ok = (indices >= 0).all(axis=1)
        if "missing_times" in ds.variables:
            missing = ds["missing_times"].values.astype("datetime64[ns]")
            ok &= ~np.isin(needed, missing).any(axis=1)
        if not ok.all():
            bad = pd.DatetimeIndex(requested[~ok])
            msg = (
                f"{len(bad)} of {len(requested)} init times lack input/target frames in the "
                f"source data (time step {self.time_step}): {', '.join(map(str, bad[:5]))}"
                + (" ..." if len(bad) > 5 else "")
            )
            if on_missing == "raise":
                raise ValueError(msg)
            warnings.warn(msg + "; skipping them", stacklevel=3)
        if not ok.any():
            raise ValueError("none of the requested init times has all frames in the source data")
        self.skipped_init_times = requested[~ok]
        self.init_times = requested[ok]
        self._time_indices = indices[ok]

    # -- properties ------------------------------------------------------------

    @property
    def ds(self) -> xr.Dataset:
        """The source dataset, opened lazily (once per process) when given as a path."""
        if self._ds is None:
            self._ds = xr.open_zarr(self._path, storage_options=self.storage_options)
        return self._ds

    @property
    def lead_times(self) -> np.ndarray:
        """Lead times ``time_step * (1 .. forecast_steps)`` as ``timedelta64[ns]``."""
        return (np.arange(1, self.forecast_steps + 1) * self.time_step.to_timedelta64()).astype(
            "timedelta64[ns]"
        )

    @property
    def grid_shape(self) -> tuple[int, int]:
        """``(H, W)`` of the verification domain (region applied)."""
        shape = []
        for name, dim in (("y", self.y_dim), ("x", self.x_dim)):
            size = self.ds.sizes[dim]
            if name in self.region:
                size = len(range(*self.region[name].indices(size)))
            shape.append(size)
        return shape[0], shape[1]

    def grid_coords(self) -> dict[str, xr.Variable]:
        """
        The ``y``/``x`` coordinates of the verification domain (region applied).

        Returns
        -------
        dict
            ``{"y": Variable, "x": Variable}`` with the source's coordinate values and
            attributes, renamed to ``y``/``x``; a dim without a coordinate is left out.
        """
        coords = {}
        for name, dim in (("y", self.y_dim), ("x", self.x_dim)):
            if dim in self.ds.coords:
                coord = self.ds[dim]
                if name in self.region:
                    coord = coord.isel({dim: self.region[name]})
                coords[name] = xr.Variable(name, coord.values, dict(coord.attrs))
        return coords

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        if self._path is not None:
            state["_ds"] = None
        return state

    def __len__(self) -> int:
        return len(self.init_times)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        t_idx = self._time_indices[idx]
        if np.all(np.diff(t_idx) == 1):
            t_sel = slice(int(t_idx[0]), int(t_idx[-1]) + 1)
        else:
            t_sel = t_idx
        isel = {self.t_dim: t_sel}
        if "y" in self.region:
            isel[self.y_dim] = self.region["y"]
        if "x" in self.region:
            isel[self.x_dim] = self.region["x"]
        da = self.ds[self.variable].isel(isel).transpose(self.t_dim, self.y_dim, self.x_dim)
        raw = np.asarray(da.values)

        normed = _normalize_numpy(raw, self.standard_name, self.scaling)
        mask = ~np.isnan(normed).any(axis=0)
        inputs = np.nan_to_num(normed[: self.input_steps], nan=self.scaling.norm_min)
        target = _truth_values(
            raw[self.input_steps :].astype(np.float32), self.standard_name, self.scaling
        )
        return {
            "input": torch.from_numpy(np.ascontiguousarray(inputs[:, None], dtype=np.float32)),
            "target": torch.from_numpy(np.ascontiguousarray(target)),
            "mask": torch.from_numpy(mask),
            "init_time": torch.tensor(
                self.init_times[idx].astype("datetime64[ns]").astype(np.int64)
            ),
        }


# ---------------------------------------------------------------------------
# Verification loop
# ---------------------------------------------------------------------------


def _member_chunks(ensemble_size: int, member_chunk: int | None) -> list[int]:
    """Split the members into model calls; never a 1-member call for an ensemble.

    ``ConvGruModel`` returns the deterministic (zero-noise) forecast for
    ``ensemble_size=1`` (unless ``noisy_decoder``), so a 1-member chunk would not be an
    ensemble member. A remainder of 1 is merged into the previous chunk.
    """
    if ensemble_size < 1:
        raise ValueError("ensemble_size must be at least 1")
    if member_chunk is None or member_chunk >= ensemble_size:
        return [ensemble_size]
    if member_chunk < 2:
        raise ValueError(
            "member_chunk must be at least 2 for an ensemble (a 1-member call is deterministic)"
        )
    chunks = [member_chunk] * (ensemble_size // member_chunk)
    rest = ensemble_size % member_chunk
    if rest == 1:
        chunks[-1] += 1
    elif rest:
        chunks.append(rest)
    return chunks


def _check_precision(precision: str) -> None:
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")


def _autocast(device: torch.device, precision: str):
    """``torch.autocast`` to bfloat16 for ``"bf16"``, a no-op context for ``"fp32"``."""
    _check_precision(precision)
    if precision == "fp32":
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def _forecast(
    model, x: torch.Tensor, forecast_steps: int, chunks: list[int], precision: str = "fp32"
) -> torch.Tensor:
    """Run the model for all members, ``(B, T, M, H, W)`` normalised, float32."""
    b, _, _, h, w = x.shape
    preds = None
    start = 0
    for k in chunks:
        with _autocast(x.device, precision):
            out = model(x, forecast_steps=forecast_steps, ensemble_size=k)
        out = out.float()  # no-op in fp32; de-normalisation and scores run in float32
        if out.shape[2] != k:
            raise ValueError(
                f"model returned {out.shape[2]} channels for {k} members; only single-variable "
                "models (one output channel per member) are supported"
            )
        if len(chunks) == 1:
            return out
        if preds is None:
            preds = out.new_empty((b, forecast_steps, sum(chunks), h, w))
        preds[:, :, start : start + k] = out
        start += k
        del out
    return preds


def verify_mlcast_model(
    model,
    dataset: VerificationDataset,
    *,
    forecast_source: str,
    ensemble_size: int,
    batch_size: int = 1,
    device: str | torch.device | None = None,
    num_workers: int = 0,
    seed: int = 0,
    compute_psd: bool = True,
    member_chunk: int | None = None,
    precision: str = "fp32",
    progress: bool = True,
) -> xr.Dataset:
    """
    Score an mlcast model on the cases of `dataset` without storing forecasts.

    For each batch: ``model(x, forecast_steps=T, ensemble_size=K)`` -> clamp to
    ``[norm_min, norm_max]`` (as in training) -> :func:`denormalize_torch` (to the
    source units) -> :meth:`ScoreAccumulator.update` on the device; the forecast tensor
    is then freed. Scores are in the native units of the source variable.

    Parameters
    ----------
    model : NowcastLightningModule or callable
        Called as ``model(x, forecast_steps=, ensemble_size=)`` with ``x`` of shape
        ``(B, T_in, 1, H, W)``, returning ``(B, T, M, H, W)`` normalised (members on
        dim 2). Moved to `device` and set to eval mode.
    dataset : VerificationDataset
        Cases to score.
    forecast_source : str
        Name of the forecast source in the score dataset.
    ensemble_size : int
        Number of members M.
    batch_size : int
        Cases per batch.
    device : str or torch.device, optional
        Default ``"cuda"`` if available, else ``"cpu"``.
    num_workers : int
        DataLoader workers reading the source data. Workers are spawned (not
        forked), so `dataset` is pickled to each worker; a dataset opened from a
        path is cheap to pickle (the store is reopened in the worker), an in-memory
        ``xr.Dataset`` source is copied.
    seed : int
        Seeds torch's global RNG (the ConvGRU noise) and the rank-histogram tie breaking.
    compute_psd : bool
        Whether to compute radial power spectra.
    member_chunk : int, optional
        Generate the members in model calls of at most this many members (at least 2)
        to lower the peak activation memory. The encoder is re-run for each call and
        the full ``(B, T, M, H, W)`` forecast is still assembled on the device.
    precision : {"fp32", "bf16"}
        Run the model in float32, or under ``torch.autocast`` to bfloat16 (about twice
        as fast on GPUs with bf16 tensor cores). The forecast is cast to float32 before
        it is de-normalised and scored either way.
    progress : bool
        Print a summary panel of the cases, model and batching to stderr (as mlcast's
        sampler does) and show a tqdm progress bar, with the mean time per batch spent
        waiting for data,
        in the model (including the transfer to the device and de-normalisation) and
        in the scores. The timing synchronises CUDA once per stage and batch.

    Returns
    -------
    xr.Dataset
        Score dataset in the spec layout, without global attributes.
    """
    acc = ScoreAccumulator(
        forecast_source,
        dataset.lead_times,
        dataset.dx_km,
        variable=dataset.score_variable,
        units=dataset.units,
        zero_threshold=dataset.zero_threshold,
        seed=seed,
        compute_psd=compute_psd,
    )
    batches = _iter_forecasts(
        model,
        dataset,
        ensemble_size,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
        seed=seed,
        member_chunk=member_chunk,
        precision=precision,
        progress=progress,
        desc=f"verifying {forecast_source}",
        consumer="score",
    )
    with torch.inference_mode():
        for preds, obs, mask, init_times in batches:
            acc.update(preds, obs, mask, init_times)
            del preds, obs, mask
    return acc.to_dataset()


def _iter_forecasts(
    model,
    dataset: VerificationDataset,
    ensemble_size: int,
    *,
    batch_size: int = 1,
    device: str | torch.device | None = None,
    num_workers: int = 0,
    seed: int = 0,
    member_chunk: int | None = None,
    precision: str = "fp32",
    progress: bool = True,
    desc: str = "forecasting",
    consumer: str = "score",
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]]:
    """
    Run the model over the cases of `dataset`, batch by batch.

    Seeds torch's global RNG with `seed` before the first batch, so the same
    arguments give the same ensemble. Each batch is computed in inference mode;
    consumers that modify the tensors in place should also run in inference mode.

    With `progress`, the progress bar shows the mean time per batch spent waiting for
    the DataLoader (``data``), in the model (``model``) and in the consumer of the
    yielded tensors (labelled `consumer`, the time until the next batch is requested).

    Yields
    ------
    fc : torch.Tensor
        Forecast ``(B, T, M, H, W)`` in the source units on the device.
    obs : torch.Tensor
        Observation ``(B, T, H, W)`` in the source units on the device, NaN replaced by 0 (NaN
        pixels are outside `mask`).
    mask : torch.Tensor
        Boolean ``(B, H, W)`` on the device: finite at all input and target steps.
    init_times : np.ndarray
        ``datetime64[ns]`` init times of the B cases.
    """
    _check_precision(precision)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device).eval()
    chunks = _member_chunks(ensemble_size, member_chunk)
    scaling = dataset.scaling
    worker_kwargs: dict[str, Any] = {}
    if num_workers > 0:
        # Forked workers can deadlock once the parent has started threads (zarr/xarray
        # I/O, torch); spawned workers start clean and reopen the source themselves.
        worker_kwargs = {"multiprocessing_context": "spawn", "persistent_workers": False}
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        **worker_kwargs,
    )
    batches = loader
    bar = None
    if progress:
        _print_summary(
            model,
            dataset,
            desc=desc,
            ensemble_size=ensemble_size,
            chunks=chunks,
            batch_size=batch_size,
            n_batches=len(loader),
            num_workers=num_workers,
            device=device,
            precision=precision,
        )
        try:
            from tqdm.auto import tqdm

            batches = bar = tqdm(loader, desc=desc, unit="batch")
        except ImportError:  # pragma: no cover
            pass
    timer = _StageTimer(device, ("data", "model", consumer)) if bar is not None else None
    started = time.perf_counter()
    torch.manual_seed(seed)  # also seeds all CUDA devices
    if timer:
        timer.start()
    for batch in batches:
        if timer:
            timer.lap("data")
        with torch.inference_mode():
            x = batch["input"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            obs = batch["target"].to(device, non_blocking=True)
            obs = torch.nan_to_num(obs, nan=0.0)  # NaN obs are outside `mask`
            init_times = batch["init_time"].numpy().astype("datetime64[ns]")

            preds = _forecast(model, x, dataset.forecast_steps, chunks, precision)
            preds = preds.clamp_(scaling.norm_min, scaling.norm_max)
            preds = denormalize_torch(preds, dataset.standard_name, scaling)
            del x
        if timer:
            timer.lap("model")
        yield preds, obs, mask, init_times
        del preds, obs, mask
        if timer:
            timer.lap(consumer)
            bar.set_postfix_str(timer.summary(), refresh=False)
    elapsed = time.perf_counter() - started
    logger.info(
        "{} done: {} cases in {} ({:.2f} s/case{})",
        desc[0].upper() + desc[1:],
        len(dataset),
        datetime.timedelta(seconds=round(elapsed)),
        elapsed / len(dataset),
        f"; per batch: {timer.summary()}" if timer else "",
    )


def _minutes(td: pd.Timedelta) -> str:
    return f"{td.total_seconds() / 60:g} min"


def _describe_model(model) -> str:
    network = getattr(model, "network", model)
    try:
        n_params = sum(p.numel() for p in model.parameters())
    except AttributeError:  # a plain callable
        return type(network).__name__
    return f"{type(network).__name__}, {n_params:,} parameters"


def _print_summary(
    model,
    dataset: VerificationDataset,
    *,
    desc: str,
    ensemble_size: int,
    chunks: list[int],
    batch_size: int,
    n_batches: int,
    num_workers: int,
    device: torch.device,
    precision: str,
) -> None:
    """Print a panel of what is about to be forecast to stderr, as mlcast's sampler does."""
    from rich.console import Console
    from rich.markup import escape
    from rich.panel import Panel
    from rich.table import Table

    h, w = dataset.grid_shape
    first, last = (pd.Timestamp(t) for t in dataset.init_times[[0, -1]])
    skipped = len(dataset.skipped_init_times)
    lead_times = pd.to_timedelta(dataset.lead_times)
    n_cases = len(dataset)
    calls = (
        "1 model call per batch"
        if len(chunks) == 1
        else f"{len(chunks)} model calls of {'/'.join(map(str, chunks))} members per batch"
    )
    device_label = str(device)
    if device.type == "cuda":
        device_label += f" ({torch.cuda.get_device_name(device)})"

    grid = Table.grid(padding=(0, 2))
    grid.add_column(justify="right", style="bold cyan")
    grid.add_column()
    grid.add_row("Source", escape(f"📂  {dataset._path or 'in-memory dataset'}"))
    grid.add_row(
        "Variable",
        escape(f"💧  {dataset.variable!r}   {dataset.standard_name} [{dataset.units}]"),
    )
    grid.add_row(
        "Domain",
        f"🗺️   {h:,} × {w:,} px   dx {dataset.dx_km:g} km   ({_describe_domain(dataset)})",
    )
    grid.add_row(
        "Init times",
        f"📅  {first:%Y-%m-%d %H:%M} → {last:%Y-%m-%d %H:%M}   {n_cases:,} cases"
        + (f", {skipped:,} skipped (missing frames)" if skipped else ""),
    )
    grid.add_row(
        "Steps",
        f"🕒  {dataset.input_steps} input + {dataset.forecast_steps} forecast × "
        f"{_minutes(dataset.time_step)}   lead {_minutes(lead_times[0])} → "
        f"{_minutes(lead_times[-1])}",
    )
    grid.add_row("Model", f"🧠  {_describe_model(model)}")
    grid.add_row("Ensemble", f"🎲  {ensemble_size} members   {calls}")
    grid.add_row(
        "Workload",
        f"🧮  {n_cases:,} cases × {ensemble_size} members × {dataset.forecast_steps} steps"
        f" = {n_cases * ensemble_size * dataset.forecast_steps:,} fields of {h:,} × {w:,}",
    )
    grid.add_row(
        "Batches", f"📦  {n_batches:,} × {batch_size} cases   {num_workers} loader workers"
    )
    grid.add_row(
        "Device",
        f"⚡  {device_label}   {'bfloat16 autocast' if precision == 'bf16' else 'float32'}",
    )
    Console(stderr=True).print(
        Panel(
            grid,
            title=f"[bold]🎯 {desc[0].upper() + desc[1:]}[/]",
            subtitle=f"[dim]{os.path.basename(str(dataset._path))}[/]" if dataset._path else None,
            border_style="blue",
            expand=False,
        )
    )


class _StageTimer:
    """Mean wall time per batch of named stages, synchronising CUDA at each lap."""

    def __init__(self, device: torch.device, stages: Sequence[str]) -> None:
        self.device = device
        self.totals = dict.fromkeys(stages, 0.0)
        self.n_batches = 0
        self._last = 0.0

    def start(self) -> None:
        self._last = time.perf_counter()

    def lap(self, stage: str) -> None:
        """Add the time since the previous lap (or start) to `stage`."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        now = time.perf_counter()
        self.totals[stage] += now - self._last
        self._last = now
        if stage == next(reversed(self.totals)):
            self.n_batches += 1

    def summary(self) -> str:
        n = max(self.n_batches, 1)
        return " ".join(f"{stage} {total / n:.2f}s" for stage, total in self.totals.items())


# ---------------------------------------------------------------------------
# Runs of a training config: forecasts, truth and on-the-fly scores
# ---------------------------------------------------------------------------


def _describe_source(
    attrs: dict[str, Any],
    path: Any,
    variable: str,
    standard_name: str,
    units: Any,
    scaling: ReflectivityScaling,
) -> str:
    name = (
        attrs.get("title") or attrs.get("mlcast_dataset_identifier") or str(path or "source data")
    )
    ident = attrs.get("mlcast_dataset_identifier")
    version = attrs.get("mlcast_dataset_version")
    tag = f" ({ident} v{version})" if ident and version else ""
    text = f"{name}{tag}, variable {variable!r} ({standard_name}, {units})"
    if standard_name == _REFLECTIVITY:
        text += f" clipped to [{scaling.dbz_floor:g}, {scaling.dbz_ceiling:g}] dBZ"
    return text


def _describe_truth(dataset: VerificationDataset) -> str:
    return _describe_source(
        dataset.ds.attrs,
        dataset._path,
        dataset.variable,
        dataset.standard_name,
        dataset.units,
        dataset.scaling,
    )


class _Run(NamedTuple):
    model: Any
    dataset: VerificationDataset
    forecast_source: str
    device: str
    config_label: str
    checkpoint: pathlib.Path
    split: str


def _default_forecast_source(checkpoint: pathlib.Path) -> str:
    """Name of the run directory (``<run>/checkpoints/<name>.ckpt`` -> ``<run>``)."""
    parent = checkpoint.resolve().parent
    run_dir = parent.parent if parent.name == "checkpoints" else parent
    return run_dir.name or checkpoint.stem


def _prepare_run(
    config: fdl.Config | str | os.PathLike,
    checkpoint: str | os.PathLike,
    *,
    split: str = "test",
    forecast_source: str | None = None,
    device: str | torch.device | None = None,
    max_cases: int | None = None,
    forecast_steps: int | None = None,
    data_root: str | os.PathLike | None = None,
    zarr_path: str | os.PathLike | None = None,
    index_path: str | os.PathLike | None = None,
    storage_options: dict[str, Any] | None = None,
    **dataset_kwargs: Any,
) -> _Run:
    """Config, model and cases shared by :func:`forecast_mlcast` and :func:`verify_mlcast_run`."""
    checkpoint = pathlib.Path(checkpoint)
    config, label = _resolve_config(
        config,
        data_root=data_root,
        zarr_path=zarr_path,
        index_path=index_path,
        storage_options=storage_options,
    )
    device = str(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_mlcast_model(checkpoint, config=config, device=device)
    dataset = VerificationDataset.from_mlcast_config(
        config, split, forecast_steps=forecast_steps, max_cases=max_cases, **dataset_kwargs
    )
    return _Run(
        model,
        dataset,
        forecast_source or _default_forecast_source(checkpoint),
        device,
        label,
        checkpoint,
        split,
    )


def _describe_domain(dataset: VerificationDataset) -> str:
    if not dataset.region:
        return "full domain"
    return "sub-domain " + ", ".join(
        f"{k}={'' if s.start is None else s.start}:{'' if s.stop is None else s.stop}"
        for k, s in dataset.region.items()
    )


def _describe_forecasts(run: _Run, ensemble_size: int, precision: str = "fp32") -> str:
    dataset = run.dataset
    first, last = (pd.Timestamp(t) for t in dataset.init_times[[0, -1]])
    domain = _describe_domain(dataset)
    autocast = ", run with bfloat16 autocast" if precision == "bf16" else ""
    return (
        f"{run.forecast_source}: mlcast model {run.checkpoint.name}{autocast}, "
        f"{ensemble_size}-member ensemble from {dataset.input_steps} input frames, "
        f"{domain} at the {len(dataset)} init times of the {run.split} split's sampled "
        f"crops ({first:%Y-%m-%dT%H:%M} to {last:%Y-%m-%dT%H:%M}) of {run.config_label}"
    )


def forecast_mlcast(
    config: fdl.Config | str | os.PathLike,
    checkpoint: str | os.PathLike,
    *,
    ensemble_size: int,
    split: str = "test",
    max_cases: int | None = None,
    forecast_steps: int | None = None,
    region: dict[str, slice] | None = None,
    data_root: str | os.PathLike | None = None,
    zarr_path: str | os.PathLike | None = None,
    index_path: str | os.PathLike | None = None,
    storage_options: dict[str, Any] | None = None,
    forecast_source: str | None = None,
    device: str | torch.device | None = None,
    batch_size: int = 1,
    num_workers: int = 0,
    member_chunk: int | None = None,
    precision: str = "fp32",
    seed: int = 0,
    out: str | os.PathLike | None = None,
    progress: bool = True,
    **dataset_kwargs: Any,
) -> xr.Dataset:
    """
    Forecasts of a trained mlcast run on the test split defined in its training config.

    The cases are those of :func:`verify_mlcast_run` with the same arguments (same
    config, checkpoint, split, ``max_cases``, ``region``, ...), and with the same
    ``seed``, ``batch_size`` and ``member_chunk`` the ensemble is the same too, so
    ``compute_scores(forecast_mlcast(config, ckpt, ...), open_truth(config))`` gives the
    scores of ``verify_mlcast_run(config, ckpt, ...)``.

    Pixels where an input or target frame of the case is invalid (NaN in the source
    data, mlcast's validity mask) are NaN in the forecasts.

    Parameters
    ----------
    config : fdl.Config or str or os.PathLike
        The run's config (:func:`load_run_config`) or the path of its ``config.yaml``.
    checkpoint : str or os.PathLike
        Lightning ``.ckpt`` file of the run.
    ensemble_size : int
        Number of members.
    split, max_cases, forecast_steps, region, data_root, zarr_path, index_path, \
storage_options, forecast_source, device, batch_size, num_workers, member_chunk, precision, \
seed, progress
        See :func:`verify_mlcast_run`.
    out : str or os.PathLike, optional
        Zarr store to write the forecasts to, batch by batch (appending along
        ``init_time``), so they never have to fit in memory; an existing store is
        replaced. Default: keep the forecasts in memory.
    **dataset_kwargs
        Passed to :class:`VerificationDataset` (``on_missing``, ``scaling``, ...).

    Returns
    -------
    xr.Dataset
        Variable ``tp`` (rain rates) or ``dbz`` (reflectivities)
        ``(forecast_source, init_time, lead_time, member, y, x)`` float32 in the units
        of the source variable, with ``forecast_source`` of length 1 and the source's
        ``x``/``y`` coordinates; global attributes describe the run. With `out`, the
        store opened lazily with ``xr.open_zarr``.
    """
    run = _prepare_run(
        config,
        checkpoint,
        split=split,
        forecast_source=forecast_source,
        device=device,
        max_cases=max_cases,
        forecast_steps=forecast_steps,
        data_root=data_root,
        zarr_path=zarr_path,
        index_path=index_path,
        storage_options=storage_options,
        region=region,
        **dataset_kwargs,
    )
    return _forecast_run(
        run,
        ensemble_size=ensemble_size,
        batch_size=batch_size,
        num_workers=num_workers,
        member_chunk=member_chunk,
        precision=precision,
        seed=seed,
        out=out,
        progress=progress,
    )


def _forecast_run(
    run: _Run,
    *,
    ensemble_size: int,
    batch_size: int,
    num_workers: int,
    member_chunk: int | None,
    precision: str,
    seed: int,
    out: str | os.PathLike | None,
    progress: bool,
) -> xr.Dataset:
    dataset = run.dataset
    static_coords: dict[str, Any] = {
        cv.FORECAST_SOURCE: np.array([run.forecast_source], dtype=str),
        cv.LEAD_TIME: dataset.lead_times,
        cv.MEMBER: np.arange(ensemble_size),
        **dataset.grid_coords(),
    }
    name = dataset.score_variable
    var_attrs = _variable_attrs(dataset.standard_name, dataset.units)
    attrs = {
        "title": f"Forecasts of {run.forecast_source} ({run.split} split)",
        "forecasts": _describe_forecasts(run, ensemble_size, precision),
        "truth": _describe_truth(dataset),
        "checkpoint": str(run.checkpoint),
        "config": run.config_label,
        "split": run.split,
        "ensemble_size": int(ensemble_size),
        "precision": precision,
        "seed": int(seed),
    }

    def to_dataset(values: np.ndarray, init_times: np.ndarray) -> xr.Dataset:
        return xr.Dataset(
            {name: (cv.FORECAST_DIMS, values[None], var_attrs)},
            coords={**static_coords, cv.INIT_TIME: init_times},
            attrs=attrs,
        )

    pieces: list[tuple[np.ndarray, np.ndarray]] = []
    written = False
    batches = _iter_forecasts(
        run.model,
        dataset,
        ensemble_size,
        batch_size=batch_size,
        device=run.device,
        num_workers=num_workers,
        seed=seed,
        member_chunk=member_chunk,
        precision=precision,
        progress=progress,
        desc=f"forecasting {run.forecast_source}",
        consumer="write",
    )
    with torch.inference_mode():
        for preds, _obs, mask, init_times in batches:
            # no forecast where mlcast's validity mask (inputs and targets) is False
            preds.masked_fill_(~mask[:, None, None], float("nan"))
            values = preds.to(torch.float32).cpu().numpy()
            del preds, _obs, mask
            if out is None:
                pieces.append((values, init_times))
                continue
            ds = to_dataset(values, init_times)
            with warnings.catch_warnings():
                # zarr warns that consolidated metadata is not part of the v3 spec
                warnings.filterwarnings("ignore", message=".*[Cc]onsolidated metadata.*")
                if not written:
                    _, t, m, h, w = values.shape
                    encoding = {
                        name: {"chunks": (1, 1, t, m, h, w)},
                        cv.INIT_TIME: {"units": "seconds since 1970-01-01", "dtype": "int64"},
                    }
                    ds.to_zarr(out, mode="w", encoding=encoding)
                    written = True
                else:
                    ds = ds.drop_vars([name for name in ds.coords if name != cv.INIT_TIME])
                    ds.to_zarr(out, append_dim=cv.INIT_TIME)
    if out is not None:
        return xr.open_zarr(out)
    return to_dataset(
        np.concatenate([v for v, _ in pieces]), np.concatenate([t for _, t in pieces])
    )


def open_truth(
    config: fdl.Config | str | os.PathLike,
    *,
    data_root: str | os.PathLike | None = None,
    zarr_path: str | os.PathLike | None = None,
    storage_options: dict[str, Any] | None = None,
    region: dict[str, slice] | None = None,
    scaling: ReflectivityScaling | None = None,
) -> xr.Dataset:
    """
    The radar truth of a training run: its source variable in its own units, lazily.

    The source Zarr and the variable are found as in :class:`VerificationDataset`
    (the run's ``data.dataset_factory``: ``zarr_path``, ``storage_options`` and the
    variable with the run's standard name). The values are those the scores of
    :func:`verify_mlcast_run` are computed against: unconverted, reflectivities
    clipped to ``[dbz_floor, dbz_ceiling]``. Times listed in a ``missing_times``
    coordinate are dropped.

    Parameters
    ----------
    config : fdl.Config or str or os.PathLike
        The run's config (:func:`load_run_config`) or the path of its ``config.yaml``.
    data_root, zarr_path, storage_options
        Where to find the data, see :func:`load_run_config`.
    region : dict, optional
        Integer index slices ``{"y": slice, "x": slice}``, the same sub-domain as given
        to :func:`forecast_mlcast` so that the grids match. Default: full domain.
    scaling : mlcast.data.normalization.ReflectivityScaling, optional
        Scaling constants (for the reflectivity clipping), default mlcast's
        ``DEFAULT_SCALING``.

    Returns
    -------
    xr.Dataset
        Variable ``tp`` (rain rates) or ``dbz`` (reflectivities) ``(time, y, x)`` in the
        units of the source variable (NaN = invalid) with the source's ``x``/``y``
        coordinates, dask-backed. Same name and units as the forecasts of
        :func:`forecast_mlcast`.
    """
    config, _ = _resolve_config(
        config, data_root=data_root, zarr_path=zarr_path, storage_options=storage_options
    )
    factory = config.data.dataset_factory
    path = factory.zarr_path
    source = xr.open_zarr(path, storage_options=getattr(factory, "storage_options", None))
    variable, standard_name = VerificationDataset._resolve_variable(
        source, None, _single_standard_name(config)
    )
    da = source[variable]
    t_dim = VerificationDataset._time_dim(da)
    y_dim, x_dim = _spatial_dims(da, t_dim)
    units = da.attrs.get("units")
    if units is None:
        raise ValueError(f"variable {variable!r} has no `units` attribute")
    scaling = scaling or DEFAULT_SCALING

    region = dict(region or {})
    if set(region) - {"y", "x"}:
        raise ValueError(f"region keys must be 'y' and/or 'x', got {sorted(region)}")
    da = da.isel({{"y": y_dim, "x": x_dim}[k]: s for k, s in region.items()})
    da = da.transpose(t_dim, y_dim, x_dim)
    if "missing_times" in source.variables:
        missing = source["missing_times"].values.astype("datetime64[ns]")
        da = da.isel({t_dim: ~np.isin(da[t_dim].values.astype("datetime64[ns]"), missing)})
    values = _truth_values(da.data, standard_name, scaling)

    coords: dict[str, Any] = {cv.TIME: (cv.TIME, da[t_dim].values, dict(da[t_dim].attrs))}
    for name, dim in ((cv.Y, y_dim), (cv.X, x_dim)):
        if dim in da.coords:
            coords[name] = (name, da[dim].values, dict(da[dim].attrs))
    return xr.Dataset(
        {
            cv.SHORT_NAMES[standard_name]: (
                cv.TRUTH_DIMS,
                values,
                {**_variable_attrs(standard_name, str(units)), "source_variable": variable},
            )
        },
        coords=coords,
        attrs={
            "truth": _describe_source(source.attrs, path, variable, standard_name, units, scaling)
        },
    )


def verify_mlcast_run(
    config: fdl.Config | str | os.PathLike,
    checkpoint: str | os.PathLike,
    *,
    ensemble_size: int,
    split: str = "test",
    forecast_source: str | None = None,
    device: str | torch.device | None = None,
    batch_size: int = 1,
    max_cases: int | None = None,
    forecast_steps: int | None = None,
    data_root: str | os.PathLike | None = None,
    zarr_path: str | os.PathLike | None = None,
    index_path: str | os.PathLike | None = None,
    storage_options: dict[str, Any] | None = None,
    region: dict[str, slice] | None = None,
    num_workers: int = 0,
    seed: int = 0,
    compute_psd: bool = True,
    member_chunk: int | None = None,
    precision: str = "fp32",
    progress: bool = True,
    **dataset_kwargs: Any,
) -> xr.Dataset:
    """
    Score a trained mlcast run on the test split defined in its training config.

    Uses the run's config (loaded with :func:`load_run_config` if a path), restores
    the model from `checkpoint` (:func:`load_mlcast_model`), builds the cases
    (:meth:`VerificationDataset.from_mlcast_config`: the unique init times of the
    crops mlcast samples for the split, forecast over the full domain) and scores
    them on the fly (:func:`verify_mlcast_model`); forecasts are never stored. The
    same as ``compute_scores(forecast_mlcast(config, checkpoint, ...),
    open_truth(config))`` with the same arguments.

    Parameters
    ----------
    config : fdl.Config or str or os.PathLike
        The run's config (:func:`load_run_config`) or the path of its ``config.yaml``
        (:func:`find_config_yaml` locates it for a checkpoint). Relative data paths of
        a ``config.yaml`` are resolved against `data_root`.
    checkpoint : str or os.PathLike
        Lightning ``.ckpt`` file of the run.
    ensemble_size : int
        Number of members.
    split : {"test", "val"}
        Which split of the config to verify on.
    forecast_source : str, optional
        Name of the forecast source; default the run directory name.
    device : str or torch.device, optional
        Default ``"cuda"`` if available, else ``"cpu"``.
    batch_size : int
        Cases per batch.
    max_cases : int, optional
        Evenly subsample the init times to at most this many (quick runs).
    forecast_steps : int, optional
        Number of lead times; default the training value.
    data_root, zarr_path, index_path, storage_options
        Where to find the data, see :func:`load_run_config`.
    region : dict, optional
        Fixed sub-domain instead of the full domain, see :class:`VerificationDataset`.
    num_workers, seed, compute_psd, member_chunk, precision, progress
        See :func:`verify_mlcast_model`.
    **dataset_kwargs
        Passed to :class:`VerificationDataset` (``on_missing``, ``zero_threshold``, ...).

    Returns
    -------
    xr.Dataset
        Score dataset in the spec layout, without global attributes (add them with
        ``write_scores``).
    """
    run = _prepare_run(
        config,
        checkpoint,
        split=split,
        forecast_source=forecast_source,
        device=device,
        max_cases=max_cases,
        forecast_steps=forecast_steps,
        data_root=data_root,
        zarr_path=zarr_path,
        index_path=index_path,
        storage_options=storage_options,
        region=region,
        **dataset_kwargs,
    )
    return verify_mlcast_model(
        run.model,
        run.dataset,
        forecast_source=run.forecast_source,
        ensemble_size=ensemble_size,
        batch_size=batch_size,
        device=run.device,
        num_workers=num_workers,
        seed=seed,
        compute_psd=compute_psd,
        member_chunk=member_chunk,
        precision=precision,
        progress=progress,
    )


# ---------------------------------------------------------------------------
# Command line (argument parsers in .mlcast, which has no heavy imports)
# ---------------------------------------------------------------------------


def _prepare_run_from_args(args: argparse.Namespace) -> _Run:
    region = {k: v for k, v in (("y", args.region_y), ("x", args.region_x)) if v is not None}
    return _prepare_run(
        args.config,
        args.checkpoint,
        split=args.split,
        forecast_source=args.forecast_source,
        device=args.device,
        max_cases=args.max_cases,
        forecast_steps=args.forecast_steps,
        data_root=args.data_root,
        zarr_path=args.zarr_path,
        index_path=args.index_path,
        storage_options=args.storage_options,
        region=region or None,
        on_missing=args.on_missing,
    )


def verify_trained_model_from_args(args: argparse.Namespace) -> int:
    """
    Run ``verify-trained-model``: load the run, score it on the fly, write the scores.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments of
        :func:`mlcast_verification_tools.integrations.mlcast.build_verify_trained_model_parser`.

    Returns
    -------
    int
        Exit code (0).
    """
    from ..io import write_scores

    run = _prepare_run_from_args(args)
    scores = verify_mlcast_model(
        run.model,
        run.dataset,
        forecast_source=run.forecast_source,
        ensemble_size=args.ensemble_size,
        batch_size=args.batch_size,
        device=run.device,
        num_workers=args.num_workers,
        seed=args.seed,
        compute_psd=not args.no_psd,
        member_chunk=args.member_chunk,
        precision=args.precision,
        progress=not args.no_progress,
    )
    path = write_scores(
        scores,
        args.out,
        title=args.title or f"Verification scores of {run.forecast_source} ({args.split} split)",
        forecasts=args.forecasts or _describe_forecasts(run, args.ensemble_size, args.precision),
        truth=args.truth or _describe_truth(run.dataset),
        created_by=args.created_by,
        created_with=args.created_with,
        license=args.license,
        dataset_version=args.dataset_version,
    )
    print(f"wrote scores of {len(run.dataset)} cases to {path}")
    return 0


def forecast_trained_model_from_args(args: argparse.Namespace) -> int:
    """
    Run ``forecast-trained-model``: load the run and write its forecasts to a Zarr store.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments of
        :func:`mlcast_verification_tools.integrations.mlcast.build_forecast_trained_model_parser`.

    Returns
    -------
    int
        Exit code (0).
    """
    run = _prepare_run_from_args(args)
    _forecast_run(
        run,
        ensemble_size=args.ensemble_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        member_chunk=args.member_chunk,
        precision=args.precision,
        seed=args.seed,
        out=args.out,
        progress=not args.no_progress,
    )
    print(f"wrote forecasts of {len(run.dataset)} cases to {args.out}")
    return 0
