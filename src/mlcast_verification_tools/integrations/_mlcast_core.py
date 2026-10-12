"""Torch/mlcast implementation of :mod:`mlcast_verification_tools.integrations.mlcast`.

Kept separate so that importing the public module (done by the CLI for every
sub-command) does not import torch or mlcast. Import from the public module.
"""

from __future__ import annotations

import argparse
import functools
import importlib
import math
import os
import pathlib
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch
import xarray as xr
from mlcast.data.normalization import DEFAULT_SCALING, ReflectivityScaling
from torch.utils.data import DataLoader, Dataset

from ..accumulator import ScoreAccumulator
from .mlcast import SUPPORTED_STANDARD_NAMES

if TYPE_CHECKING:  # pragma: no cover
    import fiddle as fdl
    from mlcast.nowcasting_module import NowcastLightningModule

_UNITS_FACTORS = {
    # rates per hour: already mm/h (1 kg m-2 of water = 1 mm)
    **dict.fromkeys(
        (
            "mm/h",
            "mmh-1",
            "mmh^-1",
            "mm/hr",
            "mmhr-1",
            "kgm-2h-1",
            "kgm^-2h^-1",
            "kg/m2/h",
            "kgm-2hr-1",
        ),
        1.0,
    ),
    # rates per second
    **dict.fromkeys(("kgm-2s-1", "kgm^-2s^-1", "kg/m2/s", "mm/s", "mms-1", "mms^-1"), 3600.0),
}

# training-time path of a TensorBoard run is <log_dir>/checkpoints/<name>.ckpt with the
# Fiddle config written to <log_dir>/config.yaml (mlcast.config.orchestrator)
_CONFIG_YAML_NAME = "config.yaml"
_NETWORK_CONFIG_PATH = "pl_module.network"


# ---------------------------------------------------------------------------
# Units and (de)normalisation
# ---------------------------------------------------------------------------


def _check_standard_name(standard_name: str) -> None:
    if standard_name not in SUPPORTED_STANDARD_NAMES:
        raise ValueError(
            f"standard_name {standard_name!r} is not supported, expected one of "
            f"{SUPPORTED_STANDARD_NAMES}"
        )


def mm_per_hour_factor(units: str | None) -> float:
    """
    Factor converting values in `units` to mm/h.

    Parameters
    ----------
    units : str or None
        CF ``units`` attribute of a rain-rate variable, e.g. ``"mm/h"``,
        ``"kg m-2 h-1"`` or ``"kg m-2 s-1"``. Case and whitespace are ignored.

    Returns
    -------
    float
        Multiply values by this factor to get mm/h.

    Raises
    ------
    ValueError
        If the units are missing or not recognised.
    """
    key = (units or "").strip().lower().replace(" ", "").replace("**", "^")
    if key not in _UNITS_FACTORS:
        raise ValueError(
            f"cannot convert units {units!r} to mm/h; pass the conversion factor explicitly "
            "(`to_mm_per_hour=` / `--to-mm-per-hour`)"
        )
    return _UNITS_FACTORS[key]


def denormalize_torch(
    x: torch.Tensor,
    standard_name: str = "rainfall_rate",
    scaling: ReflectivityScaling | None = None,
) -> torch.Tensor:
    """
    Torch port of mlcast's inverse reflectivity scaling (normalised -> rain rate).

    Maps the normalised range ``[norm_min, norm_max]`` linearly onto
    ``[dbz_floor, dbz_ceiling]`` dBZ, then applies the inverse Marshall-Palmer
    relation ``R = 10 ** ((dBZ - 10 log10(a)) / (10 b))``; values at or below
    ``dbz_floor`` become exactly 0 and NaN propagates, as in
    ``ReflectivityScaling.normalized_to_rainfall_rate``. Runs on the device of `x`.

    Parameters
    ----------
    x : torch.Tensor
        Normalised values (floating point), any shape and device.
    standard_name : {"rainfall_rate", "rainfall_flux"}
        CF standard name of the variable the model was trained on. Both use the
        same maths (mlcast reuses the mm/h relation for ``rainfall_flux``), so the
        result is in the units of the source variable; see :func:`mm_per_hour_factor`.
    scaling : mlcast.data.normalization.ReflectivityScaling, optional
        Scaling constants; defaults to mlcast's ``DEFAULT_SCALING``.

    Returns
    -------
    torch.Tensor
        Rain rate in the units of the source variable, same shape/dtype/device as `x`.
    """
    _check_standard_name(standard_name)
    if not torch.is_floating_point(x):
        raise TypeError(f"expected a floating point tensor, got {x.dtype}")
    s = scaling or DEFAULT_SCALING
    dbz = s.dbz_floor + (x - s.norm_min) * ((s.dbz_ceiling - s.dbz_floor) / (s.norm_max - s.norm_min))
    rate = torch.pow(10.0, (dbz - 10.0 * math.log10(s.a)) / (10.0 * s.b))
    return torch.where(dbz <= s.dbz_floor, torch.zeros((), dtype=rate.dtype, device=rate.device), rate)


def _normalize_numpy(values: np.ndarray, standard_name: str, scaling: ReflectivityScaling) -> np.ndarray:
    """mlcast's forward normalisation, without its per-call ``rainfall_flux`` warning."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Assuming we can use the same function")
        return scaling.normalization_registry[standard_name](values)


# ---------------------------------------------------------------------------
# Fiddle YAML configs and checkpoints
# ---------------------------------------------------------------------------


def _resolve(module: str, qualname: str) -> Any:
    obj = importlib.import_module(module)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def _yaml_loader():
    import fiddle as fdl
    import yaml

    class FiddleYamlLoader(yaml.SafeLoader):
        """SafeLoader that understands the tags written by Fiddle's ``dump_yaml``."""

    def buildable(cls, loader, node):
        kwargs = loader.construct_mapping(node, deep=True)
        target = kwargs.pop("__fn_or_cls__", None)
        if not isinstance(target, dict) or {"module", "name"} - set(target):
            raise ValueError(f"{node.tag} node without a valid __fn_or_cls__ entry: {target!r}")
        return cls(_resolve(target["module"], target["name"]), **kwargs)

    def tagged_value(loader, node):
        # tags are only used to select values when editing configs; keep the value
        return loader.construct_mapping(node, deep=True)["value"]

    def python_type(loader, node):
        # mlcast.config.orchestrator registers `!type module.qualname` for classes
        module, _, qualname = loader.construct_scalar(node).rpartition(".")
        try:
            return _resolve(module, qualname)
        except (ImportError, AttributeError):
            # qualname of a nested class: walk down from the longest importable module
            parts = loader.construct_scalar(node).split(".")
            for i in range(len(parts) - 1, 0, -1):
                try:
                    return _resolve(".".join(parts[:i]), ".".join(parts[i:]))
                except (ImportError, AttributeError):
                    continue
            raise

    FiddleYamlLoader.add_constructor("!fdl.Config", functools.partial(buildable, fdl.Config))
    FiddleYamlLoader.add_constructor("!fdl.Partial", functools.partial(buildable, fdl.Partial))
    FiddleYamlLoader.add_constructor("!fdl.TaggedValue", tagged_value)
    FiddleYamlLoader.add_constructor("!type", python_type)
    return FiddleYamlLoader


def load_fiddle_yaml(path_or_text: str | os.PathLike) -> fdl.Config:
    """
    Load a Fiddle config written by ``fiddle.experimental.yaml_serialization.dump_yaml``.

    Supports the ``!fdl.Config``, ``!fdl.Partial`` and ``!fdl.TaggedValue`` tags
    written by Fiddle 0.3 and the ``!type`` tag mlcast registers for classes.

    .. warning:: Loading imports every module named in the YAML, so only load
       configs you trust.

    Parameters
    ----------
    path_or_text : str or os.PathLike
        Path to the YAML file, or the YAML text itself.

    Returns
    -------
    fdl.Config
        The config graph (not built), e.g. ``fdl.build(cfg.pl_module.network)``.
    """
    import yaml

    text = str(path_or_text)
    if isinstance(path_or_text, os.PathLike) or ("\n" not in text and pathlib.Path(text).is_file()):
        text = pathlib.Path(path_or_text).read_text()
    return yaml.load(text, Loader=_yaml_loader())


def build_network_from_config_yaml(
    config_yaml: str | os.PathLike, key: str = _NETWORK_CONFIG_PATH
) -> torch.nn.Module:
    """
    Build the network of an mlcast run from its Fiddle ``config.yaml``.

    Parameters
    ----------
    config_yaml : str or os.PathLike
        Path to the ``config.yaml`` mlcast wrote at the start of training (or its text).
    key : str
        Dotted path of the network config in the experiment config.

    Returns
    -------
    torch.nn.Module
        Freshly initialised network, ready for ``load_state_dict``.
    """
    import fiddle as fdl

    cfg = load_fiddle_yaml(config_yaml)
    sub = cfg
    for part in key.split("."):
        try:
            sub = getattr(sub, part) if isinstance(sub, fdl.Buildable) else sub[part]
        except (AttributeError, KeyError, TypeError) as exc:
            raise KeyError(f"no {key!r} in the config (missing {part!r})") from exc
    return fdl.build(sub)


def _find_config_yaml(checkpoint_path: pathlib.Path) -> pathlib.Path | None:
    for directory in (checkpoint_path.parent, checkpoint_path.parent.parent):
        candidate = directory / _CONFIG_YAML_NAME
        if candidate.is_file():
            return candidate
    return None


def load_mlcast_model(
    checkpoint_path: str | os.PathLike,
    *,
    network: torch.nn.Module | None = None,
    config_yaml: str | os.PathLike | None = None,
    device: str | torch.device = "cpu",
) -> NowcastLightningModule:
    """
    Load a trained ``NowcastLightningModule`` from a Lightning checkpoint.

    The network architecture is not stored in the checkpoint, so it is taken from,
    in order: `network`; the Fiddle config `config_yaml`; a ``config.yaml`` next to
    the checkpoint or in its parent directory (mlcast's TensorBoard layout
    ``<run>/checkpoints/*.ckpt`` + ``<run>/config.yaml``). If none is available,
    mlcast's own ``from_checkpoint`` is tried (it fails for current mlcast
    checkpoints, which do not store the network).

    Parameters
    ----------
    checkpoint_path : str or os.PathLike
        Lightning ``.ckpt`` file.
    network : torch.nn.Module, optional
        Network with the architecture used in training (weights are overwritten).
    config_yaml : str or os.PathLike, optional
        The run's Fiddle ``config.yaml``, used to rebuild the network.
    device : str or torch.device
        Device to load the weights to.

    Returns
    -------
    NowcastLightningModule
        Model in eval mode on `device`.
    """
    from mlcast.nowcasting_module import NowcastLightningModule

    checkpoint_path = pathlib.Path(checkpoint_path)
    if network is None:
        if config_yaml is None:
            config_yaml = _find_config_yaml(checkpoint_path)
        if config_yaml is not None:
            network = build_network_from_config_yaml(config_yaml)

    kwargs: dict[str, Any] = {
        "map_location": torch.device(device),
        "strict": True,
        "weights_only": False,
    }
    if network is None:
        try:
            model = NowcastLightningModule.load_from_checkpoint(checkpoint_path, **kwargs)
        except TypeError as exc:
            raise ValueError(
                f"cannot restore {checkpoint_path}: the network is not stored in mlcast "
                "checkpoints. Pass `network=` or `config_yaml=` (the run's Fiddle config.yaml)."
            ) from exc
    else:
        model = NowcastLightningModule.load_from_checkpoint(checkpoint_path, network=network, **kwargs)
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
        raise ValueError(f"cannot infer the grid spacing from coordinate {coord.name!r}; pass dx_km")
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
    to_mm_per_hour : float, optional
        Factor converting the source values to mm/h; default from the ``units``
        attribute (:func:`mm_per_hour_factor`).
    dx_km : float, optional
        Grid spacing in km; default from the X coordinate spacing (metres -> km).
    storage_options : dict, optional
        Passed to ``xr.open_zarr`` when `source` is a path/URL.

    Notes
    -----
    Items are dicts with ``input`` (T_in, 1, H, W) float32 normalised, ``target``
    (T_out, H, W) float32 in mm/h with NaN kept, ``mask`` (H, W) bool and
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
        to_mm_per_hour: float | None = None,
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
        self.units = da.attrs.get("units")
        self.to_mm_per_hour = (
            float(to_mm_per_hour) if to_mm_per_hour is not None else mm_per_hour_factor(self.units)
        )

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
            raise ValueError(f"variable {variable!r} has no standard_name attribute; pass standard_name")
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
        offsets = np.arange(-(self.input_steps - 1), self.forecast_steps + 1) * self.time_step.to_timedelta64()
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
        target = raw[self.input_steps :].astype(np.float32) * np.float32(self.to_mm_per_hour)
        return {
            "input": torch.from_numpy(np.ascontiguousarray(inputs[:, None], dtype=np.float32)),
            "target": torch.from_numpy(np.ascontiguousarray(target)),
            "mask": torch.from_numpy(mask),
            "init_time": torch.tensor(self.init_times[idx].astype("datetime64[ns]").astype(np.int64)),
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
        raise ValueError("member_chunk must be at least 2 for an ensemble (a 1-member call is deterministic)")
    chunks = [member_chunk] * (ensemble_size // member_chunk)
    rest = ensemble_size % member_chunk
    if rest == 1:
        chunks[-1] += 1
    elif rest:
        chunks.append(rest)
    return chunks


def _forecast(model, x: torch.Tensor, forecast_steps: int, chunks: list[int]) -> torch.Tensor:
    """Run the model for all members, ``(B, T, M, H, W)`` normalised."""
    b, _, _, h, w = x.shape
    preds = None
    start = 0
    for k in chunks:
        out = model(x, forecast_steps=forecast_steps, ensemble_size=k)
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
    progress: bool = True,
) -> xr.Dataset:
    """
    Score an mlcast model on the cases of `dataset` without storing forecasts.

    For each batch: ``model(x, forecast_steps=T, ensemble_size=K)`` -> clamp to
    ``[norm_min, norm_max]`` (as in training) -> :func:`denormalize_torch` -> mm/h ->
    :meth:`ScoreAccumulator.update` on the device; the forecast tensor is then freed.

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
        DataLoader workers reading the source data.
    seed : int
        Seeds torch's global RNG (the ConvGRU noise) and the rank-histogram tie breaking.
    compute_psd : bool
        Whether to compute radial power spectra.
    member_chunk : int, optional
        Generate the members in model calls of at most this many members (at least 2)
        to lower the peak activation memory. The encoder is re-run for each call and
        the full ``(B, T, M, H, W)`` forecast is still assembled on the device.
    progress : bool
        Show a tqdm progress bar.

    Returns
    -------
    xr.Dataset
        Score dataset in the spec layout, without global attributes.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device).eval()
    chunks = _member_chunks(ensemble_size, member_chunk)
    scaling = dataset.scaling
    acc = ScoreAccumulator(
        forecast_source, dataset.lead_times, dataset.dx_km, seed=seed, compute_psd=compute_psd
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    batches = loader
    if progress:
        try:
            from tqdm.auto import tqdm

            batches = tqdm(loader, desc=f"verifying {forecast_source}", unit="batch")
        except ImportError:  # pragma: no cover
            pass

    torch.manual_seed(seed)  # also seeds all CUDA devices
    with torch.inference_mode():
        for batch in batches:
            x = batch["input"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            obs = batch["target"].to(device, non_blocking=True)
            obs = torch.nan_to_num(obs, nan=0.0)  # NaN obs are outside `mask`
            init_times = batch["init_time"].numpy().astype("datetime64[ns]")

            preds = _forecast(model, x, dataset.forecast_steps, chunks)
            preds = preds.clamp_(scaling.norm_min, scaling.norm_max)
            preds = denormalize_torch(preds, dataset.standard_name, scaling)
            if dataset.to_mm_per_hour != 1.0:
                preds.mul_(dataset.to_mm_per_hour)
            acc.update(preds, obs, mask, init_times)
            del preds, x, obs, mask
    return acc.to_dataset()


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def _describe_truth(dataset: VerificationDataset) -> str:
    attrs = dataset.ds.attrs
    name = attrs.get("title") or attrs.get("mlcast_dataset_identifier") or str(dataset._path or "source data")
    ident = attrs.get("mlcast_dataset_identifier")
    version = attrs.get("mlcast_dataset_version")
    tag = f" ({ident} v{version})" if ident and version else ""
    return (
        f"{name}{tag}, variable {dataset.variable!r} ({dataset.standard_name}, "
        f"{dataset.units}) converted to mm/h"
    )


def run_mlcast_from_args(args: argparse.Namespace) -> int:
    """
    Run ``mlcast.verify run-mlcast``: load, verify and write the scores.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments of
        :func:`mlcast_verification_tools.integrations.mlcast.add_run_mlcast_arguments`.

    Returns
    -------
    int
        Exit code (0).
    """
    from ..io import write_scores

    region = {k: v for k, v in (("y", args.region_y), ("x", args.region_x)) if v is not None}
    dataset = VerificationDataset(
        args.source,
        init_times_from_range(args.start, args.end, args.freq),
        args.input_steps,
        args.forecast_steps,
        variable=args.variable,
        region=region or None,
        standard_name=args.standard_name,
        time_step=args.time_step,
        on_missing=args.on_missing,
        to_mm_per_hour=args.to_mm_per_hour,
        storage_options=args.storage_options,
    )
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_mlcast_model(args.checkpoint, config_yaml=args.config_yaml, device=device)
    scores = verify_mlcast_model(
        model,
        dataset,
        forecast_source=args.forecast_source,
        ensemble_size=args.ensemble_size,
        batch_size=args.batch_size,
        device=device,
        num_workers=args.num_workers,
        seed=args.seed,
        compute_psd=not args.no_psd,
        member_chunk=args.member_chunk,
        progress=not args.no_progress,
    )
    forecasts = args.forecasts or (
        f"{args.forecast_source}: mlcast model {pathlib.Path(args.checkpoint).name}, "
        f"{args.ensemble_size}-member ensemble from {args.input_steps} input frames"
    )
    path = write_scores(
        scores,
        args.out,
        title=args.title,
        forecasts=forecasts,
        truth=args.truth or _describe_truth(dataset),
        created_by=args.created_by,
        created_with=args.created_with,
        license=args.license,
        dataset_version=args.dataset_version,
    )
    print(f"wrote scores of {len(dataset)} cases to {path}")
    return 0
