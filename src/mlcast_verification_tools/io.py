"""Reading, writing and validating score datasets.

Score datasets are stored in the MLCast `verification/forecast_scoring_metrics`
layout (spec v0.1.0), either as a netCDF file (suffix `.nc`, `.nc4` or `.cdf`)
or as a Zarr v3 store (any other path). The spec's validator,
`mlcast-dataset-validator`, is an optional dependency that is only needed by
`validate_scores`.
"""

from __future__ import annotations

import datetime
import os
import pathlib
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr

if TYPE_CHECKING:  # pragma: no cover
    from mlcast_dataset_validator.specs.reporting import ValidationReport

NETCDF_SUFFIXES = (".nc", ".nc4", ".cdf")

# Validator version claimed in `mlcast_dataset_validator_version` when the
# validator isn't installed (same default as the validator's example script).
DEFAULT_VALIDATOR_VERSION = "0.4.0"

PathLike = str | os.PathLike


def _validator_version() -> str:
    """Release version of the installed mlcast-dataset-validator, if any."""
    try:
        from mlcast_dataset_validator import __version__ as validator_version
        from packaging.version import Version
    except ImportError:
        return DEFAULT_VALIDATOR_VERSION
    return Version(validator_version).base_version


def _is_netcdf(path: PathLike) -> bool:
    return str(path).lower().endswith(NETCDF_SUFFIXES)


def add_global_attrs(
    ds: xr.Dataset,
    *,
    title: str,
    forecasts: str,
    truth: str,
    created_by: str,
    created_with: str,
    license: str = "CC-BY-4.0",
    dataset_version: str = "0.1.0",
    created_on: str | None = None,
) -> xr.Dataset:
    """
    Add the global attributes required/recommended by the spec (§5).

    Parameters
    ----------
    ds : xr.Dataset
        Score dataset.
    title : str
        Short description of the dataset (`title`, §5.1).
    forecasts : str
        Description of the forecast sources verified (`forecasts`, §5.1).
    truth : str
        Description of the truth source the forecasts are verified against
        (`truth`, §5.1).
    created_by : str
        Creator contact in `Name <email>` format (`mlcast_created_by`, §5.4).
    created_with : str
        GitHub URL with `@version` of the repository that created the dataset
        (`mlcast_created_with`, §5.4), e.g.
        `https://github.com/mlcast-community/mlcast-dataset-dmi-nowcast-scores@v0.1.0`.
        The spec requires the repository to be named
        `mlcast-dataset-{organisation_id}-{dataset_name}`, so this is the repository
        of the score dataset, not of mlcast-verification-tools.
    license : str, optional
        SPDX license identifier (`license`, §5.2).
    dataset_version : str, optional
        Dataset specification version (`mlcast_dataset_version`, §5.4).
    created_on : str, optional
        ISO 8601 creation time (`mlcast_created_on`, §5.4). Defaults to now
        (UTC, seconds precision).

    Returns
    -------
    xr.Dataset
        A shallow copy of `ds` with the global attributes set.
    """
    if created_on is None:
        created_on = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
    ds = ds.copy()
    ds.attrs.update(
        {
            "title": title,
            "forecasts": forecasts,
            "truth": truth,
            "license": license,
            "mlcast_created_on": created_on,
            "mlcast_created_by": created_by,
            "mlcast_created_with": created_with,
            "mlcast_dataset_version": dataset_version,
            "mlcast_dataset_validator_version": _validator_version(),
        }
    )
    return ds


def _netcdf_encoding(ds: xr.Dataset) -> dict[str, dict[str, Any]]:
    """netCDF encodings: no `_FillValue` on coordinates, explicit time units."""
    encoding: dict[str, dict[str, Any]] = {}
    for name, coord in ds.coords.items():
        enc: dict[str, Any] = {"_FillValue": None}
        if np.issubdtype(coord.dtype, np.timedelta64):
            enc.update(units="seconds", dtype="int64")
        elif np.issubdtype(coord.dtype, np.datetime64):
            enc.update(units="seconds since 1970-01-01 00:00:00", dtype="int64")
        encoding[str(name)] = enc
    for name, var in ds.data_vars.items():
        if np.issubdtype(var.dtype, np.floating):
            encoding[str(name)] = {"_FillValue": np.nan}
    return encoding


def write_scores(ds: xr.Dataset, path: PathLike, **attrs_kwargs: Any) -> pathlib.Path:
    """
    Write a score dataset to netCDF or Zarr.

    Parameters
    ----------
    ds : xr.Dataset
        Score dataset in the spec layout.
    path : str or os.PathLike
        Output path. A suffix of `.nc`, `.nc4` or `.cdf` writes netCDF4, any
        other path a Zarr v3 store (with consolidated metadata, overwriting an
        existing store).
    **attrs_kwargs
        If given, passed to `add_global_attrs` before writing.

    Returns
    -------
    pathlib.Path
        The path written to.
    """
    path = pathlib.Path(path)
    if attrs_kwargs:
        ds = add_global_attrs(ds, **attrs_kwargs)

    # drop encodings carried over from a file `ds` was read from, they may
    # conflict with the target format
    ds = ds.copy()
    for var in ds.variables.values():
        var.encoding = {}

    if _is_netcdf(path):
        ds.to_netcdf(path, format="NETCDF4", engine="netcdf4", encoding=_netcdf_encoding(ds))
    else:
        with warnings.catch_warnings():
            # zarr warns that consolidated metadata is not part of the v3 spec
            warnings.filterwarnings("ignore", message=".*[Cc]onsolidated metadata.*")
            ds.to_zarr(path, mode="w", zarr_format=3, consolidated=True)
    return path


def open_scores(path: PathLike) -> xr.Dataset:
    """
    Open a score dataset written by `write_scores` (or any spec-conforming one).

    Parameters
    ----------
    path : str or os.PathLike
        netCDF file (suffix `.nc`, `.nc4` or `.cdf`) or Zarr store, the same
        distinction as made by the validator CLI.

    Returns
    -------
    xr.Dataset
        Lazily loaded dataset with `lead_time` as timedelta64 and `init_time`
        as datetime64.
    """
    if _is_netcdf(path):
        return xr.open_dataset(path, engine="netcdf4", decode_timedelta=True)
    return xr.open_zarr(path, decode_timedelta=True)


def validate_scores(ds_or_path: xr.Dataset | PathLike) -> ValidationReport:
    """
    Validate a score dataset against the `forecast_scoring_metrics` spec.

    Parameters
    ----------
    ds_or_path : xr.Dataset or str or os.PathLike
        Dataset, or path to one (opened with `open_scores`). The storage
        format check (§5.3) needs to know where the dataset was read from, so
        a dataset that was not opened from a file (no `ds.encoding["source"]`)
        fails §5.3.

    Returns
    -------
    mlcast_dataset_validator.specs.reporting.ValidationReport
        The validation report, e.g. use `.has_fails()` or `.console_print()`.

    Raises
    ------
    ImportError
        If `mlcast-dataset-validator` is not installed.
    """
    try:
        from mlcast_dataset_validator.specs.verification import forecast_scoring_metrics
    except ImportError as exc:
        raise ImportError(
            "validate_scores needs mlcast-dataset-validator (with the "
            "verification/forecast_scoring_metrics spec), install it with e.g. "
            "`pip install git+https://github.com/leifdenby/mlcast-dataset-validator"
            "@feat/verification-metrics-spec`"
        ) from exc

    ds = ds_or_path if isinstance(ds_or_path, xr.Dataset) else open_scores(ds_or_path)
    report, _ = forecast_scoring_metrics.validate_dataset(ds)
    return report
