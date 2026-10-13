from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from mlcast_verification_tools import conventions as cv
from mlcast_verification_tools.io import (
    REPOSITORY_URL,
    add_global_attrs,
    default_created_with,
    open_scores,
    validate_scores,
    write_scores,
)
from mlcast_verification_tools.testing import make_synthetic_scores

# explicit provenance; any GitHub repository is accepted by the spec
CREATED_WITH = f"{REPOSITORY_URL}@v0.1.0"

ATTRS = dict(
    title="Synthetic precipitation nowcast scores",
    forecasts="two synthetic ensembles",
    truth="synthetic radar composite",
    created_by="Jane Doe <jane@example.com>",
    created_with=CREATED_WITH,
    created_on="2026-10-12T00:00:00+00:00",
)

SUFFIXES = [".nc", ".zarr"]


@pytest.fixture(autouse=True)
def offline_metadata(monkeypatch):
    """Skip the GitHub lookups of `mlcast_created_with`, as the validator's tests do."""
    mlcast_metadata = pytest.importorskip(
        "mlcast_dataset_validator.checks.global_attributes.mlcast_metadata"
    )
    monkeypatch.setattr(mlcast_metadata, "github_repo_exists", lambda *a: True)
    monkeypatch.setattr(mlcast_metadata, "github_ref_exists", lambda *a: "tags/v0.1.0")


def _problems(report, statuses=("FAIL",)):
    return [r for r in report.results if r.status in statuses]


def test_add_global_attrs_returns_copy():
    ds = make_synthetic_scores()
    ds_attrs = add_global_attrs(ds, **ATTRS)
    assert ds.attrs == {}
    for key in [
        "title",
        "forecasts",
        "truth",
        "license",
        "mlcast_created_on",
        "mlcast_created_by",
        "mlcast_created_with",
        "mlcast_dataset_version",
        "mlcast_dataset_validator_version",
    ]:
        assert key in ds_attrs.attrs
    assert ds_attrs.attrs["license"] == "CC-BY-4.0"


def test_add_global_attrs_defaults():
    attrs = add_global_attrs(make_synthetic_scores(), title="t", forecasts="f", truth="r").attrs
    created_on = np.datetime64(attrs["mlcast_created_on"].removesuffix("+00:00"))
    assert abs(np.datetime64("now") - created_on) < np.timedelta64(1, "D")
    # provenance is optional in the spec: no creator contact, this package as software
    assert "mlcast_created_by" not in attrs
    assert attrs["mlcast_created_with"] == default_created_with()


@pytest.mark.parametrize(
    "version, revision",
    [
        ("0.1.dev2+gacc2d8aff", "acc2d8aff"),
        ("0.1.dev2+gacc2d8aff.d20261012", "acc2d8aff"),
        ("0.2.0", "v0.2.0"),
    ],
)
def test_default_created_with(version, revision):
    assert default_created_with(version) == f"{REPOSITORY_URL}@{revision}"


@pytest.mark.parametrize("suffix", SUFFIXES)
def test_validate_without_provenance(tmp_path, suffix):
    attrs = {k: v for k, v in ATTRS.items() if k not in ("created_by", "created_with")}
    path = write_scores(make_synthetic_scores(), tmp_path / f"scores{suffix}", **attrs)
    report = validate_scores(path)
    assert not report.has_fails(), _problems(report)


@pytest.mark.parametrize("suffix", SUFFIXES)
def test_round_trip(tmp_path, suffix):
    ds = make_synthetic_scores()
    path = write_scores(ds, tmp_path / f"scores{suffix}", **ATTRS)
    assert path.exists()
    ds_read = open_scores(path)

    assert ds_read[cv.LEAD_TIME].dtype.kind == "m"
    assert ds_read[cv.INIT_TIME].dtype.kind == "M"
    assert ds_read[cv.FORECAST_SOURCE].dtype.kind == "U"
    for name in ds.variables:
        np.testing.assert_array_equal(ds_read[name].values, ds[name].values)
        assert ds_read[name].dims == ds[name].dims
        assert ds_read[name].attrs == ds[name].attrs
        if name in ds.data_vars:
            assert ds_read[name].dtype == ds[name].dtype
    assert ds_read.attrs == add_global_attrs(ds, **ATTRS).attrs

    # rank histogram of the smaller ensemble is NaN-padded
    rank_hist = ds_read[cv.var_name(cv.RANK_HISTOGRAM)].sel(forecast_source="model_b")
    assert rank_hist.sel(rank=slice(6, None)).isnull().all()
    assert rank_hist.sel(rank=slice(1, 5)).notnull().all()


def test_netcdf_coords_have_no_fill_value(tmp_path):
    path = write_scores(make_synthetic_scores(), tmp_path / "scores.nc", **ATTRS)
    with xr.open_dataset(path, decode_cf=False) as raw:
        for name in raw.coords:
            assert "_FillValue" not in raw[name].attrs, name


def test_write_overwrites_zarr(tmp_path):
    path = tmp_path / "scores.zarr"
    write_scores(make_synthetic_scores(seed=0), path, **ATTRS)
    ds = make_synthetic_scores(seed=1)
    write_scores(ds, path, **ATTRS)
    xr.testing.assert_allclose(open_scores(path)[cv.var_name(cv.MAE)], ds[cv.var_name(cv.MAE)])


def test_rewrite_dataset_read_from_other_format(tmp_path):
    nc_path = write_scores(make_synthetic_scores(), tmp_path / "scores.nc", **ATTRS)
    zarr_path = write_scores(open_scores(nc_path), tmp_path / "scores.zarr")
    xr.testing.assert_identical(open_scores(zarr_path).load(), open_scores(nc_path).load())


@pytest.mark.parametrize("suffix", SUFFIXES)
def test_validate_scores_passes(tmp_path, suffix):
    path = write_scores(make_synthetic_scores(), tmp_path / f"scores{suffix}", **ATTRS)
    report = validate_scores(path)
    assert not report.has_fails(), _problems(report)
    assert not report.has_warnings(), _problems(report, ("WARNING",))
    # the same when passing the opened dataset
    assert not validate_scores(open_scores(path)).has_fails()


def test_validate_in_memory_dataset_fails_storage_format():
    """§5.3 needs to know the storage format, which an in-memory dataset doesn't have."""
    report = validate_scores(add_global_attrs(make_synthetic_scores(), **ATTRS))
    assert {r.section for r in _problems(report)} == {"5.3"}
