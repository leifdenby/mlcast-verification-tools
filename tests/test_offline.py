from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from mlcast_verification_tools import conventions as cv
from mlcast_verification_tools.offline import compute_scores, infer_dx_km

SIZE = 32
N_INIT, N_LEAD = 6, 4
STEP = pd.Timedelta("5min")
MEMBERS = {"model_a": 5, "model_b": 3}


def _synthetic(seed=0):
    """`ds_forecasts` with a forecast_source dim (model_b NaN-padded) and `ds_truth`."""
    rng = np.random.default_rng(seed)
    x = np.arange(SIZE) * 1000.0  # metres -> dx = 1 km
    y = np.arange(SIZE) * 1000.0
    init_time = pd.date_range("2025-06-01", periods=N_INIT, freq="10min")
    lead_time = pd.timedelta_range(STEP, periods=N_LEAD, freq=STEP)
    # truth covers all valid times except the last one (-> those cases are all-NaN)
    times = pd.date_range(init_time[0] + STEP, init_time[-1] + lead_time[-2], freq=STEP)
    truth = rng.gamma(0.5, 2.0, size=(len(times), SIZE, SIZE))
    truth[:, :4, :] = np.nan  # a permanently invalid strip
    ds_truth = xr.Dataset(
        {"tp": (cv.TRUTH_DIMS, truth, {"units": "mm h-1"})},
        coords={"time": times, "y": y, "x": x},
    )

    k_max = max(MEMBERS.values())
    fc = np.full((len(MEMBERS), N_INIT, N_LEAD, k_max, SIZE, SIZE), np.nan)
    for i, k in enumerate(MEMBERS.values()):
        fc[i, :, :, :k] = rng.gamma(0.5, 2.0, size=(N_INIT, N_LEAD, k, SIZE, SIZE))
        fc[i, :, :, 0, -2:, :] = np.nan  # one member missing at a strip -> invalid
    ds_forecasts = xr.Dataset(
        {"tp": (cv.FORECAST_DIMS, fc, {"units": "mm h-1"})},
        coords={
            "forecast_source": list(MEMBERS),
            "init_time": init_time,
            "lead_time": lead_time,
            "member": np.arange(k_max),
            "y": y,
            "x": x,
        },
    )
    return ds_forecasts, ds_truth


def _last_valid(ds):
    return (ds[cv.INIT_TIME] == ds[cv.INIT_TIME][-1]) & (ds[cv.LEAD_TIME] == ds[cv.LEAD_TIME][-1])


def test_compute_scores_end_to_end():
    ds_forecasts, ds_truth = _synthetic()
    ds = compute_scores(ds_forecasts, ds_truth, batch_size=4, device="cpu")

    assert list(ds[cv.FORECAST_SOURCE].values) == ["model_a", "model_b"]
    assert ds.sizes == {
        cv.FORECAST_SOURCE: 2,
        cv.INIT_TIME: N_INIT,
        cv.LEAD_TIME: N_LEAD,
        cv.RANK: 6,
        cv.WAVELENGTH: SIZE // 2,
    }
    np.testing.assert_allclose(ds[cv.WAVELENGTH].values, SIZE * 1.0 / np.arange(1, SIZE // 2 + 1))

    mae = ds[cv.var_name(cv.MAE)]
    last_valid = _last_valid(ds)
    assert mae.where(last_valid).count() == 0  # no truth at the last valid time
    assert mae.where(~last_valid).count() == 2 * (N_INIT * N_LEAD - 1)

    rh = ds[cv.var_name(cv.RANK_HISTOGRAM)]
    # model_b is scored with its 3 members (padding dropped): ranks 5 and 6 are NaN
    assert rh.sel(forecast_source="model_b", rank=[5, 6]).isnull().all()
    assert (
        rh.sel(forecast_source="model_b", rank=[1, 2, 3, 4]).where(~last_valid, 0.0).notnull().all()
    )
    sums = rh.sum(cv.RANK, min_count=1).where(~last_valid)
    np.testing.assert_allclose(sums.values[~np.isnan(sums.values)], 1.0, rtol=1e-5)

    assert (ds[cv.var_name(cv.CRPS)].where(~last_valid, 1.0) > 0).all()
    assert (ds[cv.var_name(cv.PSD)].where(~last_valid, 1.0) > 0).all()


def test_padded_source_equals_unpadded():
    ds_forecasts, ds_truth = _synthetic(seed=2)
    both = compute_scores(ds_forecasts, ds_truth, batch_size=2, device="cpu")
    alone = compute_scores(
        ds_forecasts.sel(forecast_source=["model_b"], member=slice(0, 2)),
        ds_truth,
        batch_size=2,
        device="cpu",
    )
    xr.testing.assert_allclose(both.sel(forecast_source=["model_b"], rank=[1, 2, 3, 4]), alone)


def test_scalar_forecast_source_matches_manual_mae():
    ds_forecasts, ds_truth = _synthetic(seed=1)
    ds_b = ds_forecasts.sel(forecast_source="model_b")  # scalar coordinate
    assert cv.FORECAST_SOURCE not in ds_b.dims
    ds = compute_scores(ds_b, ds_truth, dx_km=1.0, device="cpu")
    assert list(ds[cv.FORECAST_SOURCE].values) == ["model_b"]
    assert ds.sizes[cv.RANK] == MEMBERS["model_b"] + 1

    fc = ds_b["tp"].isel(init_time=2, lead_time=1, member=slice(0, MEMBERS["model_b"]))
    valid_time = fc.init_time.values + fc.lead_time.values
    o = ds_truth["tp"].sel(time=valid_time)
    valid = np.isfinite(o) & np.isfinite(fc).all("member")
    expected = float(np.abs(fc.mean("member") - o).where(valid).mean())
    got = float(ds[cv.var_name(cv.MAE)].isel(forecast_source=0, init_time=2, lead_time=1))
    assert got == pytest.approx(expected, rel=1e-5)


def test_forecast_source_argument():
    ds_forecasts, ds_truth = _synthetic()
    ds_a = ds_forecasts.isel(forecast_source=0, drop=True)
    with pytest.raises(ValueError, match="forecast_source"):
        compute_scores(ds_a, ds_truth, device="cpu")
    ds = compute_scores(ds_a, ds_truth, forecast_source="mine", device="cpu")
    assert list(ds[cv.FORECAST_SOURCE].values) == ["mine"]
    with pytest.raises(ValueError, match="forecast_source"):
        compute_scores(ds_forecasts, ds_truth, forecast_source="mine", device="cpu")


def test_native_units():
    """Scores are in the units of the data; the rank threshold follows the units."""
    ds_forecasts, ds_truth = _synthetic(seed=3)
    expected = compute_scores(ds_forecasts, ds_truth, batch_size=3, device="cpu")
    per_second = {"units": "kg m-2 s-1"}
    fc_s = ds_forecasts.assign(tp=(ds_forecasts["tp"] / 3600).assign_attrs(per_second))
    truth_s = ds_truth.assign(tp=(ds_truth["tp"] / 3600).assign_attrs(per_second))
    got = compute_scores(fc_s, truth_s, batch_size=3, device="cpu")
    assert got["mae.tp"].attrs["units"] == "kg m-2 s-1"
    assert got["psd.tp"].attrs["units"] == "kg2 m-4 s-2 km"
    for name in ("mae.tp", "crps.tp"):
        np.testing.assert_allclose(got[name], expected[name] / 3600, rtol=1e-4)
    # 0.037 mm/h -> 0.037 / 3600 kg m-2 s-1: same ranks
    xr.testing.assert_allclose(got["rank_histogram.tp"], expected["rank_histogram.tp"])
    for name in ("psd.tp", "psd_obs.tp"):
        np.testing.assert_allclose(got[name], expected[name] / 3600**2, rtol=1e-4)


def test_mismatched_units_raise():
    ds_forecasts, ds_truth = _synthetic()
    ds_truth["tp"].attrs["units"] = "kg m-2 s-1"
    with pytest.raises(ValueError, match="different units"):
        compute_scores(ds_forecasts, ds_truth, device="cpu")
    ds_truth["tp"].attrs["units"] = "mm  H-1"  # case and whitespace are ignored
    compute_scores(ds_forecasts, ds_truth, device="cpu")


def test_reflectivity_in_dbz():
    """Any variable name and units: scores named after the variable, in its units."""
    ds_forecasts, ds_truth = _synthetic(seed=4)
    as_dbz = {"units": "dBZ"}
    fc = ds_forecasts.rename(tp="dbz").assign(dbz=lambda d: (d.dbz * 10).assign_attrs(as_dbz))
    truth = ds_truth.rename(tp="dbz").assign(dbz=lambda d: (d.dbz * 10).assign_attrs(as_dbz))
    got = compute_scores(fc, truth, variable="dbz", device="cpu")
    assert {"mae.dbz", "crps.dbz", "rank_histogram.dbz", "psd.dbz", "psd_obs.dbz"} == set(
        got.data_vars
    )
    assert got["mae.dbz"].attrs["units"] == "dBZ" and got["psd.dbz"].attrs["units"] == "dBZ2 km"
    with pytest.raises(ValueError, match="zero threshold"):
        compute_scores(
            fc.assign(dbz=fc.dbz.assign_attrs(units="K")),
            truth.assign(dbz=truth.dbz.assign_attrs(units="K")),
            variable="dbz",
            device="cpu",
        )
    compute_scores(
        fc.assign(dbz=fc.dbz.assign_attrs(units="K")),
        truth.assign(dbz=truth.dbz.assign_attrs(units="K")),
        variable="dbz",
        zero_threshold=0.0,
        device="cpu",
    )


def test_missing_units_raises():
    ds_forecasts, ds_truth = _synthetic()
    no_units = ds_forecasts.assign(tp=ds_forecasts["tp"].copy())
    no_units["tp"].attrs.pop("units")
    with pytest.raises(ValueError, match="units"):
        compute_scores(no_units, ds_truth, device="cpu")
    ds_truth["tp"].attrs.pop("units")
    with pytest.raises(ValueError, match="units"):
        compute_scores(ds_forecasts, ds_truth, device="cpu")


def test_bad_layouts_raise():
    ds_forecasts, ds_truth = _synthetic()
    with pytest.raises(ValueError, match="no data variable"):
        compute_scores(ds_forecasts, ds_truth, variable="rr", device="cpu")
    with pytest.raises(ValueError, match="dims"):
        compute_scores(ds_forecasts.isel(member=0), ds_truth, device="cpu")
    with pytest.raises(ValueError, match="coordinates"):
        compute_scores(ds_forecasts, ds_truth.assign_coords(x=ds_truth.x + 500), device="cpu")


def test_dask_backed_inputs():
    pytest.importorskip("dask")
    ds_forecasts, ds_truth = _synthetic(seed=4)
    expected = compute_scores(ds_forecasts, ds_truth, batch_size=2, device="cpu")
    got = compute_scores(
        ds_forecasts.chunk({"init_time": 1}),
        ds_truth.chunk({"time": 5}),
        batch_size=2,
        device="cpu",
    )
    xr.testing.assert_allclose(got, expected)


def test_infer_dx_km():
    da = xr.DataArray(np.zeros(4), dims="x", coords={"x": [0.0, 500.0, 1000.0, 1500.0]})
    assert infer_dx_km(da) == pytest.approx(0.5)
    da = da.assign_coords(x=[0.0, 2.0, 4.0, 6.0])
    assert infer_dx_km(da) == pytest.approx(2.0)
    assert infer_dx_km(da.to_dataset(name="tp")) == pytest.approx(2.0)
