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


def _synthetic(seed=0):
    rng = np.random.default_rng(seed)
    x = np.arange(SIZE) * 1000.0  # metres -> dx = 1 km
    y = np.arange(SIZE) * 1000.0
    init_time = pd.date_range("2025-06-01", periods=N_INIT, freq="10min")
    lead_time = pd.timedelta_range(STEP, periods=N_LEAD, freq=STEP)
    # obs cover all valid times except the last one (-> those cases are all-NaN)
    times = pd.date_range(init_time[0] + STEP, init_time[-1] + lead_time[-2], freq=STEP)
    obs_vals = rng.gamma(0.5, 2.0, size=(len(times), SIZE, SIZE))
    obs_vals[:, :4, :] = np.nan  # a permanently invalid strip
    obs = xr.DataArray(obs_vals, dims=("time", "y", "x"), coords={"time": times, "y": y, "x": x})

    forecasts = {}
    for name, k in (("model_a", 5), ("model_b", 3)):
        vals = rng.gamma(0.5, 2.0, size=(N_INIT, N_LEAD, k, SIZE, SIZE))
        vals[:, :, 0, -2:, :] = np.nan  # one member missing at a strip -> invalid
        forecasts[name] = xr.DataArray(
            vals,
            dims=("init_time", "lead_time", "member", "y", "x"),
            coords={"init_time": init_time, "lead_time": lead_time, "y": y, "x": x},
        )
    return forecasts, obs


def test_compute_scores_end_to_end():
    forecasts, obs = _synthetic()
    ds = compute_scores(forecasts, obs, batch_size=4, device="cpu")

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
    last_valid = (ds[cv.INIT_TIME] == ds[cv.INIT_TIME][-1]) & (
        ds[cv.LEAD_TIME] == ds[cv.LEAD_TIME][-1]
    )
    assert mae.where(last_valid).count() == 0  # no obs at the last valid time
    assert mae.where(~last_valid).count() == 2 * (N_INIT * N_LEAD - 1)

    rh = ds[cv.var_name(cv.RANK_HISTOGRAM)]
    assert rh.sel(forecast_source="model_b", rank=[5, 6]).isnull().all()
    sums = rh.sum(cv.RANK, min_count=1).where(~last_valid)
    np.testing.assert_allclose(sums.values[~np.isnan(sums.values)], 1.0, rtol=1e-5)

    assert (ds[cv.var_name(cv.CRPS)].where(~last_valid, 1.0) > 0).all()
    assert (ds[cv.var_name(cv.PSD)].where(~last_valid, 1.0) > 0).all()


def test_compute_scores_matches_manual_mae():
    forecasts, obs = _synthetic(seed=1)
    ds = compute_scores({"model_b": forecasts["model_b"]}, obs, dx_km=1.0)
    fc = forecasts["model_b"].isel(init_time=2, lead_time=1)
    valid_time = fc.init_time.values + fc.lead_time.values
    o = obs.sel(time=valid_time)
    valid = np.isfinite(o) & np.isfinite(fc).all("member")
    expected = float(np.abs(fc.mean("member") - o).where(valid).mean())
    got = float(ds[cv.var_name(cv.MAE)].isel(forecast_source=0, init_time=2, lead_time=1))
    assert got == pytest.approx(expected, rel=1e-5)


def test_infer_dx_km():
    da = xr.DataArray(np.zeros(4), dims="x", coords={"x": [0.0, 500.0, 1000.0, 1500.0]})
    assert infer_dx_km(da) == pytest.approx(0.5)
    da = da.assign_coords(x=[0.0, 2.0, 4.0, 6.0])
    assert infer_dx_km(da) == pytest.approx(2.0)
