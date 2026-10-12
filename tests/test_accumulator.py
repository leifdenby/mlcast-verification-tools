from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from mlcast_verification_tools import conventions as cv
from mlcast_verification_tools import testing
from mlcast_verification_tools.accumulator import ScoreAccumulator, merge_sources
from mlcast_verification_tools.scores import crps_ensemble

N_LEAD, SIZE, DX = 3, 16, 2.0
LEAD_TIMES = pd.timedelta_range("5min", periods=N_LEAD, freq="5min")


def _batch(b, k, seed):
    gen = torch.Generator().manual_seed(seed)
    fc = torch.rand(b, N_LEAD, k, SIZE, SIZE, generator=gen)
    obs = torch.rand(b, N_LEAD, SIZE, SIZE, generator=gen)
    mask = torch.rand(b, SIZE, SIZE, generator=gen) > 0.1
    return fc, obs, mask


def _accumulate(source, k, init_times, batch_size=2):
    acc = ScoreAccumulator(source, LEAD_TIMES, DX)
    for start in range(0, len(init_times), batch_size):
        its = init_times[start : start + batch_size]
        acc.update(*_batch(len(its), k, seed=start), its)
    return acc


def _assert_same_structure(ds: xr.Dataset, ref: xr.Dataset):
    assert set(ds.data_vars) == set(ref.data_vars)
    for name in ref.data_vars:
        assert ds[name].dims == ref[name].dims, name
        assert ds[name].dtype == ref[name].dtype, name
        assert ds[name].attrs == ref[name].attrs, name
    assert set(ds.coords) == set(ref.coords)
    for name in ref.coords:
        assert ds[name].dtype.kind == ref[name].dtype.kind, name
        if name not in (cv.FORECAST_SOURCE, cv.INIT_TIME, cv.LEAD_TIME):
            assert ds[name].dtype == ref[name].dtype, name
        assert ds[name].attrs == ref[name].attrs, name
    # with pandas 3 the synthetic times are [us]; computed scores always use [ns]
    assert ds[cv.INIT_TIME].dtype == np.dtype("datetime64[ns]")
    assert ds[cv.LEAD_TIME].dtype == np.dtype("timedelta64[ns]")


def test_to_dataset_layout_matches_synthetic():
    init_times = pd.date_range("2025-01-01", periods=5, freq="1D")
    ds = _accumulate("model_a", 4, list(init_times)).to_dataset()
    ref = testing.make_synthetic_scores(
        ensemble_sizes={"model_a": 4},
        n_init_times=5,
        n_lead_times=N_LEAD,
        domain_size=SIZE,
        dx_km=DX,
    )
    _assert_same_structure(ds, ref)
    for name in ref.coords:
        np.testing.assert_array_equal(ds[name].values, ref[name].values)
    for name in ref.data_vars:
        assert ds[name].shape == ref[name].shape
    np.testing.assert_allclose(ds[cv.var_name(cv.RANK_HISTOGRAM)].sum(cv.RANK), 1.0)


def test_to_dataset_sorts_and_matches_kernel():
    init_times = pd.date_range("2025-01-01", periods=4, freq="1h")[::-1]
    acc = _accumulate("m", 3, list(init_times), batch_size=4)
    ds = acc.to_dataset()
    assert (np.diff(ds[cv.INIT_TIME].values) > np.timedelta64(0, "ns")).all()
    fc, obs, mask = _batch(4, 3, seed=0)
    expected = crps_ensemble(fc, obs, mask).double().numpy()[::-1]
    np.testing.assert_allclose(ds[cv.var_name(cv.CRPS)].values[0], expected, rtol=1e-6)


def test_duplicate_init_time_raises():
    t = np.datetime64("2025-01-01T00:00", "ns")
    acc = _accumulate("m", 3, [t, t + np.timedelta64(1, "h")])
    acc.update(*_batch(1, 3, seed=9), [t])
    with pytest.raises(ValueError, match="duplicate"):
        acc.to_dataset()


def test_lead_time_mismatch_raises():
    acc = ScoreAccumulator("m", LEAD_TIMES[:2], DX)
    with pytest.raises(ValueError, match="lead times"):
        acc.update(*_batch(1, 3, seed=0), [np.datetime64("2025-01-01")])


def test_no_psd():
    acc = ScoreAccumulator("m", LEAD_TIMES.values, DX, compute_psd=False)
    acc.update(*_batch(1, 3, seed=0), [pd.Timestamp("2025-01-01")])
    ds = acc.to_dataset()
    assert cv.var_name(cv.PSD) not in ds
    assert cv.WAVELENGTH not in ds.coords


def test_merge_sources_nan_pads_rank():
    init_times = list(pd.date_range("2025-01-01", periods=4, freq="1D"))
    ds_a = _accumulate("model_a", 5, init_times).to_dataset()
    ds_b = _accumulate("model_b", 3, init_times[1:]).to_dataset()
    merged = merge_sources([ds_a, ds_b])

    ref = testing.make_synthetic_scores(
        ensemble_sizes={"model_a": 5, "model_b": 3},
        n_init_times=4,
        n_lead_times=N_LEAD,
        domain_size=SIZE,
        dx_km=DX,
    )
    _assert_same_structure(merged, ref)
    np.testing.assert_array_equal(merged[cv.FORECAST_SOURCE].values, ["model_a", "model_b"])
    np.testing.assert_array_equal(merged[cv.RANK].values, np.arange(1, 7))
    rh = merged[cv.var_name(cv.RANK_HISTOGRAM)]
    b = rh.sel(forecast_source="model_b")
    assert b.sel(rank=[5, 6]).isnull().all()
    assert b.isel(init_time=0).isnull().all()  # model_b has no first init time
    assert b.isel(init_time=slice(1, None), rank=slice(0, 4)).notnull().all()
    np.testing.assert_allclose(b.isel(init_time=slice(1, None)).sum(cv.RANK), 1.0, rtol=1e-6)
    assert rh.sel(forecast_source="model_a").notnull().all()


def test_merge_sources_rejects_duplicate_source():
    ds = _accumulate("m", 3, [pd.Timestamp("2025-01-01")]).to_dataset()
    with pytest.raises(ValueError, match="duplicate"):
        merge_sources([ds, ds])
