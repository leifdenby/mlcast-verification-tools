from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from mlcast_verification_tools import conventions as cv
from mlcast_verification_tools import testing
from mlcast_verification_tools.accumulator import ScoreAccumulator, merge_sources
from mlcast_verification_tools.scores import (
    crps_ensemble,
    ensemble_radial_psd,
    mae,
    rank_histogram,
)

N_LEAD, SIZE, DX = 3, 16, 2.0
LEAD_TIMES = pd.timedelta_range("5min", periods=N_LEAD, freq="5min")


def _batch(b, k, seed):
    gen = torch.Generator().manual_seed(seed)
    fc = torch.rand(b, N_LEAD, k, SIZE, SIZE, generator=gen)
    obs = torch.rand(b, N_LEAD, SIZE, SIZE, generator=gen)
    mask = torch.rand(b, SIZE, SIZE, generator=gen) > 0.1
    return fc, obs, mask


def _accumulate(source, k, init_times, batch_size=2, **kwargs):
    acc = ScoreAccumulator(source, LEAD_TIMES, DX, **kwargs)
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


def test_variable_and_units():
    init_times = list(pd.date_range("2025-01-01", periods=2, freq="1D"))
    ds = _accumulate("m", 3, init_times, variable="dbz", units="dBZ").to_dataset()
    assert set(ds.data_vars) == {
        "mae.dbz", "crps.dbz", "rank_histogram.dbz", "psd.dbz", "psd_obs.dbz"
    }  # fmt: skip
    assert ds["mae.dbz"].attrs == cv.metric_attrs("dBZ")[cv.MAE]
    assert ds["crps.dbz"].attrs["units"] == "dBZ"
    assert ds["psd_obs.dbz"].attrs["units"] == "dBZ2 km"
    assert ds["rank_histogram.dbz"].attrs["units"] == "1"


@pytest.mark.parametrize(
    ("units", "expected"),
    [
        ("mm h-1", "mm2 h-2 km"),
        ("kg m-2 s-1", "kg2 m-4 s-2 km"),
        ("dBZ", "dBZ2 km"),
        ("mm/h", "(mm/h)2 km"),
    ],
)
def test_psd_units(units, expected):
    assert cv.psd_units(units) == expected


def test_merge_sources_rejects_mixed_units():
    init_times = list(pd.date_range("2025-01-01", periods=2, freq="1D"))
    a = _accumulate("a", 3, init_times).to_dataset()
    b = _accumulate("b", 3, init_times, units="kg m-2 s-1").to_dataset()
    with pytest.raises(ValueError, match="different units"):
        merge_sources([a, b])


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


T_A = np.datetime64("2025-01-01T00:00", "ns")
T_B = np.datetime64("2025-01-01T01:00", "ns")


def _crop_batch():
    """Crops 0, 2, 3 at T_A (2 has lead 0 masked, 3 fully masked) and crop 1 at T_B."""
    fc, obs, _ = _batch(4, 3, seed=5)
    gen = torch.Generator().manual_seed(6)
    mask = torch.rand(4, N_LEAD, SIZE, SIZE, generator=gen) > 0.3
    mask[2] = torch.rand(SIZE, SIZE, generator=gen) > 0.6  # different mask
    mask[2, 0] = False
    mask[3] = False
    return fc, obs, mask, [T_A, T_B, T_A, T_A]


# no values below the threshold -> no ties -> rank histogram is deterministic
POOL_KWARGS = dict(pool_by_init_time=True, zero_threshold=-1.0)


def test_pool_by_init_time_matches_pixel_pooled_scores():
    fc, obs, mask, init_times = _crop_batch()
    acc = ScoreAccumulator("m", LEAD_TIMES, DX, **POOL_KWARGS)
    acc.update(fc, obs, mask, init_times)
    ds = acc.to_dataset()
    np.testing.assert_array_equal(ds[cv.INIT_TIME].values, [T_A, T_B])

    # pooling over crops == scoring the crops placed side by side on one grid
    for t, rows in ((T_A, [0, 2, 3]), (T_B, [1])):
        side = (
            fc[rows].movedim(0, -2).flatten(-2)[None],
            obs[rows].movedim(0, -2).flatten(-2)[None],
        )
        side_mask = mask[rows].movedim(0, -2).flatten(-2)[None]
        scores = {
            cv.MAE: mae(*side, side_mask),
            cv.CRPS: crps_ensemble(*side, side_mask),
            cv.RANK_HISTOGRAM: rank_histogram(*side, side_mask, -1.0),
        }
        for metric, expected in scores.items():
            np.testing.assert_allclose(
                ds[cv.var_name(metric)].sel(init_time=t).values[0],
                expected[0].double().numpy(),
                rtol=1e-5,
                err_msg=metric,
            )

    # PSDs: plain mean over the crops with valid pixels (per lead time)
    psd, psd_obs = ensemble_radial_psd(fc, obs, mask, DX)
    for metric, value in ((cv.PSD, psd), (cv.PSD_OBS, psd_obs)):
        value = value.double().numpy()
        got = ds[cv.var_name(metric)].values[0]
        np.testing.assert_allclose(got[0, 0], value[0, 0], rtol=1e-6)  # crop 2 masked
        np.testing.assert_allclose(got[0, 1:], value[[0, 2], 1:].mean(axis=0), rtol=1e-6)
        np.testing.assert_allclose(got[1], value[1], rtol=1e-6)


def test_pool_by_init_time_nan_without_valid_pixels():
    fc, obs, mask, _ = _crop_batch()
    acc = ScoreAccumulator("m", LEAD_TIMES, DX, **POOL_KWARGS)
    acc.update(fc[3:], obs[3:], mask[3:], [T_A])
    ds = acc.to_dataset()
    for name in ds.data_vars:
        assert ds[name].isnull().all(), name


def test_pool_by_init_time_across_updates():
    fc, obs, mask, init_times = _crop_batch()
    one = ScoreAccumulator("m", LEAD_TIMES, DX, **POOL_KWARGS)
    one.update(fc, obs, mask, init_times)
    split = ScoreAccumulator("m", LEAD_TIMES, DX, **POOL_KWARGS)
    for rows in ([2], [1, 0], [3]):  # T_A continues across batches; T_A seen first
        split.update(fc[rows], obs[rows], mask[rows], [init_times[r] for r in rows])
    xr.testing.assert_allclose(one.to_dataset(), split.to_dataset())
    _assert_same_structure(split.to_dataset(), one.to_dataset())


def test_pool_by_init_time_unique_matches_unpooled():
    init_times = list(pd.date_range("2025-01-01", periods=5, freq="1D")[::-1])
    kwargs = dict(zero_threshold=-1.0)
    ref = _accumulate("m", 3, init_times, **kwargs).to_dataset()
    ds = _accumulate("m", 3, init_times, pool_by_init_time=True, **kwargs).to_dataset()
    _assert_same_structure(ds, ref)
    xr.testing.assert_allclose(ds, ref)


def test_merge_sources_rejects_duplicate_source():
    ds = _accumulate("m", 3, [pd.Timestamp("2025-01-01")]).to_dataset()
    with pytest.raises(ValueError, match="duplicate"):
        merge_sources([ds, ds])
