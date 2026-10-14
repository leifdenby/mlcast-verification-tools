"""Tests of the mlcast integration (scoring an mlcast model while it forecasts)."""

from __future__ import annotations

import importlib
import re

import numpy as np
import pandas as pd
import pytest
import xarray as xr

torch = pytest.importorskip("torch")
pytest.importorskip("mlcast.data.normalization")

# imported after the skips above, which need torch and mlcast
from mlcast.data.normalization import DEFAULT_SCALING, ReflectivityScaling  # noqa: E402
from mlcast.models.convgru import ConvGruModel  # noqa: E402

from mlcast_verification_tools import conventions as cv  # noqa: E402
from mlcast_verification_tools.integrations import mlcast as mi  # noqa: E402
from mlcast_verification_tools.integrations._mlcast_core import _member_chunks  # noqa: E402


def _lightning_module():
    """NowcastLightningModule, or skip (mlcast imports torchvision via its visualization module)."""
    try:
        return importlib.import_module("mlcast.nowcasting_module").NowcastLightningModule
    except ImportError as exc:
        pytest.skip(f"mlcast.nowcasting_module not importable: {exc}")


T0 = np.datetime64("2023-07-01T00:00", "ns")
STEP = np.timedelta64(5, "m")
NT, NY, NX = 40, 24, 20
INPUT_STEPS, FORECAST_STEPS = 3, 4


def _source_dataset(
    units: str = "kg m-2 h-1", missing_time_index: int | None = 30, nt: int = NT
) -> xr.Dataset:
    """Tiny dataset in the layout of the mlcast source data (cf. IT-DPC-SRI v0.1.0)."""
    rng = np.random.default_rng(1)
    rr = rng.gamma(0.5, 2.0, size=(nt, NY, NX)).astype(np.float32)
    rr[rng.random(rr.shape) < 0.4] = 0.0
    rr[:, 0, 0] = np.nan  # never valid
    rr[10, 5, 5] = np.nan  # invalid in one frame only
    time = T0 + STEP * np.arange(nt)
    keep = np.ones(nt, bool)
    missing = np.array([], dtype="datetime64[ns]")
    if missing_time_index is not None:  # frame absent from the time axis, as in the real data
        keep[missing_time_index] = False
        missing = time[[missing_time_index]]
    ds = xr.Dataset(
        {
            "RR": (
                ("time", "y", "x"),
                rr[keep],
                {
                    "grid_mapping": "crs",
                    "long_name": "Total precipitation rate",
                    "standard_name": "rainfall_flux",
                    "units": units,
                },
            ),
            "crs": ((), np.float32(0), {"grid_mapping_name": "transverse_mercator"}),
        },
        coords={
            "time": ("time", time[keep], {"standard_name": "time", "long_name": "Time"}),
            "y": ("y", 649500.0 - 1000.0 * np.arange(NY), {"units": "m", "axis": "Y"}),
            "x": ("x", -599500.0 + 1000.0 * np.arange(NX), {"units": "m", "axis": "X"}),
            "missing_times": ("missing_times", missing),
        },
        attrs={
            "title": "Synthetic radar archive",
            "mlcast_dataset_identifier": "XX-TEST",
            "mlcast_dataset_version": "0.1.0",
        },
    )
    return ds


DBZ_NO_ECHO = -32.0  # "no echo" value of e.g. DMI's radar composite, below mlcast's 0 dBZ floor


def _dbz_source_dataset(missing_time_index: int | None = 30, nt: int = NT) -> xr.Dataset:
    """The rain rates of `_source_dataset` as a reflectivity variable in dBZ."""
    src = _source_dataset(missing_time_index=missing_time_index, nt=nt)
    rr = src.RR.values
    with np.errstate(divide="ignore"):
        dbz = (10 * np.log10(200.0) + 16 * np.log10(rr)).astype(np.float32)
    dbz[rr == 0] = DBZ_NO_ECHO
    dbz[rr > 8] = 70.0  # above mlcast's 60 dBZ ceiling
    attrs = {
        "grid_mapping": "crs",
        "long_name": "10 min radar reflectivity",
        "standard_name": "equivalent_reflectivity_factor",
        "units": "dBZ",
    }
    return src.drop_vars("RR").assign(dbz=(("time", "y", "x"), dbz, attrs))


@pytest.fixture
def source_path(tmp_path):
    path = tmp_path / "source.zarr"
    _source_dataset().to_zarr(path, zarr_format=2)
    return path


def _valid_init_times(n: int = 4) -> np.ndarray:
    # init time = last input frame; frames 30 is missing, so stay well before it
    return T0 + STEP * (INPUT_STEPS - 1 + 3 * np.arange(n))


class AdapterModel(torch.nn.Module):
    """ConvGruModel with the NowcastLightningModule.forward signature (no Lightning needed)."""

    def __init__(self, num_blocks: int = 2) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.network = ConvGruModel(input_channels=1, num_blocks=num_blocks)

    def forward(self, x, forecast_steps, ensemble_size=None):
        return self.network(x, steps=forecast_steps, ensemble_size=ensemble_size or 1)


# --- units and denormalisation ---------------------------------------------------------------


@pytest.mark.parametrize("standard_name", ["rainfall_rate", "rainfall_flux"])
@pytest.mark.parametrize(
    "scaling", [None, ReflectivityScaling(a=300.0, b=1.4, dbz_floor=5.0, dbz_ceiling=55.0)]
)
def test_denormalize_torch_matches_mlcast(standard_name, scaling):
    rng = np.random.default_rng(0)
    x = rng.uniform(-1.2, 1.2, size=(3, 5, 7)).astype(np.float32)
    x[0, 0, :3] = [-1.0, 1.0, np.nan]
    s = scaling or DEFAULT_SCALING
    with pytest.warns(UserWarning) if standard_name == "rainfall_flux" else _nullcontext():
        expected = s.denormalization_registry[standard_name](x.copy())
    got = mi.denormalize_torch(torch.from_numpy(x), standard_name, scaling).numpy()
    assert got.dtype == np.float32
    np.testing.assert_allclose(got, expected, rtol=2e-5, atol=0, equal_nan=True)
    # at or below the floor maps to exactly 0
    assert np.all(got[x <= -1.0] == 0)


@pytest.mark.parametrize(
    "scaling", [None, ReflectivityScaling(a=300.0, b=1.4, dbz_floor=5.0, dbz_ceiling=55.0)]
)
def test_denormalize_torch_reflectivity_matches_mlcast(scaling):
    x = np.array([-1.0, -0.5, 0.0, 0.3, 1.0, np.nan], dtype=np.float32)
    s = scaling or DEFAULT_SCALING
    expected = s.denormalization_registry["equivalent_reflectivity_factor"](x.copy())
    got = mi.denormalize_torch(
        torch.from_numpy(x), "equivalent_reflectivity_factor", scaling
    ).numpy()
    np.testing.assert_allclose(got, expected, rtol=1e-6, equal_nan=True)
    # dBZ, not a rain rate: the normalised range maps onto [dbz_floor, dbz_ceiling]
    assert got[0] == pytest.approx(s.dbz_floor) and got[4] == pytest.approx(s.dbz_ceiling)


def test_denormalize_torch_rejects_unsupported():
    with pytest.raises(ValueError, match="not supported"):
        mi.denormalize_torch(torch.zeros(2), "rainfall_amount")
    with pytest.raises(TypeError):
        mi.denormalize_torch(torch.zeros(2, dtype=torch.int64))


class _nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.mark.parametrize(
    ("units", "factor"),
    [("mm/h", 1.0), ("kg m-2 h-1", 1.0), ("mm h-1", 1.0), ("kg m-2 s-1", 3600.0), ("mm/s", 3600.0)],
)
def test_mm_per_hour_factor(units, factor):
    assert mi.mm_per_hour_factor(units) == factor


def test_mm_per_hour_factor_unknown():
    with pytest.raises(ValueError, match="cannot convert"):
        mi.mm_per_hour_factor("dBZ")


def test_init_times_from_range():
    expected = pd.date_range("2023-07-01", "2023-07-02", freq="6h").values
    np.testing.assert_array_equal(
        mi.init_times_from_range("2023-07-01", "2023-07-02", "PT6H"), expected
    )
    np.testing.assert_array_equal(
        mi.init_times_from_range("2023-07-01", "2023-07-02", "6h"), expected
    )


# --- dataset ------------------------------------------------------------------------------------


def test_dataset_items_match_mlcast_preprocessing(source_path):
    raw_ds = xr.open_zarr(source_path)
    init_times = _valid_init_times()
    ds = mi.VerificationDataset(source_path, init_times, INPUT_STEPS, FORECAST_STEPS)

    assert len(ds) == len(init_times)
    assert ds.variable == "RR" and ds.standard_name == "rainfall_flux"
    assert ds.dx_km == pytest.approx(1.0)
    assert ds.time_step == pd.Timedelta("5min")
    np.testing.assert_array_equal(ds.lead_times, STEP.astype("timedelta64[ns]") * np.arange(1, 5))

    item = ds[1]
    assert item["input"].shape == (INPUT_STEPS, 1, NY, NX) and item["input"].dtype == torch.float32
    assert (
        item["target"].shape == (FORECAST_STEPS, NY, NX) and item["target"].dtype == torch.float32
    )
    assert item["mask"].shape == (NY, NX) and item["mask"].dtype == torch.bool
    assert np.datetime64(int(item["init_time"]), "ns") == init_times[1]

    # frames: init_time - (T_in - 1) steps ... init_time + T_out steps
    t_init = int(np.where(raw_ds.time.values == init_times[1])[0][0])
    frames = raw_ds.RR.values[t_init - INPUT_STEPS + 1 : t_init + FORECAST_STEPS + 1]
    with pytest.warns(UserWarning):
        normed = DEFAULT_SCALING.normalization_registry["rainfall_flux"](frames)
    np.testing.assert_array_equal(
        item["input"][:, 0].numpy(),
        np.nan_to_num(normed[:INPUT_STEPS], nan=-1.0).astype(np.float32),
    )
    # kg m-2 h-1 == mm/h, NaN kept
    np.testing.assert_array_equal(item["target"].numpy(), frames[INPUT_STEPS:])
    np.testing.assert_array_equal(item["mask"].numpy(), ~np.isnan(normed).any(axis=0))
    assert not item["mask"][0, 0]


def test_dataset_mask_covers_inputs(source_path):
    # frame 10 has a NaN at (5, 5): every case using frame 10 masks that pixel
    init = T0 + STEP * 11
    ds = mi.VerificationDataset(source_path, [init], INPUT_STEPS, FORECAST_STEPS)
    assert not ds[0]["mask"][5, 5]
    assert ds[0]["mask"][6, 6]


def test_dataset_region_and_native_units(tmp_path):
    path = tmp_path / "flux_si.zarr"
    src = _source_dataset(units="kg m-2 s-1")
    src.to_zarr(path, zarr_format=2)
    region = {"y": slice(2, 18), "x": slice(4, 16)}
    ds = mi.VerificationDataset(
        path, _valid_init_times(2), INPUT_STEPS, FORECAST_STEPS, region=region
    )
    item = ds[0]
    assert item["input"].shape == (INPUT_STEPS, 1, 16, 12)
    # native units: no conversion, the rank threshold is 0.037 mm/h in kg m-2 s-1
    assert ds.units == "kg m-2 s-1" and ds.score_variable == "tp"
    assert ds.zero_threshold == pytest.approx(0.037 / 3600)
    t_init = int(np.where(src.time.values == ds.init_times[0])[0][0])
    expected = src.RR.values[t_init + 1 : t_init + 1 + FORECAST_STEPS, 2:18, 4:16]
    np.testing.assert_array_equal(item["target"].numpy(), expected)


def test_dataset_reflectivity_source():
    src = _dbz_source_dataset()
    init_times = _valid_init_times(2)
    ds = mi.VerificationDataset(src, init_times, INPUT_STEPS, FORECAST_STEPS)
    assert ds.variable == "dbz" and ds.standard_name == "equivalent_reflectivity_factor"
    assert ds.units == "dBZ" and ds.score_variable == "dbz"
    assert ds.zero_threshold == DEFAULT_SCALING.dbz_floor

    item = ds[1]
    t_init = int(np.where(src.time.values == init_times[1])[0][0])
    frames = src.dbz.values[t_init - INPUT_STEPS + 1 : t_init + FORECAST_STEPS + 1]
    assert (frames == DBZ_NO_ECHO).any() and (frames > 60).any()
    normed = DEFAULT_SCALING.normalization_registry["equivalent_reflectivity_factor"](frames)
    np.testing.assert_array_equal(
        item["input"][:, 0].numpy(),
        np.nan_to_num(normed[:INPUT_STEPS], nan=-1.0).astype(np.float32),
    )
    # dBZ clipped to the range the model can predict, NaN kept
    np.testing.assert_array_equal(item["target"].numpy(), np.clip(frames[INPUT_STEPS:], 0, 60))
    assert np.isnan(item["target"][:, 0, 0]).all()

    scaling = ReflectivityScaling(dbz_floor=5.0, dbz_ceiling=55.0)
    ds = mi.VerificationDataset(src, init_times, INPUT_STEPS, FORECAST_STEPS, scaling=scaling)
    target = ds[1]["target"].numpy()
    assert ds.zero_threshold == 5.0
    assert np.nanmin(target) == 5.0 and np.nanmax(target) == 55.0


def test_dataset_missing_frames(source_path):
    good = _valid_init_times(2)
    bad = [T0, T0 + STEP * 28]  # not enough input frames / frame 30 absent from the time axis
    with pytest.warns(UserWarning, match="2 of 4 init times"):
        ds = mi.VerificationDataset(source_path, [*good, *bad], INPUT_STEPS, FORECAST_STEPS)
    np.testing.assert_array_equal(ds.init_times, good)
    np.testing.assert_array_equal(ds.skipped_init_times, np.array(bad, dtype="datetime64[ns]"))
    with pytest.raises(ValueError, match="lack input/target frames"):
        mi.VerificationDataset(source_path, bad, INPUT_STEPS, FORECAST_STEPS, on_missing="raise")
    with pytest.raises(ValueError, match="duplicates"):
        mi.VerificationDataset(source_path, [good[0], good[0]], INPUT_STEPS, FORECAST_STEPS)


def test_dataset_missing_times_coordinate(tmp_path):
    # frame present on the time axis but listed in `missing_times` is also treated as missing
    src = _source_dataset(missing_time_index=None)
    src = src.assign_coords(missing_times=("missing_times", src.time.values[[20]]))
    init = T0 + STEP * 18
    with pytest.raises(ValueError, match="lack input/target frames"):
        mi.VerificationDataset(src, [init], INPUT_STEPS, FORECAST_STEPS, on_missing="raise")


def test_dataset_variable_lookup_errors():
    src = _source_dataset()
    src["RR"].attrs["standard_name"] = "rainfall_amount"  # not supported (accumulation)
    with pytest.raises(ValueError, match="found 0 data variables"):
        mi.VerificationDataset(src, _valid_init_times(1), INPUT_STEPS, FORECAST_STEPS)
    with pytest.raises(ValueError, match="not supported"):
        mi.VerificationDataset(
            src, _valid_init_times(1), INPUT_STEPS, FORECAST_STEPS, variable="RR"
        )


# --- verification loop --------------------------------------------------------------------------


def test_member_chunks():
    assert _member_chunks(5, None) == [5]
    assert _member_chunks(5, 8) == [5]
    assert _member_chunks(6, 2) == [2, 2, 2]
    assert _member_chunks(7, 3) == [3, 4]  # no 1-member (deterministic) call
    assert _member_chunks(8, 3) == [3, 3, 2]
    with pytest.raises(ValueError):
        _member_chunks(4, 1)


def _check_scores(
    scores: xr.Dataset, n_cases: int, n_members: int, psd: bool = True, variable: str = "tp"
) -> None:
    """The model is the first source; persistence (scored by default) may follow."""
    sources = scores[cv.FORECAST_SOURCE].values.tolist()
    assert len(sources) == 1 or sources[1:] == ["persistence"], sources
    assert scores.sizes[cv.INIT_TIME] == n_cases
    assert scores.sizes[cv.LEAD_TIME] == FORECAST_STEPS
    assert scores.sizes[cv.RANK] == n_members + 1
    names = {cv.var_name(m, variable) for m in (cv.MAE, cv.CRPS, cv.RANK_HISTOGRAM)}
    if psd:
        names |= {cv.var_name(cv.PSD, variable), cv.var_name(cv.PSD_OBS, variable)}
    assert names <= set(scores.data_vars)
    rank_name = cv.var_name(cv.RANK_HISTOGRAM, variable)
    for source in sources:
        one = scores.sel({cv.FORECAST_SOURCE: source})
        if source == "persistence":  # one member: 2 ranks, NaN-padded to the model's
            one = one.isel({cv.RANK: slice(0, 2)})
            np.testing.assert_allclose(one[rank_name].sum(cv.RANK).values, 1.0)
        for name in names:
            assert np.isfinite(one[name].values).all(), (source, name)
    assert scores[cv.LEAD_TIME].values[0] == np.timedelta64(5, "m")


def test_verify_smoke(source_path):
    init_times = _valid_init_times(4)
    ds = mi.VerificationDataset(source_path, init_times, INPUT_STEPS, FORECAST_STEPS)
    kwargs = {
        "forecast_source": "convgru",
        "ensemble_size": 3,
        "batch_size": 3,
        "device": "cpu",
        "progress": False,
    }
    scores = mi.verify_mlcast_model(AdapterModel(), ds, **kwargs)
    _check_scores(scores, 4, 3)
    np.testing.assert_array_equal(scores[cv.INIT_TIME].values, init_times)
    assert scores[cv.FORECAST_SOURCE].values.tolist() == ["convgru", "persistence"]
    # same seed -> same ensemble -> same scores
    again = mi.verify_mlcast_model(AdapterModel(), ds, **kwargs)
    xr.testing.assert_identical(scores, again)

    # the model's scores don't depend on the baseline
    alone = mi.verify_mlcast_model(AdapterModel(), ds, persistence=False, **kwargs)
    assert alone[cv.FORECAST_SOURCE].values.tolist() == ["convgru"]
    xr.testing.assert_allclose(alone, scores.sel({cv.FORECAST_SOURCE: ["convgru"]}))

    # persistence: the last input frame at every lead time, one member
    pers = scores.sel({cv.FORECAST_SOURCE: "persistence"})
    np.testing.assert_allclose(pers["crps.tp"].values, pers["mae.tp"].values, rtol=1e-6)
    for i in range(len(ds)):
        item = ds[i]
        mask = item["mask"].numpy()
        last = item["last_obs"].numpy()
        expected = [np.abs(last - obs)[mask].mean() for obs in item["target"].numpy()]
        np.testing.assert_allclose(pers["mae.tp"].values[i], expected, rtol=1e-5)
    raw = xr.open_zarr(source_path)["RR"].sel(time=init_times[0]).values
    np.testing.assert_array_equal(ds[0]["last_obs"].numpy(), raw)

    with pytest.raises(ValueError, match="taken by the baseline"):
        mi.verify_mlcast_model(AdapterModel(), ds, **{**kwargs, "forecast_source": "persistence"})


def test_verify_reflectivity_source():
    """A dBZ source is scored in dBZ, as `dbz`."""
    ds = mi.VerificationDataset(
        _dbz_source_dataset(), _valid_init_times(3), INPUT_STEPS, FORECAST_STEPS
    )
    scores = mi.verify_mlcast_model(
        AdapterModel(), ds, forecast_source="convgru", ensemble_size=3, device="cpu",
        progress=False,
    )  # fmt: skip
    _check_scores(scores, 3, 3, variable="dbz")
    assert "mae.tp" not in scores
    assert scores["mae.dbz"].attrs["units"] == "dBZ"
    assert scores["crps.dbz"].attrs["units"] == "dBZ"
    assert scores["psd.dbz"].attrs["units"] == "dBZ2 km"
    # both forecast and (clipped) truth lie in [0, 60] dBZ
    assert (scores["mae.dbz"].values <= 60).all()


def test_verify_member_chunks(source_path):
    ds = mi.VerificationDataset(source_path, _valid_init_times(3), INPUT_STEPS, FORECAST_STEPS)
    scores = mi.verify_mlcast_model(
        AdapterModel(),
        ds,
        forecast_source="convgru",
        ensemble_size=5,
        member_chunk=2,
        batch_size=2,
        num_workers=0,
        compute_psd=False,
        device="cpu",
        progress=False,
    )
    _check_scores(scores, 3, 5, psd=False)


def test_verify_bf16_autocast(source_path):
    """bf16 autocast gives float32 forecasts and scores close to (not equal to) fp32."""
    ds = mi.VerificationDataset(source_path, _valid_init_times(3), INPUT_STEPS, FORECAST_STEPS)
    kwargs = {"forecast_source": "convgru", "ensemble_size": 3, "device": "cpu", "progress": False}
    fp32 = mi.verify_mlcast_model(AdapterModel(), ds, **kwargs)
    bf16 = mi.verify_mlcast_model(AdapterModel(), ds, precision="bf16", **kwargs)
    _check_scores(bf16, 3, 3)
    assert not np.array_equal(bf16["mae.tp"].values, fp32["mae.tp"].values)
    np.testing.assert_allclose(bf16["mae.tp"].values, fp32["mae.tp"].values, rtol=0.2, atol=0.05)
    with pytest.raises(ValueError, match="precision"):
        mi.verify_mlcast_model(AdapterModel(), ds, precision="fp16", **kwargs)


def test_verify_prints_summary_and_logs_time(source_path, capsys, monkeypatch):
    from loguru import logger

    monkeypatch.setenv("COLUMNS", "300")  # rich wraps at the terminal width otherwise
    ds = mi.VerificationDataset(
        source_path, _valid_init_times(3), INPUT_STEPS, FORECAST_STEPS,
        region={"y": slice(0, 8)},
    )  # fmt: skip
    messages: list[str] = []
    sink = logger.add(messages.append, format="{message}", level="INFO")
    try:
        mi.verify_mlcast_model(
            AdapterModel(), ds, forecast_source="convgru", ensemble_size=5, member_chunk=2,
            batch_size=2, compute_psd=False, device="cpu", precision="bf16", progress=True,
        )  # fmt: skip
    finally:
        logger.remove(sink)
    panel = capsys.readouterr().err
    w = ds.grid_shape[1]
    assert "🎯 Verifying convgru" in panel
    assert f"8 × {w} px   dx 1 km   (sub-domain y=0:8)" in panel
    assert "3 cases" in panel
    assert f"{INPUT_STEPS} input + {FORECAST_STEPS} forecast × 5 min" in panel
    assert "ConvGruModel" in panel
    assert "5 members   2 model calls of 2/3 members per batch" in panel
    assert f"3 cases × 5 members × {FORECAST_STEPS} steps = {3 * 5 * FORECAST_STEPS}" in panel
    assert "2 × 2 cases   0 loader workers" in panel
    assert "cpu   bfloat16 autocast" in panel
    assert re.search(r"Verifying convgru done: 3 cases in 0:00:\d\d \(", "".join(messages))


def test_verify_progress_shows_stage_times(source_path, monkeypatch):
    from tqdm.auto import tqdm

    ds = mi.VerificationDataset(source_path, _valid_init_times(2), INPUT_STEPS, FORECAST_STEPS)
    postfixes = []
    original = tqdm.set_postfix_str

    def record(self, s="", refresh=True):
        postfixes.append(s)
        return original(self, s, refresh=refresh)

    monkeypatch.setattr(tqdm, "set_postfix_str", record)
    scores = mi.verify_mlcast_model(
        AdapterModel(), ds, forecast_source="convgru", ensemble_size=2, batch_size=1,
        compute_psd=False, device="cpu", progress=True,
    )  # fmt: skip
    _check_scores(scores, 2, 2, psd=False)
    assert len(postfixes) == 2
    assert re.fullmatch(r"data \d+\.\d\ds model \d+\.\d\ds score \d+\.\d\ds", postfixes[-1])


@pytest.mark.timeout(120)
def test_verify_with_spawned_workers(source_path):
    """The only test of the DataLoader worker path (workers are spawned, not forked)."""
    ds = mi.VerificationDataset(source_path, _valid_init_times(2), INPUT_STEPS, FORECAST_STEPS)
    kwargs = {
        "forecast_source": "convgru",
        "ensemble_size": 2,
        "compute_psd": False,
        "device": "cpu",
        "progress": False,
    }
    with_workers = mi.verify_mlcast_model(AdapterModel(), ds, num_workers=1, **kwargs)
    _check_scores(with_workers, 2, 2, psd=False)
    xr.testing.assert_identical(
        with_workers, mi.verify_mlcast_model(AdapterModel(), ds, num_workers=0, **kwargs)
    )


def test_verify_multichannel_model_rejected(source_path):
    class TwoChannel(AdapterModel):
        def forward(self, x, forecast_steps, ensemble_size=None):
            out = super().forward(x, forecast_steps, ensemble_size)
            return torch.cat([out, out], dim=2)

    ds = mi.VerificationDataset(source_path, _valid_init_times(1), INPUT_STEPS, FORECAST_STEPS)
    with pytest.raises(ValueError, match="single-variable"):
        mi.verify_mlcast_model(
            TwoChannel(), ds, forecast_source="x", ensemble_size=2, progress=False
        )


# --- checkpoints and Fiddle configs -------------------------------------------------------------


def _save_checkpoint(module, path) -> None:
    """Write a checkpoint in Lightning's format (the keys load_from_checkpoint reads)."""
    import pytorch_lightning as pl

    torch.save(
        {
            "state_dict": module.state_dict(),
            "hyper_parameters": dict(module.hparams),
            "pytorch-lightning_version": pl.__version__,
        },
        path,
    )


# a training run: source Zarr + sampling index under <root>/data, config with relative paths
RUN_NT = 80  # frames of the run's source data (no gaps)
TIME_DEPTH = 8  # sampled window, >= INPUT_STEPS + FORECAST_STEPS
RATIO_SPLITS = {"time": {"train": 0.5, "val": 0.25, "test": 0.25}}  # test: frames 60..79


def _write_sampling_index(
    path,
    zarr_path: str,
    times: np.ndarray,
    data_var: str = "RR",
    data_kind: str = "rainrate",
    units: str = "kg m-2 h-1",
) -> None:
    """Sampling index of all (t, x, y) crop corners, written with mlcast's own schema."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from mlcast.data.source_data.sampling.sampling_index_spec import (
        SamplingParameters,
        build_schema,
    )

    t, x, y = np.meshgrid(np.arange(len(times) - TIME_DEPTH + 1), [0, 8], [0, 8], indexing="ij")
    n = t.size
    rng = np.random.default_rng(3)
    mean = rng.random(n).astype(np.float32)
    columns = {
        "t": t.ravel().astype(np.int32),
        "x": x.ravel().astype(np.int32),
        "y": y.ravel().astype(np.int32),
        "nan_count": np.zeros(n, np.int32),
        "sum": mean * 512,
        "mean": mean,
        "frac_wet": rng.random(n).astype(np.float32),
    }
    params = SamplingParameters(
        zarr_path=zarr_path,
        data_var=data_var,
        time_var="time",
        start_date=pd.Timestamp(times[0]).to_pydatetime(),
        end_date=pd.Timestamp(times[-1]).to_pydatetime(),
        time_step_minutes=5,
        time_depth=TIME_DEPTH,
        width=8,
        height=8,
        step_t=1,
        step_x=8,
        step_y=8,
        max_nan=TIME_DEPTH * 64,
        wet_threshold=0.1,
        data_kind=data_kind,
        units=units,
    )
    schema = build_schema(params)
    pq.write_table(pa.Table.from_pydict(columns, schema=schema), path)


def _experiment_config(splits=None, num_blocks: int = 2, standard_name: str | None = None):
    """mlcast's default experiment config, made tiny and pointed at ./data (relative)."""
    pytest.importorskip("fiddle")
    _lightning_module()
    from mlcast.config.base import training_experiment

    cfg = training_experiment.as_buildable()  # calling it would build the experiment
    factory = cfg.data.dataset_factory
    factory.zarr_path = "./data/source.zarr"
    factory.index_path = "./data/sampled_datacubes.parquet"
    factory.input_steps = INPUT_STEPS
    factory.forecast_steps = FORECAST_STEPS
    factory.time_depth = TIME_DEPTH
    factory.width = factory.height = 8
    cfg.data.splits = splits or RATIO_SPLITS
    cfg.data.eval_selector.keep_fraction = 0.5
    cfg.data.num_workers = 0
    cfg.pl_module.network.num_blocks = num_blocks
    if standard_name is not None:
        factory.standard_names = [standard_name]
    return cfg


def _dump_config(cfg, path) -> None:
    """Write the config as mlcast's orchestrator does at the start of training."""
    import mlcast.config.orchestrator  # noqa: F401  registers mlcast's `!type` representer
    from fiddle.experimental.yaml_serialization import dump_yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_yaml(cfg))


@pytest.fixture
def data_root(tmp_path):
    """Directory training was started from: data/source.zarr + data/sampled_datacubes.parquet."""
    root = tmp_path / "mlcast"
    (root / "data").mkdir(parents=True)
    src = _source_dataset(missing_time_index=None, nt=RUN_NT)
    src.to_zarr(root / "data" / "source.zarr", zarr_format=2)
    _write_sampling_index(
        root / "data" / "sampled_datacubes.parquet", "./data/source.zarr", src.time.values
    )
    return root


@pytest.fixture
def run_dir(data_root):
    """mlcast TensorBoard run: <run>/config.yaml + <run>/checkpoints/*.ckpt (tiny ConvGRU)."""
    NowcastLightningModule = _lightning_module()
    run = data_root / "logs" / "mlcast" / "version_0"
    _dump_config(_experiment_config(), run / "config.yaml")
    torch.manual_seed(0)
    module = NowcastLightningModule(network=ConvGruModel(num_blocks=2), ensemble_size=2)
    (run / "checkpoints").mkdir(parents=True)
    _save_checkpoint(module, run / "checkpoints" / "epoch=0-step=1.ckpt")
    return run


@pytest.fixture
def dbz_data_root(tmp_path):
    """Like `data_root`, with a reflectivity (dBZ) source."""
    root = tmp_path / "mlcast_dbz"
    (root / "data").mkdir(parents=True)
    src = _dbz_source_dataset(missing_time_index=None, nt=RUN_NT)
    src.to_zarr(root / "data" / "source.zarr", zarr_format=2)
    _write_sampling_index(
        root / "data" / "sampled_datacubes.parquet",
        "./data/source.zarr",
        src.time.values,
        data_var="dbz",
        data_kind="reflectivity",
        units="dBZ",
    )
    return root


@pytest.fixture
def dbz_run_dir(dbz_data_root):
    """Like `run_dir`, for a model trained on the dBZ source."""
    NowcastLightningModule = _lightning_module()
    run = dbz_data_root / "logs" / "mlcast" / "version_0"
    _dump_config(
        _experiment_config(standard_name="equivalent_reflectivity_factor"), run / "config.yaml"
    )
    torch.manual_seed(0)
    module = NowcastLightningModule(network=ConvGruModel(num_blocks=2), ensemble_size=2)
    (run / "checkpoints").mkdir(parents=True)
    _save_checkpoint(module, run / "checkpoints" / "epoch=0-step=1.ckpt")
    return run


def _test_dataset(cfg):
    import fiddle as fdl

    dm = fdl.build(cfg.data)
    dm.setup("test")
    return dm.test_dataset


def test_load_run_config_resolves_paths(data_root, run_dir, tmp_path, monkeypatch):
    import fiddle as fdl

    cfg = mi.load_run_config(run_dir / "config.yaml", data_root=data_root)
    assert isinstance(cfg, fdl.Config)
    factory = cfg.data.dataset_factory
    assert factory.zarr_path == str(data_root / "data" / "source.zarr")
    assert factory.index_path == str(data_root / "data" / "sampled_datacubes.parquet")
    net = fdl.build(cfg.pl_module.network)
    assert isinstance(net, ConvGruModel) and net.num_blocks == 2

    # default root: the current directory
    monkeypatch.chdir(data_root)
    cfg = mi.load_run_config(run_dir / "config.yaml")
    assert factory.zarr_path == cfg.data.dataset_factory.zarr_path

    # overrides; URLs and absolute paths are kept as they are
    cfg = mi.load_run_config(
        run_dir / "config.yaml",
        data_root=tmp_path,
        zarr_path="s3://bucket/radar.zarr",
        index_path="/abs/index.parquet",
        storage_options={"anon": True},
    )
    factory = cfg.data.dataset_factory
    assert factory.zarr_path == "s3://bucket/radar.zarr"
    assert factory.index_path == "/abs/index.parquet"
    assert factory.storage_options == {"anon": True}


def test_find_config_yaml(run_dir, tmp_path):
    ckpt = run_dir / "checkpoints" / "epoch=0-step=1.ckpt"
    assert mi.find_config_yaml(ckpt) == run_dir / "config.yaml"
    assert mi.find_config_yaml(tmp_path / "nowhere" / "x.ckpt") is None


def test_split_init_times(data_root, run_dir):
    cfg = mi.load_run_config(run_dir / "config.yaml", data_root=data_root)
    init_times = mi.split_init_times(cfg)
    assert isinstance(init_times, pd.DatetimeIndex)
    assert init_times.is_monotonic_increasing and init_times.is_unique
    assert len(init_times) > 3

    # the crops mlcast selects for the test split (deterministic: seed 42)
    dataset = _test_dataset(cfg)
    t_abs = dataset.candidates["t"].to_numpy() + dataset._time_index_slice.start
    times = xr.open_zarr(data_root / "data" / "source.zarr").time.values
    expected = np.unique(times[t_abs + INPUT_STEPS - 1])
    np.testing.assert_array_equal(init_times.values, expected)
    assert len(expected) < len(dataset.candidates)  # several crops per time

    # all input and target frames are in the test range (frames 60..79)
    test_start, test_end = times[60], times[-1]
    assert (init_times - (INPUT_STEPS - 1) * pd.Timedelta(STEP) >= test_start).all()
    assert (init_times + FORECAST_STEPS * pd.Timedelta(STEP) <= test_end).all()

    # the validation split ends before the test split
    val_times = mi.split_init_times(cfg, "val")
    assert val_times.max() < test_start
    with pytest.raises(ValueError, match="split must be"):
        mi.split_init_times(cfg, "train")


def test_split_init_times_tuple_splits(data_root, tmp_path):
    times = pd.DatetimeIndex(xr.open_zarr(data_root / "data" / "source.zarr").time.values)
    iso = [str(t) for t in times]
    splits = {
        "time": {"train": (iso[0], iso[39]), "val": (iso[40], iso[59]), "test": (iso[60], iso[79])}
    }
    config_yaml = tmp_path / "tuple_run" / "config.yaml"
    _dump_config(_experiment_config(splits=splits), config_yaml)
    assert "- " in config_yaml.read_text()  # YAML stores the ranges as lists ...

    cfg = mi.load_run_config(config_yaml, data_root=data_root)
    assert cfg.data.splits["time"]["test"] == (iso[60], iso[79])  # ... coerced back to tuples
    init_times = mi.split_init_times(cfg)
    assert len(init_times) > 0
    assert init_times.min() - (INPUT_STEPS - 1) * pd.Timedelta(STEP) >= times[60]
    # same ranges as the ratio splits -> same cases
    ratio_cfg = mi.load_run_config(config_yaml, data_root=data_root)
    ratio_cfg.data.splits = RATIO_SPLITS
    pd.testing.assert_index_equal(init_times, mi.split_init_times(ratio_cfg))


def test_dataset_from_mlcast_config(data_root, run_dir):
    cfg = mi.load_run_config(run_dir / "config.yaml", data_root=data_root)
    all_times = mi.split_init_times(cfg)
    ds = mi.VerificationDataset.from_mlcast_config(cfg)
    np.testing.assert_array_equal(ds.init_times, all_times.values)
    assert ds.input_steps == INPUT_STEPS and ds.forecast_steps == FORECAST_STEPS
    assert ds.variable == "RR" and ds.standard_name == "rainfall_flux"
    assert ds[0]["input"].shape == (INPUT_STEPS, 1, NY, NX)  # full domain, not the crop

    few = mi.VerificationDataset.from_mlcast_config(cfg, max_cases=3, forecast_steps=2)
    assert len(few) == 3 and few.forecast_steps == 2
    np.testing.assert_array_equal(few.init_times[[0, -1]], all_times.values[[0, -1]])
    assert set(few.init_times) <= set(all_times.values)


# --- crop mode -----------------------------------------------------------------------------------


def test_split_crops(data_root, run_dir):
    cfg = mi.load_run_config(run_dir / "config.yaml", data_root=data_root)
    crops, crop_size = mi.split_crops(cfg)
    assert crop_size == (8, 8)
    assert list(crops.columns) == ["init_time", "y", "x"]
    np.testing.assert_array_equal(crops["init_time"].unique(), mi.split_init_times(cfg).values)

    # one row per crop mlcast selects for the test split, at its corner
    dataset = _test_dataset(cfg)
    t_abs = dataset.candidates["t"].to_numpy() + dataset._time_index_slice.start
    times = xr.open_zarr(data_root / "data" / "source.zarr").time.values
    expected = pd.DataFrame(
        {
            "init_time": times[t_abs + INPUT_STEPS - 1],
            "y": dataset.candidates["y"].to_numpy(),
            "x": dataset.candidates["x"].to_numpy(),
        }
    ).sort_values(["init_time", "y", "x"])
    np.testing.assert_array_equal(crops.to_numpy(), expected.to_numpy())
    assert len(crops) > crops["init_time"].nunique()  # several crops per init time


def test_dataset_from_mlcast_config_crops(data_root, run_dir):
    cfg = mi.load_run_config(run_dir / "config.yaml", data_root=data_root)
    crops, _ = mi.split_crops(cfg)
    ds = mi.VerificationDataset.from_mlcast_config(cfg, crops=True)
    assert ds.crop_size == (8, 8) and ds.grid_shape == (8, 8) and ds.grid_coords() == {}
    assert ds.n_crops == len(crops) and len(ds) == crops["init_time"].nunique()
    item = ds[0]
    assert item["input"].shape == (INPUT_STEPS, 1, NY, NX)  # read once per init time
    first = crops[crops["init_time"] == ds.init_times[0]]
    np.testing.assert_array_equal(item["crops"].numpy(), first[["y", "x"]].to_numpy())

    few = mi.VerificationDataset.from_mlcast_config(cfg, crops=True, max_cases=3)
    assert len(few) == 3 and few.n_crops == crops["init_time"].isin(few.init_times).sum()
    fewer = mi.VerificationDataset.from_mlcast_config(cfg, crops=True, max_crops=5)
    assert fewer.n_crops == 5

    with pytest.raises(ValueError, match="max_crops needs crops"):
        mi.VerificationDataset.from_mlcast_config(cfg, max_crops=5)
    with pytest.raises(ValueError, match="crops and region"):
        mi.VerificationDataset.from_mlcast_config(cfg, crops=True, region={"y": slice(0, 8)})


def test_dataset_crops_errors(source_path):
    init_times = _valid_init_times(2)
    crops = pd.DataFrame({"init_time": init_times, "y": [0, 16], "x": [0, 0]})
    with pytest.raises(ValueError, match="crop_size"):
        mi.VerificationDataset(source_path, None, INPUT_STEPS, FORECAST_STEPS, crops=crops)
    with pytest.raises(ValueError, match="extend beyond"):
        mi.VerificationDataset(
            source_path, None, INPUT_STEPS, FORECAST_STEPS, crops=crops, crop_size=(16, 8)
        )
    with pytest.raises(ValueError, match="init_times is needed"):
        mi.VerificationDataset(source_path, None, INPUT_STEPS, FORECAST_STEPS)


class PersistenceModel(torch.nn.Module):
    """Deterministic 'ensemble': the last input frame, for every member and lead time."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))

    def forward(self, x, forecast_steps, ensemble_size=None):
        last = x[:, -1:] * self.scale  # (B, 1, 1, H, W)
        return last.expand(-1, forecast_steps, ensemble_size or 1, -1, -1).contiguous()


_CROP_KWARGS = {"forecast_source": "convgru", "device": "cpu", "progress": False}


def test_verify_crops_whole_grid_equals_full_domain(source_path):
    """One crop covering the grid per init time == the full-domain run (same forecasts)."""
    init_times = _valid_init_times(3)
    crops = pd.DataFrame({"init_time": init_times, "y": 0, "x": 0})
    by_crop = mi.VerificationDataset(
        source_path, None, INPUT_STEPS, FORECAST_STEPS, crops=crops, crop_size=(NY, NX)
    )
    full = mi.VerificationDataset(source_path, init_times, INPUT_STEPS, FORECAST_STEPS)
    kwargs = {**_CROP_KWARGS, "ensemble_size": 3, "batch_size": 2}
    xr.testing.assert_allclose(
        mi.verify_mlcast_model(AdapterModel(), by_crop, **kwargs),
        mi.verify_mlcast_model(AdapterModel(), full, **kwargs),
    )


def test_verify_crops_forecast_full_domain_once(source_path, monkeypatch):
    """Crops are cut out of one full-domain forecast per batch, not forecast themselves."""
    init_times = _valid_init_times(3)
    crops = pd.DataFrame(
        [(t, y, x) for t in init_times for y, x in [(0, 0), (8, 4), (16, 12)]],
        columns=["init_time", "y", "x"],
    )
    ds = mi.VerificationDataset(
        source_path, None, INPUT_STEPS, FORECAST_STEPS, crops=crops, crop_size=(8, 8)
    )
    shapes = []

    class Recording(PersistenceModel):
        def forward(self, x, forecast_steps, ensemble_size=None):
            shapes.append(tuple(x.shape))
            return super().forward(x, forecast_steps, ensemble_size)

    scores = mi.verify_mlcast_model(Recording(), ds, ensemble_size=2, batch_size=2, **_CROP_KWARGS)
    assert shapes == [(2, INPUT_STEPS, 1, NY, NX), (1, INPUT_STEPS, 1, NY, NX)]
    _check_scores(scores, 3, 2)

    # scoring the crops one at a time (small chunks) gives the same pooled scores
    from mlcast_verification_tools.integrations import _mlcast_core

    monkeypatch.setattr(_mlcast_core, "_CROP_CHUNK_ELEMENTS", 1)
    one_by_one = mi.verify_mlcast_model(
        PersistenceModel(), ds, ensemble_size=2, batch_size=2, **_CROP_KWARGS
    )
    xr.testing.assert_allclose(one_by_one, scores)


def test_verify_crops_pooled(source_path):
    """Crop scores of an init time are pooled over the valid pixels of all its crops."""
    init_times = _valid_init_times(2)
    corners = [(0, 0), (8, 4), (16, 12)]  # (0, 0) is never valid -> fewer valid pixels
    crops = pd.DataFrame(
        [(t, y, x) for t in init_times for y, x in corners], columns=["init_time", "y", "x"]
    )
    by_crop = mi.VerificationDataset(
        source_path, None, INPUT_STEPS, FORECAST_STEPS, crops=crops, crop_size=(8, 8)
    )
    assert by_crop.n_crops == 6
    kwargs = {**_CROP_KWARGS, "ensemble_size": 2, "compute_psd": True}
    pooled = mi.verify_mlcast_model(PersistenceModel(), by_crop, batch_size=2, **kwargs)

    values, weights, psds = [], [], []
    for y, x in corners:
        region = {"y": slice(y, y + 8), "x": slice(x, x + 8)}
        ds = mi.VerificationDataset(
            source_path, init_times, INPUT_STEPS, FORECAST_STEPS, region=region
        )
        scores = mi.verify_mlcast_model(PersistenceModel(), ds, **kwargs)
        values.append(scores[["mae.tp", "crps.tp"]])
        weights.append(
            xr.DataArray([float(ds[i]["mask"].sum()) for i in range(2)], dims="init_time")
        )
        psds.append(scores["psd.tp"])
    weight = xr.concat(weights, "crop").assign_coords(init_time=pooled.init_time)
    expected = (xr.concat(values, "crop") * weight).sum("crop") / weight.sum("crop")
    for name in ("mae.tp", "crps.tp"):
        np.testing.assert_allclose(pooled[name].values, expected[name].values, rtol=1e-6)
    np.testing.assert_allclose(
        pooled["psd.tp"].values, xr.concat(psds, "crop").mean("crop").values, rtol=1e-6
    )
    np.testing.assert_allclose(pooled["rank_histogram.tp"].sum("rank").values, 1.0)
    assert pooled.sizes["wavelength"] == 4  # 8-pixel crops


def test_verify_mlcast_run_crops(data_root, run_dir):
    kwargs = _run_kwargs(data_root, max_cases=3, region=None)
    scores = mi.verify_mlcast_run(
        run_dir / "config.yaml", run_dir / "checkpoints" / "epoch=0-step=1.ckpt",
        crops=True, max_crops=7, **kwargs,
    )  # fmt: skip
    assert scores.sizes["wavelength"] == 4
    assert 1 <= scores.sizes[cv.INIT_TIME] <= 3
    with pytest.raises(ValueError, match="cannot be stored"):
        mi.forecast_mlcast(
            run_dir / "config.yaml", run_dir / "checkpoints" / "epoch=0-step=1.ckpt",
            crops=True, **kwargs,
        )  # fmt: skip


def test_load_mlcast_model(tmp_path):
    NowcastLightningModule = _lightning_module()
    torch.manual_seed(0)
    module = NowcastLightningModule(
        network=ConvGruModel(num_blocks=2), ensemble_size=2, loss_class="mse"
    )
    run_dir = tmp_path / "version_0"
    (run_dir / "checkpoints").mkdir(parents=True)
    ckpt = run_dir / "checkpoints" / "epoch=0-step=1.ckpt"
    _save_checkpoint(module, ckpt)

    def assert_same_weights(loaded):
        assert not loaded.training
        for (k, v), (k2, v2) in zip(
            module.state_dict().items(), loaded.state_dict().items(), strict=True
        ):
            assert k == k2
            torch.testing.assert_close(v, v2)

    # explicit network
    assert_same_weights(mi.load_mlcast_model(ckpt, config=None, network=ConvGruModel(num_blocks=2)))

    # the config is a required argument, never looked up next to the checkpoint
    with pytest.raises(TypeError):
        mi.load_mlcast_model(ckpt)
    with pytest.raises(ValueError, match="network is not stored"):
        mi.load_mlcast_model(ckpt, config=None)

    # explicit config (path or loaded)
    config = tmp_path / "elsewhere.yaml"
    _dump_config(_experiment_config(num_blocks=2), config)
    assert_same_weights(mi.load_mlcast_model(ckpt, config=config))
    assert_same_weights(mi.load_mlcast_model(ckpt, config=mi.load_run_config(config)))


@pytest.fixture
def offline_metadata(monkeypatch):
    """Skip the validator's GitHub lookups of `mlcast_created_with` (as in test_io)."""
    mlcast_metadata = pytest.importorskip(
        "mlcast_dataset_validator.checks.global_attributes.mlcast_metadata"
    )
    monkeypatch.setattr(mlcast_metadata, "github_repo_exists", lambda *a: True)
    monkeypatch.setattr(mlcast_metadata, "github_ref_exists", lambda *a: "tags/v0.1.0")


def test_verify_mlcast_run(data_root, run_dir, tmp_path, offline_metadata):
    from mlcast_verification_tools.io import validate_scores, write_scores

    ckpt = run_dir / "checkpoints" / "epoch=0-step=1.ckpt"
    scores = mi.verify_mlcast_run(
        run_dir / "config.yaml",
        ckpt,
        ensemble_size=3,
        data_root=data_root,
        device="cpu",
        max_cases=3,
        batch_size=2,
        progress=False,
    )
    _check_scores(scores, 3, 3)
    # the run directory name, and the persistence baseline
    assert scores[cv.FORECAST_SOURCE].values.tolist() == ["version_0", "persistence"]
    cfg = mi.load_run_config(run_dir / "config.yaml", data_root=data_root)
    assert set(scores[cv.INIT_TIME].values) <= set(mi.split_init_times(cfg).values)

    path = write_scores(
        scores,
        tmp_path / "scores.zarr",
        title="test scores",
        forecasts="tiny ConvGRU",
        truth="synthetic radar",
    )
    report = validate_scores(path)
    assert not report.has_fails(), [r for r in report.results if r.status == "FAIL"]

    # the config is required: there is no auto-discovery next to the checkpoint
    with pytest.raises(TypeError):
        mi.verify_mlcast_run(ckpt, ensemble_size=2)
    with pytest.raises(ValueError, match="config"):
        mi.verify_mlcast_run(None, ckpt, ensemble_size=2)


# --- forecasts and truth of a run, and their equivalence with the on-the-fly scores ---------

REGION = {"y": slice(0, 16), "x": slice(4, None)}


def _run_kwargs(data_root, **extra):
    kwargs = {
        "ensemble_size": 3,
        "data_root": data_root,
        "device": "cpu",
        "max_cases": 3,
        "batch_size": 2,
        "seed": 7,
        "progress": False,
    }
    return {**kwargs, **extra}


def _check_forecasts(
    ds: xr.Dataset,
    n_cases: int,
    n_members: int,
    ny: int,
    nx: int,
    variable: str = "tp",
    units: str = "kg m-2 h-1",
) -> None:
    assert list(ds.data_vars) == [variable]
    tp = ds[variable]
    assert tp.dims == cv.FORECAST_DIMS
    assert tp.shape == (1, n_cases, FORECAST_STEPS, n_members, ny, nx)
    assert tp.dtype == np.float32
    assert tp.attrs["units"] == units  # the source's units
    assert ds[cv.FORECAST_SOURCE].values.tolist() == ["version_0"]
    assert np.issubdtype(ds[cv.INIT_TIME].dtype, np.datetime64)
    assert np.issubdtype(ds[cv.LEAD_TIME].dtype, np.timedelta64)
    assert ds[cv.LEAD_TIME].values[0] == np.timedelta64(5, "m")
    np.testing.assert_array_equal(ds[cv.MEMBER].values, np.arange(n_members))
    assert ds[cv.X].attrs["units"] == "m" and ds.sizes[cv.X] == nx
    values = tp.values
    assert (values[np.isfinite(values)] >= 0).all()
    # pixel (0, 0) is NaN in every frame of the source: no forecast there
    if ny == NY and nx == NX:
        assert np.isnan(values[..., 0, 0]).all()
    assert np.isfinite(values).mean() > 0.9


def test_forecast_mlcast_in_memory_and_zarr(data_root, run_dir, tmp_path):
    config = mi.load_run_config(run_dir / "config.yaml", data_root=data_root)
    ckpt = run_dir / "checkpoints" / "epoch=0-step=1.ckpt"
    kwargs = _run_kwargs(None)

    in_memory = mi.forecast_mlcast(config, ckpt, **kwargs)
    _check_forecasts(in_memory, 3, 3, NY, NX)
    assert in_memory.attrs["split"] == "test" and in_memory.attrs["checkpoint"] == str(ckpt)
    assert "init times of the test split" in in_memory.attrs["forecasts"]
    # same cases as the on-the-fly verification
    scores = mi.verify_mlcast_run(config, ckpt, **kwargs)
    np.testing.assert_array_equal(in_memory[cv.INIT_TIME].values, scores[cv.INIT_TIME].values)

    # written batch by batch (2 + 1 cases) to Zarr: the same forecasts
    out = tmp_path / "forecasts.zarr"
    on_disk = mi.forecast_mlcast(config, ckpt, out=out, **kwargs)
    assert on_disk[cv.VARIABLE].chunks is not None  # lazily opened from the store
    _check_forecasts(on_disk, 3, 3, NY, NX)
    xr.testing.assert_identical(on_disk.load(), in_memory)


def test_open_truth(data_root, run_dir):
    truth = mi.open_truth(run_dir / "config.yaml", data_root=data_root)
    tp = truth[cv.VARIABLE]
    assert tp.dims == cv.TRUTH_DIMS and tp.attrs["units"] == "kg m-2 h-1"
    assert tp.attrs["standard_name"] == "rainfall_flux" and tp.attrs["source_variable"] == "RR"
    assert tp.chunks is not None  # lazy
    src = xr.open_zarr(data_root / "data" / "source.zarr")
    np.testing.assert_array_equal(tp.values, src.RR.values)  # native units
    np.testing.assert_array_equal(truth[cv.TIME].values, src.time.values)
    assert "XX-TEST v0.1.0" in truth.attrs["truth"]

    cropped = mi.open_truth(run_dir / "config.yaml", data_root=data_root, region=REGION)
    assert cropped.sizes == {cv.TIME: RUN_NT, cv.Y: 16, cv.X: NX - 4}
    np.testing.assert_array_equal(cropped[cv.X].values, src.x.values[4:])


def test_open_truth_units_and_missing_times(tmp_path, run_dir):
    root = tmp_path / "si"
    (root / "data").mkdir(parents=True)
    src = _source_dataset(units="kg m-2 s-1", missing_time_index=None, nt=RUN_NT)
    src = src.assign_coords(missing_times=("missing_times", src.time.values[[5]]))
    src.to_zarr(root / "data" / "source.zarr", zarr_format=2)
    truth = mi.open_truth(run_dir / "config.yaml", data_root=root)
    assert truth.sizes[cv.TIME] == RUN_NT - 1
    assert src.time.values[5] not in truth[cv.TIME].values
    assert truth[cv.VARIABLE].attrs["units"] == "kg m-2 s-1"  # not converted
    np.testing.assert_array_equal(truth[cv.VARIABLE].values[:5], src.RR.values[:5])


def test_open_truth_reflectivity(dbz_data_root, dbz_run_dir):
    truth = mi.open_truth(dbz_run_dir / "config.yaml", data_root=dbz_data_root)
    dbz = truth["dbz"]
    assert list(truth.data_vars) == ["dbz"] and dbz.attrs["units"] == "dBZ"
    assert dbz.attrs["standard_name"] == "equivalent_reflectivity_factor"
    src = xr.open_zarr(dbz_data_root / "data" / "source.zarr")
    np.testing.assert_array_equal(dbz.values, np.clip(src.dbz.values, 0, 60))
    assert "clipped to [0, 60] dBZ" in truth.attrs["truth"]


def _assert_scores_equal(got: xr.Dataset, expected: xr.Dataset) -> None:
    """`got` equals the sources of `expected` it has (forecast-then-score has no persistence)."""
    sources = got[cv.FORECAST_SOURCE].values
    expected = expected.sel({cv.FORECAST_SOURCE: sources})
    assert set(got.data_vars) == set(expected.data_vars)
    for coord in (cv.FORECAST_SOURCE, cv.INIT_TIME, cv.LEAD_TIME):
        np.testing.assert_array_equal(got[coord].values, expected[coord].values)
    for name in expected.data_vars:
        np.testing.assert_allclose(
            got[name].values, expected[name].values, rtol=1e-6, atol=1e-12, err_msg=name
        )


@pytest.mark.parametrize("region", [None, REGION], ids=["full", "region"])
def test_forecast_then_score_equals_verify(data_root, run_dir, region):
    """compute_scores(forecast_mlcast(...), open_truth(config)) == verify_mlcast_run(...)."""
    from mlcast_verification_tools.offline import compute_scores

    config = mi.load_run_config(run_dir / "config.yaml", data_root=data_root)
    ckpt = run_dir / "checkpoints" / "epoch=0-step=1.ckpt"
    kwargs = _run_kwargs(None, region=region, member_chunk=2)

    expected = mi.verify_mlcast_run(config, ckpt, **kwargs)
    ds_forecasts = mi.forecast_mlcast(config, ckpt, **kwargs)
    ds_truth = mi.open_truth(config, region=region)
    got = compute_scores(
        ds_forecasts, ds_truth, device="cpu", batch_size=kwargs["batch_size"], seed=kwargs["seed"]
    )
    _check_scores(got, 3, 3)
    _assert_scores_equal(got, expected)


def test_forecast_then_score_equals_verify_reflectivity(
    dbz_data_root, dbz_run_dir, tmp_path, offline_metadata
):
    """A model trained on dBZ: forecasts, truth and scores in dBZ, both paths agree."""
    from mlcast_verification_tools.io import validate_scores, write_scores
    from mlcast_verification_tools.offline import compute_scores

    config = mi.load_run_config(dbz_run_dir / "config.yaml", data_root=dbz_data_root)
    ckpt = dbz_run_dir / "checkpoints" / "epoch=0-step=1.ckpt"
    kwargs = _run_kwargs(None)

    expected = mi.verify_mlcast_run(config, ckpt, **kwargs)
    _check_scores(expected, 3, 3, variable="dbz")
    assert expected["mae.dbz"].attrs["units"] == "dBZ"

    ds_forecasts = mi.forecast_mlcast(config, ckpt, **kwargs)
    _check_forecasts(ds_forecasts, 3, 3, NY, NX, variable="dbz", units="dBZ")
    assert np.nanmax(ds_forecasts["dbz"].values) <= 60
    got = compute_scores(
        ds_forecasts,
        mi.open_truth(config),
        variable="dbz",
        device="cpu",
        batch_size=kwargs["batch_size"],
        seed=kwargs["seed"],
    )
    _assert_scores_equal(got, expected)

    path = write_scores(
        expected, tmp_path / "scores.zarr", title="t", forecasts="f", truth="synthetic dBZ"
    )
    report = validate_scores(path)
    assert not report.has_fails(), [r for r in report.results if r.status == "FAIL"]


def test_config_object_not_modified(data_root, run_dir, tmp_path):
    config = mi.load_run_config(run_dir / "config.yaml", data_root=data_root)
    zarr_path = config.data.dataset_factory.zarr_path
    with pytest.raises(Exception):  # noqa: B017 - the overridden store does not exist
        mi.open_truth(config, zarr_path=tmp_path / "nowhere.zarr")
    assert config.data.dataset_factory.zarr_path == zarr_path


# --- command line -------------------------------------------------------------------------------


def _cli_case_args(run_dir) -> list[str]:
    return [
        "--config", str(run_dir / "config.yaml"),
        "--checkpoint", str(run_dir / "checkpoints" / "epoch=0-step=1.ckpt"),
        "--ensemble-size", "3",
        "--max-cases", "4",
        "--device", "cpu",
        "--batch-size", "2",
        "--seed", "3",
        "--region-y", "0:16",
        "--region-x", "4:",
        "--no-progress",
    ]  # fmt: skip


def test_verify_trained_model_cli(data_root, run_dir, monkeypatch):
    from mlcast_verification_tools.cli import main
    from mlcast_verification_tools.io import open_scores

    monkeypatch.chdir(data_root)  # the config's data paths are relative to it
    out = data_root / "scores.nc"
    argv = ["verify-trained-model", *_cli_case_args(run_dir), "--out", str(out)]
    assert main(argv) == 0
    scores = open_scores(out).load()
    _check_scores(scores, 4, 3)
    assert scores[cv.FORECAST_SOURCE].values.tolist() == ["version_0", "persistence"]
    assert "; persistence: the last observed input frame" in scores.attrs["forecasts"]
    # the model and persistence plot together (persistence's rank histogram is NaN-padded)
    assert main(["plot", str(out), "--out", str(data_root / "figures")]) == 0
    assert any((data_root / "figures").iterdir())
    no_baseline = data_root / "no_baseline.nc"
    assert main([*argv[:-1], str(no_baseline), "--no-persistence"]) == 0
    assert open_scores(no_baseline)[cv.FORECAST_SOURCE].values.tolist() == ["version_0"]
    assert "version_0" in scores.attrs["title"] and "test split" in scores.attrs["title"]
    assert "XX-TEST v0.1.0" in scores.attrs["truth"]
    assert "init times of the test split" in scores.attrs["forecasts"]
    assert "sub-domain y=0:16, x=4:" in scores.attrs["forecasts"]
    # provenance is optional: this tool's repository@revision, no creator contact
    assert scores.attrs["mlcast_created_with"].startswith(
        "https://github.com/leifdenby/mlcast-verification-tools@"
    )
    assert "mlcast_created_by" not in scores.attrs

    # the mlcast plugin entry point gives the same scores
    out2 = data_root / "scores2.nc"
    assert mi.verify_trained_model_main([*_cli_case_args(run_dir), "--out", str(out2)]) == 0
    scores2 = open_scores(out2).load()
    # the two files are written at different times
    scores2.attrs["mlcast_created_on"] = scores.attrs["mlcast_created_on"]
    xr.testing.assert_identical(scores2, scores)

    # the arguments the config provides are gone, and the config is required
    parser = mi.build_verify_trained_model_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([*argv[1:], "--source", "x.zarr"])
    without_config = [a for a in argv[1:] if a not in ("--config", str(run_dir / "config.yaml"))]
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(without_config)
    assert exc.value.code == 2


@pytest.mark.parametrize("crops", [False, True])
def test_verify_trained_model_cli_multi_device(data_root, run_dir, monkeypatch, crops):
    """`--devices 2` (two CPU processes) scores the same cases as one process."""
    import subprocess
    import sys

    from mlcast_verification_tools.cli import main
    from mlcast_verification_tools.io import open_scores

    monkeypatch.chdir(data_root)
    case_args = _cli_case_args(run_dir)
    if crops:  # full domain, scored on the crops
        case_args = case_args[: case_args.index("--region-y")] + ["--crops", "--no-progress"]
    single = data_root / "single.nc"
    assert main(["verify-trained-model", *case_args, "--out", str(single)]) == 0
    multi = data_root / "multi.nc"
    # Fabric re-runs the command line for the second process, so not from within pytest
    cmd = [sys.executable, "-m", "mlcast_verification_tools.cli", "verify-trained-model",
           *case_args, "--devices", "2", "--out", str(multi)]  # fmt: skip
    subprocess.run(cmd, cwd=data_root, check=True, timeout=600)

    got, expected = open_scores(multi).load(), open_scores(single).load()
    _check_scores(got, expected.sizes[cv.INIT_TIME], 3)
    np.testing.assert_array_equal(got[cv.INIT_TIME].values, expected[cv.INIT_TIME].values)
    # persistence is deterministic, except the rank histogram's random tie breaking, which
    # (like the model's ensemble) is seeded per process
    deterministic = [name for name in got.data_vars if not name.startswith(cv.RANK_HISTOGRAM)]
    xr.testing.assert_allclose(
        got[deterministic].sel({cv.FORECAST_SOURCE: "persistence"}),
        expected[deterministic].sel({cv.FORECAST_SOURCE: "persistence"}),
    )
    assert "scored in 2 processes" in got.attrs["forecasts"]

    with pytest.raises(SystemExit, match="only the device type"):
        main(["verify-trained-model", *case_args, "--device", "cuda:1", "--devices", "2",
              "--out", str(multi)])  # fmt: skip


def test_verify_trained_model_cli_crops(data_root, run_dir, monkeypatch):
    from mlcast_verification_tools.cli import main
    from mlcast_verification_tools.io import open_scores

    monkeypatch.chdir(data_root)
    case_args = [a for a in _cli_case_args(run_dir) if a not in ("--region-y", "0:16")]
    case_args = (
        case_args[: case_args.index("--region-x")] + case_args[case_args.index("--region-x") + 2 :]
    )
    out = data_root / "scores.nc"
    argv = ["verify-trained-model", *case_args, "--crops", "--max-crops", "6", "--out", str(out)]
    assert main(argv) == 0
    scores = open_scores(out).load()
    _check_scores(scores, scores.sizes[cv.INIT_TIME], 3)
    assert scores.sizes["wavelength"] == 4
    assert "full domain scored on the 6 sampled 8x8 crops" in scores.attrs["forecasts"]

    with pytest.raises(SystemExit, match="--region"):
        main([*argv[:-2], "--region-y", "0:8", "--out", str(out)])
    no_crops = [a for a in argv if a != "--crops"]
    with pytest.raises(SystemExit, match="--max-crops needs --crops"):
        main(no_crops)


def test_forecast_then_score_cli_equals_verify_cli(data_root, run_dir, monkeypatch):
    """`forecast-trained-model` + `score` == `verify-trained-model`, also via `plot`."""
    from mlcast_verification_tools.cli import main
    from mlcast_verification_tools.io import open_scores

    monkeypatch.chdir(data_root)
    case_args = _cli_case_args(run_dir)
    forecasts = data_root / "forecasts.zarr"
    assert main(["forecast-trained-model", *case_args, "--out", str(forecasts)]) == 0
    ds_forecasts = xr.open_zarr(forecasts)
    _check_forecasts(ds_forecasts, 4, 3, 16, NX - 4)

    via_score = data_root / "via_score.zarr"
    argv = [
        "score", str(forecasts),
        "--truth-from-config", str(run_dir / "config.yaml"),
        "--region-y", "0:16", "--region-x", "4:",
        "--out", str(via_score),
        "--device", "cpu", "--batch-size", "2", "--seed", "3",
    ]  # fmt: skip
    assert main(argv) == 0
    direct = data_root / "direct.zarr"
    assert main(["verify-trained-model", *case_args, "--out", str(direct)]) == 0

    got, expected = open_scores(via_score).load(), open_scores(direct).load()
    _assert_scores_equal(got, expected)
    # the on-the-fly scores also describe the persistence baseline
    assert expected.attrs["forecasts"].startswith(got.attrs["forecasts"] + "; persistence: ")
    assert got.attrs["truth"] == expected.attrs["truth"]

    # a stored truth dataset works the same
    truth_path = data_root / "truth.zarr"
    mi.open_truth(run_dir / "config.yaml", region=REGION).to_zarr(truth_path)
    via_truth = data_root / "via_truth.nc"
    argv = ["score", str(forecasts), "--truth", str(truth_path), "--out", str(via_truth),
            "--device", "cpu", "--batch-size", "2", "--seed", "3"]  # fmt: skip
    assert main(argv) == 0
    _assert_scores_equal(open_scores(via_truth).load(), expected)

    figures = data_root / "figures"
    assert main(["plot", str(via_score), "--out", str(figures)]) == 0
    assert len(list(figures.iterdir())) == 4
