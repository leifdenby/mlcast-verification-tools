"""Tests of the mlcast integration (scoring an mlcast model while it forecasts)."""

from __future__ import annotations

import argparse
import importlib

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


def test_denormalize_torch_rejects_unsupported():
    with pytest.raises(ValueError, match="not supported"):
        mi.denormalize_torch(torch.zeros(2), "equivalent_reflectivity_factor")
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


def test_dataset_region_and_units_conversion(tmp_path):
    path = tmp_path / "flux_si.zarr"
    src = _source_dataset(units="kg m-2 s-1")
    src.to_zarr(path, zarr_format=2)
    region = {"y": slice(2, 18), "x": slice(4, 16)}
    ds = mi.VerificationDataset(
        path, _valid_init_times(2), INPUT_STEPS, FORECAST_STEPS, region=region
    )
    item = ds[0]
    assert item["input"].shape == (INPUT_STEPS, 1, 16, 12)
    assert ds.to_mm_per_hour == 3600.0
    t_init = int(np.where(src.time.values == ds.init_times[0])[0][0])
    expected = src.RR.values[t_init + 1 : t_init + 1 + FORECAST_STEPS, 2:18, 4:16] * np.float32(
        3600
    )
    np.testing.assert_allclose(item["target"].numpy(), expected, rtol=1e-6)


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
    src["RR"].attrs["standard_name"] = "rainfall_amount"
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


def _check_scores(scores: xr.Dataset, n_cases: int, n_members: int, psd: bool = True) -> None:
    assert dict(scores.sizes)[cv.FORECAST_SOURCE] == 1
    assert scores.sizes[cv.INIT_TIME] == n_cases
    assert scores.sizes[cv.LEAD_TIME] == FORECAST_STEPS
    assert scores.sizes[cv.RANK] == n_members + 1
    names = {cv.var_name(m) for m in (cv.MAE, cv.CRPS, cv.RANK_HISTOGRAM)}
    if psd:
        names |= {cv.var_name(cv.PSD), cv.var_name(cv.PSD_OBS)}
    assert names <= set(scores.data_vars)
    for name in names:
        assert np.isfinite(scores[name].values).all(), name
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
    assert scores[cv.FORECAST_SOURCE].values.tolist() == ["convgru"]
    # same seed -> same ensemble -> same scores
    again = mi.verify_mlcast_model(AdapterModel(), ds, **kwargs)
    xr.testing.assert_identical(scores, again)


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


def _write_sampling_index(path, zarr_path: str, times: np.ndarray) -> None:
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
        data_var="RR",
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
        data_kind="rainrate",
        units="kg m-2 h-1",
    )
    schema = build_schema(params)
    pq.write_table(pa.Table.from_pydict(columns, schema=schema), path)


def _experiment_config(splits=None, num_blocks: int = 2):
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
    assert_same_weights(mi.load_mlcast_model(ckpt, network=ConvGruModel(num_blocks=2)))

    # no network, no config: mlcast's own loading cannot restore the network
    with pytest.raises(ValueError, match="network is not stored"):
        mi.load_mlcast_model(ckpt)

    # explicit config (path or loaded), and auto-discovered <run>/config.yaml
    config = tmp_path / "elsewhere.yaml"
    _dump_config(_experiment_config(num_blocks=2), config)
    assert_same_weights(mi.load_mlcast_model(ckpt, config=config))
    assert_same_weights(mi.load_mlcast_model(ckpt, config=mi.load_run_config(config)))
    (run_dir / "config.yaml").write_text(config.read_text())
    assert_same_weights(mi.load_mlcast_model(ckpt))


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
        ckpt,
        ensemble_size=3,
        data_root=data_root,
        device="cpu",
        max_cases=3,
        batch_size=2,
        progress=False,
    )
    _check_scores(scores, 3, 3)
    assert scores[cv.FORECAST_SOURCE].values.tolist() == ["version_0"]  # run directory name
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

    with pytest.raises(FileNotFoundError, match="config.yaml"):
        mi.verify_mlcast_run(tmp_path / "model.ckpt", ensemble_size=2)


def test_run_mlcast_cli(data_root, run_dir, monkeypatch):
    from mlcast_verification_tools.cli import main
    from mlcast_verification_tools.io import open_scores

    monkeypatch.chdir(data_root)  # the config's data paths are relative to it
    out = data_root / "scores.nc"
    argv = [
        "run-mlcast",
        "--checkpoint", str(run_dir / "checkpoints" / "epoch=0-step=1.ckpt"),
        "--out", str(out),
        "--ensemble-size", "3",
        "--max-cases", "4",
        "--device", "cpu",
        "--batch-size", "2",
        "--region-y", "0:16",
        "--region-x", "4:",
        "--no-progress",
    ]  # fmt: skip
    assert main(argv) == 0
    scores = open_scores(out).load()
    _check_scores(scores, 4, 3)
    assert "version_0" in scores.attrs["title"] and "test split" in scores.attrs["title"]
    assert "XX-TEST v0.1.0" in scores.attrs["truth"]
    assert "init times of the test split" in scores.attrs["forecasts"]
    # provenance is optional: this tool's repository@revision, no creator contact
    assert scores.attrs["mlcast_created_with"].startswith(
        "https://github.com/leifdenby/mlcast-verification-tools@"
    )
    assert "mlcast_created_by" not in scores.attrs

    # the arguments the config provides are gone
    parser = argparse.ArgumentParser()
    mi.add_run_mlcast_arguments(parser)
    with pytest.raises(SystemExit):
        parser.parse_args([*argv[1:], "--source", "x.zarr"])
