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


def _source_dataset(units: str = "kg m-2 h-1", missing_time_index: int | None = 30) -> xr.Dataset:
    """Tiny dataset in the layout of the mlcast source data (cf. IT-DPC-SRI v0.1.0)."""
    rng = np.random.default_rng(1)
    rr = rng.gamma(0.5, 2.0, size=(NT, NY, NX)).astype(np.float32)
    rr[rng.random(rr.shape) < 0.4] = 0.0
    rr[:, 0, 0] = np.nan  # never valid
    rr[10, 5, 5] = np.nan  # invalid in one frame only
    time = T0 + STEP * np.arange(NT)
    keep = np.ones(NT, bool)
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
            "y": ("y", 649500.0 - 1000.0 * np.arange(NY), {"units": "m"}),
            "x": ("x", -599500.0 + 1000.0 * np.arange(NX), {"units": "m"}),
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


def test_verify_member_chunks_and_workers(source_path):
    ds = mi.VerificationDataset(source_path, _valid_init_times(3), INPUT_STEPS, FORECAST_STEPS)
    scores = mi.verify_mlcast_model(
        AdapterModel(),
        ds,
        forecast_source="convgru",
        ensemble_size=5,
        member_chunk=2,
        batch_size=2,
        num_workers=1,
        compute_psd=False,
        device="cpu",
        progress=False,
    )
    _check_scores(scores, 3, 5, psd=False)


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


def _tiny_experiment_yaml(num_blocks: int = 2) -> str:
    pytest.importorskip("fiddle")
    _lightning_module()
    import mlcast.config.orchestrator  # noqa: F401  registers mlcast's `!type` representer
    from fiddle.experimental.yaml_serialization import dump_yaml
    from mlcast.config.base import training_experiment

    cfg = training_experiment.as_buildable()
    cfg.pl_module.network.num_blocks = num_blocks
    return dump_yaml(cfg)


def test_load_fiddle_yaml_roundtrip():
    import fiddle as fdl

    text = _tiny_experiment_yaml(num_blocks=3)
    cfg = mi.load_fiddle_yaml(text)
    assert isinstance(cfg, fdl.Config)
    assert cfg.pl_module.ensemble_size == 2
    assert isinstance(cfg.pl_module.optimizer, fdl.Partial)
    net = fdl.build(cfg.pl_module.network)
    assert isinstance(net, ConvGruModel) and net.num_blocks == 3
    net2 = mi.build_network_from_config_yaml(text)
    assert {k: v.shape for k, v in net.state_dict().items()} == {
        k: v.shape for k, v in net2.state_dict().items()
    }
    with pytest.raises(KeyError, match="no 'pl_module.nope'"):
        mi.build_network_from_config_yaml(text, key="pl_module.nope")


def test_load_fiddle_yaml_type_tag():
    loaded = mi.load_fiddle_yaml("a: !type torch.nn.modules.conv.Conv2d\n")
    assert loaded["a"] is torch.nn.Conv2d


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

    # explicit config.yaml, and auto-discovered <run>/config.yaml
    config = tmp_path / "elsewhere.yaml"
    config.write_text(_tiny_experiment_yaml(num_blocks=2))
    assert_same_weights(mi.load_mlcast_model(ckpt, config_yaml=config))
    (run_dir / "config.yaml").write_text(config.read_text())
    assert_same_weights(mi.load_mlcast_model(ckpt))


def test_run_mlcast_cli(tmp_path, source_path):
    NowcastLightningModule = _lightning_module()
    from mlcast_verification_tools.io import open_scores

    torch.manual_seed(0)
    module = NowcastLightningModule(network=ConvGruModel(num_blocks=2), ensemble_size=2)
    ckpt = tmp_path / "model.ckpt"
    _save_checkpoint(module, ckpt)
    (tmp_path / "config.yaml").write_text(_tiny_experiment_yaml(num_blocks=2))

    parser = argparse.ArgumentParser()
    mi.add_run_mlcast_arguments(parser)
    out = tmp_path / "scores.nc"
    args = parser.parse_args(
        [
            "--checkpoint", str(ckpt),
            "--source", str(source_path),
            "--start", str(_valid_init_times(4)[0]),
            "--end", str(_valid_init_times(4)[-1]),
            "--freq", "PT15M",
            "--input-steps", str(INPUT_STEPS),
            "--forecast-steps", str(FORECAST_STEPS),
            "--ensemble-size", "3",
            "--forecast-source", "convgru-test",
            "--device", "cpu",
            "--batch-size", "2",
            "--region-y", "0:16",
            "--region-x", "4:",
            "--no-progress",
            "--out", str(out),
            "--title", "test scores",
            "--created-by", "Test <test@example.com>",
            "--created-with", "https://github.com/mlcast-community/mlcast-dataset-xx-test-scores@v0.1.0",
        ]
    )  # fmt: skip
    assert mi.run_mlcast_from_args(args) == 0
    scores = open_scores(out).load()
    _check_scores(scores, 4, 3)
    assert scores.attrs["title"] == "test scores"
    assert "XX-TEST v0.1.0" in scores.attrs["truth"]
    assert "convgru-test" in scores.attrs["forecasts"]
    assert scores.attrs["mlcast_created_with"].endswith("@v0.1.0")
