"""Tests of the `mlcast.verify` command line interface."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pytest

from mlcast_verification_tools.cli import build_parser, main
from mlcast_verification_tools.testing import make_synthetic_scores


@pytest.fixture
def scores_nc(tmp_path):
    path = tmp_path / "scores.nc"
    make_synthetic_scores(n_init_times=4, n_lead_times=4, domain_size=32).to_netcdf(path)
    return path


def test_plot_writes_pngs(scores_nc, tmp_path, capsys):
    out = tmp_path / "figs"
    assert main(["plot", str(scores_nc), "--out", str(out)]) == 0
    names = sorted(p.name for p in out.iterdir())
    assert names == ["crps_tp.png", "mae_tp.png", "rank_histogram_tp.png", "spectra_tp.png"]
    assert all((out / n).read_bytes().startswith(b"\x89PNG") for n in names)
    assert "mae_tp.png" in capsys.readouterr().out


def test_plot_zarr_and_format(tmp_path):
    path = tmp_path / "scores.zarr"
    make_synthetic_scores(n_init_times=3, n_lead_times=3, domain_size=32)[["mae.tp"]].to_zarr(path)
    out = tmp_path / "figs"
    assert main(["plot", str(path), "--out", str(out), "--format", "pdf"]) == 0
    assert [p.name for p in out.iterdir()] == ["mae_tp.pdf"]


def test_subcommands_registered():
    parser = build_parser()
    help_text = parser.format_help()
    for name in ("plot", "score", "verify-trained-model", "forecast-trained-model"):
        assert name in help_text
    assert "run-mlcast" not in help_text


def test_requires_subcommand():
    with pytest.raises(SystemExit):
        main([])


@pytest.mark.parametrize("command", ["verify-trained-model", "forecast-trained-model"])
def test_trained_model_commands_require_config(command, capsys):
    with pytest.raises(SystemExit) as exc:
        main([command, "--checkpoint", "x.ckpt", "--out", "out.zarr"])
    assert exc.value.code == 2
    assert "--config" in capsys.readouterr().err


def test_mlcast_plugin_entry_points(capsys):
    """`mlcast.commands` contract: main(argv) -> int | None, help text = first doc line."""
    from mlcast_verification_tools.integrations import mlcast as mi

    for func in (mi.verify_trained_model_main, mi.forecast_trained_model_main):
        first_line = func.__doc__.strip().splitlines()[0]
        assert first_line.endswith(".") and "trained mlcast model" in first_line
        with pytest.raises(SystemExit) as exc:
            func(["--checkpoint", "x.ckpt", "--out", "out.zarr"])
        assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        mi.verify_trained_model_main(["--help"])
    assert exc.value.code == 0
    assert "mlcast verify-trained-model" in capsys.readouterr().out


def test_score_truth_options_exclusive(tmp_path):
    with pytest.raises(SystemExit):
        main(["score", "fc.zarr", "--out", str(tmp_path / "s.nc")])  # no truth
    with pytest.raises(SystemExit):
        main(
            ["score", "fc.zarr", "--truth", "t.zarr", "--truth-from-config", "c.yaml",
             "--out", str(tmp_path / "s.nc")]
        )  # fmt: skip


def test_score_cli(tmp_path):
    """`score` gives compute_scores' result, written with write_scores."""
    pytest.importorskip("torch")
    import numpy as np
    import xarray as xr

    from mlcast_verification_tools.io import open_scores
    from mlcast_verification_tools.offline import compute_scores

    rng = np.random.default_rng(0)
    init = np.datetime64("2023-07-01T00:00", "ns") + np.timedelta64(15, "m") * np.arange(3)
    lead = np.timedelta64(5, "m").astype("timedelta64[ns]") * np.arange(1, 4)
    time = np.datetime64("2023-07-01T00:00", "ns") + np.timedelta64(5, "m") * np.arange(16)
    y, x = 1000.0 * np.arange(8), 1000.0 * np.arange(10)
    attrs = {"units": "mm h-1"}
    fc = xr.Dataset(
        {"tp": (("forecast_source", "init_time", "lead_time", "member", "y", "x"),
                rng.gamma(0.5, 2.0, (1, 3, 3, 4, 8, 10)).astype("float32"), attrs)},
        coords={"forecast_source": ["a"], "init_time": init, "lead_time": lead,
                "member": np.arange(4), "y": y, "x": x},
        attrs={"forecasts": "synthetic forecasts"},
    )  # fmt: skip
    truth = xr.Dataset(
        {"tp": (("time", "y", "x"), rng.gamma(0.5, 2.0, (16, 8, 10)).astype("float32"), attrs)},
        coords={"time": time, "y": y, "x": x},
    )
    fc.to_zarr(tmp_path / "fc.zarr")
    truth.to_zarr(tmp_path / "truth.zarr")
    out = tmp_path / "scores.nc"
    argv = ["score", str(tmp_path / "fc.zarr"), "--truth", str(tmp_path / "truth.zarr"),
            "--out", str(out), "--device", "cpu", "--batch-size", "2"]  # fmt: skip
    assert main(argv) == 0
    got = open_scores(out).load()
    expected = compute_scores(fc, truth, device="cpu", batch_size=2)
    for name in expected.data_vars:
        np.testing.assert_allclose(got[name].values, expected[name].values, rtol=1e-6)
    assert got.attrs["forecasts"] == "synthetic forecasts"
