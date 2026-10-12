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


def test_run_mlcast_subcommand_registered():
    parser = build_parser()
    help_text = parser.format_help()
    assert "plot" in help_text and "run-mlcast" in help_text


def test_requires_subcommand():
    with pytest.raises(SystemExit):
        main([])
