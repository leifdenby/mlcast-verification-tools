"""Tests of the plotting module and its derived-quantity helpers."""

from __future__ import annotations

import subprocess
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from matplotlib.figure import Figure

from mlcast_verification_tools import plots
from mlcast_verification_tools.plots import _style
from mlcast_verification_tools.testing import make_synthetic_scores


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def ds():
    return make_synthetic_scores(n_init_times=6, n_lead_times=6, domain_size=64)


@pytest.fixture
def ds_truncated(ds):
    """model_b has no scores beyond the 3rd lead time (like a model stopping at 90 min)."""
    ds = ds.copy(deep=True)
    for name in ds.data_vars:
        ds[name].loc[{"forecast_source": "model_b", "lead_time": ds.lead_time[3:]}] = np.nan
    return ds


def _tiny(sources=("a", "b"), sizes=(3, 1), n_init=2, lead_minutes=(10, 20)):
    """Hand-made dataset: flat rank histograms, psd = 2 * psd_obs."""
    k_max = max(sizes)
    rank = np.arange(1, k_max + 2)
    lead = pd.to_timedelta(list(lead_minutes), unit="min")
    init = pd.date_range("2025-01-01", periods=n_init, freq="1D")
    rh = np.full((len(sources), n_init, len(lead), k_max + 1), np.nan)
    for i, k in enumerate(sizes):
        rh[i, ..., : k + 1] = 1.0 / (k + 1)
    wl = np.array([80.0, 40.0, 16.0, 4.0])
    psd_obs = np.ones((len(sources), n_init, len(lead), wl.size)) * wl
    dims = ("forecast_source", "init_time", "lead_time")
    return xr.Dataset(
        {
            "rank_histogram.tp": ((*dims, "rank"), rh),
            "psd.tp": ((*dims, "wavelength"), 2 * psd_obs),
            "psd_obs.tp": ((*dims, "wavelength"), psd_obs),
            "mae.tp": (dims, np.ones((len(sources), n_init, len(lead)))),
        },
        coords={
            "forecast_source": list(sources),
            "init_time": init,
            "lead_time": lead,
            "rank": rank,
            "wavelength": wl,
        },
    )


def test_ensemble_sizes_from_nan_padding():
    assert _style.ensemble_sizes(_tiny()["rank_histogram.tp"]) == {"a": 3, "b": 1}


def test_lead_time_minutes():
    lt = pd.to_timedelta([5, 90], unit="min").values
    np.testing.assert_allclose(_style.lead_time_minutes(lt), [5.0, 90.0])


def test_rmsd_flat_histogram_is_zero():
    rmsd = plots.rank_histogram_rmsd(_tiny())
    assert rmsd.dims == ("forecast_source", "lead_time")
    np.testing.assert_allclose(rmsd.values, 0.0, atol=1e-12)


def test_rmsd_uses_case_averaged_histogram():
    ds = _tiny(sources=("a",), sizes=(1,), n_init=2, lead_minutes=(10,))
    # two cases with opposite deviations average to a flat histogram -> RMSD 0
    ds["rank_histogram.tp"][0, 0, 0] = [0.8, 0.2]
    ds["rank_histogram.tp"][0, 1, 0] = [0.2, 0.8]
    assert plots.rank_histogram_rmsd(ds).item() == pytest.approx(0.0)
    # both cases skewed the same way: pbar = [0.8, 0.2], deviation 0.3 at each rank
    ds["rank_histogram.tp"][0, 1, 0] = [0.8, 0.2]
    assert plots.rank_histogram_rmsd(ds).item() == pytest.approx(0.3)


def test_rmsd_nan_where_source_has_no_data():
    ds = _tiny()
    ds["rank_histogram.tp"].loc[{"forecast_source": "b", "lead_time": ds.lead_time[1]}] = np.nan
    rmsd = plots.rank_histogram_rmsd(ds)
    assert np.isnan(rmsd.sel(forecast_source="b").values[1])
    assert np.isfinite(rmsd.sel(forecast_source="a").values).all()


def test_excess_power_doubling_is_3db():
    excess = plots.excess_power_db(_tiny(), scales_km=(74, 16, 4))
    assert excess.dims == ("forecast_source", "lead_time", "wavelength")
    np.testing.assert_allclose(excess.wavelength.values, [80.0, 16.0, 4.0])
    np.testing.assert_allclose(excess.target_wavelength.values, [74.0, 16.0, 4.0])
    np.testing.assert_allclose(excess.values, 10 * np.log10(2.0))
    assert excess.values[0, 0, 0] == pytest.approx(3.0103, abs=1e-4)


def test_case_mean_skips_nan():
    da = xr.DataArray([[1.0, np.nan, 3.0]], dims=("lead_time", "init_time"))
    assert _style.case_mean(da).item() == pytest.approx(2.0)


@pytest.mark.parametrize("metric", ["mae", "crps"])
def test_plot_score_vs_lead_time(ds_truncated, metric):
    fig = plots.plot_score_vs_lead_time(ds_truncated, metric)
    assert isinstance(fig, Figure)
    lines = {ln.get_label(): ln for ln in fig.axes[0].get_lines()}
    assert set(lines) == {"model_a", "model_b"}
    y_b = np.asarray(lines["model_b"].get_ydata(), dtype=float)
    assert np.isfinite(y_b[:3]).all() and np.isnan(y_b[3:]).all()


def test_plot_score_vs_lead_time_into_axes(ds):
    fig, ax = plt.subplots()
    assert plots.plot_score_vs_lead_time(ds, "mae", ax=ax) is fig


def test_plot_score_vs_lead_time_rejects_extra_dims(ds):
    with pytest.raises(ValueError):
        plots.plot_score_vs_lead_time(ds, "rank_histogram")


def test_source_colours_consistent(ds):
    fig1 = plots.plot_score_vs_lead_time(ds, "mae")
    fig2 = plots.plot_rank_rmsd(ds)
    c1 = {ln.get_label(): ln.get_color() for ln in fig1.axes[0].get_lines()}
    c2 = {ln.get_label(): ln.get_color() for ln in fig2.axes[0].get_lines()}
    assert c1 == c2


def test_plot_rank_histograms(ds_truncated):
    fig = plots.plot_rank_histograms(ds_truncated)
    assert len(fig.axes) == 3  # two sources + RMSD
    fig = plots.plot_rank_histograms(ds_truncated, lead_times=[10, 30], include_rmsd=False)
    assert len(fig.axes) == 2
    labels = [ln.get_label() for ln in fig.axes[0].get_lines()]
    assert "10 min" in labels and "30 min" in labels


def test_plot_spectra(ds_truncated):
    assert isinstance(plots.plot_psd(ds_truncated), Figure)
    assert isinstance(plots.plot_excess_power(ds_truncated), Figure)
    fig = plots.plot_spectra(ds_truncated)
    # 2 source panels + averaged + colorbar + 3 excess panels
    assert len(fig.axes) == 7
    assert fig.axes[0].xaxis_inverted()


def test_plot_all_writes_files(ds, tmp_path):
    paths = plots.plot_all(ds, tmp_path / "out")
    assert sorted(p.name for p in paths) == [
        "crps_tp.png",
        "mae_tp.png",
        "rank_histogram_tp.png",
        "spectra_tp.png",
    ]
    assert all(p.stat().st_size > 0 for p in paths)


def test_plot_all_skips_missing_variables(ds, tmp_path):
    paths = plots.plot_all(ds[["mae.tp"]], tmp_path, fmt="svg")
    assert [p.name for p in paths] == ["mae_tp.svg"]


def test_plots_do_not_import_torch():
    code = (
        "import sys, mlcast_verification_tools.plots, mlcast_verification_tools.cli; "
        "sys.exit('torch' in sys.modules)"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
