"""Rank histograms and their deviation from uniformity (IRENE paper Fig. 6)."""

from __future__ import annotations

import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .. import conventions as cv
from . import _style as st

RANK_HIST_NAME = cv.var_name(cv.RANK_HISTOGRAM)


def _valid_rank_mask(rh: xr.DataArray) -> xr.DataArray:
    """Boolean (forecast_source, rank) mask of the ranks 1..K+1 of each source."""
    sizes = st.ensemble_sizes(rh)
    n_ranks = xr.DataArray(
        [sizes[s] + 1 for s in st.source_names(rh)],
        dims=cv.FORECAST_SOURCE,
        coords={cv.FORECAST_SOURCE: rh[cv.FORECAST_SOURCE]},
    )
    position = xr.DataArray(
        np.arange(rh.sizes[cv.RANK]), dims=cv.RANK, coords={cv.RANK: rh[cv.RANK]}
    )
    return position < n_ranks


def rank_histogram_rmsd(ds: xr.Dataset, variable: str = cv.VARIABLE) -> xr.DataArray:
    """
    Root-mean-square deviation of the case-averaged rank histogram from uniformity.

    ``RMSD(tau) = sqrt(mean_r (pbar_r(tau) - 1/(K+1))**2)`` where ``pbar_r(tau)`` is the
    rank frequency averaged over all forecast cases (`init_time`) at lead time ``tau``
    and the mean runs over the ``K+1`` ranks of each source (``K`` = ensemble size,
    detected from the non-NaN ranks).

    Parameters
    ----------
    ds : xr.Dataset
        Score dataset with ``rank_histogram.{variable}``.
    variable : str
        Verified variable short name.

    Returns
    -------
    xr.DataArray
        RMSD with dims (`forecast_source`, `lead_time`); NaN where a source has no data.
    """
    rh = st.require(ds, cv.var_name(cv.RANK_HISTOGRAM, variable))
    pbar = st.case_mean(rh)
    mask = _valid_rank_mask(rh)
    n_ranks = mask.sum(cv.RANK)
    dev2 = ((pbar - 1.0 / n_ranks) ** 2).where(mask)
    rmsd = np.sqrt(dev2.mean(cv.RANK, skipna=True))
    # lead times where the source has no histogram at all stay NaN
    rmsd = rmsd.where(pbar.where(mask).notnull().any(cv.RANK))
    rmsd.name = "rank_histogram_rmsd"
    rmsd.attrs = {"long_name": "RMSD of rank histogram from uniformity", "units": "1"}
    return rmsd


def _resolve_lead_times(lead_times, available_min: np.ndarray) -> np.ndarray:
    """Lead times (minutes) to plot: nearest available values to the request."""
    if lead_times is None:
        return st.pick_lead_times(available_min, 4)
    req = st.lead_time_minutes(np.atleast_1d(np.asarray(lead_times)))
    idx = [int(np.argmin(np.abs(available_min - r))) for r in req]
    return available_min[sorted(set(idx))]


def plot_rank_rmsd(ds: xr.Dataset, ax: Axes | None = None, variable: str = cv.VARIABLE) -> Figure:
    """
    Plot the rank-histogram RMSD from uniformity against lead time, one line per source.

    Parameters
    ----------
    ds : xr.Dataset
        Score dataset with ``rank_histogram.{variable}``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure is made if not given.
    variable : str
        Verified variable short name.

    Returns
    -------
    matplotlib.figure.Figure
    """
    rmsd = st.with_minutes(rank_histogram_rmsd(ds, variable))
    with st.style_context():
        if ax is None:
            fig, ax = st.new_figure(figsize=(5.0, 3.8))
        else:
            fig = ax.figure
        for src in st.source_names(ds):
            y = rmsd.sel({cv.FORECAST_SOURCE: src})
            ax.plot(
                y[cv.LEAD_TIME].values,
                y.values,
                color=st.source_color(ds, src),
                linestyle=st.source_linestyle(ds, src),
                marker="o",
                markersize=4,
                label=src,
            )
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("lead time [min]")
        ax.set_ylabel("RMSD from uniform")
        ax.set_title("Deviation from uniformity", loc="left")
        ax.legend()
    return fig


def _plot_source_panel(ax: Axes, ds: xr.Dataset, rh: xr.DataArray, src: str, lts_min, k: int):
    """Rank histogram panel of one forecast source."""
    rh_src = st.with_minutes(rh.sel({cv.FORECAST_SOURCE: src}))
    ranks = rh_src[cv.RANK].values[: k + 1]
    full = rh_src.mean([cv.INIT_TIME, cv.LEAD_TIME], skipna=True).values[: k + 1]
    pbar = st.case_mean(rh_src)

    ax.bar(ranks, full, width=0.8, color=st.FULL_FORECAST_GREY, label="full forecast", zorder=1)
    colors = st.lead_time_colors(lts_min) if len(lts_min) > 1 else [st.LEAD_TIME_CMAP(0.7)]
    all_lt = pbar[cv.LEAD_TIME].values
    for lt, c in zip(lts_min, colors, strict=False):
        y = pbar.sel({cv.LEAD_TIME: all_lt[np.argmin(np.abs(all_lt - lt))]}).values[: k + 1]
        if np.all(np.isnan(y)):
            continue
        ax.plot(
            ranks, y, color=c, marker="o", markersize=3.5, label=st.format_minutes(lt), zorder=3
        )
    ax.axhline(
        1.0 / (k + 1),
        color=st.INK_SECONDARY,
        linestyle="--",
        linewidth=1.2,
        zorder=2,
        label="uniform 1/(K+1)",
    )
    ax.set_xlabel("rank")
    ax.set_xlim(ranks[0] - 0.6, ranks[-1] + 0.6)
    ax.set_xticks(ranks if len(ranks) <= 12 else ranks[:: int(np.ceil(len(ranks) / 12))])
    ax.set_ylim(bottom=0)
    ax.grid(axis="x", visible=False)
    ax.set_title(f"{src} (K = {k})", loc="left", color=st.INK_PRIMARY)
    # identity cue: the source's colour on the baseline, matching the RMSD panel
    ax.spines["bottom"].set_color(st.source_color(ds, src))
    ax.spines["bottom"].set_linewidth(2.0)


def plot_rank_histograms(
    ds: xr.Dataset,
    lead_times=None,
    include_rmsd: bool = True,
    variable: str = cv.VARIABLE,
) -> Figure:
    """
    Plot rank histograms per forecast source, optionally with the RMSD panel.

    Each source panel shows the case-averaged rank frequency at selected lead times
    (sequential colours), the frequency over the full forecast (mean over all cases and
    lead times, grey columns) and the uniform frequency ``1/(K+1)`` (dashed).

    Parameters
    ----------
    ds : xr.Dataset
        Score dataset with ``rank_histogram.{variable}``.
    lead_times : sequence, optional
        Lead times to draw, as timedelta64 values or minutes; the nearest available
        lead times are used. Defaults to four evenly spaced lead times.
    include_rmsd : bool
        Add a final panel with the RMSD from uniformity vs lead time.
    variable : str
        Verified variable short name.

    Returns
    -------
    matplotlib.figure.Figure
    """
    rh = st.require(ds, cv.var_name(cv.RANK_HISTOGRAM, variable))
    sources = st.source_names(ds)
    sizes = st.ensemble_sizes(rh)
    lts_min = _resolve_lead_times(lead_times, st.lead_time_minutes(rh[cv.LEAD_TIME]))

    n_panels = len(sources) + int(include_rmsd)
    with st.style_context():
        fig, axes = st.new_figure(1, n_panels, figsize=(3.6 * n_panels + 0.4, 3.9), squeeze=False)
        axes = axes[0]
        for ax, src in zip(axes, sources, strict=False):
            _plot_source_panel(ax, ds, rh, src, lts_min, sizes[src])
        axes[0].set_ylabel("rank frequency")
        # one shared legend for the lead-time / reference entries, below the panels so
        # it never hides the histograms
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, fontsize=8, loc="outside lower center", ncols=len(labels))
        if include_rmsd:
            plot_rank_rmsd(ds, ax=axes[-1], variable=variable)
        fig.suptitle("Rank histograms", x=0.01, ha="left", fontsize=12, color=st.INK_PRIMARY)
    return fig
