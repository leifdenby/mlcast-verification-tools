"""Radially averaged power spectra and excess power (IRENE paper Fig. 7)."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure

from .. import conventions as cv
from . import _style as st

OBS_LABEL = "observation"
OBS_LINEWIDTH = 3.0


def excess_power_db(
    ds: xr.Dataset, scales_km=(74, 16, 4), variable: str = cv.VARIABLE
) -> xr.DataArray:
    """
    Excess forecast power over the observation, in dB, at selected spatial scales.

    ``excess(tau, lambda) = 10 log10(mean_cases psd / mean_cases psd_obs)`` evaluated at
    the wavelength bin nearest to each requested scale.

    Parameters
    ----------
    ds : xr.Dataset
        Score dataset with ``psd.{variable}`` and ``psd_obs.{variable}``.
    scales_km : sequence of float
        Requested wavelengths in km.
    variable : str
        Verified variable short name.

    Returns
    -------
    xr.DataArray
        Dims (`forecast_source`, `lead_time`, `wavelength`); the `wavelength` coordinate
        holds the actual bin values used and the `target_wavelength` coordinate the
        requested scales.
    """
    psd = st.case_mean(st.require(ds, cv.var_name(cv.PSD, variable)))
    psd_obs = st.case_mean(st.require(ds, cv.var_name(cv.PSD_OBS, variable)))
    wl = psd[cv.WAVELENGTH].values
    idx = [int(np.argmin(np.abs(wl - s))) for s in scales_km]
    with np.errstate(divide="ignore", invalid="ignore"):
        excess = 10.0 * np.log10(psd / psd_obs)
    excess = excess.isel({cv.WAVELENGTH: idx})
    excess = excess.assign_coords(
        target_wavelength=(cv.WAVELENGTH, np.asarray(scales_km, dtype=float))
    )
    excess.name = "excess_power"
    excess.attrs = {"long_name": "excess power of forecast over observation", "units": "dB"}
    return excess.transpose(cv.FORECAST_SOURCE, cv.LEAD_TIME, cv.WAVELENGTH)


def _default_psd_lead_times(available_min: np.ndarray) -> np.ndarray:
    if available_min.size <= 12:
        return available_min
    return st.pick_lead_times(available_min, 8)


def _resolve_lead_times(lead_times, available_min: np.ndarray) -> np.ndarray:
    if lead_times is None:
        return _default_psd_lead_times(available_min)
    req = st.lead_time_minutes(np.atleast_1d(np.asarray(lead_times)))
    idx = [int(np.argmin(np.abs(available_min - r))) for r in req]
    return available_min[sorted(set(idx))]


def _format_loglog(ax: Axes):
    ax.set_xscale("log")
    ax.set_yscale("log")
    if not ax.xaxis_inverted():
        ax.invert_xaxis()
    ax.set_xlabel("wavelength [km]")
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.grid(which="minor", visible=False)


def _plot_obs(ax: Axes, psd_obs: xr.DataArray):
    obs = psd_obs.mean([d for d in psd_obs.dims if d != cv.WAVELENGTH], skipna=True)
    ax.plot(
        obs[cv.WAVELENGTH].values,
        obs.values,
        color=st.INK_PRIMARY,
        linewidth=OBS_LINEWIDTH,
        label=OBS_LABEL,
        zorder=4,
    )


def _draw_psd_panels(axes, ds: xr.Dataset, lead_times, variable: str, average_panel: bool = True):
    """Draw per-source PSD panels (+ averaged panel) into `axes`; return the norm used."""
    psd = st.with_minutes(st.require(ds, cv.var_name(cv.PSD, variable)))
    psd_obs = st.with_minutes(st.require(ds, cv.var_name(cv.PSD_OBS, variable)))
    all_lt = psd[cv.LEAD_TIME].values
    lts = _resolve_lead_times(lead_times, all_lt)
    norm = st.lead_time_norm(lts)
    sources = st.source_names(ds)
    psd_mean = st.case_mean(psd)
    wl = psd[cv.WAVELENGTH].values

    for ax, src in zip(axes, sources, strict=False):
        p = psd_mean.sel({cv.FORECAST_SOURCE: src})
        for lt in lts:
            y = p.sel({cv.LEAD_TIME: lt}).values
            if np.all(np.isnan(y)):
                continue
            ax.plot(wl, y, color=st.LEAD_TIME_CMAP(norm(lt)), linewidth=1.4, zorder=3)
        _plot_obs(ax, psd_obs.sel({cv.FORECAST_SOURCE: src}))
        _format_loglog(ax)
        ax.set_title(src, loc="left")
        ax.spines["bottom"].set_color(st.source_color(ds, src))
        ax.spines["bottom"].set_linewidth(2.0)

    if average_panel:
        ax = axes[len(sources)]
        for src in sources:
            y = psd.sel({cv.FORECAST_SOURCE: src}).mean([cv.INIT_TIME, cv.LEAD_TIME], skipna=True)
            ax.plot(
                wl,
                y.values,
                color=st.source_color(ds, src),
                linestyle=st.source_linestyle(ds, src),
                label=src,
                zorder=3,
            )
        _plot_obs(ax, psd_obs)
        _format_loglog(ax)
        ax.set_title("lead-time averaged", loc="left")
        ax.legend(loc="lower left")
    axes[0].set_ylabel(st.axis_label(xr.DataArray(attrs={"units": psd.attrs.get("units")}), "PSD"))
    axes[0].legend(handles=axes[0].get_lines()[-1:], loc="lower left")
    for ax in axes[1:]:
        ax.sharey(axes[0])
    return norm


def _add_lead_time_colorbar(fig, axes, norm):
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=st.LEAD_TIME_CMAP)
    cb = fig.colorbar(sm, ax=axes, shrink=0.85, aspect=25, pad=0.01)
    cb.set_label("lead time [min]")
    cb.outline.set_visible(False)
    return cb


def plot_psd(ds: xr.Dataset, lead_times=None, variable: str = cv.VARIABLE) -> Figure:
    """
    Plot case-averaged radial power spectra per forecast source, coloured by lead time.

    Each source panel shows the forecast PSD at the selected lead times and the mean
    observed PSD (thick black). A last panel compares the lead-time-averaged PSD of all
    sources with the observation. The wavelength axis is inverted (large scales left).

    Parameters
    ----------
    ds : xr.Dataset
        Score dataset with ``psd.{variable}`` and ``psd_obs.{variable}``.
    lead_times : sequence, optional
        Lead times (timedelta64 or minutes) to draw; nearest available are used.
        Defaults to all lead times (or 8 evenly spaced ones if there are more than 12).
    variable : str
        Verified variable short name.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n = len(st.source_names(ds)) + 1
    with st.style_context():
        fig, axes = st.new_figure(1, n, figsize=(3.4 * n + 1.0, 3.8), squeeze=False)
        norm = _draw_psd_panels(list(axes[0]), ds, lead_times, variable)
        _add_lead_time_colorbar(fig, list(axes[0][:-1]), norm)
        fig.suptitle("Radially averaged power spectra", x=0.01, ha="left", fontsize=12)
    return fig


def _draw_excess_panels(axes, ds: xr.Dataset, scales_km, variable: str):
    excess = st.with_minutes(excess_power_db(ds, scales_km, variable))
    for ax, j in zip(axes, range(excess.sizes[cv.WAVELENGTH]), strict=False):
        e = excess.isel({cv.WAVELENGTH: j})
        ax.axhline(0.0, color=st.INK_SECONDARY, linewidth=1.0, zorder=1)
        for src in st.source_names(ds):
            y = e.sel({cv.FORECAST_SOURCE: src})
            ax.plot(
                y[cv.LEAD_TIME].values,
                y.values,
                color=st.source_color(ds, src),
                linestyle=st.source_linestyle(ds, src),
                marker="o",
                markersize=4,
                label=src,
                zorder=3,
            )
        ax.set_title(f"λ ≈ {float(e[cv.WAVELENGTH]):.3g} km", loc="left")
        ax.set_xlabel("lead time [min]")
        ax.set_xlim(left=0)
    axes[0].set_ylabel("excess power [dB]")
    axes[0].legend()
    return excess


def plot_excess_power(ds: xr.Dataset, scales_km=(74, 16, 4), variable: str = cv.VARIABLE) -> Figure:
    """
    Plot excess power (dB) vs lead time, one panel per spatial scale.

    Parameters
    ----------
    ds : xr.Dataset
        Score dataset with ``psd.{variable}`` and ``psd_obs.{variable}``.
    scales_km : sequence of float
        Requested wavelengths in km; the nearest bin is used and named in the title.
    variable : str
        Verified variable short name.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n = len(scales_km)
    with st.style_context():
        fig, axes = st.new_figure(1, n, figsize=(3.6 * n, 3.4), squeeze=False, sharey=True)
        _draw_excess_panels(list(axes[0]), ds, scales_km, variable)
        fig.suptitle("Excess power relative to observation", x=0.01, ha="left", fontsize=12)
    return fig


def plot_spectra(
    ds: xr.Dataset, lead_times=None, scales_km=(74, 16, 4), variable: str = cv.VARIABLE
) -> Figure:
    """
    Combined spectra figure like IRENE Fig. 7: PSD panels on top, excess power below.

    Parameters
    ----------
    ds : xr.Dataset
        Score dataset with ``psd.{variable}`` and ``psd_obs.{variable}``.
    lead_times : sequence, optional
        Lead times for the per-source PSD panels (see `plot_psd`).
    scales_km : sequence of float
        Scales for the excess-power panels (see `plot_excess_power`).
    variable : str
        Verified variable short name.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_top = len(st.source_names(ds)) + 1
    n_bottom = len(scales_km)
    width = max(3.4 * n_top + 1.0, 3.6 * n_bottom)
    with st.style_context():
        fig = plt.figure(figsize=(width, 7.6), layout="constrained")
        top, bottom = fig.subfigures(2, 1, height_ratios=[1.1, 1.0], hspace=0.04)
        axes_top = top.subplots(1, n_top, squeeze=False)[0]
        norm = _draw_psd_panels(list(axes_top), ds, lead_times, variable)
        _add_lead_time_colorbar(top, list(axes_top[:-1]), norm)
        axes_bottom = bottom.subplots(1, n_bottom, squeeze=False, sharey=True)[0]
        _draw_excess_panels(list(axes_bottom), ds, scales_km, variable)
        _subtitle(top, "Radially averaged power spectra")
        _subtitle(bottom, "Excess power relative to observation")
    return fig


def _subtitle(subfig: SubFigure, text: str):
    subfig.suptitle(text, x=0.01, ha="left", fontsize=12)
