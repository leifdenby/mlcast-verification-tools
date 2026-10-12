"""Shared colours, styling and small data helpers for the verification plots.

Colour roles follow one system across all figures:

- each `forecast_source` gets a fixed categorical hue, assigned in the order the
  sources appear in the dataset (never by rank of the score),
- lead time is encoded with a single-hue (blue) sequential ramp, light = short,
  dark = long lead time,
- the observation is drawn in primary ink (near-black), reference lines in muted grey.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap

from .. import conventions as cv

# categorical slots (light mode), fixed order -- validated for CVD separation on
# adjacent pairs; slots beyond 8 are not generated, they wrap with a dashed style
SOURCE_COLORS = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#6250d6",  # violet
    "#e34948",  # red
]

# single-hue sequential ramp for lead time (ordinal steps 250 -> 700)
_LEAD_TIME_RAMP = [
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
]
LEAD_TIME_CMAP = LinearSegmentedColormap.from_list("lead_time_blue", _LEAD_TIME_RAMP)

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
FULL_FORECAST_GREY = "#d6d5ce"

RC_PARAMS = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_SECONDARY,
    "axes.titlecolor": INK_PRIMARY,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "xtick.labelcolor": INK_SECONDARY,
    "ytick.labelcolor": INK_SECONDARY,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "lines.linewidth": 2.0,
    "font.family": "sans-serif",
}


def style_context():
    """Return a matplotlib rc context manager with the package style."""
    return mpl.rc_context(RC_PARAMS)


def new_figure(*args, **kwargs):
    """`plt.subplots` with constrained layout by default."""
    kwargs.setdefault("layout", "constrained")
    return plt.subplots(*args, **kwargs)


def source_names(ds: xr.Dataset | xr.DataArray) -> list[str]:
    """Forecast source names in dataset order."""
    return [str(s) for s in ds[cv.FORECAST_SOURCE].values]


def source_color(ds: xr.Dataset | xr.DataArray, source: str) -> str:
    """Colour of `source`, fixed by its position in the dataset's `forecast_source`."""
    idx = source_names(ds).index(str(source))
    return SOURCE_COLORS[idx % len(SOURCE_COLORS)]


def source_linestyle(ds: xr.Dataset | xr.DataArray, source: str) -> str:
    """Solid for the first 8 sources, dashed for the next 8 (secondary encoding)."""
    idx = source_names(ds).index(str(source))
    return "-" if idx < len(SOURCE_COLORS) else "--"


def lead_time_minutes(lead_time) -> np.ndarray:
    """
    Convert lead time values to minutes.

    Parameters
    ----------
    lead_time : array-like or xr.DataArray
        timedelta64 values, or numbers that are taken to already be minutes.

    Returns
    -------
    np.ndarray of float
    """
    values = np.asarray(getattr(lead_time, "values", lead_time))
    if np.issubdtype(values.dtype, np.timedelta64):
        return values.astype("timedelta64[s]").astype(float) / 60.0
    return values.astype(float)


def with_minutes(da: xr.DataArray) -> xr.DataArray:
    """Replace the `lead_time` coordinate with float minutes."""
    if cv.LEAD_TIME not in da.dims:
        return da
    return da.assign_coords({cv.LEAD_TIME: lead_time_minutes(da[cv.LEAD_TIME])})


def case_mean(da: xr.DataArray) -> xr.DataArray:
    """Mean over forecast cases (`init_time`), skipping NaN."""
    return da.mean(cv.INIT_TIME, skipna=True, keep_attrs=True)


def ensemble_sizes(rank_hist: xr.DataArray) -> dict[str, int]:
    """
    Ensemble size K per forecast source from a NaN-padded rank histogram.

    K is the number of ranks that hold a finite value in at least one case, minus one.

    Parameters
    ----------
    rank_hist : xr.DataArray
        Rank histogram with dims including `forecast_source` and `rank`.

    Returns
    -------
    dict of str to int
    """
    other = [d for d in rank_hist.dims if d not in (cv.FORECAST_SOURCE, cv.RANK)]
    has_value = rank_hist.notnull().any(other) if other else rank_hist.notnull()
    n_ranks = has_value.sum(cv.RANK)
    return {str(s): int(n) - 1 for s, n in zip(n_ranks[cv.FORECAST_SOURCE].values, n_ranks.values)}


def lead_time_colors(lead_times_min) -> list:
    """Sequential colours for lead times (light = short, dark = long)."""
    lt = np.asarray(lead_times_min, dtype=float)
    if lt.size == 1:
        return [LEAD_TIME_CMAP(1.0)]
    frac = (lt - lt.min()) / (lt.max() - lt.min())
    return [LEAD_TIME_CMAP(f) for f in frac]


def lead_time_norm(lead_times_min):
    """Normalisation of the lead-time colormap over the given minutes."""
    lt = np.asarray(lead_times_min, dtype=float)
    return mpl.colors.Normalize(vmin=float(lt.min()), vmax=float(max(lt.max(), lt.min() + 1)))


def pick_lead_times(lead_times_min, n: int = 4) -> np.ndarray:
    """About `n` evenly spaced lead times (minutes) including the first and last."""
    lt = np.asarray(lead_times_min, dtype=float)
    if lt.size <= n:
        return lt
    idx = np.unique(np.round(np.linspace(0, lt.size - 1, n)).astype(int))
    return lt[idx]


def format_minutes(minutes: float) -> str:
    """Label like '30 min'."""
    return f"{minutes:g} min"


def axis_label(da: xr.DataArray, fallback: str) -> str:
    """Axis label from `long_name` and `units` attributes."""
    name = da.attrs.get("long_name", fallback)
    units = da.attrs.get("units")
    if units and units != "1":
        units = units.replace("mm h-1", "mm/h").replace("h-2", "h⁻²").replace("mm2", "mm²")
        return f"{name} [{units}]"
    return name


def require(ds: xr.Dataset, name: str) -> xr.DataArray:
    """Return `ds[name]` or raise a KeyError naming the missing variable."""
    if name not in ds:
        raise KeyError(f"score dataset has no variable {name!r} (has {list(ds.data_vars)})")
    return ds[name]
