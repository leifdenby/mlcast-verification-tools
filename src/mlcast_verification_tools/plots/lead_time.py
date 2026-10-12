"""Case-averaged scores as a function of lead time (IRENE paper Figs. 4 and 5)."""

from __future__ import annotations

import textwrap

import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .. import conventions as cv
from . import _style as st


def plot_score_vs_lead_time(
    ds: xr.Dataset,
    metric: str,
    variable: str = cv.VARIABLE,
    ax: Axes | None = None,
) -> Figure:
    """
    Plot the case-mean of a score against lead time, one line per forecast source.

    Parameters
    ----------
    ds : xr.Dataset
        Score dataset in the spec layout.
    metric : str
        Metric name, e.g. ``"mae"`` or ``"crps"``. The variable ``{metric}.{variable}``
        must only have the case dims (`forecast_source`, `init_time`, `lead_time`).
    variable : str
        Verified variable short name.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure is made if not given.

    Returns
    -------
    matplotlib.figure.Figure
        The figure holding the axes. Lines stop where a source has no (NaN) scores.
    """
    name = cv.var_name(metric, variable)
    da = st.require(ds, name)
    extra = set(da.dims) - set(cv.CASE_DIMS)
    if extra:
        raise ValueError(f"{name} has non-case dims {sorted(extra)}; cannot plot vs lead time")
    mean = st.with_minutes(st.case_mean(da))

    with st.style_context():
        if ax is None:
            fig, ax = st.new_figure(figsize=(6.0, 4.0))
        else:
            fig = ax.figure
        for src in st.source_names(ds):
            y = mean.sel({cv.FORECAST_SOURCE: src})
            ax.plot(
                y[cv.LEAD_TIME].values,
                y.values,
                color=st.source_color(ds, src),
                linestyle=st.source_linestyle(ds, src),
                marker="o",
                markersize=4,
                label=src,
            )
        lt = mean[cv.LEAD_TIME].values
        ax.set_xlim(0, float(np.nanmax(lt)) * 1.03 if lt.size else 1)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("lead time [min]")
        ax.set_ylabel("\n".join(textwrap.wrap(st.axis_label(da, metric.upper()), 40)))
        ax.set_title(f"{metric.upper()} ({variable})", loc="left")
        ax.legend(title="forecast source", title_fontsize=9)
    return fig
