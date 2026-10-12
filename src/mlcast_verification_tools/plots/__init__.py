"""Verification plots made from a score dataset in the package's spec layout.

All functions take the stored score dataset (one value per forecast case) and do the
aggregation over cases themselves. Figures reproduce the verification figures of the
IRENE paper (Figs. 4-7).
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import xarray as xr

from .. import conventions as cv
from ._style import case_mean, ensemble_sizes, lead_time_minutes
from .lead_time import plot_score_vs_lead_time
from .rank_histogram import plot_rank_histograms, plot_rank_rmsd, rank_histogram_rmsd
from .spectra import excess_power_db, plot_excess_power, plot_psd, plot_spectra

__all__ = [
    "case_mean",
    "ensemble_sizes",
    "excess_power_db",
    "lead_time_minutes",
    "plot_all",
    "plot_excess_power",
    "plot_psd",
    "plot_rank_histograms",
    "plot_rank_rmsd",
    "plot_score_vs_lead_time",
    "plot_spectra",
    "rank_histogram_rmsd",
]

logger = logging.getLogger(__name__)


def plot_all(ds: xr.Dataset, out_dir, fmt: str = "png", dpi: int = 150) -> list[Path]:
    """
    Save every figure the score dataset has variables for.

    Produces, when the variables are present: one score-vs-lead-time figure per
    ``{metric}.{variable}`` that only has case dims (e.g. ``mae_tp``, ``crps_tp``),
    ``rank_histogram_{variable}`` (needs ``rank_histogram.*``) and
    ``spectra_{variable}`` (needs both ``psd.*`` and ``psd_obs.*``).

    Parameters
    ----------
    ds : xr.Dataset
        Score dataset in the spec layout.
    out_dir : str or Path
        Output directory (created if missing).
    fmt : str
        Image format / file suffix passed to `Figure.savefig`.
    dpi : int
        Resolution for raster formats.

    Returns
    -------
    list of Path
        Paths of the written files.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def save(fig, stem: str):
        path = out_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=dpi, format=fmt)
        plt.close(fig)
        written.append(path)
        logger.info("wrote %s", path)

    names = [str(n) for n in ds.data_vars]
    for name in names:
        metric, variable = cv.split_var_name(name)
        if not metric or set(ds[name].dims) != set(cv.CASE_DIMS):
            continue
        save(plot_score_vs_lead_time(ds, metric, variable), f"{metric}_{variable}")

    for name in names:
        metric, variable = cv.split_var_name(name)
        if metric == cv.RANK_HISTOGRAM:
            save(plot_rank_histograms(ds, variable=variable), f"rank_histogram_{variable}")
        elif metric == cv.PSD and cv.var_name(cv.PSD_OBS, variable) in ds:
            save(plot_spectra(ds, variable=variable), f"spectra_{variable}")
    return written
