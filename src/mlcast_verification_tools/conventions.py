"""Names, units and attributes shared by every module that writes or reads score datasets.

The score dataset follows the MLCast `verification/forecast_scoring_metrics` spec
(v0.1.0): one data variable per `{metric}.{variable}`. A forecast *case* is one
initialisation time (`init_time`) over a fixed verification domain; each value is the
score of one forecast source, case and lead time, i.e. reduced over the domain's valid
pixels (and members) but not over cases. Required dims are
`forecast_source`, `init_time` and `lead_time`; this package adds `rank` (rank
histograms) and `wavelength` (radial power spectra).
"""

from __future__ import annotations

FORECAST_SOURCE = "forecast_source"
INIT_TIME = "init_time"
LEAD_TIME = "lead_time"
RANK = "rank"
WAVELENGTH = "wavelength"

CASE_DIMS = (FORECAST_SOURCE, INIT_TIME, LEAD_TIME)

# Dims of the *input* datasets of `offline.compute_scores` (see the `offline` module):
# forecasts `(forecast_source, init_time, lead_time, member, y, x)` and truth
# `(time, y, x)`, with `x`/`y` projection coordinates on the same grid.
MEMBER = "member"
TIME = "time"
X = "x"
Y = "y"
FORECAST_DIMS = (FORECAST_SOURCE, INIT_TIME, LEAD_TIME, MEMBER, Y, X)
TRUTH_DIMS = (TIME, Y, X)

# short name of the verified physical variable (precipitation rate)
VARIABLE = "tp"

# observed and forecast values below this rain rate are set to zero before ranking
# (IRENE paper, Sect. 4: 0.037 mm/h)
RANK_ZERO_THRESHOLD_MM_H = 0.037

MAE = "mae"
CRPS = "crps"
RANK_HISTOGRAM = "rank_histogram"
PSD = "psd"
PSD_OBS = "psd_obs"

# Data variable attributes per metric. `metric_polarity` is only given for metrics
# that are not in the spec's list of known metrics (§4.2).
METRIC_ATTRS: dict[str, dict[str, str]] = {
    MAE: {
        "units": "mm h-1",
        "long_name": "mean absolute error of the ensemble mean",
    },
    CRPS: {
        "units": "mm h-1",
        "long_name": "continuous ranked probability score",
    },
    RANK_HISTOGRAM: {
        "units": "1",
        "long_name": "relative frequency of the observation's rank among the ensemble members",
        "metric_polarity": "neutral",
    },
    PSD: {
        "units": "mm2 h-2 km",
        "long_name": "radially averaged power spectral density of the forecast (mean over members)",
        "metric_polarity": "activity",
    },
    PSD_OBS: {
        "units": "mm2 h-2 km",
        "long_name": "radially averaged power spectral density of the observation",
        "metric_polarity": "activity",
    },
}

COORD_ATTRS: dict[str, dict[str, str]] = {
    RANK: {"long_name": "rank of the observation among the ensemble members", "units": "1"},
    WAVELENGTH: {"long_name": "spatial wavelength", "units": "km"},
}


def var_name(metric: str, variable: str = VARIABLE) -> str:
    """Data variable name `{metric}.{variable}` (spec §4.1)."""
    return f"{metric}.{variable}"


def split_var_name(name: str) -> tuple[str, str]:
    """Split a data variable name into (metric, variable) on the last `.`."""
    metric, _, variable = name.rpartition(".")
    return metric, variable
