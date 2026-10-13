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

import re

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

# short name of the verified physical variable (precipitation rate); the default where no
# source is involved
VARIABLE = "tp"

# short name of the score variables per CF standard name of the source variable. Scores
# are in the native units of the source (no unit conversion).
SHORT_NAMES: dict[str, str] = {
    "rainfall_rate": "tp",
    "rainfall_flux": "tp",
    "equivalent_reflectivity_factor": "dbz",
}

# observed and forecast rain rates below this value count as zero (tied) when ranking
# (IRENE paper, Sect. 4: 0.037 mm/h); see `units.default_zero_threshold` for other units
RANK_ZERO_THRESHOLD_MM_H = 0.037

MAE = "mae"
CRPS = "crps"
RANK_HISTOGRAM = "rank_histogram"
PSD = "psd"
PSD_OBS = "psd_obs"


def psd_units(units: str) -> str:
    """
    Units of the radial PSD of a field in `units`: the units squared, times km.

    Terms like ``kg m-2 s-1`` get their exponents doubled (``kg2 m-4 s-2 km``,
    ``mm h-1`` -> ``mm2 h-2 km``, ``dBZ`` -> ``dBZ2 km``); other unit strings are
    parenthesised (``(mm/h)2 km``).
    """
    terms = []
    for term in units.split():
        match = re.fullmatch(r"([A-Za-z]+)(-?\d+)?", term)
        if match is None:
            return f"({units})2 km"
        symbol, exponent = match.groups()
        terms.append(f"{symbol}{2 * int(exponent or 1)}")
    return " ".join([*terms, "km"]) if terms else "km"


def metric_attrs(units: str) -> dict[str, dict[str, str]]:
    """
    Data variable attributes per metric, for a verified variable in `units`.

    `metric_polarity` is only given for metrics that are not in the spec's list of
    known metrics (§4.2).

    Parameters
    ----------
    units : str
        Units of the verified variable, e.g. ``"mm h-1"`` or ``"dBZ"``.

    Returns
    -------
    dict
        ``{metric: attrs}`` for every metric of this package.
    """
    return {
        MAE: {
            "units": units,
            "long_name": "mean absolute error of the ensemble mean",
        },
        CRPS: {
            "units": units,
            "long_name": "continuous ranked probability score",
        },
        RANK_HISTOGRAM: {
            "units": "1",
            "long_name": "relative frequency of the observation's rank among the ensemble members",
            "metric_polarity": "neutral",
        },
        PSD: {
            "units": psd_units(units),
            "long_name": (
                "radially averaged power spectral density of the forecast (mean over members)"
            ),
            "metric_polarity": "activity",
        },
        PSD_OBS: {
            "units": psd_units(units),
            "long_name": "radially averaged power spectral density of the observation",
            "metric_polarity": "activity",
        },
    }


# attributes for a rain rate in mm/h (the default of `ScoreAccumulator`)
METRIC_ATTRS: dict[str, dict[str, str]] = metric_attrs("mm h-1")

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
