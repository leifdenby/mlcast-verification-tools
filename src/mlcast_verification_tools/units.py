"""Units of the verified variable (no torch needed).

Scores are computed in the native units of the source; units only matter for the
rank-histogram zero threshold, which is defined in mm/h for rain rates (IRENE paper)
and at mlcast's reflectivity floor for reflectivities.
"""

from __future__ import annotations

from . import conventions as cv

_DBZ_UNITS = ("dbz",)

_UNITS_FACTORS = {
    # rates per hour: already mm/h (1 kg m-2 of water = 1 mm)
    **dict.fromkeys(
        (
            "mm/h",
            "mmh-1",
            "mmh^-1",
            "mm/hr",
            "mmhr-1",
            "kgm-2h-1",
            "kgm^-2h^-1",
            "kg/m2/h",
            "kgm-2hr-1",
        ),
        1.0,
    ),
    # rates per second
    **dict.fromkeys(("kgm-2s-1", "kgm^-2s^-1", "kg/m2/s", "mm/s", "mms-1", "mms^-1"), 3600.0),
}


def mm_per_hour_factor(units: str | None) -> float:
    """
    Factor converting values in `units` to mm/h.

    Parameters
    ----------
    units : str or None
        CF ``units`` attribute of a rain-rate variable, e.g. ``"mm/h"``,
        ``"kg m-2 h-1"`` or ``"kg m-2 s-1"``. Case and whitespace are ignored.

    Returns
    -------
    float
        Multiply values by this factor to get mm/h.

    Raises
    ------
    ValueError
        If the units are missing or not recognised.
    """
    key = (units or "").strip().lower().replace(" ", "").replace("**", "^")
    if key not in _UNITS_FACTORS:
        raise ValueError(f"cannot convert units {units!r} to mm/h")
    return _UNITS_FACTORS[key]


def default_zero_threshold(units: str | None, dbz_floor: float = 0.0) -> float:
    """
    Rank-histogram zero threshold for a variable in `units`.

    Values below the threshold count as zero (tied, below all others) when ranking.

    Parameters
    ----------
    units : str or None
        CF ``units`` attribute of the verified variable.
    dbz_floor : float
        Reflectivity (dBZ) that corresponds to zero rain, used for reflectivities;
        mlcast's ``ReflectivityScaling.dbz_floor`` (0 dBZ by default).

    Returns
    -------
    float
        0.037 mm/h (IRENE paper) expressed in `units` for rain rates, `dbz_floor` for
        reflectivities in dBZ.

    Raises
    ------
    ValueError
        If no default is known for `units`; pass ``zero_threshold`` explicitly then.
    """
    key = (units or "").strip().lower().replace(" ", "")
    if key in _DBZ_UNITS:
        return float(dbz_floor)
    try:
        factor = mm_per_hour_factor(units)
    except ValueError:
        raise ValueError(
            f"no default rank-histogram zero threshold for units {units!r}; "
            "pass `zero_threshold` explicitly"
        ) from None
    return cv.RANK_ZERO_THRESHOLD_MM_H / factor
