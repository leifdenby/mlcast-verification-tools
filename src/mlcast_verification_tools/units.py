"""Conversion of rain-rate units to mm/h (no torch needed)."""

from __future__ import annotations

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
        raise ValueError(
            f"cannot convert units {units!r} to mm/h; pass the conversion factor explicitly "
            "(`to_mm_per_hour=` / `--to-mm-per-hour`)"
        )
    return _UNITS_FACTORS[key]
