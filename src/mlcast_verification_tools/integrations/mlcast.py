"""Score an `mlcast <https://github.com/mlcast-community/mlcast>`_ model while it forecasts.

Forecasts are generated batch by batch on the GPU (or CPU), converted to mm/h and
passed straight to a :class:`~mlcast_verification_tools.accumulator.ScoreAccumulator`,
so no forecast is ever written to disk. Only the per-case scores are kept.

Needs the ``mlcast`` extra (torch, pytorch-lightning, mlcast). Importing this module
is cheap: the CLI helpers below have no heavy dependencies, and the torch/mlcast
API (:func:`denormalize_torch`, :func:`load_mlcast_model`,
:class:`VerificationDataset`, :func:`verify_mlcast_model`, ...) is imported on
first access from :mod:`._mlcast_core`.

Pipeline
--------
1. :func:`load_mlcast_model` restores a ``NowcastLightningModule`` from a Lightning
   checkpoint. The network is not part of the checkpoint's hyper-parameters (see
   below), so it is rebuilt from the run's Fiddle ``config.yaml`` or passed in.
2. :class:`VerificationDataset` reads full-domain (or fixed-region) cases from an
   mlcast source-data Zarr, normalised exactly as mlcast's training datasets do.
3. :func:`verify_mlcast_model` runs the model, clamps to the normalised range (as
   ``NowcastLightningModule.shared_step`` does), de-normalises on the device with
   :func:`denormalize_torch` and accumulates the scores.

Units
-----
mlcast applies its Marshall-Palmer Z-R scaling directly to the stored values,
whatever ``standard_name`` they carry: ``rainfall_flux`` reuses the mm/h
relation unchanged (mlcast warns about this). The model therefore predicts in the
*units of the source variable*, and :func:`denormalize_torch` returns values in
those units, like mlcast's ``DENORMALIZATION_REGISTRY``. Both forecasts and
observations are then multiplied by a factor derived from the variable's
``units`` attribute (:func:`mm_per_hour_factor`) to get mm/h: 1 for ``mm/h`` and
``kg m-2 h-1`` (e.g. IT-DPC-SRI, ``rainfall_flux`` in ``kg m-2 h-1``), 3600 for
``kg m-2 s-1`` / ``mm/s``.

Checkpoints
-----------
``NowcastLightningModule.__init__`` only saves ``ensemble_size``, ``loss_class``,
``loss_params`` and ``masked_loss`` as hyper-parameters, not ``network``, so
mlcast's ``NowcastLightningModule.from_checkpoint`` cannot restore a checkpoint
on its own (``__init__`` is called without its required ``network`` argument).
:func:`load_mlcast_model` passes the network explicitly to
``load_from_checkpoint``. Fiddle (0.3) can dump its configs to YAML but not load
them back, so :func:`load_fiddle_yaml` implements the inverse of
``fiddle.experimental.yaml_serialization.dump_yaml`` (plus mlcast's ``!type``
tag) and returns a ``fdl.Config`` graph.
"""

from __future__ import annotations

import argparse
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - for type checkers/IDEs; imported lazily at runtime
    from ._mlcast_core import (  # re-exported lazily via __getattr__
        VerificationDataset,
        build_network_from_config_yaml,
        denormalize_torch,
        init_times_from_range,
        load_fiddle_yaml,
        load_mlcast_model,
        mm_per_hour_factor,
        run_mlcast_from_args,
        verify_mlcast_model,
    )

#: CF standard names of rain-rate-like variables mlcast's Z-R scaling is applied to
#: and that can be scored as mm/h.
SUPPORTED_STANDARD_NAMES = ("rainfall_rate", "rainfall_flux")

_LAZY = (
    "VerificationDataset",
    "build_network_from_config_yaml",
    "denormalize_torch",
    "init_times_from_range",
    "load_fiddle_yaml",
    "load_mlcast_model",
    "mm_per_hour_factor",
    "run_mlcast_from_args",
    "verify_mlcast_model",
)

__all__ = [  # noqa: RUF022 - grouped: light helpers first, then the lazily imported API
    "SUPPORTED_STANDARD_NAMES",
    "add_run_mlcast_arguments",
    "VerificationDataset",
    "build_network_from_config_yaml",
    "denormalize_torch",
    "init_times_from_range",
    "load_fiddle_yaml",
    "load_mlcast_model",
    "mm_per_hour_factor",
    "run_mlcast_from_args",
    "verify_mlcast_model",
]


def __getattr__(name: str) -> Any:
    """Import the torch/mlcast API on first access (PEP 562)."""
    if name in _LAZY:
        from . import _mlcast_core

        value = getattr(_mlcast_core, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


def _parse_slice(text: str) -> slice:
    try:
        start, stop = text.split(":")
        return slice(int(start) if start else None, int(stop) if stop else None)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an index range 'start:stop', got {text!r}") from exc


def add_run_mlcast_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add the arguments of ``mlcast.verify run-mlcast`` to `parser`.

    Imports nothing beyond the standard library, so building the CLI stays cheap.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser (sub-command) to add the arguments to.
    """
    g = parser.add_argument_group("model")
    g.add_argument("--checkpoint", required=True, help="Lightning checkpoint (.ckpt) of the mlcast run")
    g.add_argument(
        "--config-yaml",
        help="the run's Fiddle config.yaml (default: config.yaml next to the checkpoint or one level up)",
    )
    g.add_argument("--ensemble-size", type=int, default=20, help="number of members (default: 20)")
    g.add_argument("--member-chunk", type=int, help="generate members in calls of at most this many (>= 2)")
    g.add_argument("--device", help="torch device (default: cuda if available, else cpu)")
    g.add_argument("--seed", type=int, default=0)

    g = parser.add_argument_group("data")
    g.add_argument("--source", required=True, help="source-data Zarr store (path or URL)")
    g.add_argument(
        "--storage-options", type=json.loads, help="JSON dict for xr.open_zarr, e.g. '{\"anon\": true}'"
    )
    g.add_argument("--variable", help="data variable (default: the one with a rain-rate standard_name)")
    g.add_argument("--standard-name", choices=SUPPORTED_STANDARD_NAMES)
    g.add_argument("--start", required=True, help="first init time (ISO 8601)")
    g.add_argument("--end", required=True, help="last init time (ISO 8601, inclusive)")
    g.add_argument("--freq", required=True, help="init time spacing, e.g. PT6H or 6h")
    g.add_argument("--input-steps", type=int, default=6)
    g.add_argument("--forecast-steps", type=int, default=12)
    g.add_argument("--time-step", help="frame spacing (default: most common spacing in the source)")
    g.add_argument("--region-y", type=_parse_slice, help="y index range start:stop")
    g.add_argument("--region-x", type=_parse_slice, help="x index range start:stop")
    g.add_argument("--to-mm-per-hour", type=float, help="factor to mm/h (default: from units attr)")
    g.add_argument("--on-missing", choices=("skip", "raise"), default="skip")
    g.add_argument("--batch-size", type=int, default=1)
    g.add_argument("--num-workers", type=int, default=0)

    g = parser.add_argument_group("output")
    g.add_argument("--forecast-source", required=True, help="name of the forecast source")
    g.add_argument("--out", required=True, help="output score file (.nc) or Zarr store")
    g.add_argument("--no-psd", action="store_true", help="skip the radial power spectra")
    g.add_argument("--no-progress", action="store_true")
    g.add_argument("--title", required=True)
    g.add_argument("--created-by", required=True, help="'Name <email>'")
    g.add_argument(
        "--created-with",
        required=True,
        help="GitHub URL@version of the score dataset repo, e.g. "
        "https://github.com/mlcast-community/mlcast-dataset-dmi-nowcast-scores@v0.1.0",
    )
    g.add_argument("--forecasts", help="description of the forecasts (default: generated)")
    g.add_argument("--truth", help="description of the truth (default: from the source attrs)")
    g.add_argument("--license", default="CC-BY-4.0")
    g.add_argument("--dataset-version", default="0.1.0")
