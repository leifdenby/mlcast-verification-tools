"""Forecast with and score trained `mlcast <https://github.com/mlcast-community/mlcast>`_ models.

The training config of the run is always an explicit input (an ``fdl.Config`` from
:func:`load_run_config`, or the path of the run's ``config.yaml``). Two paths:

- forecasts first, then scores::

      ds_forecasts = forecast_mlcast(config, checkpoint, ensemble_size=20, out="fc.zarr")
      ds_truth = open_truth(config)
      scores = compute_scores(ds_forecasts, ds_truth)  # mlcast_verification_tools.offline

- on the fly, never storing forecasts: forecasts are generated batch by batch on the
  GPU (or CPU), de-normalised and passed straight to a
  :class:`~mlcast_verification_tools.accumulator.ScoreAccumulator`::

      scores = verify_mlcast_run(config, checkpoint, ensemble_size=20)

With the same arguments and seed, both give the same scores.

Needs the ``mlcast`` extra (torch, pytorch-lightning, mlcast). Importing this module
is cheap: the command line helpers below have no heavy dependencies, and the
torch/mlcast API (:func:`forecast_mlcast`, :func:`open_truth`,
:func:`verify_mlcast_run`, :func:`load_mlcast_model`, :class:`VerificationDataset`,
:func:`verify_mlcast_model`, ...) is imported on first access from
:mod:`._mlcast_core`.

Pipeline
--------
1. :func:`load_run_config` loads the config with mlcast's own
   ``mlcast.config.load_yaml_config`` and resolves its relative data paths
   (:func:`find_config_yaml` locates ``<run>/config.yaml`` for a checkpoint).
2. :func:`load_mlcast_model` restores a ``NowcastLightningModule`` from the
   checkpoint. The network is not part of the checkpoint's hyper-parameters (see
   below), so it is rebuilt from ``pl_module.network`` of the config.
3. :func:`split_init_times` builds the run's data module and takes the unique init
   times of the crops mlcast's sampler selects for the test (or validation) split;
   :meth:`VerificationDataset.from_mlcast_config` reads full-domain cases at those
   init times from the source Zarr, normalised exactly as mlcast's training
   datasets do. The :class:`VerificationDataset` constructor also takes an explicit
   source and init times (e.g. :func:`init_times_from_range`). With ``crops=True``
   (``--crops``) each crop of the split (:func:`split_crops`) is instead forecast on
   its own, as mlcast's test step does, and the scores of the crops of an init time
   are pooled (on-the-fly scoring only).
4. The model is run, clamped to the normalised range (as
   ``NowcastLightningModule.shared_step`` does) and de-normalised on the device with
   :func:`denormalize_torch`; :func:`verify_mlcast_model` accumulates the scores,
   :func:`forecast_mlcast` returns (or writes) the forecasts.

Command line
------------
Two mlcast sub-commands (``mlcast verify-trained-model`` / ``mlcast
forecast-trained-model``, registered under the ``mlcast.commands`` entry-point group,
also available as ``mlcast.verify verify-trained-model`` / ``forecast-trained-model``):
:func:`verify_trained_model_main` and :func:`forecast_trained_model_main`.

Units
-----
Forecasts and observations are scored in the *native units of the source variable*,
without conversion; the scores are named after the variable's ``standard_name``
(:data:`SUPPORTED_STANDARD_NAMES`, ``tp`` for rain rates, ``dbz`` for reflectivities)
and carry its ``units``.

- Rain rates (``rainfall_rate``, ``rainfall_flux``): mlcast applies its
  Marshall-Palmer Z-R scaling directly to the stored values (``rainfall_flux`` reuses
  the mm/h relation unchanged; mlcast warns about this), so the model predicts in the
  units of the source variable and :func:`denormalize_torch` returns values in those
  units, like mlcast's ``DENORMALIZATION_REGISTRY``. The rank-histogram zero threshold
  is 0.037 mm/h expressed in those units (:func:`.units.default_zero_threshold`).
- Reflectivities (``equivalent_reflectivity_factor``, dBZ): mlcast clips them to
  ``[dbz_floor, dbz_ceiling]`` and maps that range linearly onto the normalised range;
  :func:`denormalize_torch` maps back to dBZ. The observations are clipped to the
  same range, since values outside it (e.g. a "no echo" value far below the floor)
  cannot be predicted. The rank-histogram zero threshold is ``dbz_floor``.

Checkpoints
-----------
``NowcastLightningModule.__init__`` only saves ``ensemble_size``, ``loss_class``,
``loss_params`` and ``masked_loss`` as hyper-parameters, not ``network``, so
mlcast's ``NowcastLightningModule.from_checkpoint`` cannot restore a checkpoint
on its own (``__init__`` is called without its required ``network`` argument).
:func:`load_mlcast_model` passes the network explicitly to
``load_from_checkpoint``.
"""

from __future__ import annotations

import argparse
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - for type checkers/IDEs; imported lazily at runtime
    from ._mlcast_core import (  # re-exported lazily via __getattr__
        VerificationDataset,
        denormalize_torch,
        find_config_yaml,
        forecast_mlcast,
        init_times_from_range,
        load_mlcast_model,
        load_run_config,
        mm_per_hour_factor,
        open_truth,
        split_crops,
        split_init_times,
        verify_mlcast_model,
        verify_mlcast_run,
    )

#: CF standard names of the source variables that can be verified: rain rates, which
#: mlcast normalises through its Z-R relation, and reflectivities in dBZ.
SUPPORTED_STANDARD_NAMES = ("rainfall_rate", "rainfall_flux", "equivalent_reflectivity_factor")

#: Precisions the model can be run in: ``fp32`` (default) or ``bf16`` (``torch.autocast``
#: to bfloat16; the forecast is cast back to float32 before de-normalisation).
PRECISIONS = ("fp32", "bf16")

_LAZY = (
    "VerificationDataset",
    "denormalize_torch",
    "find_config_yaml",
    "forecast_mlcast",
    "init_times_from_range",
    "load_mlcast_model",
    "load_run_config",
    "mm_per_hour_factor",
    "open_truth",
    "split_crops",
    "split_init_times",
    "verify_mlcast_model",
    "verify_mlcast_run",
)

__all__ = [  # noqa: RUF022 - grouped: light helpers first, then the lazily imported API
    "PRECISIONS",
    "SUPPORTED_STANDARD_NAMES",
    "build_forecast_trained_model_parser",
    "build_verify_trained_model_parser",
    "forecast_trained_model_main",
    "verify_trained_model_main",
    "VerificationDataset",
    "denormalize_torch",
    "find_config_yaml",
    "forecast_mlcast",
    "init_times_from_range",
    "load_mlcast_model",
    "load_run_config",
    "mm_per_hour_factor",
    "open_truth",
    "split_crops",
    "split_init_times",
    "verify_mlcast_model",
    "verify_mlcast_run",
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


# ---------------------------------------------------------------------------
# Command line (standard library only)
# ---------------------------------------------------------------------------

VERIFY_HELP = "score a trained mlcast model on its run's test split, without storing forecasts"
FORECAST_HELP = "forecast with a trained mlcast model on its run's test split, to a Zarr store"


def _parse_slice(text: str) -> slice:
    try:
        start, stop = text.split(":")
        return slice(int(start) if start else None, int(stop) if stop else None)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected an index range 'start:stop', got {text!r}"
        ) from exc


def _add_run_arguments(parser: argparse.ArgumentParser, action: str) -> None:
    """Arguments defining the run, the model and the cases (shared by both commands)."""
    g = parser.add_argument_group("model")
    g.add_argument(
        "--config",
        required=True,
        help="the run's Fiddle config.yaml (mlcast writes it to <run>/config.yaml)",
    )
    g.add_argument(
        "--checkpoint", required=True, help="Lightning checkpoint (.ckpt) of the mlcast run"
    )
    g.add_argument("--ensemble-size", type=int, default=20, help="number of members (default: 20)")
    g.add_argument(
        "--member-chunk", type=int, help="generate members in calls of at most this many (>= 2)"
    )
    g.add_argument("--device", help="torch device (default: cuda if available, else cpu)")
    g.add_argument(
        "--precision",
        choices=PRECISIONS,
        default="fp32",
        help="run the model in float32 (default) or with bfloat16 autocast "
        "(about twice as fast on GPUs with bf16 tensor cores, e.g. A100/H100)",
    )
    g.add_argument("--seed", type=int, default=0)

    g = parser.add_argument_group("data (default: from the run config)")
    g.add_argument(
        "--split",
        choices=("test", "val"),
        default="test",
        help=f"split of the run config to {action} (default: test)",
    )
    g.add_argument(
        "--data-root",
        help="directory relative data paths of the config are resolved against "
        "(default: the current directory)",
    )
    g.add_argument("--zarr-path", help="source-data Zarr store (path or URL) to use instead")
    g.add_argument("--index-path", help="sampling index (parquet) to use instead")
    g.add_argument(
        "--storage-options",
        type=json.loads,
        help="JSON dict for xr.open_zarr, e.g. '{\"anon\": true}'",
    )
    g.add_argument(
        "--forecast-steps",
        type=int,
        help="number of lead times (default: as in training)",
    )
    g.add_argument(
        "--max-cases", type=int, help="evenly subsample the init times to at most this many"
    )
    g.add_argument("--region-y", type=_parse_slice, help="y index range start:stop")
    g.add_argument("--region-x", type=_parse_slice, help="x index range start:stop")
    g.add_argument("--on-missing", choices=("skip", "raise"), default="skip")
    g.add_argument("--batch-size", type=int, default=1)
    g.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker processes reading the source data (default: 0, i.e. read "
        "in the main process); a few (2-4) suffice to overlap reading with the GPU",
    )


def build_verify_trained_model_parser(
    prog: str | None = "mlcast verify-trained-model", *, add_help: bool = True
) -> argparse.ArgumentParser:
    """
    Argument parser of the ``verify-trained-model`` command.

    Imports nothing beyond the standard library. The source data, variable,
    input/forecast steps and the cases (the unique init times of the crops mlcast
    samples for the test split, forecast over the full domain) come from the run's
    ``config.yaml``, which must be given with ``--config``.

    Parameters
    ----------
    prog : str, optional
        Program name shown in the usage.
    add_help : bool
        Add ``-h/--help`` (False to use the parser as a parent of another parser).

    Returns
    -------
    argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog=prog, description=VERIFY_HELP[0].upper() + VERIFY_HELP[1:] + ".", add_help=add_help
    )
    _add_run_arguments(parser, "verify on")
    g = parser.add_argument_group("crops (default: forecast the full domain at each init time)")
    g.add_argument(
        "--crops",
        action="store_true",
        help="forecast and score each crop the sampler selects for the split on its own, "
        "as mlcast's test step does; the scores of the crops of an init time are pooled. "
        "--batch-size is then the number of crops per model call",
    )
    g.add_argument(
        "--max-crops",
        type=int,
        help="with --crops, evenly subsample the crops to at most this many",
    )
    g = parser.add_argument_group("output")
    g.add_argument(
        "--forecast-source", help="name of the forecast source (default: the run directory name)"
    )
    g.add_argument("--out", required=True, help="output score file (.nc) or Zarr store")
    g.add_argument("--no-psd", action="store_true", help="skip the radial power spectra")
    g.add_argument("--no-progress", action="store_true")
    g.add_argument("--title", help="title of the score dataset (default: generated)")
    g.add_argument(
        "--created-by", help="creator contact 'Name <email>' (optional; default: omitted)"
    )
    g.add_argument(
        "--created-with",
        help="GitHub URL@revision of the creating software "
        "(optional; default: this tool's repository@revision)",
    )
    g.add_argument("--forecasts", help="description of the forecasts (default: generated)")
    g.add_argument("--truth", help="description of the truth (default: from the source attrs)")
    g.add_argument("--license", default="CC-BY-4.0")
    g.add_argument("--dataset-version", default="0.1.0")
    return parser


def build_forecast_trained_model_parser(
    prog: str | None = "mlcast forecast-trained-model", *, add_help: bool = True
) -> argparse.ArgumentParser:
    """
    Argument parser of the ``forecast-trained-model`` command.

    Imports nothing beyond the standard library; the cases are defined as for
    :func:`build_verify_trained_model_parser`.

    Parameters
    ----------
    prog : str, optional
        Program name shown in the usage.
    add_help : bool
        Add ``-h/--help`` (False to use the parser as a parent of another parser).

    Returns
    -------
    argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog=prog,
        description=FORECAST_HELP[0].upper()
        + FORECAST_HELP[1:]
        + ". The store is written batch by batch (appending along init_time) and can "
        "be scored with `mlcast.verify score`.",
        add_help=add_help,
    )
    _add_run_arguments(parser, "forecast")
    g = parser.add_argument_group("output")
    g.add_argument(
        "--forecast-source", help="name of the forecast source (default: the run directory name)"
    )
    g.add_argument(
        "--out", required=True, help="output Zarr store of the forecasts (replaced if it exists)"
    )
    g.add_argument("--no-progress", action="store_true")
    return parser


def _core():
    try:
        from . import _mlcast_core
    except ImportError as exc:
        raise SystemExit(
            "this command needs the `mlcast` extra "
            "(pip install 'mlcast-verification-tools[mlcast]'); "
            f"import failed with: {exc}"
        ) from exc
    return _mlcast_core


def run_verify_trained_model(args: argparse.Namespace) -> int:
    """Run ``verify-trained-model`` with parsed arguments (imports torch/mlcast)."""
    return _core().verify_trained_model_from_args(args)


def run_forecast_trained_model(args: argparse.Namespace) -> int:
    """Run ``forecast-trained-model`` with parsed arguments (imports torch/mlcast)."""
    return _core().forecast_trained_model_from_args(args)


def verify_trained_model_main(argv: list[str] | None = None) -> int:
    """Score a trained mlcast model on its run's test split, without storing forecasts.

    Entry point of ``mlcast verify-trained-model`` (``mlcast.commands`` entry point).
    Needs ``--config`` (the run's ``config.yaml``), ``--checkpoint`` and ``--out``.

    Parameters
    ----------
    argv : list of str, optional
        Arguments without the command name; default ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit code.
    """
    args = build_verify_trained_model_parser().parse_args(argv)
    return run_verify_trained_model(args)


def forecast_trained_model_main(argv: list[str] | None = None) -> int:
    """Forecast with a trained mlcast model on its run's test split, to a Zarr store.

    Entry point of ``mlcast forecast-trained-model`` (``mlcast.commands`` entry
    point). Needs ``--config`` (the run's ``config.yaml``), ``--checkpoint`` and
    ``--out``; score the store with ``mlcast.verify score``.

    Parameters
    ----------
    argv : list of str, optional
        Arguments without the command name; default ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit code.
    """
    args = build_forecast_trained_model_parser().parse_args(argv)
    return run_forecast_trained_model(args)
