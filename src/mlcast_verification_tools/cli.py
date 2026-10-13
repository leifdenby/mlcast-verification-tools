"""Command line interface `mlcast.verify`.

Subcommands
-----------
verify-trained-model
    Score a trained mlcast model on its run's test split on the fly, without storing
    forecasts (needs the ``mlcast`` extra); also ``mlcast verify-trained-model``.
forecast-trained-model
    Write the forecasts of a trained mlcast model on its run's test split to a Zarr
    store (needs the ``mlcast`` extra); also ``mlcast forecast-trained-model``.
score
    Score stored forecasts against a truth dataset (or the truth of an mlcast run).
plot
    Make all verification figures from a stored score dataset.

The mlcast subcommands' arguments and handlers live in
`mlcast_verification_tools.integrations.mlcast`.
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import xarray as xr

NETCDF_SUFFIXES = (".nc", ".nc4", ".cdf")


def _open_dataset(path) -> xr.Dataset:
    """Open a netCDF file or a Zarr store lazily."""
    import xarray as xr

    path = pathlib.Path(path)
    if path.suffix.lower() in NETCDF_SUFFIXES:
        return xr.open_dataset(path, chunks={})
    return xr.open_zarr(path)


def _cmd_plot(args: argparse.Namespace) -> int:
    import matplotlib

    matplotlib.use("Agg")
    from .io import open_scores
    from .plots import plot_all

    ds = open_scores(args.scores_path)
    written = plot_all(ds, args.out, fmt=args.format, dpi=args.dpi)
    if not written:
        print(f"no plottable score variables found in {args.scores_path}", file=sys.stderr)
        return 1
    for path in written:
        print(path)
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    from .accumulator import merge_sources
    from .io import write_scores
    from .offline import compute_scores

    if args.truth is not None:
        ds_truth = _open_dataset(args.truth)
        truth_description = (
            ds_truth.attrs.get("truth") or ds_truth.attrs.get("title") or str(args.truth)
        )
    else:
        from .integrations.mlcast import open_truth

        ds_truth = open_truth(
            args.truth_from_config,
            data_root=args.data_root,
            zarr_path=args.zarr_path,
            storage_options=args.storage_options,
            region={k: v for k, v in (("y", args.region_y), ("x", args.region_x)) if v is not None}
            or None,
        )
        truth_description = ds_truth.attrs["truth"]

    scores, descriptions = [], []
    for path in args.forecasts_paths:
        ds_forecasts = _open_dataset(path)
        variable = args.variable
        if variable is None:
            if len(ds_forecasts.data_vars) != 1:
                raise SystemExit(
                    f"{path} has data variables {list(ds_forecasts.data_vars)}; pick one "
                    "with --variable"
                )
            variable = str(next(iter(ds_forecasts.data_vars)))
        scores.append(
            compute_scores(
                ds_forecasts,
                ds_truth,
                variable=variable,
                forecast_source=args.forecast_source,
                dx_km=args.dx_km,
                device=args.device,
                batch_size=args.batch_size,
                seed=args.seed,
            )
        )
        descriptions.append(ds_forecasts.attrs.get("forecasts") or str(path))
    ds_scores = merge_sources(scores)
    sources = ", ".join(str(s) for s in ds_scores["forecast_source"].values)
    path = write_scores(
        ds_scores,
        args.out,
        title=args.title or f"Verification scores of {sources}",
        forecasts=args.forecasts_description or "; ".join(descriptions),
        truth=args.truth_description or truth_description,
        created_by=args.created_by,
        created_with=args.created_with,
        license=args.license,
        dataset_version=args.dataset_version,
    )
    print(f"wrote scores of {ds_scores.sizes['init_time']} cases to {path}")
    return 0


def _add_score_subcommand(subparsers) -> None:
    from .integrations.mlcast import _parse_slice

    help_text = "score stored forecasts against a truth dataset and write a score dataset"
    p = subparsers.add_parser(
        "score",
        help=help_text,
        description=(
            "Score forecast datasets (variable (forecast_source, init_time, lead_time, "
            "member, y, x) with a units attr, e.g. from forecast-trained-model) against a "
            "truth dataset (the same variable (time, y, x) in the same units) and write a "
            "score dataset, in the units of the data. Several forecast stores are scored "
            "separately and merged along forecast_source."
        ),
    )
    p.add_argument(
        "forecasts_paths", metavar="FORECASTS", nargs="+", help="forecast datasets (.zarr or .nc)"
    )
    truth = p.add_mutually_exclusive_group(required=True)
    truth.add_argument("--truth", metavar="PATH", help="truth dataset (.zarr or .nc)")
    truth.add_argument(
        "--truth-from-config",
        metavar="CONFIG",
        help="the truth of an mlcast run: its config.yaml (needs the `mlcast` extra)",
    )
    g = p.add_argument_group("truth from an mlcast config")
    g.add_argument("--data-root", help="directory the config's relative data paths start from")
    g.add_argument("--zarr-path", help="source-data Zarr store (path or URL) to use instead")
    g.add_argument(
        "--storage-options", type=json.loads, help="JSON dict for xr.open_zarr of the source data"
    )
    g.add_argument("--region-y", type=_parse_slice, help="y index range start:stop")
    g.add_argument("--region-x", type=_parse_slice, help="x index range start:stop")

    g = p.add_argument_group("scoring")
    g.add_argument(
        "--variable",
        help="data variable to score, in forecasts and truth "
        "(default: the single data variable of the forecasts)",
    )
    g.add_argument(
        "--forecast-source",
        help="name of the source, for forecasts without a forecast_source dim/coordinate",
    )
    g.add_argument("--dx-km", type=float, help="grid spacing in km (default: from x)")
    g.add_argument("--device", help="torch device (default: cuda if available, else cpu)")
    g.add_argument("--batch-size", type=int, default=1, help="init times per batch")
    g.add_argument("--seed", type=int, default=0, help="seed of the rank-histogram tie-breaking")

    g = p.add_argument_group("output")
    g.add_argument("--out", required=True, help="output score file (.nc) or Zarr store")
    g.add_argument("--title", help="title of the score dataset (default: generated)")
    g.add_argument(
        "--forecasts-description",
        help="description of the forecasts (default: the stores' `forecasts` attrs)",
    )
    g.add_argument(
        "--truth-description",
        help="description of the truth (default: the truth's `truth`/`title` attr)",
    )
    g.add_argument("--created-by", help="creator contact 'Name <email>' (optional)")
    g.add_argument(
        "--created-with",
        help="GitHub URL@revision of the creating software (default: this tool's)",
    )
    g.add_argument("--license", default="CC-BY-4.0")
    g.add_argument("--dataset-version", default="0.1.0")
    p.set_defaults(func=_cmd_score)


def _add_mlcast_subcommands(subparsers) -> None:
    from .integrations import mlcast as mi

    for name, help_text, build, run in (
        (
            "verify-trained-model",
            mi.VERIFY_HELP,
            mi.build_verify_trained_model_parser,
            mi.run_verify_trained_model,
        ),
        (
            "forecast-trained-model",
            mi.FORECAST_HELP,
            mi.build_forecast_trained_model_parser,
            mi.run_forecast_trained_model,
        ),
    ):
        template = build(prog=None, add_help=False)
        p = subparsers.add_parser(
            name, parents=[template], help=help_text, description=template.description
        )
        p.set_defaults(func=run)


def build_parser() -> argparse.ArgumentParser:
    """Build the `mlcast.verify` argument parser."""
    parser = argparse.ArgumentParser(
        prog="mlcast.verify",
        description="Verification scores and plots for precipitation nowcasts.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="log progress")
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_mlcast_subcommands(subparsers)
    _add_score_subcommand(subparsers)

    p_plot = subparsers.add_parser(
        "plot",
        help="make all verification figures from a score dataset",
        description="Make all verification figures from a score dataset (netCDF or zarr).",
    )
    p_plot.add_argument("scores_path", metavar="SCORES_PATH", help="score dataset (.nc or .zarr)")
    p_plot.add_argument("--out", "-o", required=True, type=pathlib.Path, help="output directory")
    p_plot.add_argument("--format", default="png", help="image format (default: png)")
    p_plot.add_argument("--dpi", type=int, default=150, help="resolution of raster images")
    p_plot.set_defaults(func=_cmd_plot)
    return parser


def main(argv: list[str] | None = None) -> int:
    """
    Entry point of `mlcast.verify`.

    Parameters
    ----------
    argv : list of str, optional
        Arguments (without the program name); defaults to `sys.argv[1:]`.

    Returns
    -------
    int
        Exit status.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    result = args.func(args)
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    sys.exit(main())
