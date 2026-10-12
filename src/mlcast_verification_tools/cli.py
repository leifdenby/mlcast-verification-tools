"""Command line interface `mlcast.verify`.

Subcommands
-----------
plot
    Make all verification figures from a stored score dataset.
run-mlcast
    Score an MLCast model run (needs the ``mlcast`` extra); arguments and handler
    live in `mlcast_verification_tools.integrations.mlcast`.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import xarray as xr

NETCDF_SUFFIXES = (".nc", ".nc4", ".cdf")


def _open_scores(path) -> xr.Dataset:
    """Open a score dataset, via `io.open_scores` when available."""
    try:
        from .io import open_scores
    except ImportError:
        open_scores = None
    if open_scores is not None:
        return open_scores(path)

    import xarray as xr

    path = pathlib.Path(path)
    if path.suffix.lower() in NETCDF_SUFFIXES:
        return xr.open_dataset(path)
    return xr.open_zarr(path)


def _cmd_plot(args: argparse.Namespace) -> int:
    import matplotlib

    matplotlib.use("Agg")
    from .plots import plot_all

    ds = _open_scores(args.scores_path)
    written = plot_all(ds, args.out, fmt=args.format, dpi=args.dpi)
    if not written:
        print(f"no plottable score variables found in {args.scores_path}", file=sys.stderr)
        return 1
    for path in written:
        print(path)
    return 0


def _add_run_mlcast_subcommand(subparsers) -> None:
    help_text = "score an MLCast model run and write a score dataset"
    try:
        from .integrations.mlcast import add_run_mlcast_arguments
    except ImportError as exc:
        error = exc
        p = subparsers.add_parser(
            "run-mlcast",
            help=f"{help_text} (unavailable: install the `mlcast` extra)",
            description=f"{help_text}. Requires `pip install mlcast-verification-tools[mlcast]`.",
        )
        p.add_argument("rest", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)

        def _unavailable(args: argparse.Namespace) -> int:
            raise SystemExit(
                "`mlcast.verify run-mlcast` needs the `mlcast` extra "
                "(pip install 'mlcast-verification-tools[mlcast]'); "
                f"import failed with: {error}"
            )

        p.set_defaults(func=_unavailable)
        return

    p = subparsers.add_parser("run-mlcast", help=help_text, description=help_text)
    add_run_mlcast_arguments(p)

    def _run(args: argparse.Namespace) -> int:
        from .integrations.mlcast import run_mlcast_from_args

        result = run_mlcast_from_args(args)
        return result if isinstance(result, int) else 0

    p.set_defaults(func=_run)


def build_parser() -> argparse.ArgumentParser:
    """Build the `mlcast.verify` argument parser."""
    parser = argparse.ArgumentParser(
        prog="mlcast.verify",
        description="Verification scores and plots for precipitation nowcasts.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="log progress")
    subparsers = parser.add_subparsers(dest="command", required=True)

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

    _add_run_mlcast_subcommand(subparsers)
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
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
