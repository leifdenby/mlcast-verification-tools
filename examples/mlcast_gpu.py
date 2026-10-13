"""Score a trained mlcast run on its test split, on the GPU.

The run's training config is an explicit input: mlcast writes it to
``<run>/config.yaml`` (next to ``<run>/checkpoints/``), and it defines the network,
the source data, the variable, the input/forecast steps and the test split. The
cases are the unique init times of the crops mlcast samples for the test split, each
forecast over the full radar domain.

Two ways to get the same scores (same arguments and seed):

a) ``--forecasts forecasts.zarr``: forecast first, writing the forecasts to a Zarr
   store batch by batch (:func:`forecast_mlcast`), then score them against the run's
   radar truth (:func:`open_truth`, :func:`compute_scores`). The forecasts are kept,
   e.g. to score them again or to look at them.
b) default: score on the fly (:func:`verify_mlcast_run`), never storing forecasts.

Then write the scores, optionally merge them with the scores of other forecast
sources, and plot.

Needs the ``mlcast`` extra::

    pip install 'mlcast-verification-tools[mlcast]'

Example, run from the directory training was started from (the config's data
paths, e.g. ``./data/radar.zarr``, are relative to it; see ``--data-root``)::

    python examples/mlcast_gpu.py \\
        --config logs/mlcast/version_3/config.yaml \\
        --checkpoint logs/mlcast/version_3/checkpoints/best.ckpt \\
        --other-scores pysteps_scores.zarr --plots plots/

The same steps are available on the command line as ``mlcast.verify
verify-trained-model`` (b), and ``mlcast.verify forecast-trained-model`` followed by
``mlcast.verify score`` (a).
"""

from __future__ import annotations

import argparse

from mlcast_verification_tools.accumulator import merge_sources
from mlcast_verification_tools.integrations.mlcast import (
    forecast_mlcast,
    load_run_config,
    open_truth,
    verify_mlcast_run,
)
from mlcast_verification_tools.io import open_scores, write_scores
from mlcast_verification_tools.offline import compute_scores
from mlcast_verification_tools.plots import plot_all


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", required=True, help="the run's config.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", help="directory the config's relative data paths start from")
    p.add_argument("--ensemble-size", type=int, default=20)
    p.add_argument("--max-cases", type=int, help="evenly subsample the test init times")
    p.add_argument("--member-chunk", type=int, help="members per model call, to save GPU memory")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument(
        "--forecasts", help="write the forecasts to this Zarr store first, then score them"
    )
    p.add_argument("--out", default="mlcast_scores.zarr")
    p.add_argument(
        "--other-scores", nargs="*", default=[], help="score files of other sources to merge"
    )
    p.add_argument("--plots", help="directory for the figures")
    args = p.parse_args()

    # The run's training config, with its relative data paths resolved.
    config = load_run_config(args.config, data_root=args.data_root)
    run_kwargs = dict(
        ensemble_size=args.ensemble_size,
        max_cases=args.max_cases,
        batch_size=args.batch_size,
        member_chunk=args.member_chunk,
        seed=0,
    )

    if args.forecasts:
        # a) Forecast (on the GPU if available), written batch by batch, then score the
        #    stored forecasts against the run's radar truth. The forecast source is
        #    named after the run directory (pass forecast_source= to change it).
        ds_forecasts = forecast_mlcast(config, args.checkpoint, out=args.forecasts, **run_kwargs)
        ds_truth = open_truth(config)
        scores = compute_scores(
            ds_forecasts, ds_truth, batch_size=args.batch_size, seed=run_kwargs["seed"]
        )
    else:
        # b) Forecast and score batch by batch on the GPU; only the scores are kept.
        scores = verify_mlcast_run(config, args.checkpoint, **run_kwargs)
    source = str(scores.forecast_source.values[0])

    # Write the scores with the global attributes the spec requires.
    write_scores(
        scores,
        args.out,
        title=f"Verification scores of {source}",
        forecasts=f"{source}: {args.ensemble_size}-member mlcast ConvGRU ensemble, test split",
        truth="mlcast source data of the run, converted to mm/h",
    )
    print(f"wrote scores of {scores.sizes['init_time']} cases to {args.out}")

    # Optionally compare with other forecast sources scored on the same cases,
    # e.g. a pysteps baseline written with write_scores.
    if args.other_scores:
        scores = merge_sources([scores, *(open_scores(path).load() for path in args.other_scores)])

    # Plots: score vs lead time, rank histograms, power spectra.
    if args.plots:
        for path in plot_all(scores, args.plots):
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
