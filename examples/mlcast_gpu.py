"""Score a trained mlcast run on its test split, on the GPU (no forecasts on disk).

Only the checkpoint is needed: the run's ``config.yaml`` (mlcast writes it to
``<run>/config.yaml``, next to ``<run>/checkpoints/``) defines the network, the
source data, the variable, the input/forecast steps and the test split. The cases
are the unique init times of the crops mlcast samples for the test split, each
forecast over the full radar domain.

Steps: verify on the GPU, write the scores, optionally merge them with the scores
of other forecast sources, and plot.

Needs the ``mlcast`` extra::

    pip install 'mlcast-verification-tools[mlcast]'

Example, run from the directory training was started from (the config's data
paths, e.g. ``./data/radar.zarr``, are relative to it; see ``--data-root``)::

    python examples/mlcast_gpu.py \\
        --checkpoint logs/mlcast/version_3/checkpoints/best.ckpt \\
        --other-scores pysteps_scores.zarr --plots plots/

The same, without merging or plotting, is available as ``mlcast.verify run-mlcast``.
"""

from __future__ import annotations

import argparse

from mlcast_verification_tools.accumulator import merge_sources
from mlcast_verification_tools.integrations.mlcast import verify_mlcast_run
from mlcast_verification_tools.io import open_scores, write_scores
from mlcast_verification_tools.plots import plot_all


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", help="directory the config's relative data paths start from")
    p.add_argument("--ensemble-size", type=int, default=20)
    p.add_argument("--max-cases", type=int, help="evenly subsample the test init times")
    p.add_argument("--member-chunk", type=int, help="members per model call, to save GPU memory")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--out", default="mlcast_scores.zarr")
    p.add_argument(
        "--other-scores", nargs="*", default=[], help="score files of other sources to merge"
    )
    p.add_argument("--plots", help="directory for the figures")
    args = p.parse_args()

    # 1. Load config + model, build the test cases, forecast and score on the GPU
    #    (if available), batch by batch; only the scores are kept. The forecast
    #    source is named after the run directory (pass forecast_source= to change it).
    scores = verify_mlcast_run(
        args.checkpoint,
        ensemble_size=args.ensemble_size,
        data_root=args.data_root,
        max_cases=args.max_cases,
        batch_size=args.batch_size,
        member_chunk=args.member_chunk,
    )
    source = str(scores.forecast_source.values[0])

    # 2. Write the scores with the global attributes the spec requires.
    write_scores(
        scores,
        args.out,
        title=f"Verification scores of {source}",
        forecasts=f"{source}: {args.ensemble_size}-member mlcast ConvGRU ensemble, test split",
        truth="mlcast source data of the run, converted to mm/h",
        created_by="Your Name <you@example.com>",
        created_with="https://github.com/mlcast-community/mlcast-dataset-dmi-nowcast-scores@v0.1.0",
    )
    print(f"wrote scores of {scores.sizes['init_time']} cases to {args.out}")

    # 3. Optionally compare with other forecast sources scored on the same cases,
    #    e.g. a pysteps baseline written with write_scores.
    if args.other_scores:
        scores = merge_sources([scores, *(open_scores(path).load() for path in args.other_scores)])

    # 4. Plots: score vs lead time, rank histograms, power spectra.
    if args.plots:
        for path in plot_all(scores, args.plots):
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
