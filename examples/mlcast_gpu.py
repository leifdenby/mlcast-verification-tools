"""Score a trained mlcast model on the GPU while it forecasts (no forecasts on disk).

Steps: load the model from its Lightning checkpoint, build forecast cases over a
date range from the mlcast source-data Zarr, verify on the GPU, write the scores,
optionally merge them with the scores of other forecast sources, and plot.

Needs the ``mlcast`` extra::

    pip install 'mlcast-verification-tools[mlcast]'

Example (an mlcast TensorBoard run directory holds ``config.yaml`` and
``checkpoints/*.ckpt``; the network is rebuilt from ``config.yaml``)::

    python examples/mlcast_gpu.py \\
        --checkpoint logs/mlcast/version_3/checkpoints/best.ckpt \\
        --source data/mlcast-source-datasets/IT-DPC-SRI/v0.1.0/italian-radar-dpc-sri.zarr \\
        --start 2023-07-01T00:25 --end 2023-07-01T02:25 --freq 30min \\
        --other-scores pysteps_scores.zarr --plots plots/

The same, without merging or plotting, is available as ``mlcast.verify run-mlcast``.
"""

from __future__ import annotations

import argparse

import torch

from mlcast_verification_tools.accumulator import merge_sources
from mlcast_verification_tools.integrations.mlcast import (
    VerificationDataset,
    init_times_from_range,
    load_mlcast_model,
    verify_mlcast_model,
)
from mlcast_verification_tools.io import open_scores, write_scores
from mlcast_verification_tools.plots import plot_all


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--config-yaml", help="the run's Fiddle config.yaml (default: found next to the checkpoint)"
    )
    p.add_argument("--source", required=True, help="mlcast source-data Zarr (path or URL)")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--freq", default="6h", help="spacing of the init times (e.g. 6h or PT6H)")
    p.add_argument("--input-steps", type=int, default=6, help="as in training")
    p.add_argument("--forecast-steps", type=int, default=12)
    p.add_argument("--ensemble-size", type=int, default=20)
    p.add_argument("--member-chunk", type=int, help="members per model call, to save GPU memory")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--forecast-source", default="mlcast-convgru")
    p.add_argument("--out", default="mlcast_scores.zarr")
    p.add_argument(
        "--other-scores", nargs="*", default=[], help="score files of other sources to merge"
    )
    p.add_argument("--plots", help="directory for the figures")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Model. mlcast checkpoints don't store the network architecture, so it is
    #    rebuilt from the run's Fiddle config.yaml (or pass `network=` yourself).
    model = load_mlcast_model(args.checkpoint, config_yaml=args.config_yaml, device=device)

    # 2. Cases: full domain (or pass region={"y": slice(..), "x": slice(..)}), one per
    #    init time = valid time of the last input frame. Init times whose frames are
    #    missing in the source data are skipped with a warning.
    dataset = VerificationDataset(
        args.source,
        init_times_from_range(args.start, args.end, args.freq),
        input_steps=args.input_steps,
        forecast_steps=args.forecast_steps,
    )
    print(
        f"{len(dataset)} cases, variable {dataset.variable!r} "
        f"({dataset.standard_name}, {dataset.units}), dx = {dataset.dx_km} km, "
        f"lead times up to {dataset.lead_times[-1].astype('timedelta64[m]')}"
    )

    # 3. Forecast and score on the GPU, batch by batch; only scores are kept.
    scores = verify_mlcast_model(
        model,
        dataset,
        forecast_source=args.forecast_source,
        ensemble_size=args.ensemble_size,
        batch_size=args.batch_size,
        member_chunk=args.member_chunk,
        device=device,
        num_workers=2,  # read the next cases while the GPU works
    )

    # 4. Write the scores with the global attributes the spec requires.
    write_scores(
        scores,
        args.out,
        title=f"Verification scores of {args.forecast_source}",
        forecasts=f"{args.forecast_source}: {args.ensemble_size}-member mlcast ConvGRU ensemble",
        truth=f"{dataset.ds.attrs.get('title', args.source)}, converted to mm/h",
        created_by="Your Name <you@example.com>",
        created_with="https://github.com/mlcast-community/mlcast-dataset-dmi-nowcast-scores@v0.1.0",
    )
    print(f"wrote {args.out}")

    # 5. Optionally compare with other forecast sources scored on the same cases,
    #    e.g. a pysteps baseline written with write_scores.
    if args.other_scores:
        scores = merge_sources([scores, *(open_scores(path).load() for path in args.other_scores)])

    # 6. Plots: score vs lead time, rank histograms, power spectra.
    if args.plots:
        for path in plot_all(scores, args.plots):
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
