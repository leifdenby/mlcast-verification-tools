# mlcast-verification-tools

Verification scores and plots for precipitation nowcasts, reproducing the verification
figures of the IRENE paper ([arXiv:2609.17175](https://arxiv.org/html/2609.17175v1)),
with the MLCast `verification/forecast_scoring_metrics` format (spec v0.1.0,
[mlcast-dataset-validator](https://github.com/leifdenby/mlcast-dataset-validator)) as
the intermediate file between scoring and plotting:

```
forecasts + radar ──(torch, CPU/GPU)──▶ per-case scores ──▶ scores.zarr / .nc ──▶ plots
  (on disk, or produced on the fly       (one value per            (spec-valid)
   by a trained mlcast model)             source × case × lead time)
```

## What is a "case"?

A **forecast case** is one initialisation time (`init_time`): the forecast started from
the radar observations up to that time, over a fixed verification domain. Every forecast
source is run for the same cases, so cases are what the sources are compared over.

Each stored score value belongs to one **forecast source × case × lead time**. To get it,
the forecast (all ensemble members) for that lead time is compared with the radar
observation at the valid time `init_time + lead_time`. The comparison is reduced over all
valid pixels of the domain (and over members, as the metric requires) to a single number,
or to a small vector for the rank histogram (over `rank`) and the spectra (over
`wavelength`).

Scores are **not** averaged over cases in the file. Averaging over cases (over
`init_time`) happens at plot time. Keeping the per-case values is what the spec asks for:
it lets the scores be bootstrapped over cases (e.g. by mlwp-scorecard), and it allows
subsets of cases (a season, rainy days) to be selected later.

## Metrics

Every metric is computed per case and lead time, then stored. Averaging over cases and
the quantities derived from those averages are computed at plot time.

| Metric | Paper | Stored as | Extra dim | Units | Polarity | Plot-time derivation |
|---|---|---|---|---|---|---|
| MAE of the ensemble mean | Eq. 21, Fig. 4 | `mae.tp` | – | mm h-1 | known | mean over cases |
| CRPS (ensemble estimator) | Eqs. 8, 22, Fig. 5 | `crps.tp` | – | mm h-1 | known | mean over cases |
| Rank histogram (rel. frequency; values < 0.037 mm/h set to 0, random tie-breaking) | Eq. 23, Fig. 6 | `rank_histogram.tp` | `rank` | 1 | `neutral` (declared) | mean over cases (and lead times for "full forecast") |
| Rank-histogram RMSD from uniform | Eq. 24, Fig. 6 | — | | | | from the case-averaged histogram |
| Radial PSD, forecast (mean over members) | Eqs. 16–17, Fig. 7 | `psd.tp` | `wavelength` (km) | mm2 h-2 km | `activity` (declared) | mean over cases |
| Radial PSD, observation | Eqs. 16–17, Fig. 7 | `psd_obs.tp` | `wavelength` (km) | mm2 h-2 km | `activity` (declared) | mean over cases |
| Excess power | Fig. 7 | — | | dB | | 10·log10(mean psd / mean psd_obs) at ≈74, 16, 4 km |

MAE = Mean Absolute Error, CRPS = Continuous Ranked Probability Score, RMSD = Root Mean
Square Deviation, PSD = Power Spectral Density (radially averaged, a.k.a. RAPSD).

The paper's case-study maps (Fig. 3) show raw fields rather than scores, so they are not
part of the score file.

## Install

PyTorch is installed through one of the `cpu`, `gpu-cu126`, `gpu-cu128` or
`gpu-cu130` extras, as in mlcast:

```bash
uv sync                                    # reading, writing and plotting score files
uv sync --extra cpu                        # + torch score kernels on CPU (offline scoring)
uv sync --extra gpu-cu128 --extra mlcast   # + on-the-fly GPU scoring of mlcast models
```

## Development

```bash
uv sync --extra cpu --extra mlcast   # dev tools (pytest, pre-commit, ruff, validator) included
uv run pre-commit install            # run ruff, pyproject-fmt etc. on every commit
uv run pytest
```

CI runs the tests and the pre-commit hooks on every push and pull request. Pull
requests must add an entry to `CHANGELOG.md` (or carry the `no changelog` label).
Publishing a GitHub release uploads the package to PyPI via trusted publishing; the
version comes from the git tag.

## Usage

Score forecasts stored on disk (`(init_time, lead_time, member, y, x)` in mm/h, radar as
`(time, y, x)`):

```python
from mlcast_verification_tools.offline import compute_scores
from mlcast_verification_tools.io import write_scores, validate_scores
from mlcast_verification_tools.plots import plot_all

scores = compute_scores({"irene": fc_irene, "steps": fc_steps}, radar)  # GPU if available
path = write_scores(
    scores, "scores.zarr",
    title="...", forecasts="...", truth="...",
    created_by="Name <email>",
    created_with="https://github.com/<org>/mlcast-dataset-<org>-<name>@v0.1.0",
)
validate_scores(path).console_print()
plot_all(scores, "figures/")
```

Score a trained mlcast model on the GPU without writing forecasts to disk (see
`examples/mlcast_gpu.py`), or from the command line:

```bash
mlcast.verify run-mlcast --help
mlcast.verify plot scores.zarr --out figures/
```

Synthetic end-to-end demo (scoring → spec file → validation → figures):

```bash
uv run --extra cpu python examples/synthetic_demo.py --out demo_output
```

## Notes on the format

- **Weighting.** The paper pools all pixels of all samples, while averaging the stored
  per-case scores weights every case equally. The two are identical when every case has
  the same number of valid pixels.
- **Nonlinear quantities are not stored.** RMSD and excess power are computed from
  case-averaged histograms/spectra; averaging per-case values would give different
  numbers.
- **`init_time` must be unique per source**, so verification uses a fixed domain per
  initialisation time (not random crops).
- **Ensemble sizes may differ** between sources; `rank` is NaN-padded to the largest K+1.
- **`mlcast_created_with`** must point to a repository named
  `mlcast-dataset-{organisation_id}-{dataset_name}` (validator §5.4), i.e. the repository
  of the score dataset, so `write_scores` requires it explicitly.
- The spec's polarity classes have no natural fit for a distribution such as the rank
  histogram; `neutral` is used until the spec gains e.g. a `distribution` class.

## Tests

```bash
uv run --extra mlcast pytest
```
