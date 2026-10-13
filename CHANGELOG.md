# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Changed

- Scores are computed in the native units of the verified variable instead of being
  converted to mm/h, and named after it (`mae.tp`, `mae.dbz`, ...); their `units`
  attributes follow the variable (e.g. `dBZ` and `dBZ2 km` for MAE and PSD). Rain-rate
  sources in e.g. kg m-2 s-1 are now scored in kg m-2 s-1. `compute_scores` requires
  forecasts and truth in the same units, and `merge_sources` refuses to merge scores in
  different units. The `to_mm_per_hour` arguments and `--to-mm-per-hour` are removed;
  `open_truth` and `forecast_mlcast` return the variable in the source's units.
  `mlcast.verify score --variable` defaults to the forecasts' single data variable.
- The rank histogram treats values below its zero threshold as tied below all other
  values, which works in any units (same ranks as before for rain rates). The threshold
  defaults to 0.037 mm/h expressed in the variable's units (`default_zero_threshold`).

### Added

- `verify-trained-model --crops` (`verify_mlcast_run(crops=True)`,
  `VerificationDataset(crops=, crop_size=)`): forecast each crop the sampler selects for
  the split on its own, as mlcast's test step does, instead of the full domain, with the
  scores of the crops of an init time pooled (`ScoreAccumulator(pool_by_init_time=True)`);
  `--max-crops` subsamples the crops. `split_crops` returns the split's crops.
- `--precision bf16` for `verify-trained-model` / `forecast-trained-model` (`precision=`
  in `verify_mlcast_model`, `verify_mlcast_run` and `forecast_mlcast`): runs the model
  under bfloat16 autocast, about twice as fast on GPUs with bf16 tensor cores; the
  forecasts are cast back to float32 before they are de-normalised and scored.
- A summary panel (rich, in the style of mlcast's sampler) printed before the first
  batch: source, variable, domain, init times, steps, model, ensemble, workload
  (cases × members × steps), batching and device/precision. Log output (loguru) of the
  test split's crops and unique init times, the loaded model and the elapsed time per
  case at the end. The progress bar shows the mean time per batch spent waiting for data, in the model and
  in the scores (or writing the forecasts).
- `VerificationDataset.grid_shape`.

- Verification of mlcast models trained on radar reflectivity
  (`equivalent_reflectivity_factor`, dBZ): forecasts are de-normalised to dBZ, the
  observations are clipped to mlcast's `[dbz_floor, dbz_ceiling]`, and the rank-histogram
  zero threshold is `dbz_floor`.

- Per-case verification scores for precipitation nowcasts, computed with torch on
  CPU or GPU: MAE of the ensemble mean, CRPS, rank histogram and radially averaged
  power spectra of forecast and observation.
- Reading, writing and validating score files in the MLCast
  `verification/forecast_scoring_metrics` format (netCDF or Zarr).
- Plots reproducing the verification figures of the IRENE paper (arXiv:2609.17175),
  made from the score file only, and the `mlcast.verify plot` command.
- On-the-fly GPU scoring of trained mlcast models without writing forecasts to disk
  (`verify_mlcast_model`, `mlcast.verify run-mlcast`).
- Verification of an mlcast run from its checkpoint alone, on the test split of its
  training config: `verify_mlcast_run`, `load_run_config`, `split_init_times`,
  `find_config_yaml` and `VerificationDataset.from_mlcast_config`. The cases are the
  unique init times of the crops mlcast samples for the split, forecast over the full
  domain. `mlcast.verify run-mlcast` now takes `--checkpoint` (plus the output
  attributes), with optional `--split`, `--data-root`, `--zarr-path`, `--index-path`,
  `--max-cases` and `--forecast-steps`. It no longer takes `--source`, `--start`,
  `--end`, `--freq`, `--input-steps`, `--variable`, `--standard-name` or `--time-step`.
  The run config is loaded with mlcast's `load_yaml_config`, which replaces our own
  Fiddle YAML loader (`load_fiddle_yaml` and `build_network_from_config_yaml` are
  removed).
- `mlcast_created_by` and `mlcast_created_with` are optional, following the
  forecast_scoring_metrics spec (validator `58d46b5`): `write_scores` and
  `mlcast.verify run-mlcast` no longer require `created_by`/`created_with`.
  `mlcast_created_with` defaults to this package's repository and revision, and
  `mlcast_created_by` is only written when given.
- Three-step Python API with the training config as explicit input:
  `forecast_mlcast(config, checkpoint, ...)` makes a forecast dataset (in memory or
  written to Zarr batch by batch), `open_truth(config)` gives the run's radar truth in
  mm/h, and `compute_scores(ds_forecasts, ds_truth)` scores them (it takes xr.Datasets
  instead of a dict of DataArrays). `verify_mlcast_run(config, checkpoint, ...)` does the
  same on the fly; the config is no longer discovered next to the checkpoint.
- `mlcast forecast-trained-model` and `mlcast verify-trained-model` subcommands,
  registered through mlcast's CLI plugin hook (entry-point group `mlcast.commands`), and
  the same commands plus `score` under `mlcast.verify`. `--config` is required;
  `mlcast.verify run-mlcast` is removed.
- DataLoader workers of `verify_mlcast_model` are spawned instead of forked. Forked
  workers could deadlock.
- Development setup: pre-commit hooks (ruff, pyproject-fmt), and CI workflows for
  tests, linting, changelog checks and PyPI releases.
