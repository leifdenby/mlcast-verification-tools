# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

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
