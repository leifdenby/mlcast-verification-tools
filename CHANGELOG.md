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
- Development setup: pre-commit hooks (ruff, pyproject-fmt), and CI workflows for
  tests, linting, changelog checks and PyPI releases.
