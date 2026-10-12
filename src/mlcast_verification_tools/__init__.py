"""Verification scores and plots for precipitation nowcasts.

Scores are computed per forecast case (torch kernels, CPU or GPU), stored in the
MLCast `verification/forecast_scoring_metrics` format, and all plots are made from
that stored format.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mlcast-verification-tools")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "unknown"
