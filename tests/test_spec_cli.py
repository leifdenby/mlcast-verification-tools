"""Validate written score datasets with the validator's command line interface.

The CLI runs in a subprocess, so its GitHub lookup of `mlcast_created_with` (this
package's repository and revision by default) can't be patched out. Without network
access that lookup gives a WARNING (not a FAIL), so the exit code is 0 either way.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys

import pytest

from mlcast_verification_tools.io import write_scores
from mlcast_verification_tools.testing import make_synthetic_scores

pytest.importorskip("mlcast_dataset_validator")

ATTRS = dict(
    title="Synthetic precipitation nowcast scores",
    forecasts="two synthetic ensembles",
    truth="synthetic radar composite",
)


def _cli_command() -> list[str]:
    script = pathlib.Path(sys.executable).parent / "mlcast.validate_dataset"
    if script.exists():
        return [str(script)]
    found = shutil.which("mlcast.validate_dataset")
    if found:
        return [found]
    return [sys.executable, "-m", "mlcast_dataset_validator.specs.cli"]


@pytest.mark.parametrize("suffix", [".nc", ".zarr"])
def test_cli_validates_written_scores(tmp_path, suffix):
    path = write_scores(make_synthetic_scores(), tmp_path / f"scores{suffix}", **ATTRS)
    result = subprocess.run(
        [*_cli_command(), "verification", "forecast_scoring_metrics", str(path)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "0 fail(s)" in result.stdout, result.stdout
