"""The NJ4-220 example must run and pass its own calibration check."""

import subprocess
import sys
from pathlib import Path

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "example_comau_nj4_220.py"


def test_example_nj4_runs_and_calibrates() -> None:
    result = subprocess.run(
        [sys.executable, str(EXAMPLE)],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "Calibration check: match" in result.stdout
    assert "family: two_intersecting_two_parallel" in result.stdout
    assert "NaN rows: 0" in result.stdout
