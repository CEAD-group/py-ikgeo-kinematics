import importlib.util
from pathlib import Path

import pytest

_MODULE = Path(__file__).resolve().parents[2] / "build_backend" / "ikgeo_version.py"


def _load():
    spec = importlib.util.spec_from_file_location("ikgeo_version", _MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("described", "expected"),
    [
        ("v0.0.1-0-g58b7753", "0.0.1"),
        ("v0.0.1-0-g58b7753-dirty", "0.0.2-dev.0+g58b7753.dirty"),
        ("v0.0.1-3-g58b7753", "0.0.2-dev.3+g58b7753"),
        ("v1.4.9-12-gabc1234-dirty", "1.4.10-dev.12+gabc1234.dirty"),
        ("vnext-1-gabc1234", None),
    ],
)
def test_version_from_describe(described, expected):
    assert _load().version_from_describe(described) == expected
