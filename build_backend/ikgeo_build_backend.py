"""In-tree PEP 517 backend: give local builds a version from the git tag.

`Cargo.toml` carries the placeholder `0.0.0-dev`, which the release CI rewrites
from the tag before building. Without this shim every other build, notably an
editable install in a sibling checkout, reports `0.0.0.dev0` and fails a
consumer's `>=0.0.1` floor.

When the placeholder is in place inside a git checkout with a `v*` tag, it is
rewritten from `git describe` (see `ikgeo_version`). An exact clean tag gives
the tag version (`0.0.1`). Anything past the tag or dirty bumps the patch, as
hatch-vcs and setuptools-scm do, so it sorts above the tag it builds on:
`0.0.2-dev.3+g1234abc`, which maturin normalises to the PEP 440
`0.0.2.dev3+g1234abc`. Any other version in `Cargo.toml` is left alone, so the
CI release path is untouched.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from pathlib import Path

from ikgeo_version import PLACEHOLDER, git_version
from maturin import build_editable as _maturin_build_editable
from maturin import build_sdist as _maturin_build_sdist
from maturin import build_wheel as _maturin_build_wheel

# Hooks this module does not wrap are re-exported unchanged.
from maturin import (  # noqa: F401
    get_requires_for_build_editable,
    get_requires_for_build_sdist,
    get_requires_for_build_wheel,
)
from maturin import (
    prepare_metadata_for_build_editable as _maturin_prepare_metadata_for_build_editable,
)
from maturin import (
    prepare_metadata_for_build_wheel as _maturin_prepare_metadata_for_build_wheel,
)

_VERSION_LINE = re.compile(r'^version = ".*"$', re.MULTILINE)


@contextmanager
def _versioned_cargo_toml():
    root = Path(__file__).resolve().parent.parent
    cargo_toml = root / "Cargo.toml"
    cargo_lock = root / "Cargo.lock"
    original = cargo_toml.read_text()
    if f'version = "{PLACEHOLDER}"' not in original:
        # CI already stamped the release version.
        yield
        return
    version = git_version(root)
    if version is None:
        yield
        return
    lock_original = cargo_lock.read_text() if cargo_lock.exists() else None
    cargo_toml.write_text(_VERSION_LINE.sub(f'version = "{version}"', original, count=1))
    try:
        yield
    finally:
        cargo_toml.write_text(original)
        # cargo rewrites the lock to match the patched version during the build.
        if lock_original is not None:
            cargo_lock.write_text(lock_original)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    with _versioned_cargo_toml():
        return _maturin_build_wheel(wheel_directory, config_settings, metadata_directory)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    with _versioned_cargo_toml():
        return _maturin_build_editable(wheel_directory, config_settings, metadata_directory)


def build_sdist(sdist_directory, config_settings=None):
    with _versioned_cargo_toml():
        return _maturin_build_sdist(sdist_directory, config_settings)


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    with _versioned_cargo_toml():
        return _maturin_prepare_metadata_for_build_wheel(metadata_directory, config_settings)


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    with _versioned_cargo_toml():
        return _maturin_prepare_metadata_for_build_editable(metadata_directory, config_settings)
