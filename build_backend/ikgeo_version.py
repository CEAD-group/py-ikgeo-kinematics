"""Map `git describe` onto a Cargo version; no maturin import, so tests can load it."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

PLACEHOLDER = "0.0.0-dev"
# v0.0.1-3-g1234abc[-dirty]
_DESCRIBE = re.compile(
    r"^v(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)"
    r"-(?P<distance>\d+)-g(?P<sha>[0-9a-f]+)(?P<dirty>-dirty)?$"
)


def version_from_describe(described: str) -> str | None:
    """Cargo semver for a `git describe --long --dirty` string, or None if unparseable."""
    match = _DESCRIBE.match(described)
    if not match:
        return None
    major, minor, patch = match["major"], match["minor"], int(match["patch"])
    distance, sha, dirty = int(match["distance"]), match["sha"], bool(match["dirty"])
    if distance == 0 and not dirty:
        return f"{major}.{minor}.{patch}"
    local = f"g{sha}.dirty" if dirty else f"g{sha}"
    return f"{major}.{minor}.{patch + 1}-dev.{distance}+{local}"


def git_version(root: Path) -> str | None:
    """Version derived from the checkout at `root`, or None outside a tagged git tree."""
    try:
        described = subprocess.run(
            ["git", "describe", "--tags", "--long", "--dirty", "--match", "v[0-9]*"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return version_from_describe(described)
