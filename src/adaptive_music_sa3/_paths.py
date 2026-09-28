"""Filesystem roots for this package."""

from pathlib import Path


def package_root() -> Path:
    """Return the repo root (parent of ``src``)."""
    return Path(__file__).resolve().parents[2]


def default_g0_path() -> Path:
    """Return the default path for 1-D Phase-0 ``g0_endpoints.json``."""
    return package_root() / "artifacts" / "g0_endpoints.json"


def default_g0_ts_path() -> Path:
    """Return the default path for T/S corner ``g0_ts.json``."""
    return package_root() / "artifacts" / "g0_ts.json"
