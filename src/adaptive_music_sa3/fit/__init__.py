"""Bayesian optimization: library Phase 0 and per-clip Phase 1."""

from .clip import OptimizedConfig, optimize_audio
from .library import (
    clip_records,
    discover_library,
    run_phase0,
    run_phase0_ts,
    stratified_sample,
)

__all__ = [
    "OptimizedConfig",
    "clip_records",
    "discover_library",
    "optimize_audio",
    "run_phase0",
    "run_phase0_ts",
    "stratified_sample",
]
