"""FlowEdit four-corner bilinear map for independent T / S.

Only ``t_start`` and ``tgt_cfg`` vary per corner (8-D BO). 
``src_cfg`` and ``t_stop`` are frozen at the moment. #TODO: make this configurable

Overview
--------
- ``theta(T,S)`` bilinear over corners 00 / 10 / 01 / 11
- BO vector: free keys at 00, 10, 01, 11
- JSON: ``artifacts/g0_ts.json``
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from adaptive_music_sa3._paths import default_g0_ts_path
from adaptive_music_sa3.control.endpoints import (
    BOUNDS,
    FLOWEDIT_KEYS,
    load_g0,
)

# BO-free keys, in vector order within each corner.
CORNER_FREE_KEYS = ("t_start", "tgt_cfg")
CORNER_FROZEN_KEYS = ("src_cfg", "t_stop")
CORNER_ORDER = ("00", "10", "01", "11")  # (T,S) = (0,0), (1,0), (0,1), (1,1)


def _clip_window(params: dict[str, float]) -> dict[str, float]:
    """Clamp free keys and keep ``t_stop < t_start``."""
    out = dict(params)
    for k in FLOWEDIT_KEYS:
        lo, hi = BOUNDS[k]
        out[k] = float(min(max(out[k], lo), hi))
    lo_ts, _ = BOUNDS["t_stop"]
    out["t_stop"] = min(out["t_stop"], out["t_start"] - 1e-3)
    out["t_stop"] = max(out["t_stop"], lo_ts)
    return out


def _apply_frozen(params: dict[str, float], frozen: dict[str, float]) -> dict[str, float]:
    out = dict(params)
    out.update(frozen)
    return _clip_window(out)


@dataclass
class Corners:
    """FlowEdit parameters at the four (T, S) corners.

    Attributes
    ----------
    theta00, theta10, theta01, theta11 : dict[str, float]
        Full FlowEdit dicts at (T,S) = (0,0), (1,0), (0,1), (1,1).
    frozen : dict[str, float]
        Shared ``src_cfg`` / ``t_stop`` (from 1-D g0 ``theta0``).
    """

    theta00: dict[str, float]
    theta10: dict[str, float]
    theta01: dict[str, float]
    theta11: dict[str, float]
    frozen: dict[str, float]

    def _corner(self, name: str) -> dict[str, float]:
        return {
            "00": self.theta00,
            "10": self.theta10,
            "01": self.theta01,
            "11": self.theta11,
        }[name]

    def at(self, T: float, S: float) -> dict[str, float]:
        """Bilinear interpolate FlowEdit params at ``(T, S)``.

        Frozen keys are taken from ``frozen``, not from the lerp.
        """
        T, S = float(T), float(S)
        w00 = (1.0 - T) * (1.0 - S)
        w10 = T * (1.0 - S)
        w01 = (1.0 - T) * S
        w11 = T * S
        out: dict[str, float] = {}
        for k in CORNER_FREE_KEYS:
            out[k] = (
                w00 * self.theta00[k]
                + w10 * self.theta10[k]
                + w01 * self.theta01[k]
                + w11 * self.theta11[k]
            )
        out.update(self.frozen)
        return _clip_window(out)

    def to_vector(self) -> list[float]:
        """Flatten free keys at corners ``00, 10, 01, 11`` (length 8)."""
        vec: list[float] = []
        for name in CORNER_ORDER:
            corner = self._corner(name)
            vec.extend(float(corner[k]) for k in CORNER_FREE_KEYS)
        return vec

    @classmethod
    def from_vector(
        cls,
        vec: list[float] | tuple[float, ...],
        *,
        frozen: dict[str, float],
    ) -> Corners:
        """Rebuild corners from an 8-D BO vector plus frozen keys."""
        if len(vec) != 8:
            raise ValueError(f"expected 8 dims for Corners, got {len(vec)}")
        corners: dict[str, dict[str, float]] = {}
        i = 0
        for name in CORNER_ORDER:
            raw = {k: float(vec[i + j]) for j, k in enumerate(CORNER_FREE_KEYS)}
            i += len(CORNER_FREE_KEYS)
            corners[name] = _apply_frozen(raw, frozen)
        return cls(
            theta00=corners["00"],
            theta10=corners["10"],
            theta01=corners["01"],
            theta11=corners["11"],
            frozen=dict(frozen),
        )

    def clip_to_bounds(self) -> Corners:
        """Clamp every corner and re-apply frozen keys."""
        return Corners(
            theta00=_apply_frozen(self.theta00, self.frozen),
            theta10=_apply_frozen(self.theta10, self.frozen),
            theta01=_apply_frozen(self.theta01, self.frozen),
            theta11=_apply_frozen(self.theta11, self.frozen),
            frozen=dict(self.frozen),
        )


def corner_search_space() -> list[tuple[float, float]]:
    """Return 8 ``(lo, hi)`` boxes for ``t_start`` / ``tgt_cfg`` × four corners."""
    return [BOUNDS[k] for _ in CORNER_ORDER for k in CORNER_FREE_KEYS]


def trust_region_bounds_corners(
    corners: Corners, frac: float = 0.2
) -> list[tuple[float, float]]:
    """Local BO box around ``corners`` free keys, intersected with ``BOUNDS``.

    Parameters
    ----------
    corners : Corners
        Center of the trust region (typically loaded ``g0_ts``).
    frac : float
        Half-width as a fraction of ``|value|`` (or of the global span if near 0).

    Returns
    -------
    list[tuple[float, float]]
        Eight ``(lo, hi)`` pairs aligned with ``corners.to_vector()``.
    """
    vec = corners.to_vector()
    dims: list[tuple[float, float]] = []
    for i, val in enumerate(vec):
        k = CORNER_FREE_KEYS[i % len(CORNER_FREE_KEYS)]
        lo_b, hi_b = BOUNDS[k]
        half = abs(val) * frac if abs(val) > 1e-6 else (hi_b - lo_b) * frac
        lo = max(lo_b, val - half)
        hi = min(hi_b, val + half)
        if hi <= lo:
            lo, hi = lo_b, hi_b
        dims.append((lo, hi))
    return dims


def frozen_from_g0_theta0(g0_path: Path | None = None) -> dict[str, float]:
    """Shared ``src_cfg`` / ``t_stop`` from fitted 1-D FlowEdit ``theta0``."""
    fe = load_g0(g0_path)["flowedit"]
    return {k: float(fe.theta0[k]) for k in CORNER_FROZEN_KEYS}


def default_corners(g0_path: Path | None = None) -> Corners:
    """Seed corners from 1-D FlowEdit g0: 00←θ0, 11←θ1, off-diagonal midpoint."""
    fe = load_g0(g0_path)["flowedit"]
    frozen = {k: float(fe.theta0[k]) for k in CORNER_FROZEN_KEYS}
    mid = {
        k: 0.5 * (float(fe.theta0[k]) + float(fe.theta1[k]))
        for k in FLOWEDIT_KEYS
    }
    return Corners(
        theta00=_apply_frozen(fe.theta0, frozen),
        theta11=_apply_frozen(fe.theta1, frozen),
        theta10=_apply_frozen(mid, frozen),
        theta01=_apply_frozen(mid, frozen),
        frozen=frozen,
    ).clip_to_bounds()


def load_g0_ts(path: Path | None = None) -> Corners:
    """Load T/S corners from JSON, or seed from 1-D g0."""
    path = path or default_g0_ts_path()
    if not path.is_file():
        return default_corners()
    data = json.loads(path.read_text())
    block = data.get("corners") or data
    frozen = {k: float(block["frozen"][k]) for k in CORNER_FROZEN_KEYS}
    return Corners(
        theta00=_apply_frozen(block["theta00"], frozen),
        theta10=_apply_frozen(block["theta10"], frozen),
        theta01=_apply_frozen(block["theta01"], frozen),
        theta11=_apply_frozen(block["theta11"], frozen),
        frozen=frozen,
    ).clip_to_bounds()


def save_g0_ts(
    corners: Corners,
    path: Path | None = None,
    *,
    meta: dict[str, Any] | None = None,
) -> Path:
    """Write corners to ``g0_ts.json``."""
    path = path or default_g0_ts_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "corners": {
            "theta00": corners.theta00,
            "theta10": corners.theta10,
            "theta01": corners.theta01,
            "theta11": corners.theta11,
            "frozen": corners.frozen,
        },
        "meta": meta or {},
    }
    path.write_text(json.dumps(payload, indent=2))
    return path
