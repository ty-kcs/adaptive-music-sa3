"""1-D endpoint-linear parameter schedules.

Maps a musicality knob ``m`` to edit parameters via
``theta(m) = (1-m) theta0 + m theta1``, with box constraints for Bayesian
optimization. Invert ``gamma`` is fixed at 0.

Overview
--------
- Endpoints <-> flat BO vector
- ``m`` -> concrete parameter dict
- g0 JSON load/save with backfill for missing keys
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from adaptive_music_sa3._paths import default_g0_path

MethodName = Literal["flowedit", "rfinv_mp", "rfinv_fp"]

# Box constraints for the BO search space.
BOUNDS: dict[str, tuple[float, float]] = {
    "t_start": (0.3, 1.0),
    "tgt_cfg": (1.0, 12.0),
    "src_cfg": (1.0, 3.0),
    "t_stop": (0.0, 0.25),
    "eta": (0.0, 1.0),
    "cfg": (1.0, 12.0),
    "stop": (0.5, 1.0),
    "start": (0.0, 0.3),
}

FLOWEDIT_KEYS = ("t_start", "tgt_cfg", "src_cfg", "t_stop")
RFINV_KEYS = ("eta", "cfg", "stop", "start")

METHOD_KEYS: dict[MethodName, tuple[str, ...]] = {
    "flowedit": FLOWEDIT_KEYS,
    "rfinv_mp": RFINV_KEYS,
    "rfinv_fp": RFINV_KEYS,
}

DEFAULT_ENDPOINTS: dict[MethodName, dict[str, Any]] = {
    "flowedit": {
        # FE winner: t_start 0.50→0.95, tgt_cfg 4→7; src_cfg fixed at 1.8
        "theta0": {"t_start": 0.50, "tgt_cfg": 4.0, "src_cfg": 1.8, "t_stop": 0.0},
        "theta1": {"t_start": 0.95, "tgt_cfg": 7.0, "src_cfg": 1.8, "t_stop": 0.0},
    },
    "rfinv_mp": {
        # mp winner: η 0.30→0.08, cfg 3→5, stop 1.0→0.85
        "theta0": {"eta": 0.30, "cfg": 3.0, "stop": 1.0, "start": 0.0},
        "theta1": {"eta": 0.08, "cfg": 5.0, "stop": 0.85, "start": 0.0},
    },
    "rfinv_fp": {
        # fp winner: η ≈ 3.5×mp → 1.00→0.28; cfg/stop same as mp
        "theta0": {"eta": 1.00, "cfg": 3.0, "stop": 1.0, "start": 0.0},
        "theta1": {"eta": 0.28, "cfg": 5.0, "stop": 0.85, "start": 0.0},
    },
}


@dataclass
class Endpoints:
    """Parameter endpoints at musicality ``m=0`` and ``m=1``.

    Attributes
    ----------
    method : MethodName
        Edit method whose key set defines theta.
    theta0 : dict[str, float]
        Parameters at ``m=0``.
    theta1 : dict[str, float]
        Parameters at ``m=1``.
    """

    method: MethodName
    theta0: dict[str, float]  # musicality m = 0
    theta1: dict[str, float]  # musicality m = 1

    def keys(self) -> tuple[str, ...]:
        """Return the ordered parameter names for this method."""
        return METHOD_KEYS[self.method]

    def at(self, m: float) -> dict[str, float]:
        """Linearly interpolate parameters at musicality ``m``.

        Parameters
        ----------
        m : float
            Musicality in ``[0, 1]`` (values outside are still lerped).

        Returns
        -------
        dict[str, float]
            ``theta(m) = (1-m) theta0 + m theta1`` for each key.
        """
        m = float(m)
        out = {}
        for k in self.keys():
            a, b = self.theta0[k], self.theta1[k]
            out[k] = a + (b - a) * m
        return out

    def to_vector(self) -> list[float]:
        """Flatten endpoints for BO: ``[theta0..., theta1...]``.

        Returns
        -------
        list[float]
            Values in ``keys()`` order for both ends.
        """
        keys = self.keys()
        return [self.theta0[k] for k in keys] + [self.theta1[k] for k in keys]

    @classmethod
    def from_vector(
        cls, method: MethodName, vec: list[float] | tuple[float, ...]
    ) -> Endpoints:
        """Rebuild endpoints from a flat BO vector.

        Parameters
        ----------
        method : MethodName
            Method that defines key order and dimension.
        vec : list[float] | tuple[float, ...]
            Length ``2 * len(METHOD_KEYS[method])``.

        Returns
        -------
        Endpoints
            Reconstructed endpoints (not yet clipped).
        """
        keys = METHOD_KEYS[method]
        n = len(keys)
        if len(vec) != 2 * n:
            raise ValueError(f"expected {2 * n} dims for {method}, got {len(vec)}")
        theta0 = {k: float(vec[i]) for i, k in enumerate(keys)}
        theta1 = {k: float(vec[n + i]) for i, k in enumerate(keys)}
        return cls(method=method, theta0=theta0, theta1=theta1)

    def clip_to_bounds(self) -> Endpoints:
        """Clamp to ``BOUNDS`` and enforce window constraints.

        FlowEdit requires ``t_stop < t_start``. RF-Inv requires ``start <= stop``.

        Returns
        -------
        Endpoints
            A new clipped instance.
        """
        keys = self.keys()
        t0, t1 = {}, {}
        for k in keys:
            lo, hi = BOUNDS[k]
            t0[k] = float(min(max(self.theta0[k], lo), hi))
            t1[k] = float(min(max(self.theta1[k], lo), hi))

        if self.method == "flowedit":
            t0["t_stop"] = min(t0["t_stop"], t0["t_start"] - 1e-3)
            t1["t_stop"] = min(t1["t_stop"], t1["t_start"] - 1e-3)
            lo_ts, _ = BOUNDS["t_stop"]
            t0["t_stop"] = max(t0["t_stop"], lo_ts)
            t1["t_stop"] = max(t1["t_stop"], lo_ts)
        else:
            t0["start"] = min(t0["start"], t0["stop"])
            t1["start"] = min(t1["start"], t1["stop"])

        return Endpoints(self.method, t0, t1)


def search_space(method: MethodName) -> list[tuple[float, float]]:
    """Return scikit-optimize dimensions for ``theta0`` then ``theta1``."""
    keys = METHOD_KEYS[method]
    return [BOUNDS[k] for k in keys] + [BOUNDS[k] for k in keys]


def default_endpoints(method: MethodName) -> Endpoints:
    """Return a copy of the default endpoints for ``method``."""
    raw = DEFAULT_ENDPOINTS[method]
    return Endpoints(method=method, theta0=dict(raw["theta0"]), theta1=dict(raw["theta1"]))


def _merge_theta(method: MethodName, side: str, raw: dict[str, Any]) -> dict[str, float]:
    """Fill missing keys from ``DEFAULT_ENDPOINTS`` when loading legacy JSON."""
    base = dict(DEFAULT_ENDPOINTS[method][side])
    for k, v in raw.items():
        if k in METHOD_KEYS[method]:
            base[k] = float(v)
    return base


def load_g0(path: Path | None = None) -> dict[MethodName, Endpoints]:
    """Load Phase-0 endpoints from JSON, or fall back to defaults."""
    path = path or default_g0_path()
    if not path.is_file():
        return {m: default_endpoints(m) for m in METHOD_KEYS}
    data = json.loads(path.read_text())
    out: dict[MethodName, Endpoints] = {}
    for method in METHOD_KEYS:
        block = data.get(method) or data.get("methods", {}).get(method)
        if block is None:
            out[method] = default_endpoints(method)
            continue
        out[method] = Endpoints(
            method=method,  # type: ignore[arg-type]
            theta0=_merge_theta(method, "theta0", block["theta0"]),
            theta1=_merge_theta(method, "theta1", block["theta1"]),
        ).clip_to_bounds()
    return out


def save_g0(
    endpoints: dict[str, Endpoints],
    path: Path | None = None,
    *,
    meta: dict[str, Any] | None = None,
) -> Path:
    """Write endpoints to JSON (both nested ``methods`` and flat keys)."""
    path = path or default_g0_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"methods": {}, "meta": meta or {}}
    for name, ep in endpoints.items():
        payload["methods"][name] = {
            "theta0": ep.theta0,
            "theta1": ep.theta1,
        }
        payload[name] = payload["methods"][name]
    path.write_text(json.dumps(payload, indent=2))
    return path


def trust_region_bounds(ep: Endpoints, frac: float = 0.2) -> list[tuple[float, float]]:
    """Build a local BO box around ``ep`` intersected with global ``BOUNDS``."""
    keys = ep.keys()
    vec = ep.to_vector()
    dims: list[tuple[float, float]] = []
    for i, val in enumerate(vec):
        k = keys[i % len(keys)]
        lo_b, hi_b = BOUNDS[k]
        half = abs(val) * frac if abs(val) > 1e-6 else (hi_b - lo_b) * frac
        lo = max(lo_b, val - half)
        hi = min(hi_b, val + half)
        if hi <= lo:
            lo, hi = lo_b, hi_b
        dims.append((lo, hi))
    return dims


def endpoints_to_dict(ep: Endpoints) -> dict[str, Any]:
    """Serialize endpoints to a plain dict (includes dataclass fields)."""
    return {"method": ep.method, "theta0": ep.theta0, "theta1": ep.theta1, **asdict(ep)}


__all__ = [
    "BOUNDS",
    "DEFAULT_ENDPOINTS",
    "Endpoints",
    "FLOWEDIT_KEYS",
    "METHOD_KEYS",
    "MethodName",
    "RFINV_KEYS",
    "default_endpoints",
    "default_g0_path",
    "endpoints_to_dict",
    "load_g0",
    "save_g0",
    "search_space",
    "trust_region_bounds",
]
