"""Scalar objectives ``J`` for 1-D curves and 2-D T/S grids.

Overview
--------
- ``score_curve``: Spearman of commanded ``m`` vs ``M_hat`` / drift
- ``score_grid``: on-axis T/tc and S/hc, minus |cross| correlations,
  plus ``max(T,S)`` vs drift and a loud-corner drift hinge
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

_LOUD_CORNERS: tuple[tuple[float, float], ...] = ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0))
# 5×5 and 3×3 corners sit on the targets; farther nearest-cells count as missing.
_CORNER_MAX_DIST2 = 0.26**2


def spearman(x: NDArray[np.floating], y: NDArray[np.floating]) -> float:
    """Spearman rho, or 0 if either series has near-zero variance.

    Parameters
    ----------
    x, y : array-like
        Paired series.

    Returns
    -------
    float
        Correlation in ``[-1, 1]``, or ``0.0`` if undefined.
    """
    from scipy import stats

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    rho, _ = stats.spearmanr(x, y)
    if rho is None or np.isnan(rho):
        return 0.0
    return float(rho)


def mono_violations(y: NDArray[np.floating]) -> int:
    """Count adjacent decreases (non-monotonic steps).

    Parameters
    ----------
    y : array-like
        Ordered series along increasing command.

    Returns
    -------
    int
        Number of pairs with ``y[i] >= y[i+1]``.
    """
    y = np.asarray(y, dtype=np.float64)
    return int(sum(1 for i in range(len(y) - 1) if y[i] >= y[i + 1] - 1e-12))


def _viol_grouped(
    command: NDArray[np.floating],
    group: NDArray[np.floating],
    values: NDArray[np.floating],
) -> int:
    """Monotonicity violations of ``values`` vs ``command``, within each ``group``."""
    buckets: dict[float, list[tuple[float, float]]] = defaultdict(list)
    for c, g, v in zip(command, group, values, strict=True):
        buckets[round(float(g), 6)].append((float(c), float(v)))
    total = 0
    for pts in buckets.values():
        pts.sort(key=lambda p: p[0])
        total += mono_violations(np.array([v for _, v in pts], dtype=np.float64))
    return total


def _nearest_drift(
    t: NDArray[np.floating],
    s: NDArray[np.floating],
    drift: NDArray[np.floating],
    target: tuple[float, float],
) -> float | None:
    """Drift at the cell nearest ``target`` in (T, S), or None if missing."""
    if t.size == 0:
        return None
    dist2 = (t - target[0]) ** 2 + (s - target[1]) ** 2
    i = int(np.argmin(dist2))
    if float(dist2[i]) > _CORNER_MAX_DIST2:
        return None
    return float(drift[i])


def _drift_hinge(
    t: NDArray[np.floating],
    s: NDArray[np.floating],
    drift: NDArray[np.floating],
    *,
    margin: float,
) -> float:
    """Relative drift floor: loud corners should exceed ``d00 + margin``.

    Looks up nearest cells to ``(0,0)``, ``(1,0)``, ``(0,1)``, ``(1,1)``.
    Returns 0 if any of those corners is missing.
    """
    d00 = _nearest_drift(t, s, drift, (0.0, 0.0))
    if d00 is None:
        return 0.0
    total = 0.0
    for corner in _LOUD_CORNERS:
        dc = _nearest_drift(t, s, drift, corner)
        if dc is None:
            return 0.0
        total += max(0.0, d00 + margin - dc)
    return float(total)


@dataclass
class ObjectiveBreakdown:
    """Components of the 1-D curve objective ``J``.

    Attributes
    ----------
    J : float
        Combined score (higher is better).
    spearman_mhat, spearman_drift : float
        Rank correlations of ``m`` with each series.
    viol_mhat, viol_drift : int
        Monotonicity violation counts.
    """

    J: float
    spearman_mhat: float
    spearman_drift: float
    viol_mhat: int
    viol_drift: int


@dataclass
class GridObjective:
    """Components of the 2-D T/S grid objective ``J``.

    Attributes
    ----------
    J : float
        Combined score (higher is better).
    spearman_T_tc, spearman_S_hc : float
        On-axis rank correlations.
    spearman_T_hc, spearman_S_tc : float
        Cross-talk rank correlations (penalized in absolute value).
    spearman_maxTS_drift : float
        Rank correlation of ``max(T, S)`` with drift.
    drift_hinge : float
        Sum of ``max(0, d00 + margin - d_c)`` over loud corners.
    viol_T_tc, viol_S_hc : int
        Monotonicity violations along T (at fixed S) and S (at fixed T).
    """

    J: float
    spearman_T_tc: float
    spearman_S_hc: float
    spearman_T_hc: float
    spearman_S_tc: float
    spearman_maxTS_drift: float
    drift_hinge: float
    viol_T_tc: int
    viol_S_hc: int


def score_curve(
    m: NDArray[np.floating] | list[float],
    m_hat: NDArray[np.floating] | list[float],
    drift: NDArray[np.floating] | list[float],
    *,
    w_mhat: float = 0.5,
    w_drift: float = 0.5,
    lam_mono: float = 0.3,
) -> ObjectiveBreakdown:
    """Score one control curve (several ``m`` levels) as scalar ``J``.

    ::

        J = w1 Spearman(m, M_hat) + w2 Spearman(m, drift)
            - lam (viol(M_hat) + viol(drift))

    Parameters
    ----------
    m : sequence of float
        Commanded musicality levels.
    m_hat : sequence of float
        DSP musicality estimates.
    drift : sequence of float
        Latent drift from the source.
    w_mhat, w_drift : float
        Weights on the two Spearman terms.
    lam_mono : float
        Penalty weight per monotonicity violation.

    Returns
    -------
    ObjectiveBreakdown
        ``J`` and its components.
    """
    m_arr = np.asarray(m, dtype=np.float64)
    mh = np.asarray(m_hat, dtype=np.float64)
    dr = np.asarray(drift, dtype=np.float64)
    s_m = spearman(m_arr, mh)
    s_d = spearman(m_arr, dr)
    v_m = mono_violations(mh)
    v_d = mono_violations(dr)
    J = w_mhat * s_m + w_drift * s_d - lam_mono * (v_m + v_d)
    return ObjectiveBreakdown(
        J=float(J),
        spearman_mhat=s_m,
        spearman_drift=s_d,
        viol_mhat=v_m,
        viol_drift=v_d,
    )


def score_grid(
    T: NDArray[np.floating] | list[float],
    S: NDArray[np.floating] | list[float],
    tc: NDArray[np.floating] | list[float],
    hc: NDArray[np.floating] | list[float],
    drift: NDArray[np.floating] | list[float],
    *,
    lam_x: float = 0.5,
    lam_mono: float = 0.3,
    lam_c: float = 1.0,
    margin: float = 0.08,
) -> GridObjective:
    """Score a T/S control grid, rewarding axis independence.

    ::

        J = Spearman(T, tc) + Spearman(S, hc)
            - lam_x (|Spearman(T, hc)| + |Spearman(S, tc)|)
            + Spearman(max(T, S), drift)
            - lam_m (viol_T(tc) + viol_S(hc))
            - lam_c * hinge

    ``hinge`` is ``sum max(0, d00 + margin - d_c)`` for corners
    ``(1,0)``, ``(0,1)``, ``(1,1)`` vs nearest-cell ``(0,0)``.

    Parameters
    ----------
    T, S : sequence of float
        Commanded temporal / spectral levels.
    tc, hc, drift : sequence of float
        Measured DSP scores and latent drift.
    lam_x : float
        Weight on absolute cross-axis Spearman terms.
    lam_mono : float
        Weight on 2-D monotonicity violations.
    lam_c : float
        Weight on the loud-corner drift hinge.
    margin : float
        Required extra drift at loud corners vs ``(0,0)``.

    Returns
    -------
    GridObjective
        ``J`` and its components.
    """
    t = np.asarray(T, dtype=np.float64)
    s = np.asarray(S, dtype=np.float64)
    tc_a = np.asarray(tc, dtype=np.float64)
    hc_a = np.asarray(hc, dtype=np.float64)
    dr = np.asarray(drift, dtype=np.float64)
    max_ts = np.maximum(t, s)

    s_t_tc = spearman(t, tc_a)
    s_s_hc = spearman(s, hc_a)
    s_t_hc = spearman(t, hc_a)
    s_s_tc = spearman(s, tc_a)
    s_max_d = spearman(max_ts, dr)
    hinge = _drift_hinge(t, s, dr, margin=margin)
    v_t = _viol_grouped(t, s, tc_a)
    v_s = _viol_grouped(s, t, hc_a)
    J = (
        s_t_tc
        + s_s_hc
        - lam_x * (abs(s_t_hc) + abs(s_s_tc))
        + s_max_d
        - lam_mono * (v_t + v_s)
        - lam_c * hinge
    )
    return GridObjective(
        J=float(J),
        spearman_T_tc=s_t_tc,
        spearman_S_hc=s_s_hc,
        spearman_T_hc=s_t_hc,
        spearman_S_tc=s_s_tc,
        spearman_maxTS_drift=s_max_d,
        drift_hinge=hinge,
        viol_T_tc=v_t,
        viol_S_hc=v_s,
    )
