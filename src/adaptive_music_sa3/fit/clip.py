"""Phase 1: per-clip trust-region BO on T/S FlowEdit corners.

Loads ``g0_ts.json``, then refines the 8 free keys
(``t_start`` / ``tgt_cfg`` at four corners) in a ±frac box so the
uploaded clip scores higher on the independence objective ``J``.

Overview
--------
- Input: model + latent + Phase-0 corners
- Scoring: 3×3 ``(T, S)`` grid via ``evaluate_grid`` / ``score_grid``
- Output: ``OptimizedConfig``
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from skopt import gp_minimize
from skopt.space import Real

from adaptive_music_sa3.control.corners import (
    Corners,
    load_g0_ts,
    trust_region_bounds_corners,
)
from adaptive_music_sa3.control.prompts import LadderName
from adaptive_music_sa3.dsp.objectives import GridObjective
from adaptive_music_sa3.eval.engine import (
    TS_EVAL_PHASE1,
    GridResult,
    evaluate_grid,
)


def _fmt_dur(seconds: float) -> str:
    """Format seconds as ``HhMmSs`` / ``MmSs`` for progress lines."""
    s = max(0, int(round(seconds)))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{sec:02d}s"
    return f"{m}m{sec:02d}s"


def _log(msg: str) -> None:
    """Print a Phase-1 progress line (flush for Gradio / tmux)."""
    print(msg, flush=True)


@dataclass
class OptimizedConfig:
    """Phase-1 result: refined T/S corners and grid objective.

    Attributes
    ----------
    corners : Corners
        Locally refined FlowEdit map.
    objective : GridObjective
        ``J`` and Spearman / violation breakdown.
    grid : GridResult | None
        Last evaluated 3×3 grid at the winner.
    n_local_calls : int
        Number of BO objective evaluations.
    """

    corners: Corners
    objective: GridObjective
    grid: GridResult | None = None
    n_local_calls: int = 0

    def to_meta(self) -> dict[str, Any]:
        obj = self.objective
        return {
            "method": "flowedit",
            "theta00": self.corners.theta00,
            "theta10": self.corners.theta10,
            "theta01": self.corners.theta01,
            "theta11": self.corners.theta11,
            "frozen": self.corners.frozen,
            "J": obj.J,
            "spearman_T_tc": obj.spearman_T_tc,
            "spearman_S_hc": obj.spearman_S_hc,
            "spearman_T_hc": obj.spearman_T_hc,
            "spearman_S_tc": obj.spearman_S_tc,
            "spearman_maxTS_drift": obj.spearman_maxTS_drift,
            "drift_hinge": obj.drift_hinge,
            "viol_T_tc": obj.viol_T_tc,
            "viol_S_hc": obj.viol_S_hc,
            "n_local_calls": self.n_local_calls,
        }


def local_refine_corners(
    model,
    latent,
    seconds_total: float,
    start: Corners,
    *,
    ladder: LadderName,
    prompt_mode: Literal["ladder", "fixed"],
    fixed_prompt: str | None,
    n_calls: int = 12,
    steps: int = 25,
    seed: int = 0,
    frac: float = 0.2,
    patience: int = 3,
    ts_grid: tuple[tuple[float, float], ...] = TS_EVAL_PHASE1,
) -> tuple[Corners, GridResult, int]:
    """Local BO in a ±``frac`` trust region around ``start`` corners.

    Each trial scores ``ts_grid`` (default 3×3). Stops early when best
    ``J`` does not improve for ``patience`` calls after the initial
    random points.

    Returns
    -------
    Corners
        Best corners found.
    GridResult
        Grid at those corners.
    int
        Number of objective evaluations.
    """
    dims = trust_region_bounds_corners(start, frac=frac)
    space = [Real(lo, hi) for lo, hi in dims]
    frozen = dict(start.frozen)
    n_initial = max(3, min(5, n_calls // 3))

    best_corners = start
    best_grid: GridResult | None = None
    best_J = float("-inf")
    calls = 0
    stale = 0
    stopped_early = False
    t0 = time.monotonic()

    def objective(vec):
        nonlocal best_corners, best_grid, best_J, calls, stale
        calls += 1
        call_i = calls
        _log(f"[phase1] local BO ts-corners call {call_i}/{n_calls} start")
        corners = Corners.from_vector(vec, frozen=frozen).clip_to_bounds()
        gr = evaluate_grid(
            model,
            latent,
            seconds_total,
            corners,
            ladder=ladder,
            prompt_mode=prompt_mode,
            fixed_prompt=fixed_prompt,
            ts_grid=ts_grid,
            steps=steps,
            seed=seed,
        )
        if gr.objective.J > best_J:
            best_J = gr.objective.J
            best_corners = corners
            best_grid = gr
            stale = 0
        else:
            stale += 1
        elapsed = time.monotonic() - t0
        eta = (elapsed / call_i) * (n_calls - call_i)
        _log(
            f"[phase1] local BO ts-corners call {call_i}/{n_calls} done "
            f"J={gr.objective.J:.4f} best={best_J:.4f} stale={stale}/{patience} "
            f"elapsed={_fmt_dur(elapsed)} eta={_fmt_dur(eta)}"
        )
        return -gr.objective.J

    def stop_cb(_res):
        nonlocal stopped_early
        if patience <= 0:
            return False
        if calls < n_initial:
            return False
        if stale >= patience:
            stopped_early = True
            _log(
                f"[phase1] early stop after {calls}/{n_calls} "
                f"(no best-J improve for {patience} calls)"
            )
            return True
        return False

    _log(
        f"[phase1] local BO ts-corners n_calls={n_calls} "
        f"n_initial={n_initial} patience={patience} frac={frac} "
        f"ts_grid={len(ts_grid)}"
    )
    gp_minimize(
        objective,
        space,
        n_calls=n_calls,
        random_state=seed,
        x0=start.to_vector(),
        n_initial_points=n_initial,
        callback=stop_cb,
    )
    assert best_grid is not None
    _log(
        f"[phase1] local BO done best_J={best_J:.4f} "
        f"calls={calls}/{n_calls} early_stop={stopped_early} "
        f"elapsed={_fmt_dur(time.monotonic() - t0)}"
    )
    return best_corners, best_grid, calls


def optimize_audio(
    model,
    latent,
    seconds_total: float,
    *,
    ladder: LadderName = "glitch",
    prompt_mode: Literal["ladder", "fixed"] = "ladder",
    fixed_prompt: str | None = None,
    g0_ts_path: Path | None = None,
    n_local_calls: int = 12,
    patience: int = 3,
    steps_search: int = 25,
    seed: int = 0,
) -> OptimizedConfig:
    """Run Phase 1 for one audio: local refine of T/S corners.

    Parameters
    ----------
    g0_ts_path : Path | None
        Phase-0 JSON; default ``artifacts/g0_ts.json``.
    n_local_calls, patience, steps_search, seed
        Local BO budget (3×3 grid per call).

    Returns
    -------
    OptimizedConfig
        Refined corners and ``GridObjective``.
    """
    t_run = time.monotonic()
    start = load_g0_ts(g0_ts_path)
    _log(
        f"[phase1] start ts-corners ladder={ladder} "
        f"prompt_mode={prompt_mode} n_local_calls={n_local_calls} "
        f"patience={patience} steps={steps_search}"
    )
    _log("[phase1] evaluate g0_ts on this clip …")
    gr0 = evaluate_grid(
        model,
        latent,
        seconds_total,
        start,
        ladder=ladder,
        prompt_mode=prompt_mode,
        fixed_prompt=fixed_prompt,
        ts_grid=TS_EVAL_PHASE1,
        steps=steps_search,
        seed=seed,
    )
    _log(f"[phase1] g0_ts J={gr0.objective.J:.4f}")

    corners, grid, n_calls = local_refine_corners(
        model,
        latent,
        seconds_total,
        start,
        ladder=ladder,
        prompt_mode=prompt_mode,
        fixed_prompt=fixed_prompt,
        n_calls=n_local_calls,
        steps=steps_search,
        seed=seed,
        patience=patience,
    )
    _log(
        f"[phase1] done J={grid.objective.J:.4f} "
        f"n_local_calls={n_calls} "
        f"total_elapsed={_fmt_dur(time.monotonic() - t_run)}"
    )
    return OptimizedConfig(
        corners=corners,
        objective=grid.objective,
        grid=grid,
        n_local_calls=n_calls,
    )
