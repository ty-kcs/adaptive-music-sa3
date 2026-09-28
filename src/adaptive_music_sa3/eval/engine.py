"""Shared SA3 edit evaluation for a 1-D curve or a T/S grid.

Phase 0 / Phase 1 both score a candidate schedule by editing the same
source latent at several command points, then folding DSP scores and
latent drift into a scalar ``J`` (higher is better). RF methods invert
once and reuse that noise; FlowEdit skips inversion.

Overview
--------
- Input: model + source latent + ``Endpoints`` (1-D ``m``) or ``Corners``
  (2-D ``T``/``S``) + a prompt ladder
- Per point: one SA3 edit, then ``tc`` / ``hc`` / ``M_hat`` and drift (M_hat is the predicted musicality from the model)
- 1-D ``evaluate_curve``: five ``m`` rungs → ``CurveResult``. ``J`` rewards
  Spearman of commanded ``m`` vs ``M_hat`` and vs drift, minus monotonicity
  violations
- 2-D ``evaluate_grid``: T/S cells → ``GridResult``. ``J`` rewards
  on-axis T↔tc and S↔hc, penalizes cross-talk, includes
  ``max(T,S)``↔drift, and a loud-corner drift hinge. Phase 0 uses
  5×5; Phase 1 local BO uses 3×3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from adaptive_music_sa3.control.corners import Corners
from adaptive_music_sa3.control.endpoints import Endpoints, MethodName
from adaptive_music_sa3.control.prompts import LEVELS, LadderName, prompt_for
from adaptive_music_sa3.dsp.musicality import score_musicality
from adaptive_music_sa3.dsp.objectives import (
    GridObjective,
    ObjectiveBreakdown,
    score_curve,
    score_grid,
)
from adaptive_music_sa3.sa3 import decode_latent, drift, flow_edit, invert, make_cond, sample

# Five-point musicality grid shared by Phase 0 and Phase 1 Optimize.
M_EVAL = (0.0, 0.25, 0.5, 0.75, 1.0)
M_EVAL_PHASE0 = M_EVAL

# 5×5 T/S grid for Phase-0 corner BO (25 edits per clip; matches prompt rungs).
TS_EVAL: tuple[tuple[float, float], ...] = tuple(
    (t, s) for t in LEVELS for s in LEVELS
)
# 3×3 grid for Phase-1 local refine (9 edits; Gradio-scale).
TS_EVAL_PHASE1: tuple[tuple[float, float], ...] = tuple(
    (t, s) for t in (0.0, 0.5, 1.0) for s in (0.0, 0.5, 1.0)
)


@dataclass
class LevelResult:
    m: float
    prompt: str
    params: dict[str, float]
    drift: float
    tc: float
    hc: float
    m_hat: float
    latent: torch.Tensor | None = None
    T: float | None = None
    S: float | None = None


@dataclass
class CurveResult:
    method: MethodName
    levels: list[LevelResult]
    objective: ObjectiveBreakdown
    inverted: torch.Tensor | None = None

    @property
    def m_values(self) -> list[float]:
        return [lv.m for lv in self.levels]

    @property
    def drifts(self) -> list[float]:
        return [lv.drift for lv in self.levels]

    @property
    def m_hats(self) -> list[float]:
        return [lv.m_hat for lv in self.levels]


@dataclass
class GridResult:
    """5×5 (or other) T/S evaluation plus ``GridObjective``."""

    cells: list[LevelResult]
    objective: GridObjective


def _sample_solver(method: MethodName) -> tuple[str, str, str]:
    """Return ``(invert_solver, invert_schedule, sample_solver)`` for ``method``."""
    if method == "rfinv_mp":
        return "midpoint", "logsnr", "midpoint"
    if method == "rfinv_fp":
        return "fixed-point", "model", "euler"
    raise ValueError(method)


@torch.inference_mode()
def invert_once(
    model,
    latent: torch.Tensor,
    seconds_total: float,
    method: MethodName,
    *,
    steps: int,
    seed: int,
    gamma: float = 0.0,
) -> torch.Tensor:
    """Invert once under an empty prompt (RF methods only)."""
    inv_solver, inv_sched, _ = _sample_solver(method)
    torch.manual_seed(seed)
    src_cond = make_cond(
        model, "", seconds_total, latent.shape[-1], latent.shape[0],
    )
    return invert(
        model,
        latent,
        src_cond,
        steps=steps,
        gamma=float(gamma),
        schedule=inv_sched,
        solver=inv_solver,
        disable_tqdm=True,
    )


@torch.inference_mode()
def edit_at_level(
    model,
    latent: torch.Tensor,
    seconds_total: float,
    endpoints: Endpoints,
    m: float,
    prompt: str,
    *,
    steps: int,
    seed: int,
    inverted: torch.Tensor | None = None,
    return_latent: bool = False,
    T: float | None = None,
    S: float | None = None,
) -> LevelResult:
    """Edit at a single musicality ``m`` and return drift plus DSP scores."""
    params = endpoints.at(m)
    method = endpoints.method

    if method == "flowedit":
        out = flow_edit(
            model,
            latent,
            target_prompt=prompt,
            seconds_total=seconds_total,
            source_prompt="",
            steps=steps,
            t_start=params["t_start"],
            t_stop=params["t_stop"],
            n_avg=1,
            src_cfg=params["src_cfg"],
            tgt_cfg=params["tgt_cfg"],
            schedule="model",
            seed=seed,
            disable_tqdm=True,
        )
    else:
        if inverted is None:
            inverted = invert_once(
                model,
                latent,
                seconds_total,
                method,
                steps=steps,
                seed=seed,
            )
        _, inv_sched, sample_solver = _sample_solver(method)
        tgt_cond = make_cond(
            model, prompt, seconds_total, latent.shape[-1], latent.shape[0],
        )
        out = sample(
            model,
            inverted,
            tgt_cond,
            steps=steps,
            cfg_scale=params["cfg"],
            eta=params["eta"],
            source_latent=latent,
            start=params["start"],
            stop=params["stop"],
            schedule=inv_sched,
            solver=sample_solver,
            disable_tqdm=True,
        )

    d = drift(out, latent)
    wav = decode_latent(model, out)
    sr = int(model.model.sample_rate)
    scores = score_musicality(wav.numpy(), sr)
    return LevelResult(
        m=float(m),
        prompt=prompt,
        params=params,
        drift=d,
        tc=scores.tc,
        hc=scores.hc,
        m_hat=scores.m_hat,
        latent=out if return_latent else None,
        T=None if T is None else float(T),
        S=None if S is None else float(S),
    )


@torch.inference_mode()
def evaluate_curve(
    model,
    latent: torch.Tensor,
    seconds_total: float,
    endpoints: Endpoints,
    *,
    ladder: LadderName = "glitch",
    prompt_mode: Literal["ladder", "fixed"] = "ladder",
    fixed_prompt: str | None = None,
    m_grid: tuple[float, ...] = M_EVAL,
    steps: int = 25,
    seed: int = 0,
    inverted: torch.Tensor | None = None,
    keep_latents: bool = False,
) -> CurveResult:
    """Generate the full ``m_grid`` curve and score objective ``J``."""
    method = endpoints.method
    if method != "flowedit" and inverted is None:
        inverted = invert_once(
            model,
            latent,
            seconds_total,
            method,
            steps=steps,
            seed=seed,
        )

    levels: list[LevelResult] = []
    for m in m_grid:
        prompt = prompt_for(
            ladder, m, mode=prompt_mode, fixed_prompt=fixed_prompt,
        )
        lv = edit_at_level(
            model,
            latent,
            seconds_total,
            endpoints,
            m,
            prompt,
            steps=steps,
            seed=seed + int(m * 100),
            inverted=inverted,
            return_latent=keep_latents,
        )
        levels.append(lv)

    obj = score_curve(
        [lv.m for lv in levels],
        [lv.m_hat for lv in levels],
        [lv.drift for lv in levels],
    )
    return CurveResult(
        method=method, levels=levels, objective=obj, inverted=inverted,
    )


def _one_shot_endpoints(params: dict[str, float]) -> Endpoints:
    """Endpoints whose lerp is constant at ``params`` (FlowEdit)."""
    return Endpoints(method="flowedit", theta0=dict(params), theta1=dict(params))


@torch.inference_mode()
def edit_at_ts(
    model,
    latent: torch.Tensor,
    seconds_total: float,
    corners: Corners,
    T: float,
    S: float,
    prompt: str,
    *,
    steps: int,
    seed: int,
    return_latent: bool = False,
) -> LevelResult:
    """Edit at commanded ``(T, S)`` using bilinear FlowEdit corners.

    Parameters
    ----------
    corners : Corners
        Four-corner schedule; ``corners.at(T, S)`` supplies FlowEdit params.
    T, S : float
        Temporal / spectral commands in ``[0, 1]``.
    prompt : str
        Already-composed T/S (or fixed) conditioning text.

    Returns
    -------
    LevelResult
        Drift, DSP scores, and the commanded ``T`` / ``S``.
    """
    params = corners.at(T, S)
    m = 0.5 * float(T) + 0.5 * float(S)
    return edit_at_level(
        model,
        latent,
        seconds_total,
        _one_shot_endpoints(params),
        m,
        prompt,
        steps=steps,
        seed=seed,
        return_latent=return_latent,
        T=T,
        S=S,
    )


@torch.inference_mode()
def evaluate_grid(
    model,
    latent: torch.Tensor,
    seconds_total: float,
    corners: Corners,
    *,
    ladder: LadderName = "glitch",
    prompt_mode: Literal["ladder", "fixed"] = "ladder",
    fixed_prompt: str | None = None,
    ts_grid: tuple[tuple[float, float], ...] = TS_EVAL,
    steps: int = 25,
    seed: int = 0,
    keep_latents: bool = False,
) -> GridResult:
    """Generate a T/S grid and score the independence objective."""
    cells: list[LevelResult] = []
    for i, (T, S) in enumerate(ts_grid):
        prompt = prompt_for(
            ladder, T=T, S=S, mode=prompt_mode, fixed_prompt=fixed_prompt,
        )
        lv = edit_at_ts(
            model,
            latent,
            seconds_total,
            corners,
            T,
            S,
            prompt,
            steps=steps,
            seed=seed + i,
            return_latent=keep_latents,
        )
        cells.append(lv)

    obj = score_grid(
        [c.T if c.T is not None else c.m for c in cells],
        [c.S if c.S is not None else 0.0 for c in cells],
        [c.tc for c in cells],
        [c.hc for c in cells],
        [c.drift for c in cells],
    )
    return GridResult(cells=cells, objective=obj)


def mean_J(curves: list[CurveResult]) -> float:
    if not curves:
        return float("-inf")
    return float(np.mean([c.objective.J for c in curves]))


def mean_grid_J(grids: list[GridResult]) -> float:
    if not grids:
        return float("-inf")
    return float(np.mean([g.objective.J for g in grids]))
