"""Phase 0: library-average Bayesian optimization of endpoints or T/S corners.

Overview
--------
- 1-D: per-method ``Endpoints`` -> ``g0_endpoints.json``
- T/S: FlowEdit ``Corners`` (8-D) -> ``g0_ts.json``
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from skopt import gp_minimize
from skopt.space import Real

from adaptive_music_sa3.control.corners import (
    Corners,
    corner_search_space,
    default_corners,
    save_g0_ts,
)
from adaptive_music_sa3.control.endpoints import (
    Endpoints,
    MethodName,
    default_endpoints,
    save_g0,
    search_space,
)
from adaptive_music_sa3.control.prompts import LadderName
from adaptive_music_sa3.eval.engine import (
    M_EVAL_PHASE0,
    TS_EVAL,
    CurveResult,
    GridResult,
    evaluate_curve,
    evaluate_grid,
    mean_J,
    mean_grid_J,
)
from adaptive_music_sa3.sa3 import load_latent

AUDIO_EXTS = {
    ".wav",
    ".wave",
    ".mp3",
    ".flac",
    ".ogg",
    ".oga",
    ".aiff",
    ".aif",
    ".aifc",
    ".m4a",
    ".aac",
    ".opus",
    ".wma",
    ".webm",
}
EXCLUDE_CATEGORIES = frozenset({"generated", "used_for_training"})


def _fmt_dur(seconds: float) -> str:
    """Format seconds as ``HhMmSs`` / ``MmSs`` for progress lines."""
    s = max(0, int(round(seconds)))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{sec:02d}s"
    return f"{m}m{sec:02d}s"


def _log(msg: str) -> None:
    """Print a Phase-0 progress line (flush for tmux / nohup)."""
    print(msg, flush=True)


@dataclass
class LibraryClip:
    path: Path
    category: str


def clip_records(clips: Sequence[LibraryClip]) -> list[dict[str, str]]:
    """Serialize clips as ``path`` / ``category`` dicts for Phase-0 JSON meta."""
    return [{"path": str(c.path), "category": c.category} for c in clips]


def _log_clip_split(label: str, clips: Sequence[LibraryClip]) -> None:
    """Print each selected clip path under ``label`` (search / val)."""
    _log(f"[phase0] {label} clips ({len(clips)}):")
    for c in clips:
        _log(f"[phase0]   {c.category}\t{c.path}")


def _is_audio(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in AUDIO_EXTS


def discover_library(root: Path) -> list[LibraryClip]:
    """List audio under ``{root}/{category}/**``, skipping excluded categories."""
    root = Path(root)
    clips: list[LibraryClip] = []
    subdirs = sorted(
        p for p in root.iterdir()
        if p.is_dir() and not p.name.startswith(".") and p.name not in EXCLUDE_CATEGORIES
    )
    for d in subdirs:
        for p in sorted(d.rglob("*")):
            if _is_audio(p):
                clips.append(LibraryClip(path=p, category=d.name))
    if clips:
        return clips
    for p in sorted(root.rglob("*")):
        if not _is_audio(p):
            continue
        if any(part in EXCLUDE_CATEGORIES for part in p.relative_to(root).parts):
            continue
        clips.append(LibraryClip(path=p, category="uncategorized"))
    return clips


def stratified_sample(
    clips: Sequence[LibraryClip],
    n_per_cat: int = 1,
    *,
    seed: int = 0,
) -> list[LibraryClip]:
    """Take up to ``n_per_cat`` clips from each discovered category (default 1)."""
    rng = random.Random(seed)
    by_cat: dict[str, list[LibraryClip]] = {}
    for c in clips:
        by_cat.setdefault(c.category, []).append(c)
    cats = sorted(by_cat.keys())
    out: list[LibraryClip] = []
    for cat in cats:
        pool = by_cat[cat]
        k = min(n_per_cat, len(pool))
        out.extend(rng.sample(pool, k))
    target = n_per_cat * max(len(cats), 1)
    if len(out) < target:
        chosen = {c.path for c in out}
        rest = [c for c in clips if c.path not in chosen]
        need = target - len(out)
        out.extend(rng.sample(rest, min(need, len(rest))))
    return out


def subsample(clips: Sequence[LibraryClip], n: int, *, seed: int = 1) -> list[LibraryClip]:
    rng = random.Random(seed)
    if len(clips) <= n:
        return list(clips)
    return rng.sample(list(clips), n)


def _load_clip_latents(model, clips: Sequence[LibraryClip], seconds: float, *, label: str = "load"):
    data = []
    n = len(clips)
    t0 = time.monotonic()
    for i, c in enumerate(clips, start=1):
        latent, secs = load_latent(model, c.path, seconds=seconds)
        data.append((c, latent, secs))
        if i == 1 or i == n or i % 8 == 0:
            elapsed = time.monotonic() - t0
            eta = (elapsed / i) * (n - i) if i else 0.0
            _log(
                f"[phase0] {label} {i}/{n} "
                f"elapsed={_fmt_dur(elapsed)} eta={_fmt_dur(eta)}"
            )
    return data


def evaluate_endpoints_on_set(
    model,
    clip_latents,
    endpoints: Endpoints,
    *,
    ladder: LadderName,
    steps: int,
    seed: int,
    prompt_mode: str = "ladder",
    fixed_prompt: str | None = None,
    m_grid: tuple[float, ...] = M_EVAL_PHASE0,
    progress: str | None = None,
    progress_every: int = 1,
) -> tuple[float, list[CurveResult]]:
    """Score mean ``J`` over a clip set."""
    curves: list[CurveResult] = []
    n = len(clip_latents)
    t0 = time.monotonic()
    for i, (_c, latent, secs) in enumerate(clip_latents):
        cr = evaluate_curve(
            model,
            latent,
            secs,
            endpoints,
            ladder=ladder,
            prompt_mode=prompt_mode,  # type: ignore[arg-type]
            fixed_prompt=fixed_prompt,
            m_grid=m_grid,
            steps=steps,
            seed=seed + i,
        )
        curves.append(cr)
        done = i + 1
        if progress and (done == n or done % max(1, progress_every) == 0):
            elapsed = time.monotonic() - t0
            eta = (elapsed / done) * (n - done)
            _log(
                f"[phase0] {progress} clip {done}/{n} "
                f"J_running={mean_J(curves):.4f} "
                f"elapsed={_fmt_dur(elapsed)} eta={_fmt_dur(eta)}"
            )
    return mean_J(curves), curves


def evaluate_corners_on_set(
    model,
    clip_latents,
    corners: Corners,
    *,
    ladder: LadderName,
    steps: int,
    seed: int,
    prompt_mode: str = "ladder",
    fixed_prompt: str | None = None,
    ts_grid: tuple[tuple[float, float], ...] = TS_EVAL,
    progress: str | None = None,
    progress_every: int = 1,
) -> tuple[float, list[GridResult]]:
    """Score mean grid ``J`` over a clip set."""
    grids: list[GridResult] = []
    n = len(clip_latents)
    t0 = time.monotonic()
    for i, (_c, latent, secs) in enumerate(clip_latents):
        gr = evaluate_grid(
            model,
            latent,
            secs,
            corners,
            ladder=ladder,
            prompt_mode=prompt_mode,  # type: ignore[arg-type]
            fixed_prompt=fixed_prompt,
            ts_grid=ts_grid,
            steps=steps,
            seed=seed + i,
        )
        grids.append(gr)
        done = i + 1
        if progress and (done == n or done % max(1, progress_every) == 0):
            elapsed = time.monotonic() - t0
            eta = (elapsed / done) * (n - done)
            _log(
                f"[phase0] {progress} clip {done}/{n} "
                f"J_running={mean_grid_J(grids):.4f} "
                f"elapsed={_fmt_dur(elapsed)} eta={_fmt_dur(eta)}"
            )
    return mean_grid_J(grids), grids


def fit_method_endpoints(
    model,
    search_latents,
    method: MethodName,
    *,
    ladder: LadderName = "glitch",
    n_calls: int = 40,
    steps: int = 50,
    seed: int = 0,
    x0: Endpoints | None = None,
    m_grid: tuple[float, ...] = M_EVAL_PHASE0,
    callback: Callable[[Endpoints, float], None] | None = None,
) -> tuple[Endpoints, float, list[tuple[Endpoints, float]]]:
    """Optimize endpoints for one method with ``gp_minimize``."""
    space = [Real(lo, hi, name=f"d{i}") for i, (lo, hi) in enumerate(search_space(method))]
    history: list[tuple[Endpoints, float]] = []
    start = x0 or default_endpoints(method)
    t0 = time.monotonic()
    best_so_far = float("-inf")

    def objective(vec):
        nonlocal best_so_far
        call_i = len(history) + 1
        ep = Endpoints.from_vector(method, vec).clip_to_bounds()
        _log(f"[phase0] BO {method} call {call_i}/{n_calls} start")
        J, _ = evaluate_endpoints_on_set(
            model,
            search_latents,
            ep,
            ladder=ladder,
            steps=steps,
            seed=seed,
            m_grid=m_grid,
            progress=f"BO {method} call {call_i}/{n_calls}",
            progress_every=max(1, len(search_latents) // 3),
        )
        history.append((ep, J))
        if J > best_so_far:
            best_so_far = J
        elapsed = time.monotonic() - t0
        eta = (elapsed / call_i) * (n_calls - call_i)
        _log(
            f"[phase0] BO {method} call {call_i}/{n_calls} done "
            f"J={J:.4f} best={best_so_far:.4f} "
            f"elapsed={_fmt_dur(elapsed)} eta={_fmt_dur(eta)}"
        )
        if callback:
            callback(ep, J)
        return -J

    res = gp_minimize(
        objective,
        space,
        n_calls=n_calls,
        random_state=seed,
        x0=start.to_vector(),
        n_initial_points=max(5, min(10, n_calls // 4)),
    )
    best = Endpoints.from_vector(method, res.x).clip_to_bounds()
    best_J = -float(res.fun)
    if history:
        hist_best = max(history, key=lambda t: t[1])
        if hist_best[1] > best_J:
            best, best_J = hist_best[0], hist_best[1]
    return best, best_J, history


def fit_corners(
    model,
    search_latents,
    *,
    ladder: LadderName = "glitch",
    n_calls: int = 15,
    steps: int = 25,
    seed: int = 0,
    x0: Corners | None = None,
    ts_grid: tuple[tuple[float, float], ...] = TS_EVAL,
) -> tuple[Corners, float, list[tuple[Corners, float]]]:
    """Optimize FlowEdit corners (8-D) with ``gp_minimize``."""
    start = x0 or default_corners()
    frozen = dict(start.frozen)
    space = [Real(lo, hi, name=f"d{i}") for i, (lo, hi) in enumerate(corner_search_space())]
    history: list[tuple[Corners, float]] = []
    t0 = time.monotonic()
    best_so_far = float("-inf")

    def objective(vec):
        nonlocal best_so_far
        call_i = len(history) + 1
        corners = Corners.from_vector(vec, frozen=frozen).clip_to_bounds()
        _log(f"[phase0] BO ts-corners call {call_i}/{n_calls} start")
        J, _ = evaluate_corners_on_set(
            model,
            search_latents,
            corners,
            ladder=ladder,
            steps=steps,
            seed=seed,
            ts_grid=ts_grid,
            progress=f"BO ts-corners call {call_i}/{n_calls}",
            progress_every=max(1, len(search_latents) // 3),
        )
        history.append((corners, J))
        if J > best_so_far:
            best_so_far = J
        elapsed = time.monotonic() - t0
        eta = (elapsed / call_i) * (n_calls - call_i)
        _log(
            f"[phase0] BO ts-corners call {call_i}/{n_calls} done "
            f"J={J:.4f} best={best_so_far:.4f} "
            f"elapsed={_fmt_dur(elapsed)} eta={_fmt_dur(eta)}"
        )
        return -J

    res = gp_minimize(
        objective,
        space,
        n_calls=n_calls,
        random_state=seed,
        x0=start.to_vector(),
        n_initial_points=max(5, min(10, n_calls // 4)),
    )
    best = Corners.from_vector(res.x, frozen=frozen).clip_to_bounds()
    best_J = -float(res.fun)
    if history:
        hist_best = max(history, key=lambda t: t[1])
        if hist_best[1] > best_J:
            best, best_J = hist_best[0], hist_best[1]
    return best, best_J, history


def reevaluate_top(
    model,
    val_latents,
    candidates: list[Endpoints],
    *,
    ladder: LadderName,
    steps: int,
    seed: int,
    method: MethodName | str = "?",
    m_grid: tuple[float, ...] = M_EVAL_PHASE0,
) -> tuple[Endpoints, float]:
    scored = []
    n = len(candidates)
    for i, ep in enumerate(candidates, start=1):
        _log(f"[phase0] val {method} candidate {i}/{n}")
        J, _ = evaluate_endpoints_on_set(
            model,
            val_latents,
            ep,
            ladder=ladder,
            steps=steps,
            seed=seed,
            m_grid=m_grid,
            progress=f"val {method} cand {i}/{n}",
            progress_every=max(1, len(val_latents) // 4),
        )
        _log(f"[phase0] val {method} candidate {i}/{n} J={J:.4f}")
        scored.append((ep, J))
    return max(scored, key=lambda t: t[1])


def reevaluate_top_corners(
    model,
    val_latents,
    candidates: list[Corners],
    *,
    ladder: LadderName,
    steps: int,
    seed: int,
) -> tuple[Corners, float]:
    scored = []
    n = len(candidates)
    for i, corners in enumerate(candidates, start=1):
        _log(f"[phase0] val ts-corners candidate {i}/{n}")
        J, _ = evaluate_corners_on_set(
            model,
            val_latents,
            corners,
            ladder=ladder,
            steps=steps,
            seed=seed,
            progress=f"val ts-corners cand {i}/{n}",
            progress_every=max(1, len(val_latents) // 4),
        )
        _log(f"[phase0] val ts-corners candidate {i}/{n} J={J:.4f}")
        scored.append((corners, J))
    return max(scored, key=lambda t: t[1])


def _prepare_library(
    library_root: Path,
    *,
    n_per_cat: int,
    n_val: int,
    seed: int,
) -> tuple[list[LibraryClip], list[LibraryClip], list[LibraryClip], list[str]]:
    all_clips = discover_library(library_root)
    if not all_clips:
        raise FileNotFoundError(f"no audio under {library_root}")
    search_clips = stratified_sample(all_clips, n_per_cat=n_per_cat, seed=seed)
    val_clips = subsample(all_clips, n_val, seed=seed + 1)
    cats = sorted({c.category for c in all_clips})
    return all_clips, search_clips, val_clips, cats


def run_phase0(
    model,
    library_root: Path,
    *,
    out_path: Path | None = None,
    ladder: LadderName = "glitch",
    methods: Sequence[MethodName] = ("flowedit", "rfinv_mp", "rfinv_fp"),
    n_per_cat: int = 1,
    n_val: int = 32,
    n_calls: int = 40,
    top_k: int = 3,
    steps: int = 50,
    seconds: float = 10.0,
    seed: int = 0,
    report_all: bool = False,
    m_grid: tuple[float, ...] = M_EVAL_PHASE0,
) -> dict[str, Any]:
    """Run 1-D Phase 0 and save ``g0_endpoints.json``."""
    library_root = Path(library_root)
    all_clips, search_clips, val_clips, cats = _prepare_library(
        library_root, n_per_cat=n_per_cat, n_val=n_val, seed=seed,
    )
    methods = tuple(methods)
    t_run = time.monotonic()

    _log(f"[phase0] library={library_root} total={len(all_clips)} categories={cats}")
    _log(
        f"[phase0] search={len(search_clips)} "
        f"({sorted({c.category for c in search_clips})}) "
        f"val={len(val_clips)}"
    )
    _log_clip_split("search", search_clips)
    _log_clip_split("val", val_clips)
    _log(
        f"[phase0] plan methods={list(methods)} n_calls={n_calls} "
        f"top_k={top_k} report_all={report_all} steps={steps} seconds={seconds} "
        f"m_grid={list(m_grid)}"
    )

    search_latents = _load_clip_latents(
        model, search_clips, seconds, label="encode search",
    )
    val_latents = _load_clip_latents(
        model, val_clips, seconds, label="encode val",
    )

    g0: dict[str, Endpoints] = {}
    meta: dict[str, Any] = {
        "search_n": len(search_clips),
        "val_n": len(val_clips),
        "steps": steps,
        "seconds": seconds,
        "seed": seed,
        "ladder": ladder,
        "m_grid": list(m_grid),
        "library_root": str(library_root),
        "categories": cats,
        "search_clips": clip_records(search_clips),
        "val_clips": clip_records(val_clips),
        "methods": {},
    }

    for mi, method in enumerate(methods, start=1):
        _log(
            f"[phase0] === method {mi}/{len(methods)}: {method} "
            f"(run elapsed {_fmt_dur(time.monotonic() - t_run)}) ==="
        )
        best, best_J, history = fit_method_endpoints(
            model,
            search_latents,
            method,
            ladder=ladder,
            n_calls=n_calls,
            steps=steps,
            seed=seed,
            m_grid=m_grid,
        )
        ranked = sorted(history, key=lambda t: t[1], reverse=True)
        uniq: list[Endpoints] = []
        seen = set()
        for ep, _J in ranked:
            key = tuple(round(x, 4) for x in ep.to_vector())
            if key in seen:
                continue
            seen.add(key)
            uniq.append(ep)
            if len(uniq) >= top_k:
                break
        if best not in uniq:
            uniq = [best] + uniq[: top_k - 1]

        _log(f"[phase0] re-eval top {len(uniq)} on val set ({method})")
        chosen, val_J = reevaluate_top(
            model,
            val_latents,
            uniq,
            ladder=ladder,
            steps=steps,
            seed=seed,
            method=method,
            m_grid=m_grid,
        )
        g0[method] = chosen
        meta["methods"][method] = {
            "search_J": best_J,
            "val_J": val_J,
            "theta0": chosen.theta0,
            "theta1": chosen.theta1,
        }
        _log(
            f"[phase0] {method} done search_J={best_J:.4f} val_J={val_J:.4f} "
            f"run elapsed={_fmt_dur(time.monotonic() - t_run)}"
        )

    if report_all:
        _log("[phase0] full-library report (expensive)…")
        all_latents = _load_clip_latents(
            model, all_clips, seconds, label="encode full",
        )
        full_scores = {}
        for mi, (method, ep) in enumerate(g0.items(), start=1):
            _log(f"[phase0] full report {mi}/{len(g0)} method={method}")
            J, _ = evaluate_endpoints_on_set(
                model,
                all_latents,
                ep,
                ladder=ladder,
                steps=steps,
                seed=seed,
                m_grid=m_grid,
                progress=f"full {method}",
                progress_every=max(1, len(all_latents) // 8),
            )
            full_scores[method] = J
            _log(f"[phase0] full_J[{method}]={J:.4f}")
        meta["full_library_J"] = full_scores

    path = save_g0(g0, out_path, meta=meta)
    _log(
        f"[phase0] wrote {path} "
        f"total_elapsed={_fmt_dur(time.monotonic() - t_run)}"
    )
    return {"g0": g0, "meta": meta, "path": str(path)}


def run_phase0_ts(
    model,
    library_root: Path,
    *,
    out_path: Path | None = None,
    ladder: LadderName = "glitch",
    n_per_cat: int = 1,
    n_val: int = 8,
    n_calls: int = 15,
    top_k: int = 3,
    steps: int = 25,
    seconds: float = 10.0,
    seed: int = 0,
) -> dict[str, Any]:
    """Run T/S corner Phase 0 (FlowEdit, 8-D) and save ``g0_ts.json``."""
    library_root = Path(library_root)
    all_clips, search_clips, val_clips, cats = _prepare_library(
        library_root, n_per_cat=n_per_cat, n_val=n_val, seed=seed,
    )
    t_run = time.monotonic()
    ts_grid = TS_EVAL

    _log(f"[phase0] ts-corners library={library_root} total={len(all_clips)} categories={cats}")
    _log(
        f"[phase0] search={len(search_clips)} "
        f"({sorted({c.category for c in search_clips})}) "
        f"val={len(val_clips)}"
    )
    _log_clip_split("search", search_clips)
    _log_clip_split("val", val_clips)
    _log(
        f"[phase0] plan ts-corners n_calls={n_calls} top_k={top_k} "
        f"steps={steps} seconds={seconds} ts_grid={list(ts_grid)}"
    )

    search_latents = _load_clip_latents(
        model, search_clips, seconds, label="encode search",
    )
    val_latents = _load_clip_latents(
        model, val_clips, seconds, label="encode val",
    )

    best, best_J, history = fit_corners(
        model,
        search_latents,
        ladder=ladder,
        n_calls=n_calls,
        steps=steps,
        seed=seed,
        ts_grid=ts_grid,
    )
    ranked = sorted(history, key=lambda t: t[1], reverse=True)
    uniq: list[Corners] = []
    seen = set()
    for corners, _J in ranked:
        key = tuple(round(x, 4) for x in corners.to_vector())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(corners)
        if len(uniq) >= top_k:
            break
    if best not in uniq:
        uniq = [best] + uniq[: top_k - 1]

    _log(f"[phase0] re-eval top {len(uniq)} on val set (ts-corners)")
    chosen, val_J = reevaluate_top_corners(
        model,
        val_latents,
        uniq,
        ladder=ladder,
        steps=steps,
        seed=seed,
    )
    meta: dict[str, Any] = {
        "mode": "ts_corners",
        "search_n": len(search_clips),
        "val_n": len(val_clips),
        "steps": steps,
        "seconds": seconds,
        "seed": seed,
        "ladder": ladder,
        "ts_grid": [list(p) for p in ts_grid],
        "library_root": str(library_root),
        "categories": cats,
        "search_clips": clip_records(search_clips),
        "val_clips": clip_records(val_clips),
        "search_J": best_J,
        "val_J": val_J,
        "frozen": chosen.frozen,
    }
    path = save_g0_ts(chosen, out_path, meta=meta)
    _log(
        f"[phase0] wrote {path} search_J={best_J:.4f} val_J={val_J:.4f} "
        f"total_elapsed={_fmt_dur(time.monotonic() - t_run)}"
    )
    return {"corners": chosen, "meta": meta, "path": str(path)}
