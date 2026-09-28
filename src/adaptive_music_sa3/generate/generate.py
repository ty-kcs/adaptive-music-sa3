"""Single-shot generation at commanded ``(T, S)``.

Thin API for Gradio: bilinear FlowEdit corners plus independent T/S
prompts. ``M = 0.5 T + 0.5 S`` is reported only, not used as a control.

Overview
--------
- Input: model + latent + ``Corners`` + T, S
- Output: ``GenerateResult``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch

from adaptive_music_sa3.control.corners import Corners
from adaptive_music_sa3.control.prompts import LadderName, prompt_for
from adaptive_music_sa3.dsp.musicality import MusicalityScores, score_musicality
from adaptive_music_sa3.eval.engine import edit_at_ts
from adaptive_music_sa3.sa3 import decode_latent


@dataclass
class GenerateResult:
    wav: torch.Tensor  # (C, T) cpu
    sr: int
    m: float
    T: float
    S: float
    prompt: str
    params: dict[str, float]
    drift: float
    scores: MusicalityScores
    method: str
    meta: dict[str, Any]


def musicality_from_TS(T: float, S: float) -> float:
    """Return display-only ``M = 0.5 T + 0.5 S``."""
    return 0.5 * float(T) + 0.5 * float(S)


@torch.inference_mode()
def generate(
    model,
    latent: torch.Tensor,
    seconds_total: float,
    corners: Corners,
    *,
    T: float = 0.5,
    S: float = 0.5,
    ladder: LadderName = "glitch",
    prompt_mode: Literal["ladder", "fixed"] = "ladder",
    fixed_prompt: str | None = None,
    steps: int = 50,
    seed: int = 0,
) -> GenerateResult:
    """Edit one clip at independent ``(T, S)``.

    Parameters
    ----------
    corners : Corners
        Four-corner FlowEdit map (Phase-0 ``g0_ts`` or Phase-1 refine).
    T, S : float
        Temporal / spectral knobs. Prompt rungs and FlowEdit params
        are chosen separately (not collapsed to ``M``).
    """
    T, S = float(T), float(S)
    M = musicality_from_TS(T, S)
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
        seed=seed,
        return_latent=True,
    )
    assert lv.latent is not None
    wav = decode_latent(model, lv.latent)
    sr = int(model.model.sample_rate)
    scores = score_musicality(wav.numpy(), sr)

    meta = {
        "method": "flowedit",
        "T": T,
        "S": S,
        "M": float(M),
        "prompt": prompt,
        "params": lv.params,
        "drift": lv.drift,
        "tc": scores.tc,
        "hc": scores.hc,
        "m_hat": scores.m_hat,
        "steps": steps,
        "theta00": corners.theta00,
        "theta10": corners.theta10,
        "theta01": corners.theta01,
        "theta11": corners.theta11,
        "frozen": corners.frozen,
    }
    return GenerateResult(
        wav=wav,
        sr=sr,
        m=M,
        T=T,
        S=S,
        prompt=prompt,
        params=lv.params,
        drift=lv.drift,
        scores=scores,
        method="flowedit",
        meta=meta,
    )


def wav_to_numpy_stereo(wav: torch.Tensor) -> np.ndarray:
    """Convert a torch wav to float32 ``(T, C)`` for Gradio."""
    w = wav.detach().cpu().float()
    if w.dim() == 1:
        w = w.unsqueeze(0)
    return w.transpose(0, 1).numpy()
