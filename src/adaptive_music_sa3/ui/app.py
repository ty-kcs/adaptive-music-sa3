"""Gradio UI for adaptive-music-sa3.

Upload audio, optionally run Phase-1 T/S corner refine, set T and S
independently, and generate with FlowEdit corners plus DSP metrics.

Overview
--------
- Input: browser controls
- Output: wav + metadata text
- ``--demo`` / ``ADAPTIVE_MUSIC_DEMO=1``: client-facing layout (no Optimize)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch

from adaptive_music_sa3._paths import default_g0_ts_path
from adaptive_music_sa3.control.corners import load_g0_ts
from adaptive_music_sa3.control.prompts import LADDERS
from adaptive_music_sa3.fit.clip import optimize_audio
from adaptive_music_sa3.generate import generate, musicality_from_TS, wav_to_numpy_stereo
from adaptive_music_sa3.sa3 import load_latent, load_latent_from_wav

_STATE: dict[str, Any] = {
    "model": None,
    "model_name": None,
    "device": None,
    "latent": None,
    "seconds_total": None,
    "sr_in": None,
    "config": None,
    "demo": False,
}


def _resolve_device(device: str | None) -> str:
    if device and device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def ensure_model(model_name: str, device: str | None = None):
    from stable_audio_3.model import StableAudioModel

    dev = _resolve_device(device)
    if _STATE["model"] is not None and _STATE["model_name"] == model_name and _STATE["device"] == dev:
        return _STATE["model"]
    _STATE["config"] = None
    _STATE["latent"] = None
    print(f"[app] loading {model_name} on {dev}…")
    model = StableAudioModel.from_pretrained(model_name, device=dev)
    _STATE["model"] = model
    _STATE["model_name"] = model_name
    _STATE["device"] = dev
    return model


def _clear_audio_cache() -> None:
    """Drop latent / optimize config when the upload changes."""
    _STATE["latent"] = None
    _STATE["seconds_total"] = None
    _STATE["config"] = None


def on_audio_change(_audio):
    """Invalidate session audio state when the input file changes."""
    _clear_audio_cache()
    if _STATE.get("demo"):
        return ""
    return "", ""


def _load_upload(audio, model, seconds: float):
    """Encode Gradio audio (filepath or ``(sr, ndarray)``) to a latent."""
    if audio is None:
        raise gr.Error("Upload an audio file first")
    if isinstance(audio, tuple) and len(audio) == 2:
        sr, data = audio
        data = np.asarray(data)
        if data.ndim == 1:
            wav = torch.from_numpy(data.astype(np.float32)).unsqueeze(0)
        else:
            wav = torch.from_numpy(data.astype(np.float32).T)
        peak = wav.abs().max().clamp(min=1e-6)
        wav = wav / peak
        return load_latent_from_wav(model, wav, int(sr), seconds=seconds)
    path = Path(audio)
    if not path.is_file():
        raise gr.Error(f"Audio file not found: {path}")
    return load_latent(model, path, seconds=seconds)


def _corners_for_generate():
    """Phase-1 corners if present, else loaded ``g0_ts``."""
    cfg = _STATE["config"]
    if cfg is not None:
        return cfg.corners, ""
    path = default_g0_ts_path()
    corners = load_g0_ts(path if path.is_file() else None)
    if path.is_file():
        note = f"(no optimize — using {path})\n"
    else:
        note = (
            "(no optimize — g0_ts.json missing; seeded from 1-D FlowEdit g0)\n"
        )
    return corners, note


def _ts_figure(T: float, S: float, *, show_m: bool):
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("T (temporal)")
    ax.set_ylabel("S (spectral/harmonic)")
    if show_m:
        M = musicality_from_TS(T, S)
        ax.set_title(f"T={T:.2f}  S={S:.2f}  (M={M:.2f} display only)")
    else:
        ax.set_title(f"T={T:.2f}  S={S:.2f}")
    ax.grid(True, alpha=0.3)
    ax.scatter([T], [S], s=80, c="#2a78d6", zorder=3)
    fig.tight_layout()
    return fig


def on_ts_change(T: float, S: float):
    M = musicality_from_TS(T, S)
    return M, _ts_figure(T, S, show_m=True)


def on_ts_change_demo(T: float, S: float):
    return _ts_figure(T, S, show_m=False)


def do_optimize(
    audio,
    model_name: str,
    ladder: str,
    prompt_mode: str,
    fixed_prompt: str,
    seconds: float,
    n_local: int,
    patience: int,
    steps_search: int,
    seed: int,
):
    model = ensure_model(model_name)
    latent, secs = _load_upload(audio, model, seconds)
    _STATE["latent"] = latent
    _STATE["seconds_total"] = secs

    g0_path = default_g0_ts_path()
    g0_note = (
        f"g0_ts: {g0_path}" if g0_path.is_file()
        else "g0_ts: seeded from 1-D g0 (run python -m adaptive_music_sa3.fit --ts-corners)"
    )

    cfg = optimize_audio(
        model,
        latent,
        secs,
        ladder=ladder,  # type: ignore[arg-type]
        prompt_mode=prompt_mode,  # type: ignore[arg-type]
        fixed_prompt=fixed_prompt or None,
        g0_ts_path=g0_path if g0_path.is_file() else None,
        n_local_calls=int(n_local),
        patience=int(patience),
        steps_search=int(steps_search),
        seed=int(seed),
    )
    _STATE["config"] = cfg
    meta = cfg.to_meta()
    text = (
        f"{g0_note}\n"
        f"method=flowedit (T/S corners)\n"
        f"J={meta['J']:.4f}\n"
        f"ρ(T,tc)={meta['spearman_T_tc']:.3f}  ρ(S,hc)={meta['spearman_S_hc']:.3f}\n"
        f"ρ(T,hc)={meta['spearman_T_hc']:.3f}  ρ(S,tc)={meta['spearman_S_tc']:.3f}\n"
        f"ρ(max(T,S),drift)={meta['spearman_maxTS_drift']:.3f}\n"
        f"drift hinge={meta['drift_hinge']:.4f}\n"
        f"viol T→tc={meta['viol_T_tc']}  viol S→hc={meta['viol_S_hc']}\n"
        f"theta00={json.dumps(meta['theta00'])}\n"
        f"theta10={json.dumps(meta['theta10'])}\n"
        f"theta01={json.dumps(meta['theta01'])}\n"
        f"theta11={json.dumps(meta['theta11'])}\n"
        f"frozen={json.dumps(meta['frozen'])}\n"
        f"local BO calls={meta['n_local_calls']} "
        f"(max={int(n_local)}, patience={int(patience)}, 3×3 grid)"
    )
    return text


def _format_generate_meta_demo(result) -> str:
    return (
        "Prompt\n"
        f"  {result.prompt}\n"
        "\n"
        "Musicality (DSP)\n"
        f"  Temporal Regularity:  {result.scores.tc:.3f}\n"
        f"  Spectral / Harmonic Coherence:  {result.scores.hc:.3f}\n"
        "\n"
        f"Drift (latent distance from source):  {result.drift:.4f}"
    )


def do_generate(
    audio,
    model_name: str,
    T: float,
    S: float,
    ladder: str,
    prompt_mode: str,
    fixed_prompt: str,
    seconds: float,
    steps: int,
    seed: int,
):
    model = ensure_model(model_name)
    latent, secs = _load_upload(audio, model, seconds)
    _STATE["latent"] = latent
    _STATE["seconds_total"] = secs

    corners, note = _corners_for_generate()
    result = generate(
        model,
        latent,
        secs,
        corners,
        T=float(T),
        S=float(S),
        ladder=ladder,  # type: ignore[arg-type]
        prompt_mode=prompt_mode,  # type: ignore[arg-type]
        fixed_prompt=fixed_prompt or None,
        steps=int(steps),
        seed=int(seed),
    )
    audio_out = (result.sr, wav_to_numpy_stereo(result.wav))
    if _STATE.get("demo"):
        return audio_out, _format_generate_meta_demo(result)
    meta = (
        f"{note}"
        f"method={result.method} (T/S corners)\n"
        f"T={result.T:.2f} S={result.S:.2f} M={result.m:.2f} (M is display-only)\n"
        f"params={json.dumps(result.params)}\n"
        f"drift={result.drift:.4f}\n"
        f"tc={result.scores.tc:.3f} hc={result.scores.hc:.3f} M̂={result.scores.m_hat:.3f}\n"
        f"prompt={result.prompt}\n"
        f"theta00={json.dumps(result.meta['theta00'])}\n"
        f"theta10={json.dumps(result.meta['theta10'])}\n"
        f"theta01={json.dumps(result.meta['theta01'])}\n"
        f"theta11={json.dumps(result.meta['theta11'])}"
    )
    return audio_out, meta


def build_ui(*, demo: bool = False) -> gr.Blocks:
    _STATE["demo"] = bool(demo)
    title = "adaptive-music-sa3" + (" (demo)" if demo else "")
    if demo:
        intro = (
            "## adaptive-music-sa3\n"
            "Upload ambient audio → set **T** (temporal regularity) and "
            "**S** (spectral / harmonic coherence) → **Generate**."
        )
    else:
        intro = (
            "## adaptive-music-sa3\n"
            "Upload ambient audio → set **T** and **S** independently → **Generate**.\n"
            "Uses FlowEdit **four-corner** map from Phase-0 `g0_ts.json` "
            "(prompt = genre + T phrase + S phrase; θ bilinear in T and S).\n"
            "Optional **Optimize** locally refines those corners on this clip "
            "(3×3 grid, independence objective \\(J\\)).\n"
            r"\(M = 0.5T + 0.5S\) is shown only; it is not the control."
        )

    with gr.Blocks(title=title) as blocks:
        gr.Markdown(intro)
        with gr.Row():
            with gr.Column():
                audio_in = gr.Audio(
                    type="filepath",
                    label="Input audio",
                    sources=["upload"],
                )
                model_name = gr.Dropdown(
                    choices=["medium-base", "small-music-base"],
                    value="medium-base",
                    label="Model",
                )
                ladder = gr.Dropdown(
                    choices=list(LADDERS.keys()),
                    value="glitch",
                    label="Genre prefix",
                )
                prompt_mode = gr.Radio(
                    choices=["ladder", "fixed"],
                    value="ladder",
                    label="Prompt mode",
                )
                fixed_prompt = gr.Textbox(
                    label="Fixed prompt (when mode=fixed)",
                    lines=2,
                    placeholder="Leave empty for T/S ladder phrases",
                )
                with gr.Row():
                    T = gr.Slider(0, 1, value=0.5, step=0.01, label="T (temporal regularity)")
                    S = gr.Slider(0, 1, value=0.5, step=0.01, label="S (spectral/harmonic)")
                M_out = None
                if not demo:
                    M_out = gr.Number(label="M = 0.5T+0.5S (display only)", interactive=False)
                ts_plot = gr.Plot(label="(T, S) map")
                with gr.Accordion("Advanced", open=False):
                    seconds = gr.Slider(4, 20, value=10, step=1, label="Seconds")
                    steps_final = gr.Slider(20, 100, value=50, step=5, label="Generate steps")
                    seed = gr.Number(value=0, precision=0, label="Seed")
                    steps_search = None
                    n_local = None
                    patience = None
                    if not demo:
                        steps_search = gr.Slider(10, 50, value=25, step=5, label="Optimize steps")
                        n_local = gr.Slider(3, 30, value=12, step=1, label="Local BO max trials")
                        patience = gr.Slider(
                            0, 10, value=3, step=1,
                            label="Local BO patience (0=disabled)",
                        )
                with gr.Row():
                    btn_opt = None
                    if not demo:
                        btn_opt = gr.Button("Optimize in this setting", variant="secondary")
                    btn_gen = gr.Button("Generate", variant="primary")
            with gr.Column():
                audio_out = gr.Audio(label="Output", type="numpy")
                opt_meta = None
                if not demo:
                    opt_meta = gr.Textbox(label="Optimize meta", lines=14)
                gen_meta = gr.Textbox(label="Generate meta", lines=10 if demo else 14)

        if demo:
            T.change(on_ts_change_demo, [T, S], [ts_plot])
            S.change(on_ts_change_demo, [T, S], [ts_plot])
            blocks.load(lambda: on_ts_change_demo(0.5, 0.5), outputs=[ts_plot])
            audio_in.change(on_audio_change, [audio_in], [gen_meta])
        else:
            T.change(on_ts_change, [T, S], [M_out, ts_plot])
            S.change(on_ts_change, [T, S], [M_out, ts_plot])
            blocks.load(lambda: on_ts_change(0.5, 0.5), outputs=[M_out, ts_plot])
            audio_in.change(on_audio_change, [audio_in], [opt_meta, gen_meta])
            assert btn_opt is not None and opt_meta is not None
            assert steps_search is not None and n_local is not None and patience is not None
            btn_opt.click(
                do_optimize,
                [
                    audio_in, model_name, ladder, prompt_mode, fixed_prompt,
                    seconds, n_local, patience, steps_search, seed,
                ],
                [opt_meta],
            )

        btn_gen.click(
            do_generate,
            [
                audio_in, model_name, T, S, ladder, prompt_mode, fixed_prompt,
                seconds, steps_final, seed,
            ],
            [audio_out, gen_meta],
        )
    return blocks


def _demo_enabled(argv_demo: bool) -> bool:
    env = os.environ.get("ADAPTIVE_MUSIC_DEMO", "").strip().lower()
    return bool(argv_demo) or env in ("1", "true", "yes")


def main():
    parser = argparse.ArgumentParser(description="adaptive-music-sa3 Gradio UI")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Client-facing UI: simple generate meta, no Optimize",
    )
    args, _unknown = parser.parse_known_args()
    demo = _demo_enabled(args.demo)
    if demo:
        print("[app] demo mode (no Optimize; short generate meta)", flush=True)
    blocks = build_ui(demo=demo)
    blocks.queue().launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
