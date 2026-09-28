"""CLI: fit endpoint schedules or T/S corners on an environmental library.

Writes ``artifacts/g0_endpoints.json`` (1-D) or ``artifacts/g0_ts.json``
(``--ts-corners``). Does not launch a fit unless you run this module.

Overview
--------
- ``python -m adaptive_music_sa3.fit`` -> 1-D endpoints
- ``python -m adaptive_music_sa3.fit --ts-corners`` -> T/S corners
"""

from __future__ import annotations

import argparse
from pathlib import Path

from adaptive_music_sa3._paths import default_g0_path, default_g0_ts_path, package_root


def main() -> None:
    """Parse argv, load SA3, and run Phase 0."""
    parser = argparse.ArgumentParser(
        description="Phase 0: fit g0 endpoints or T/S corners",
    )
    parser.add_argument(
        "--library-root",
        type=Path,
        default=package_root().parent / "environmental_dataset",
        help="Library root with category subdirs (default: ../environmental_dataset)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON (default: g0_endpoints.json or g0_ts.json)",
    )
    parser.add_argument("--model", default="medium-base", choices=("medium-base", "small-music-base"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--ladder", default="glitch", choices=("glitch", "acoustic_perc", "downtempo"))
    parser.add_argument("--n-calls", type=int, default=None)
    parser.add_argument("--n-per-cat", type=int, default=1,
                        help="Search clips per category (default 1 for speed)")
    parser.add_argument("--n-val", type=int, default=None)
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Diffusion steps (1-D default 50; --ts-corners default 25)",
    )
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--report-all",
        action="store_true",
        help="After 1-D fitting, evaluate mean J on the full library once",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["flowedit", "rfinv_mp", "rfinv_fp"],
    )
    parser.add_argument(
        "--ts-corners",
        action="store_true",
        help="Fit FlowEdit T/S corners (8-D, 5×5 eval) to artifacts/g0_ts.json",
    )
    args = parser.parse_args()

    import torch
    from stable_audio_3.model import StableAudioModel

    from adaptive_music_sa3.fit.library import run_phase0, run_phase0_ts

    device = args.device
    if device is None:
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    print(f"device={device} model={args.model} library={args.library_root}")
    model = StableAudioModel.from_pretrained(args.model, device=device)

    if args.ts_corners:
        out = args.out or default_g0_ts_path()
        run_phase0_ts(
            model,
            args.library_root,
            out_path=out,
            ladder=args.ladder,
            n_per_cat=args.n_per_cat,
            n_val=args.n_val if args.n_val is not None else 8,
            n_calls=args.n_calls if args.n_calls is not None else 15,
            steps=args.steps if args.steps is not None else 25,
            seconds=args.seconds,
            seed=args.seed,
        )
        return

    out = args.out or default_g0_path()
    run_phase0(
        model,
        args.library_root,
        out_path=out,
        ladder=args.ladder,
        methods=tuple(args.methods),  # type: ignore[arg-type]
        n_per_cat=args.n_per_cat,
        n_val=args.n_val if args.n_val is not None else 32,
        n_calls=args.n_calls if args.n_calls is not None else 40,
        steps=args.steps if args.steps is not None else 50,
        seconds=args.seconds,
        seed=args.seed,
        report_all=args.report_all,
    )


if __name__ == "__main__":
    main()
