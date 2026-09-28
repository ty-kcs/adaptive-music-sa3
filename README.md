# adaptive-music-sa3

*English below / [日本語はこちら](#日本語)*

T/S uses four FlowEdit corners and independent prompts. Details:
[docs/optimization.md](docs/optimization.md).

## Setup

Python 3.10+, NVIDIA GPU, uv, and Hugging
Face access to SA3 `-base` checkpoints (`medium-base` default).

[Stable Audio 3](https://github.com/Stability-AI/stable-audio-3) must
be cloned **next to** this repo (`../stable-audio-3`). `medium-base`
needs flash-attn — see that README, then `uv sync --inexact` so uv
does not remove it.

```bash
git clone https://github.com/Stability-AI/stable-audio-3.git
git clone https://github.com/ty-kcs/adaptive-music-sa3.git

cd stable-audio-3
uv sync
# install flash-attn for your CUDA / torch / Python
uv sync --inexact

cd ../adaptive-music-sa3
uv sync
uv run huggingface-cli login
```



## Run

```bash
uv run python -m adaptive_music_sa3 --demo
```

## Phase 0 / Phase 1

```bash
# Phase 0 — T/S corners → artifacts/g0_ts.json  (hours)
uv run python -m adaptive_music_sa3.fit --device cuda --ts-corners \
  --library-root /path/to/ambient/library

# Phase 1 — optional per-clip T/S refine: Optimize in the UI
#   (3×3 grid per trial; uses artifacts/g0_ts.json)
```

Library root: category folders of audio (`generated` is skipped).
Defaults: 5×5 eval grid, `--n-calls 15 --steps 25 --n-val 8`.

## How it works

Commanded `(T, S)` picks `{genre}, {T phrase}, {S phrase}` and
bilinear-interpolates FlowEdit `t_start` / `tgt_cfg` at four corners.
DSP measures `tc` (pulse) and `hc` (tonal / harmonic).
See [docs/optimization.md](docs/optimization.md).

---



# 日本語

T/S は FlowEdit の四隅と独立プロンプト。説明:
[docs/optimization.md](docs/optimization.md)（英語）。

## セットアップ

Python 3.10+、NVIDIA GPU、[uv](https://docs.astral.sh/uv/)、Hugging
Face の SA3 `-base`（既定 `medium-base`）。ライセンス同意のあと
一度ログイン。

[Stable Audio 3](https://github.com/Stability-AI/stable-audio-3) を
このリポジトリの **隣** に置く（`../stable-audio-3`）。
`medium-base` は flash-attn が必要 — そちらの README を見て入れたあと
`uv sync --inexact`。

```bash
git clone https://github.com/Stability-AI/stable-audio-3.git
git clone https://github.com/ty-kcs/adaptive-music-sa3.git

cd stable-audio-3
uv sync
# 自分の CUDA / torch / Python 向けに flash-attn を入れる
uv sync --inexact

cd ../adaptive-music-sa3
uv sync
uv run huggingface-cli login
```



## 実行

```bash
uv run python -m adaptive_music_sa3 --demo
```


[http://localhost:7860](http://localhost:7860)（`0.0.0.0` で待ち受け。
認証なし）。別マシンから: `ssh -L 7860:localhost:7860 user@host`。

## Phase 0 / Phase 1

```bash
# Phase 0 — T/S 四隅 → artifacts/g0_ts.json（数時間）
uv run python -m adaptive_music_sa3.fit --device cuda --ts-corners \
  --library-root /path/to/ambient/library

# Phase 1 — 任意: そのaudio・モデル設定に対するOptimize（3×3、g0_ts.json を局所 refine）
```

ライブラリはカテゴリ別フォルダ（`generated` は除外）。
既定: 5×5 評価格子、`--n-calls 15 --steps 25 --n-val 8`。

## 仕組み

指令 `(T, S)` で `{ジャンル}, {T 句}, {S 句}` を選び、FlowEdit の
`t_start` / `tgt_cfg` を四隅から bilinear 補間。DSP は `tc`（パルス）
と `hc`（調性 / 和声）。
[docs/optimization.md](docs/optimization.md)。
