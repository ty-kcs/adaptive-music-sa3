"""DSP proxies for temporal regularity and harmonic coherence.

``tc`` / ``hc`` are simple averages of a few [0, 1] features;
``M_hat = 0.5 * tc + 0.5 * hc``.

Overview
--------
- Input: mono/stereo waveform + sample rate
- Temporal: pulse clarity + onset-interval regularity (on short overlapping windows)
- Harmonic: peak tonalness + chroma concentration + key clarity (on short windows,
  ignoring quiet frames)
- Output: ``MusicalityScores`` for BO / UI
"""

from __future__ import annotations

from dataclasses import dataclass, field

import librosa
import numpy as np
from numpy.typing import NDArray

# Spectrogram
HOP = 512
N_FFT = 2048

# Pulse: keep lags that correspond to this BPM range (empirical)
BPM_MIN = 30.0
BPM_MAX = 300.0

# Harmonic: ignore frames this many dB below the loudest frame (empirical)
SILENCE_DB = 40.0
# Harmonic: a bin is a "peak" if it is this many dB above the frame median (empirical)
PEAK_DB = 12.0

# Analysis window lengths in seconds (empirical; keep TEMP longer than HARM)
TEMP_WIN_S = 5.0
TEMP_HOP_S = 2.5
HARM_WIN_S = 2.0
HARM_HOP_S = 1.0

# Krumhansl–Kessler major / minor templates (12 pitch classes).
# Shifting by 0..11 gives all 24 major/minor keys.
_KK_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    dtype=np.float64,
)
_KK_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    dtype=np.float64,
)


@dataclass
class MusicalityScores:
    """DSP estimate of musicality for one clip.

    Attributes
    ----------
    tc : float
        Temporal regularity in ``[0, 1]``.
    hc : float
        Harmonic / spectral coherence in ``[0, 1]``.
    m_hat : float
        ``0.5 * tc + 0.5 * hc``.
    components : dict[str, float]
        Per-feature values that were averaged into ``tc`` / ``hc``.
    """

    tc: float
    hc: float
    m_hat: float
    components: dict[str, float] = field(default_factory=dict)


def _to_mono(y: NDArray[np.floating], sr: int) -> tuple[NDArray[np.floating], int]:
    """Convert to mono float64 and peak-normalize to about [-1, 1]."""
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 2:
        # (channels, time) if first dim is 1 or 2, else (time, channels)
        y = y.mean(axis=0 if y.shape[0] <= 2 else 1)
    peak = np.max(np.abs(y)) + 1e-12
    return y / peak, sr


def _window_starts(n: int, win: int, hop: int) -> list[int]:
    """Start indices for overlapping windows. One window if the clip is short."""
    if n <= int(win * 1.2):
        return [0]
    return list(range(0, n - win + 1, hop))


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


# ---------------------------------------------------------------------------
# Temporal features
# ---------------------------------------------------------------------------


def _pulse(onset_env: NDArray[np.floating], sr_env: float) -> float:
    """How periodic the onset envelope is inside the BPM band (autocorr peak)."""
    if onset_env.size < 8:
        return 0.0
    x = onset_env - onset_env.mean()
    corr = np.correlate(x, x, mode="full")
    corr = corr[corr.size // 2 :]  # lag >= 0
    if corr[0] <= 1e-12:
        return 0.0
    corr = corr / corr[0]
    # lag (frames) for BPM_MAX .. BPM_MIN
    min_lag = max(1, int(sr_env * 60.0 / BPM_MAX))
    max_lag = min(len(corr) - 1, int(sr_env * 60.0 / BPM_MIN))
    if max_lag <= min_lag:
        return 0.0
    return float(np.clip(np.max(corr[min_lag : max_lag + 1]), 0.0, 1.0))


def _onset_regularity(onset_env: NDArray[np.floating], sr: int) -> float:
    """How even the gaps between detected onsets are: exp(-CV of intervals)."""
    frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=HOP, units="frames"
    )
    if len(frames) < 4:
        return 0.0
    times = librosa.frames_to_time(frames, sr=sr, hop_length=HOP)
    gaps = np.diff(times)
    gaps = gaps[gaps > 1e-3]
    if gaps.size < 3:
        return 0.0
    cv = float(np.std(gaps) / (np.mean(gaps) + 1e-12))
    return float(np.clip(np.exp(-cv), 0.0, 1.0))


def _score_temporal(y: NDArray[np.floating], sr: int) -> dict[str, float]:
    """Average pulse and onset regularity over TEMP_WIN_S windows."""
    n = y.shape[-1]
    win = int(TEMP_WIN_S * sr)
    hop = max(1, int(TEMP_HOP_S * sr))
    pulses: list[float] = []
    onsets: list[float] = []
    for start in _window_starts(n, win, hop):
        seg = y[start : start + min(win, n - start)]
        env = librosa.onset.onset_strength(y=seg, sr=sr, hop_length=HOP)
        pulses.append(_pulse(env, sr / HOP))
        onsets.append(_onset_regularity(env, sr))
    return {"pulse": _mean(pulses), "onset_regularity": _mean(onsets)}


# ---------------------------------------------------------------------------
# Harmonic features
# ---------------------------------------------------------------------------


def _loud_frames(S: NDArray[np.floating]) -> NDArray[np.bool_]:
    """True where frame energy is within SILENCE_DB of the peak frame."""
    energy = np.sum(S * S, axis=0)
    peak = float(np.max(energy)) + 1e-12
    return energy > peak * (10.0 ** (-SILENCE_DB / 10.0))


def _spectral_tonalness(S: NDArray[np.floating], loud: NDArray[np.bool_]) -> float:
    """Fraction of loud-frame energy that sits in narrow spectral peaks."""
    if S.size == 0 or not np.any(loud):
        return 0.0
    Sa = S[:, loud]
    floor = np.median(Sa, axis=0, keepdims=True) + 1e-12
    peaks = Sa > (floor * (10.0 ** (PEAK_DB / 20.0)))
    return float(np.clip((Sa * peaks).sum() / (Sa.sum() + 1e-12), 0.0, 1.0))


def _chroma_concentration(chroma: NDArray[np.floating]) -> float:
    """1 - normalized entropy of a 12-bin chroma histogram (peakier => higher)."""
    p = chroma / (chroma.sum() + 1e-12)
    ent = -np.sum(p * np.log(p + 1e-12))
    return float(np.clip(1.0 - ent / np.log(12.0), 0.0, 1.0))


def _key_clarity(chroma: NDArray[np.floating]) -> float:
    """Best correlation of chroma with any of the 24 major/minor KK profiles."""
    p = chroma - chroma.mean()
    p = p / (np.linalg.norm(p) + 1e-12)
    best = 0.0
    for profile in (_KK_MAJOR, _KK_MINOR):
        base = profile - profile.mean()
        base = base / (np.linalg.norm(base) + 1e-12)
        for shift in range(12):
            best = max(best, float(np.dot(p, np.roll(base, shift))))
    return float(np.clip(best, 0.0, 1.0))


def _score_harmonic(y: NDArray[np.floating], sr: int) -> dict[str, float]:
    """Average harmonic features over HARM_WIN_S windows (loud frames only)."""
    n = y.shape[-1]
    win = int(HARM_WIN_S * sr)
    hop = max(1, int(HARM_HOP_S * sr))
    tonals: list[float] = []
    chromas: list[float] = []
    keys: list[float] = []

    for start in _window_starts(n, win, hop):
        seg = y[start : start + min(win, n - start)]
        S = np.abs(librosa.stft(seg, n_fft=N_FFT, hop_length=HOP))
        loud = _loud_frames(S)
        if not np.any(loud):
            tonals.append(0.0)
            chromas.append(0.0)
            keys.append(0.0)
            continue

        tonals.append(_spectral_tonalness(S, loud))

        # One chroma vector per window = mean over loud frames
        chroma = librosa.feature.chroma_cqt(y=seg, sr=sr, hop_length=HOP)
        n_fr = min(chroma.shape[1], loud.shape[0])
        loud_c = loud[:n_fr]
        if not np.any(loud_c):
            chromas.append(0.0)
            keys.append(0.0)
            continue
        p = chroma[:, :n_fr][:, loud_c].mean(axis=1)
        chromas.append(_chroma_concentration(p))
        keys.append(_key_clarity(p))

    return {
        "spectral_tonalness": _mean(tonals),
        "chroma_concentration": _mean(chromas),
        "key_clarity": _mean(keys),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _score_segment(y: NDArray[np.floating], sr: int) -> MusicalityScores:
    """Score one (already mono, peak-normalized) segment."""
    temporal = _score_temporal(y, sr)
    harmonic = _score_harmonic(y, sr)
    tc = _mean(list(temporal.values()))
    hc = _mean(list(harmonic.values()))
    return MusicalityScores(
        tc=tc,
        hc=hc,
        m_hat=0.5 * tc + 0.5 * hc,
        components={**temporal, **harmonic},
    )


def score_musicality(
    y: NDArray[np.floating] | object,
    sr: int,
    *,
    window_s: float = 15.0,
) -> MusicalityScores:
    """Estimate ``tc``, ``hc``, and ``M_hat`` for a waveform.

    Parameters
    ----------
    y : array-like or torch tensor
        Mono or stereo waveform.
    sr : int
        Sample rate in Hz.
    window_s : float
        If the clip is longer than this, average scores over overlapping
        outer windows of this length. Inner TEMP/HARM windows are fixed.

    Returns
    -------
    MusicalityScores
        ``tc``, ``hc``, ``m_hat``, and feature components.
    """
    if hasattr(y, "detach"):
        y = y.detach().cpu().numpy()
    y, sr = _to_mono(np.asarray(y, dtype=np.float64), sr)

    n = y.shape[-1]
    win = int(window_s * sr)
    if n <= win * 1.2:
        return _score_segment(y, sr)

    hops = max(1, win // 2)
    tcs: list[float] = []
    hcs: list[float] = []
    comp_acc: dict[str, list[float]] = {}
    for start in range(0, n - win + 1, hops):
        sc = _score_segment(y[start : start + win], sr)
        tcs.append(sc.tc)
        hcs.append(sc.hc)
        for k, v in sc.components.items():
            comp_acc.setdefault(k, []).append(v)

    tc = _mean(tcs)
    hc = _mean(hcs)
    components = {k: _mean(vs) for k, vs in comp_acc.items()}
    return MusicalityScores(tc=tc, hc=hc, m_hat=0.5 * tc + 0.5 * hc, components=components)


# Alias used by musicality_validation/build_map.py
_score_window = _score_segment
