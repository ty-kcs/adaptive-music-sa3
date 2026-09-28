"""Audio encode / decode / drift helpers (vendored from sa3-inversion bench_common).

Minimal wav <-> latent utilities and relative latent distance for measuring
how far an edit moved from the source.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from stable_audio_3.inference.audio_utils import prepare_audio


def _read_audio_file(path: Path) -> tuple[torch.Tensor, int]:
    """Load ``(channels, samples)`` float waveform via torchaudio or soundfile.

    Parameters
    ----------
    path : Path
        Audio file path (wav / aiff / flac / mp3 / … depending on backends).

    Returns
    -------
    wav : torch.Tensor
        Shape ``(C, T)``, float32.
    sample_rate : int
        Native sample rate.
    """
    try:
        wav, in_sr = torchaudio.load(str(path))
        return wav, int(in_sr)
    except Exception:
        data, in_sr = sf.read(str(path), always_2d=True, dtype="float32")
        wav = torch.from_numpy(np.asarray(data, dtype=np.float32).T)
        return wav, int(in_sr)


def rel_err(output: torch.Tensor, reference: torch.Tensor) -> float:
    """Return ``||output - reference|| / ||reference||``."""
    return ((output - reference).norm() / reference.norm()).item()


def drift(output: torch.Tensor, source: torch.Tensor) -> float:
    """Relative latent distance from the source after editing."""
    return rel_err(output, source)


def load_latent(model, path: Path | str, seconds: float = 10.0):
    """Peak-normalize an audio file and encode to latent.

    Parameters
    ----------
    model
        Loaded Stable Audio model.
    path : Path | str
        Audio file path (wav / aiff / flac / mp3 / …).
    seconds : float
        Crop length in seconds.

    Returns
    -------
    latent : torch.Tensor
        Encoded latent.
    seconds_total : float
        Actual duration used after downsampling alignment.
    """
    path = Path(path)
    device = model.device
    sr = model.model.sample_rate
    ds = model.same.downsampling_ratio

    wav, in_sr = _read_audio_file(path)
    available = int(wav.shape[-1] / in_sr * sr)
    n = max((min(int(seconds * sr), available) // ds) * ds, ds)
    audio = prepare_audio(
        wav,
        in_sr=in_sr,
        target_sr=sr,
        target_length=n,
        target_channels=2,
        device=device,
    )
    audio = audio / audio.abs().max().clamp(min=1e-6)
    latent = model.same.encode(audio.to(next(model.same.parameters()).dtype))
    return latent, n / sr


def load_latent_from_wav(model, wav: torch.Tensor, in_sr: int, seconds: float = 10.0):
    """Encode an in-memory waveform (e.g. Gradio upload) to latent.

    Parameters
    ----------
    wav : torch.Tensor
        Shape ``(C, T)`` or ``(T,)``.
    in_sr : int
        Native sample rate of ``wav``.
    seconds : float
        Crop length in seconds.

    Returns
    -------
    latent, seconds_total
        Same as ``load_latent``.
    """
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    device = model.device
    sr = model.model.sample_rate
    ds = model.same.downsampling_ratio
    available = int(wav.shape[-1] / in_sr * sr)
    n = max((min(int(seconds * sr), available) // ds) * ds, ds)
    audio = prepare_audio(
        wav,
        in_sr=in_sr,
        target_sr=sr,
        target_length=n,
        target_channels=2,
        device=device,
    )
    audio = audio / audio.abs().max().clamp(min=1e-6)
    latent = model.same.encode(audio.to(next(model.same.parameters()).dtype))
    return latent, n / sr


def decode_latent(model, latent: torch.Tensor) -> torch.Tensor:
    """Decode latent to CPU float wav ``(channels, samples)``."""
    with torch.inference_mode():
        dtype = next(model.same.parameters()).dtype
        wav = model.same.decode(latent.to(dtype)).float().clamp(-1, 1).cpu()
    return wav[0] if wav.dim() == 3 else wav


def write_wav(path: Path | str, wav: torch.Tensor, sr: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    torchaudio.save(str(path), wav.cpu(), sr)
