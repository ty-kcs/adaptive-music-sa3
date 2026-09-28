"""Preset musicality prompt ladders (1-D) and T/S phrase tables.

1-D ladders (glitch / acoustic_perc / downtempo) stay for the existing
curve path. T/S control concatenates a genre prefix with independent
temporal and spectral phrases.

Overview
--------
- ``prompt_for(ladder, m=...)`` -> 1-D DLT ladder step
- ``prompt_for(ladder, T=..., S=...)`` -> ``prefix, T_phrase, S_phrase``
"""

from __future__ import annotations

from typing import Literal, overload

LadderName = Literal["glitch", "acoustic_perc", "downtempo"]

LEVELS: tuple[float, ...] = (0.00, 0.25, 0.50, 0.75, 1.00)

# Genre prefixes shared by 1-D ladders and T/S composition.
PREFIXES: dict[LadderName, str] = {
    "glitch": "glitch electronica",
    "acoustic_perc": "acoustic percussion free improvisation",
    "downtempo": "downtempo acoustic abstract hiphop",
}

# Temporal / pulse wording (from MUSICALITY_TIER_WORDS + DIMENSIONS.time/pulse).
T_PHRASES: dict[float, str] = {
    0.00: (
        "non-metric, arrhythmic, irregular, non-repetitive, sparse, "
        "no pulse, unstructured"
    ),
    0.25: (
        "fragmented rhythm, stochastic, irregular pulse, sparse events, "
        "unstructured, no clear beat"
    ),
    0.50: (
        "loose pulse, implied pulse, subtle repetition, quasi-periodic, "
        "slowly emerging pattern"
    ),
    0.75: (
        "broken rhythm, textural rhythm, quasi-metric, evolving repetition, "
        "emerging regular pulse"
    ),
    1.00: (
        "steady pulse, metric, periodic, regular pulse, structured rhythm, "
        "beat-driven, repeating pattern"
    ),
}

# Texture / tone wording (tier texture words + harmonic anchors for hc).
S_PHRASES: dict[float, str] = {
    0.00: (
        "noisy, atonal, inharmonic, broadband texture, evolving granular, "
        "unstable, diffuse"
    ),
    0.25: (
        "diffuse acoustic texture, environmental, organic, faint inharmonic "
        "color, granular"
    ),
    0.50: (
        "soft tonal color, weakly harmonic, granular ambient, somewhat "
        "coherent texture"
    ),
    0.75: (
        "warmer harmonic color, emerging pitch, more consistent texture, "
        "restrained transients"
    ),
    1.00: (
        "tonal, harmonic, sustained pitch, clear harmony, key-stable, "
        "consistent texture"
    ),
}

LADDERS: dict[LadderName, dict[float, str]] = {
    "glitch": {
        0.00: (
            "glitch electronica, electroacoustic free improvisation, non-metric, "
            "sparse, irregular, evolving sound texture"
        ),
        0.25: (
            "glitch electronica, musique concrète, granular acoustic texture, "
            "fragmented, stochastic, unstructured"
        ),
        0.50: (
            "glitch electronica, experimental ambient, microsound, glitch texture, "
            "loose pulse, subtle repetition, slowly emerging pattern"
        ),
        0.75: (
            "glitch electronica, rhythmic ambient, minimal electronica, broken rhythm, "
            "quasi-periodic, consistent texture"
        ),
        1.00: (
            "glitch electronica, minimal dub, IDM, beat-driven, steady pulse, "
            "repetitive textural pattern, regular pulse"
        ),
    },
    "acoustic_perc": {
        0.00: (
            "acoustic percussion free improvisation, musique concrète, sound texture, "
            "non-metric, irregular, sparse, unstructured, incidental"
        ),
        0.25: (
            "acoustic percussion free improvisation, electroacoustic, acoustic texture, "
            "non-metric, fragmented, evolving texture, organic, gestural"
        ),
        0.50: (
            "acoustic percussion free improvisation, experimental ambient, glitch, "
            "deconstructed, broken rhythm, loose pulse, implied pulse, subtle repetition"
        ),
        0.75: (
            "acoustic percussion free improvisation, rhythmic ambient, minimal electronica, "
            "beat-driven, repeating pattern, metric, recurring motif"
        ),
        1.00: (
            "acoustic percussion free improvisation, minimal dub, consistent texture, "
            "steady pulse, periodic, structured rhythm, clear rhythmic hierarchy, "
            "regular pulse"
        ),
    },
    "downtempo": {
        0.00: (
            "downtempo acoustic abstract hiphop, musique concrète, environmental, "
            "fragmented acoustic texture, non-metric, irregular, non-repetitive"
        ),
        0.25: (
            "downtempo acoustic abstract hiphop, electroacoustic free improvisation, "
            "evolving acoustic texture, sparse, stochastic, gestural"
        ),
        0.50: (
            "downtempo acoustic abstract hiphop, experimental ambient, glitch texture, "
            "soft transients, loose pulse, subtle repetition, evolving repetition"
        ),
        0.75: (
            "downtempo acoustic abstract hiphop, rhythmic ambient, minimal dub, "
            "consistent texture, steady pulse, repeating pattern, structured rhythm"
        ),
        1.00: (
            "downtempo acoustic abstract hiphop, abstract hip hop, minimal electronica, "
            "consistent texture, periodic, regular pulse, recurring motif"
        ),
    },
}


def nearest_level(m: float) -> float:
    """Return the ladder level closest to ``m``."""
    return min(LEVELS, key=lambda lv: abs(lv - m))


@overload
def prompt_for(
    ladder: LadderName,
    m: float,
    *,
    mode: Literal["ladder", "fixed"] = "ladder",
    fixed_prompt: str | None = None,
) -> str: ...


@overload
def prompt_for(
    ladder: LadderName,
    m: None = None,
    *,
    T: float,
    S: float,
    mode: Literal["ladder", "fixed"] = "ladder",
    fixed_prompt: str | None = None,
) -> str: ...


def prompt_for(
    ladder: LadderName,
    m: float | None = None,
    *,
    T: float | None = None,
    S: float | None = None,
    mode: Literal["ladder", "fixed"] = "ladder",
    fixed_prompt: str | None = None,
) -> str:
    """Return the prompt for 1-D ``m`` or independent ``T`` / ``S``.

    Parameters
    ----------
    ladder : LadderName
        Genre prefix / 1-D ladder when ``mode='ladder'``.
    m : float | None
        1-D musicality; used when ``T`` and ``S`` are omitted.
    T, S : float | None
        Temporal / spectral commands. Both required for the 2-D path.
    mode : {'ladder', 'fixed'}
        ``fixed`` always returns ``fixed_prompt``.
    fixed_prompt : str | None
        Required when ``mode='fixed'``.

    Returns
    -------
    str
        Conditioning prompt text.
    """
    if mode == "fixed":
        if not fixed_prompt:
            raise ValueError("fixed mode requires fixed_prompt")
        return fixed_prompt
    if T is not None and S is not None:
        t_lv = nearest_level(T)
        s_lv = nearest_level(S)
        return f"{PREFIXES[ladder]}, {T_PHRASES[t_lv]}, {S_PHRASES[s_lv]}"
    if m is None:
        raise ValueError("prompt_for requires m= or both T= and S=")
    return LADDERS[ladder][nearest_level(m)]
