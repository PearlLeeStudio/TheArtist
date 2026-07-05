"""Re-rank chord-sequence candidates by mean Sethares roughness.

Lower mean roughness → higher rank. This is a first-pass consonance
filter to layer on top of model-likelihood ranking; voice-leading,
harmonicity, or RAG-grounded scores can be combined downstream.
"""
from __future__ import annotations

from .chord_consonance import chord_roughness
from .voicing import voice_chord

# Special tokens always start with these prefixes — never a chord symbol.
_SPECIAL_PREFIXES = ("[", "<")


def _is_chord_token(tok: str) -> bool:
    return bool(tok) and not tok.startswith(_SPECIAL_PREFIXES)


def sequence_roughness(
    tokens: list[str], root_octave: int = 4
) -> tuple[float, int]:
    """Sum Sethares roughness over all parsable chord tokens in `tokens`.

    Returns:
        (total_roughness, n_scored). Tokens that fail to parse are skipped
        and not counted toward `n_scored`.
    """
    total = 0.0
    n = 0
    for tok in tokens:
        if not _is_chord_token(tok):
            continue
        voicing = voice_chord(tok, root_octave)
        if voicing is None:
            continue
        total += chord_roughness(voicing)
        n += 1
    return total, n


def mean_roughness(tokens: list[str], root_octave: int = 4) -> float:
    """Per-chord mean roughness; NaN if no parsable chords."""
    total, n = sequence_roughness(tokens, root_octave)
    return total / n if n > 0 else float("nan")


def rerank(
    candidates: list[list[str]], root_octave: int = 4
) -> list[int]:
    """Indices of `candidates` sorted by mean roughness ascending.

    Use as: ``[candidates[i] for i in rerank(candidates)]``.
    """
    scored = [(mean_roughness(c, root_octave), i) for i, c in enumerate(candidates)]
    scored.sort(key=lambda x: x[0])
    return [i for _, i in scored]
