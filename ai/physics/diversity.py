"""Computational diversity proxies for chord progressions.

Guards against a rerank-induced collapse to safe, repetitive outputs.
When R1 (physics rerank) or R2 (theory rerank) promotes the most
consonant or most theory-justified option on every step, a side effect
can be to keep selecting the same handful of safe chords — which these
metrics catch.

Intended reading:
- If rerank-on vs rerank-off show comparable diversity, the rerank is
  selecting *better* candidates, not *safer* candidates.
- If rerank-on diversity drops significantly, the rerank is collapsing
  the output and its weight should be reduced (or a diversity bonus
  added).
"""
from __future__ import annotations

import math
from collections import Counter


def chord_entropy(chords: list[str]) -> float:
    """Shannon entropy of the chord-type distribution, in bits.

    Range: 0 (one chord) to ``log2(n_distinct_chords)`` (uniform).
    Higher = more varied vocabulary.
    """
    if not chords:
        return 0.0
    counts = Counter(chords)
    total = len(chords)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def unique_chord_ratio(chords: list[str]) -> float:
    """Fraction of distinct chord types over total chord count.

    Range: 1/n (one type) to 1.0 (all distinct). Complements entropy by
    being insensitive to distribution shape.
    """
    if not chords:
        return 0.0
    return len(set(chords)) / len(chords)


def ngram_repetition_rate(chords: list[str], n: int = 2) -> float:
    """Fraction of n-grams that occur more than once in the sequence.

    Range: 0.0 (all n-grams unique) to 1.0 (every n-gram repeats).
    High values indicate loop-like progressions — often a symptom of a
    collapsed generator or an over-aggressive rerank.
    """
    if len(chords) < n:
        return 0.0
    ngrams = [tuple(chords[i : i + n]) for i in range(len(chords) - n + 1)]
    counts = Counter(ngrams)
    if not counts:
        return 0.0
    repeated_occurrences = sum(v for v in counts.values() if v > 1)
    return repeated_occurrences / len(ngrams)


def diversity_profile(chords: list[str]) -> dict[str, float]:
    """Bundle of diversity metrics for a single chord sequence."""
    return {
        "entropy": chord_entropy(chords),
        "unique_ratio": unique_chord_ratio(chords),
        "bigram_repetition": ngram_repetition_rate(chords, n=2),
        "trigram_repetition": ngram_repetition_rate(chords, n=3),
        "n_chords": len(chords),
        "n_distinct": len(set(chords)),
    }
