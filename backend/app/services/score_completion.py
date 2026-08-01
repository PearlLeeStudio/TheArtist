"""Offer own-model harmonizations for readable sparse notes, never fake transcription."""
from __future__ import annotations

import re

from app.models.schemas import ContextMeasure
from app.models.score_import import ScoreImportResponse
from app.services.score_context import score_context

MAX_COMPLETION_BARS = 8


def complete_sparse_harmony(draft: ScoreImportResponse) -> ScoreImportResponse:
    from app.services.score_analysis import _bounded_warnings
    result = _complete_sparse_harmony(draft)
    result.warnings = _bounded_warnings(result.warnings)
    return result


def _complete_sparse_harmony(draft: ScoreImportResponse) -> ScoreImportResponse:
    """Local F1 proposals use complete known harmony; printed symbols/N.C. stay fixed.

    Targets need read notes and no existing chord. Unreadable bars are not fabricated.
    The checkpoint proposes candidates; this does not recover unique composer intent.
    """
    targets = [i for i, bar in enumerate(draft.measures) if not bar.chords and not bar.noChord and bar.notes]
    if not targets or not draft.key or not draft.timeSignature:
        return draft
    if len(targets) > MAX_COMPLETION_BARS or len(draft.measures) > 128:
        draft.warnings.append("Own-model harmonization supports up to 8 unresolved note-bearing bars in 128 bars; split this score.")
        return draft
    from app.services.inference import generate_ai_result

    context = [ContextMeasure(measure=i, chords=list(bar.chords)) for i, bar in enumerate(draft.measures)]
    issues = []
    def phase(_, payload):
        if "skipped" in payload.get("label", "").lower():
            issues.append(payload["label"])
    result = generate_ai_result(draft.key, None, context, targets, tuple(draft.timeSignature),
                                model_key="ft_f1", seed=1380, on_phase=phase)
    if result is None or issues:
        draft.warnings.append((issues[0] if issues else "The F1 checkpoint is unavailable; no own-model harmonization was made.")[:400])
        return draft
    # Source note lists are independent of generated chord labels. Rank candidates
    # by their overlap first; the model's whole-context ordering breaks ties.
    from music21 import harmony
    def chord_pitch_classes(symbol):
        symbol = re.sub(r"(^|/)([A-G])b", r"\1\2-", symbol).replace("mMaj", "mM")
        return {pitch.pitchClass for pitch in harmony.ChordSymbol(symbol).pitches}
    evidence = score_context(draft)
    def overlap(candidate):
        score = 0.0
        for i in targets:
            pitches = set(evidence["measures"][i]["pitchClasses"])
            proposed = set()
            for symbol in candidate.chords.get(str(i), []):
                proposed.update(chord_pitch_classes(symbol))
            score += len(pitches & proposed) / len(pitches) if pitches else 0
        return score
    try:
        candidates = sorted(result.suggestions, key=overlap, reverse=True)
    except ValueError:
        draft.warnings.append("Own-model note comparison could not parse a candidate; unresolved bars were retained.")
        return draft
    if not candidates:
        draft.warnings.append("The F1 checkpoint returned no harmonization candidates.")
        return draft
    for i in targets:
        chords = candidates[0].chords.get(str(i), [])
        if not chords:
            continue
        bar = draft.measures[i]
        bar.chords = [symbol for j, symbol in enumerate(chords) if not j or symbol != chords[j - 1]]
        bar.chordBasis, bar.confidence, bar.needsReview = "inferred", "low", True
        bar.model = result.model_key
        bar.note = ("Own-model contextual harmonization, not recovered source chords. Read notes: "
                    + ", ".join(bar.notes[:12]) + ". Available known harmony informed candidate ranking. "
                    + bar.note)[:400]
    draft.model = result.model_key
    draft.warnings.append("Sparse note passages include F1 harmonization suggestions. Review them against the melody; source notes were not changed.")
    return draft
