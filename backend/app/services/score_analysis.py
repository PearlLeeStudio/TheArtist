"""Local vector-score reading and deterministic harmonic analysis.

No API client, key or network is used here. PDF reading and harmonic inference
are separate boundaries, so unread notation never turns into invented notes.
"""

from __future__ import annotations

import base64
import re
from typing import TYPE_CHECKING

from app.models.score_import import (
    MAX_CHORDS_PER_MEASURE, MAX_NOTE_CHARS, MAX_PITCH_NAMES, MAX_SCORE_WARNINGS, MAX_TITLE_CHARS,
    ImportedMeasure, ScoreImportRequest, ScoreImportResponse,
)

if TYPE_CHECKING:
    from app.services.harmony_analysis import NoteEvent
    from app.services.score_pdf import NoteEvent as ReadNote, ParsedScore, ScoreMeasure

_OCTAVE_SUFFIX = re.compile(r"-?\d+$")


class LocalScoreUnsupported(ValueError):
    """The local reader cannot establish the requested score evidence."""


def _without_octave(spelling: str) -> str:
    return _OCTAVE_SUFFIX.sub("", spelling)


def _note_name(midi: int, spelling: str) -> str:
    """The spelling's letter and accidentals with the octave of the MIDI pitch, e.g. B#3 for MIDI 60."""
    root = _without_octave(spelling)
    natural = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[root[0]]
    alteration = root.count("#") - root.count("b") + 2 * root.count("x")
    octave = (midi - natural - alteration) // 12 - 1
    return f"{root}{octave}"


def _written_name(note: ReadNote) -> str:
    """The pitch as written before an octave-span shift; an unshifted note is written as it sounds."""
    written_midi = getattr(note, "written_midi", None)
    return _note_name(written_midi if written_midi is not None else note.midi,
                      getattr(note, "written_spelling", None) or note.spelling)


def _harmonic_spelling(spelling: str) -> str | None:
    root = _without_octave(spelling)
    # The evidence retains double accidentals. Candidate naming uses an
    # enharmonic name when the harmonic engine's single-accidental vocabulary
    # cannot preserve that spelling; MIDI pitch is never changed.
    return root if re.fullmatch(r"[A-G][#b]?", root) else None


def _beats(time_signature: tuple[int, int]) -> float:
    """Quarter-note beats in one measure."""
    return time_signature[0] * 4 / time_signature[1]


def _bass_evidence(notes: list[ReadNote]) -> list[str]:
    """Lowest read active pitches, independent of staff order or crossed voices."""
    result: list[str] = []
    boundaries = {note.onset for note in notes}
    boundaries.update(note.onset + note.duration for note in notes if note.duration is not None)
    for onset in sorted(boundaries):
        active = [note for note in notes if note.onset <= onset and
                  ((note.duration is not None and note.onset + note.duration > onset) or note.onset == onset)]
        if active:
            note = min(active, key=lambda item: item.midi)
            name = _note_name(note.midi, note.spelling)
            if name not in result:
                result.append(name)
    return result[:MAX_PITCH_NAMES]


def _bounded_warnings(messages: list[str]) -> list[str]:
    """Unique, nonempty warnings cut to the note length; past the response bound, a count replaces the rest."""
    unique = list(dict.fromkeys(message[:MAX_NOTE_CHARS] for message in messages if message))
    if len(unique) <= MAX_SCORE_WARNINGS:
        return unique
    kept = MAX_SCORE_WARNINGS - 1
    return [*unique[:kept], f"Additional reading issues: {len(unique) - kept}. Check the source measure evidence before importing."]


def _score_events(score: ParsedScore) -> list[NoteEvent]:
    """Every read note on one continuous beat clock, for the whole-score key estimate.

    Raises LocalScoreUnsupported past harmony_analysis.MAX_NOTE_EVENTS note events.
    """
    from app.services.harmony_analysis import MAX_NOTE_EVENTS, NoteEvent

    events: list[NoteEvent] = []
    cursor = 0.0
    for bar in score.measures:
        beats = _beats(bar.time_signature)
        for note in bar.notes:
            duration = note.duration if note.duration is not None and note.duration > 0 else min(1.0, beats)
            spelling = _harmonic_spelling(note.spelling)
            events.append(NoteEvent(note.midi, cursor + max(0.0, note.onset), duration,
                                    bar.index, staff=str(note.staff), spelling=spelling))
            if len(events) > MAX_NOTE_EVENTS:
                raise LocalScoreUnsupported("The score exceeds the local note-event limit; split it into shorter parts.")
        cursor += beats
    return events


def _inferred_harmony(bar: ScoreMeasure, key: str | None, notes: list[str], evidence: str, label: str,
                      warnings: list[str]) -> tuple[list[str], str, str, str]:
    """Chords proposed from one bar's read notes, as (chords, basis, confidence, evidence).

    A bar with more than MAX_CHORDS_PER_MEASURE changes is reported in warnings and left unresolved.
    """
    from app.services.harmony_analysis import NoteEvent, analyze_harmony

    beats = _beats(bar.time_signature)
    events: list[NoteEvent] = []
    for note in bar.notes:
        onset = min(max(0.0, note.onset), max(0.0, beats - 0.01))
        duration = note.duration if note.duration is not None and note.duration > 0 else min(1.0, beats - onset)
        duration = min(duration, beats - onset)
        if duration <= 0:
            continue
        events.append(NoteEvent(note.midi, onset, duration, bar.index,
                                staff=str(note.staff), spelling=_harmonic_spelling(note.spelling)))
    analysis = analyze_harmony(events, key=key, time_signature=bar.time_signature,
                               total_duration=beats, timing_is_estimated=not bar.rhythm_exact)
    chords: list[str] = []
    reasons: list[str] = []
    scores: list[float] = []
    unresolved_segments = 0
    for segment in analysis.segments:
        if segment.candidates:
            candidate = segment.candidates[0]
            if not chords or candidate.symbol != chords[-1]:
                chords.append(candidate.symbol)
            alternatives = " / ".join(option.symbol for option in segment.candidates[:3])
            reasons.append(f"Candidates {alternatives}: {candidate.evidence}")
            scores.append(candidate.confidence)
        elif segment.kind == "insufficient":
            unresolved_segments += 1
    if len(chords) > MAX_CHORDS_PER_MEASURE:
        warnings.append(f"{label} has more than {MAX_CHORDS_PER_MEASURE} harmonic changes; review a shorter passage.")
        chords = []
    basis = "inferred" if chords else "unreadable"
    confidence = "medium" if scores and min(scores) >= .65 and bar.rhythm_exact else "low"
    evidence = (("Reading limits: " + evidence + " " if evidence else "") +
                "Read notes: " + ", ".join(notes[:16]) + ". " + " ".join(reasons[:3])).strip()
    if not chords:
        evidence = (evidence + " Insufficient simultaneous note evidence for a chord; no harmony was invented.").strip()
    elif unresolved_segments:
        evidence = (f"{unresolved_segments} time slices have insufficient harmony evidence. " + evidence).strip()
        confidence = "low"
    if not bar.rhythm_exact:
        evidence = ("Timing is approximate; check notation and chord changes. " + evidence).strip()
    return chords, basis, confidence, evidence


def _import_measure(mode: str, bar: ScoreMeasure, key: str | None, previous: ImportedMeasure | None,
                    warnings: list[str]) -> ImportedMeasure:
    """One source bar for review: its printed chords, a written repeat of the previous bar, or analyzed notes."""
    notes = list(dict.fromkeys(_note_name(note.midi, note.spelling) for note in bar.notes))[:MAX_PITCH_NAMES]
    written = list(dict.fromkeys(_written_name(note) for note in bar.notes))[:MAX_PITCH_NAMES]
    bass = _bass_evidence(bar.notes)
    label = f"Page {bar.page}, system {bar.system}, bar {bar.number}"
    chords = [chord.symbol for chord in bar.chords]
    no_chord = bool(bar.no_chord)
    evidence = " ".join(bar.warnings)
    basis = "printed"
    confidence = "high"
    needs_review = bool(bar.warnings)

    # A written repeat-measure sign reuses the previous written measure;
    # D.C./D.S./ending jumps never expand the score's written order.
    if not chords and not no_chord and any(marker in ("%", "repeat_measure") for marker in bar.repeat_markers):
        if previous is not None and (previous.chords or previous.noChord):
            chords = list(previous.chords)
            no_chord = previous.noChord
            basis = previous.chordBasis
            confidence = previous.confidence
            needs_review = needs_review or previous.needsReview
            evidence += " Written repeat-measure sign reuses the preceding measure's chord symbols."
        else:
            evidence += " Repeat-measure sign has no preceding readable harmony."

    if mode == "infer_harmony" and not chords and not no_chord:
        chords, basis, confidence, evidence = _inferred_harmony(bar, key, notes, evidence, label, warnings)
        needs_review = True

    if not chords and not no_chord:
        basis, confidence, needs_review = "unreadable", "low", True
    if len(chords) > 2:
        needs_review = True
    return ImportedMeasure(chords=chords, sourceLabel=label, needsReview=needs_review,
                           note=evidence[:MAX_NOTE_CHARS], chordBasis=basis, confidence=confidence,
                           notes=notes, writtenNotes=written, bassNotes=bass, noChord=no_chord,
                           timeSignature=list(bar.time_signature), analysisSource="local")


def analyze_local_score(req: ScoreImportRequest) -> ScoreImportResponse:
    if req.mimeType != "application/pdf":
        raise LocalScoreUnsupported("Local reading currently requires a vector PDF with extractable music symbols; raster images need an optical reader.")
    # Imported per call: tests substitute the parser module and patch the engine's functions.
    from app.services.score_pdf import ScorePDFError, parse_vector_score
    from app.services.harmony_analysis import estimate_key

    try:
        score = parse_vector_score(base64.b64decode(req.dataBase64, validate=True))
    except ScorePDFError as exc:
        raise LocalScoreUnsupported(str(exc)) from None
    if not score.supported or not score.measures:
        raise LocalScoreUnsupported("No supported vector music notation or printed chord measures were found in this PDF.")
    if not score.complete:
        raise LocalScoreUnsupported("The complete written measure order could not be established. Split the PDF into shorter parts.")

    key_events = _score_events(score)
    estimated_timing = any(not bar.rhythm_exact for bar in score.measures)
    key_estimate = estimate_key(key_events, key=score.key, key_signature=score.key_signature,
                                time_signature=score.time_signature or (4, 4),
                                timing_is_estimated=estimated_timing)
    key = key_estimate.key
    warnings = list(score.warnings)
    if estimated_timing:
        warnings.append("Some rhythmic durations could not be decoded exactly. Duration weights and harmonic timing are approximate, and confidence is limited.")
    if key_estimate.source == "inferred":
        alternatives = ", ".join(candidate.key for candidate in key_estimate.alternatives[:3])
        warnings.append(f"The tonal key is inferred from note durations and key signature. Confirm {key or 'the key'}; alternatives: {alternatives}.")

    measures: list[ImportedMeasure] = []
    for bar in score.measures:
        previous = measures[-1] if measures else None
        measures.append(_import_measure(req.mode, bar, key, previous, warnings))

    if not any(measure.chords or measure.noChord for measure in measures):
        if req.mode == "infer_harmony" and key_events:
            warnings.insert(0, "Notes were read, but no sufficiently supported harmony was established. The passages remain unresolved.")
        else:
            message = "No printed chords were found. Choose Analyze notes (no chords) to analyze the read notes." if req.mode == "chord_symbols" else "No supported note or harmony evidence could be read from this score."
            raise LocalScoreUnsupported(message)
    meters = {tuple(measure.timeSignature) for measure in measures if measure.timeSignature}
    if len(meters) > 1:
        warnings.append("The source changes meter between measures. The chord editor uses one global meter; source meters are retained for review and its playback is a harmonic sketch.")
    warnings.extend(score.unresolved)
    return ScoreImportResponse(title=score.title[:MAX_TITLE_CHARS], key=key,
                               timeSignature=list(score.time_signature) if score.time_signature else None,
                               measures=measures, sections=[], warnings=_bounded_warnings(warnings),
                               mode=req.mode, model=None, analysisSource="local")
