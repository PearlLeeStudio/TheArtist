"""Deterministic harmony candidates from *already read*, time-aligned notes.

This module does not read PDFs, perform optical music recognition, or call a
model. A note's onset/duration are global quarter-note beats, not seconds or
per-measure offsets; measure is its zero-based source measure. Key names use
``C major`` / ``A minor`` (single sharp/flat roots are accepted). Optional note
spellings such as ``Bb`` preserve the written scale degree.

Every note start/end creates a slice. Short stepwise passing/neighbor tones are
downweighted before chord matching; notes from an entire bar are never unioned
into one chord. At least three comparably supported, non-ornamental pitch classes
must support a chord identity. A held dyad plus a short passing/neighbor tone stays
unresolved instead of manufacturing a new harmony. Scores/confidence are
transparent ranking heuristics, not calibrated probabilities. Candidate ties and
key alternatives remain visible. The engine never forces two slots per measure.
The explicit exception is a two-tone tritone shell: it can suggest both possible
rootless dominant-seventh roots, always marked ambiguous with low confidence and
missing structural root/fifth evidence. It never claims an observed full chord.
"""

from __future__ import annotations

from dataclasses import dataclass
from bisect import bisect_left, bisect_right
from math import exp, isfinite, sqrt
import re
from typing import Iterable, Iterator, Literal, NamedTuple

MAX_NOTE_EVENTS = 20_000
_EPS = 1e-8
_ANALYSIS_LIMIT = 1_000_000
_STRUCTURAL_SUPPORT_RATIO = .5
# A candidate's confidence shrinks by a factor of e per .7 score points it trails the best one.
_CONFIDENCE_DECAY = .7
_LETTERS = "CDEFGAB"
_NATURAL = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_FLATS = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")
_SHARPS = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_MAJOR = (0, 2, 4, 5, 7, 9, 11)
_MINOR = (0, 2, 3, 5, 7, 8, 10)
_ROMANS = ("I", "II", "III", "IV", "V", "VI", "VII")
_NOTE = re.compile(r"([A-Ga-g])([#b]?)\Z")
_KEY = re.compile(r"([A-Ga-g][#b]?)\s+(major|minor)\Z", re.IGNORECASE)
# Tonics of the key signatures from seven flats (index 0) to seven sharps (index 14).
_SIGNATURE_MAJOR = ("Cb", "Gb", "Db", "Ab", "Eb", "Bb", "F", "C", "G", "D", "A", "E", "B", "F#", "C#")
_SIGNATURE_MINOR = ("Ab", "Eb", "Bb", "F", "C", "G", "D", "A", "E", "B", "F#", "C#", "G#", "D#", "A#")


def _number(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < 0 or (positive and value <= 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    if value > _ANALYSIS_LIMIT:
        raise ValueError(f"{name} exceeds the analysis limit")
    return float(value)


def _pitch(name: str) -> int:
    match = _NOTE.fullmatch(name)
    if not match:
        raise ValueError("note spelling must be a letter A-G with at most one # or b")
    return (_NATURAL[match[1].upper()] + {"": 0, "#": 1, "b": -1}[match[2]]) % 12


@dataclass(frozen=True)
class NoteEvent:
    pitch: int
    onset: float
    duration: float
    measure: int = 0
    voice: str = ""
    staff: str = ""
    spelling: str | None = None

    def __post_init__(self) -> None:
        if type(self.pitch) is not int or not 0 <= self.pitch <= 127:
            raise ValueError("pitch must be an integer MIDI note 0..127")
        onset = _number(self.onset, "onset")
        duration = _number(self.duration, "duration", positive=True)
        if duration < _EPS or onset + duration > _ANALYSIS_LIMIT:
            raise ValueError("note time is outside the analysis limits")
        if type(self.measure) is not int or self.measure < 0:
            raise ValueError("measure must be a zero-based nonnegative integer")
        if not isinstance(self.voice, str) or not isinstance(self.staff, str):
            raise ValueError("voice and staff must be strings")
        if len(self.voice) > 80 or len(self.staff) > 80:
            raise ValueError("voice/staff labels must be at most 80 characters")
        if self.spelling is not None:
            if not isinstance(self.spelling, str) or _pitch(self.spelling) != self.pitch % 12:
                raise ValueError("note spelling must agree with its MIDI pitch")

    @property
    def end(self) -> float:
        return self.onset + self.duration


# The notes sounding in one time slice, as (index into the sorted notes, note) pairs.
_Sounding = tuple[tuple[int, NoteEvent], ...]


@dataclass(frozen=True)
class KeyCandidate:
    key: str
    score: float


@dataclass(frozen=True)
class KeyEstimate:
    key: str | None
    source: Literal["provided", "inferred", "none"]
    confidence: float
    alternatives: tuple[KeyCandidate, ...]
    pitch_class_durations: tuple[float, ...]


@dataclass(frozen=True)
class NoteEvidence:
    pitch: int
    spelling: str | None
    onset: float
    duration: float
    voice: str
    staff: str
    weight: float
    ornamental: bool


@dataclass(frozen=True)
class ChordCandidate:
    symbol: str
    roman: str | None
    root: str
    quality: str
    bass: str
    inversion: int | None
    score: float
    confidence: float
    chord_pitch_classes: tuple[int, ...]
    missing_pitch_classes: tuple[int, ...]
    non_chord_pitches: tuple[int, ...]
    evidence: str
    rootless_shell: bool = False


@dataclass(frozen=True)
class HarmonySegment:
    start: float
    end: float
    measure: int
    kind: Literal["harmony", "insufficient", "rest"]
    candidates: tuple[ChordCandidate, ...]
    notes: tuple[NoteEvidence, ...]
    uncertain: bool
    reason: str


@dataclass(frozen=True)
class HarmonyAnalysis:
    key: KeyEstimate
    segments: tuple[HarmonySegment, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class _Template:
    quality: str
    intervals: tuple[int, ...]
    missing_costs: tuple[float, ...]
    complexity: float


# Missing fifths cost less than missing thirds/sevenths. Added extensions must
# be sufficiently sustained to outweigh their explicit complexity cost.
_TEMPLATES = (
    _Template("maj", (0, 4, 7), (.9, .9, .35), 0),
    _Template("min", (0, 3, 7), (.9, .9, .35), 0),
    _Template("dim", (0, 3, 6), (.9, .9, .7), .04),
    _Template("aug", (0, 4, 8), (.9, .9, .7), .1),
    _Template("sus2", (0, 2, 7), (.9, .9, .35), .12),
    _Template("sus4", (0, 5, 7), (.9, .9, .35), .12),
    _Template("7", (0, 4, 7, 10), (.9, .9, .3, .85), .18),
    _Template("maj7", (0, 4, 7, 11), (.9, .9, .3, .85), .18),
    _Template("m7", (0, 3, 7, 10), (.9, .9, .3, .85), .18),
    _Template("m7b5", (0, 3, 6, 10), (.9, .9, .7, .85), .22),
    _Template("dim7", (0, 3, 6, 9), (.9, .9, .7, .85), .24),
    _Template("6", (0, 4, 7, 9), (.9, .9, .3, .75), .28),
    _Template("m6", (0, 3, 7, 9), (.9, .9, .3, .75), .28),
    _Template("add9", (0, 4, 7, 2), (.9, .9, .3, .75), .5),
    _Template("9", (0, 4, 7, 10, 2), (.9, .9, .3, .85, .75), .65),
    _Template("maj9", (0, 4, 7, 11, 2), (.9, .9, .3, .85, .75), .7),
    _Template("m9", (0, 3, 7, 10, 2), (.9, .9, .3, .85, .75), .65),
    _Template("11", (0, 4, 7, 10, 2, 5), (.9, .6, .25, .85, .45, .75), .85),
    _Template("m11", (0, 3, 7, 10, 2, 5), (.9, .9, .25, .85, .45, .75), .85),
    _Template("13", (0, 4, 7, 10, 2, 9), (.9, .9, .25, .85, .4, .8), .95),
)
_DOMINANT_SEVENTH = (0, 4, 7, 10)


class _TemplateMatch(NamedTuple):
    score: float
    root: int
    template: _Template
    tones: tuple[int, ...]
    missing: tuple[int, ...]
    covered: float


def _parse_key(key: str) -> tuple[str, int, bool]:
    """Normalized name (``bb minor`` -> ``Bb minor``), tonic pitch class and whether it is minor."""
    match = _KEY.fullmatch(key.strip()) if isinstance(key, str) else None
    if not match:
        raise ValueError("key must be a name such as 'C major' or 'A minor'")
    root = match[1][0].upper() + match[1][1:]
    mode = match[2].lower()
    return f"{root} {mode}", _pitch(root), mode == "minor"


def _metric_weight(onset: float, bar_beats: float) -> float:
    """Downbeats weigh most, then the half bar, then other beats; off-beat onsets least."""
    beat = onset % bar_beats
    if min(beat, bar_beats - beat) < _EPS:
        return 1.15
    if abs(beat - bar_beats / 2) < _EPS:
        return 1.08
    return 1.0 if abs(beat - round(beat)) < _EPS else .9


# Ornament candidates: tones of at most .75 beat whose single same-voice
# neighbors end/start within .25 beat of them, at most a whole tone away.
_ORNAMENT_MAX_DURATION = .75
_ORNAMENT_GAP = .25
_ORNAMENT_MAX_STEP = 2


def _is_stepwise_ornament(previous: int, pitch: int, following: int) -> bool:
    """Approached and left by step: a passing tone moves on, a neighbor tone returns."""
    step_in, step_out = pitch - previous, following - pitch
    if not 0 < abs(step_in) <= _ORNAMENT_MAX_STEP or not 0 < abs(step_out) <= _ORNAMENT_MAX_STEP:
        return False
    passing = step_in * step_out > 0
    neighbor = previous == following
    return passing or neighbor


def _ornament_weights(notes: tuple[NoteEvent, ...], bar_beats: float) -> dict[int, float]:
    """Only label short stepwise tones with unambiguous same-voice neighbors.

    Returns reduced weights by note index; unlisted notes keep full weight.
    """
    voices: dict[tuple[str, str], list[tuple[int, NoteEvent]]] = {}
    for index, note in enumerate(notes):
        voices.setdefault((note.staff, note.voice), []).append((index, note))
    weights: dict[int, float] = {}
    for voice_notes in voices.values():
        by_end = sorted(voice_notes, key=lambda item: item[1].end)
        by_onset = sorted(voice_notes, key=lambda item: item[1].onset)
        end_times = [note.end for _, note in by_end]
        onset_times = [note.onset for _, note in by_onset]
        for index, note in voice_notes:
            if note.duration > _ORNAMENT_MAX_DURATION:
                continue
            ending_before = by_end[bisect_left(end_times, note.onset - _ORNAMENT_GAP):
                                   bisect_right(end_times, note.onset + _EPS)]
            starting_after = by_onset[bisect_left(onset_times, note.end - _EPS):
                                      bisect_right(onset_times, note.end + _ORNAMENT_GAP)]
            if not ending_before or not starting_after:
                continue
            latest_end = max(other.end for _, other in ending_before)
            earliest_onset = min(other.onset for _, other in starting_after)
            previous = [other for _, other in ending_before if abs(other.end - latest_end) < _EPS]
            following = [other for _, other in starting_after if abs(other.onset - earliest_onset) < _EPS]
            if len(previous) != 1 or len(following) != 1:
                continue
            if _is_stepwise_ornament(previous[0].pitch, note.pitch, following[0].pitch):
                # Accented non-chord tones retain more influence than weak ones.
                weights[index] = .4 if _metric_weight(note.onset, bar_beats) > 1 else .18
    return weights


# Explicit tonal preference weights: tonic/dominant/third are strongest;
# chromatic tones still contribute rather than being silently discarded.
_MAJOR_PROFILE = (4, .3, 1.4, .3, 2.5, 1.8, .3, 3, .3, 1.6, .5, 1.1)
_MINOR_PROFILE = (4, .3, 1.1, 2.5, .3, 1.8, .6, 3, 1.8, .5, 1.5, 1.0)
_SHARP_MINOR_TONICS = (1, 6, 8)  # C#, F# and G# minor rather than Db, Gb and Ab minor


def _final_bass_pitch_class(notes: tuple[NoteEvent, ...]) -> int:
    """Pitch class of the lowest note among those that end last."""
    last_end = max(note.end for note in notes)
    return min(note.pitch for note in notes if abs(note.end - last_end) < _EPS) % 12


def _key_candidates(histogram: list[float], total: float, final_bass: int,
                    key_signature: int | None) -> list[KeyCandidate]:
    """All 24 major/minor keys scored against the weighted pitch-class histogram, best first."""
    candidates: list[KeyCandidate] = []
    for is_minor, profile in ((False, _MAJOR_PROFILE), (True, _MINOR_PROFILE)):
        norm = sqrt(sum(value * value for value in profile))
        for tonic in range(12):
            weighted = sum(histogram[pc] * profile[(pc - tonic) % 12] for pc in range(12)) / (total * norm)
            score = weighted + .14 * histogram[tonic] / total + (.08 if tonic == final_bass else 0)
            name = (_SHARPS if is_minor and tonic in _SHARP_MINOR_TONICS else _FLATS)[tonic]
            if key_signature is not None:
                signature_root = (_SIGNATURE_MINOR if is_minor else _SIGNATURE_MAJOR)[key_signature + 7]
                if tonic == _pitch(signature_root):
                    # A signature guides the relative major/minor pair but
                    # never declares its mode or overrides actual note evidence.
                    score += .12
                    name = signature_root
            candidates.append(KeyCandidate(f"{name} {'minor' if is_minor else 'major'}", round(score, 6)))
    candidates.sort(key=lambda item: (-item.score, item.key))
    return candidates


def _estimate_key(notes: tuple[NoteEvent, ...], ornaments: dict[int, float], bar_beats: float,
                  supplied: str | None, key_signature: int | None, timing_is_estimated: bool) -> KeyEstimate:
    histogram = [0.0] * 12
    for index, note in enumerate(notes):
        histogram[note.pitch % 12] += note.duration * ornaments.get(index, 1.0) * _metric_weight(note.onset, bar_beats)
    durations = tuple(round(value, 6) for value in histogram)
    if supplied is not None:
        key, _, _ = _parse_key(supplied)
        return KeyEstimate(key, "provided", 1.0, (KeyCandidate(key, 1.0),), durations)
    total = sum(histogram)
    if total == 0:
        return KeyEstimate(None, "none", 0.0, (), durations)
    candidates = _key_candidates(histogram, total, _final_bass_pitch_class(notes), key_signature)
    # Confidence follows the lead over the runner-up, scaled down until seven
    # distinct pitch classes (a full diatonic collection) have sounded.
    gap = candidates[0].score - candidates[1].score
    coverage = min(1.0, sum(value > 0 for value in histogram) / 7)
    confidence = round(min(.95, max(.05, gap * 7)) * coverage, 4)
    if timing_is_estimated:
        # Estimated timing caps an inferred key, never a manual one.
        confidence = min(.35, confidence)
    return KeyEstimate(candidates[0].key, "inferred", confidence, tuple(candidates[:5]), durations)


def _signed_interval(semitones: int) -> int:
    """Fold a semitone difference into -6..5: negative means below, positive above."""
    return ((semitones + 6) % 12) - 6


def _key_scale(key: str) -> tuple[int, tuple[int, ...]]:
    """Tonic pitch class and major or natural-minor scale steps of a key name."""
    _, tonic, minor = _parse_key(key)
    return tonic, _MINOR if minor else _MAJOR


def _nearest_degree(interval: int, scale: tuple[int, ...]) -> int:
    """Scale degree closest to ``interval``; a tie lowers the upper degree rather than raising the lower."""
    alterations = [_signed_interval(interval - step) for step in scale]
    return min(range(7), key=lambda degree: (abs(alterations[degree]), 0 if alterations[degree] <= 0 else 1))


def _spell(pc: int, key: str | None, evidence: tuple[NoteEvent, ...] = ()) -> str:
    """The most frequent written spelling, else the key's nearest scale-degree letter."""
    written = [note.spelling for note in evidence if note.pitch % 12 == pc and note.spelling]
    if written:
        spelling = max(dict.fromkeys(written), key=written.count)
        return spelling[0].upper() + spelling[1:]
    if key is None:
        return _FLATS[pc]
    tonic, scale = _key_scale(key)
    tonic_letter = _LETTERS.index(key.split(" ")[0][0])
    degree = _nearest_degree((pc - tonic) % 12, scale)
    letter = _LETTERS[(tonic_letter + degree) % 7]
    accidental = _signed_interval(pc - _NATURAL[letter])
    if abs(accidental) <= 1:
        return letter + ("#" if accidental == 1 else "b" if accidental == -1 else "")
    return _FLATS[pc]


_LOWERCASE_NUMERAL_QUALITIES = frozenset({"min", "m7", "m6", "m9", "m11", "dim", "dim7", "m7b5"})
_ROMAN_SUFFIXES = {"maj": "", "min": "", "dim": "°", "dim7": "°7", "aug": "+",
                   "m7b5": "ø7", "m7": "7", "m6": "6", "m9": "9", "m11": "11"}


def _roman(root: str, quality: str, key: str | None) -> str | None:
    if key is None:
        return None
    tonic, scale = _key_scale(key)
    tonic_letter = _LETTERS.index(key[0])
    degree = (_LETTERS.index(root[0]) - tonic_letter + 7) % 7
    interval = (_pitch(root) - tonic) % 12
    alteration = _signed_interval(interval - scale[degree])
    prefix = "#" * alteration if alteration > 0 else "b" * -alteration
    numeral = _ROMANS[degree]
    if quality in _LOWERCASE_NUMERAL_QUALITIES:
        numeral = numeral.lower()
    return prefix + numeral + _ROMAN_SUFFIXES.get(quality, quality)


def _listed(values: Iterable[int]) -> str:
    return ", ".join(map(str, values))


def _non_chord_pitches(notes: tuple[NoteEvent, ...], tones: tuple[int, ...]) -> tuple[int, ...]:
    """Distinct sounding MIDI pitches outside the chord's pitch classes, low to high."""
    return tuple(sorted({note.pitch for note in notes if note.pitch % 12 not in tones}))


def _chord_symbol(root_name: str, quality: str, bass_name: str, *, slash: bool) -> str:
    """Lead-sheet symbol such as ``C``, ``Am`` or ``G7/B``."""
    suffix = "" if quality == "maj" else "m" if quality == "min" else quality
    return root_name + suffix + (f"/{bass_name}" if slash else "")


def _tonal_context(key: str | None) -> tuple[int, set[int]]:
    """Tonic pitch class and diatonic pitch classes; ``(-1, set())`` without a key."""
    if not key:
        return -1, set()
    tonic, scale = _key_scale(key)
    return tonic, {(tonic + interval) % 12 for interval in scale}


def _with_tonal_bonus(score: float, root: int, tones: tuple[int, ...], tonic: int, diatonic: set[int]) -> float:
    """Key-context bonuses for a diatonic root, an all-diatonic chord and a tonic root.

    Adds each bonus to the running score in this order; summing the bonuses first
    would change float rounding and can reorder exactly tied candidates.
    """
    score += .12 if root in diatonic else 0
    score += .08 if set(tones) <= diatonic else 0
    score += .05 if root == tonic else 0
    return score


def _rootless_shells(sounding: _Sounding, structural_pcs: set[int],
                     key: KeyEstimate, confidence_cap: float) -> tuple[ChordCandidate, ...]:
    """A tritone is compatible with two dominant 3rd/7th shells, not one chord.

    The missing classes here lack structural support, even if a fleeting melody
    ornament happens to touch one. Raw note evidence remains unchanged.
    """
    first, second = sorted(structural_pcs)
    if (second - first) % 12 != 6:
        return ()
    active_notes = tuple(note for _, note in sounding)
    bass_pc = min(note.pitch for note in active_notes) % 12
    bass_name = _spell(bass_pc, key.key, active_notes)
    tonic, diatonic = _tonal_context(key.key)
    interpretations = []
    # Either tritone tone can be the third (a major third above the root); the other is the seventh.
    for root in ((first - 4) % 12, (second - 4) % 12):
        tones = tuple((root + interval) % 12 for interval in _DOMINANT_SEVENTH)
        # Fleeting, unsupported melody tones cannot flip the identity of an
        # unchanged structural shell. They stay in the evidence, while only
        # the shell and tonal context order these inherently ambiguous roots.
        # Base: a fully covered dominant seventh missing its root (.9) and
        # fifth (.3), less the seventh-chord complexity (.18).
        score = _with_tonal_bonus(4 - .9 - .3 - .18, root, tones, tonic, diatonic)
        interpretations.append((score, root, tones))
    interpretations.sort(key=lambda item: (-item[0], item[1]))
    best_score = interpretations[0][0]
    alternatives = ", ".join(_spell(root, key.key) + "7" for _, root, _ in interpretations)
    pitches = _listed(sorted({note.pitch for note in active_notes}))
    result = []
    for score, root, tones in interpretations:
        root_name = _spell(root, key.key)
        missing = tuple(pc for pc in tones if pc not in structural_pcs)
        inversion = _DOMINANT_SEVENTH.index((bass_pc - root) % 12) if bass_pc in tones else None
        symbol = _chord_symbol(root_name, "7", bass_name, slash=bass_pc != root)
        confidence = min(.35, confidence_cap, .25 * exp((score - best_score) / _CONFIDENCE_DECAY))
        evidence = (f"Ambiguous rootless dominant shell: structural pitch classes {first}, {second} can be "
                    f"third/seventh tones. Alternatives: {alternatives}. Root/fifth lack structural support "
                    f"(pitch classes {_listed(missing)}); this is not an observed complete chord. "
                    f"Read MIDI: {pitches}; bass {bass_name}.")
        result.append(ChordCandidate(symbol, _roman(root_name, "7", key.key), root_name, "7", bass_name,
                                     inversion, round(score, 6), round(confidence, 4), tones, missing,
                                     _non_chord_pitches(active_notes, tones), evidence, rootless_shell=True))
    return tuple(result)


def _pitch_class_support(sounding: _Sounding, weights: dict[int, float],
                         ornaments: dict[int, float]) -> tuple[dict[int, float], set[int]]:
    """Summed weight per sounding pitch class, and the classes with comparable structural support.

    Ornaments add weight but never structural support.
    """
    pc_weights: dict[int, float] = {}
    structural_support: dict[int, float] = {}
    for index, note in sounding:
        pc = note.pitch % 12
        pc_weights[pc] = pc_weights.get(pc, 0) + weights[index]
        if index not in ornaments:
            # Doubling the same fleeting melody tone in another octave must
            # not manufacture duration support for an extra harmonic degree.
            structural_support[pc] = max(structural_support.get(pc, 0), weights[index])
    strongest = max(structural_support.values(), default=0)
    structural_pcs = {pc for pc, weight in structural_support.items()
                      if weight >= strongest * _STRUCTURAL_SUPPORT_RATIO}
    return pc_weights, structural_pcs


def _template_matches(pc_weights: dict[int, float], structural_pcs: set[int], bass_pc: int,
                      key: str | None) -> list[_TemplateMatch]:
    """Score every template on every sounding root, best first.

    A template is scored only when its tones on that root include at least three
    structural pitch classes; the root itself only has to sound.
    """
    total = sum(pc_weights.values())
    tonic, diatonic = _tonal_context(key)
    matches: list[_TemplateMatch] = []
    for root in range(12):
        # No rootless-chord assertion when a simpler explanation has actual
        # root evidence. Rootless voicings can be interpreted in a later layer.
        if root not in pc_weights:
            continue
        for template in _TEMPLATES:
            tones = tuple((root + interval) % 12 for interval in template.intervals)
            if len(set(tones) & structural_pcs) < 3:
                continue
            missing = tuple(pc for pc in tones if pc not in pc_weights)
            foreign = sum(weight for pc, weight in pc_weights.items() if pc not in tones)
            covered = 1 - foreign / total
            missing_cost = sum(cost for pc, cost in zip(tones, template.missing_costs) if pc not in pc_weights)
            score = 4 * covered - 2 * foreign / total - missing_cost - template.complexity
            # A root-position bass beats an inverted one, which beats a foreign bass.
            score += .24 if bass_pc == root else .12 if bass_pc in tones else -.2
            score = _with_tonal_bonus(score, root, tones, tonic, diatonic)
            matches.append(_TemplateMatch(score, root, template, tones, missing, covered))
    matches.sort(key=lambda match: (-match.score, len(match.tones), match.root, match.template.quality))
    return matches


def _rank(sounding: _Sounding, weights: dict[int, float], ornaments: dict[int, float],
          key: KeyEstimate, top_n: int, confidence_cap: float) -> tuple[ChordCandidate, ...]:
    pc_weights, structural_pcs = _pitch_class_support(sounding, weights, ornaments)
    if len(structural_pcs) == 2:
        return _rootless_shells(sounding, structural_pcs, key, confidence_cap)
    if len(structural_pcs) < 3:
        return ()
    active_notes = tuple(note for _, note in sounding)
    bass_midi = min(note.pitch for note in active_notes)
    bass_pc = bass_midi % 12
    matches = _template_matches(pc_weights, structural_pcs, bass_pc, key.key)
    if not matches:
        return ()
    best = matches[0]
    # The winner's confidence grows with its margin over the runner-up and with its coverage.
    margin = best.score - matches[1].score if len(matches) > 1 else 1
    confidence = min(confidence_cap, min(.95, max(.05, .35 + margin * .65)) * best.covered)
    bass_name = _spell(bass_pc, key.key, active_notes)
    pitches = _listed(sorted({note.pitch for note in active_notes}))
    result: list[ChordCandidate] = []
    for score, root, template, tones, missing, covered in matches[:top_n]:
        root_name = _spell(root, key.key, active_notes)
        symbol = _chord_symbol(root_name, template.quality, bass_name, slash=bass_pc != root)
        non_chord = _non_chord_pitches(active_notes, tones)
        inversion = template.intervals.index((bass_pc - root) % 12) if bass_pc in tones else None
        relative_confidence = confidence * exp((score - best.score) / _CONFIDENCE_DECAY)
        evidence = (f"Active MIDI pitches: {pitches}; bass {bass_name} ({bass_midi}); "
                    f"weighted chord coverage {covered:.0%}. "
                    f"Structural chord pitch classes: {_listed(sorted(set(tones) & structural_pcs))}.")
        if non_chord:
            evidence += f" Non-chord pitches: {_listed(non_chord)}."
        if missing:
            evidence += f" Missing template pitch classes: {_listed(missing)}."
        result.append(ChordCandidate(symbol, _roman(root_name, template.quality, key.key), root_name,
                                     template.quality, bass_name, inversion, round(score, 6),
                                     round(relative_confidence, 4), tones, missing, non_chord, evidence))
    return tuple(result)


def _time_slices(notes: tuple[NoteEvent, ...], end: float) -> Iterator[tuple[float, float, _Sounding]]:
    """Spans between consecutive note starts/ends, 0 and ``end``, with the notes sounding in each.

    Spans shorter than ``_EPS`` are skipped; a span without sounding notes is a rest.
    """
    boundaries = sorted({0.0, end, *(note.onset for note in notes), *(note.end for note in notes)})
    starts: dict[float, list[int]] = {}
    ends: dict[float, list[int]] = {}
    for index, note in enumerate(notes):
        starts.setdefault(note.onset, []).append(index)
        ends.setdefault(note.end, []).append(index)
    active: set[int] = set()
    for start, stop in zip(boundaries, boundaries[1:]):
        active.difference_update(ends.get(start, ()))
        active.update(starts.get(start, ()))
        if stop - start < _EPS:
            continue
        yield start, stop, tuple((index, notes[index]) for index in sorted(active))


def _segment_kind_and_reason(sounding: _Sounding, candidates: tuple[ChordCandidate, ...],
                             ornaments: dict[int, float]) -> tuple[Literal["harmony", "insufficient", "rest"], str]:
    if not sounding:
        kind, reason = "rest", "No active notes; no chord inferred."
    elif not candidates:
        kind, reason = "insufficient", "Fewer than three sufficiently supported structural pitch classes; harmony is unresolved."
    elif candidates[0].rootless_shell:
        kind, reason = "harmony", "Ambiguous rootless dominant shell; both possible roots are retained and need review."
    else:
        kind, reason = "harmony", "Ranked from simultaneous notes, duration/metric weights and bass evidence."
    if any(index in ornaments for index, _ in sounding):
        reason += " Short stepwise ornament(s) were downweighted."
    return kind, reason


def _validated_notes(events: Iterable[NoteEvent]) -> tuple[NoteEvent, ...]:
    notes_list: list[NoteEvent] = []
    for note in events:
        if not isinstance(note, NoteEvent):
            raise ValueError("events must contain validated NoteEvent instances")
        notes_list.append(note)
        if len(notes_list) > MAX_NOTE_EVENTS:
            raise ValueError(f"analysis accepts at most {MAX_NOTE_EVENTS} note events")
    return tuple(sorted(notes_list, key=lambda note: (note.onset, note.pitch, note.duration,
                                                     note.measure, note.staff, note.voice, note.spelling or "")))


def _bar_beats(time_signature: tuple[int, int]) -> float:
    if not isinstance(time_signature, (tuple, list)) or len(time_signature) != 2:
        raise ValueError("time_signature must contain numerator and denominator")
    numerator, denominator = time_signature
    if type(numerator) is not int or not 1 <= numerator <= 32 or type(denominator) is not int \
            or denominator not in (1, 2, 4, 8, 16, 32):
        raise ValueError("time_signature must use numerator 1..32 and a power-of-two denominator 1..32")
    return numerator * 4 / denominator


def _key_options(key: str | None, key_signature: int | None, timing_is_estimated: bool) -> str | None:
    """Validate the options both entry points share; return the normalized manual key."""
    if key_signature is not None and (type(key_signature) is not int or not -7 <= key_signature <= 7):
        raise ValueError("key_signature must be an integer flat/sharp count -7..7")
    if type(timing_is_estimated) is not bool:
        raise ValueError("timing_is_estimated must be a boolean")
    return _parse_key(key)[0] if key is not None else None


def estimate_key(events: Iterable[NoteEvent], *, key: str | None = None,
                 key_signature: int | None = None, time_signature: tuple[int, int] = (4, 4),
                 timing_is_estimated: bool = False) -> KeyEstimate:
    """Estimate a whole-score key before analyzing measures with their own meters.

    Use global onsets for this call, then local quarter-beat onsets and the
    original zero-based measure index for individual ``analyze_harmony`` calls.
    A mixed-meter adapter may use this default meter for approximate global
    onset weighting; durations and the signature pair remain the main evidence.
    Estimated timing caps inferred-key confidence at .35, never a manual key.
    """
    key = _key_options(key, key_signature, timing_is_estimated)
    bar_beats = _bar_beats(time_signature)
    notes = _validated_notes(events)
    return _estimate_key(notes, _ornament_weights(notes, bar_beats), bar_beats, key, key_signature,
                         timing_is_estimated)


def analyze_harmony(events: Iterable[NoteEvent], *, key: str | None = None,
                    time_signature: tuple[int, int] = (4, 4), top_n: int = 3,
                    total_duration: float | None = None, key_signature: int | None = None,
                    timing_is_estimated: bool = False) -> HarmonyAnalysis:
    """Analyze notes without I/O, model state, random choices or bar-slot padding.

    ``total_duration`` optionally retains leading/trailing rests. If supplied it
    must cover all events. Results are serializable with ``dataclasses.asdict``.
    Manual key always overrides the heuristic key estimate, including its Roman
    analysis; it never transposes the evidence pitches.
    ``key_signature`` is an optional -7..7 flat/sharp count, a soft major/minor
    pair prior rather than a declaration of mode. Geometry-derived or otherwise
    uncertain timing must set ``timing_is_estimated``; chord confidence is then
    capped at .55 and all sounding segments require review.
    ``top_n`` controls ordinary candidates. The two inherently ambiguous
    rootless-shell alternatives are both retained even when ``top_n=1``.
    """
    if type(top_n) is not int or not 1 <= top_n <= 10:
        raise ValueError("top_n must be an integer 1..10")
    key = _key_options(key, key_signature, timing_is_estimated)
    bar_beats = _bar_beats(time_signature)
    notes = _validated_notes(events)
    end = max((note.end for note in notes), default=0.0)
    if total_duration is not None:
        total_duration = _number(total_duration, "total_duration")
        if total_duration + _EPS < end:
            raise ValueError("total_duration must cover every note event")
        end = total_duration
    ornaments = _ornament_weights(notes, bar_beats)
    weights = {index: min(1.0, note.duration) * _metric_weight(note.onset, bar_beats) * ornaments.get(index, 1.0)
               for index, note in enumerate(notes)}
    estimate = _estimate_key(notes, ornaments, bar_beats, key, key_signature, timing_is_estimated)
    key_is_ambiguous = estimate.source == "inferred" and estimate.confidence < .5
    warnings: list[str] = []
    if key_is_ambiguous:
        warnings.append("The global key is ambiguous; inspect its alternatives or provide a key.")
    if not notes:
        warnings.append("No note evidence was supplied; no chord has been inferred.")
    if timing_is_estimated:
        warnings.append("Note timing is estimated rather than read exactly; simultaneity, bass and chord boundaries require review.")
    confidence_cap = .55 if timing_is_estimated else .95
    segments: list[HarmonySegment] = []
    last_measure = 0
    for start, stop, sounding in _time_slices(notes, end):
        evidence = tuple(NoteEvidence(note.pitch, note.spelling, note.onset, note.duration,
                                      note.voice, note.staff, round(weights[index], 6), index in ornaments)
                         for index, note in sounding)
        if sounding:  # a rest keeps the measure of the notes before it
            last_measure = max(note.measure for _, note in sounding)
        candidates = _rank(sounding, weights, ornaments, estimate, top_n, confidence_cap)
        kind, reason = _segment_kind_and_reason(sounding, candidates, ornaments)
        uncertain = (key_is_ambiguous or not candidates or candidates[0].confidence < .65
                     or bool(candidates[0].missing_pitch_classes))
        segments.append(HarmonySegment(start, stop, last_measure, kind, candidates, evidence, uncertain, reason))
    return HarmonyAnalysis(estimate, tuple(segments), tuple(warnings))
