"""Second ear: the facts code computes about the A/B/C candidates of a Generate.

One row per candidate (chords outside the key, slots changed from the original,
repetition of the previous section, mean Sethares roughness, the slots the song-form
rules write). review_candidates, what POST /api/review/candidates calls, adds each
candidate's Music Transformer score to its row and names the choice.
"""

from __future__ import annotations

import re
import time

from app.models.review import CandidateFacts, ReviewReason, ReviewRequest, ReviewResponse
from app.services import inference
from app.services.planner import _in_key, _key_collection, _symbol
from app.services.song_form import rule_slots

# The editor's quality aliases (frontend chordParser.ts), so 'C' and 'Cmaj' or 'Am' and 'Amin'
# are one chord there and here; any other quality text is compared as written
_QUALITY_ALIASES = {"": "maj", "M": "maj", "major": "maj", "m": "min", "minor": "min", "M7": "maj7",
                    "min7": "m7", "o7": "dim7", "ø7": "m7b5", "+": "aug", "+7": "aug7", "7#5": "aug7",
                    "sus": "sus4"}
_NOTE_NAMES = frozenset("C C# Db D D# Eb E F F# Gb G G# Ab A A# Bb B B# Cb E# Fb".split())
_EXTENSIONS = re.compile(r"\(([^)]+)\)")
_ALTERED_TRIAD = re.compile(r"\A([A-G][#b]?)\([^)]*\)")


def _slot(symbol: str | None) -> str | None:
    return (symbol or "").strip() or None


def _flat(bars: list[list[str | None]]) -> list[str | None]:
    return [_slot(symbol) for measure in bars for symbol in measure]


def _chord_key(symbol: str | None, with_bass: bool = True) -> tuple | str | None:
    """A chord as the editor compares two chords (generationComparison.ts): root and slash bass as
    spelled, the quality under the editor's aliases, the extensions in parentheses; text it cannot
    read stays text."""
    text = _slot(symbol)
    if text is None:
        return None
    head, bass = text, ""
    cut = text.rfind("/")
    if cut > 0 and text[cut + 1:] in _NOTE_NAMES:
        head, bass = text[:cut], text[cut + 1:]
    root_len = 2 if len(head) >= 2 and head[1] in "#b" else 1
    root, rest = head[:root_len], head[root_len:]
    if root not in _NOTE_NAMES:
        return text
    extensions: tuple[str, ...] = ()
    if match := _EXTENSIONS.search(rest):
        extensions = tuple(sorted(part.strip() for part in match[1].split(",")))
        rest = rest.replace(match[0], "", 1)
    return root, _QUALITY_ALIASES.get(rest, rest or "maj"), extensions, bass if with_bass else ""


def _music21_symbol(symbol: str) -> str:
    """The editor's symbol as the physics code reads it: extensions without parentheses (C7(b9) -> C7b9),
    which music21 does not voice; ai/physics respells flats itself."""
    # An alteration right after a bare root ('C(b9)') would otherwise read as a flat root ('Cb9'): drop it
    bare = _ALTERED_TRIAD.sub(r"\1", symbol)
    return bare.replace("(", "").replace(")", "").replace(",", "")


def _outside_key(chord: str, collection: frozenset[int]) -> bool:
    """A chord with a tone outside the key, without its slash bass as the planner reads it; a
    symbol the parser cannot read counts as outside."""
    core = _symbol(chord)
    return core is None or not _in_key(core, collection)


def _mean_roughness(chords: list[str]) -> float | None:
    """R1 as inference's rerank computes it (ai/physics), over every chord music21 can voice."""
    if not inference.PHYSICS_AVAILABLE or not chords:
        return None
    total, scored = inference.sequence_roughness([_music21_symbol(chord) for chord in chords])
    return round(total / scored, 3) if scored else None


def candidate_facts(request: ReviewRequest) -> list[CandidateFacts]:
    """One row per candidate, in request order. Every ratio counts the candidate's filled slots."""
    original = _flat(request.original)
    previous = _flat(request.previousSection) if request.previousSection is not None else None
    collection = _key_collection(request.key)
    rules = rule_slots(request.sectionType, len(original)) if request.sectionType else []
    rows = []
    for candidate in request.candidates:
        chords = _flat(candidate.chords)
        filled = [chord for chord in chords if chord]
        outside = sum(1 for chord in filled if _outside_key(chord, collection))
        changed = sum(1 for mine, theirs in zip(chords, original) if _chord_key(mine) != _chord_key(theirs))
        repetition = None
        if previous is not None:
            placed = [i for i, chord in enumerate(chords[:len(previous)]) if chord]
            same = sum(1 for i in placed if _chord_key(chords[i], False) == _chord_key(previous[i], False))
            repetition = round(same / len(placed), 3) if placed else None
        rows.append(CandidateFacts(
            label=candidate.label,
            outOfKeyRatio=round(outside / len(filled), 3) if filled else None,
            changedSlots=changed, repetitionVsPrevious=repetition, r1Roughness=_mean_roughness(filled),
            rulesSlots=list(rules),
        ))
    return rows


def review_candidates(request: ReviewRequest) -> ReviewResponse:
    """Compare chord likelihood, with separately measured theory facts."""
    from app.models.schemas import ContextMeasure, Suggestion
    from app.services.chord_context import context_scores, fitting_context
    from app.services.reharmonize import model_symbol

    started = time.monotonic()
    key = request.modelKey or inference.DEFAULT_MODEL
    if key not in inference.CHECKPOINTS:
        raise ValueError("Unknown Music Transformer checkpoint.")
    registry = inference.ModelRegistry.get()
    loaded = registry.load_model(key)
    if loaded is None:
        raise ValueError("The selected Music Transformer is unavailable.")
    prefix = request.previousSection or []
    context = [ContextMeasure(measure=i, chords=[model_symbol(s) if s else None for s in bar])
               for i, bar in enumerate(prefix + request.original)]
    selected = list(range(len(prefix), len(context)))
    suggestions = [Suggestion.model_construct(label=c.label, chords={
        str(len(prefix) + i): [model_symbol(s) if s else None for s in bar]
        for i, bar in enumerate(c.chords)}) for c in request.candidates]
    header = inference.build_prompt(registry.tokenizer, request.key, request.genre, request.timeSignature, [], [])
    window, _ = fitting_context(loaded, registry.tokenizer, suggestions, header, context, selected)
    scores = context_scores(loaded, registry.tokenizer, suggestions, header, window, selected)
    facts = candidate_facts(request)
    for row, score in zip(facts, scores):
        row.modelScore = round(score, 5)
    best = max(range(len(scores)), key=scores.__getitem__)
    score = facts[best].modelScore
    return ReviewResponse(
        choice=request.candidates[best].label,
        reasons=[ReviewReason(fact=f"modelScore={score}", factName="modelScore", value=score,
                              claim="Highest conditional chord likelihood among these candidates; not a quality rating.",
                              confirmed=True, analyzerValue=score)],
        facts=facts, model=inference.MODEL_LABELS.get(key, key),
        latencyMs=round((time.monotonic() - started) * 1000),
    )
