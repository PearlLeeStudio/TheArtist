"""Bounded harmonic search followed by learned, two-sided sequence reranking.

The pitch distance is a symmetric nearest-pitch-class proxy, not performed voice
leading. Mode labels mean set membership in parallel modes of the stated tonic,
not a claim that a mode was detected. No melody is inferred from chord symbols.
"""
from __future__ import annotations

from functools import lru_cache
from itertools import combinations
import re

from app.models.reharmonize import Change, ReharmCandidate, ReharmRequest, Slot

ROOTS = ("C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
NATURAL = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
INTERVALS = {
    "maj": (0, 4, 7), "m": (0, 3, 7), "7": (0, 4, 7, 10), "maj7": (0, 4, 7, 11),
    "m7": (0, 3, 7, 10), "m7b5": (0, 3, 6, 10), "dim7": (0, 3, 6, 9), "dim": (0, 3, 6),
    "aug": (0, 4, 8), "sus4": (0, 5, 7), "sus2": (0, 2, 7), "6": (0, 4, 7, 9),
    "m6": (0, 3, 7, 9), "9": (0, 2, 4, 7, 10), "m9": (0, 2, 3, 7, 10),
    "maj9": (0, 2, 4, 7, 11), "11": (0, 2, 4, 5, 7, 10), "m11": (0, 2, 3, 5, 7, 10),
    "13": (0, 2, 4, 5, 7, 9, 10), "m13": (0, 2, 3, 5, 7, 9, 10),
    "add9": (0, 2, 4, 7), "mMaj7": (0, 3, 7, 11),
    "7b9": (0, 1, 4, 7, 10), "7#9": (0, 3, 4, 7, 10),
    "7#11": (0, 4, 6, 7, 10), "7b13": (0, 4, 7, 8, 10),
}
ALIASES = {"": "maj", "min": "m", "min7": "m7", "min9": "m9", "min11": "m11",
           "min13": "m13", "min6": "m6", "mM7": "mMaj7", "minmaj7": "mMaj7", "M7": "maj7",
           "M9": "maj9", "sus": "sus4", "major": "maj", "M": "maj", "minor": "m",
           "+": "aug", "+7": "aug7", "7#5": "aug7", "o": "dim", "°": "dim",
           "o7": "dim7", "°7": "dim7", "ø": "m7b5", "ø7": "m7b5",
           "M11": "maj11", "M13": "maj13", "Δ": "maj7", "Δ7": "maj7", "Δ9": "maj9",
           "augM7": "augmaj7", "+maj7": "augmaj7", "+M7": "augmaj7", "maj7#5": "augmaj7",
           "mM9": "mMaj9", "7sus": "7sus4"}
# Imported/editor chords are broader than the trained vocabulary. Keep their
# exact tones for the mathematical score, without expanding the set of generated candidates.
INPUT_INTERVALS = {**INTERVALS,
    "aug7": (0, 4, 8, 10), "augmaj7": (0, 4, 8, 11), "5": (0, 7),
    "maj11": (0, 2, 4, 5, 7, 11), "maj13": (0, 2, 4, 5, 7, 9, 11),
    "mMaj9": (0, 2, 3, 7, 11), "6/9": (0, 2, 4, 7, 9), "m6/9": (0, 2, 3, 7, 9),
    "7sus2": (0, 2, 7, 10), "7sus4": (0, 5, 7, 10),
    "9sus4": (0, 2, 5, 7, 10), "13sus4": (0, 2, 5, 7, 9, 10),
    "add2": (0, 2, 4, 7), "add4": (0, 4, 5, 7), "add6": (0, 4, 7, 9),
    "add11": (0, 4, 5, 7), "add13": (0, 4, 7, 9),
    # The editor's manual builder spells minor added tones as m(add9) and a
    # third-less seventh as 7(no3); they are scored by their actual tones.
    "madd9": (0, 2, 3, 7), "m(add9)": (0, 2, 3, 7), "madd2": (0, 2, 3, 7), "m(add2)": (0, 2, 3, 7),
    "madd4": (0, 3, 5, 7), "m(add4)": (0, 3, 5, 7), "madd11": (0, 3, 5, 7), "m(add11)": (0, 3, 5, 7),
    "7(no3)": (0, 7, 10),
}
# Explicit coarse model encodings, disclosed to the caller (modelApproximations). Never let
# the tokenizer silently turn an unfamiliar quality (e.g. aug7) into major.
MODEL_QUALITIES = {**{q: q for q in INTERVALS},
    "aug7": "7", "augmaj7": "maj7", "5": "maj", "maj11": "maj9", "maj13": "maj9",
    "mMaj9": "mMaj7", "6/9": "6", "m6/9": "m6", "7sus2": "7", "7sus4": "7",
    "9sus4": "9", "13sus4": "13", "add2": "add9", "add4": "maj", "add6": "6",
    "add11": "maj", "add13": "6",
    "madd9": "m", "m(add9)": "m", "madd2": "m", "m(add2)": "m",
    "madd4": "m", "m(add4)": "m", "madd11": "m", "m(add11)": "m", "7(no3)": "7",
}
_ALTERATION = re.compile(r"(?:b5|#5|b9|#9|#11|b13)")
_QUALITY = re.compile(
    "(" + "|".join(re.escape(q) for q in sorted(INPUT_INTERVALS.keys() | ALIASES.keys(),
                                                key=lambda q: (-len(q), q))) + ")"
    r"((?:(?:b5|#5|b9|#9|#11|b13)){0,4})(?:\(((?:b5|#5|b9|#9|#11|b13)(?:,?(?:b5|#5|b9|#9|#11|b13)){0,3})\))?\Z"
)


@lru_cache(maxsize=512)
def _quality(quality: str):
    match = _QUALITY.fullmatch(quality)
    if not match:
        raise ValueError("Unsupported chord quality")
    base, suffix, extensions = match.groups()
    base = ALIASES.get(base, base)
    tones = set(INPUT_INTERVALS[base])
    alterations = _ALTERATION.findall(suffix + (extensions or ""))
    # Remove natural degrees together, then add all alterations: b9 and #9
    # can coexist, and their order cannot change the resulting pitch set.
    natural = {"5": 7, "9": 2, "11": 5, "13": 9}
    tones.difference_update(natural[a[1:]] for a in alterations)
    tones.update((natural[a[1:]] + (1 if a[0] == "#" else -1)) % 12 for a in alterations)
    canonical = base + "".join(alterations)
    # Single alterations already in the trained vocabulary retain their tokens.
    model_quality = canonical if canonical in INTERVALS else MODEL_QUALITIES[base]
    return frozenset(tones), canonical, model_quality
MODES = {
    "ionian": (0, 2, 4, 5, 7, 9, 11), "dorian": (0, 2, 3, 5, 7, 9, 10),
    "phrygian": (0, 1, 3, 5, 7, 8, 10), "lydian": (0, 2, 4, 6, 7, 9, 11),
    "mixolydian": (0, 2, 4, 5, 7, 9, 10), "aeolian": (0, 2, 3, 5, 7, 8, 10),
    "locrian": (0, 1, 3, 5, 6, 8, 10),
}
DOMINANTS = {"7", "9", "11", "13", "7b9", "7#9", "7#11", "7b13", "aug7", "7sus2", "7sus4", "9sus4", "13sus4"}
MAJORS = {"maj", "maj7", "maj9", "6", "add9"}
MINORS = {"m", "m7", "m9", "m6", "m11", "m13"}
_SYMBOL = re.compile(r"([A-G][#b]?)(.*?)(?:/([A-G][#b]?))?\Z")


def pitch(note: str) -> int:
    return (NATURAL[note[0]] + {"": 0, "#": 1, "b": -1}[note[1:]]) % 12


@lru_cache(maxsize=2048)
def chord(symbol: str) -> tuple[int, frozenset[int], int, str]:
    match = _SYMBOL.fullmatch(symbol)
    if not match:
        raise ValueError(f"Unsupported chord symbol for substitution search: {symbol}")
    root, quality, bass = match.groups()
    try:
        tones, quality, _ = _quality(quality)
    except ValueError:
        raise ValueError(f"Unsupported chord quality for substitution search: {symbol}") from None
    root_pc = pitch(root)
    pcs = frozenset({(root_pc + interval) % 12 for interval in tones} | ({pitch(bass)} if bass else set()))
    return root_pc, pcs, pitch(bass) if bass else root_pc, quality


def model_symbol(symbol: str) -> str:
    root, _, _, _ = chord(symbol)
    quality = _SYMBOL.fullmatch(symbol).group(2)
    return ROOTS[root] + _quality(quality)[2]


def distance(a: int, b: int) -> int:
    return min((a - b) % 12, (b - a) % 12)


def pitch_distance(a: frozenset[int], b: frozenset[int]) -> float:
    return (sum(min(distance(p, q) for q in b) for p in a) / len(a)
            + sum(min(distance(q, p) for p in a) for q in b) / len(b)) / 2


def context_window(req: ReharmRequest):
    first = min(s.measureIndex for s in req.selectedSlots)
    last = max(s.measureIndex for s in req.selectedSlots)
    return [bar for bar in req.context if first - 8 <= bar.measure <= last + 8]


def validate_context(req: ReharmRequest):
    """Fail before model loading, identifying the actual score slot."""
    for bar in context_window(req):
        for slot, symbol in enumerate(bar.chords):
            if symbol:
                try:
                    chord(symbol)
                except ValueError as exc:
                    raise ValueError(f"M{bar.measure + 1}.{slot + 1}: {exc}. Edit this chord symbol and try again.") from None


def model_approximations(req: ReharmRequest):
    approximations = []
    for bar in context_window(req):
        for slot, symbol in enumerate(bar.chords):
            if not symbol:
                continue
            projected = model_symbol(symbol)
            # Slash bass is already a documented mathematical-only constraint.
            head = _SYMBOL.fullmatch(symbol)
            if chord(head.group(1) + head.group(2))[1] != chord(projected)[1]:
                approximations.append({"address": {"measureIndex": bar.measure, "slotIndex": slot},
                                       "original": symbol, "scoredAs": projected})
    return approximations


def _neighbor(symbols, index, step):
    # Imported charts often have one symbol per bar. Skip empty half-bars,
    # bounded by the same eight-bar reach used by the learned context scorer.
    for offset in range(1, 17):
        target = index + step * offset
        if not 0 <= target < len(symbols):
            break
        if symbols[target]:
            return target
    return None


def _metrics(symbols, changed):
    edges = set()
    for i in changed:
        for step in (-1, 1):
            neighbor = _neighbor(symbols, i, step)
            if neighbor is not None:
                edges.add(tuple(sorted((i, neighbor))))
    if not edges:
        return 0.0, 0.0
    pairs = [(chord(symbols[a]), chord(symbols[b])) for a, b in sorted(edges)]
    return (sum(pitch_distance(a[1], b[1]) for a, b in pairs) / len(pairs),
            sum(distance(a[2], b[2]) for a, b in pairs) / len(pairs))


def convention_preference(key, before, after, previous=None, following=None):
    """Small, explicit theory priors, not measured usage frequencies or quality.

    See OMT's Modal Mixture and Substitutions chapters. Borrowed predominants
    and resolving dominants get more support than mere membership in a mode.
    Never infer a mode from one chord or add bonuses for multiple compatible modes.
    """
    tonic = pitch(key.split()[0])
    root, pcs, _, quality = chord(after)
    old_root, _, _, old_quality = chord(before)
    prev = chord(previous) if previous else None
    nxt = chord(following) if following else None
    degree = (root - tonic) % 12
    aeolian = {(tonic + p) % 12 for p in MODES['aeolian']}
    evidence = []
    if key.endswith('major') and pcs <= aeolian:
        if degree == 2 and quality in {'dim', 'm7b5'} and (old_root - tonic) % 12 in (2, 5):
            resolves_v = nxt and nxt[0] == (tonic + 7) % 12 and nxt[3] in DOMINANTS | MAJORS
            evidence.append((.5 if resolves_v else .3, 'Borrowed iiø / ii° predominant'))
        if degree == 5 and quality in MINORS:
            resolves = nxt and (nxt[0] == tonic or
                               (nxt[0] == (tonic + 7) % 12 and nxt[3] in DOMINANTS | MAJORS))
            evidence.append((.4 if resolves else .2, 'Borrowed iv'))
        if degree in (3, 8, 10) and quality in MAJORS | {'7'}:
            evidence.append((.15, {3: 'Borrowed ♭III', 8: 'Borrowed ♭VI', 10: 'Borrowed ♭VII'}[degree]))
    if key.endswith('minor') and ((degree == 5 and quality in MAJORS | DOMINANTS) or
                                 (degree == 2 and quality in MINORS)):
        dorian = {(tonic + p) % 12 for p in MODES['dorian']}
        if pcs <= dorian and (tonic + 9) % 12 in pcs:
            evidence.append((.2, 'Dorian raised-sixth color'))
    if key.endswith('major') and degree == 2 and quality in MAJORS:
        lydian = {(tonic + p) % 12 for p in MODES['lydian']}
        if pcs <= lydian and ((prev and prev[0] == tonic) or (nxt and nxt[0] == tonic)):
            evidence.append((.15, 'Lydian II beside I'))
    if quality in DOMINANTS and nxt:
        approach = (root - nxt[0]) % 12
        if approach == 7:
            evidence.append((.25, 'Dominant resolution'))
            if degree == 7 and quality == '7b9' and nxt[0] == tonic:
                evidence.append((.45, 'V7♭9 resolving to tonic'))
        if approach == 1 and old_quality in DOMINANTS and (root - old_root) % 12 == 6:
            evidence.append((.5, 'Resolving tritone substitute'))
        if approach == 10 and nxt[0] == tonic and nxt[3] in MAJORS:
            prepared = prev and (prev[0] - tonic) % 12 == 5 and prev[3] in MINORS
            evidence.append((.45 if prepared else .3, 'Backdoor dominant to I'))
    return max((weight for weight, _ in evidence), default=0.0), [label for _, label in evidence]


def _diverse_roots(candidates, limit):
    result, signatures = [], set()
    for item in sorted(candidates, key=lambda c: (-c.mathScore, c.id)):
        signature = tuple((c.address.measureIndex, c.address.slotIndex, chord(c.after)[0]) for c in item.changes)
        if signature not in signatures:
            signatures.add(signature)
            result.append(item)
        if len(result) == limit:
            break
    return result


def search(req: ReharmRequest, family: str, *, limit: int = 6) -> list[ReharmCandidate]:
    """Enumerate selected occupied slots within the requested one-to-four change cap."""
    if family not in ("common_tones", "modal", "dominant"):
        raise ValueError("Unknown substitution family.")
    bars = sorted(req.context, key=lambda bar: bar.measure)
    # The context may be a window of a longer score: positions are relative to
    # its first bar, while every reported address stays absolute.
    base = bars[0].measure
    symbols = [symbol for bar in bars for symbol in bar.chords]
    # Validate precisely the same local context that the model will score.
    validate_context(req)
    tonic = pitch(req.key.split()[0])
    scales = {name: {(tonic + p) % 12 for p in scale} for name, scale in MODES.items()}
    home = scales["aeolian" if req.key.endswith("minor") else "ionian"]

    def position(address) -> int:
        return (address.measureIndex - base) * 2 + address.slotIndex

    selected = sorted(position(s) for s in req.selectedSlots)

    def make(replacements):
        version = list(symbols)
        for i, after in replacements:
            version[i] = after
        changes = []
        preferences = []
        for i, after in sorted(replacements):
            before = symbols[i]
            before_pcs, after_pcs = chord(before)[1], chord(after)[1]
            left, right = _neighbor(version, i, -1), _neighbor(version, i, 1)
            preference, conventions = convention_preference(req.key, before, after,
                version[left] if left is not None else None, version[right] if right is not None else None)
            preferences.append(preference)
            changes.append(Change(address=Slot(measureIndex=i // 2 + base, slotIndex=i % 2), before=before, after=after,
                                  commonTones=len(before_pcs & after_pcs),
                                  conventions=conventions,
                                  modes=[name for name, scale in scales.items() if after_pcs <= scale]))
        indexes = [i for i, _ in replacements]
        motion, bass = _metrics(version, indexes)
        old_motion, old_bass = _metrics(symbols, indexes)
        shared = sum(c.commonTones / len(chord(c.before)[1]) for c in changes) / len(changes)
        # Less movement and shared tones are useful pruning heuristics, never quality probabilities.
        preference = sum(preferences) / len(preferences)
        score = 2 * shared - motion - .12 * bass - .15 * (len(changes) - 1) + preference
        identity = ";".join(f"{i}:{symbol}" for i, symbol in sorted(replacements))
        return ReharmCandidate(id=identity, family=family, changes=changes,
                               pitchDistance=round(motion, 3), originalPitchDistance=round(old_motion, 3),
                               bassMotion=round(bass, 3), originalBassMotion=round(old_bass, 3),
                               mathScore=round(score, 4), conventionScore=round(preference, 4))

    singles = []
    for i in selected:
        root, original, bass, quality = chord(symbols[i])
        next_index = _neighbor(symbols, i, 1)
        following = chord(symbols[next_index]) if next_index is not None else None
        options = []
        for candidate_root in ROOTS:
            for candidate_quality in INTERVALS:
                after = candidate_root + candidate_quality
                cr, pcs, cb, cq = chord(after)
                # Keep an explicitly written pedal/inversion bass; it is not an invitation to replace it.
                slash_bass = _SYMBOL.fullmatch(symbols[i]).group(3)
                if slash_bass:
                    after += "/" + slash_bass
                    cr, pcs, cb, cq = chord(after)
                if pcs == original and cb == bass:
                    continue
                common = len(pcs & original)
                if req.mode != "auto" and not pcs <= scales[req.mode]:
                    continue
                if family == "common_tones" and common < 2:
                    continue
                if family == "modal" and (pcs <= home or not any(pcs <= s for s in scales.values()) or common < 1):
                    continue
                if family == "dominant":
                    tritone = quality in DOMINANTS and cq in DOMINANTS and (cr - root) % 12 == 6
                    approaches = following and cq in DOMINANTS and ((cr - following[0]) % 12 in (1, 7) or
                        ((cr - following[0]) % 12 == 10 and following[3] in MAJORS))
                    if not (tritone or approaches):
                        continue
                # Reject needless density jumps before any model scoring.
                if abs(len(pcs) - len(original)) > 1:
                    continue
                options.append(make([(i, after)]))
        singles.extend(_diverse_roots(options, 6))
    candidates = list(singles)
    if req.maxChanges >= 2:
        # Bounded beam: pair only the best single changes, then recompute shared boundaries.
        for a, b in combinations(sorted(singles, key=lambda c: -c.mathScore)[:12], 2):
            ca, cb = a.changes[0], b.changes[0]
            ia, ib = position(ca.address), position(cb.address)
            if ia != ib:
                candidates.append(make([(ia, ca.after), (ib, cb.after)]))
    if req.maxChanges >= 3:
        # Extend only 12 retained patches per depth with at most two options
        # per selected slot (at most 32). Re-score each complete patch so its
        # shared boundaries and convention evidence reflect all its changes.
        extensions = []
        for i in selected:
            extensions.extend(_diverse_roots([c for c in singles if position(c.changes[0].address) == i], 2))
        beam = _diverse_roots([c for c in candidates if len(c.changes) == 2], 12)
        for size in range(3, min(req.maxChanges, len(selected)) + 1):
            expanded = {}
            for partial in beam:
                replacements = [(position(c.address), c.after) for c in partial.changes]
                occupied = {i for i, _ in replacements}
                for single in extensions:
                    change = single.changes[0]
                    i = position(change.address)
                    if i not in occupied:
                        patch = make([*replacements, (i, change.after)])
                        expanded[patch.id] = patch
            beam = _diverse_roots(list(expanded.values()), 12)
            candidates.extend(beam)
        # Keep wider edits in the six-candidate shortlist the model scores instead
        # of letting the sparsity penalty prune every three/four-chord option.
        diverse = _diverse_roots(candidates, len(candidates))
        shortlist = []
        for size in range(min(req.maxChanges, len(selected)), 0, -1):
            best = next((c for c in diverse if len(c.changes) == size), None)
            if best is not None and len(shortlist) < limit:
                shortlist.append(best)
        seen = {c.id for c in shortlist}
        for item in diverse:
            if len(shortlist) >= limit:
                break
            if item.id not in seen:
                shortlist.append(item)
                seen.add(item.id)
        return shortlist
    # Avoid presenting only extensions of one root as the entire shortlist.
    return _diverse_roots(candidates, limit)


class ContextScorer:
    """Load one chosen checkpoint once, and score only a small mathematical shortlist."""
    def __init__(self, req: ReharmRequest):
        validate_context(req)
        from app.services import inference
        key = req.modelKey or inference.DEFAULT_MODEL
        if key not in inference.CHECKPOINTS:
            raise ValueError("Unknown Music Transformer checkpoint.")
        registry = inference.ModelRegistry.get()
        self.loaded = registry.load_model(key)
        if self.loaded is None:
            raise ValueError("The selected Music Transformer is unavailable; no fallback candidates were returned.")
        self.tokenizer = registry.tokenizer
        self.label = inference.MODEL_LABELS.get(key, key)
        # Project exact parsed qualities into the trained chord vocabulary. Slash
        # bass is handled by the mathematical score; this tokenizer has no bass tokens.
        self.context = [bar.model_copy(update={"chords": [self._token_symbol(s) if s else None for s in bar.chords]})
                        for bar in context_window(req)]
        self.header = inference.build_prompt(self.tokenizer, req.key, None, req.timeSignature, [], [])
        self.selected = sorted({s.measureIndex for s in req.selectedSlots})
        self.cache = {}

    @staticmethod
    def _token_symbol(symbol):
        return model_symbol(symbol)

    def score(self, candidates):
        from app.models.schemas import Suggestion
        from app.services.chord_context import context_scores
        missing = [c for c in candidates if c.id not in self.cache]
        suggestions = []
        for item in missing:
            bars = {str(b.measure): list(b.chords) for b in self.context if b.measure in self.selected}
            for change in item.changes:
                bars[str(change.address.measureIndex)][change.address.slotIndex] = self._token_symbol(change.after)
            # sequence_tokens understands nulls, even though the older Suggestion schema does not.
            suggestions.append(Suggestion.model_construct(label=item.id, chords=bars))
        if suggestions:
            scores = context_scores(self.loaded, self.tokenizer, suggestions, self.header, self.context, self.selected)
            self.cache.update({c.id: score for c, score in zip(missing, scores)})
        return sorted([c.model_copy(update={"modelScore": round(self.cache[c.id], 5)}) for c in candidates],
                      key=lambda c: -c.modelScore)
