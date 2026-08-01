"""Pull a freshly composed progression back into the requested key.

With no chords to continue from, the model largely ignores the [KEY:...]
prompt token: an 8-bar 'C major' request with an empty context can come
back in E major (pop) or G major (rock). ``anchor_to_key`` estimates which
diatonic collection a candidate actually uses and, when that collection
clearly differs from the requested key's, transposes the whole candidate
onto the requested one.

Relative major and minor share one collection (C major = A minor), so the
estimate never moves a candidate between them; it only corrects the
transposition, which is what the model gets wrong. Rock and blues
progressions centre on their collection's 5th degree instead (mixolydian:
I-bVII-IV, I7-IV7-V7, C-Bb-F sits in F major's collection); those move so
that this centre lands on the requested tonic.
"""

from __future__ import annotations

from app.services.voice_leading import _CHORD_TONES, _NOTE_PC, _parse

_MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)
# Index = pitch class of the collection's major tonic.
_COLLECTIONS = [frozenset((t + step) % 12 for step in _MAJOR_SCALE) for t in range(12)]

# The model spells a minor triad 'Am'; the chord-tone table keys it 'min'.
_QUALITY_ALIASES = {"m": "min"}

# Transpose only when the best collection holds this share of all chord
# tones more than the requested one (and at least _MIN_MARGIN tones), so a
# lone borrowed chord or secondary dominant never moves a progression.
# Collections a fifth apart differ in one pitch class, so a pop I-IV-V-vi
# written a fifth away shows only ~8% distinguishing tones: a 10% margin
# left such G-major answers to C-major requests in place. In-key rock and
# blues (C-Bb-F, C7-F7-G7) also score a fifth away, on the flat side; the
# mixolydian reading keeps those where they are, whatever the margin.
_MARGIN_SHARE = 0.05
_MIN_MARGIN = 2

_FLAT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
_SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_SHARP_MAJOR_KEYS = frozenset({"G", "D", "A", "E", "B", "F#", "C#"})
_SHARP_MINOR_KEYS = frozenset({"E", "B", "F#", "C#", "G#", "D#", "A#"})


def _chord_pcs(sym: str) -> frozenset[int] | None:
    """Pitch classes of a chord symbol's chord tones (plus a slash bass)."""
    parsed = _parse(sym)
    if parsed is None:
        return None
    root, quality, bass = parsed
    # Unknown qualities still contribute their root.
    intervals = _CHORD_TONES.get(_QUALITY_ALIASES.get(quality, quality), (0,))
    pcs = {(root + iv) % 12 for iv in intervals}
    if bass is not None:
        pcs.add(bass)
    return frozenset(pcs)


def uses_sharps(key_root: str, is_minor: bool) -> bool:
    """Sharp keys spell with sharps; flat keys and C major / A minor with flats."""
    sharp_keys = _SHARP_MINOR_KEYS if is_minor else _SHARP_MAJOR_KEYS
    return key_root in sharp_keys or "#" in key_root


def _transpose(sym: str, shift: int, names: list[str]) -> str:
    """Move a chord's root and slash bass by ``shift`` semitones, keeping its quality text."""
    head, slash, bass = sym.strip().partition("/")
    root_len = 2 if len(head) >= 2 and head[1] in ("#", "b") else 1
    root, quality = head[:root_len], head[root_len:]
    out = names[(_NOTE_PC[root] + shift) % 12] + quality
    if slash:
        bass = bass.strip()
        out += "/" + (names[(_NOTE_PC[bass] + shift) % 12] if bass in _NOTE_PC else bass)
    return out


def _tonic_hint(tonic: int, roots: list[int]) -> tuple[bool, int]:
    """Tie-break between equally good collections: does the progression open
    on the collection's tonic (major or relative minor), and how often do its
    chord roots sit on those two tonics?"""
    tonics = (tonic, (tonic + 9) % 12)
    return (bool(roots) and roots[0] in tonics, sum(r in tonics for r in roots))


def _centres_on_fifth(tonic: int, roots: list[int]) -> bool:
    """Mixolydian reading of a collection: the progression opens on its 5th
    degree and either closes there or sits there more often than on the
    collection's own tonic (C-Bb-F-C, or a C7-F7-G7 blues, in F's collection).
    A major-key progression that merely starts on V does neither."""
    fifth = (tonic + 7) % 12
    return roots[0] == fifth and (roots[-1] == fifth or roots.count(fifth) > roots.count(tonic))


def anchor_to_key(chords: list[str], key_root: str, is_minor: bool) -> list[str]:
    """Transpose ``chords`` into the requested key when they clearly sit in another one.

    The candidate's collection is the major-scale pitch-class set (of 12)
    holding most of its chord tones. The requested key's collection wins
    ties and near-ties; otherwise every chord (root and slash bass) moves by
    the interval between the two collections' major tonics, or, when the
    progression centres on its collection's 5th degree, by the interval that
    puts that centre on the requested tonic (a mixolydian candidate already
    centred there stays as it is). Transposed chords are spelled for the
    requested key; a mixolydian result, whose tonic chord is major, for the
    major key on the same tonic. Pure: returns a new list.
    """
    target_pc = _NOTE_PC.get(key_root)
    if target_pc is None:
        return list(chords)
    target = (target_pc + 3) % 12 if is_minor else target_pc

    chord_pcs = [pcs for pcs in (_chord_pcs(c) for c in chords) if pcs is not None]
    total = sum(len(pcs) for pcs in chord_pcs)
    if total == 0:
        return list(chords)

    scores = [sum(len(pcs & coll) for pcs in chord_pcs) for coll in _COLLECTIONS]
    roots = [parsed[0] for parsed in (_parse(c) for c in chords) if parsed is not None]

    def rank(tonic: int) -> tuple:
        # Most chord tones first; ties go to the tonic hints, then to the
        # collection the fewest semitones from the requested one.
        move = (target - tonic) % 12
        return (scores[tonic], *_tonic_hint(tonic, roots), -min(move, 12 - move))

    best = max(range(12), key=rank)
    if scores[best] - scores[target] < max(_MIN_MARGIN, _MARGIN_SHARE * total):
        return list(chords)

    if _centres_on_fifth(best, roots):
        shift = (target_pc - best - 7) % 12
        sharps = uses_sharps(key_root, False)
    else:
        shift = (target - best) % 12
        sharps = uses_sharps(key_root, is_minor)
    if shift == 0:
        return list(chords)
    names = _SHARP_NAMES if sharps else _FLAT_NAMES
    return [_transpose(c, shift, names) if _parse(c) is not None else c for c in chords]
