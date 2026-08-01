"""Song form post-processor for chord generation.

Applies section-aware harmonic rules derived from statistical analysis
of 679,807 songs (Chordonomicon dataset). The model generates chords
freely; this module adjusts first/last chords and cadences to match
conventional song form expectations.

Rules are genre-agnostic and based on relative scale degrees.
"""

from __future__ import annotations

from collections.abc import Callable

from app.models.schemas import Suggestion
from app.services.key_anchor import uses_sharps

NOTES_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
NOTES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

NOTE_TO_SEMI: dict[str, int] = {}
for _i, _n in enumerate(NOTES_FLAT):
    NOTE_TO_SEMI[_n] = _i
for _i, _n in enumerate(NOTES_SHARP):
    NOTE_TO_SEMI[_n] = _i

_SHARP_KEYS = frozenset({"B", "F#", "C#"})

# The five original sections keep the spelling above (flats except in B, F#
# and C#), so their output stays exactly as it was. The sections added with
# the wider vocabulary spell by key signature, like key_anchor: E major and
# E minor write C#m7 / F#m7, not Dbm7 / Gbm7.
_ORIGINAL_SECTIONS = frozenset({"intro", "verse", "chorus", "bridge", "outro"})


def _note(semi: int, flats: bool = True) -> str:
    return NOTES_FLAT[semi % 12] if flats else NOTES_SHARP[semi % 12]


def _spells_with_flats(section: str, key: str, is_minor: bool) -> bool:
    if section in _ORIGINAL_SECTIONS:
        return key not in _SHARP_KEYS
    return not uses_sharps(key, is_minor)


def _parse_chord_root(chord: str) -> tuple[str, str] | None:
    """Return (root, quality) from a chord string like 'Bbm7'."""
    if not chord:
        return None
    root = chord[0]
    rest = chord[1:]
    if rest and rest[0] in ("b", "#"):
        root += rest[0]
        rest = rest[1:]
    if root not in NOTE_TO_SEMI:
        return None
    return root, rest


def _degree_of(chord: str, key_root: int) -> int | None:
    """Return semitone interval of chord root relative to key."""
    parsed = _parse_chord_root(chord)
    if not parsed:
        return None
    return (NOTE_TO_SEMI[parsed[0]] - key_root) % 12


def _make_chord(key_root: int, interval: int, quality: str, flats: bool) -> str:
    return f"{_note((key_root + interval) % 12, flats)}{quality}"


# ── Section rules ──────────────────────────────────────────────────────
# Based on Chordonomicon 679K song analysis:
#
# INTRO:  starts I (54%), ends varied
# VERSE:  starts I (34%), ends I(24%)/V(16%) — tension maintained
# CHORUS: starts I(28%)/IV(13%), ends I(26%)/V(13%) — V→I cadence (11%)
# BRIDGE: starts IV(11%)/vi(6%), ends V(19%) — half cadence
# OUTRO:  ends I(31%) — V→I(10%), IV→I(9%) — strong resolution
#
# The other section ids (app/models/sections.py) follow common practice
# rather than the corpus statistics above: a pre-chorus hangs on V, loop
# sections (post-chorus, drop, breakdown, vamp) repeat their opening bars,
# a turnaround leads back to the top, a tag closes like an outro, and the
# jazz head / A / B sections cadence like verse / chorus / bridge.

def apply_song_form(
    chords: list[str],
    section: str,
    key: str,
    is_minor: bool = False,
    cadence: str | None = None,
) -> list[str]:
    """Post-process generated chords based on song section conventions.

    Args:
        chords: Generated chord names (flat list for all selected measures)
        section: A section id from app/models/sections.py; ids without a
            rule (interlude, solo, unknown) pass through unchanged
        key: Key root note (e.g. 'C', 'Bb')
        is_minor: Whether the key is minor
        cadence: An ending from CADENCE_TYPES that replaces the section's
            own (the planner's choice); None keeps the section's rule

    Returns:
        Adjusted chord list (same length)
    """
    return apply_song_form_with_slots(chords, section, key, is_minor, cadence)[0]


def apply_song_form_with_slots(
    chords: list[str],
    section: str,
    key: str,
    is_minor: bool = False,
    cadence: str | None = None,
) -> tuple[list[str], list[int]]:
    """apply_song_form plus the slot indices its rules decided (rule_slots),
    so a caller can tell the model's chords from the rules' chords."""
    rules = _section_rules(section, cadence)
    if not chords or rules is None:
        return chords, []

    key_root = NOTE_TO_SEMI.get(key)
    if key_root is None:
        return chords, []
    flats = _spells_with_flats(section, key, is_minor)

    result = list(chords)

    # Static sections: repeat the opening chords across the whole section
    if rules.get("loop"):
        n = rules["loop"]
        result = [result[i % n] for i in range(len(result))]

    # Apply first chord rule
    if rules.get("force_start") and len(result) >= 1:
        start_interval, start_quality = rules["force_start"](is_minor)
        current_degree = _degree_of(result[0], key_root)
        if current_degree != start_interval:
            result[0] = _make_chord(key_root, start_interval, start_quality, flats)

    # Apply cadence rule (last 2 chords)
    if rules.get("cadence") and len(result) >= 2:
        result = _apply_cadence(result, rules["cadence"], key_root, is_minor, flats)

    # Apply last chord rule
    if rules.get("force_end") and len(result) >= 1:
        end_interval, end_quality = rules["force_end"](is_minor)
        current_degree = _degree_of(result[-1], key_root)
        if current_degree != end_interval:
            result[-1] = _make_chord(key_root, end_interval, end_quality, flats)

    return result, rule_slots(section, len(result), cadence)


def _section_rules(section: str, cadence: str | None) -> dict | None:
    """The section's rules, with a chosen cadence in place of the section's own.

    A chosen cadence also drops force_end: authentic and plagal already land on
    the bare tonic, and the open endings (half, turnaround) must not be
    overwritten with it. "none" keeps the model's own ending. The section's
    identity (forced first chord, loop) stays.
    """
    rules = SECTION_RULES.get(section)
    if rules is None or cadence is None:
        return rules
    rules = {name: rule for name, rule in rules.items() if name not in ("cadence", "force_end")}
    if cadence != "none":
        rules["cadence"] = cadence
    return rules


def _cadence_slots(cadence_type: str, n: int) -> list[int]:
    """The indices _apply_cadence writes in an n-chord sequence (n >= 2)."""
    if cadence_type in ("authentic", "plagal", "half_iv_v"):
        return [n - 2, n - 1]
    if cadence_type == "half":
        return [n - 1]
    if cadence_type == "turnaround":
        return list(range(n - 4, n)) if n >= 4 else [n - 1]
    return []


def rule_slots(section: str, n: int, cadence: str | None = None) -> list[int]:
    """The slot indices (0..n-1) the section's rules decide in an n-chord
    sequence, whether or not the chord there already complied: the loop's
    copies, the forced first chord, the cadence and the forced last chord.
    Empty for a section without rules, so a caller can predict the slots
    before any chord exists."""
    rules = _section_rules(section, cadence)
    if not rules or n <= 0:
        return []
    slots: set[int] = set()
    if rules.get("loop"):
        slots.update(range(rules["loop"], n))
    if rules.get("force_start"):
        slots.add(0)
    if rules.get("cadence") and n >= 2:
        slots.update(_cadence_slots(rules["cadence"], n))
    if rules.get("force_end"):
        slots.add(n - 1)
    return sorted(slots)


def resolves_to_tonic(section: str | None, cadence: str | None = None) -> bool:
    """Whether the section's effective ending lands on the tonic (the rule
    behind ROOT_POSITION_ENDINGS, with the planner's cadence taken into account)."""
    rules = _section_rules(section, cadence) if section else None
    return bool(rules) and rules.get("cadence") in ("authentic", "plagal")


def rewrite_chord_sequence(
    suggestion: Suggestion,
    rewrite: Callable[[list[str]], list[str]],
) -> None:
    """Replace a suggestion's chords in place with ``rewrite`` applied to its
    whole chord sequence (measures in order), keeping every measure's slot count."""
    measures = sorted(suggestion.chords, key=int)
    rewritten = rewrite([chord for m in measures for chord in suggestion.chords[m]])
    start = 0
    for m in measures:
        n_slots = len(suggestion.chords[m])
        suggestion.chords[m] = rewritten[start : start + n_slots]
        start += n_slots


def apply_song_form_to_suggestions(
    suggestions: list[Suggestion],
    section: str | None,
    key_root: str,
    is_minor: bool,
    cadence: str | None = None,
) -> None:
    """Run apply_song_form over each suggestion's whole chord sequence in place,
    keeping every measure's slot count. No-op when ``section`` is empty."""
    if not section:
        return
    for suggestion in suggestions:
        rewrite_chord_sequence(
            suggestion, lambda chords: apply_song_form(chords, section, key_root, is_minor, cadence),
        )


def _apply_cadence(
    chords: list[str],
    cadence_type: str,
    key_root: int,
    is_minor: bool,
    flats: bool,
) -> list[str]:
    """Apply cadence pattern to last 2 chords (last 4 for a turnaround)."""
    result = list(chords)

    if cadence_type == "authentic":
        # V → I (Perfect Authentic Cadence). Final tonic is a bare
        # triad (maj / min) — Imaj7 leaves a residual 7th tension that
        # reads "still going" rather than "resolved", which is the
        # opposite of what an authentic cadence should communicate.
        result[-2] = _make_chord(key_root, 7, "7", flats)
        tonic_q = "min" if is_minor else "maj"
        result[-1] = _make_chord(key_root, 0, tonic_q, flats)

    elif cadence_type == "plagal":
        # IV → I. Penultimate IV keeps the 7th colour (it's still a
        # tension chord); the resolving I is a bare triad for the same
        # reason as authentic cadence.
        iv_q = "m7" if is_minor else "maj7"
        result[-2] = _make_chord(key_root, 5, iv_q, flats)
        tonic_q = "min" if is_minor else "maj"
        result[-1] = _make_chord(key_root, 0, tonic_q, flats)

    elif cadence_type == "half":
        # x → V (half cadence, keep penultimate chord)
        result[-1] = _make_chord(key_root, 7, "7", flats)

    elif cadence_type == "half_iv_v":
        # IV → V (bridge ending)
        iv_q = "m7" if is_minor else "maj7"
        result[-2] = _make_chord(key_root, 5, iv_q, flats)
        result[-1] = _make_chord(key_root, 7, "7", flats)

    elif cadence_type == "turnaround":
        # I → vi → ii → V7 back to the top; minor: i → VI → iiø → V7.
        # Under 4 chords there is only room for the V7.
        if len(result) >= 4:
            if is_minor:
                pattern = [(0, "m7"), (8, "maj7"), (2, "m7b5"), (7, "7")]
            else:
                pattern = [(0, "maj7"), (9, "m7"), (2, "m7"), (7, "7")]
            result[-4:] = [_make_chord(key_root, iv, q, flats) for iv, q in pattern]
        else:
            result[-1] = _make_chord(key_root, 7, "7", flats)

    return result


# ── Rule definitions per section ──────────────────────────────────────

def _tonic_start(is_minor: bool) -> tuple[int, str]:
    """I or i chord."""
    return (0, "m7" if is_minor else "maj7")


def _iv_start(is_minor: bool) -> tuple[int, str]:
    """IV chord (bridge convention)."""
    return (5, "m7" if is_minor else "maj7")


def _tonic_end(is_minor: bool) -> tuple[int, str]:
    """Final tonic resolution — bare triad, never the 7th. A trailing
    Imaj7 / im7 keeps a colour tone hanging in the air; an unembellished
    I / i lands the song."""
    return (0, "min" if is_minor else "maj")


SECTION_RULES: dict[str, dict] = {
    "intro": {
        # 54% start on I
        "force_start": _tonic_start,
        # No forced cadence — intro flows into verse
    },
    "verse": {
        # 34% start on I — don't force, allow variety
        # End: resolve to tonic (V→I)
        "cadence": "authentic",
    },
    "chorus": {
        # End: resolve to tonic (V→I, 11%)
        "cadence": "authentic",
    },
    "bridge": {
        # Start on IV (11%) or vi — subdominant area
        "force_start": _iv_start,
        # End on V (19%) — half cadence preparing chorus return
        "cadence": "half_iv_v",
    },
    "outro": {
        # Strong resolution to I (31%)
        "cadence": "plagal",
        "force_end": _tonic_end,
    },
    "pre_chorus": {
        # Hang on V (half cadence) so the chorus lands as the arrival
        "cadence": "half",
    },
    "post_chorus": {
        # Static two-bar loop, no cadence
        "loop": 4,
    },
    "drop": {
        "loop": 4,
    },
    "breakdown": {
        # Static one-bar vamp, no cadence
        "loop": 2,
    },
    "vamp": {
        "loop": 2,
    },
    "turnaround": {
        # Last 4 chords: I → vi → ii → V7 (minor: i → VI → iiø → V7)
        "cadence": "turnaround",
    },
    "tag": {
        # Close like an outro, but with V → I
        "cadence": "authentic",
        "force_end": _tonic_end,
    },
    # Jazz head / A section resolve like verse and chorus; B section is the bridge
    "head": {
        "cadence": "authentic",
    },
    "a_section": {
        "cadence": "authentic",
    },
    "b_section": {
        "force_start": _iv_start,
        "cadence": "half_iv_v",
    },
    # Interlude and solo keep the model's own phrase
    "interlude": {},
    "solo": {},
}

# Sections whose rule resolves to the tonic (verse, chorus, outro, tag, head,
# A section). Voice leading must leave that final tonic in root position:
# an inverted I after the V7 does not resolve.
ROOT_POSITION_ENDINGS = frozenset(
    section for section, rules in SECTION_RULES.items()
    if rules.get("cadence") in ("authentic", "plagal")
)
