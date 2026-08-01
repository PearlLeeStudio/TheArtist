"""Song form post-processor for chord generation.

Applies section-aware harmonic rules derived from statistical analysis
of 679,807 songs (Chordonomicon dataset). The model generates chords
freely; this module adjusts first/last chords and cadences to match
conventional song form expectations.

Rules are genre-agnostic and based on relative scale degrees.
"""

from __future__ import annotations

NOTES_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
NOTES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

NOTE_TO_SEMI: dict[str, int] = {}
for _i, _n in enumerate(NOTES_FLAT):
    NOTE_TO_SEMI[_n] = _i
for _i, _n in enumerate(NOTES_SHARP):
    NOTE_TO_SEMI[_n] = _i

_SHARP_KEYS = frozenset({"B", "F#", "C#"})


def _note(semi: int, flats: bool = True) -> str:
    return NOTES_FLAT[semi % 12] if flats else NOTES_SHARP[semi % 12]


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


def _is_minor_chord(chord: str) -> bool:
    parsed = _parse_chord_root(chord)
    if not parsed:
        return False
    q = parsed[1]
    return q.startswith("m") and not q.startswith("maj")


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

def apply_song_form(
    chords: list[str],
    section: str,
    key: str,
    is_minor: bool = False,
) -> list[str]:
    """Post-process generated chords based on song section conventions.

    Args:
        chords: Generated chord names (flat list for all selected measures)
        section: One of 'intro', 'verse', 'chorus', 'bridge', 'outro'
        key: Key root note (e.g. 'C', 'Bb')
        is_minor: Whether the key is minor

    Returns:
        Adjusted chord list (same length)
    """
    if not chords or section not in SECTION_RULES:
        return chords

    key_root = NOTE_TO_SEMI.get(key)
    if key_root is None:
        return chords
    flats = key not in _SHARP_KEYS

    result = list(chords)
    rules = SECTION_RULES[section]

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

    return result


def _apply_cadence(
    chords: list[str],
    cadence_type: str,
    key_root: int,
    is_minor: bool,
    flats: bool,
) -> list[str]:
    """Apply cadence pattern to last 2 chords."""
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
}
