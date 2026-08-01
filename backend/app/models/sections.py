"""Song-form section vocabulary shared by the API, the song-form rules and the planner.

Mirrored in frontend/src/engine/songForm.ts; keep the two lists identical.
Chord counts assume the editor's fixed two chord slots per measure.
"""

from math import ceil
from typing import Literal, get_args

SectionType = Literal[
    "intro", "verse", "pre_chorus", "chorus", "post_chorus", "bridge", "outro",
    "interlude", "solo", "breakdown", "drop", "vamp", "turnaround", "tag",
    "head", "a_section", "b_section",
]

# id -> (label, dropdown group, default chord count)
SECTION_INFO: dict[str, tuple[str, str, int]] = {
    "intro": ("Intro", "common", 8),
    "verse": ("Verse", "common", 16),
    "pre_chorus": ("Pre-Chorus", "common", 8),
    "chorus": ("Chorus", "common", 16),
    "post_chorus": ("Post-Chorus", "common", 8),
    "bridge": ("Bridge", "common", 16),
    "outro": ("Outro", "common", 8),
    "interlude": ("Interlude", "instrumental", 8),
    "solo": ("Solo", "instrumental", 16),
    "breakdown": ("Breakdown", "instrumental", 16),
    "drop": ("Drop", "instrumental", 16),
    "vamp": ("Vamp", "instrumental", 4),
    "turnaround": ("Turnaround", "instrumental", 4),
    "tag": ("Tag", "instrumental", 8),
    "head": ("Head", "jazz", 64),
    "a_section": ("A Section", "jazz", 16),
    "b_section": ("B Section", "jazz", 16),
}

SECTION_TYPES: tuple[str, ...] = tuple(SECTION_INFO)
assert set(SECTION_TYPES) == set(get_args(SectionType)), "SectionType and SECTION_INFO drifted"

CHORDS_PER_BAR = 2
MAX_SECTION_CHORDS = 128
MAX_SECTION_BARS = MAX_SECTION_CHORDS // CHORDS_PER_BAR  # 64


def section_label(section: str) -> str:
    return SECTION_INFO[section][0]


def bars_for_chords(chord_count: int) -> int:
    return ceil(chord_count / CHORDS_PER_BAR)


def section_title(section: str, bars: int) -> str:
    """Default song title for a single section, e.g. 'Chorus — 8 bars'."""
    unit = "bar" if bars == 1 else "bars"
    return f"{section_label(section)} — {bars} {unit}"
