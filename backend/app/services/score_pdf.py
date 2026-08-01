"""Deterministic extraction of supported vector notation, without a model call.

The PDF text origins, not font bounding boxes, locate music glyphs. Staff lines,
barlines and stems remain vector geometry. Unsupported notation is reported; this
is not a raster OMR or a claim to reconstruct arbitrary engraving software.
Geometric tolerances are multiples of a staff's spacing, the gap between its lines.
"""

from __future__ import annotations

import io
import re
from bisect import bisect_left, bisect_right
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from statistics import median
from typing import Any

import pdfplumber

MAX_PDF_BYTES = 8 * 1024 * 1024
MAX_PDF_PAGES = 50
MAX_PDF_MEASURES = 512
MAX_PAGE_OBJECTS = 60_000
MAX_TOTAL_GLYPHS = 250_000
_LIMIT_MESSAGE = "The vector score exceeds the bounded glyph/geometry limit."


class ScorePDFError(ValueError):
    pass


@dataclass
class NoteEvent:
    staff: int
    midi: int
    spelling: str
    onset: float
    duration: float | None
    x: float
    y: float
    confidence: str = "medium"
    tie_continuation: bool = False
    tie_to_next: bool = False
    written_midi: int | None = None
    written_spelling: str | None = None
    octave_shift: int = 0


@dataclass
class PrintedChord:
    symbol: str
    onset: float
    x: float


@dataclass
class ScoreMeasure:
    index: int
    page: int
    system: int
    number: int
    time_signature: tuple[int, int]
    notes: list[NoteEvent] = field(default_factory=list)
    chords: list[PrintedChord] = field(default_factory=list)
    no_chord: bool = False
    repeat_markers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    staff_warnings: dict[int, list[str]] = field(default_factory=dict)
    rhythm_exact: bool = True
    key_signature: int | None = None


@dataclass
class ParsedScore:
    title: str = ""
    key_signature: int | None = None
    key: str | None = None
    time_signature: tuple[int, int] | None = None
    measures: list[ScoreMeasure] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    unsupported_counts: dict[str, int] = field(default_factory=dict)
    supported: bool = False
    complete: bool = True
    resolved_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class _Glyph:
    text: str
    font: str
    x: float
    y: float
    width: float
    size: float
    standard: bool = False


@dataclass
class _Staff:
    top: float
    bottom: float
    left: float
    right: float
    spacing: float
    glyphs: list[_Glyph] = field(default_factory=list)


@dataclass
class _OctaveMark:
    x: float
    y: float
    width: float
    amount: int
    direction: int
    glyph_ids: set[int]


@dataclass
class _OctaveSpan:
    staff: _Staff
    start: float
    end: float
    shift: int
    mark: _OctaveMark


@dataclass
class _Page:
    """One page's glyphs and geometry, shared by every system and measure on it."""
    number: int
    glyphs: list[_Glyph]
    staffs: list[_Staff]
    verticals: list[dict]
    shapes: list[dict]
    rects: list[dict]
    octave_spans: list[_OctaveSpan]
    unresolved_octaves: list[_OctaveMark]
    resolved_octave_ids: set[int]
    tie_links: dict[int, int]
    resolved_tie_shapes: set[int]
    notes_by_glyph: dict[int, NoteEvent] = field(default_factory=dict)
    duration_geometry: dict[int, _DurationGeometry] = field(default_factory=dict)


class _DurationGeometry:
    """Index stems once per staff, and reuse a chord's stem/beam duration work."""

    def __init__(self, staff: _Staff, verticals: list[dict], shapes: list[dict]):
        self.stems = sorted(enumerate(verticals), key=lambda item: item[1]["x0"])
        self.xs = [line["x0"] for _, line in self.stems]
        self.beams = [shape for shape in shapes if shape.get("fill") and not _is_curved(shape)
                      and shape["x1"] - shape["x0"] >= staff.spacing * .65
                      and shape["y1"] - shape["y0"] <= staff.spacing * 1.8]
        self.flags = [g for g in staff.glyphs if g.text in _FLAGS]
        self.dots = [g for g in staff.glyphs if g.text in _AUGMENTATION_DOTS]
        self.beam_counts: dict[int, int] = {}

    def near_stems(self, g: _Glyph, staff: _Staff) -> list[dict]:
        start = bisect_left(self.xs, g.x - staff.spacing * .3)
        end = bisect_right(self.xs, g.x + staff.spacing * 1.8)
        # Preserve source order for equally close stems, as the original reader did.
        return [line for _, line in sorted(self.stems[start:end])]


@dataclass
class _ReadState:
    """Notation context that carries across systems and pages, in written order."""
    parsed: ParsedScore
    max_measures: int
    clefs: dict[int, str] = field(default_factory=dict)
    keys: dict[int, int] = field(default_factory=dict)
    meter: tuple[int, int] = (4, 4)
    meter_known: bool = False
    counts: Counter[str] = field(default_factory=Counter)
    resolved_counts: Counter[str] = field(default_factory=Counter)

    def mark_incomplete(self, name: str, amount: int = 1) -> None:
        self.counts[name] += amount
        self.parsed.complete = False


_NOTEHEADS = {"œ": 1.0, "˙": 2.0, "w": 4.0, "\ue0a4": 1.0, "\ue0a3": 2.0, "\ue0a2": 4.0}
_CLEFS = {"&": "treble", "?": "bass", "B": "alto", "\ue050": "treble", "\ue062": "bass", "\ue05c": "alto"}
_ACCIDENTALS = {"b": -1, "♭": -1, "\ue260": -1, "#": 1, "♯": 1, "\ue262": 1,
                "n": 0, "♮": 0, "\ue261": 0, "𝄪": 2, "\ue263": 2, "𝄫": -2, "\ue264": -2}
_RESTS = {"Œ": 1.0, "‰": .5, "≈": .25, "Ó": 2.0, "∑": None,
          "\ue4e1": None, "\ue4e2": 2.0, "\ue4e3": 1.0, "\ue4e4": .5, "\ue4e5": .25,
          "\ue4e6": .125, "\ue4e7": .0625, "\ue4e8": .03125}
_FLAGS = {"j": .5, "J": .5, "r": .25, "R": .25,
          "\ue240": .5, "\ue241": .5, "\ue242": .25, "\ue243": .25,
          "\ue244": .125, "\ue245": .125, "\ue246": .0625, "\ue247": .0625}
_REPEATS = {"%", "÷", "𝄆", "𝄇", "\ue500", "\ue501", "\ue502"}
_MUSIC_FONTS = re.compile(r"Maestro|Bravura|Petaluma|Leland|Gonville|MuseJazz", re.I)
_MAESTRO = re.compile("Maestro", re.I)
# SMuFL's explicit alta/bassa signs; bare 8/15 use the bracket's staff side.
# https://smufl.formats.music/latest/tables/octaves.html
_OCTAVE_GLYPHS = {"\ue510": (12, 0), "\ue511": (12, 1), "\ue512": (12, -1), "\ue513": (12, -1),
                  "\ue514": (24, 0), "\ue515": (24, 1), "\ue516": (24, -1), "\ue51c": (12, -1),
                  "\ue51d": (24, -1), "\ue517": (36, 0), "\ue518": (36, 1), "\ue519": (36, -1),
                  "\ue51e": (36, -1), "\ue51f": (0, -1),
                  "𝄶": (12, 1), "𝄷": (12, -1), "𝄸": (24, 1), "𝄹": (24, -1)}
_LETTERS = "CDEFGAB"
_PCS = (0, 2, 4, 5, 7, 9, 11)
_CHORD_SYMBOL = re.compile(r"^[A-G](?:#|b)?(?:(?:maj|min|dim|aug|sus|add|omit|no|alt|m|M|Δ|ø|°|[0-9#+−-]|\([^)]*\)))*(?:/[A-G](?:#|b)?)?$")
_NO_CHORD = re.compile(r"\(?N\.?C\.?\)?", re.I)
_OCTAVE_LABEL = re.compile(r"8(?:va|vb|ba)|15m[ab]|22m[ab]")
_OCTAVE_TEXT = re.compile(r"(?:8v[ab]|8ba|15m[ab]|22m[ab]|ottava)", re.I)
_SPELLING = re.compile(r"([A-G](?:#{1,2}|b{1,2})?)(-?\d+)")
_NOTEHEADS_AND_CLEFS = (*_NOTEHEADS, *_CLEFS)
# Noteheads typed in a font that is not recognized as music.
_LEGACY_FONT_NOTEHEADS = ("œ", "˙")
_KEY_SIGNATURE_ACCIDENTALS = ("b", "#", "n", "♭", "♯", "♮", "\ue260", "\ue261", "\ue262")
_COMMON_TIME = ("c", "C", "𝄴", "\ue08a")
_CUT_TIME = ("¢", "𝄵", "\ue08b")
_METER_DENOMINATORS = (1, 2, 4, 8, 16, 32)
_AUGMENTATION_DOTS = (".", "\ue1e7")
_TUPLET_DIGITS = ("3", "5", "6", "7", "9")
# Diatonic step (octave * 7 + letter index) of each clef's bottom line: E4, G2, F3 and D3.
_CLEF_BOTTOM_STEP = {"treble": 30, "bass": 18, "alto": 24, "tenor": 22}
_SHARP_ORDER = "FCGDAEB"
_FLAT_ORDER = "BEADGCF"
_ACCIDENTAL_SUFFIX = {-2: "bb", -1: "b", 0: "", 1: "#", 2: "##"}
# SMuFL code point ranges, bounds inclusive. Composite notes stop before E1E7, the augmentation dot.
_SMUFL = range(0xE000, 0xEFFF + 1)
_SMUFL_CLEFS = range(0xE050, 0xE06F + 1)
_SMUFL_TIME_DIGITS = range(0xE080, 0xE089 + 1)
_SMUFL_NOTEHEADS = range(0xE0A0, 0xE0FF + 1)
_SMUFL_COMPOSITE_NOTES = range(0xE1D0, 0xE1E6 + 1)


def _has_code_point(text: str, code_points: range) -> bool:
    return any(ord(char) in code_points for char in text)


def _is_music(g: _Glyph) -> bool:
    """Music-font, standard SMuFL or octave-sign glyphs; everything else is text."""
    return bool(_MUSIC_FONTS.search(g.font)) or g.standard or g.text in _OCTAVE_GLYPHS \
        or _has_code_point(g.text, _SMUFL)


def _is_skewed(a: float, b: float, c: float, d: float) -> bool:
    """A text matrix sheared beyond 5% of its scale draws rotated or slanted glyphs."""
    return abs(b) + abs(c) > .05 * max(abs(a), abs(d), 1)


def _is_curved(shape: dict) -> bool:
    """A path with a Bezier segment, as ties and slurs have and straight beams do not."""
    return any(op[0] == "c" for op in shape.get("path", []))


def _clef(g: _Glyph, staff: _Staff) -> str:
    selected = _CLEFS[g.text]
    # A C clef three spaces above the bottom line marks the fourth line: a tenor clef.
    if selected == "alto" and round((g.y - staff.bottom) / staff.spacing) == 3:
        return "tenor"
    return selected


def _glyphs(chars: list[dict]) -> list[_Glyph]:
    """Glyphs at their text origins, skipping sheared or malformed matrices.

    SMuFL time-signature digits become plain digits.
    """
    result = []
    for char in chars:
        matrix = char.get("matrix")
        if not matrix or len(matrix) != 6:
            continue
        a, b, c, d, x, y = matrix
        if _is_skewed(a, b, c, d):
            continue
        text = char.get("text", "")
        standard = _has_code_point(text, _SMUFL)
        if len(text) == 1 and ord(text) in _SMUFL_TIME_DIGITS:
            text = str(ord(text) - _SMUFL_TIME_DIGITS.start)
        result.append(_Glyph(text, char.get("fontname", ""), float(x), float(y),
                             abs(float(char.get("x1", x)) - float(char.get("x0", x))),
                             max(abs(float(a)), abs(float(d)), float(char.get("size", 0))), standard))
    return result


def _cluster(values: list[float], tolerance: float) -> list[float]:
    """Medians of sorted values grouped while each stays within tolerance of its group's median."""
    groups: list[list[float]] = []
    for value in sorted(values):
        if groups and abs(value - median(groups[-1])) <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [median(group) for group in groups]


def _rows(glyphs: Iterable[_Glyph], joins: Callable[[float], bool]) -> list[list[_Glyph]]:
    """Text rows from the top down; a glyph joins the last row when joins(distance to its median height)."""
    rows: list[list[_Glyph]] = []
    for g in sorted(glyphs, key=lambda g: -g.y):
        if rows and joins(abs(g.y - median(item.y for item in rows[-1]))):
            rows[-1].append(g)
        else:
            rows.append([g])
    return rows


def _runs(glyphs: Iterable[_Glyph], joins: Callable[[float], bool]) -> list[list[_Glyph]]:
    """Runs from left to right; a glyph joins the last run when joins(its gap after the run's right edge)."""
    runs: list[list[_Glyph]] = []
    for g in sorted(glyphs, key=lambda g: g.x):
        if runs and joins(g.x - max(item.x + item.width for item in runs[-1])):
            runs[-1].append(g)
        else:
            runs.append([g])
    return runs


def _nearest_staff(staffs: list[_Staff], y: float) -> _Staff:
    """The staff vertically closest to y (distance zero inside it); the first one wins a tie."""
    return min(staffs, key=lambda staff: max(staff.bottom - y, y - staff.top, 0))


def _staff_step(y: float, staff: _Staff) -> float:
    """Height above the bottom line in half spacings: lines are even steps, spaces odd."""
    return (y - staff.bottom) / (staff.spacing / 2)


def _in_bar(x: float, left: float, right: float, spacing: float) -> bool:
    """Bar membership shifted left, so a symbol just before a barline belongs to the next bar."""
    return left - spacing * .2 <= x < right - spacing * .2


def _horizontal_rows(lines: list[dict], width: float) -> list[dict]:
    """Long horizontal rules by height, top first, each with its longest merged segment's extent."""
    horizontal = [line for line in lines if abs(line["y1"] - line["y0"]) < .3
                  and abs(line["x1"] - line["x0"]) > max(35, width * .08)]
    rows: list[dict] = []
    for line in sorted(horizontal, key=lambda item: -item["y0"]):
        segment = (min(line["x0"], line["x1"]), max(line["x0"], line["x1"]))
        if rows and abs(rows[-1]["y"] - line["y0"]) < .35:
            rows[-1]["segments"].append(segment)
        else:
            rows.append({"y": line["y0"], "segments": [segment]})
    for row in rows:
        segments = sorted(row["segments"])
        merged = [list(segments[0])]
        for left, right in segments[1:]:
            if left - merged[-1][1] <= 2:
                merged[-1][1] = max(merged[-1][1], right)
            else:
                merged.append([left, right])
        row["left"], row["right"] = max(merged, key=lambda segment: segment[1] - segment[0])
    return rows


def _staffs(lines: list[dict], width: float) -> list[_Staff]:
    """Five consecutive rows with even, plausible gaps and a shared wide extent form a staff."""
    rows = _horizontal_rows(lines, width)
    result = []
    index = 0
    while index + 4 < len(rows):
        group = rows[index:index + 5]
        gaps = [group[j]["y"] - group[j + 1]["y"] for j in range(4)]
        spacing = median(gaps)
        left, right = max(row["left"] for row in group), min(row["right"] for row in group)
        if 1.5 <= spacing <= 16 and max(abs(gap - spacing) for gap in gaps) <= max(.25, spacing * .12) \
                and right - left >= max(40, width * .16):
            result.append(_Staff(group[0]["y"], group[-1]["y"], left, right, spacing))
            index += 5
        else:
            index += 1
    return result


def _verticals(lines: list[dict]) -> list[dict]:
    """Vertical rules: barline and stem candidates."""
    return [line for line in lines if abs(line["x1"] - line["x0"]) < .3
            and abs(line["y1"] - line["y0"]) > 2]


def _systems(staffs: list[_Staff], verticals: list[dict]) -> list[list[_Staff]]:
    """Group staves top to bottom; close staves joined by a vertical rule share a system."""
    systems: list[list[_Staff]] = []
    for staff in staffs:
        if not systems:
            systems.append([staff])
            continue
        previous = systems[-1][-1]
        gap = previous.bottom - staff.top
        connected = any(min(line["y0"], line["y1"]) <= staff.bottom + staff.spacing * .3
                        and max(line["y0"], line["y1"]) >= previous.top - previous.spacing * .3
                        for line in verticals)
        if gap < 15 * max(previous.spacing, staff.spacing) and connected:
            systems[-1].append(staff)
        else:
            systems.append([staff])
    return systems


def _assign_glyphs(staffs: list[_Staff], glyphs: list[_Glyph]) -> None:
    """Give each music glyph to its nearest staff when it lies within that staff's reach."""
    for g in glyphs:
        if not _is_music(g):
            continue
        nearest = _nearest_staff(staffs, g.y)
        if nearest.left - nearest.spacing * 2 <= g.x <= nearest.right + nearest.spacing * 2 \
                and nearest.bottom - nearest.spacing * 9 <= g.y <= nearest.top + nearest.spacing * 9:
            nearest.glyphs.append(g)


def _key_signature(staff: _Staff, start: float | None = None, end: float | None = None) -> tuple[int | None, set[int]]:
    """Signed sharp (+) or flat (-) count of a key signature at the start of [start, end), and its glyph ids.

    Accidentals count only before the first notehead and close together; mixed signs
    or an implausible count leave the signature unread (None). Naturals alone give 0.
    """
    start = staff.left if start is None else start
    end = staff.right if end is None else end
    heads = [g.x for g in staff.glyphs if g.text in _NOTEHEADS and start <= g.x < end]
    first = min(heads, default=staff.right)
    candidates = sorted((g for g in staff.glyphs if g.text in _KEY_SIGNATURE_ACCIDENTALS
                         and start <= g.x < min(first - staff.spacing * 1.5, start + staff.spacing * 13, end)
                         and staff.bottom - staff.spacing <= g.y <= staff.top + staff.spacing), key=lambda g: g.x)
    group = []
    for g in candidates:
        if group and g.x - group[-1].x > staff.spacing * 2:
            break
        group.append(g)
    if not group:
        return None, set()
    altered = [g for g in group if _ACCIDENTALS[g.text] != 0]
    signs = {_ACCIDENTALS[g.text] for g in altered}
    if len(signs) > 1 or len(altered) > 7 or len(group) > 14:
        return None, set()
    return (next(iter(signs)) * len(altered) if signs else 0), {id(g) for g in group}


def _has_empty_key_signature(staff: _Staff) -> bool:
    """A clef but no accidental before the first notehead: an explicitly empty key signature."""
    first_head = min((g.x for g in staff.glyphs if g.text in _NOTEHEADS), default=staff.right)
    header = [g for g in staff.glyphs if g.x < first_head and staff.bottom - staff.spacing <= g.y <= staff.top + staff.spacing]
    return any(g.text in _CLEFS for g in header) and not any(g.text in _ACCIDENTALS for g in header)


def _meters(staff: _Staff) -> list[tuple[float, tuple[int, int]]]:
    """Time signatures on a staff as (x, (count, unit)), left to right."""
    candidates = [g for g in staff.glyphs if staff.bottom - staff.spacing * .4 <= g.y <= staff.top + staff.spacing * .4]
    result = [(g.x, (4, 4)) for g in candidates if g.text in _COMMON_TIME]
    result += [(g.x, (2, 2)) for g in candidates if g.text in _CUT_TIME]
    digits = [g for g in candidates if g.text.isdigit()]
    center = (staff.top + staff.bottom) / 2
    for group in _runs(digits, lambda gap: gap < staff.spacing):
        numerator = sorted((g for g in group if g.y > center + staff.spacing * .2), key=lambda g: g.x)
        denominator = sorted((g for g in group if g.y < center - staff.spacing * .2), key=lambda g: g.x)
        if numerator and denominator:
            top = int("".join(g.text for g in numerator))
            bottom = int("".join(g.text for g in denominator))
            if 1 <= top <= 32 and bottom in _METER_DENOMINATORS:
                result.append((min(g.x for g in group), (top, bottom)))
    return sorted(result)


def _boundaries(system: list[_Staff], verticals: list[dict]) -> list[float]:
    """Measure boundaries: barlines across the system, plus its open edges where notes need them.

    A vertical that crosses one staff next to a notehead is a stem, not a barline.
    """
    spacing = median(staff.spacing for staff in system)
    left, right = max(staff.left for staff in system), min(staff.right for staff in system)
    values = []
    for line in verticals:
        x = line["x0"]
        if not left - spacing <= x <= right + spacing:
            continue
        crossed = sum(min(line["y0"], line["y1"]) <= staff.bottom + spacing * .3
                      and max(line["y0"], line["y1"]) >= staff.top - spacing * .3 for staff in system)
        if crossed:
            near_head = any(abs(g.x - x) < spacing * 1.7 for staff in system for g in staff.glyphs if g.text in _NOTEHEADS)
            if crossed >= 2 or not near_head:
                values.append(x)
    values = _cluster(values, spacing * .65)
    if not values or values[0] > left + spacing:
        values.insert(0, left)
    if values[-1] < right - spacing and any(g.x > values[-1] + spacing and g.text in (*_NOTEHEADS, *_RESTS)
                                          for staff in system for g in staff.glyphs):
        values.append(right)
    return values


def _boxed(g: _Glyph, rects: list[dict], spacing: float) -> bool:
    """Inside a small rectangle, such as a boxed rehearsal mark."""
    return any(rect["x0"] - 1 <= g.x <= rect["x1"] + 1 and rect["y0"] <= g.y <= rect["y1"]
               and rect["x1"] - rect["x0"] < spacing * 8 and rect["y1"] - rect["y0"] < spacing * 8
               for rect in rects)


def _chord_text(system: list[_Staff], glyphs: list[_Glyph], rects: list[dict]) -> list[tuple[float, str]]:
    """Printed chord symbols and N.C. above a system's top staff, as (x, symbol)."""
    staff = system[0]
    candidates = [g for g in glyphs if (not _is_music(g) or (g.text in _ACCIDENTALS and g.size < staff.spacing * 3.5))
                  and staff.top + staff.spacing * .5 <= g.y <= staff.top + staff.spacing * 6
                  and staff.left <= g.x <= staff.right and not _boxed(g, rects, staff.spacing)]
    result = []
    for row in _rows(candidates, lambda distance: distance < staff.spacing * 1.2):
        for group in _runs(row, lambda gap: gap <= staff.spacing * 1.15):
            text = "".join(g.text for g in group).replace(" ", "").replace("♯", "#").replace("♭", "b").replace("−", "-")
            if _NO_CHORD.fullmatch(text):
                result.append((min(g.x for g in group), "N.C."))
            elif _CHORD_SYMBOL.fullmatch(text):
                result.append((min(g.x for g in group), text))
    return sorted(result)


def _pitch(y: float, staff: _Staff, clef: str, signature: int, accidental: int | None) -> tuple[int, str]:
    """MIDI pitch and spelling at a staff height; without a written accidental the key signature applies."""
    diatonic = _CLEF_BOTTOM_STEP[clef] + round(_staff_step(y, staff))
    octave, letter = divmod(diatonic, 7)
    if accidental is None:
        altered = _SHARP_ORDER[:signature] if signature >= 0 else _FLAT_ORDER[:abs(signature)]
        accidental = (1 if signature > 0 else -1) if _LETTERS[letter] in altered else 0
    midi = (octave + 1) * 12 + _PCS[letter] + accidental
    return midi, _LETTERS[letter] + _ACCIDENTAL_SUFFIX[accidental] + str(octave)


def _stem(g: _Glyph, staff: _Staff, verticals: list[dict]) -> dict | None:
    """The nearest long vertical beside a notehead that reaches its height."""
    stems = [line for line in verticals if g.x - staff.spacing * .3 <= line["x0"] <= g.x + staff.spacing * 1.8
             and min(line["y0"], line["y1"]) - staff.spacing * .5 <= g.y <= max(line["y0"], line["y1"]) + staff.spacing * .5
             and abs(line["y1"] - line["y0"]) >= staff.spacing * 1.8]
    return min(stems, key=lambda line: abs(line["x0"] - g.x), default=None)


def _beam_count(stem: dict, staff: _Staff, shapes: list[dict]) -> int:
    """Distinct levels of filled straight bars that cross a stem."""
    beam_levels = []
    for shape in shapes:
        if not shape.get("fill") or _is_curved(shape):
            continue
        if shape["x1"] - shape["x0"] < staff.spacing * .65 or shape["y1"] - shape["y0"] > staff.spacing * 1.8:
            continue
        if shape["x0"] - staff.spacing * .1 <= stem["x0"] <= shape["x1"] + staff.spacing * .1 \
                and min(stem["y0"], stem["y1"]) - staff.spacing * .4 <= (shape["y0"] + shape["y1"]) / 2 <= max(stem["y0"], stem["y1"]) + staff.spacing * .4:
            beam_levels.append((shape["y0"] + shape["y1"]) / 2)
    return len(_cluster(beam_levels, staff.spacing * .3))


def _duration(g: _Glyph, staff: _Staff, geometry: _DurationGeometry) -> float | None:
    """Beats of a notehead: flags or beams on a filled head's stem shorten it, dots lengthen it."""
    base = _NOTEHEADS[g.text]
    if base == 1:
        stem = _stem(g, staff, geometry.near_stems(g, staff))
        if stem is None:
            return None
        flags = [flag for flag in geometry.flags if abs(flag.x - stem["x0"]) <= staff.spacing
                 and min(stem["y0"], stem["y1"]) - staff.spacing <= flag.y <= max(stem["y0"], stem["y1"]) + staff.spacing]
        if flags:
            base = min(_FLAGS[flag.text] for flag in flags)
        else:
            if id(stem) not in geometry.beam_counts:
                geometry.beam_counts[id(stem)] = _beam_count(stem, staff, geometry.beams)
            beams = geometry.beam_counts[id(stem)]
            if beams > 4:  # Shorter than a 64th note is not read.
                return None
            base = 1 / (2 ** beams)
    dots = [dot for dot in geometry.dots if g.x + g.width <= dot.x <= g.x + g.width + staff.spacing * 2.5
            and abs(dot.y - g.y) <= staff.spacing * .8]
    if dots:  # One dot adds half the value; two or more add three quarters.
        count = min(2, len(_cluster([dot.x for dot in dots], staff.spacing * .3)))
        base *= 1.5 if count == 1 else 1.75
    return base


def _tie_pair(staff: _Staff, heads: list[_Glyph], shape: dict, first_y: float, last_y: float) \
        -> tuple[_Glyph, _Glyph] | None:
    """The best (source, target) notehead pair at a curve's ends; None when there is none or it is ambiguous."""
    sources = [g for g in heads if 0 <= shape["x0"] - g.x <= staff.spacing * 3
               and staff.spacing * .15 <= abs(first_y - g.y) <= staff.spacing * 1.5]
    targets = [g for g in heads if abs(shape["x1"] - g.x) <= staff.spacing * 1.5
               and staff.spacing * .15 <= abs(last_y - g.y) <= staff.spacing * 1.5]
    pairs = []
    for source in sources:
        for target in targets:
            if source.x >= target.x or abs(source.y - target.y) > staff.spacing * .15:
                continue
            if any(source.x < g.x < target.x and g.text in _CLEFS for g in staff.glyphs):
                continue
            if any(source.x + staff.spacing * 1.8 < head.x < target.x - staff.spacing * 1.8 for head in heads):
                continue
            score = abs(first_y - source.y) + abs(last_y - target.y) \
                + abs(shape["x0"] - source.x - source.width) * .25 + abs(shape["x1"] - target.x) * .25
            pairs.append((score, source, target))
    pairs.sort(key=lambda item: item[0])
    if not pairs or (len(pairs) > 1 and pairs[1][0] - pairs[0][0] < staff.spacing * .25):
        return None
    _, source, target = pairs[0]
    return source, target


def _tie_links(staff: _Staff, shapes: list[dict], page_height: float) -> tuple[dict[int, int], set[int]]:
    """Resolve only adjacent equal-position notes with aligned curve endpoints.

    A slur spanning intermediate notes, different staff positions, a partial
    system-end curve, or ambiguous chord endpoints remains unsupported.
    Returns target notehead id -> source notehead id, and the ids of resolved curves.
    """
    heads = [g for g in staff.glyphs if g.text in _NOTEHEADS]
    links: dict[int, int] = {}
    resolved: set[int] = set()
    for shape in shapes:
        path = shape.get("path", [])
        if not _is_curved(shape) or shape["x1"] - shape["x0"] < staff.spacing * 2:
            continue
        endpoints = [op[-1] for op in path if op[0] in ("m", "l", "c") and len(op) > 1]
        if not endpoints:
            continue
        first = min(endpoints, key=lambda point: point[0])
        last = max(endpoints, key=lambda point: point[0])
        # pdfplumber gives path points top-down; flip them to the bottom-up y of glyphs and staff lines.
        first_y, last_y = page_height - first[1], page_height - last[1]
        if abs(first_y - last_y) > staff.spacing * .3:
            continue
        pair = _tie_pair(staff, heads, shape, first_y, last_y)
        if pair is None:
            continue
        source, target = pair
        if id(target) in links and links[id(target)] != id(source):
            continue
        links[id(target)] = id(source)
        resolved.add(id(shape))
    return links, resolved


def _octave_marks(glyphs: list[_Glyph], spacing: float) -> list[_OctaveMark]:
    """Octave signs from SMuFL or Maestro glyphs and text such as 8va or 15mb, merged by position."""
    marks = []
    for g in glyphs:
        if g.text in _OCTAVE_GLYPHS:
            amount, direction = _OCTAVE_GLYPHS[g.text]
            marks.append(_OctaveMark(g.x, g.y, g.width, amount, direction, {id(g)}))
        elif g.text == "√" and _MAESTRO.search(g.font):
            marks.append(_OctaveMark(g.x, g.y, g.width, 12, 0, {id(g)}))
    text_glyphs = (g for g in glyphs if not _is_music(g) and g.text.strip())
    for row in _rows(text_glyphs, lambda distance: distance <= spacing * .8):
        for group in _runs(row, lambda gap: gap <= spacing * .8):
            text = "".join(g.text for g in group).replace(" ", "").strip("()").lower()
            if _OCTAVE_LABEL.fullmatch(text):
                amount = 12 if text.startswith("8") else (24 if text.startswith("15") else 36)
                direction = -1 if text.endswith(("vb", "ba", "mb")) else 0
                start = min(g.x for g in group)
                marks.append(_OctaveMark(start, median(g.y for g in group), max(g.x + g.width for g in group) - start,
                                         amount, direction, {id(g) for g in group}))
    unique: dict[tuple, _OctaveMark] = {}
    for mark in marks:
        key = (round(mark.x / (spacing * .2)), round(mark.y / (spacing * .2)), mark.amount, mark.direction)
        if key in unique:
            unique[key].glyph_ids.update(mark.glyph_ids)
        else:
            unique[key] = mark
    return list(unique.values())


def _bracket_owner(staffs: list[_Staff], mark: _OctaveMark, end: float, y: float) -> tuple[_Staff, int] | None:
    """The one staff a bracket line clearly belongs to, with +1 above or -1 below it; None when unclear."""
    owners = []
    for staff in staffs:
        if not staff.left - staff.spacing <= mark.x < end <= staff.right + staff.spacing:
            continue
        if y > staff.top + staff.spacing * .4:
            direction, gap = 1, y - staff.top
        elif y < staff.bottom - staff.spacing * .4:
            direction, gap = -1, staff.bottom - y
        else:
            continue
        if gap > staff.spacing * 10 or (mark.direction and mark.direction != direction):
            continue
        owners.append((gap / staff.spacing, staff, direction))
    owners.sort(key=lambda item: item[0])
    if not owners or (len(owners) > 1 and owners[1][0] - owners[0][0] <= 1):
        return None
    _, staff, direction = owners[0]
    return staff, direction


def _has_end_hook(lines: list[dict], staff: _Staff, end: float, y: float, direction: int) -> bool:
    """A short solid vertical at the bracket's end that turns toward its staff."""
    for hook in lines:
        if hook.get("dash") and hook["dash"][0]:
            continue
        low, high = min(hook["y0"], hook["y1"]), max(hook["y0"], hook["y1"])
        if abs(hook["x1"] - hook["x0"]) > staff.spacing * .08 \
                or abs(hook["x0"] - end) > staff.spacing * .3 \
                or not staff.spacing * .25 <= high - low <= staff.spacing * 3:
            continue
        if direction > 0 and abs(high - y) <= staff.spacing * .2 and low < y - staff.spacing * .2:
            return True
        if direction < 0 and abs(low - y) <= staff.spacing * .2 and high > y + staff.spacing * .2:
            return True
    return False


def _octave_spans(staffs: list[_Staff], glyphs: list[_Glyph], lines: list[dict]) \
        -> tuple[list[_OctaveSpan], list[_OctaveMark], set[int]]:
    """Read only a single-system dashed bracket with a correctly directed hook."""
    spacing = median(staff.spacing for staff in staffs)
    marks = _octave_marks(glyphs, spacing)
    dashed = [line for line in lines if line.get("dash") and line["dash"][0]
              and any(length > 0 for length in line["dash"][0])
              and abs(line["y1"] - line["y0"]) <= spacing * .08
              and line["x1"] - line["x0"] >= spacing * 3]
    spans = []
    for mark in marks:
        if mark.amount not in (12, 24):
            continue
        candidates = []
        for line in dashed:
            start, end, y = line["x0"], line["x1"], (line["y0"] + line["y1"]) / 2
            if start < mark.x or not -spacing <= start - mark.x - mark.width <= spacing * 6 \
                    or abs(y - mark.y) > spacing * 2:
                continue
            owner = _bracket_owner(staffs, mark, end, y)
            if owner is None:
                continue
            staff, direction = owner
            if _has_end_hook(lines, staff, end, y, direction):
                candidates.append(_OctaveSpan(staff, mark.x, end, mark.amount * direction, mark))
        unique = {(id(span.staff), round(span.start, 3), round(span.end, 3), span.shift): span for span in candidates}
        if len(unique) == 1:
            spans.append(next(iter(unique.values())))
    ambiguous = {id(first) for first in spans for second in spans if first is not second and first.staff is second.staff
                 and min(first.end, second.end) - max(first.start, second.start) > first.staff.spacing * .2}
    spans = [span for span in spans if id(span) not in ambiguous]
    resolved_ids = {identity for span in spans for identity in span.mark.glyph_ids}
    return spans, marks, resolved_ids


def _octave_shift(g: _Glyph, staff: _Staff, spans: list[_OctaveSpan]) -> int:
    return next((span.shift for span in spans if span.staff is staff
                 and span.start - staff.spacing * .1 <= g.x <= span.end + staff.spacing * .1), 0)


def _shift_spelling(spelling: str, shift: int) -> str:
    match = _SPELLING.fullmatch(spelling)
    if match is None:
        return spelling
    return match[1] + str(int(match[2]) + shift // 12)


def _warn(measure: ScoreMeasure, staff: int, message: str) -> None:
    """Record a staff's reading issue once; any issue makes the measure's rhythm provisional."""
    if message not in measure.warnings:
        measure.warnings.append(message)
    values = measure.staff_warnings.setdefault(staff, [])
    if message not in values:
        values.append(message)
    measure.rhythm_exact = False


def _rotated_music_count(chars: list[dict]) -> int:
    """Noteheads and clefs drawn rotated, slanted or mirrored, which staff positions cannot place."""
    count = 0
    for char in chars:
        matrix = char.get("matrix")
        if matrix and len(matrix) == 6 and (char.get("text") in _NOTEHEADS_AND_CLEFS):
            a, b, c, d, _, _ = matrix
            if _is_skewed(a, b, c, d) or a <= 0 or d <= 0:
                count += 1
    return count


def _title(glyphs: list[_Glyph], staff: _Staff) -> str:
    """The largest text well above the first staff, left to right; empty when there is none."""
    candidates = [g for g in glyphs if not _is_music(g) and g.y > staff.top + staff.spacing * 8 and g.text]
    if not candidates:
        return ""
    largest = max(g.size for g in candidates)
    chosen = sorted((g for g in candidates if g.size >= largest * .8), key=lambda g: g.x)
    return "".join(g.text for g in chosen).strip()[:120]


def _unsupported_glyphs(glyphs: list[_Glyph], staffs: list[_Staff]) -> tuple[tuple[str, list[_Glyph]], ...]:
    """Notation that is recognizable but not readable here, by unsupported-count name."""
    unknown_heads = [g for g in glyphs if not _is_music(g) and g.text in _LEGACY_FONT_NOTEHEADS
                     and any(staff.bottom - staff.spacing * 6 < g.y < staff.top + staff.spacing * 6 for staff in staffs)]
    unsupported_heads = [g for g in glyphs if _has_code_point(g.text, _SMUFL_NOTEHEADS) and g.text not in _NOTEHEADS]
    unsupported_clefs = [g for g in glyphs if _has_code_point(g.text, _SMUFL_CLEFS) and g.text not in _CLEFS]
    composite_notes = [g for g in glyphs if _has_code_point(g.text, _SMUFL_COMPOSITE_NOTES)]
    return (("unknown_legacy_music_font", unknown_heads), ("unsupported_notehead", unsupported_heads),
            ("unsupported_clef", unsupported_clefs), ("unsupported_composite_note", composite_notes))


def _page_tie_links(staffs: list[_Staff], shapes: list[dict], page_height: float) -> tuple[dict[int, int], set[int]]:
    links: dict[int, int] = {}
    resolved: set[int] = set()
    for staff in staffs:
        staff_links, staff_resolved = _tie_links(staff, shapes, page_height)
        links.update(staff_links)
        resolved.update(staff_resolved)
    return links, resolved


def _prepare_page(state: _ReadState, page_number: int, page: Any, chars: list[dict], lines: list[dict]) -> _Page | None:
    """Locate a page's staves, count what cannot be read, and resolve its octave spans and ties.

    A page without supported staff geometry is counted and returns None.
    """
    glyphs = _glyphs(chars)
    rotated = _rotated_music_count(chars)
    if rotated:
        state.mark_incomplete("rotated_music_glyph", rotated)
    staffs = _staffs(lines, float(page.width))
    if not staffs:
        state.mark_incomplete("pages_without_supported_staff_geometry")
        return None
    if not state.parsed.title:
        state.parsed.title = _title(glyphs, staffs[0])
    _assign_glyphs(staffs, glyphs)
    for name, items in _unsupported_glyphs(glyphs, staffs):
        if items:
            state.mark_incomplete(name, len(items))
    verticals = _verticals(lines)
    shapes = list(getattr(page, "curves", [])) + list(getattr(page, "rects", []))
    if len(chars) + len(lines) + len(shapes) > MAX_PAGE_OBJECTS:
        raise ScorePDFError(_LIMIT_MESSAGE)
    octave_spans, octave_marks, resolved_octave_ids = _octave_spans(staffs, glyphs, lines)
    unresolved_octaves = [mark for mark in octave_marks if not mark.glyph_ids <= resolved_octave_ids]
    state.resolved_counts["ottava_spans"] += len(octave_spans)
    state.counts["ottava"] += len(unresolved_octaves)
    tie_links, resolved_tie_shapes = _page_tie_links(staffs, shapes, float(page.height))
    return _Page(page_number, glyphs, staffs, verticals, shapes, list(getattr(page, "rects", [])), octave_spans,
                 unresolved_octaves, resolved_octave_ids, tie_links, resolved_tie_shapes)


def _apply_opening_key_signatures(state: _ReadState, system: list[_Staff]) -> set[int]:
    """Apply each staff's key signature at the start of a system; return its accidental glyph ids."""
    signature_ids: set[int] = set()
    for staff_index, staff in enumerate(system):
        signature, ids = _key_signature(staff)
        # A staff's first header with a clef and meter but no accidentals states an empty signature.
        if signature is None and staff_index not in state.keys and _meters(staff) and _has_empty_key_signature(staff):
            signature = 0
        signature_ids.update(ids)
        if signature is not None:
            state.keys[staff_index] = signature
            if state.parsed.key_signature is None:
                state.parsed.key_signature = signature
    return signature_ids


def _read_system(state: _ReadState, page: _Page, system_number: int, system: list[_Staff]) -> None:
    """Read a system's measures left to right, carrying meter and key signature changes forward."""
    bounds = _boundaries(system, page.verticals)
    if len(bounds) < 2:
        state.mark_incomplete("systems_without_measure_boundaries")
        return
    meter_events = _meters(system[0])
    chords = _chord_text(system, page.glyphs, page.rects)
    signature_ids = _apply_opening_key_signatures(state, system)
    for left, right in zip(bounds, bounds[1:]):
        if len(state.parsed.measures) >= state.max_measures:
            state.mark_incomplete("measure_limit")
            break
        for x, meter in meter_events:
            if x < right:
                state.meter = meter
                state.meter_known = True
        for staff_index, staff in enumerate(system):
            signature, ids = _key_signature(staff, left, right)
            if signature is not None:
                state.keys[staff_index] = signature
                signature_ids.update(ids)
        if state.parsed.time_signature is None and state.meter_known:
            state.parsed.time_signature = state.meter
        measure = _read_measure(state, page, system_number, system, left, right, chords, signature_ids)
        state.parsed.measures.append(measure)
    # The system's last meter, including one after its final barline, carries into the next system.
    for _, meter in meter_events:
        state.meter = meter


def _read_measure(state: _ReadState, page: _Page, system_number: int, system: list[_Staff], left: float,
                  right: float, chords: list[tuple[float, str]], signature_ids: set[int]) -> ScoreMeasure:
    """Read every staff of one measure, then attach its printed chords and octave warnings."""
    measure = ScoreMeasure(len(state.parsed.measures), page.number, system_number, len(state.parsed.measures) + 1,
                           state.meter, key_signature=state.keys.get(0))
    beats = state.meter[0] * 4 / state.meter[1]
    if not state.meter_known:
        measure.warnings.append("No readable time signature; 4/4 timing is provisional.")
        measure.rhythm_exact = False
        state.counts["missing_meter"] += 1
    onsets: list[tuple[float, float]] = []
    for staff_index, staff in enumerate(system):
        onsets += _read_staff(state, page, measure, staff_index, staff, left, right, beats, signature_ids)
    _attach_chords(measure, chords, left, right, system[0].spacing, onsets)
    _flag_octave_text(state, page, measure, system, left, right)
    if not measure.rhythm_exact:
        for note in measure.notes:
            note.confidence = "medium"
    measure.notes.sort(key=lambda note: (note.onset, note.staff, note.midi))
    return measure


def _read_staff(state: _ReadState, page: _Page, measure: ScoreMeasure, staff_index: int, staff: _Staff,
                left: float, right: float, beats: float, signature_ids: set[int]) -> list[tuple[float, float]]:
    """Read one staff of one measure into notes and rests; return (x, onset) for each note group and rest."""
    records = sorted((g for g in staff.glyphs if _in_bar(g.x, left, right, staff.spacing)), key=lambda g: g.x)
    clef, clef_shown = _starting_clef(staff, state.clefs.get(staff_index), left, right)
    if staff_index not in state.clefs and not clef_shown:
        _warn(measure, staff_index, "No supported clef was present; treble position is provisional.")
        state.counts["missing_clef"] += 1
    accidentals: dict[int, int] = {}  # Written accidentals by staff step, until the barline.
    note_records: list[tuple[_Glyph, NoteEvent]] = []
    rests = []
    for g in records:
        if g.text in _CLEFS:
            clef = _clef(g, staff)
        elif g.text in _ACCIDENTALS and id(g) not in signature_ids:
            step = round(_staff_step(g.y, staff))
            accidentals[step] = _ACCIDENTALS[g.text]
        elif g.text in _NOTEHEADS:
            note = _read_note(state, page, measure, staff_index, staff, g, clef, accidentals)
            if note is not None:
                note_records.append((g, note))
        elif g.text in _RESTS:
            rests.append((g.x, _RESTS[g.text] if _RESTS[g.text] is not None else beats))
        elif g.text in _REPEATS:
            measure.repeat_markers.append(g.text)
            state.counts["unexpanded_repeat"] += 1
    state.clefs[staff_index] = clef
    placed = _place_notes(state, measure, staff_index, staff, beats, _rhythm_atoms(note_records, rests, staff.spacing))
    _flag_unresolved_curves(state, page, measure, staff_index, staff, left, right)
    _flag_tuplets(state, measure, staff_index, staff, records)
    return placed


def _starting_clef(staff: _Staff, previous: str | None, left: float, right: float) -> tuple[str, bool]:
    """The clef in force at a bar's left edge, and whether the staff shows any clef before the bar ends."""
    clef_events = [(g.x, _clef(g, staff), g) for g in staff.glyphs if g.text in _CLEFS and g.x < right]
    clef = "treble" if previous is None else previous
    # Key on x alone: a redrawn clef at the same x would otherwise make the sort compare glyph objects.
    for x, selected, _ in sorted(clef_events, key=lambda event: event[0]):
        if x <= left:
            clef = selected
    return clef, bool(clef_events)


def _read_note(state: _ReadState, page: _Page, measure: ScoreMeasure, staff_index: int, staff: _Staff,
               g: _Glyph, clef: str, accidentals: dict[int, int]) -> NoteEvent | None:
    """Pitch, octave span, tie and duration of one notehead; None when it falls outside the MIDI range."""
    step = _staff_step(g.y, staff)
    if abs(step - round(step)) > .22:
        _warn(measure, staff_index, "A notehead does not align reliably to a staff line or space.")
        state.counts["unaligned_notehead"] += 1
    written_midi, written_spelling = _pitch(g.y, staff, clef, state.keys.get(staff_index, 0), accidentals.get(round(step)))
    shift = _octave_shift(g, staff, page.octave_spans)
    tied_source = page.notes_by_glyph.get(page.tie_links.get(id(g), -1))
    if tied_source is not None:
        # A tie carries its source's written pitch unless an accidental or an octave-span edge contradicts it.
        source_written_midi = tied_source.written_midi if tied_source.written_midi is not None else tied_source.midi
        source_written_spelling = tied_source.written_spelling or tied_source.spelling
        if round(step) in accidentals and written_midi != source_written_midi:
            _warn(measure, staff_index, "A possible tie conflicts with a written accidental; its pitch needs review.")
            state.counts["tie_accidental_conflict"] += 1
            tied_source = None
        elif source_written_midi + shift != tied_source.midi:
            _warn(measure, staff_index, "A possible tie crosses an octave-span boundary; sounding-pitch continuity needs review.")
            state.counts["tie_octave_conflict"] += 1
            tied_source = None
        else:
            written_midi, written_spelling = source_written_midi, source_written_spelling
    midi, spelling = written_midi + shift, _shift_spelling(written_spelling, shift)
    if not 0 <= midi <= 127:
        state.mark_incomplete("out_of_range_pitch")
        return None
    if id(staff) not in page.duration_geometry:
        page.duration_geometry[id(staff)] = _DurationGeometry(staff, page.verticals, page.shapes)
    duration = _duration(g, staff, page.duration_geometry[id(staff)])
    note = NoteEvent(staff_index, midi, spelling, 0, duration, g.x, g.y)
    if shift:
        note.written_midi, note.written_spelling, note.octave_shift = written_midi, written_spelling, shift
        state.resolved_counts["octave_shifted_notes"] += 1
    if tied_source is not None:
        note.tie_continuation = True
        tied_source.tie_to_next = True
    page.notes_by_glyph[id(g)] = note
    if duration is None:
        _warn(measure, staff_index, "A note duration could not be resolved from its stem, beams or flags.")
        state.counts["unknown_duration"] += 1
    return note


def _rhythm_atoms(note_records: list[tuple[_Glyph, NoteEvent]], rests: list[tuple[float, float]], spacing: float) \
        -> list[tuple[float, float | None, list[tuple[_Glyph, NoteEvent]]]]:
    """Note groups and rests left to right as (x, duration, notes).

    Noteheads within 1.4 spacings of a group's first head join that group, whose
    duration is its shortest known one.
    """
    groups: list[list[tuple[_Glyph, NoteEvent]]] = []
    for record in note_records:
        if groups and record[0].x - groups[-1][0][0].x <= spacing * 1.4:
            groups[-1].append(record)
        else:
            groups.append([record])
    atoms = [(group[0][0].x, min((note.duration for _, note in group if note.duration is not None), default=None), group)
             for group in groups]
    atoms.extend((x, duration, []) for x, duration in rests)
    atoms.sort(key=lambda item: item[0])
    return atoms


def _place_notes(state: _ReadState, measure: ScoreMeasure, staff_index: int, staff: _Staff, beats: float,
                 atoms: list[tuple[float, float | None, list[tuple[_Glyph, NoteEvent]]]]) -> list[tuple[float, float]]:
    """Give each note group its onset and confidence and add its notes to the measure.

    Durations that fill the meter are summed; otherwise onsets follow horizontal position
    and the staff is warned. Returns (x, onset) for each note group and rest.
    """
    if not atoms:
        return []
    total = sum(duration or 0 for _, duration, _ in atoms)
    sequential = all(duration is not None for _, duration, _ in atoms) and abs(total - beats) <= .06
    if not sequential:
        _warn(measure, staff_index, "Rhythm/voices do not sum to the written meter; onsets use provisional horizontal alignment.")
        state.counts["rhythm_alignment"] += 1
    placed = []
    cursor = 0.0
    first_x, last_x = atoms[0][0], atoms[-1][0]
    for x, duration, group in atoms:
        onset = cursor if sequential else max(0, min(beats, (x - first_x) / max(last_x - first_x + staff.spacing * 4, 1) * beats))
        for _, note in group:
            note.onset = round(onset, 6)
            note.confidence = "high" if sequential and measure.rhythm_exact else "medium"
            measure.notes.append(note)
        placed.append((x, onset))
        cursor += duration or 0
    return placed


def _flag_unresolved_curves(state: _ReadState, page: _Page, measure: ScoreMeasure, staff_index: int,
                            staff: _Staff, left: float, right: float) -> None:
    """Warn about tie or slur curves near this staff in the measure that were not resolved as ties."""
    for shape in page.shapes:
        if shape["x1"] < left or shape["x0"] > right:
            continue
        unresolved_curve = id(shape) not in page.resolved_tie_shapes and _is_curved(shape)
        if unresolved_curve and shape["x1"] - shape["x0"] > staff.spacing * 2 \
                and 0 < shape["y1"] - shape["y0"] < staff.spacing * 4 \
                and staff.bottom - staff.spacing * 7 < shape["y1"] < staff.top + staff.spacing * 7:
            _warn(measure, staff_index, "Tie/slur geometry is present; tied duration and voice continuity require review.")
            state.counts["tie_or_slur"] += 1


def _flag_tuplets(state: _ReadState, measure: ScoreMeasure, staff_index: int, staff: _Staff,
                  records: list[_Glyph]) -> None:
    for g in records:
        if g.text in _TUPLET_DIGITS and g.y > staff.top + staff.spacing * .5:
            _warn(measure, staff_index, "A possible tuplet needs review; tuplet timing has not been applied.")
            state.counts["possible_tuplet"] += 1


def _attach_chords(measure: ScoreMeasure, chords: list[tuple[float, str]], left: float, right: float,
                   spacing: float, onsets: list[tuple[float, float]]) -> None:
    """Attach the bar's printed chords at the onset of the horizontally nearest note group or rest."""
    for x, symbol in chords:
        if not _in_bar(x, left, right, spacing):
            continue
        if symbol == "N.C.":
            measure.no_chord = True
        else:
            onset = min(onsets, key=lambda item: abs(item[0] - x))[1] if onsets else 0.0
            measure.chords.append(PrintedChord(symbol, round(onset, 6), x))


def _flag_octave_text(state: _ReadState, page: _Page, measure: ScoreMeasure, system: list[_Staff],
                      left: float, right: float) -> None:
    """Warn when an octave sign in the bar has no resolved bracket; its pitches stay as written."""
    text = "".join(g.text for g in sorted(page.glyphs, key=lambda g: g.x)
                   if not _is_music(g) and id(g) not in page.resolved_octave_ids and left <= g.x < right
                   and system[-1].bottom - system[-1].spacing * 10 < g.y < system[0].top + system[0].spacing * 12)
    unresolved_here = [mark for mark in page.unresolved_octaves if left <= mark.x < right
                       and _nearest_staff(page.staffs, mark.y) in system]
    if unresolved_here:
        measure.warnings.append("An octave indication has no unambiguous bounded bracket; its pitches remain written until the span is reviewed.")
        measure.rhythm_exact = False
    elif _OCTAVE_TEXT.search(text):
        measure.warnings.append("Octave text is present but its bounded span is unresolved; pitches remain written.")
        measure.rhythm_exact = False
        state.counts["ottava"] += 1


def _summarize(state: _ReadState) -> ParsedScore:
    """Finish the score: support and completeness, unresolved counts and score-level warnings."""
    parsed, counts, resolved_counts = state.parsed, state.counts, state.resolved_counts
    parsed.supported = any(measure.notes or measure.chords or measure.no_chord for measure in parsed.measures)
    if not parsed.supported:
        parsed.complete = False
        parsed.warnings.append("No supported vector noteheads or printed chord labels could be extracted.")
    if counts:
        parsed.unsupported_counts = {name: count for name, count in counts.items() if count}
        parsed.unresolved = [f"{name}: {count}" for name, count in sorted(counts.items()) if count]
    parsed.resolved_counts = {name: count for name, count in resolved_counts.items() if count}
    if resolved_counts["ottava_spans"]:
        parsed.warnings.append(f"Applied {resolved_counts['ottava_spans']} bounded octave spans; original written pitches are retained alongside sounding pitches.")
    if any(not measure.rhythm_exact for measure in parsed.measures):
        parsed.warnings.append("Some durations or onsets are provisional; review the per-measure notation warnings.")
    if parsed.key_signature is not None:
        parsed.warnings.append("Key signature is extracted independently of tonal-center analysis; relative major/minor is not assumed.")
    return parsed


def _parse_pages(pages: Any, *, max_measures: int = MAX_PDF_MEASURES) -> ParsedScore:
    state = _ReadState(ParsedScore(), max_measures)
    total_glyphs = 0
    for page_number, page in enumerate(pages, 1):
        chars, lines = page.chars, page.lines
        total_glyphs += len(chars)
        if total_glyphs > MAX_TOTAL_GLYPHS or len(chars) + len(lines) > MAX_PAGE_OBJECTS:
            raise ScorePDFError(_LIMIT_MESSAGE)
        prepared = _prepare_page(state, page_number, page, chars, lines)
        if prepared is None:
            continue
        for system_number, system in enumerate(_systems(prepared.staffs, prepared.verticals), 1):
            _read_system(state, prepared, system_number, system)
        if len(state.parsed.measures) >= max_measures and page_number < len(pages):
            state.mark_incomplete("measure_limit")
            break
    return _summarize(state)


def parse_vector_score(data: bytes, *, max_pages: int = MAX_PDF_PAGES,
                       max_measures: int = MAX_PDF_MEASURES) -> ParsedScore:
    """Decode in request-local memory; uploaded bytes and notation are never cached."""
    if not isinstance(data, bytes) or not data or len(data) > MAX_PDF_BYTES or not data.startswith(b"%PDF-"):
        raise ScorePDFError("Choose a nonempty PDF up to 8 MiB for local vector analysis.")
    if not 1 <= max_pages <= MAX_PDF_PAGES or not 1 <= max_measures <= MAX_PDF_MEASURES:
        raise ScorePDFError(f"Local vector analysis supports 1..{MAX_PDF_PAGES} pages and 1..{MAX_PDF_MEASURES} measures.")
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            if not 1 <= len(pdf.pages) <= max_pages:
                raise ScorePDFError(f"The PDF exceeds the local limit of {max_pages} pages.")
            score = _parse_pages(pdf.pages, max_measures=max_measures)
    except ScorePDFError:
        raise
    except Exception as exc:
        raise ScorePDFError("The PDF could not be decoded into supported vector geometry.") from exc
    return score
