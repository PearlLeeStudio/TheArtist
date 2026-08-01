"""Logic Pro Scripter export: the arrangement as a MIDI FX script that plays inside Logic.

The script carries the same harmony and bass notes the MIDI export writes (smf_export.arrangement),
so Logic plays what Play and the .mid file play; drums stay with Logic or the MIDI file.
"""

import json
import math
from pathlib import Path

from app.models.export import ExportSpec
from app.services.smf_export import ExportError, Note, arrangement, safe_title

TEMPLATE = Path(__file__).with_name("scripter_template.js")
SONG_MARKER = "/*__SONG_JSON__*/"
MAX_SCRIPT_BYTES = 4 * 1024 * 1024


def _notes(part: str, notes: list[Note]) -> list[dict]:
    # The MIDI encoder drops the same impossible notes (smf_export._is_valid_note)
    return [{"part": part, "pitch": note.pitch, "start": round(note.start, 6), "end": round(note.end, 6),
             "velocity": note.velocity}
            for note in notes
            if 0 <= note.pitch <= 127 and math.isfinite(note.start) and math.isfinite(note.end) and note.end > note.start]


def scripter_song(spec: ExportSpec) -> dict:
    """The SONG object the template reads: times in quarter notes from the start of bar 1."""
    parts = arrangement(spec)
    grid = parts.grid
    chords = [{"bar": bar, "slot": slot, "symbol": chord.symbol,
               "startQuarter": round((bar * grid.numerator + slot * grid.half) * grid.quarters_per_beat, 6)}
              for bar, measure in enumerate(spec.measures)
              for slot, chord in enumerate(measure.chords) if chord is not None]
    notes = sorted(_notes("bass", parts.bass) + _notes("harmony", parts.harmony),
                   key=lambda note: (note["start"], note["part"] != "bass", note["pitch"]))
    # The song form the script plays from: the whole form or one of these sections, never beyond it
    sections = [{"label": section.label, "startBar": section.startMeasure, "bars": section.length}
                for section in sorted(spec.sections, key=lambda section: section.startMeasure)]
    return {"title": spec.title, "key": spec.key, "bpm": spec.bpm, "timeSignature": list(spec.timeSignature),
            "quartersPerBar": grid.numerator * grid.quarters_per_beat, "lengthQuarters": grid.song_quarters,
            "sections": sections, "chords": chords, "notes": notes}


def render_scripter(spec: ExportSpec) -> str:
    """The template with this song's SONG object in place of the placeholder."""
    template = TEMPLATE.read_text(encoding="utf-8")
    start = template.index(SONG_MARKER) + len(SONG_MARKER)
    end = template.index(";\n", start)
    payload = json.dumps(scripter_song(spec), ensure_ascii=False, separators=(",", ":"))
    # Valid JSON, but not valid in an older JavaScript string literal
    payload = payload.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    script = template[:start] + payload + template[end:]
    if len(script.encode("utf-8")) > MAX_SCRIPT_BYTES:
        raise ExportError("The Scripter script exceeds the 4 MiB export limit.", 413)
    return script


def scripter_filename(title: str) -> str:
    return f"{safe_title(title)}.theartist.js"
