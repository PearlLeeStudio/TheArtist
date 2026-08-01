"""Deterministic type-1 MIDI export of the editor's actual two-slot voicings."""

import io
import math
import re
import threading
import unicodedata
from dataclasses import dataclass

import mido

from app.models.export import (
    MAX_EXPORT_BYTES, MAX_EXPORT_NOTES, NOTE_PC, ExportMeasure, ExportPlay, ExportSpec,
)
from app.services.genre_tables import (
    DRUM_PATTERNS, DRUM_VOICE_TO_GM_NOTE, GM_PROGRAM, bass_instrument, bass_pattern,
    genre_drum_pattern_id, harmony_instrument, harmony_pattern, harmony_rhythm_preset,
    harmony_strikes,
)

PPQ = 480
_MIDI_CHARSET_LOCK = threading.Lock()  # mido's text charset is process-global while saving
_MAJOR_KEYS = {"Cb", "Gb", "Db", "Ab", "Eb", "Bb", "F", "C", "G", "D", "A", "E", "B", "F#", "C#"}
_MINOR_KEYS = {"Ab", "Eb", "Bb", "F", "C", "G", "D", "A", "E", "B", "F#", "C#", "G#", "D#", "A#"}
_CANONICAL_MAJOR = ("C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
_CANONICAL_MINOR = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B")
_UNTITLED = "Untitled"
_MAX_FILENAME_TITLE_CHARS = 48

# Note velocities are 0..1 and lengths are in written beats.
_HARMONY_VELOCITY = 0.7
_BASS_VELOCITY = 0.6
_MAX_BASS_BEATS = 0.9
_DRUM_HIT_BEATS = 0.25
_DRUM_STEPS_PER_BEAT = 4  # drum patterns are written as 16 steps per 4/4 bar

_HARMONY_CHANNEL = 0
_BASS_CHANNEL = 1
_DRUM_CHANNEL = 9  # General MIDI percussion
_VOLUME_CONTROL = 7  # CC7 channel volume


class ExportError(ValueError):
    """An export that cannot be rendered, with its HTTP status (422 invalid data, 413 too large)."""

    def __init__(self, message: str, status_code: int = 422) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class Note:
    start: float  # quarter-note beats, after meter scaling and swing
    end: float
    pitch: int
    velocity: float


@dataclass(frozen=True)
class _Grid:
    """Places written-beat bar positions on the song's quarter-note time line."""

    numerator: int  # beats per bar
    quarters_per_beat: float  # pattern beats are written denominator units; SMF beats are quarters
    song_quarters: float
    swing: float  # how late off-beat eighths land, in written beats

    @property
    def half(self) -> float:
        return self.numerator / 2

    def onset(self, bar: int, position: float) -> float:
        """Swung onset of a bar position, in quarters."""
        return (bar * self.numerator + position + _swing_delay(position, self.swing)) * self.quarters_per_beat


def safe_title(title: str) -> str:
    """Filename-safe title: NFKC letters, digits, spaces, '-' and '_', or "Untitled"."""
    text = unicodedata.normalize("NFKC", title)
    text = "".join(char for char in text if char.isalnum() or char in " -_")
    return re.sub(r"\s+", " ", text).strip(" -_")[:_MAX_FILENAME_TITLE_CHARS] or _UNTITLED


def midi_key(key: str) -> str:
    """mido key name; a spelling with no SMF key signature (e.g. D# major) uses its enharmonic."""
    root, mode = key.split(" ")
    minor = mode == "minor"
    if root not in (_MINOR_KEYS if minor else _MAJOR_KEYS):
        root = (_CANONICAL_MINOR if minor else _CANONICAL_MAJOR)[NOTE_PC[root]]
    return root + ("m" if minor else "")


def _normalize_genre(genre: str | None) -> str:
    """The genre-table spelling of the song's genre (frontend twin: normalizeGenre in playSettings.ts)."""
    name = (genre or "").strip().lower()
    return "bossa" if name == "bossa nova" else name


def _chosen_harmony_pattern(play: ExportPlay, genre: str) -> list[float]:
    """The picked comping preset's pattern, else the genre's default rhythm."""
    preset = harmony_rhythm_preset(play.harmonyRhythmId) if play.harmonyRhythmId else None
    return preset["pattern"] if preset else harmony_pattern(genre)


def _meter_rhythm(pattern: list[float], numerator: int) -> list[float]:
    # Slot-only patterns [0,2] follow half-bars in every meter, as in Play.
    if pattern and all(position in (0, 2) for position in pattern):
        return [0 if position == 0 else numerator / 2 for position in pattern]
    return pattern


def _swing_delay(position: float, swing: float) -> float:
    """Swing delays only off-beat eighths (positions ending in .5)."""
    return swing if abs(position % 1 - 0.5) < 1e-9 else 0.0


def _harmony_notes(measures: list[ExportMeasure], pattern: list[float], grid: _Grid) -> list[Note]:
    """Each chord's voicing struck on the comping rhythm within its half-bar slot."""
    numerator, half = grid.numerator, grid.half
    quarters_per_beat, song_quarters = grid.quarters_per_beat, grid.song_quarters
    rhythm = _meter_rhythm(pattern, numerator)
    notes: list[Note] = []
    for bar, measure in enumerate(measures):
        first, second = measure.chords
        changed = (first.symbol if first else None) != (second.symbol if second else None)
        positions = [position for position in rhythm if 0 <= position < numerator]
        # A rhythm that never strikes in a chord's half-bar would drop that chord.
        if positions and first and not any(position < half for position in positions):
            positions.append(0)
        if positions and second and changed and not any(position >= half for position in positions):
            positions.append(half)
        for position, duration in harmony_strikes(positions, numerator):
            chord = first if position < half else second
            if chord is None:
                continue
            if changed and position < half:
                duration = min(duration, half - position)
            # Swing moves the onset; harmony still releases at the original intended time.
            start = grid.onset(bar, position)
            end = min(song_quarters, (bar * numerator + min(numerator, position + duration)) * quarters_per_beat)
            for pitch in sorted(set(chord.voicing)):
                notes.append(Note(start, end, pitch, _HARMONY_VELOCITY))
    return notes


def _bass_notes(measures: list[ExportMeasure], pattern: list[float], grid: _Grid) -> list[Note]:
    """The chord's bass on each hit, released by the next hit or the end of its chord slot."""
    numerator, half = grid.numerator, grid.half
    quarters_per_beat = grid.quarters_per_beat
    positions = sorted({position for position in _meter_rhythm(pattern, numerator) if 0 <= position < numerator})
    notes: list[Note] = []
    for bar, measure in enumerate(measures):
        first, second = measure.chords
        for index, position in enumerate(positions):
            chord = first if position < half else second
            if chord is None:
                continue
            start = grid.onset(bar, position)
            next_position = positions[index + 1] if index + 1 < len(positions) else numerator
            slot_end = half if position < half else numerator
            duration = min(_MAX_BASS_BEATS, next_position - position, slot_end - position)
            end = (bar * numerator + position + duration) * quarters_per_beat
            notes.append(Note(start, end, chord.bass_pitch, _BASS_VELOCITY))
    return notes


def _drum_notes(bar_count: int, drums: dict, grid: _Grid) -> list[Note]:
    """The one-bar drum pattern in every bar, each hit a short note."""
    notes: list[Note] = []
    for bar in range(bar_count):
        for voice, hits in drums["voices"].items():
            for step, velocity in hits:
                position = step / _DRUM_STEPS_PER_BEAT
                if not 0 <= position < grid.numerator:
                    continue
                start = grid.onset(bar, position)
                notes.append(Note(start, min(grid.song_quarters, start + _DRUM_HIT_BEATS * grid.quarters_per_beat),
                                  DRUM_VOICE_TO_GM_NOTE[voice], velocity))
    return notes


def _is_finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _is_valid_note(note: Note) -> bool:
    return (all(_is_finite_number(value) for value in (note.start, note.end, note.velocity))
            and 0 <= note.start < note.end and 0 < note.velocity <= 1
            and isinstance(note.pitch, int) and not isinstance(note.pitch, bool) and 0 <= note.pitch <= 127)


def _note_messages(notes: list[Note], channel: int, end_tick: int) -> list[mido.Message]:
    """Validate, deduplicate and cut retriggers of each pitch before delta encoding."""
    combined: dict[tuple[int, int], tuple[int, int]] = {}  # (on, pitch) -> (off, velocity)
    for note in notes:
        if not _is_valid_note(note):
            raise ExportError("Cannot export an invalid MIDI note or timing.")
        on = round(note.start * PPQ)
        off = min(end_tick, max(on + 1, round(note.end * PPQ)))
        if not 0 <= on < off <= end_tick:
            raise ExportError("A MIDI note falls outside the song.")
        velocity = max(1, min(127, round(note.velocity * 127)))
        previous_off, previous_velocity = combined.get((on, note.pitch), (off, velocity))
        combined[(on, note.pitch)] = (max(off, previous_off), max(velocity, previous_velocity))
    # Walk onsets latest first so each note can end at the next onset of its pitch.
    next_on: dict[int, int] = {}
    events: list[tuple[int, int, int, int]] = []
    for (on, pitch), (off, velocity) in sorted(combined.items(), reverse=True):
        off = min(off, next_on.get(pitch, off))
        next_on[pitch] = on
        events.extend(((on, 1, pitch, velocity), (off, 0, pitch, 0)))
    messages = []
    last_tick = 0
    for tick, is_on, pitch, velocity in sorted(events):  # at one tick, note-offs (0) precede note-ons (1)
        messages.append(mido.Message("note_on" if is_on else "note_off", channel=channel,
                                     note=pitch, velocity=velocity, time=tick - last_tick))
        last_tick = tick
    return messages


def _meta_track(spec: ExportSpec, quarters_per_beat: float, end_tick: int) -> mido.MidiTrack:
    """Title, tempo, meter and key, then one marker per section."""
    numerator, denominator = spec.timeSignature
    track = mido.MidiTrack([
        mido.MetaMessage("track_name", name=spec.title or _UNTITLED, time=0),
        mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(spec.bpm), time=0),
        mido.MetaMessage("time_signature", numerator=numerator, denominator=denominator, time=0),
        mido.MetaMessage("key_signature", key=midi_key(spec.key), time=0),
    ])
    last_tick = 0
    for section in sorted(spec.sections, key=lambda part: part.startMeasure):
        tick = round(section.startMeasure * numerator * quarters_per_beat * PPQ)
        track.append(mido.MetaMessage("marker", text=section.label, time=tick - last_tick))
        last_tick = tick
    track.append(mido.MetaMessage("end_of_track", time=end_tick - last_tick))
    return track


def _part_track(name: str, notes: list[Note], channel: int, program: int | None, level: float,
                end_tick: int) -> mido.MidiTrack:
    """One part: its name, GM program (none for drums), CC7 volume from its 0..1 level, then notes."""
    track = mido.MidiTrack([mido.MetaMessage("track_name", name=name, time=0)])
    if program is not None:
        track.append(mido.Message("program_change", channel=channel, program=program, time=0))
    track.append(mido.Message("control_change", channel=channel, control=_VOLUME_CONTROL,
                              value=round(level * 127), time=0))
    track.extend(_note_messages(notes, channel, end_tick))
    track.append(mido.MetaMessage("end_of_track", time=end_tick - sum(message.time for message in track)))
    return track


def _encode(midi: mido.MidiFile) -> bytes:
    buffer = io.BytesIO()
    with _MIDI_CHARSET_LOCK:
        midi.save(file=buffer)
    data = buffer.getvalue()
    if not data.startswith(b"MThd") or len(data) > MAX_EXPORT_BYTES:
        raise ExportError("The MIDI exceeds the 4 MiB export limit or could not be encoded.", 413)
    return data


@dataclass(frozen=True)
class Arrangement:
    """Every part's notes on the song's quarter-note time line: what the MIDI file and the
    Logic Scripter script both play."""

    grid: _Grid
    harmony: list[Note]
    bass: list[Note]
    drums: list[Note]
    has_drums: bool  # a drum pattern applies, so its track is written even without a hit


def arrangement(spec: ExportSpec) -> Arrangement:
    """The notes of every part, within the export's note limit."""
    numerator, denominator = spec.timeSignature
    quarters_per_beat = 4 / denominator
    song_quarters = len(spec.measures) * numerator * quarters_per_beat
    genre = _normalize_genre(spec.genre)
    play = spec.play
    drum_id = play.drumPatternId if play.drumMode == "manual" else genre_drum_pattern_id(genre)
    drums = DRUM_PATTERNS.get(drum_id)
    grid = _Grid(numerator, quarters_per_beat, song_quarters, drums["swing"] if drums else 0)

    harmony_notes = _harmony_notes(spec.measures, _chosen_harmony_pattern(play, genre), grid)
    bass_notes = _bass_notes(spec.measures, bass_pattern(genre), grid) if play.bassEnabled else []
    drum_notes = _drum_notes(len(spec.measures), drums, grid) if drums else []
    if sum(map(len, (harmony_notes, bass_notes, drum_notes))) > MAX_EXPORT_NOTES:
        raise ExportError("The arrangement exceeds the 100,000-note export limit.", 413)
    return Arrangement(grid, harmony_notes, bass_notes, drum_notes, bool(drums))


def render_midi(spec: ExportSpec) -> bytes:
    """A meta track, then Harmony, Bass (when enabled) and Drums (when a pattern applies)."""
    parts = arrangement(spec)
    end_tick = round(parts.grid.song_quarters * PPQ)
    genre = _normalize_genre(spec.genre)
    play = spec.play

    midi = mido.MidiFile(type=1, ticks_per_beat=PPQ, charset="utf8")
    midi.tracks.append(_meta_track(spec, parts.grid.quarters_per_beat, end_tick))
    instrument = play.harmonyOverride or harmony_instrument(genre)
    midi.tracks.append(_part_track(f"Harmony – {instrument.replace('_', ' ')}", parts.harmony, _HARMONY_CHANNEL,
                                   GM_PROGRAM[instrument], play.masterVolume * play.harmonyVolume, end_tick))
    if play.bassEnabled:
        midi.tracks.append(_part_track("Bass", parts.bass, _BASS_CHANNEL, GM_PROGRAM[bass_instrument(genre)],
                                       play.masterVolume * play.bassVolume, end_tick))
    if parts.has_drums:
        midi.tracks.append(_part_track("Drums", parts.drums, _DRUM_CHANNEL, None,
                                       play.masterVolume * play.drumVolume, end_tick))
    return _encode(midi)
