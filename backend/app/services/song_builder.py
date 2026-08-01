"""Compose multi-track events from a chord progression + genre.

Used by `/api/generate/song` — voyager sends genre+length, the
chord stage produces bars, this builder turns those bars into 3 layers of
symbolic events (harmony / bass / drum) plus a convenience MIDI render.

3-track only (melody dropped 2026-05-10). Each layer is rule-based:
  - harmony: chord voicing struck at a comping rhythm's positions (the
    genre's preset unless the caller passes one), lengths from
    harmony_strikes
  - bass: chord root struck at GENRE_BASS_RHYTHM positions (C2 base)
  - drum: a DRUM_PATTERNS voice/timing pattern (the genre's unless the
    caller passes one or turns the drums off)

Sparse rhythms add a boundary strike when a different second chord would
otherwise be omitted. Repeated single-chord bars retain the original rhythm.
"""

from __future__ import annotations

import base64
import io

import mido

from app.services.genre_tables import (
    DRUM_VOICE_TO_GM_NOTE,
    GM_PROGRAM,
    bass_instrument,
    bass_pattern,
    drum_pattern,
    harmony_instrument as genre_harmony_instrument,
    harmony_pattern,
    harmony_strikes,
)
from app.services.voice_leading import _CHORD_TONES, _parse


# Bass register: C2 (MIDI 36) is the standard electric-bass / contrabass
# anchor; chord roots placed there sit below the harmony register.
BASS_BASE_MIDI = 36

# Harmony register: build voicings around C4 (MIDI 60) so they sit above
# the bass and don't crowd the listening register's centre.
HARMONY_BASE_MIDI = 60


def _bass_pc(chord_sym: str) -> int | None:
    """Use the slash-bass note if specified, otherwise the root."""
    parsed = _parse(chord_sym)
    if parsed is None:
        return None
    root_pc, _q, bass_pc = parsed
    return bass_pc if bass_pc is not None else root_pc


def _harmony_voicing(chord_sym: str) -> list[int]:
    """Build a 3-4 note MIDI voicing for the chord, anchored near C4.
    Returns [] if the chord can't be parsed.
    """
    parsed = _parse(chord_sym)
    if parsed is None:
        return []
    root_pc, quality, _bass = parsed
    intervals = _CHORD_TONES.get(quality, _CHORD_TONES["maj"])
    return [HARMONY_BASE_MIDI + root_pc + iv for iv in intervals]


def _slot_chord(bar_chords: list[str | None], pos: float, beats_per_slot: float) -> str | None:
    """Pick the chord (slot 0 or 1) that's active at `pos` beats into the bar."""
    if not bar_chords:
        return None
    slot_idx = 0 if pos < beats_per_slot else 1
    if slot_idx >= len(bar_chords):
        slot_idx = len(bar_chords) - 1
    return bar_chords[slot_idx]


def _bar_strikes(
    pattern: list[float],
    chords: list[str | None],
    beats_per_measure: int,
) -> list[tuple[float, float]]:
    """(position, length) in beats of each harmony strike in one bar of the
    comping rhythm `pattern` (beat positions, as a preset holds them).

    Positions outside the bar drop out first (a 4/4 preset can reach past a
    3/4 bar). A rhythm that then skips a chord slot gains a strike at the
    slot's start: beat 0 for the first chord, the half bar for a different
    second chord. While the chord changes, a strike in the first slot stops
    at the half bar.
    """
    beats_per_slot = beats_per_measure / 2
    # The rhythm's positions inside one bar of this meter, sorted and deduplicated.
    positions = [pos for pos, _ in harmony_strikes(pattern, beats_per_measure)]
    first = _slot_chord(chords, 0, beats_per_slot)
    second = _slot_chord(chords, beats_per_slot, beats_per_slot)
    changed = first != second
    if positions and first and not any(pos < beats_per_slot for pos in positions):
        positions.append(0)
    if positions and second and changed and not any(pos >= beats_per_slot for pos in positions):
        positions.append(beats_per_slot)
    strikes: list[tuple[float, float]] = []
    for pos, dur in harmony_strikes(positions, beats_per_measure):
        if changed and pos < beats_per_slot:
            dur = min(dur, beats_per_slot - pos)
        strikes.append((pos, dur))
    return strikes


def build_harmony_events(
    bars: list[dict],
    genre: str,
    *,
    beats_per_measure: int,
    pattern: list[float] | None = None,
) -> list[dict]:
    """Schedule chord strikes at a 1-bar comping rhythm: `pattern` (beat
    positions, e.g. a HARMONY_RHYTHM_PRESETS entry's), else the genre's.
    Strike lengths come from harmony_strikes (sparse → sustained, dense →
    stabby, never into the next strike)."""
    base_pattern = pattern if pattern is not None else harmony_pattern(genre)
    beats_per_slot = beats_per_measure / 2

    events: list[dict] = []
    for bar_idx, bar in enumerate(bars):
        chords = bar.get("chords") or []
        for pos, dur in _bar_strikes(base_pattern, chords, beats_per_measure):
            chord_sym = _slot_chord(chords, pos, beats_per_slot)
            if not chord_sym:
                continue
            voicing = _harmony_voicing(chord_sym)
            if not voicing:
                continue
            for pitch in voicing:
                events.append({
                    "bar": bar_idx,
                    "beat": pos,
                    "pitch": pitch,
                    "duration": dur,
                    "velocity": 0.7,
                })
    return events


def build_bass_events(
    bars: list[dict],
    genre: str,
    *,
    beats_per_measure: int,
) -> list[dict]:
    """Schedule chord-root strikes at the genre's bass rhythm positions."""
    pattern = bass_pattern(genre)
    beats_per_slot = beats_per_measure / 2
    strike_dur = 0.9   # ~quarter; mostly hidden under the harmony layer

    events: list[dict] = []
    for bar_idx, bar in enumerate(bars):
        chords = bar.get("chords") or []
        for pos in pattern:
            if pos < 0 or pos >= beats_per_measure:
                continue
            chord_sym = _slot_chord(chords, pos, beats_per_slot)
            if not chord_sym:
                continue
            pc = _bass_pc(chord_sym)
            if pc is None:
                continue
            events.append({
                "bar": bar_idx,
                "beat": float(pos),
                "pitch": BASS_BASE_MIDI + pc,
                "duration": strike_dur,
                "velocity": 0.6,
            })
    return events


def build_drum_events(
    n_bars: int,
    genre: str,
    *,
    beats_per_measure: int,
    pattern: dict | None = None,
    enabled: bool = True,
) -> list[dict]:
    """Repeat a drum pattern across every bar: `pattern` (a DRUM_PATTERNS
    entry), else the genre's. `enabled=False` means no drums (the Play
    panel's Off) and returns [] whatever `pattern` is.
    Drum patterns are encoded on a 16th-note grid (pos16 in 0..15);
    convert to bar-relative beats for the response. A 16th is 0.25 beats
    in every meter, so steps past the bar drop out (3/4 plays steps
    0-11), as in the editor's drums.ts.
    """
    if not enabled:
        return []
    p = pattern if pattern is not None else drum_pattern(genre)
    if p is None:
        return []
    grid_step = 0.25   # one 16th, in beats

    events: list[dict] = []
    for bar_idx in range(n_bars):
        for voice, hits in p["voices"].items():
            for hit in hits:
                pos16, vel = hit
                beat = pos16 * grid_step
                if beat >= beats_per_measure:
                    continue
                events.append({
                    "bar": bar_idx,
                    "beat": float(beat),
                    "voice": voice,
                    "duration": grid_step,
                    "velocity": float(vel),
                })
    return events


def render_midi(
    bars: list[dict],
    genre: str,
    *,
    bpm: int,
    beats_per_measure: int,
    harmony_events: list[dict],
    bass_events: list[dict],
    drum_events: list[dict],
    harmony_instrument: str | None = None,
    bass_instrument_name: str | None = None,
) -> bytes:
    """Pack the three event lists into a multi-track MIDI binary.
    Each track gets a GM program-change so the receiver can render with
    matching timbres without needing the instrument lookup.
    `harmony_instrument` and `bass_instrument_name` replace the genre's instruments.
    A note ends no later than the next strike of its pitch on its channel.
    """
    mid = mido.MidiFile()
    ppq = mid.ticks_per_beat   # default 480

    def to_ticks(beat: float) -> int:
        return int(round(beat * ppq))

    def event_to_messages(track_events: list[dict], channel: int, *, drum: bool) -> list[mido.Message]:
        # Convert each event to (note_on_tick, note_off_tick, note, velocity)
        # then sort and emit with delta-times.
        notes: list[tuple[int, int, int, int]] = []
        for e in track_events:
            start_beat = e["bar"] * beats_per_measure + e["beat"]
            end_beat = start_beat + e["duration"]
            note = (DRUM_VOICE_TO_GM_NOTE.get(e["voice"], 36) if drum
                    else int(e["pitch"]))
            vel = max(1, min(127, int(round(e["velocity"] * 127))))
            notes.append((to_ticks(start_beat), to_ticks(end_beat), note, vel))

        # Cut a note at the next strike of the same pitch: one channel can't
        # sound a pitch twice, and players pair the note-offs differently
        # (funk's bass re-strikes its root 0.5 beats into a 0.9-beat note).
        # The event JSON keeps the editor's lengths.
        notes.sort()
        next_on: dict[int, int] = {}   # pitch -> tick of its next strike, walking backwards
        flat: list[tuple[int, str, int, int]] = []
        for on, off, note, vel in reversed(notes):
            next_strike = next_on.get(note)
            if next_strike is not None and on < next_strike < off:
                off = next_strike
            next_on[note] = on
            flat.append((on,  "on",  note, vel))
            flat.append((off, "off", note, 0))
        flat.sort(key=lambda x: (x[0], 0 if x[1] == "off" else 1))

        msgs: list[mido.Message] = []
        last_tick = 0
        for tick, kind, note, vel in flat:
            delta = tick - last_tick
            last_tick = tick
            if kind == "on":
                msgs.append(mido.Message("note_on", note=note, velocity=vel,
                                         channel=channel, time=delta))
            else:
                msgs.append(mido.Message("note_off", note=note, velocity=0,
                                         channel=channel, time=delta))
        return msgs

    # Tempo + time-sig meta
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    meta.append(mido.MetaMessage("time_signature",
                                 numerator=beats_per_measure, denominator=4,
                                 clocks_per_click=24, notated_32nd_notes_per_beat=8,
                                 time=0))
    mid.tracks.append(meta)

    # Channel 0 = harmony, 1 = bass, 9 = drums (GM convention). Programs
    # come from GM_PROGRAM; voyager can ignore them.
    h_instrument = harmony_instrument or genre_harmony_instrument(genre)
    h_track = mido.MidiTrack()
    h_track.append(mido.Message("program_change",
                                program=GM_PROGRAM.get(h_instrument, 0),
                                channel=0, time=0))
    h_track.extend(event_to_messages(harmony_events, channel=0, drum=False))
    mid.tracks.append(h_track)

    b_track = mido.MidiTrack()
    b_track.append(mido.Message("program_change",
                                program=GM_PROGRAM.get(bass_instrument_name or bass_instrument(genre), 33),
                                channel=1, time=0))
    b_track.extend(event_to_messages(bass_events, channel=1, drum=False))
    mid.tracks.append(b_track)

    d_track = mido.MidiTrack()
    d_track.extend(event_to_messages(drum_events, channel=9, drum=True))
    mid.tracks.append(d_track)

    buf = io.BytesIO()
    mid.save(file=buf)
    return buf.getvalue()


def midi_to_b64(midi_bytes: bytes) -> str:
    return base64.b64encode(midi_bytes).decode("ascii")
