"""Compose multi-track events from a chord progression + genre.

Used by `/api/generate/song` — voyager sends genre+length, the
chord stage produces bars, this builder turns those bars into 3 layers of
symbolic events (harmony / bass / drum) plus a convenience MIDI render.

3-track only (melody dropped 2026-05-10). Each layer is rule-based:
  - harmony: chord voicing struck at GENRE_HARMONY_RHYTHM positions
  - bass: chord root struck at GENRE_BASS_RHYTHM positions (C2 base)
  - drum: per-genre voice/timing pattern from GENRE_DRUM_PATTERN

Rhythm patterns are 1-bar literal — if a slot has no pattern position
landing in it, that slot's chord simply isn't struck (the previous strike
sustains).
"""

from __future__ import annotations

import base64
import io

import mido

from app.services.genre_tables import (
    DRUM_VOICE_TO_GM_NOTE,
    bass_instrument,
    bass_pattern,
    drum_pattern,
    harmony_instrument,
    harmony_pattern,
)
from app.services.voice_leading import _CHORD_TONES, _parse


# Bass register: C2 (MIDI 36) is the standard electric-bass / contrabass
# anchor; chord roots placed there sit below the harmony register.
BASS_BASE_MIDI = 36

# Harmony register: build voicings around C4 (MIDI 60) so they sit above
# the bass and don't crowd the listening register's centre.
HARMONY_BASE_MIDI = 60


def _root_pc(chord_sym: str) -> int | None:
    """Extract root pitch class from a chord symbol (e.g. 'Cmaj7' → 0)."""
    parsed = _parse(chord_sym)
    if parsed is None:
        return None
    root_pc, _q, _bass = parsed
    return root_pc


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


def build_harmony_events(
    bars: list[dict],
    genre: str,
    *,
    beats_per_measure: int,
) -> list[dict]:
    """Schedule chord stabs at the genre's harmony rhythm positions."""
    pattern = harmony_pattern(genre)
    beats_per_slot = beats_per_measure / 2
    # Stab duration auto-scales to pattern density (sparse → sustained,
    # dense → stabby). 0.85 keeps a hair of breath between strikes.
    n = max(1, len(pattern))
    strike_dur = (beats_per_measure / n) * 0.85

    events: list[dict] = []
    for bar_idx, bar in enumerate(bars):
        chords = bar.get("chords") or []
        for pos in pattern:
            if pos < 0 or pos >= beats_per_measure:
                continue
            chord_sym = _slot_chord(chords, pos, beats_per_slot)
            if not chord_sym:
                continue
            voicing = _harmony_voicing(chord_sym)
            if not voicing:
                continue
            for pitch in voicing:
                events.append({
                    "bar": bar_idx,
                    "beat": float(pos),
                    "pitch": pitch,
                    "duration": strike_dur,
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
) -> list[dict]:
    """Repeat the genre's drum pattern across every bar.
    Drum patterns are encoded on a 16th-note grid (pos16 in 0..15);
    convert to bar-relative beats for the response.
    """
    p = drum_pattern(genre)
    if p is None:
        return []
    grid_step = beats_per_measure / 16.0   # 4/4 → 0.25 beats per 16th

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
) -> bytes:
    """Pack the three event lists into a multi-track MIDI binary.
    Each track gets a GM program-change so the receiver can render with
    matching timbres without needing the instrument lookup.
    """
    mid = mido.MidiFile()
    ppq = mid.ticks_per_beat   # default 480

    def to_ticks(beat: float) -> int:
        return int(round(beat * ppq))

    def event_to_messages(track_events: list[dict], channel: int, *, drum: bool) -> list[mido.Message]:
        # Convert each event to (note_on_tick, note_off_tick, note, velocity)
        # then sort and emit with delta-times.
        flat: list[tuple[int, str, int, int]] = []
        for e in track_events:
            start_beat = e["bar"] * beats_per_measure + e["beat"]
            end_beat = start_beat + e["duration"]
            note = (DRUM_VOICE_TO_GM_NOTE.get(e["voice"], 36) if drum
                    else int(e["pitch"]))
            vel = max(1, min(127, int(round(e["velocity"] * 127))))
            flat.append((to_ticks(start_beat), "on",  note, vel))
            flat.append((to_ticks(end_beat),   "off", note, 0))
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

    # Per-track GM program (best-effort approximations — voyager can ignore).
    GM_HARMONY = {
        "acoustic_grand_piano": 0,
        "electric_piano_1":     4,
        "hammond_organ":        16,
        "acoustic_guitar_nylon": 24,
        "acoustic_guitar_steel": 25,
        "electric_guitar_clean": 27,
        "overdriven_guitar":     29,
        "pad_2_warm":            89,
    }
    GM_BASS = {
        "contrabass":             32,
        "electric_bass_finger":   33,
    }

    # Channel 0 = harmony, 1 = bass, 9 = drums (GM convention).
    h_track = mido.MidiTrack()
    h_track.append(mido.Message("program_change",
                                program=GM_HARMONY.get(harmony_instrument(genre), 0),
                                channel=0, time=0))
    h_track.extend(event_to_messages(harmony_events, channel=0, drum=False))
    mid.tracks.append(h_track)

    b_track = mido.MidiTrack()
    b_track.append(mido.Message("program_change",
                                program=GM_BASS.get(bass_instrument(genre), 33),
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
