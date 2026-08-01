/**
 * Arrangement layer — bass scheduling under the harmony + drum layers, in
 * the editor (Play panel bass on/off) and in MyPage session playback.
 *
 * Design:
 *   - Bass = chord root (or slash bass), struck on the genre's 1-bar
 *     bass rhythm (GENRE_BASS_RHYTHM in genreTables.ts).
 *
 * **Melody layer dropped 2026-05-10**: 4-track → 3-track (harmony + bass +
 * drum). WJazzD ~283 songs is too small to train a real chord-conditioned
 * melody from scratch; Chordonomicon-derived data is jazz-only, and
 * licensing rules out commercial use. The harmony layer's instrument picker
 * and comping-rhythm presets absorb the "variety" role melody used to play.
 *
 * **Storage shape:** an `Arrangement` object with 1-bar `bassPattern` +
 * `bassInstrument` + `beatsPerMeasure`. Repeated across all measures at
 * play time. Beat positions are floats in `[0, beatsPerMeasure)`.
 */
import * as Tone from 'tone';
import type { Arrangement, Chord, Song } from '../models/types';
import { NOTE_VALUES } from './constants';
import { playChordOnInstrument } from './instruments';
import { swingDelay } from './drums';
import { bassInstrumentFor, bassPatternFor, meterRhythm } from './playSettings';

export type { Arrangement } from '../models/types';

/** Longest bass note, in beats; the next bass note or the end of the chord
 *  slot can cut it shorter. */
const MAX_BASS_NOTE_BEATS = 0.9;
/** Shortest bass note, in seconds, once its swing delay is taken off. */
const MIN_BASS_NOTE_SECONDS = 0.01;
const BASS_VELOCITY = 0.6;

/** Read when each bass note fires, so the Play panel's Bass key mutes or
 *  unmutes the bass mid-playback without a reschedule. */
let bassEnabled = true;

export function setBassEnabled(on: boolean): void {
  bassEnabled = on;
}

/**
 * Tag each rhythm-pattern position with the chord slot it falls into
 * (slot 0 covers beats `[0, beatsPerSlot)`, slot 1 covers
 * `[beatsPerSlot, beatsPerMeasure)`). Patterns are 1-bar by design —
 * if a slot is empty in the pattern, that slot's chord simply isn't
 * struck. An earlier version inherited slot 0 → slot 1 to "ensure both
 * chords are heard," but that produced unnatural 4-strike Charleston etc.
 * The natural reading is "rhythm is 1-bar, harmony is incidental — write
 * 1 chord per bar to match sparse patterns."
 */
export function expandPatternPerSlot(
  pattern: number[],
  beatsPerSlot: number,
  beatsPerMeasure: number,
): { pos: number; slotIdx: 0 | 1 }[] {
  const out: { pos: number; slotIdx: 0 | 1 }[] = [];
  for (const p of pattern) {
    if (p < 0 || p >= beatsPerMeasure) continue;
    out.push({ pos: p, slotIdx: p < beatsPerSlot ? 0 : 1 });
  }
  return out;
}

function bassNoteFor(chord: Chord): number | null {
  const bassRoot = chord.bass ?? chord.root;
  const v = NOTE_VALUES[bassRoot];
  if (v === undefined) return null;
  // C2 base (MIDI 36) — chord root in standard electric-bass register.
  return 36 + v;
}

/** The pattern's first position after `pos` inside the bar, else the bar's end. */
function nextPatternPos(pattern: number[], pos: number, beatsPerMeasure: number): number {
  return pattern.filter((p) => p > pos && p < beatsPerMeasure).sort((a, b) => a - b)[0] ?? beatsPerMeasure;
}

/**
 * The 1-bar bass pattern + bass instrument for a song's genre. MyPage
 * caches it on the session at save time; the editor computes it at play.
 *
 * Drum stays out of this — drums.ts has its own per-genre patterns.
 * Harmony also stays in playback.ts (comping rhythm + instrument).
 */
export function computeArrangement(song: Song): Arrangement {
  return {
    bassPattern: bassPatternFor(song.genre),
    bassInstrument: bassInstrumentFor(song.genre),
    beatsPerMeasure: song.timeSignature?.[0] ?? 4,
  };
}

/**
 * Schedule the bass notes of one chord slot on Tone.Transport, from the
 * slot's start time — playback.ts calls it per scheduled slot, so the bass
 * lines up with the harmony for a selection or a mid-song start too. Pushes
 * scheduled IDs into `scheduledIds` so playback cleanup can clear them.
 *
 * Pattern positions ≥ the song's beatsPerMeasure are dropped (3/4, 6/8:
 * patterns written for 4/4 still play, minus the over-bar positions); a
 * pattern on the chord-slot starts only (boom-chick [0, 2]) follows the
 * chord slots instead (meterRhythm).
 */
export function scheduleBassSlot(
  arrangement: Arrangement,
  chord: Chord,
  slotIndex: 0 | 1,
  slotTime: number,
  beatsPerMeasure: number,
  secondsPerBeat: number,
  scheduledIds: number[],
  beatUnit = 1,
): void {
  const bassMidi = bassNoteFor(chord);
  if (bassMidi === null) return;
  const beatsPerSlot = beatsPerMeasure / 2;
  const slotStartBeat = slotIndex * beatsPerSlot;
  const slotEndBeat = slotStartBeat + beatsPerSlot;
  const pattern = meterRhythm(arrangement.bassPattern, beatsPerMeasure);
  for (const { pos, slotIdx } of expandPatternPerSlot(pattern, beatsPerSlot, beatsPerMeasure)) {
    if (slotIdx !== slotIndex) continue;
    const id = Tone.getTransport().schedule((t) => {
      if (!bassEnabled) return;
      // The length follows the tempo playing now (the BPM knob changes it
      // live); a swung "&" starts late and still ends on time.
      const beats = Math.min(MAX_BASS_NOTE_BEATS, nextPatternPos(pattern, pos, beatsPerMeasure) - pos, slotEndBeat - pos);
      const secondsPerBeatNow = (60 * beatUnit) / Tone.getTransport().bpm.value;
      const delay = swingDelay(pos, beatUnit);
      const dur = Math.max(MIN_BASS_NOTE_SECONDS, beats * secondsPerBeatNow - delay);
      playChordOnInstrument(arrangement.bassInstrument, [bassMidi], t + delay, dur, BASS_VELOCITY, 'bass');
    }, slotTime + (pos - slotStartBeat) * secondsPerBeat);
    scheduledIds.push(id);
  }
}
