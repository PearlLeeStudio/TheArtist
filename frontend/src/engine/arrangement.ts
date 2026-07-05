/**
 * Arrangement layer — bass scheduling for "play with full multi-track"
 * mode (Pearl 2026-05-10 R4 MVP, MyPage Play button).
 *
 * Design:
 *   - Bass = chord root, struck per genre rhythm pattern.
 *
 * **Melody layer dropped 2026-05-10**: Pearl decided 4-track → 3-track
 * (harmony + bass + drum). Reasons: WJazzD ~283 songs is too small to
 * train a real chord-conditioned melody from scratch; Chordonomicon-
 * derived data is jazz-only, and licensing rules out commercial use.
 * Harmony's instrument-override grid (VoicingViewer) absorbs the
 * "variety" role melody used to provide.
 *
 * **Storage shape:** an `Arrangement` object with 1-bar `bassPattern` +
 * `bassInstrument` + `beatsPerMeasure`. Repeated across all measures at
 * play time. Beat positions are floats in `[0, beatsPerMeasure)`.
 */
import * as Tone from 'tone';
import type { Arrangement, Chord, Song } from '../models/types';
import { NOTE_VALUES } from './constants';
import {
  bassInstrumentForGenre,
  playChordOnInstrument,
} from './instruments';

export type { Arrangement } from '../models/types';

/**
 * Beat positions where the harmony chord strikes within a 1-bar window.
 * Pearl 2026-05-10: harmony absorbs the rhythmic-variety role melody used
 * to play, since the melody track was dropped. Each pattern was picked to
 * be the most-recognized comping rhythm for the genre — Charleston for
 * jazz, boom-chick for country/gospel, 8th pulse for pop/rock, etc.
 *
 * Strike duration auto-scales to the pattern density (sparse patterns
 * sustain longer, dense ones feel stabby). Read by playback.ts at chord
 * trigger time so harmonyOverride / live BPM swap stays responsive.
 */
export const GENRE_HARMONY_RHYTHM: Record<string, number[]> = {
  // Beatles / Coldplay strum — 8th-note pulse over the bar.
  // Data-validated 2026-05-10 against POP909 (909 songs, 73,290 bars):
  // top 8 hit positions matched this exact pattern at 0.67–0.93 ratio;
  // 16th positions (0.75/1.75/2.75/3.25) clustered at ~0.21 (below the
  // 0.30 threshold).
  pop:        [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5],
  // Country boom-chick — chord on beats 2 & 4 (bass on 1 & 3).
  country:    [1, 3],
  // AC/DC / Foo Fighters driving 8ths.
  rock:       [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5],
  // 12-bar shuffle blues — triplet stab on every beat + "& of triplet"
  // (Stevie Ray Vaughan / Albert King). Pearl 2026-05-10: chose shuffle
  // over the slow-blues whole-bar sustain so the genre identity is
  // immediately audible against the chord-stab harmony layer.
  blues:      [0, 2/3, 1, 5/3, 2, 8/3, 3, 11/3],
  // Bossa nova syncopation — 1-bar approximation of 3-2 clave first half.
  bossa:      [0, 1.5, 3],
  // Chamber-music / hymn-style — chord on every beat (4-part chorale).
  classical:  [0, 1, 2, 3],
  // Joni Mitchell / Bill Withers pocket — half + 4 eighths.
  folk:       [0, 2, 2.5, 3, 3.5],
  // Neo-soul (D'Angelo / J Dilla) pocket — offbeat-only behind-beat feel.
  rnb_soul:   [1.5, 2.5, 3.5],
  // Trap / boom-bap drop — chord on beat 1 + "& of 3".
  hip_hop:    [0, 2.5],
  // House / EDM sidechain pump — chord on offbeats only (drop the 1).
  electronic: [0.5, 1.5, 2.5, 3.5],
  // JB "Funky Drummer" anticipation — two 8th clusters around beats 1, 3.
  funk:       [0, 0.5, 2, 2.5],
  // Gospel B3 organ backbeat — chord stabs on beats 2 & 4.
  gospel:     [1, 3],
  // Charleston (Count Basie / Red Garland) — beat 1 + "& of 2".
  jazz:       [0, 1.5],
};

const GENRE_BASS_RHYTHM: Record<string, number[]> = {
  pop:        [0, 1, 2, 3],
  country:    [0, 2],          // boom-chick: bass on 1 & 3
  rock:       [0, 1, 2, 3],
  blues:      [0, 1, 2, 3],
  bossa:      [0, 1.5],        // 1-bar bossa bass (was 2-bar [0, 1.5, 4, 5.5])
  classical:  [0],
  folk:       [0, 1, 2, 3],
  rnb_soul:   [0, 1, 2, 3],
  hip_hop:    [0, 3],
  electronic: [0, 1, 2, 3],    // 4-on-floor sub
  funk:       [0, 0.5, 2, 2.5],
  gospel:     [0, 2],          // boom-chick: bass on 1 & 3
  jazz:       [0, 1, 2, 3],    // walking deferred to v2; root for now
};

/**
 * Tag each rhythm-pattern position with the chord slot it falls into
 * (slot 0 covers beats `[0, beatsPerSlot)`, slot 1 covers
 * `[beatsPerSlot, beatsPerMeasure)`). Patterns are 1-bar by design —
 * if a slot is empty in the pattern, that slot's chord simply isn't
 * struck. Pearl 2026-05-10: an earlier version inherited slot 0 →
 * slot 1 to "ensure both chords are heard," but that produced
 * unnatural 4-strike Charleston etc. The natural reading is "rhythm
 * is 1-bar, harmony is incidental — write 1 chord per bar to match
 * sparse patterns."
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

/**
 * Compute the 1-bar arrangement pattern + bass instrument selection for
 * a song. Called at SAVE time so the result can be cached on the session
 * — Play just reads this and expands per measure.
 *
 * Drum stays out of this — drums.ts has its own per-genre patterns.
 * Harmony also stays in playback.ts (chord block on every slot, with
 * user-pickable instrument override on top).
 */
export function computeArrangement(song: Song): Arrangement {
  const genre = (song.genre ?? '').toLowerCase();
  const beatsPerMeasure = song.timeSignature?.[0] ?? 4;

  return {
    bassPattern: GENRE_BASS_RHYTHM[genre] ?? [0, 1, 2, 3],
    bassInstrument: bassInstrumentForGenre(genre) ?? 'electric_bass_finger',
    beatsPerMeasure,
  };
}

/**
 * Schedule the pre-computed arrangement on Tone.Transport — expands
 * the 1-bar pattern across every measure of the song. Called at play
 * time. Pushes scheduled IDs into `scheduledIds` so playback cleanup
 * can clear them on stop.
 *
 * Pattern positions ≥ song's actual beatsPerMeasure are dropped at
 * expand (handles 3/4, 6/8 gracefully — patterns optimized for 4/4
 * still play, just with the over-bar positions filtered out).
 */
export function scheduleArrangement(
  arrangement: Arrangement,
  song: Song,
  baseTime: number,
  secondsPerBeat: number,
  scheduledIds: number[],
): void {
  const beatsPerMeasure = song.timeSignature?.[0] ?? arrangement.beatsPerMeasure;
  const beatsPerSlot = beatsPerMeasure / 2;

  for (let mi = 0; mi < song.measures.length; mi++) {
    const measure = song.measures[mi];
    if (!measure) continue;
    const measureStartTime = baseTime + mi * beatsPerMeasure * secondsPerBeat;

    // Bass — chord root strikes. Pattern expanded so a sparse 1-bar
    // pattern (e.g. bossa `[0, 1.5]` slot-0-only) still strikes the
    // second slot's chord by inheriting slot 0's sub-pattern.
    const expandedBass = expandPatternPerSlot(arrangement.bassPattern, beatsPerSlot, beatsPerMeasure);
    for (const { pos, slotIdx } of expandedBass) {
      const chord = measure.chords[slotIdx];
      if (!chord) continue;
      const bassMidi = bassNoteFor(chord);
      if (bassMidi === null) continue;
      const time = measureStartTime + pos * secondsPerBeat;
      const dur = 0.9 * secondsPerBeat;
      const id = Tone.getTransport().schedule((t) => {
        playChordOnInstrument(arrangement.bassInstrument, [bassMidi], t, dur, 0.6);
      }, time);
      scheduledIds.push(id);
    }
  }
}
