import type { Chord } from '../models/types';
import { NOTE_VALUES, NOTE_NAMES } from './constants.ts';

// Intervals relative to root for each chord quality.
//
// Voicings are capped at 4 notes — that's how jazz pianists actually
// play these on the keyboard. For 5+ note tension chords (9/11/13 and
// their alterations), the conventional drop is the 5th (and the 9th
// for 11/13 chords), keeping root–3rd–7th–tension. This matches the
// "텐션 코드도 결국 4개 누르는" reality without throwing the tension
// away.
//
// Triads (3 notes) and basic 7-chords (4 notes) pass through unchanged.
export const QUALITY_INTERVALS: Record<string, number[]> = {
  // Triads — 3 notes
  'maj':    [0, 4, 7],
  'min':    [0, 3, 7],
  'dim':    [0, 3, 6],
  'aug':    [0, 4, 8],
  'sus2':   [0, 2, 7],
  'sus4':   [0, 5, 7],

  // 7-chords / 6-chords — 4 notes
  '7':      [0, 4, 7, 10],
  'maj7':   [0, 4, 7, 11],
  'm7':     [0, 3, 7, 10],
  'aug7':   [0, 4, 8, 10],
  'dim7':   [0, 3, 6, 9],
  'm7b5':   [0, 3, 6, 10],
  '6':      [0, 4, 7, 9],
  'm6':     [0, 3, 7, 9],
  'mMaj7':  [0, 3, 7, 11],
  'add9':   [0, 4, 7, 14],   // already 4-note (root, 3rd, 5th, 9th)

  // Extended (≥9th) — drop 5th, keep tension. 4 notes.
  '9':      [0, 4, 10, 14],   // root–3–b7–9
  'm9':     [0, 3, 10, 14],   // root–b3–b7–9
  'maj9':   [0, 4, 11, 14],   // root–3–maj7–9

  // Doubly-extended (11th, 13th) — drop 5th AND 9th; keep root–3–b7–top
  '11':     [0, 4, 10, 17],
  'm11':    [0, 3, 10, 17],
  '13':     [0, 4, 10, 21],
  'm13':    [0, 3, 10, 21],

  // Altered dominants — drop 5th, keep alteration
  '7b9':    [0, 4, 10, 13],
  '7#9':    [0, 4, 10, 15],
  '7#11':   [0, 4, 10, 18],
  '7b13':   [0, 4, 10, 20],

  // Suspended sevenths and ninths (charts write D7sus4, G9sus4) — keep the 4th
  '7sus4':  [0, 5, 7, 10],
  '7sus2':  [0, 2, 7, 10],
  '9sus4':  [0, 5, 10, 14],
  '9sus2':  [0, 2, 10, 14],
  '13sus4': [0, 5, 10, 21],

  // Sixth/ninth and added tones — 4 notes, drop the 5th where a tension replaces it
  '6/9':    [0, 4, 9, 14],
  'm6/9':   [0, 3, 9, 14],
  'add2':   [0, 2, 4, 7],
  'add4':   [0, 4, 5, 7],
  'add11':  [0, 4, 7, 17],
  'madd9':  [0, 3, 7, 14],
  'm(add9)': [0, 3, 7, 14],
  'm(add2)': [0, 2, 3, 7],
  'm(add4)': [0, 3, 5, 7],
  'm(add11)': [0, 3, 7, 17],

  // Major sevenths with upper extensions — root–3–maj7–top
  'maj7#11': [0, 4, 11, 18],
  'maj11':  [0, 4, 11, 17],
  'maj13':  [0, 4, 11, 21],
  'mMaj9':  [0, 3, 11, 14],
  '7b5':    [0, 4, 6, 10],

  // Power chords (manual entry's "5") — root, fifth, octave
  '5':      [0, 7, 12],
  '7(no3)': [0, 7, 10],
};

/** MIDI middle C: an inverted slash chord whose bass lands at or above it drops an octave. */
const MIDDLE_C = 60;

/** A note name's pitch class, or undefined for any name that is not a key of NOTE_VALUES.
 *  Indexing the plain object directly also finds inherited names ('constructor',
 *  '__proto__', ...), which a hand-edited project can carry; the function or object that
 *  comes back equals no pitch class, so the bass search below would never end. */
function notePc(name: string): number | undefined {
  return Object.hasOwn(NOTE_VALUES, name) ? NOTE_VALUES[name] : undefined;
}

/** Slash bass as a pitch class, or null when the chord has none (or it is the root). */
function slashBassPc(chord: Chord): number | null {
  if (!chord.bass) return null;
  const bass = notePc(chord.bass);
  const root = notePc(chord.root);
  if (bass === undefined || root === undefined || bass === root) return null;
  return bass;
}

/** Root-position stack from the root in octave 3 (the voicing before slash basses were honoured). */
export function rootPositionVoicing(chord: Chord): number[] {
  const rootValue = notePc(chord.root);
  if (rootValue === undefined) return [];

  const intervals = Object.hasOwn(QUALITY_INTERVALS, chord.quality) ? QUALITY_INTERVALS[chord.quality] : QUALITY_INTERVALS['maj'];
  const baseMidi = 48 + rootValue;
  return intervals.map(interval => baseMidi + interval);
}

// Slash chords put the named bass at the bottom. When the bass is a chord
// tone the chord is inverted (F/C = C F A, C/E = E G C); otherwise the bass
// goes under the root-position chord (F/G = G | F A C), dropping the 5th of
// a 4-note chord so the voicing stays within the 4-note cap plus the bass.
export function generatePianoVoicing(chord: Chord): number[] {
  const voicing = rootPositionVoicing(chord);
  const bass = slashBassPc(chord);
  if (voicing.length === 0 || bass === null) return voicing;

  if (voicing.some(m => m % 12 === bass)) {
    const inverted = [...voicing];
    // tensions reach up to a 13th, so a note may need two octave moves
    for (let i = 0; i < voicing.length * 2 && inverted[0] % 12 !== bass; i++) {
      inverted.push(inverted.shift()! + 12);
      inverted.sort((a, b) => a - b);
    }
    return inverted[0] >= MIDDLE_C ? inverted.map(m => m - 12) : inverted;
  }

  const fifth = voicing[0] + 7;
  const upper = voicing.length >= 4 && voicing.includes(fifth) ? voicing.filter(m => m !== fifth) : voicing;
  let bassMidi = upper[0] - 1;
  while (bassMidi % 12 !== bass) bassMidi--;
  return [bassMidi, ...upper];
}

/** The chord's stored voicing, except an empty one or a slash chord still stored in root
 *  position (saved before slash basses were honoured); those are recomputed. A voicing
 *  played in from a MIDI keyboard is never root-position-only for a slash chord, so it stays. */
export function resolvePianoVoicing(chord: Chord): number[] {
  const stored = chord.voicing ?? [];
  if (stored.length === 0) return generatePianoVoicing(chord);
  const savedBeforeSlashBasses = slashBassPc(chord) !== null && sameNotes(rootPositionVoicing(chord), stored);
  return savedBeforeSlashBasses ? generatePianoVoicing(chord) : stored;
}

function sameNotes(expected: number[], actual: number[]): boolean {
  return expected.length === actual.length && expected.every((m, i) => m === actual[i]);
}

const OPEN_STRINGS = [40, 45, 50, 55, 59, 64]; // E A D G B E
/** Guitar shapes stay in the open position: frets 0 (open) to 5. */
const HIGHEST_FRET = 5;

/** The lowest fret on a string tuned to `openMidi` whose pitch class
 *  `wanted` accepts, or -1 (the string is muted). */
function lowestFret(openMidi: number, wanted: (pitchClass: number) => boolean): number {
  for (let fret = 0; fret <= HIGHEST_FRET; fret++) {
    if (wanted((openMidi + fret) % 12)) return fret;
  }
  return -1;
}

export function generateGuitarVoicing(chord: Chord): number[] {
  const rootValue = notePc(chord.root);
  if (rootValue === undefined) return [-1, -1, -1, -1, -1, -1];

  const intervals = Object.hasOwn(QUALITY_INTERVALS, chord.quality) ? QUALITY_INTERVALS[chord.quality] : QUALITY_INTERVALS['maj'];
  const chordNotes = new Set(intervals.map(i => (rootValue + i) % 12));
  const isChordTone = (pitchClass: number) => chordNotes.has(pitchClass);

  // Slash chord: the named bass goes on the lowest string that reaches it
  // within the first five frets (F/C = x33211, C/E = 032010, G/B = x20003);
  // strings below it are muted and the rest take the lowest-fret chord tone.
  const bass = slashBassPc(chord);
  if (bass !== null) {
    const isBass = (pitchClass: number) => pitchClass === bass;
    const bassString = OPEN_STRINGS.findIndex(open => lowestFret(open, isBass) >= 0);
    if (bassString >= 0) {
      return OPEN_STRINGS.map((open, i) => {
        if (i < bassString) return -1;
        return lowestFret(open, i === bassString ? isBass : isChordTone);
      });
    }
  }

  const frets = OPEN_STRINGS.map(open => lowestFret(open, isChordTone));

  // Ensure root is in the bass — mute strings below the lowest root occurrence
  let rootStringIndex = -1;
  for (let i = 0; i < frets.length; i++) {
    if (frets[i] >= 0) {
      const note = (OPEN_STRINGS[i] + frets[i]) % 12;
      if (note === rootValue) {
        rootStringIndex = i;
        break;
      }
    }
  }

  if (rootStringIndex > 0) {
    // Mute strings below the root
    for (let i = 0; i < rootStringIndex; i++) {
      frets[i] = -1;
    }
  }
  // If no root found, keep all voiced strings (don't mute everything)

  return frets;
}

/** The MIDI notes of the guitar voicing, low string first; muted strings are left out. */
export function guitarVoicingNotes(chord: Chord): number[] {
  return generateGuitarVoicing(chord).flatMap((fret, i) => (fret >= 0 ? [OPEN_STRINGS[i] + fret] : []));
}

export function midiToNoteName(midi: number): string {
  const octave = Math.floor(midi / 12) - 1;
  return NOTE_NAMES[midi % 12] + octave;
}
