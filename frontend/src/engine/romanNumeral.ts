import type { Chord } from '../models/types';
import { NOTE_VALUES, MAJOR_SCALE_INTERVALS, MINOR_SCALE_INTERVALS } from './constants.ts';

const ROMAN_NUMERALS = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII'];
const NOTE_LETTERS = ['C', 'D', 'E', 'F', 'G', 'A', 'B'];

const MINOR_QUALITIES = new Set([
  'min', 'm7', 'dim', 'dim7', 'm7b5', 'm6', 'm9', 'm11', 'm13', 'mMaj7', 'mMaj9', 'm6/9',
  'madd9', 'm(add9)', 'm(add2)', 'm(add4)', 'm(add11)',
]);

/** Conventional spelling of a non-diatonic root when its written letter would
 *  need a double accidental: by semitone above the tonic, for major and minor keys. */
const CHROMATIC_DEGREES: Record<'major' | 'minor', Record<number, string>> = {
  major: { 1: 'bII', 3: 'bIII', 6: '#IV', 8: 'bVI', 10: 'bVII' },
  minor: { 1: 'bII', 4: '#III', 6: '#IV', 9: '#VI', 11: '#VII' },
};

function parseKey(key: string): { root: number; letter: number; isMinor: boolean } {
  const parts = key.split(' ');
  const rootName = parts[0];
  const isMinor = parts[1]?.toLowerCase() === 'minor';
  return { root: NOTE_VALUES[rootName] ?? 0, letter: NOTE_LETTERS.indexOf(rootName[0]), isMinor };
}

export function calculateRomanNumeral(chord: Chord, key: string): string {
  const { root: keyRoot, letter: keyLetter, isMinor } = parseKey(key);
  const chordRoot = NOTE_VALUES[chord.root];
  const chordLetter = NOTE_LETTERS.indexOf(chord.root[0]);
  if (chordRoot === undefined || chordLetter < 0 || keyLetter < 0) return '?';

  const interval = (chordRoot - keyRoot + 12) % 12;
  const scaleIntervals = isMinor ? MINOR_SCALE_INTERVALS : MAJOR_SCALE_INTERVALS;

  // The written note letter determines the degree. Pitch distance alone
  // cannot distinguish F# (#IV) from Gb (bV), or Bb (bVII) from A# (#VI).
  let degree = (chordLetter - keyLetter + 7) % 7;
  // A root spelled enharmonically to a scale tone (D# in Ab major, G# for the
  // Ab tonic) reads as that scale tone rather than as a doubly altered degree.
  const diatonicDegree = scaleIntervals.indexOf(interval);
  if (diatonicDegree >= 0) degree = diatonicDegree;

  let numeral = ROMAN_NUMERALS[degree];
  const expectedInterval = scaleIntervals[degree];
  if (interval !== expectedInterval) {
    // Semitones away from the scale degree, wrapped into -6..+5: sharps above, flats below.
    const alteration = ((interval - expectedInterval + 18) % 12) - 6;
    numeral = Math.abs(alteration) === 1
      ? (alteration > 0 ? '#' : 'b') + numeral
      // Double accidentals never appear in chord analysis: use the conventional chromatic degree.
      : CHROMATIC_DEGREES[isMinor ? 'minor' : 'major'][interval] ?? numeral;
  }
  if (MINOR_QUALITIES.has(chord.quality)) {
    numeral = numeral.replace(/[IV]+/, (roman) => roman.toLowerCase());
  }

  return numeral + getQualitySuffix(chord.quality);
}

/**
 * Diatonic ⇔ Roman numeral has no chromatic prefix (#/b).
 * Used by `rankCandidatesByContext` to float in-key interpretations
 * to the top of MIDI candidate lists.
 */
export function isDiatonic(chord: Chord, key: string): boolean {
  const numeral = calculateRomanNumeral(chord, key);
  return !numeral.startsWith('#') && !numeral.startsWith('b');
}

/**
 * Rank candidate chord interpretations by harmonic context — diatonic
 * (in-key) interpretations float to the top, then ties break on the
 * candidates' original ordering (note-match score from `detectChordCandidates`).
 */
export function rankCandidatesByContext(
  candidates: Chord[],
  key: string,
): Chord[] {
  return [...candidates].sort((a, b) => {
    const aDiatonic = isDiatonic(a, key) ? 1 : 0;
    const bDiatonic = isDiatonic(b, key) ? 1 : 0;
    return bDiatonic - aDiatonic;
  });
}

function getQualitySuffix(quality: string): string {
  switch (quality) {
    case 'maj': case 'min': return '';
    case '7': return '7';
    case 'maj7': return 'maj7';
    case 'm7': return '7';
    case 'dim': return '\u00B0';
    case 'dim7': return '\u00B07';
    case 'm7b5': return '\u00F87';
    case 'aug': return '+';
    case 'aug7': return '+7';
    case 'sus2': return 'sus2';
    case 'sus4': return 'sus4';
    case '6': case 'm6': return '6';
    case '9': case 'm9': return '9';
    case 'maj9': return 'maj9';
    case '11': case 'm11': return '11';
    case '13': case 'm13': return '13';
    case 'mMaj7': return 'maj7';
    case 'mMaj9': return 'maj9';
    case 'add9': case 'madd9': case 'm(add9)': return 'add9';
    case 'm(add2)': return 'add2';
    case 'm(add4)': return 'add4';
    case 'm(add11)': return 'add11';
    case '6/9': case 'm6/9': return '6/9';
    case '7b9': return '7b9';
    case '7#9': return '7#9';
    case '7#11': return '7#11';
    case '7b13': return '7b13';
    default: return quality;
  }
}
