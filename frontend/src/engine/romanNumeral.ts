import type { Chord } from '../models/types';
import { NOTE_VALUES, MAJOR_SCALE_INTERVALS, MINOR_SCALE_INTERVALS } from './constants';

const ROMAN_NUMERALS = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII'];

const MINOR_QUALITIES = new Set([
  'min', 'm7', 'dim', 'dim7', 'm7b5', 'm6', 'm9', 'm11', 'm13', 'mMaj7',
]);

function parseKey(key: string): { root: number; isMinor: boolean } {
  const parts = key.split(' ');
  const rootName = parts[0];
  const isMinor = parts[1]?.toLowerCase() === 'minor';
  return { root: NOTE_VALUES[rootName] ?? 0, isMinor };
}

export function calculateRomanNumeral(chord: Chord, key: string): string {
  const { root: keyRoot, isMinor } = parseKey(key);
  const chordRoot = NOTE_VALUES[chord.root];
  if (chordRoot === undefined) return '?';

  const interval = (chordRoot - keyRoot + 12) % 12;
  const scaleIntervals = isMinor ? MINOR_SCALE_INTERVALS : MAJOR_SCALE_INTERVALS;

  let degree = 0;
  let minDist = 12;
  for (let i = 0; i < scaleIntervals.length; i++) {
    const dist = Math.abs(scaleIntervals[i] - interval);
    if (dist < minDist) {
      minDist = dist;
      degree = i;
    }
  }

  let numeral = ROMAN_NUMERALS[degree];
  if (MINOR_QUALITIES.has(chord.quality)) {
    numeral = numeral.toLowerCase();
  }

  const expectedInterval = scaleIntervals[degree];
  if (interval !== expectedInterval) {
    const diff = interval - expectedInterval;
    if (diff === 1 || diff === -11) numeral = '#' + numeral;
    else if (diff === -1 || diff === 11) numeral = 'b' + numeral;
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
 *
 * `_prevChord` / `_nextChord` are reserved for future cadential heuristics
 * (V→I bonus, ii→V setup bonus, etc.); v1 ignores them but the signature
 * is stable.
 */
export function rankCandidatesByContext(
  candidates: Chord[],
  key: string,
  _prevChord?: Chord | null,
  _nextChord?: Chord | null,
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
    case 'sus2': return 'sus2';
    case 'sus4': return 'sus4';
    case '6': case 'm6': return '6';
    case '9': case 'm9': return '9';
    case 'maj9': return 'maj9';
    case '11': return '11';
    case '13': return '13';
    case 'mMaj7': return 'maj7';
    case 'add9': return 'add9';
    case '7b9': return '7b9';
    case '7#9': return '7#9';
    case '7#11': return '7#11';
    case '7b13': return '7b13';
    default: return quality;
  }
}
