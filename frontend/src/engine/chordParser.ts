import type { Chord } from '../models/types';
import { NOTE_VALUES } from './constants.ts';

const QUALITY_MAP: Record<string, string> = {
  '': 'maj',
  'M': 'maj',
  'maj': 'maj',
  'major': 'maj',
  'm': 'min',
  'min': 'min',
  'minor': 'min',
  '7': '7',
  'maj7': 'maj7',
  'M7': 'maj7',
  'm7': 'm7',
  'min7': 'm7',
  'dim': 'dim',
  'dim7': 'dim7',
  'o7': 'dim7',
  'm7b5': 'm7b5',
  'ø7': 'm7b5',
  'aug': 'aug',
  '+': 'aug',
  'aug7': 'aug7',
  '+7': 'aug7',
  '7#5': 'aug7',
  'sus2': 'sus2',
  'sus4': 'sus4',
  'sus': 'sus4',
  '6': '6',
  'm6': 'm6',
  '9': '9',
  'm9': 'm9',
  'maj9': 'maj9',
  '11': '11',
  'm11': 'm11',
  '13': '13',
  'm13': 'm13',
  'add9': 'add9',
  'mMaj7': 'mMaj7',
  '7b9': '7b9',
  '7#9': '7#9',
  '7#11': '7#11',
  '7b13': '7b13',
  // Chart spellings of the qualities the voicing engine knows
  '7sus': '7sus4',
  'sus7': '7sus4',
  '7sus4': '7sus4',
  '7sus2': '7sus2',
  '9sus4': '9sus4',
  '9sus2': '9sus2',
  '13sus4': '13sus4',
  '69': '6/9',
  '6/9': '6/9',
  'm69': 'm6/9',
  'm6/9': 'm6/9',
  'M9': 'maj9',
  'M11': 'maj11',
  'M13': 'maj13',
  'maj11': 'maj11',
  'maj13': 'maj13',
  'M7#11': 'maj7#11',
  'maj7#11': 'maj7#11',
  'mM9': 'mMaj9',
  'mMaj9': 'mMaj9',
  'minmaj7': 'mMaj7',
  'mM7': 'mMaj7',
  'Δ': 'maj7',
  'Δ7': 'maj7',
  'Δ9': 'maj9',
  '°': 'dim',
  '°7': 'dim7',
  'ø': 'm7b5',
  '-': 'min',
  '-7': 'm7',
  '-9': 'm9',
  'madd9': 'madd9',
  'add2': 'add2',
  'add4': 'add4',
  'add11': 'add11',
  '7b5': '7b5',
  '5': '5',
};

export function parseChordName(input: string): Chord | null {
  const trimmed = input.trim();
  if (!trimmed) return null;

  // Handle slash chords: split on last '/' that has a valid bass note after it
  let mainPart = trimmed;
  let bass: string | undefined;
  const slashIdx = trimmed.lastIndexOf('/');
  if (slashIdx > 0) {
    const possibleBass = trimmed.slice(slashIdx + 1);
    if (Object.hasOwn(NOTE_VALUES, possibleBass)) {
      bass = possibleBass;
      mainPart = trimmed.slice(0, slashIdx);
    }
  }

  let root = '';
  let rest = mainPart;

  if (rest.length >= 2 && (rest[1] === '#' || rest[1] === 'b')) {
    root = rest.slice(0, 2);
    rest = rest.slice(2);
  } else if (rest.length >= 1) {
    root = rest[0];
    rest = rest.slice(1);
  }

  if (!Object.hasOwn(NOTE_VALUES, root)) return null;

  const extensions: string[] = [];
  const extMatch = rest.match(/\(([^)]+)\)/);
  if (extMatch) {
    extensions.push(...extMatch[1].split(',').map(s => s.trim()));
    rest = rest.replace(extMatch[0], '');
  }

  // Own keys only: "Cconstructor" must not pick up Object.prototype members as a quality.
  const quality = Object.hasOwn(QUALITY_MAP, rest) ? QUALITY_MAP[rest] : (rest || 'maj');

  return { root, quality, bass, romanNumeral: '', voicing: [], extensions };
}

export function chordToString(chord: Chord): string {
  const q = chord.quality === 'maj' ? '' : chord.quality === 'min' ? 'm' : chord.quality;
  const base = `${chord.root}${q}`;
  return chord.bass ? `${base}/${chord.bass}` : base;
}
