import type { Chord } from '../models/types';

export interface ChordDisplayParts {
  root: string;       // C, Db, F# — large
  sup: string;        // 7, maj7, 9, 11, 13, b9, #11 — superscript (top-right)
  sub: string;        // m, dim, aug, sus4, sus2 — subscript (bottom-right)
  bass: string;       // slash bass note, e.g. "G" in C/G
}

const COMPOUND_MAP: Record<string, { sub: string; sup: string }> = {
  'maj':    { sub: '', sup: '' },
  'min':    { sub: 'm', sup: '' },
  '7':      { sub: '', sup: '7' },
  'maj7':   { sub: '', sup: 'maj7' },
  'm7':     { sub: 'm', sup: '7' },
  'dim':    { sub: 'dim', sup: '' },
  'dim7':   { sub: 'dim', sup: '7' },
  'm7b5':   { sub: 'm', sup: '7b5' },
  'aug':    { sub: '+', sup: '' },
  'sus2':   { sub: 'sus2', sup: '' },
  'sus4':   { sub: 'sus4', sup: '' },
  '6':      { sub: '', sup: '6' },
  'm6':     { sub: 'm', sup: '6' },
  '9':      { sub: '', sup: '9' },
  'm9':     { sub: 'm', sup: '9' },
  'maj9':   { sub: '', sup: 'maj9' },
  '11':     { sub: '', sup: '11' },
  'm11':    { sub: 'm', sup: '11' },
  '13':     { sub: '', sup: '13' },
  'm13':    { sub: 'm', sup: '13' },
  'add9':   { sub: '', sup: 'add9' },
  'mMaj7':  { sub: 'm', sup: 'maj7' },
  '7b9':    { sub: '', sup: '7b9' },
  '7#9':    { sub: '', sup: '7#9' },
  '7#11':   { sub: '', sup: '7#11' },
  '7b13':   { sub: '', sup: '7b13' },
};

export function getChordDisplayParts(chord: Chord): ChordDisplayParts {
  const mapped = COMPOUND_MAP[chord.quality];
  const bass = chord.bass || '';
  if (mapped) {
    return { root: chord.root, sup: mapped.sup, sub: mapped.sub, bass };
  }
  return { root: chord.root, sup: chord.quality, sub: '', bass };
}
