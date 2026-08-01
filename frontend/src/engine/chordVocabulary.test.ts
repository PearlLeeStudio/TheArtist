import { test } from 'node:test';
import assert from 'node:assert/strict';
import { parseChordName, chordToString } from './chordParser.ts';
import { QUALITY_INTERVALS, generatePianoVoicing } from './voicingEngine.ts';
import { getChordDisplayParts } from './chordDisplay.ts';
import { parseImportedChord } from './scoreImport.ts';
import { resolveQuality } from './manualChordDraft.ts';

const pcs = (notes: number[]) => [...new Set(notes.map((n) => ((n % 12) + 12) % 12))].sort((a, b) => a - b);

test('chart spellings of suspended sevenths, 6/9, maj7#11 and added tones parse to voiced qualities', () => {
  const cases: Array<[string, string, number[]]> = [
    ['D7sus4', '7sus4', [2, 7, 9, 0]],
    ['G7sus', '7sus4', [7, 0, 2, 5]],
    ['Gsus7', '7sus4', [7, 0, 2, 5]],
    ['C6/9', '6/9', [0, 4, 9, 2]],
    ['C69', '6/9', [0, 4, 9, 2]],
    ['Am6/9', 'm6/9', [9, 0, 6, 11]],
    ['EbM7#11', 'maj7#11', [3, 7, 2, 9]],
    ['Fmaj7#11', 'maj7#11', [5, 9, 4, 11]],
    ['Dmadd9', 'madd9', [2, 5, 9, 4]],
    ['C5', '5', [0, 7]],
    ['A9sus4', '9sus4', [9, 2, 7, 11]],
  ];
  for (const [symbol, quality, expected] of cases) {
    const chord = parseChordName(symbol);
    assert.ok(chord, symbol);
    assert.equal(chord.quality, quality, symbol);
    assert.ok(Object.hasOwn(QUALITY_INTERVALS, quality), `${quality} is voiced`);
    assert.deepEqual(pcs(generatePianoVoicing(chord)), pcs(expected.map((pc) => 60 + pc)), symbol);
    const imported = parseImportedChord(symbol, 'C major');
    assert.ok(imported, `${symbol} imports`);
    assert.equal(imported.quality, quality);
  }
});

test('slash basses still parse beside the 6/9 spelling and unknown suffixes stay unsupported on import', () => {
  const slash = parseImportedChord('Dm7/G', 'C major');
  assert.equal(slash?.bass, 'G');
  assert.equal(parseImportedChord('C6/9/E', 'C major')?.bass, 'E', 'a 6/9 keeps a written slash bass');
  assert.equal(parseImportedChord('C7alt', 'C major'), null);
  assert.equal(parseImportedChord('Cfoo', 'C major'), null);
});

test('display parts place the suspension below and the extension above the root', () => {
  assert.deepEqual(getChordDisplayParts(parseChordName('D7sus4')!), { root: 'D', sup: '7', sub: 'sus4', bass: '' });
  assert.deepEqual(getChordDisplayParts({ root: 'D', quality: 'm(add9)', extensions: [], voicing: [], romanNumeral: '' }),
    { root: 'D', sup: 'add9', sub: 'm', bass: '' });
  // Manual alterations are no longer dropped from the staff symbol.
  assert.deepEqual(getChordDisplayParts({ root: 'G', quality: '7', extensions: ['b9', '#11'], voicing: [], romanNumeral: '' }),
    { root: 'G', sup: '7(b9,#11)', sub: '', bass: '' });
  assert.deepEqual(getChordDisplayParts(parseChordName('C6/9')!), { root: 'C', sup: '6/9', sub: '', bass: '' });
  assert.deepEqual(getChordDisplayParts(parseChordName('C5')!), { root: 'C', sup: '', sub: '5', bass: '' });
  assert.equal(chordToString(parseChordName('G7sus4')!), 'G7sus4');
});

test('the manual builder\'s common qualities all have voicings', () => {
  const builder = [
    resolveQuality('maj', '7', '', 'sus4'), resolveQuality('maj', '9', '', 'sus4'), resolveQuality('maj', '7', '', 'sus2'),
    resolveQuality('maj', '', 'add9', ''), resolveQuality('min', '', 'add9', ''), resolveQuality('min', '', 'add2', ''),
    resolveQuality('min', '', 'add4', ''), resolveQuality('min', '', 'add11', ''), resolveQuality('maj', '', 'add2', ''),
    resolveQuality('5', '', '', ''), resolveQuality('5', '7', '', ''), resolveQuality('min', 'maj7', '', ''),
    resolveQuality('dim', '7', '', ''), resolveQuality('dim', '9', '', ''), resolveQuality('aug', '7', '', ''),
    resolveQuality('min', '6', '', ''), resolveQuality('maj', '13', '', ''), resolveQuality('min', '11', '', ''),
  ];
  assert.deepEqual(builder.filter((quality) => !Object.hasOwn(QUALITY_INTERVALS, quality)), []);
});
