import { test } from 'node:test';
import assert from 'node:assert/strict';
import { VexFlow } from 'vexflow/bravura';
import { STANDARD_KEYS, STANDARD_KEY_SIGNATURES, keySignatureFor, normalizeKeySignature } from './keySignatures.ts';
import { buildImportedSong, createScoreImportDraft, importMeasureRomanPreview, parseImportedChord, validateScoreImportDraft } from './scoreImport.ts';

test('all 30 standard major/minor signatures exist in VexFlow with the expected accidental count and type', () => {
  assert.equal(STANDARD_KEYS.length, 30);
  assert.equal(new Set(STANDARD_KEYS).size, 30);
  assert.equal(STANDARD_KEYS.filter((key) => key.endsWith(' major')).length, 15);
  assert.equal(STANDARD_KEYS.filter((key) => key.endsWith(' minor')).length, 15);
  for (const signature of STANDARD_KEY_SIGNATURES) {
    assert.equal(normalizeKeySignature(signature.label), signature.label);
    assert.equal(keySignatureFor(signature.label), signature.vexflow);
    const actual = VexFlow.keySignature(signature.vexflow);
    assert.equal(actual.length, Math.abs(signature.accidentals), signature.label);
    assert.ok(actual.every((accidental) => accidental.type === (signature.accidentals < 0 ? 'b' : '#')), signature.label);
  }
});

test('seven-flat/sharp and previously missing minor keys retain their written signature and tonic Roman analysis on import', () => {
  const cases = [
    ['Cb major', 'Cb', 'Cb', 'I'], ['C# major', 'C#', 'C#', 'I'],
    ['D# minor', 'D#m', 'D#m', 'i'], ['Ab minor', 'Abm', 'Abm', 'i'], ['A# minor', 'A#m', 'A#m', 'i'],
  ];
  for (const [key, symbol, vexflow, numeral] of cases) {
    assert.equal(keySignatureFor(key), vexflow, key);
    assert.equal(importMeasureRomanPreview(symbol, key), numeral, key);
    const draft = createScoreImportDraft({
      title: 'Written key', key, timeSignature: [4, 4], sections: [], warnings: [],
      analysisSource: 'local', model: null, requestId: null,
      measures: [{ chords: [symbol], sourceLabel: '', needsReview: false, note: '' }],
    }, 'key.pdf');
    const song = buildImportedSong(draft);
    assert.equal(song.key, key);
    assert.deepEqual(song.measures[0].chords.map((chord) => chord?.romanNumeral), [numeral, numeral]);
  }
});

test('standard key spelling is normalized while theoretical signatures require an explicit supported key choice', () => {
  assert.equal(normalizeKeySignature(' c♯ MAJOR '), 'C# major');
  assert.equal(normalizeKeySignature('a♭ minor'), 'Ab minor');
  for (const key of ['E# major', 'Fb major', 'G# major', 'D# major', 'A# major', 'B# minor', 'Db minor', 'Gb minor']) {
    assert.equal(normalizeKeySignature(key), null, key);
    assert.equal(keySignatureFor(key), null, key);
    const draft = createScoreImportDraft({
      title: 'Needs key review', key, timeSignature: [4, 4], sections: [], warnings: [],
      analysisSource: 'local', model: null, requestId: null,
      measures: [{ chords: ['C'], sourceLabel: '', needsReview: false, note: '' }],
    }, 'key.pdf');
    assert.match(validateScoreImportDraft(draft).errors.join(' '), /30 standard.*Correct unsupported keys manually/);
    assert.throws(() => buildImportedSong(draft));
  }
  // Chord spelling is independent of the set of supported key signatures.
  assert.equal(parseImportedChord('E#', 'C# major')?.romanNumeral, 'III');
  assert.equal(parseImportedChord('Cb', 'Cb major')?.romanNumeral, 'I');
});
