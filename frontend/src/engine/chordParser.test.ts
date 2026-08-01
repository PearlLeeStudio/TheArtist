import { test } from 'node:test';
import assert from 'node:assert/strict';
import { chordToString, parseChordName } from './chordParser.ts';

test('common spellings parse to the shared quality names', () => {
  assert.deepEqual(
    ['C', 'Dm', 'G7', 'Fmaj7', 'Bbm7b5', 'F/C'].map((text) => {
      const chord = parseChordName(text);
      return chord && [chord.root, chord.quality, chord.bass ?? null];
    }),
    [['C', 'maj', null], ['D', 'min', null], ['G', '7', null], ['F', 'maj7', null],
      ['Bb', 'm7b5', null], ['F', 'maj', 'C']],
  );
});

test('an unknown root or an empty name is rejected', () => {
  assert.equal(parseChordName('H'), null);
  assert.equal(parseChordName(''), null);
});

test('an inherited property name is a plain unknown quality, never an object', () => {
  for (const suffix of ['constructor', 'toString', '__proto__', 'hasOwnProperty']) {
    const chord = parseChordName(`C${suffix}`);
    assert.ok(chord, `C${suffix} still parses`);
    assert.equal(typeof chord.quality, 'string');
    assert.equal(chord.quality, suffix);
    assert.equal(chordToString(chord), `C${suffix}`);
  }
});
