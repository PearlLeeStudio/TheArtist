import { test } from 'node:test';
import assert from 'node:assert/strict';
import type { Chord } from '../models/types.ts';
import { calculateRomanNumeral } from './romanNumeral.ts';

const chord = (root: string, quality = 'maj', bass?: string): Chord => ({
  root, quality, bass, extensions: [], voicing: [], romanNumeral: '',
});

test('C major analysis uses the chord quality for Roman case and seventh symbols', () => {
  const progression = [chord('C'), chord('D', 'min'), chord('E', 'min'), chord('F'),
    chord('G', '7'), chord('A', 'min'), chord('B', 'dim')];
  assert.deepEqual(progression.map((c) => calculateRomanNumeral(c, 'C major')),
    ['I', 'ii', 'iii', 'IV', 'V7', 'vi', 'vii°']);
});

test('the same unchanged chord is analyzed relative to the selected review key', () => {
  const c = chord('C');
  assert.equal(calculateRomanNumeral(c, 'C major'), 'I');
  assert.equal(calculateRomanNumeral(c, 'G major'), 'IV');
  assert.equal(calculateRomanNumeral(c, 'F major'), 'V');
  assert.equal(c.root, 'C');
});

test('borrowed and altered roots retain their written degree instead of the nearest pitch', () => {
  const roots = ['Bb', 'Eb', 'Ab', 'F#', 'Gb', 'C#', 'Db'];
  assert.deepEqual(roots.map((root) => calculateRomanNumeral(chord(root), 'C major')),
    ['bVII', 'bIII', 'bVI', '#IV', 'bV', '#I', 'bII']);
  assert.equal(calculateRomanNumeral(chord('F'), 'G major'), 'bVII');
  assert.equal(calculateRomanNumeral(chord('Ab'), 'Eb major'), 'IV');
});

test('minor keys retain their natural-minor degrees and minor extensions do not duplicate m', () => {
  const progression = [chord('A', 'min'), chord('B', 'dim'), chord('C'), chord('D', 'min'),
    chord('E', 'min'), chord('F'), chord('G')];
  assert.deepEqual(progression.map((c) => calculateRomanNumeral(c, 'A minor')),
    ['i', 'ii°', 'III', 'iv', 'v', 'VI', 'VII']);
  assert.equal(calculateRomanNumeral(chord('D', 'm11'), 'C major'), 'ii11');
  assert.equal(calculateRomanNumeral(chord('D', 'm13'), 'C major'), 'ii13');
});

test('slash bass and voicing do not change the harmonic root degree', () => {
  const inversion = { ...chord('C', 'maj', 'E'), voicing: [52, 55, 60] };
  assert.equal(calculateRomanNumeral(inversion, 'C major'), 'I');
  assert.equal(calculateRomanNumeral(chord('B', 'm7b5'), 'C major'), 'viiø7');
  assert.equal(calculateRomanNumeral(chord('H'), 'C major'), '?');
});

test('roots spelled enharmonically to a scale tone read as that scale tone, never as a double accidental', () => {
  // A reader spelled these with sharps inside Ab major (Hiromi score import).
  assert.equal(calculateRomanNumeral(chord('D#', '7'), 'Ab major'), 'V7');
  assert.equal(calculateRomanNumeral(chord('A#', 'm7'), 'Ab major'), 'ii7');
  assert.equal(calculateRomanNumeral(chord('G#', 'sus4'), 'Ab major'), 'Isus4');
  assert.equal(calculateRomanNumeral(chord('D#', 'aug', 'A'), 'Ab major'), 'V+');
  assert.equal(calculateRomanNumeral(chord('C#', 'maj7'), 'Ab major'), 'IVmaj7');
  // Written borrowed degrees keep their single accidental and letter.
  assert.equal(calculateRomanNumeral(chord('F#'), 'C major'), '#IV');
  assert.equal(calculateRomanNumeral(chord('Gb'), 'C major'), 'bV');
  assert.equal(calculateRomanNumeral(chord('Bb'), 'G major'), 'bIII');
  // An enharmonic spelling of a diatonic degree in a sharp key reads as that degree too.
  assert.equal(calculateRomanNumeral(chord('Gb', '7'), 'E major'), 'II7');
  assert.equal(calculateRomanNumeral(chord('A#', 'dim'), 'E minor'), '#iv°');
});

test('new chord vocabulary keeps minor case and readable suffixes', () => {
  assert.equal(calculateRomanNumeral(chord('D', '7sus4'), 'C major'), 'II7sus4');
  assert.equal(calculateRomanNumeral(chord('D', 'madd9'), 'C major'), 'iiadd9');
  assert.equal(calculateRomanNumeral(chord('D', 'm(add9)'), 'C major'), 'iiadd9');
  assert.equal(calculateRomanNumeral(chord('F', '6/9'), 'C major'), 'IV6/9');
  assert.equal(calculateRomanNumeral(chord('E', 'm6/9'), 'C major'), 'iii6/9');
  assert.equal(calculateRomanNumeral(chord('Bb', 'maj7#11'), 'C major'), 'bVIImaj7#11');
});
