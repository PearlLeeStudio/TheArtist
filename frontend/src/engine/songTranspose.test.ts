import { test } from 'node:test';
import assert from 'node:assert/strict';
import type { Chord, Song } from '../models/types.ts';
import { transposeSongKey } from './songTranspose.ts';
import { NOTE_VALUES } from './constants.ts';

const chord = (root: string, quality = 'maj', bass?: string): Chord => ({
  root, quality, bass, extensions: [], voicing: [], romanNumeral: '',
});
function songWith(chords: [Chord | null, Chord | null]): Song {
  return { id: 'transpose-score', title: 'Scale degrees', key: 'C major', bpm: 80,
    genre: null, timeSignature: [4, 4], measures: [{ id: 'bar-1', index: 0, chords }] };
}

test('the Key control still transposes actual pitches while preserving tonic and leading-tone degrees', () => {
  const original = songWith([chord('C'), chord('B', 'dim')]);
  const before = structuredClone(original);
  const moved = transposeSongKey(original, 'D major');
  assert.deepEqual(moved.measures[0].chords.map((c) => [c?.root, c?.romanNumeral]), [['D', 'I'], ['C#', 'vii°']]);
  assert.ok(moved.measures[0].chords.every((c) => c!.voicing.length > 0));
  assert.deepEqual(original, before);
});

test('sharp and flat keys keep E-sharp and C-flat playable with the right scale degree', () => {
  const original = songWith([chord('C'), chord('B', 'dim')]);
  const sharp = transposeSongKey(original, 'F# major');
  assert.deepEqual(sharp.measures[0].chords.map((c) => [c?.root, c?.romanNumeral]), [['F#', 'I'], ['E#', 'vii°']]);
  assert.equal(sharp.measures[0].chords[1]?.voicing[0] % 12, 5);
  const flat = transposeSongKey(songWith([chord('F'), chord('G', '7')]), 'Gb major');
  assert.deepEqual(flat.measures[0].chords.map((c) => [c?.root, c?.romanNumeral]), [['Cb', 'IV'], ['Db', 'V7']]);
  assert.equal(flat.measures[0].chords[0]?.voicing[0] % 12, 11);
});

test('slash bass transposes in tune and rests and song metadata survive', () => {
  const original = songWith([chord('F', 'maj', 'A'), null]);
  const moved = transposeSongKey(original, 'D major');
  const result = moved.measures[0].chords[0]!;
  assert.equal(result.root, 'G');
  assert.equal(result.bass, 'B');
  assert.equal(result.romanNumeral, 'IV');
  assert.equal(result.voicing[0] % 12, NOTE_VALUES.B);
  assert.equal(moved.measures[0].chords[1], null);
  assert.equal(moved.title, original.title);
  assert.equal(moved.id, original.id);
});

test('changing major/minor mode at the same root reanalyzes without moving chord pitches', () => {
  const original = songWith([chord('A', 'min'), chord('G', '7')]);
  const moved = transposeSongKey(original, 'C minor');
  assert.deepEqual(moved.measures[0].chords.map((c) => [c?.root, c?.romanNumeral]), [['A', '#vi'], ['G', 'V7']]);
});
