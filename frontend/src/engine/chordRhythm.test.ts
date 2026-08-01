import { test } from 'node:test';
import assert from 'node:assert/strict';
import { harmonyStrikesForChords, secondsPerMeterBeat } from './playSettings.ts';
import type { Chord } from '../models/types.ts';
const c: Chord = { root: 'C', quality: 'maj', romanNumeral: '', voicing: [], extensions: [] };
const f: Chord = { ...c, root: 'F' };
test('6/8 uses three quarter notes per bar and follows live tempo changes', () => {
  assert.equal(secondsPerMeterBeat(120, 8) * 6, 1.5);
  assert.equal(secondsPerMeterBeat(60, 8) * 6, 3);
  assert.equal(secondsPerMeterBeat(120, 4) * 4, 2);
});
test('Charleston and half time sound both different chords, without doubling repeated chords', () => {
  assert.deepEqual(harmonyStrikesForChords([0, 1.5], 4, [c, f]).map((s) => s.pos), [0, 1.5, 2]);
  assert.deepEqual(harmonyStrikesForChords([0], 4, [c, f]).map((s) => s.pos), [0, 2]);
  assert.deepEqual(harmonyStrikesForChords([0], 4, [c, c]).map((s) => s.pos), [0]);
});
test('second-slot preview, slash bass change and odd meter preserve the new chord', () => {
  assert.deepEqual(harmonyStrikesForChords([0], 4, [null, c]).map((s) => s.pos), [0, 2]);
  assert.deepEqual(harmonyStrikesForChords([0], 3, [c, f]).map((s) => s.pos), [0, 1.5]);
  assert.deepEqual(harmonyStrikesForChords([0], 4, [c, { ...c, bass: 'E' }]).map((s) => s.pos), [0, 2]);
  for (const s of harmonyStrikesForChords([0, 1.5, 3], 4, [c, f])) {
    assert.ok(s.pos >= 2 || s.pos + s.dur <= 2);
  }
});
