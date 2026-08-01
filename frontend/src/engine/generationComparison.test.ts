import { test } from 'node:test';
import assert from 'node:assert/strict';
import type { Chord, Song } from '../models/types.ts';
import {
  canApplyComparison,
  captureGenerationComparison,
  compareCandidate,
  comparisonUpdates,
  matchesCapturedRange,
  type TargetChord,
} from './generationComparison.ts';

function chord(root: string, quality = 'maj'): Chord {
  return { root, quality, voicing: [48, 52, 55], extensions: [], romanNumeral: 'I' };
}

function song(): Song {
  return {
    id: 'working-song', title: 'Sketch', key: 'C major', genre: 'pop', bpm: 80, timeSignature: [4, 4],
    measures: [
      { id: 'm0', index: 0, chords: [chord('C'), chord('G', '7')] },
      { id: 'm1', index: 1, chords: [null, chord('F')] },
      { id: 'm2', index: 2, chords: [chord('D', 'm7'), chord('G')] },
    ],
  };
}

/** Use the returned patch as a caller would. No whole-song replacement can
 * accidentally hide an unselected edit in these assertions. */
function apply(song: Song, updates: TargetChord[]): Song {
  const measures = song.measures.map((measure, measureIndex) => {
    const selected = updates.filter((u) => u.address.measureIndex === measureIndex);
    if (selected.length === 0) return measure;
    const chords = [...measure.chords] as [Chord | null, Chord | null];
    for (const update of selected) chords[update.address.slotIndex] = update.chord;
    return { ...measure, chords };
  });
  return { ...song, measures };
}

test('original captures only the requested slots and owns nested chord data', () => {
  const before = song();
  const slots = [{ measureIndex: 0, slotIndex: 1 as const }, { measureIndex: 1, slotIndex: 0 as const }];
  const original = captureGenerationComparison(before, [...slots, slots[0]], 7);
  before.measures[0].chords[1]!.voicing[0] = 99;
  before.measures[0].chords[1]!.extensions.push('b9');
  slots[0].measureIndex = 2;
  assert.equal(original.original.length, 2);
  assert.deepEqual(original.original[0].address, { measureIndex: 0, slotIndex: 1 });
  assert.deepEqual(original.original[0].chord?.voicing, [48, 52, 55]);
  assert.deepEqual(original.original[0].chord?.extensions, []);
  assert.equal(original.original[1].chord, null);
});

test('A/B/Original preserves outside edits and restores empty slots and exact original voicings', () => {
  const before = song();
  const original = captureGenerationComparison(before, [
    { measureIndex: 0, slotIndex: 1 }, { measureIndex: 1, slotIndex: 0 },
  ], 2);
  const a = new Map([['0-0', chord('A')], ['0-1', chord('Bb')], ['1-0', chord('E')]]);
  const versionA = apply(before, comparisonUpdates(original, before, 2, a));
  assert.equal(versionA.measures[0].chords[0], before.measures[0].chords[0]);
  const outsideEdit = chord('Ab');
  const withOutsideEdit = apply(versionA, [{ address: { measureIndex: 2, slotIndex: 1 }, chord: outsideEdit }]);
  // A partial B response must not retain A in the missing slot.
  const b = new Map([['0-1', chord('D', '7')]]);
  const versionB = apply(withOutsideEdit, comparisonUpdates(original, withOutsideEdit, 2, b));
  assert.equal(versionB.measures[1].chords[0], null);
  const restored = apply(versionB, comparisonUpdates(original, versionB, 2));
  assert.deepEqual(restored.measures[0].chords[1], before.measures[0].chords[1]);
  assert.equal(restored.measures[1].chords[0], null);
  assert.equal(restored.measures[1].chords[1], before.measures[1].chords[1]);
  assert.equal(restored.measures[2].chords[1], outsideEdit);
  assert.equal(restored.measures[2], versionB.measures[2]);
  assert.ok(matchesCapturedRange(original, restored, 2));
  // Applying/restoring must not expose the snapshot's mutable arrays.
  restored.measures[0].chords[1]!.voicing[0] = 12;
  assert.equal(original.original[0].chord?.voicing[0], 48);
});

test('late results cannot address a loaded copy, another song, or a replaced measure', () => {
  const before = song();
  const original = captureGenerationComparison(before, [{ measureIndex: 1, slotIndex: 1 }], 5);
  const candidate = new Map([['1-1', chord('A')]]);
  assert.deepEqual(comparisonUpdates(original, structuredClone(before), 6, candidate), []);
  assert.deepEqual(comparisonUpdates(original, { ...before, id: 'other-song' }, 5, candidate), []);
  const replacedMeasure = structuredClone(before);
  replacedMeasure.measures[1].id = 'replacement';
  assert.deepEqual(comparisonUpdates(original, replacedMeasure, 5, candidate), []);
  const unrelatedEdit = apply(before, [{ address: { measureIndex: 0, slotIndex: 0 }, chord: chord('E') }]);
  assert.ok(canApplyComparison(original, unrelatedEdit, 5));
});

test('change previews use only target symbols, including slash bass, extensions and rests', () => {
  const before = song();
  const original = captureGenerationComparison(before, [
    { measureIndex: 0, slotIndex: 0 }, { measureIndex: 0, slotIndex: 1 }, { measureIndex: 1, slotIndex: 0 },
  ], 0);
  const candidate = new Map<string, Chord | null>([
    ['0-0', { ...chord('C'), voicing: [], romanNumeral: '' }],
    ['0-1', { ...chord('G', '7'), bass: 'B', extensions: ['b9'] }],
    ['1-0', chord('F')],
    ['2-0', chord('Ab')],
  ]);
  const rows = compareCandidate(original, candidate);
  assert.equal(rows.length, 3);
  assert.deepEqual(rows.map((r) => r.changed), [false, true, true]);
  assert.equal(rows[2].before, null);
  assert.equal(rows[2].after?.root, 'F');
});

test('new section Original means empty chords without removing its section or earlier music', () => {
  const before = song();
  before.measures.push({ id: 'new', index: 3, chords: [null, null] });
  before.sections = [{ id: 'chorus', type: 'chorus', startMeasure: 3, length: 1 }];
  const original = captureGenerationComparison(before, [
    { measureIndex: 3, slotIndex: 0 }, { measureIndex: 3, slotIndex: 1 },
  ], 3, 'section');
  const generated = apply(before, comparisonUpdates(original, before, 3,
    new Map([['3-0', chord('F')], ['3-1', chord('C')]])));
  const restored = apply(generated, comparisonUpdates(original, generated, 3));
  assert.equal(original.kind, 'section');
  assert.deepEqual(restored.measures[3].chords, [null, null]);
  assert.equal(restored.sections, before.sections);
  assert.equal(restored.measures[0], before.measures[0]);
});

test('an applied version remains active through outside edits but not a target voicing change', () => {
  const before = song();
  const applied = captureGenerationComparison(before, [{ measureIndex: 0, slotIndex: 0 }], 4);
  const outside = apply(before, [{ address: { measureIndex: 2, slotIndex: 1 }, chord: chord('Bb') }]);
  assert.ok(matchesCapturedRange(applied, outside, 4));
  const changedVoicing = apply(outside, [{
    address: { measureIndex: 0, slotIndex: 0 }, chord: { ...chord('C'), voicing: [60, 64, 67] },
  }]);
  assert.equal(matchesCapturedRange(applied, changedVoicing, 4), false);
});
