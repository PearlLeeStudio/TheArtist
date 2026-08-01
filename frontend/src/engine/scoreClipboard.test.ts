import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  CHORD_CLIPBOARD_TYPE,
  clipboardText,
  copySelection,
  holdsWholeBars,
  orderedSlots,
  parseClipboardText,
  planPaste,
  readClipboardPayload,
  scoreEditKey,
  selectionHasChords,
  serializeClipboard,
} from './scoreClipboard.ts';
import { parseChordName } from './chordParser.ts';
import type { Chord, Measure, SlotAddress } from '../models/types';

function chord(name: string, voicing: number[] = [60, 64, 67]): Chord {
  const parsed = parseChordName(name);
  if (!parsed) throw new Error(`fixture chord ${name}`);
  return { ...parsed, romanNumeral: 'I', voicing };
}

/** Bars from chord names; '' is an empty slot. */
function measures(...bars: [string, string][]): Measure[] {
  return bars.map(([a, b], index) => ({
    id: `m${index}`, index,
    chords: [a ? chord(a) : null, b ? chord(b) : null],
  }));
}

function slot(measureIndex: number, slotIndex: 0 | 1): SlotAddress {
  return { measureIndex, slotIndex };
}

function range(from: SlotAddress, to: SlotAddress): SlotAddress[] {
  const out: SlotAddress[] = [];
  for (let i = from.measureIndex * 2 + from.slotIndex; i <= to.measureIndex * 2 + to.slotIndex; i++) {
    out.push(slot(Math.floor(i / 2), (i % 2) as 0 | 1));
  }
  return out;
}

const names = (slots: (Chord | null)[]) => slots.map((c) => c ? `${c.root}${c.quality}` : null);

test('copy keeps the selected slots in score order with their voicings and empty slots', () => {
  const score = measures(['Dm7', 'G7'], ['Cmaj7', ''], ['Am7', 'Am7']);
  const clip = copySelection(score, [slot(1, 1), slot(0, 1), slot(1, 0), slot(0, 1)]);
  assert.ok(clip);
  assert.deepEqual(names(clip.slots), ['G7', 'Cmaj7', null]);
  assert.equal(clip.firstSlotIndex, 1);
  assert.deepEqual(clip.slots[0]?.voicing, [60, 64, 67]);
  // The copy is independent of the score.
  clip.slots[0]!.voicing.push(72);
  assert.deepEqual(score[0].chords[1]?.voicing, [60, 64, 67]);
  assert.equal(copySelection(score, []), null);
  assert.equal(copySelection(score, [slot(9, 0)]), null);
});

test('clipboard text writes bar lines, repeats and empty slots the way charts do', () => {
  const score = measures(['Dm7', 'G7'], ['Cmaj7', 'Cmaj7'], ['', ''], ['F', 'G/B']);
  const clip = copySelection(score, range(slot(0, 0), slot(3, 1)))!;
  assert.equal(clipboardText(clip), 'Dm7 G7 | Cmaj7 | - | F G/B');
  const fromSecondSlot = copySelection(score, range(slot(0, 1), slot(1, 1)))!;
  assert.equal(clipboardText(fromSecondSlot), 'G7 | Cmaj7');
});

test('the structured payload round-trips and rejects foreign data', () => {
  const score = measures(['Dm7', 'G7'], ['Cmaj7', '']);
  const clip = copySelection(score, range(slot(0, 0), slot(1, 1)))!;
  const back = readClipboardPayload(serializeClipboard(clip));
  assert.ok(back);
  assert.deepEqual(names(back.slots), ['Dm7', 'G7', 'Cmaj7', null]);
  assert.deepEqual(back.slots[1]?.voicing, [60, 64, 67]);
  assert.equal(CHORD_CLIPBOARD_TYPE, 'application/x-theartist-chords');
  assert.equal(readClipboardPayload('not json'), null);
  assert.equal(readClipboardPayload('{"format":"other","version":1,"slots":[]}'), null);
  assert.equal(readClipboardPayload('{"format":"theartist-chords","version":1,"slots":[{"root":"H","quality":"maj"}]}'), null);
  assert.equal(readClipboardPayload('{"format":"theartist-chords","version":1,"slots":[]}'), null);
});

test('pasted chord text follows the bar-line convention', () => {
  assert.deepEqual(names(parseClipboardText('Dm7 G7 | Cmaj7 | - | % | C Am F G')!),
    ['Dm7', 'G7', 'Cmaj7', 'Cmaj7', null, null, null, null, 'Cmaj', 'Fmaj']);
  assert.deepEqual(names(parseClipboardText('Am7 D7 Gmaj7')!), ['Am7', 'D7', 'Gmaj7']);
  assert.deepEqual(names(parseClipboardText('Am7\nD7\n')!), ['Am7', 'Am7', 'D7', 'D7']);
  assert.deepEqual(names(parseClipboardText('N.C. | F#m7b5')!), [null, null, 'F#m7b5', 'F#m7b5']);
  assert.equal(parseClipboardText('hello world'), null);
  assert.equal(parseClipboardText('   '), null);
  assert.equal(parseClipboardText('- | -'), null);
});

test('a clipboard that divides the selection is tiled over it', () => {
  const plan = planPaste(range(slot(2, 0), slot(5, 1)), [chord('Dm7'), chord('G7')], 8);
  assert.ok(plan);
  assert.equal(plan.measuresToAdd, 0);
  assert.deepEqual(plan.targets.map((t) => [t.address.measureIndex, t.address.slotIndex, t.chord?.root ?? null]),
    [[2, 0, 'D'], [2, 1, 'G'], [3, 0, 'D'], [3, 1, 'G'], [4, 0, 'D'], [4, 1, 'G'], [5, 0, 'D'], [5, 1, 'G']]);
  assert.deepEqual(plan.select, [slot(2, 0), slot(5, 1)]);
  // One chord fills every selected slot.
  const fill = planPaste(range(slot(0, 0), slot(1, 1)), [chord('C')], 4)!;
  assert.deepEqual(fill.targets.map((t) => t.chord?.root), ['C', 'C', 'C', 'C']);
});

test('otherwise paste writes forward from the first selected slot and grows the score', () => {
  const plan = planPaste([slot(3, 1)], [chord('Dm7'), chord('G7'), null, chord('C')], 4);
  assert.ok(plan);
  assert.deepEqual(plan.targets.map((t) => [t.address.measureIndex, t.address.slotIndex, t.chord?.root ?? null]),
    [[3, 1, 'D'], [4, 0, 'G'], [4, 1, null], [5, 0, 'C']]);
  assert.equal(plan.measuresToAdd, 2);
  assert.deepEqual(plan.select, [slot(3, 1), slot(5, 0)]);
  // A three-slot selection does not tile two chords; the anchor is the earliest slot.
  const partial = planPaste([slot(1, 1), slot(1, 0), slot(2, 0)], [chord('Dm7'), chord('G7')], 4)!;
  assert.deepEqual(partial.targets.map((t) => [t.address.measureIndex, t.address.slotIndex]), [[1, 0], [1, 1]]);
  assert.equal(partial.measuresToAdd, 0);
  // Pasted chords are copies.
  const source = [chord('Dm7')];
  const copy = planPaste([slot(0, 0)], source, 1)!;
  copy.targets[0].chord!.voicing.push(1);
  assert.deepEqual(source[0].voicing, [60, 64, 67]);
  assert.equal(planPaste([], [chord('C')], 4), null);
  assert.equal(planPaste([slot(0, 0)], [], 4), null);
  assert.equal(planPaste([slot(7, 0)], [chord('C')], 4), null);
});

test('whole bars are pasted from the start of the clicked bar, never straddling a bar line', () => {
  const twoBars = [chord('Dm7'), chord('G7'), chord('Cmaj7'), chord('Cmaj7')];
  // Clicking the second half of bar 3 still pastes the two bars as bars 3 and 4.
  const plan = planPaste([slot(2, 1)], twoBars, 8, { wholeBars: true })!;
  assert.deepEqual(plan.targets.map((t) => [t.address.measureIndex, t.address.slotIndex]), [[2, 0], [2, 1], [3, 0], [3, 1]]);
  assert.equal(plan.measuresToAdd, 0);
  assert.deepEqual(plan.select, [slot(2, 0), slot(3, 1)]);
  // At the last bar of a four-bar score it adds exactly the one bar that is missing.
  const atEnd = planPaste([slot(3, 1)], twoBars, 4, { wholeBars: true })!;
  assert.equal(atEnd.measuresToAdd, 1);
  assert.equal(atEnd.targets[0].address.measureIndex, 3);
  // Dragging bars 4–7 (any slots) and pasting one bar tiles it over those whole bars.
  const tiled = planPaste([slot(4, 1), slot(5, 0), slot(6, 1), slot(7, 0)], [chord('F'), chord('G')], 8, { wholeBars: true })!;
  assert.deepEqual(tiled.targets.map((t) => `${t.address.measureIndex}.${t.address.slotIndex}:${t.chord?.root}`),
    ['4.0:F', '4.1:G', '5.0:F', '5.1:G', '6.0:F', '6.1:G', '7.0:F', '7.1:G']);
  // A half-bar copy keeps slot precision.
  const half = planPaste([slot(2, 1)], [chord('G7')], 8, { wholeBars: false })!;
  assert.deepEqual(half.targets.map((t) => [t.address.measureIndex, t.address.slotIndex]), [[2, 1]]);
  assert.equal(holdsWholeBars({ slots: twoBars, firstSlotIndex: 0 }), true);
  assert.equal(holdsWholeBars({ slots: twoBars.slice(1), firstSlotIndex: 1 }), false);
  assert.equal(holdsWholeBars({ slots: twoBars.slice(0, 3), firstSlotIndex: 0 }), false);
});

test('selection helpers', () => {
  const score = measures(['', ''], ['C', '']);
  assert.equal(selectionHasChords(score, [slot(0, 0), slot(0, 1)]), false);
  assert.equal(selectionHasChords(score, [slot(0, 1), slot(1, 0)]), true);
  assert.deepEqual(orderedSlots([slot(1, 1), slot(0, 0), slot(1, 1)]), [slot(0, 0), slot(1, 1)]);
});

test('edit keys: Ctrl/Cmd X C V Z and Delete/Backspace outside text controls', () => {
  const key = (overrides: Partial<Parameters<typeof scoreEditKey>[0]>) => scoreEditKey({
    key: 'c', ctrlKey: false, metaKey: false, altKey: false, shiftKey: false, isComposing: false, defaultPrevented: false, ...overrides,
  }, false);
  assert.equal(key({ key: 'c', ctrlKey: true }), 'copy');
  assert.equal(key({ key: 'X', metaKey: true }), 'cut');
  assert.equal(key({ key: 'v', metaKey: true }), 'paste');
  assert.equal(key({ key: 'z', ctrlKey: true }), 'undo');
  assert.equal(key({ key: 'Delete' }), 'clear');
  assert.equal(key({ key: 'Backspace' }), 'clear');
  assert.equal(key({ key: 'z', metaKey: true, shiftKey: true }), null, 'redo is not claimed');
  assert.equal(key({ key: 'c' }), null, 'plain typing');
  assert.equal(key({ key: 'c', ctrlKey: true, altKey: true }), null);
  assert.equal(key({ key: 'v', ctrlKey: true, isComposing: true }), null);
  assert.equal(key({ key: 'v', ctrlKey: true, defaultPrevented: true }), null);
  assert.equal(key({ key: 'Backspace', shiftKey: true }), null);
  assert.equal(scoreEditKey({ key: 'v', ctrlKey: true, metaKey: false, altKey: false, shiftKey: false, isComposing: false, defaultPrevented: false }, true), null, 'blocked in text controls and dialogs');
});
