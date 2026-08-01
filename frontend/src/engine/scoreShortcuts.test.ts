import { test } from 'node:test';
import assert from 'node:assert/strict';
import { selectAllScore } from './scoreShortcuts.ts';
import type { SlotAddress } from '../models/types';

function key(overrides: Partial<Parameters<typeof selectAllScore>[0]> = {}) {
  let prevented = false;
  const event = {
    key: 'a', ctrlKey: true, metaKey: false, altKey: false, shiftKey: false,
    isComposing: false, defaultPrevented: false,
    preventDefault: () => { prevented = true; },
    ...overrides,
  };
  return { event, prevented: () => prevented };
}

test('Ctrl+A and Command+A select both chord slots of every measure', () => {
  for (const modifiers of [{ ctrlKey: true }, { ctrlKey: false, metaKey: true }]) {
    const input = key(modifiers);
    const selections: SlotAddress[][] = [];
    assert.equal(selectAllScore(input.event, 32, false, (...range) => selections.push(range)), true);
    assert.deepEqual(selections, [[{ measureIndex: 0, slotIndex: 0 }, { measureIndex: 31, slotIndex: 1 }]]);
    assert.equal(input.prevented(), true);
  }
});

test('text controls and open dialogs retain their native select-all', () => {
  const input = key();
  assert.equal(selectAllScore(input.event, 8, true, () => assert.fail('changed score selection')), false);
  assert.equal(input.prevented(), false);
});

test('plain typing, modified shortcuts, composition and handled events leave selection alone', () => {
  for (const options of [
    { ctrlKey: false }, { key: 'c' }, { altKey: true }, { shiftKey: true },
    { isComposing: true }, { defaultPrevented: true },
  ]) {
    const input = key(options);
    assert.equal(selectAllScore(input.event, 8, false, () => assert.fail('changed score selection')), false);
    assert.equal(input.prevented(), false);
  }
});

test('an empty score does not claim the browser shortcut', () => {
  const input = key();
  assert.equal(selectAllScore(input.event, 0, false, () => assert.fail('selected a nonexistent measure')), false);
  assert.equal(input.prevented(), false);
});
