import { test } from 'node:test';
import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';
import { parseChordName } from '../engine/chordParser.ts';
import { copySelection, planPaste } from '../engine/scoreClipboard.ts';
import type { Chord, SlotAddress } from '../models/types.ts';

// Same loading arrangement as generationStore.test.ts: resolve the stores'
// extensionless imports, stand in for the Web Audio playback module and
// give Node a localStorage.
const SRC_URL = new URL('../', import.meta.url).href;
registerHooks({
  resolve(specifier, context, nextResolve) {
    if (context.parentURL?.startsWith(SRC_URL) && /^\.\.?\//.test(specifier) && !/\.\w+$/.test(specifier)) {
      specifier += '.ts';
    }
    return nextResolve(specifier, context);
  },
  load(url, context, nextLoad) {
    if (url === `${SRC_URL}engine/playback.ts`) {
      return {
        format: 'module',
        shortCircuit: true,
        source: 'export function cancelPendingStart() {}\nexport function setLiveBpm() {}\nexport function stopPlayback() {}\n',
      };
    }
    return nextLoad(url, context);
  },
});
const storage = new Map<string, string>();
Object.defineProperty(globalThis, 'localStorage', { configurable: true, value: {
  getItem: (key: string) => storage.get(key) ?? null,
  setItem: (key: string, value: string) => { storage.set(key, value); },
} });
// The autosave subscription schedules its write through window.
Object.defineProperty(globalThis, 'window', { configurable: true, value: globalThis });

const { useSongStore } = await import('./songStore.ts');

const slot = (measureIndex: number, slotIndex: 0 | 1): SlotAddress => ({ measureIndex, slotIndex });
const chord = (name: string): Chord => parseChordName(name)!;
const symbols = () => useSongStore.getState().song.measures.map((m) => m.chords.map((c) => c ? `${c.root}${c.quality}` : null));

function freshScore(): void {
  useSongStore.getState().clearSong();
  const { setChord } = useSongStore.getState();
  setChord(slot(0, 0), chord('Dm7'));
  setChord(slot(0, 1), chord('G7'));
  setChord(slot(1, 0), chord('Cmaj7'));
  setChord(slot(1, 1), chord('Cmaj7'));
}

test('clearSlots empties only the given slots and records one undo step', () => {
  freshScore();
  const before = useSongStore.getState().song;
  useSongStore.getState().clearSlots([slot(0, 1), slot(1, 0)]);
  assert.deepEqual(symbols(), [['Dm7', null], [null, 'Cmaj7'], [null, null], [null, null]]);
  assert.equal(useSongStore.getState().editHistory.length, 1);
  // Clearing already-empty slots changes nothing and records nothing.
  useSongStore.getState().clearSlots([slot(2, 0), slot(3, 1)]);
  assert.equal(useSongStore.getState().editHistory.length, 1);
  assert.equal(useSongStore.getState().undoEdit(), true);
  assert.deepEqual(useSongStore.getState().song.measures, before.measures);
  assert.equal(useSongStore.getState().editHistory.length, 0);
  assert.equal(useSongStore.getState().undoEdit(), false);
});

test('pasteSlots appends the bars a forward paste needs, analyzes the chords in the song key and selects the range', () => {
  freshScore();
  const { song } = useSongStore.getState();
  const clip = copySelection(song.measures, [slot(0, 0), slot(0, 1), slot(1, 0), slot(1, 1)])!;
  const plan = planPaste([slot(3, 0)], clip.slots, song.measures.length)!;
  assert.equal(plan.measuresToAdd, 1);
  useSongStore.getState().pasteSlots(plan);
  const state = useSongStore.getState();
  assert.equal(state.song.measures.length, 5);
  assert.deepEqual(symbols().slice(3), [['Dm7', 'G7'], ['Cmaj7', 'Cmaj7']]);
  assert.equal(state.song.measures[3].chords[0]?.romanNumeral, 'ii7');
  assert.deepEqual(state.selectedSlots, [slot(3, 0), slot(3, 1), slot(4, 0), slot(4, 1)]);
  assert.equal(state.song.measures[4].index, 4);
  // Undo removes the appended bars again and keeps the selection inside the score.
  assert.equal(state.undoEdit(), true);
  assert.equal(useSongStore.getState().song.measures.length, 4);
  assert.deepEqual(useSongStore.getState().selectedSlots, [slot(3, 0), slot(3, 1)]);
});

test('a tiled paste repeats the pattern over the selection and the pasted chords are independent copies', () => {
  freshScore();
  const { song } = useSongStore.getState();
  const clip = copySelection(song.measures, [slot(0, 0), slot(0, 1)])!;
  const selection = [slot(1, 0), slot(1, 1), slot(2, 0), slot(2, 1), slot(3, 0), slot(3, 1)];
  useSongStore.getState().pasteSlots(planPaste(selection, clip.slots, 4)!);
  assert.deepEqual(symbols(), [['Dm7', 'G7'], ['Dm7', 'G7'], ['Dm7', 'G7'], ['Dm7', 'G7']]);
  const measures = useSongStore.getState().song.measures;
  assert.notEqual(measures[1].chords[0], measures[2].chords[0]);
  assert.notEqual(measures[1].chords[0], measures[0].chords[0]);
});

test('a song replacement drops the edit history', () => {
  freshScore();
  useSongStore.getState().clearSlots([slot(0, 0)]);
  assert.equal(useSongStore.getState().editHistory.length, 1);
  useSongStore.getState().clearSong();
  assert.equal(useSongStore.getState().editHistory.length, 0);
  assert.equal(useSongStore.getState().undoEdit(), false);
});
