import { test } from 'node:test';
import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';
import { parseChordName } from '../engine/chordParser.ts';
import type { Song } from '../models/types.ts';

const sourceRoot = new URL('../', import.meta.url).href;
registerHooks({
  resolve(specifier, context, nextResolve) {
    if (context.parentURL?.startsWith(sourceRoot) && /^\.\.?\//.test(specifier) && !/\.\w+$/.test(specifier)) specifier += '.ts';
    return nextResolve(specifier, context);
  },
  load(url, context, nextLoad) {
    if (url === `${sourceRoot}engine/playback.ts`) return { format: 'module', shortCircuit: true,
      source: 'export function cancelPendingStart() {}\nexport function setLiveBpm() {}\nexport function stopPlayback() {}\n' };
    return nextLoad(url, context);
  },
});

/** A song as an earlier build stored it: source content still attached, and the
 *  reading provenance of that build (no analysisSource, a model name, request ids). */
function legacySong(): Song {
  return {
    id: 'legacy', title: 'My arrangement title', bpm: 90, key: 'C major', genre: 'jazz', timeSignature: [4, 4],
    measures: [{ id: 'bar', index: 0, chords: [parseChordName('C')!, null] }],
    sections: [{ id: 'verse', type: 'verse', label: 'My verse', startMeasure: 0, length: 1 }],
    importSource: { filename: 'private-original.pdf', model: 'muse-spark-1.3', requestId: 'request-accounting', mode: 'infer_harmony',
      measures: [{ chordBasis: 'inferred', confidence: 'medium', evidence: 'Private duration and pitch text', originalChords: ['C'],
        notes: ['C6'], writtenNotes: ['C5'], bassNotes: ['C3'],
        analysisSource: 'meta', model: 'muse-spark-1.3', requestId: 'bar-request' }] },
  };
}

const storage = new Map<string, string>([['theartist-song', JSON.stringify(legacySong())]]);
Object.defineProperty(globalThis, 'localStorage', { configurable: true, value: {
  getItem: (key: string) => storage.get(key) ?? null,
  setItem: (key: string, value: string) => { storage.set(key, value); },
} });
Object.defineProperty(globalThis, 'window', { configurable: true, value: globalThis });

const { useSongStore } = await import('./songStore.ts');
const { useSessionStore } = await import('./sessionStore.ts');
const noSourceContent = (value: unknown) => assert.doesNotMatch(JSON.stringify(value), /private-original|Private duration|C6|C5|C3|"notes"|"writtenNotes"|"bassNotes"/);

test('legacy current-song recovery immediately cleans browser storage and keeps the arrangement', () => {
  const recovered = useSongStore.getState().song;
  noSourceContent(recovered);
  noSourceContent(JSON.parse(storage.get('theartist-song')!));
  assert.equal(recovered.title, 'My arrangement title');
  assert.deepEqual(recovered.measures, JSON.parse(JSON.stringify(legacySong().measures)));
  assert.deepEqual(recovered.sections, legacySong().sections);
  assert.equal(recovered.importSource?.requestId, 'request-accounting');
  assert.equal(recovered.importSource?.model, 'muse-spark-1.3');
  // The bar keeps its stored reading provenance; only source content is removed.
  assert.deepEqual(recovered.importSource?.measures?.[0], {
    chordBasis: 'inferred', confidence: 'medium', evidence: '', originalChords: ['C'],
    analysisSource: 'meta', model: 'muse-spark-1.3', requestId: 'bar-request',
  });
});

test('import undo, edit history, saved projects and autosave never retain extracted source content', async () => {
  const songs = useSongStore.getState();
  songs.setSong(legacySong());
  songs.importSong({ ...legacySong(), id: 'replacement' });
  noSourceContent(useSongStore.getState().song);
  noSourceContent(useSongStore.getState().previousImportSong);
  useSongStore.getState().clearSlots([{ measureIndex: 0, slotIndex: 0 }]);
  noSourceContent(useSongStore.getState().editHistory);
  assert.equal(useSongStore.getState().editHistory[0].measures[0].chords[0]?.root, 'C');
  useSongStore.getState().undoEdit();
  assert.equal(useSongStore.getState().song.measures[0].chords[0]?.root, 'C');
  const saved = useSessionStore.getState().add(legacySong(), { model: 'ft_f1', genre: 'jazz' });
  noSourceContent(saved);
  noSourceContent(useSessionStore.getState().sessions);
  noSourceContent(JSON.parse(storage.get('theartist-sessions')!));
  useSessionStore.getState().update(saved.id, { song: { ...legacySong(), title: 'User edited title' } });
  assert.equal(useSessionStore.getState().sessions[0].song.title, 'User edited title');
  noSourceContent(useSessionStore.getState().sessions);
  await new Promise((resolve) => setTimeout(resolve, 550));
  noSourceContent(JSON.parse(storage.get('theartist-song')!));
  assert.equal(JSON.parse(storage.get('theartist-song')!).measures[0].chords[0].root, 'C');
  useSongStore.getState().undoImport();
  assert.equal(useSongStore.getState().song.id, 'legacy');
  noSourceContent(useSongStore.getState().song);
});
