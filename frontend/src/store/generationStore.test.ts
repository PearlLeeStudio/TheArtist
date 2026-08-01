import { test } from 'node:test';
import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';
import type { SavedSession, SlotAddress, Song } from '../models/types.ts';
import type { Suggestion } from '../engine/chordGenerator.ts';

// The stores import their modules without extensions (Vite resolves them),
// and songStore reaches the Web Audio stack through engine/playback. Resolve
// `.ts` for the relative imports under src/ and stand in for playback with
// the three no-ops songStore uses, so the stores load without a bundler.
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
// Node's own localStorage wants a storage file and warns without one; the
// stores read it while loading.
const storage = new Map<string, string>();
Object.defineProperty(globalThis, 'localStorage', { configurable: true, value: {
  getItem: (key: string) => storage.get(key) ?? null,
  setItem: (key: string, value: string) => { storage.set(key, value); },
} });

const { useGenerationStore } = await import('./generationStore.ts');
const { useSongStore } = await import('./songStore.ts');

type Snapshot = Pick<ReturnType<typeof useGenerationStore.getState>, 'loading' | 'phase' | 'showSuggestions' | 'error'>;

/** One store write as 'loading phase list error' words, to check a run's sequence of writes. */
function mark(state: Snapshot): string {
  return [
    state.loading ? 'loading' : 'idle',
    state.phase?.name ?? '-',
    state.showSuggestions ? 'shown' : 'hidden',
    state.error === null ? 'ok' : 'error',
  ].join(' ');
}

/** The store writes made while `run` is in progress, oldest first. */
async function writesDuring(run: () => Promise<unknown>): Promise<string[]> {
  const writes: string[] = [];
  const unsubscribe = useGenerationStore.subscribe((state) => { writes.push(mark(state)); });
  try { await run(); } finally { unsubscribe(); }
  return writes;
}

const slots: SlotAddress[] = [{ measureIndex: 0, slotIndex: 0 }, { measureIndex: 0, slotIndex: 1 }];
const suggestion: Suggestion = { label: 'A', chords: { 0: ['Dm7', 'G7'] } };
const frame = (event: string, data: unknown) => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;

function streamResponse(parts: string[]): Response {
  const encoder = new TextEncoder();
  return new Response(new ReadableStream({
    start(controller) {
      for (const part of parts) controller.enqueue(encoder.encode(part));
      controller.close();
    },
  }), { headers: { 'Content-Type': 'text/event-stream' } });
}

/** A song object that breaks captureGenerationComparison before any request. */
function brokenSong(): Song {
  return { ...useSongStore.getState().song, measures: undefined } as unknown as Song;
}

function starredPreset(): SavedSession {
  return {
    id: 'preset', name: 'Warm', createdAt: 0, starred: true,
    song: useSongStore.getState().song, generation: { model: 'ft_f4', genre: null },
  };
}

/** A list as a previous run left it, to check that a failed run clears it. */
function showPreviousList(): void {
  const song = useSongStore.getState().song;
  useGenerationStore.setState({
    suggestions: [suggestion],
    targetSlots: slots.map((s) => ({ ...s })),
    comparison: { songEpoch: 0, songId: song.id, kind: 'selection', original: [] },
    source: 'ai',
    showSuggestions: true,
    activeSuggestionIdx: 0,
  });
}

function assertFailedWithoutRequest(fetchCalls: number, writes: string[]): void {
  const state = useGenerationStore.getState();
  assert.equal(fetchCalls, 0);
  assert.deepEqual(writes, ['loading - hidden ok', 'loading - hidden error', 'idle - hidden error']);
  assert.equal(state.loading, false);
  assert.equal(state.phase, null);
  assert.match(state.error ?? '', /^Generation failed\. .+ Your score has not changed\.$/);
  assert.deepEqual(state.suggestions, []);
  assert.deepEqual(state.targetSlots, []);
  assert.equal(state.comparison, null);
  assert.equal(state.showSuggestions, false);
  assert.equal(state.activeSuggestionIdx, null);
}

test('a generate run with a stubbed stream lands its suggestions for the selected slots', async (t) => {
  const song = useSongStore.getState().song;
  const fetchMock = t.mock.method(globalThis, 'fetch', async (input: RequestInfo | URL, init?: RequestInit) => {
    assert.equal(input, '/api/generate/stream');
    const body = JSON.parse(String(init?.body));
    assert.deepEqual(body.selectedMeasures, [0]);
    assert.equal(body.modelKey, 'ft_f1');
    return streamResponse([
      frame('composing', { label: 'Composing', step: 1, total: 3, creativity: 'balanced' }),
      frame('result', { suggestions: [suggestion], source: 'ai' }),
    ]);
  });
  const writes = await writesDuring(() => useGenerationStore.getState().runGenerate(song, slots, 'ft_f1', null));
  const state = useGenerationStore.getState();
  assert.equal(fetchMock.mock.callCount(), 1);
  assert.deepEqual(writes, [
    'loading - hidden ok', 'loading composing hidden ok', 'loading composing shown ok', 'idle - shown ok',
  ]);
  assert.equal(state.error, null);
  assert.deepEqual(state.suggestions, [suggestion]);
  assert.equal(state.source, 'ai');
  assert.equal(state.showSuggestions, true);
  assert.equal(state.activeSuggestionIdx, null);
  assert.deepEqual(state.targetSlots, slots);
  assert.equal(state.comparison?.kind, 'selection');
  assert.equal(state.comparison?.songId, song.id);
  assert.deepEqual(state.comparison?.original.map((o) => o.address), slots);
  // The run keeps the section it sent (here none), so a later review reads it the same way
  assert.equal(state.comparison?.sectionType, null);
  assert.deepEqual(useSongStore.getState().song, song);
});

test('a song without measures fails the generate run before any request, with the error shown and loading off', async (t) => {
  showPreviousList();
  const fetchMock = t.mock.method(globalThis, 'fetch', async () => { throw new Error('no request expected'); });
  const writes = await writesDuring(() => useGenerationStore.getState().runGenerate(brokenSong(), slots, 'ft_f1', null));
  assertFailedWithoutRequest(fetchMock.mock.callCount(), writes);
});

test('a song without measures fails the favorite run the same way', async (t) => {
  showPreviousList();
  const fetchMock = t.mock.method(globalThis, 'fetch', async () => { throw new Error('no request expected'); });
  const writes = await writesDuring(() =>
    useGenerationStore.getState().runFavoriteGenerate(brokenSong(), slots, [starredPreset()], null));
  assertFailedWithoutRequest(fetchMock.mock.callCount(), writes);
});
