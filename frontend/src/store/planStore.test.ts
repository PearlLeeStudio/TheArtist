import { test } from 'node:test';
import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';
import type { Chord, Measure, SectionType, Song } from '../models/types.ts';
import type { PlanOptions, PlanSectionResult, PlanTrace } from '../engine/planClient.ts';

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
// stores read it while loading. songStore's autosave reaches the timers
// through window.
const storage = new Map<string, string>();
Object.defineProperty(globalThis, 'localStorage', { configurable: true, value: {
  getItem: (key: string) => storage.get(key) ?? null,
  setItem: (key: string, value: string) => { storage.set(key, value); },
} });
Object.defineProperty(globalThis, 'window', { configurable: true, value: globalThis });

const { usePlanStore } = await import('./planStore.ts');
const { useSongStore } = await import('./songStore.ts');
const { useSessionStore } = await import('./sessionStore.ts');
const { startNewSongFromForm } = await import('./newSong.ts');

const chord = (root: string, quality: string): Chord => ({ root, quality, extensions: [], romanNumeral: '', voicing: [] });
const measure = (index: number, chords: [Chord | null, Chord | null]): Measure => ({ id: `m${index}`, index, chords });
const frame = (event: string, data: unknown) => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
const options: PlanOptions = { targetBars: 16, wantOutro: true, contrast: 'medium', modelKey: 'ft_f1' };

/** A 2-bar verse sketch in C major with two empty bars after it. */
function sketchSong(): Song {
  return {
    id: 'sketch', title: 'Verse — 2 bars', key: 'C major', bpm: 110, genre: 'pop', timeSignature: [4, 4],
    measures: [
      measure(0, [chord('C', 'maj7'), chord('A', 'm7')]),
      measure(1, [chord('D', 'm7'), chord('G', '7')]),
      measure(2, [null, null]),
      measure(3, [null, null]),
    ],
    sections: [{ id: 'verse', type: 'verse', startMeasure: 0, length: 2 }],
    autoTitle: true,
  };
}

function sectionResult(index: number, sectionType: SectionType, startMeasure: number, bars: number, extra: Partial<PlanSectionResult> = {}): PlanSectionResult {
  const chords: Record<string, [string, string]> = {};
  for (let m = startMeasure; m < startMeasure + bars; m++) chords[String(m)] = m % 2 === 0 ? ['Fmaj7', 'G7'] : ['Em7', 'Am7'];
  return {
    sectionType, title: `${sectionType} — ${bars} bars`, startMeasure, bars, chordCount: bars * 2,
    suggestions: [{ label: 'Option A', chords }], source: 'ai', modelUsed: 'ft_f1',
    cadence: 'authentic', rulesSlots: [bars * 2 - 2, bars * 2 - 1], reason: 'Lift', cites: [], regenerated: false,
    ...extra,
  };
}

function trace(overrides: Partial<PlanTrace> = {}): PlanTrace {
  return {
    status: 'ok', genre: 'pop', key: 'C major', bpm: 110, timeSignature: [4, 4], plan: { sections: [] },
    validation: [{ attempt: 1, ok: true, errors: [] }], sections: [],
    elapsedMs: 20700, generationMs: 0, error: null, ...overrides,
  };
}

function streamResponse(parts: string[]): Response {
  const encoder = new TextEncoder();
  return new Response(new ReadableStream({
    start(controller) {
      for (const part of parts) controller.enqueue(encoder.encode(part));
      controller.close();
    },
  }), { headers: { 'Content-Type': 'text/event-stream' } });
}

/** The genre form after the 2-bar sketch: a chorus of 8 bars and an outro of 6, 16 bars in all. */
const PLAN = { sections: [
  { type: 'chorus', bars: 8, cadence: 'authentic', reason: 'pop default form (rules)', cites: [] },
  { type: 'outro', bars: 6, cadence: 'plagal', reason: 'pop default form (rules)', cites: [] },
] };

/** A chorus of 8 bars and an outro of 6 after the 2-bar sketch: 16 bars. */
function completedRun() {
  const sections = [sectionResult(0, 'chorus', 2, 8), sectionResult(1, 'outro', 10, 6, { cadence: 'plagal', rulesSlots: [0, 10, 11] })];
  const final = trace({ plan: PLAN, sections });
  return [
    frame('plan', { attempt: 1, source: 'rules', plan: PLAN, t_ms: 2 }),
    frame('validate', { attempt: 1, ok: true, errors: [], t_ms: 3 }),
    frame('generate_section', { index: 0, ...sections[0], t_ms: 5000 }),
    frame('generate_section', { index: 1, ...sections[1], t_ms: 8000 }),
    frame('complete', final),
  ];
}

function chipSummary() {
  return usePlanStore.getState().chips.map((chip) => [chip.label, chip.badge, chip.note, chip.ok]);
}

test('a progress frame sets the label the dialog shows, and a single long section applies and undoes cleanly', async (t) => {
  useSongStore.getState().setSong(sketchSong());
  const original = structuredClone(useSongStore.getState().song);
  const section = sectionResult(0, 'chorus', 2, 14, { rulesSlots: [] });
  t.mock.method(globalThis, 'fetch', async (_input: RequestInfo | URL, init?: RequestInit) => {
    const body = JSON.parse(String(init?.body));
    // The request names no planner and no chord source: there is one of each
    assert.equal('planner' in body, false);
    assert.equal('chordSource' in body, false);
    return streamResponse([
      frame('progress', { label: 'Composing chorus (1/1)…' }),
      frame('generate_section', { index: 0, ...section, t_ms: 10 }),
      frame('complete', trace({ sections: [section] })),
    ]);
  });
  await usePlanStore.getState().startPlan(options);
  assert.equal(usePlanStore.getState().progress, 'Composing chorus (1/1)…');
  assert.deepEqual(chipSummary(), [['Chorus · 14 bars', 'own model', null, true]]);
  usePlanStore.getState().applyPlan();
  assert.equal(usePlanStore.getState().applied, true);
  assert.equal(useSongStore.getState().song.measures.length, 16);
  assert.ok(useSongStore.getState().song.measures.slice(2).every(bar => bar.chords.every(Boolean)));
  usePlanStore.getState().undoPlan();
  assert.deepEqual(useSongStore.getState().song, original);
});

test('starting a new song preserves the complete score and clears the old form run', async (t) => {
  useSessionStore.setState({ sessions: [] });
  useSongStore.getState().setSong(sketchSong());
  t.mock.method(globalThis, 'fetch', async () => streamResponse(completedRun()));
  await usePlanStore.getState().startPlan(options);
  usePlanStore.getState().applyPlan();
  const previous = structuredClone(useSongStore.getState().song);
  const generation = { model: 'ft_f1', genre: previous.genre };
  startNewSongFromForm(generation);
  assert.deepEqual(useSessionStore.getState().sessions[0].song, previous);
  assert.deepEqual(useSessionStore.getState().sessions[0].generation, generation);
  const fresh = useSongStore.getState().song;
  assert.notEqual(fresh.id, previous.id);
  assert.equal(fresh.title, 'Untitled');
  assert.equal(fresh.genre, previous.genre);
  assert.equal(fresh.key, previous.key);
  assert.equal(fresh.bpm, previous.bpm);
  assert.deepEqual(fresh.timeSignature, previous.timeSignature);
  assert.deepEqual(fresh.sections, []);
  assert.equal(fresh.measures.length, 4);
  assert.ok(fresh.measures.every(bar => bar.chords.every(value => value === null)));
  assert.equal(usePlanStore.getState().status, 'idle');
  // Opening the stored project and starting again must not duplicate it.
  useSongStore.getState().setSong(previous);
  startNewSongFromForm(generation);
  assert.equal(useSessionStore.getState().sessions.length, 1);
});

test('a failed project save leaves the current score intact when starting a new song', (t) => {
  useSessionStore.setState({ sessions: [] });
  useSongStore.getState().setSong(sketchSong());
  const previous = structuredClone(useSongStore.getState().song);
  t.mock.method(useSessionStore.getState(), 'add', () => { throw new Error('Storage is full'); });
  assert.throws(() => startNewSongFromForm({ model: 'ft_f1', genre: 'pop' }), /Storage is full/);
  assert.deepEqual(useSongStore.getState().song, previous);
  assert.equal(useSessionStore.getState().sessions.length, 0);
});

test('a plan cannot apply to a sketch edited after the request, but keeps its trace', async (t) => {
  t.mock.method(globalThis, 'fetch', async () => streamResponse(completedRun()));
  const edits = [
    () => useSongStore.getState().setKey('D major'),
    () => useSongStore.getState().setGenre('jazz'),
    () => useSongStore.getState().setChord({ measureIndex: 0, slotIndex: 0 }, chord('F', 'maj7')),
  ];
  for (const edit of edits) {
    useSongStore.getState().setSong(sketchSong());
    await usePlanStore.getState().startPlan(options);
    edit();
    const edited = structuredClone(useSongStore.getState().song);
    usePlanStore.getState().applyPlan();
    assert.deepEqual(useSongStore.getState().song, edited);
    assert.equal(usePlanStore.getState().status, 'failed');
    assert.match(usePlanStore.getState().error ?? '', /score changed/i);
    assert.equal(usePlanStore.getState().applied, false);
    assert.notEqual(usePlanStore.getState().trace, null);
  }
});

test('a run lands nothing until Apply, which appends the sections after the sketch; Undo restores the exact score', async (t) => {
  useSongStore.getState().setSong(sketchSong());
  const before = structuredClone(useSongStore.getState().song);
  const fetchMock = t.mock.method(globalThis, 'fetch', async (input: RequestInfo | URL, init?: RequestInit) => {
    assert.equal(input, '/api/plan/song/stream');
    const body = JSON.parse(String(init?.body));
    assert.equal(body.measures.length, 2);
    assert.deepEqual(body.measures[1].roman, ['ii7', 'V7']);
    assert.deepEqual(body.sections, [{ type: 'verse', start: 0, length: 2, label: null }]);
    assert.equal(body.targetBars, 16);
    assert.equal(body.contrast, 'medium');
    assert.equal(body.modelKey, 'ft_f1');
    return streamResponse(completedRun());
  });

  await usePlanStore.getState().startPlan(options);
  const done = usePlanStore.getState();
  assert.equal(fetchMock.mock.callCount(), 1);
  assert.equal(done.status, 'done');
  assert.deepEqual(chipSummary(), [
    ['plan', 'rules', null, true],
    ['validate ok', null, null, true],
    ['Chorus · 8 bars', 'own model', 'cadence: rules (2)', true],
    ['Outro · 6 bars', 'own model', 'cadence: rules (3)', true],
  ]);
  assert.equal(done.sections.length, 2);
  assert.equal(done.applied, false);
  assert.deepEqual(useSongStore.getState().song, before);

  usePlanStore.getState().applyPlan();
  const song = useSongStore.getState().song;
  assert.equal(usePlanStore.getState().applied, true);
  assert.deepEqual(usePlanStore.getState().snapshot, before);
  assert.equal(song.measures.length, 16);
  assert.deepEqual(song.measures.slice(0, 2), before.measures.slice(0, 2));
  assert.deepEqual((song.sections ?? []).map((s) => [s.type, s.startMeasure, s.length, s.pending]),
    [['verse', 0, 2, undefined], ['chorus', 2, 8, undefined], ['outro', 10, 6, undefined]]);
  assert.equal(song.title, 'Verse–Chorus–Outro');
  const first = song.measures[2].chords[0];
  assert.equal(first?.root, 'F');
  assert.equal(first?.quality, 'maj7');
  assert.equal(first?.romanNumeral, 'IVmaj7');
  assert.ok((first?.voicing.length ?? 0) > 0);
  assert.deepEqual(song.measures[15].chords.map((c) => c && `${c.root}${c.quality}`), ['Em7', 'Am7']);
  // Apply is idempotent: a second call changes nothing.
  usePlanStore.getState().applyPlan();
  assert.equal(useSongStore.getState().song.measures.length, 16);

  usePlanStore.getState().undoPlan();
  assert.deepEqual(useSongStore.getState().song, before);
  assert.equal(usePlanStore.getState().applied, false);
  assert.equal(usePlanStore.getState().snapshot, null);
  // The band stays after Undo, so the sections can be applied again.
  assert.equal(usePlanStore.getState().status, 'done');
  assert.equal(JSON.parse(usePlanStore.getState().exportTrace() ?? 'null')?.status, 'ok');
  usePlanStore.getState().dismiss();
  assert.equal(usePlanStore.getState().status, 'idle');
});

test('a run that fails on the server shows a red chip with its reason and leaves the song untouched', async (t) => {
  useSongStore.getState().setSong(sketchSong());
  const before = structuredClone(useSongStore.getState().song);
  const reason = 'the run\'s 45 s of chord-model time is used up';
  const failed = trace({ status: 'error', error: { kind: 'timeout', message: reason } });
  t.mock.method(globalThis, 'fetch', async () => streamResponse([
    frame('plan', { attempt: 1, source: 'rules', plan: { sections: [] }, t_ms: 2 }),
    frame('validate', { attempt: 1, ok: false, errors: [{ code: 'total_bars', section: null, message: 'the planned bars make 28, not 32' }], t_ms: 3 }),
    frame('error', failed),
  ]));
  await usePlanStore.getState().startPlan(options);
  const state = usePlanStore.getState();
  assert.equal(state.status, 'failed');
  assert.equal(state.error, reason);
  assert.deepEqual(chipSummary(), [
    ['plan', 'rules', null, true],
    ['validate failed (1)', null, null, false],
    ['failed: timeout', null, null, false],
  ]);
  assert.equal(state.chips[2].message, reason);
  assert.equal(state.chips[1].message, 'total_bars: the planned bars make 28, not 32');
  assert.deepEqual(useSongStore.getState().song, before);
  usePlanStore.getState().applyPlan();
  assert.deepEqual(useSongStore.getState().song, before);
  assert.equal(usePlanStore.getState().applied, false);
  assert.equal(JSON.parse(usePlanStore.getState().exportTrace() ?? 'null')?.error?.kind, 'timeout');
  usePlanStore.getState().dismiss();
});

test('an unreachable server and a request the client refuses both fail without a trace', async (t) => {
  useSongStore.getState().setSong(sketchSong());
  const before = structuredClone(useSongStore.getState().song);
  t.mock.method(globalThis, 'fetch', async () => { throw new TypeError('Failed to fetch'); });
  await usePlanStore.getState().startPlan(options);
  assert.equal(usePlanStore.getState().status, 'failed');
  assert.match(usePlanStore.getState().error ?? '', /Could not reach the server/);
  assert.deepEqual(chipSummary(), [['failed', null, null, false]]);
  assert.equal(usePlanStore.getState().exportTrace(), null);

  useSongStore.getState().setGenre(null);
  await usePlanStore.getState().startPlan(options);
  assert.equal(usePlanStore.getState().status, 'failed');
  assert.match(usePlanStore.getState().error ?? '', /Pick a genre/);
  assert.deepEqual(useSongStore.getState().song, { ...before, genre: null });
  usePlanStore.getState().dismiss();
});

test('cancelling aborts the request, hides the band and leaves the song untouched', async (t) => {
  useSongStore.getState().setSong(sketchSong());
  const before = structuredClone(useSongStore.getState().song);
  let signal: AbortSignal | undefined;
  t.mock.method(globalThis, 'fetch', (_input: RequestInfo | URL, init?: RequestInit) => new Promise<Response>((_resolve, reject) => {
    signal = init?.signal ?? undefined;
    signal?.addEventListener('abort', () => reject(signal?.reason), { once: true });
  }));
  const run = usePlanStore.getState().startPlan(options);
  assert.equal(usePlanStore.getState().status, 'running');
  usePlanStore.getState().dismiss();
  assert.equal(signal?.aborted, true);
  await run;
  assert.equal(usePlanStore.getState().status, 'idle');
  assert.deepEqual(usePlanStore.getState().chips, []);
  assert.deepEqual(useSongStore.getState().song, before);
});

test('the genre form is labelled rules on the plan chip and the sections keep their own source', async (t) => {
  useSongStore.getState().setSong(sketchSong());
  t.mock.method(globalThis, 'fetch', async () => streamResponse(completedRun()));
  await usePlanStore.getState().startPlan(options);
  assert.equal(usePlanStore.getState().status, 'done');
  assert.deepEqual(chipSummary(), [
    ['plan', 'rules', null, true],
    ['validate ok', null, null, true],
    ['Chorus · 8 bars', 'own model', 'cadence: rules (2)', true],
    ['Outro · 6 bars', 'own model', 'cadence: rules (3)', true],
  ]);
  // The tooltips: the plan as one line, then each section's model, cadence and reason
  assert.deepEqual(usePlanStore.getState().chips.map((chip) => chip.message), [
    'chorus 8 (authentic) · outro 6 (plagal)',
    null,
    'model ft_f1 · cadence authentic · Lift',
    'model ft_f1 · cadence plagal · Lift',
  ]);
  usePlanStore.getState().dismiss();
});

test('a reused chorus retains its source through the stream and Apply/Undo', async (t) => {
  useSongStore.getState().setSong(sketchSong());
  const before = structuredClone(useSongStore.getState().song);
  const reused = sectionResult(0, 'chorus', 2, 8, { source: 'reuse', modelUsed: null,
    reuse: { origin: 'sketch', sectionIndex: null, startMeasure: 0, source: 'sketch', modelUsed: null,
      retainedSlots: Array.from({ length: 14 }, (_, i) => i), changedSlots: [14, 15] } });
  const sections = [reused, sectionResult(1, 'outro', 10, 6)];
  t.mock.method(globalThis, 'fetch', async () => streamResponse([
    frame('generate_section', { index: 0, ...reused, t_ms: 1 }),
    frame('complete', trace({ sections })),
  ]));
  await usePlanStore.getState().startPlan({ ...options, chorusMode: 'reuse' });
  assert.deepEqual(chipSummary()[0], ['Chorus · 8 bars', 'reused chorus', 'from bar 1 · 2 ending slots changed', true]);
  assert.deepEqual(usePlanStore.getState().trace?.sections[0].reuse, reused.reuse);
  usePlanStore.getState().applyPlan();
  assert.equal(useSongStore.getState().song.measures.length, 16);
  assert.deepEqual(useSongStore.getState().song.measures.slice(0, 2), before.measures.slice(0, 2));
  usePlanStore.getState().undoPlan();
  assert.deepEqual(useSongStore.getState().song, before);
  usePlanStore.getState().dismiss();
});

test('a section the backend\'s rule-based generator answered is labelled rules', async (t) => {
  useSongStore.getState().setSong(sketchSong());
  const sections = [sectionResult(0, 'chorus', 2, 8, { source: 'rules', modelUsed: null }), sectionResult(1, 'outro', 10, 6)];
  t.mock.method(globalThis, 'fetch', async () => streamResponse([
    frame('plan', { attempt: 1, source: 'rules', plan: { sections: [] }, t_ms: 1 }),
    frame('validate', { attempt: 1, ok: true, errors: [], t_ms: 2 }),
    frame('generate_section', { index: 0, ...sections[0], t_ms: 3 }),
    frame('generate_section', { index: 1, ...sections[1], t_ms: 4 }),
    frame('complete', trace({ sections })),
  ]));
  await usePlanStore.getState().startPlan(options);
  assert.deepEqual(chipSummary().slice(2), [
    ['Chorus · 8 bars', 'rules', 'cadence: rules (2)', true],
    ['Outro · 6 bars', 'own model', 'cadence: rules (2)', true],
  ]);
  assert.deepEqual(usePlanStore.getState().sections.map((section) => [section.sectionType, section.source]),
    [['chorus', 'rules'], ['outro', 'ai']]);
  usePlanStore.getState().dismiss();
});

test('a replaced song drops the run, while Undo\'s own replacement keeps it', async (t) => {
  useSongStore.getState().setSong(sketchSong());
  t.mock.method(globalThis, 'fetch', async () => streamResponse(completedRun()));
  await usePlanStore.getState().startPlan(options);
  usePlanStore.getState().applyPlan();
  usePlanStore.getState().undoPlan();
  assert.equal(usePlanStore.getState().status, 'done');
  useSongStore.getState().clearSong();
  assert.equal(usePlanStore.getState().status, 'idle');
});
