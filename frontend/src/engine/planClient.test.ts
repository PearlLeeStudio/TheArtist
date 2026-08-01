import { test } from 'node:test';
import assert from 'node:assert/strict';
import type { Chord, Measure, SectionType, Song } from '../models/types.ts';
import {
  availableTargetBars,
  buildPlanRequest,
  planBlocker,
  resolveTargetBars,
  runPlan,
  sketchBars,
  type PlanEvent,
  type PlanOptions,
  type PlanSectionResult,
  type PlanTrace,
} from './planClient.ts';

const chord = (root: string, quality: string): Chord => ({ root, quality, extensions: [], romanNumeral: '', voicing: [] });
const measure = (index: number, chords: [Chord | null, Chord | null]): Measure => ({ id: `m${index}`, index, chords });

// A 3-bar pop sketch in C major with a verse and an unnamed region, then an
// empty trailing bar that is not part of the sketch.
const song: Song = {
  id: 'sketch', title: 'Sketch', key: 'C major', bpm: 110.4, genre: 'pop', timeSignature: [4, 4],
  measures: [
    measure(0, [chord('C', 'maj7'), chord('A', 'm7')]),
    measure(1, [chord('D', 'm7'), { ...chord('G', '7'), bass: 'B' }]),
    measure(2, [chord('C', 'maj'), null]),
    measure(3, [null, null]),
  ],
  sections: [
    { id: 'tail', type: null, label: ' tail ', startMeasure: 2, length: 1 },
    { id: 'verse', type: 'verse', startMeasure: 0, length: 2 },
  ],
};
const options: PlanOptions = { targetBars: 32, wantOutro: true, contrast: 'medium', modelKey: 'ft_f1' };

// The backend tests' sample run (tests/test_planner.py, tests/test_plan_api.py)
const VALID_PLAN = { sections: [
  { type: 'chorus', bars: 8, cadence: 'authentic', reason: 'Lift after the ii-V in bars 1 and 3', cites: [{ bar: 1, roman: 'V7' }] },
  { type: 'verse', bars: 8, cadence: 'half', reason: 'Second verse, left open', cites: [] },
  { type: 'outro', bars: 8, cadence: 'plagal', reason: 'IV-I close like bar 6', cites: [{ bar: 6, roman: 'IVmaj7' }] },
] };
const frame = (event: string, data: unknown) => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;

function sectionResult(index: number, sectionType: SectionType, startMeasure: number, bars: number, extra: Partial<PlanSectionResult> = {}) {
  const chords: Record<string, [string, string]> = {};
  for (let m = startMeasure; m < startMeasure + bars; m++) chords[String(m)] = m % 2 === 0 ? ['Cmaj7', 'Am7'] : ['Dm7', 'G7'];
  return {
    index, sectionType, title: `${sectionType} — ${bars} bars`, startMeasure, bars, chordCount: bars * 2,
    suggestions: [{ label: 'Option A', chords }], source: 'ai', modelUsed: 'ft_f1',
    cadence: 'authentic', rulesSlots: [bars * 2 - 2, bars * 2 - 1], reason: 'Lift', cites: [], regenerated: false, t_ms: 100 * (index + 1),
    ...extra,
  };
}

function trace(overrides: Partial<PlanTrace> = {}): PlanTrace {
  return {
    status: 'ok', genre: 'pop', key: 'C major', bpm: 110, timeSignature: [4, 4], plan: VALID_PLAN,
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

test('the request carries the sketch up to its last used bar with the staff Roman numerals, null slots and locked sections', () => {
  const before = structuredClone(song);
  assert.equal(sketchBars(song), 3);
  const request = buildPlanRequest(song, options);
  assert.deepEqual(request, {
    genre: 'pop', key: 'C major', timeSignature: [4, 4], bpm: 110,
    measures: [
      { index: 0, chords: ['Cmaj7', 'Am7'], roman: ['Imaj7', 'vi7'] },
      { index: 1, chords: ['Dm7', 'G7/B'], roman: ['ii7', 'V7'] },
      { index: 2, chords: ['C', null], roman: ['I', null] },
    ],
    sections: [
      { type: 'verse', start: 0, length: 2, label: null },
      { type: null, start: 2, length: 1, label: 'tail' },
    ],
    targetBars: 32, wantOutro: true, contrast: 'medium', modelKey: 'ft_f1',
  });
  assert.deepEqual(song, before);
});

test('an empty score sends no sketch, and a song saved with the old bossa nova genre is sent as bossa', () => {
  const empty: Song = { ...song, genre: 'bossa nova', measures: [measure(0, [null, null]), measure(1, [null, null])], sections: [] };
  const request = buildPlanRequest(empty, { ...options, modelKey: null });
  assert.equal(sketchBars(empty), 0);
  assert.deepEqual(request.measures, []);
  assert.equal(request.genre, 'bossa');
  assert.equal(request.modelKey, null);
});

test('chorus choice is carried explicitly without mutating a locked sketch', () => {
  const before = structuredClone(song);
  assert.equal(buildPlanRequest(song, { ...options, chorusMode: 'reuse' }).chorusMode, 'reuse');
  assert.equal(buildPlanRequest(song, { ...options, chorusMode: 'new' }).chorusMode, 'new');
  assert.deepEqual(song, before);
});

test('a completed form and Undo never leave an unavailable target selected', () => {
  assert.equal(resolveTargetBars(64, 32), null);
  assert.equal(resolveTargetBars(51, 32), 64);
  assert.equal(resolveTargetBars(32, 32), 48);
  assert.equal(resolveTargetBars(0, 32), 32);
  assert.equal(resolveTargetBars(8, 64), 64);
});

test('no genre, a target that leaves no room and a sketch too long to plan are refused before any request', () => {
  assert.match(planBlocker({ ...song, genre: null }) ?? '', /Pick a genre/);
  assert.throws(() => buildPlanRequest({ ...song, genre: null }, options), /Pick a genre/);
  assert.equal(planBlocker(song), null);
  assert.deepEqual(availableTargetBars(3), [16, 32, 48, 64]);
  assert.deepEqual(availableTargetBars(15), [32, 48, 64]);
  assert.deepEqual(availableTargetBars(63), []);
  const fifteenBars: Song = { ...song, sections: [{ id: 'long', type: 'verse', startMeasure: 0, length: 15 }],
    measures: Array.from({ length: 15 }, (_, i) => measure(i, [null, null])) };
  assert.throws(() => buildPlanRequest(fifteenBars, { ...options, targetBars: 16 }), /at least 17 bars/);
  const tooLong: Song = { ...song, sections: [{ id: 'long', type: 'verse', startMeasure: 0, length: 63 }],
    measures: Array.from({ length: 63 }, (_, i) => measure(i, [null, null])) };
  assert.match(planBlocker(tooLong) ?? '', /already uses 63 bars/);
});

test('a fragmented stream yields the typed events in run order and resolves with the complete trace', async (t) => {
  const final = trace({
    sections: [0, 1, 2].map((i) => sectionResult(i, 'chorus', 8 + 8 * i, 8)) as unknown as PlanSectionResult[],
  });
  const wire =
    frame('progress', { label: 'Planning the song form from the sketch…', t_ms: 1 }) +
    frame('plan', { attempt: 1, source: 'rules', plan: VALID_PLAN, t_ms: 2 }) +
    frame('validate', { attempt: 1, ok: true, errors: [], t_ms: 3 }) +
    frame('generate_section', sectionResult(0, 'chorus', 8, 8)) +
    frame('generate_section', sectionResult(1, 'verse', 16, 8, { cadence: 'half', rulesSlots: [15] })) +
    frame('generate_section', sectionResult(2, 'outro', 24, 8, { cadence: 'plagal' })) +
    frame('heartbeat', { t_ms: 9001 }) +
    frame('complete', final);
  const fetchMock = t.mock.method(globalThis, 'fetch', async (input: RequestInfo | URL, init?: RequestInit) => {
    assert.equal(input, '/api/plan/song/stream');
    assert.equal(init?.method, 'POST');
    assert.equal(JSON.parse(String(init?.body)).targetBars, 32);
    const headers = new Headers(init?.headers);
    assert.equal(headers.get('Content-Type'), 'application/json');
    assert.equal(headers.get('X-TheArtist-Session'), null);  // no session header goes with a form run
    return streamResponse([wire.slice(0, 23), wire.slice(23, 140), wire.slice(140, 700), wire.slice(700)]);
  });
  const events: PlanEvent[] = [];
  const result = await runPlan(buildPlanRequest(song, options), { onEvent: (event) => events.push(event) });
  assert.equal(fetchMock.mock.callCount(), 1);
  assert.deepEqual(events.map((e) => e.event),
    ['progress', 'plan', 'validate', 'generate_section', 'generate_section', 'generate_section', 'complete']);
  const [progress, plan, validate, first, second] = events;
  assert.deepEqual(progress, { event: 'progress', data: { label: 'Planning the song form from the sketch…' } });
  assert.equal(plan.event, 'plan');
  if (plan.event !== 'plan') return;
  assert.equal(plan.data.source, 'rules');
  assert.equal(plan.data.plan.sections[0].type, 'chorus');
  assert.equal(validate.event, 'validate');
  if (validate.event !== 'validate') return;
  assert.equal(validate.data.ok, true);
  assert.equal(first.event, 'generate_section');
  if (first.event !== 'generate_section') return;
  assert.equal(first.data.index, 0);
  assert.deepEqual(first.data.rulesSlots, [14, 15]);
  assert.deepEqual(first.data.suggestions[0].chords['8'], ['Cmaj7', 'Am7']);
  if (second.event !== 'generate_section') return;
  assert.deepEqual(second.data.rulesSlots, [15]);
  assert.deepEqual(result, final);
  assert.equal(result.status, 'ok');
  assert.equal(result.sections.length, 3);
});

test('a run that failed on the server resolves with its error trace, so the failure stays visible', async (t) => {
  const issues = [
    { code: 'total_bars', section: null, message: 'the sketch\'s 8 bars plus the planned 20 bars make 28, not targetBars 32' },
    { code: 'final_cadence', section: 1, message: 'the last section ends with half; the song must land on authentic or plagal' },
  ];
  const failed = trace({ status: 'error', validation: [{ attempt: 1, ok: false, errors: issues }],
    error: { kind: 'invalid', message: 'the genre-default form failed validation' } });
  t.mock.method(globalThis, 'fetch', async () => streamResponse([
    frame('plan', { attempt: 1, source: 'rules', plan: VALID_PLAN, t_ms: 2 }),
    frame('validate', { attempt: 1, ok: false, errors: issues, t_ms: 3 }),
    frame('error', failed),
  ]));
  const events: PlanEvent[] = [];
  const result = await runPlan(buildPlanRequest(song, options), { onEvent: (event) => events.push(event) });
  assert.deepEqual(events.map((event) => event.event), ['plan', 'validate', 'error']);
  const validate = events[1];
  assert.equal(validate.event, 'validate');
  if (validate.event !== 'validate') return;
  assert.equal(validate.data.ok, false);
  assert.deepEqual(validate.data.errors.map((issue) => issue.code), ['total_bars', 'final_cadence']);
  assert.equal(result.status, 'error');
  assert.equal(result.error?.kind, 'invalid');
  assert.deepEqual(result, failed);
});

test('an error frame without a trace rejects with the server message', async (t) => {
  t.mock.method(globalThis, 'fetch', async () => streamResponse([frame('error', { detail: 'Song planning failed' })]));
  await assert.rejects(runPlan(buildPlanRequest(song, options)), /Song planning failed/);
});

test('HTTP errors keep the backend validation message', async (t) => {
  t.mock.method(globalThis, 'fetch', async () => new Response(JSON.stringify({
    detail: [{ loc: ['body', 'targetBars'], msg: 'Input should be 16, 32, 48 or 64' }],
  }), { status: 422 }));
  await assert.rejects(runPlan(buildPlanRequest(song, options)), /targetBars: Input should be 16, 32, 48 or 64/);
});

test('HTTP failures without JSON show their status', async (t) => {
  t.mock.method(globalThis, 'fetch', async () => new Response('Backend unavailable', { status: 503 }));
  await assert.rejects(runPlan(buildPlanRequest(song, options)), /Server error \(HTTP 503\)/);
});

test('a disconnected backend rejects as unreachable', async (t) => {
  t.mock.method(globalThis, 'fetch', async () => { throw new TypeError('Failed to fetch'); });
  await assert.rejects(runPlan(buildPlanRequest(song, options)), /Could not reach the server/);
});

test('a stream that ends after progress rejects without a trace', async (t) => {
  t.mock.method(globalThis, 'fetch', async () => streamResponse([
    frame('plan', { attempt: 1, source: 'rules', plan: VALID_PLAN, t_ms: 2 }), frame('validate', { attempt: 1, ok: true, errors: [], t_ms: 3 }),
  ]));
  await assert.rejects(runPlan(buildPlanRequest(song, options)), /ended without a result/);
});

test('an invalid section result or terminal frame rejects instead of looking like a plan', async (t) => {
  for (const [event, payload] of [
    ['generate_section', { index: 0, suggestions: 'none' }],
    ['complete', { status: 'ok' }],
    ['plan', { attempt: 1, source: 'rules' }],
    ['validate', { attempt: 1, ok: 'yes', errors: [] }],
  ] as const) {
    const mock = t.mock.method(globalThis, 'fetch', async () => streamResponse([frame(event, payload)]));
    await assert.rejects(runPlan(buildPlanRequest(song, options)), /invalid/);
    mock.mock.restore();
  }
});

test('aborting the signal rejects with the AbortError itself', async (t) => {
  const controller = new AbortController();
  t.mock.method(globalThis, 'fetch', (_input: RequestInfo | URL, init?: RequestInit) => new Promise<Response>((_resolve, reject) => {
    init?.signal?.addEventListener('abort', () => reject(init.signal?.reason), { once: true });
  }));
  const pending = runPlan(buildPlanRequest(song, options), { signal: controller.signal });
  controller.abort();
  await assert.rejects(pending, (err: unknown) => err instanceof Error && err.name === 'AbortError');
});
