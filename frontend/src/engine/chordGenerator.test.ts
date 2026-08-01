import { test } from 'node:test';
import assert from 'node:assert/strict';
import { generateChords } from './chordGenerator.ts';
import type { Song, SlotAddress } from '../models/types.ts';
import type { GeneratePhase } from './chordGenerator.ts';

const song: Song = {
  id: 'existing-score', title: 'Existing work', key: 'C major', bpm: 80.4,
  genre: 'pop', timeSignature: [4, 4],
  measures: [{ id: 'bar-one', index: 0, chords: [
    { root: 'C', quality: 'maj7', extensions: [], romanNumeral: 'Imaj7', voicing: [60, 64, 67, 71] },
    { root: 'G', quality: '7', extensions: [], romanNumeral: 'V7', voicing: [55, 59, 62, 65] },
  ] }],
};
const slots: SlotAddress[] = [{ measureIndex: 0, slotIndex: 0 }];
const suggestions = [{ label: 'A', chords: { 0: ['Dm7', 'G7'] } }];
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

test('partial generation reads fragmented SSE progress and the actual server result without changing the score', async (t) => {
  const before = structuredClone(song);
  const phases: GeneratePhase[] = [];
  const wire = frame('composing', { label: 'Composing', step: 1, total: 3, creativity: 'balanced' }) +
    frame('result', { suggestions, source: 'ai' });
  t.mock.method(globalThis, 'fetch', async (input, init) => {
    assert.equal(input, '/api/generate/stream');
    const body = JSON.parse(String(init?.body));
    assert.deepEqual(body.selectedMeasures, [0]);
    assert.deepEqual(body.context[0].chords, ['Cmaj7', 'G7']);
    assert.equal(body.bpm, 80);
    assert.equal(body.sectionType, 'verse');
    return streamResponse([wire.slice(0, 19), wire.slice(19, 53), wire.slice(53)]);
  });
  assert.deepEqual(await generateChords(song, slots, 'verse', 'ft_f1', (phase) => phases.push(phase)), { suggestions, source: 'ai' });
  assert.equal(phases[0].name, 'composing');
  assert.deepEqual(song, before);
});

test('explicit backend rule sources remain identified as rules', async (t) => {
  for (const source of ['rules', 'ai+rules'] as const) {
    const mock = t.mock.method(globalThis, 'fetch', async () => streamResponse([frame('result', { suggestions, source })]));
    assert.deepEqual(await generateChords(song, slots, null), { suggestions, source });
    mock.mock.restore();
  }
});

test('HTTP errors retain the backend validation message and return no candidates', async (t) => {
  t.mock.method(globalThis, 'fetch', async () => new Response(JSON.stringify({
    detail: [{ loc: ['body', 'selectedMeasures'], msg: 'Select at least one bar' }],
  }), { status: 422 }));
  await assert.rejects(generateChords(song, slots, null), /selectedMeasures: Select at least one bar/);
});

test('HTTP failures without JSON show their status', async (t) => {
  t.mock.method(globalThis, 'fetch', async () => new Response('Backend unavailable', { status: 503 }));
  await assert.rejects(generateChords(song, slots, null), /Server error \(HTTP 503\)/);
});

test('a disconnected backend rejects instead of manufacturing offline chords and preserves the input', async (t) => {
  const before = structuredClone(song);
  t.mock.method(globalThis, 'fetch', async () => { throw new TypeError('Failed to fetch'); });
  await assert.rejects(generateChords(song, slots, null), /Could not reach the server/);
  assert.deepEqual(song, before);
});

test('SSE error events reject even when preceded by a result', async (t) => {
  t.mock.method(globalThis, 'fetch', async () => streamResponse([
    frame('result', { suggestions, source: 'ai' }), frame('error', { detail: 'Generation was interrupted' }),
  ]));
  await assert.rejects(generateChords(song, slots, null), /Generation was interrupted/);
});

test('a stream that ends after progress rejects without candidates', async (t) => {
  t.mock.method(globalThis, 'fetch', async () => streamResponse([frame('ranking', { label: 'Ranking' })]));
  await assert.rejects(generateChords(song, slots, null), /ended without a result/);
});

test('an incomplete final SSE frame rejects without candidates', async (t) => {
  t.mock.method(globalThis, 'fetch', async () => streamResponse([
    frame('composing', { label: 'Composing' }), 'event: result\ndata: {"suggestions":[',
  ]));
  await assert.rejects(generateChords(song, slots, null), /ended without a result/);
});

test('malformed SSE JSON rejects without candidates', async (t) => {
  t.mock.method(globalThis, 'fetch', async () => streamResponse(['event: result\ndata: {bad JSON}\n\n']));
  await assert.rejects(generateChords(song, slots, null), SyntaxError);
});

test('a transport interruption while reading SSE rejects without candidates', async (t) => {
  t.mock.method(globalThis, 'fetch', async () => new Response(new ReadableStream({
    start(controller) { controller.error(new Error('Connection lost during generation')); },
  })));
  await assert.rejects(generateChords(song, slots, null), /Connection lost during generation/);
});

test('missing, empty and malformed result payloads reject instead of appearing successful', async (t) => {
  for (const payload of [
    null, {}, { suggestions: [] }, { suggestions: [null] },
    { suggestions: [{ label: 'A', chords: { 0: ['C'] } }] },
    { suggestions: [{ label: 'A', chords: { 0: ['C', 7] } }] },
    { suggestions: [{ label: 'A', chords: {} }] },
  ]) {
    const mock = t.mock.method(globalThis, 'fetch', async () => streamResponse([frame('result', payload)]));
    await assert.rejects(generateChords(song, slots, null), /invalid generation result/);
    mock.mock.restore();
  }
});
