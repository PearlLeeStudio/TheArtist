import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readSSEFrames } from './sseStream.ts';

const encode = (value: string) => new TextEncoder().encode(value);
const frame = 'event: result\ndata: {"label":"후렴 ♭"}\n\n';

test('fragmented UTF-8 frames retain their text and release the reader at EOF', async () => {
  const response = new Response(new ReadableStream({
    start(controller) {
      for (const byte of encode(frame + frame)) controller.enqueue(new Uint8Array([byte]));
      controller.close();
    },
  }));
  const frames = [];
  for await (const value of readSSEFrames(response)) frames.push(value);
  assert.deepEqual(frames, [
    { event: 'result', data: { label: '후렴 ♭' } },
    { event: 'result', data: { label: '후렴 ♭' } },
  ]);
  assert.equal(response.body?.locked, false);
});

test('an incomplete frame is not treated as a result', async () => {
  const response = new Response(frame.slice(0, -1));
  const frames = [];
  for await (const value of readSSEFrames(response)) frames.push(value);
  assert.deepEqual(frames, []);
  assert.equal(response.body?.locked, false);
});

test('stopping after a frame cancels the open response and releases its lock', async () => {
  let cancelled = false;
  const response = new Response(new ReadableStream({
    start(controller) { controller.enqueue(encode(frame)); },
    cancel() { cancelled = true; },
  }));
  for await (const value of readSSEFrames(response)) {
    assert.equal(value.event, 'result');
    break;
  }
  assert.equal(cancelled, true);
  assert.equal(response.body?.locked, false);
});

test('a consumer failure cancels the response without replacing the original error', async () => {
  const failure = new Error('The server reported a generation error');
  let cancelled = false;
  const response = new Response(new ReadableStream({
    start(controller) { controller.enqueue(encode(frame)); },
    cancel() { cancelled = true; throw new Error('cleanup failed'); },
  }));
  await assert.rejects(async () => {
    for await (const value of readSSEFrames(response)) {
      assert.equal(value.event, 'result');
      throw failure;
    }
  }, (error) => error === failure);
  assert.equal(cancelled, true);
  assert.equal(response.body?.locked, false);
});

test('malformed JSON cancels the open response instead of leaving it locked', async () => {
  let cancelled = false;
  const response = new Response(new ReadableStream({
    start(controller) { controller.enqueue(encode('event: result\ndata: {broken}\n\n')); },
    cancel() { cancelled = true; },
  }));
  await assert.rejects(async () => {
    for await (const value of readSSEFrames(response)) assert.fail(JSON.stringify(value));
  }, SyntaxError);
  assert.equal(cancelled, true);
  assert.equal(response.body?.locked, false);
});

test('a transport abort preserves its error and releases the reader', async () => {
  const failure = new DOMException('Aborted', 'AbortError');
  const response = new Response(new ReadableStream({
    pull(controller) { controller.error(failure); },
  }));
  await assert.rejects(async () => {
    for await (const value of readSSEFrames(response)) assert.fail(JSON.stringify(value));
  }, (error) => error === failure);
  assert.equal(response.body?.locked, false);
});

test('a response without a body fails explicitly', async () => {
  await assert.rejects(async () => {
    for await (const value of readSSEFrames(new Response(null))) assert.fail(JSON.stringify(value));
  }, /SSE response has no body/);
});
