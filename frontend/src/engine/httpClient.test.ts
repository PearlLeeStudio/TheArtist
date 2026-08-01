import { test } from 'node:test';
import assert from 'node:assert/strict';
import { postJson } from './httpClient.ts';

test('JSON POST sends only the JSON content type, keeps cancellation and leaves the response for the caller', async (t) => {
  const controller = new AbortController();
  const response = new Response('event: complete\ndata: {}\n\n');
  t.mock.method(globalThis, 'fetch', async (url, init) => {
    assert.equal(url, '/api/plan/song/stream');
    assert.equal(init?.method, 'POST');
    assert.equal(init?.signal, controller.signal);
    assert.deepEqual([...new Headers(init?.headers).entries()], [['content-type', 'application/json']]);
    assert.deepEqual(JSON.parse(String(init?.body)), { targetBars: 32 });
    return response;
  });
  assert.equal(await postJson('/api/plan/song/stream', { targetBars: 32 }, { signal: controller.signal }), response);
  assert.equal(response.bodyUsed, false);
});

test('network errors retain their cause and the endpoint-specific message', async (t) => {
  const failure = new TypeError('connection refused');
  t.mock.method(globalThis, 'fetch', async () => { throw failure; });
  await assert.rejects(postJson('/api/generate/stream', {}), (error: Error) =>
    /Could not reach the server/.test(error.message) && error.cause === failure);
  await assert.rejects(postJson('/api/generate/section/stream', {}, { unreachableMessage: 'Section unavailable' }),
    (error: Error) => error.message === 'Section unavailable' && error.cause === failure);
});

test('cancellation is not reclassified as a connectivity failure', async (t) => {
  const controller = new AbortController();
  const failure = new DOMException('Aborted', 'AbortError');
  t.mock.method(globalThis, 'fetch', async () => { controller.abort(); throw failure; });
  await assert.rejects(postJson('/api/plan/song/stream', {}, { signal: controller.signal }),
    (error) => error === failure);
});

test('HTTP failures preserve validation locations, alternate messages, and non-JSON status', async (t) => {
  const cases: Array<[number, unknown, string]> = [
    [422, { detail: [{ loc: ['body', 'sections', 0, 'bars'], msg: 'Must be positive' }] }, 'sections.0.bars: Must be positive'],
    [429, { detail: '  ', error: 'Too many requests' }, 'Too many requests'],
    [400, null, 'Request failed (HTTP 400)'],
    [503, '<html>Unavailable</html>', 'Server error (HTTP 503)'],
  ];
  for (const [status, body, message] of cases) {
    const mock = t.mock.method(globalThis, 'fetch', async () =>
      new Response(typeof body === 'string' ? body : JSON.stringify(body), { status }));
    await assert.rejects(postJson('/api/generate/stream', {}), { message });
    mock.mock.restore();
  }
});
