import { test } from 'node:test';
import assert from 'node:assert/strict';
import { analyzeScoreFile, ScoreAnalysisError, recognitionErrorMessage } from './scoreImportClient.ts';

const frame = (event: string, data: unknown) => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
function stream(parts: string[], init: ResponseInit = {}): Response {
  const encoder = new TextEncoder();
  return new Response(new ReadableStream({
    start(controller) {
      for (const part of parts) controller.enqueue(encoder.encode(part));
      controller.close();
    },
  }), { status: 200, headers: { 'Content-Type': 'text/event-stream' }, ...init });
}
const reading = {
  title: 'Sketch', key: 'C major', timeSignature: [4, 4], warnings: [], sections: [], model: null,
  analysisSource: 'local', mode: 'infer_harmony',
  measures: [{ chords: ['C'], sourceLabel: 'Bar 1', needsReview: false, note: '', chordBasis: 'printed', confidence: 'high', analysisSource: 'local' }],
};
const file = new File([new Uint8Array([0x25, 0x50, 0x44, 0x46, 0x2d])], 'score.pdf', { type: 'application/pdf' });

test('the client sends the file once, relays reader stages and returns the validated reading', async () => {
  const calls: Array<{ url: string; body: Record<string, unknown>; headers: Record<string, string> }> = [];
  const statuses: string[] = [];
  const fetcher = (async (url: string | URL | Request, init?: RequestInit) => {
    assert.equal(init?.cache, 'no-store');
    calls.push({ url: String(url), body: JSON.parse(String(init?.body)), headers: init?.headers as Record<string, string> });
    return stream([frame('progress', { label: 'Reading the score…' }), frame('result', reading)]);
  }) as typeof fetch;
  const result = await analyzeScoreFile(file, 'application/pdf', 'infer_harmony', { fetcher, onStatus: (label) => statuses.push(label) });
  assert.equal(result.measures.length, 1);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, '/api/import/score/stream');
  // The server refuses unknown fields: the body is the file and the reading mode only.
  assert.deepEqual(calls[0].body, { filename: 'score.pdf', mimeType: 'application/pdf', dataBase64: btoa('%PDF-'), mode: 'infer_harmony' });
  assert.deepEqual(calls[0].headers, { 'Content-Type': 'application/json' });
  assert.deepEqual(statuses, ['Uploading the score for analysis…', 'Reading the score…']);
});

test('server errors keep their status, and a stream without a result is an error, not a score', async () => {
  const errorFrame = (async () => stream([frame('progress', { label: 'x' }), frame('error', { detail: 'Not a vector PDF', status: 422 })])) as typeof fetch;
  await assert.rejects(analyzeScoreFile(file, 'application/pdf', 'infer_harmony', { fetcher: errorFrame }),
    (error: unknown) => error instanceof ScoreAnalysisError && error.status === 422 && /Not a vector PDF/.test(error.message));
  const refused = (async () => Response.json({ detail: 'The score reader is unavailable.' }, { status: 503 })) as typeof fetch;
  await assert.rejects(analyzeScoreFile(file, 'application/pdf', 'infer_harmony', { fetcher: refused }),
    (error: unknown) => error instanceof ScoreAnalysisError && error.status === 503 && error.message === 'The score reader is unavailable.');
  const silent = (async () => stream([frame('progress', { label: 'x' })])) as typeof fetch;
  await assert.rejects(analyzeScoreFile(file, 'application/pdf', 'infer_harmony', { fetcher: silent }), /ended before analysis finished/);
  assert.equal(recognitionErrorMessage({ detail: [{ msg: 'a' }, { msg: 'b' }] }, 422), 'a; b');
  assert.equal(recognitionErrorMessage(null, 500), 'Score recognition failed (500). Please try again.');
});
