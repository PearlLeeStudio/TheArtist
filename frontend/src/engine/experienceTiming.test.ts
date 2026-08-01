import { test } from 'node:test';
import assert from 'node:assert/strict';
import { beginTiming, clearTimings, setTimingEnabled, timeOperation, timingSamples, timingSummary } from './experienceTiming.ts';

test('timing is opt-in, bounded, idempotent, and returned records cannot mutate it', () => {
  clearTimings(); setTimingEnabled(false);
  beginTiming('import.review')();
  assert.deepEqual(timingSamples(), []);
  setTimingEnabled(true);
  const finish = beginTiming('import.review', 'local');
  finish(); finish('error');
  assert.equal(timingSamples().length, 1);
  timingSamples()[0].outcome = 'error';
  assert.equal(timingSamples()[0].outcome, 'ok');
  for (let i = 0; i < 210; i++) beginTiming('instrument.ready', 'warm')();
  assert.equal(timingSamples().length, 200);
  setTimingEnabled(false);
});

test('failed and cancelled calls stay out of successful latency percentiles and still reject', async () => {
  clearTimings(); setTimingEnabled(true);
  assert.equal(await timeOperation('generate.result', 'default', async () => 42), 42);
  const error = new Error('failed');
  await assert.rejects(timeOperation('generate.result', 'default', async () => { throw error; }), error);
  await assert.rejects(timeOperation('generate.result', 'default', async () => { throw new DOMException('cancel', 'AbortError'); }));
  const summary = timingSummary()[0];
  assert.equal(summary.count, 1);
  assert.equal(summary.failures, 2);
  assert.equal(summary.p50Ms, summary.p95Ms);
  assert.deepEqual(timingSamples().map((sample) => sample.outcome), ['ok', 'error', 'cancelled']);
  setTimingEnabled(false);
});
