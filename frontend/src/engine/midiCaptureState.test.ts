import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createMidiCaptureState } from './midiCaptureState.ts';
import type { Chord } from '../models/types';

const chord: Chord = { root: 'C', quality: 'maj', voicing: [60, 64, 67], romanNumeral: 'I', extensions: [] };
const seventh: Chord = { ...chord, quality: 'maj7', voicing: [60, 64, 67, 71] };

function fixture() {
  let now = 0;
  let nextId = 0;
  const timers = new Map<number, () => void>();
  const commits: Chord[] = [];
  let live: Chord[] = [];
  const capture = createMidiCaptureState({
    clock: {
      now: () => now,
      setInterval: (callback) => { const id = ++nextId; timers.set(id, callback); return id; },
      clearInterval: (id) => { timers.delete(id); },
    },
    onCommit: (value) => { commits.push(value); return true; },
    onCandidates: (value) => { live = value; },
  });
  return {
    capture, commits, timers, live: () => live,
    advance(milliseconds: number) { now += milliseconds; [...timers.values()].forEach((callback) => callback()); },
    play(value = chord) { capture.notes(value.voicing); capture.detected([value]); },
  };
}

test('monitoring publishes notes and chords without a target, timer or commit', () => {
  const f = fixture();
  f.capture.setContext(false, null);
  f.play();
  f.advance(10_000);
  assert.deepEqual(f.live(), [chord]);
  assert.equal(f.timers.size, 0);
  f.capture.notes([]);
  f.capture.confirm();
  assert.equal(f.capture.getSnapshot().confirming, null);
  assert.deepEqual(f.live(), []);
  assert.deepEqual(f.commits, []);
});

test('leaving the input tab cancels a running hold and never commits in Play', () => {
  const f = fixture();
  f.capture.setContext(true, '0:0');
  f.play();
  f.advance(2000);
  assert.ok(f.capture.getSnapshot().holdProgress > 0);
  f.capture.setContext(false, '0:0');
  assert.equal(f.timers.size, 0);
  assert.equal(f.capture.getSnapshot().holdProgress, 0);
  f.advance(5000);
  f.capture.notes([]);
  f.capture.confirm();
  assert.deepEqual(f.commits, []);
  assert.equal(f.capture.getSnapshot().confirming, null);
});

test('returning to capture requires a fresh press rather than committing held monitor notes', () => {
  const f = fixture();
  f.play();
  f.capture.setContext(true, '0:0');
  f.capture.detected([seventh]);
  f.advance(5000);
  assert.equal(f.capture.getSnapshot().awaitingRelease, true);
  assert.deepEqual(f.commits, []);
  f.capture.notes([]);
  f.play();
  f.advance(3000);
  assert.deepEqual(f.commits, [chord]);
});

test('a hold commits once and partial key releases cannot start a second commit', () => {
  const f = fixture();
  f.capture.setContext(true, '0:0');
  f.play(seventh);
  f.advance(3000);
  f.capture.notes(chord.voicing);
  f.capture.detected([chord]);
  f.advance(5000);
  f.capture.notes([]);
  assert.deepEqual(f.commits, [seventh]);
  assert.equal(f.capture.getSnapshot().confirming, null);
});

test('early release reviews the largest voicing, but hidden confirmation cannot write it', () => {
  const f = fixture();
  f.capture.setContext(true, '0:0');
  f.play(seventh);
  f.capture.notes(chord.voicing);
  f.capture.detected([chord]);
  f.capture.notes([]);
  assert.equal(f.capture.getSnapshot().confirming?.candidates[0], seventh);
  f.capture.setContext(false, '0:0');
  f.capture.confirm();
  assert.deepEqual(f.commits, []);
  assert.equal(f.capture.getSnapshot().confirming, null);
});

test('selection changes and song replacement cancel stale capture state', () => {
  const f = fixture();
  f.capture.setContext(true, '0:0');
  f.play();
  f.advance(2000);
  f.capture.setContext(true, '1:0');
  f.advance(3000);
  assert.deepEqual(f.commits, []);
  assert.equal(f.timers.size, 0);
  f.capture.reset();
  assert.deepEqual(f.live(), []);
  assert.equal(f.capture.getSnapshot().awaitingRelease, false);
  assert.equal(f.capture.getSnapshot().confirming, null);
});

test('one selected target is needed for capture, but losing it keeps live monitoring', () => {
  const f = fixture();
  f.capture.setContext(true, '0:0');
  f.play();
  f.capture.setContext(true, null);
  f.capture.detected([seventh]);
  f.advance(3000);
  assert.deepEqual(f.live(), [seventh]);
  assert.deepEqual(f.commits, []);
  assert.equal(f.timers.size, 0);
});

test('disposal cancels all scheduled capture work', () => {
  const f = fixture();
  f.capture.setContext(true, '0:0');
  f.play();
  f.capture.dispose();
  assert.equal(f.timers.size, 0);
  f.advance(5000);
  assert.deepEqual(f.commits, []);
});

test('auto-advance cannot rearm the still-held chord through a synchronous target update', () => {
  let now = 0;
  let tick: (() => void) | null = null;
  let commits = 0;
  const capture = createMidiCaptureState({
    clock: {
      now: () => now,
      setInterval: (callback) => { tick = callback; return 1; },
      clearInterval: () => { tick = null; },
    },
    onCandidates: () => {},
    onCommit: () => { commits++; capture.setContext(true, '0:1'); return true; },
  });
  capture.setContext(true, '0:0');
  capture.notes(seventh.voicing);
  capture.detected([seventh]);
  now = 3000;
  const fire = tick as (() => void) | null;
  fire?.();
  capture.notes(chord.voicing);
  capture.detected([chord]);
  assert.equal(commits, 1);
  assert.equal(tick, null);
  capture.notes([]);
  assert.equal(capture.getSnapshot().confirming, null);
});

test('automatic input commits the chord still held, not a larger earlier voicing', () => {
  const f = fixture();
  f.capture.setContext(true, '0:0');
  f.play(seventh);
  f.advance(500);
  f.capture.notes(chord.voicing);
  f.capture.detected([chord]);
  f.advance(3000);
  assert.deepEqual(f.commits, [chord]);
});

test('dropping below three held notes cancels automatic input but retains early-release review', () => {
  const f = fixture();
  f.capture.setContext(true, '0:0');
  f.play();
  f.advance(2000);
  f.capture.notes([60, 64]);
  assert.equal(f.timers.size, 0);
  assert.equal(f.capture.getSnapshot().holdProgress, 0);
  assert.deepEqual(f.live(), []);
  f.advance(5000);
  assert.deepEqual(f.commits, []);
  f.capture.notes([]);
  assert.deepEqual(f.capture.getSnapshot().confirming?.candidates, [chord]);
});
