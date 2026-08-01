import { test } from 'node:test';
import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';

class FakeInstrument {
  name: string;
  destination: unknown;
  load: Promise<void>;
  resolve!: () => void;
  reject!: (error: Error) => void;
  starts: unknown[] = [];
  constructor(_context: unknown, options: { instrument?: string; destination: unknown }) {
    this.name = options.instrument ?? 'curated';
    this.destination = options.destination;
    this.load = new Promise((resolve, reject) => { this.resolve = resolve; this.reject = reject; });
    instances.push(this);
  }
  start(args: unknown) { this.starts.push(args); }
}
const instances: FakeInstrument[] = [];
declare global { var InstrumentTestFake: typeof FakeInstrument; }
globalThis.InstrumentTestFake = FakeInstrument;
const ENGINE_URL = new URL('./', import.meta.url).href;
const stubs: Record<string, string> = {
  tone: 'export const getContext = () => ({ rawContext: {} }); export const start = async () => {}; export const getTransport = () => ({ start() {} });',
  smplr: 'export const Soundfont = globalThis.InstrumentTestFake; export const SplendidGrandPiano = Soundfont; export const ElectricPiano = Soundfont;',
  './mixer': 'export const getInstrumentOutput = layer => layer; export const setMixerVolumes = () => {};',
  './voicingEngine': 'export const guitarVoicingNotes = () => []; export const resolvePianoVoicing = () => [];',
  './drums': 'export const preloadDrums = async () => {}; export const scheduleDrumMeasure = () => {}; export const clearDrumSchedule = () => {}; export const setCurrentDrumPattern = () => {}; export const setDrumVolume = () => {}; export const swingDelay = () => 0; export const DRUM_STEPS_PER_BEAT = 4;',
  './arrangement': 'export const scheduleBassSlot = () => {}; export const setBassEnabled = () => {};',
  './playSettings': 'export const HARMONY_GRID = []; export const autoRhythmPreset = () => []; export const genreHarmonyInstrument = () => null; export const harmonyStrikesForChords = () => []; export const secondsPerMeterBeat = () => 1;',
};
registerHooks({
  resolve(specifier, context, nextResolve) {
    if (context.parentURL?.startsWith(ENGINE_URL)) {
      if (stubs[specifier]) return { url: `data:text/javascript,${encodeURIComponent(stubs[specifier])}`, shortCircuit: true };
      if (specifier === './instruments') specifier += '.ts';
    }
    return nextResolve(specifier, context);
  },
});
const { getInstrument, isInstrumentReady, preloadInstruments, prefetchHarmonyInstruments, playChordOnInstrument } = await import('./instruments.ts');
const { prepareHarmonyFamily, prepareHarmonyInstrument, startPlayback } = await import('./playback.ts');
const flush = async () => { for (let i = 0; i < 6; i++) await Promise.resolve(); };

test('intent waits for the first Play gesture, then prepares without triggering any notes', async () => {
  prepareHarmonyInstrument('before-play');
  prepareHarmonyFamily(['before-family']);
  assert.equal(instances.length, 0);
  await startPlayback();
  prepareHarmonyInstrument('hovered');
  assert.equal(instances.length, 1);
  assert.equal(instances[0].name, 'hovered');
  assert.deepEqual(instances[0].starts, []);
  instances[0].resolve();
  await flush();
  assert.equal(isInstrumentReady('hovered'), true);
});

test('preload and selection share one load, release their timer, and play the prepared instrument', async (t) => {
  const clear = t.mock.method(globalThis, 'clearTimeout');
  const waiting = preloadInstruments(['hovered', 'hovered', null]);
  const same = getInstrument('hovered');
  await waiting;
  assert.equal(instances.filter((inst) => inst.name === 'hovered').length, 1);
  assert.equal(same, getInstrument('hovered'));
  assert.equal(clear.mock.calls.length, 1);
  assert.equal(playChordOnInstrument('hovered', [60, 64, 67], 2, 1, 0.5), true);
  assert.deepEqual(instances[0].starts, [60, 64, 67].map((note) => ({ note, velocity: 64, time: 2, duration: 1 })));
  getInstrument('hovered', 'bass');
  assert.equal(instances.at(-1)!.destination, 'bass');
  assert.equal(isInstrumentReady('hovered', 'bass'), false);
  instances.at(-1)!.resolve();
});

test('background loads are bounded, stale families drop out, and a selection bypasses the queue', async () => {
  prefetchHarmonyInstruments(['family-a', 'family-b', 'stale-c', 'family-a']);
  assert.deepEqual(instances.slice(-2).map((inst) => inst.name), ['family-a', 'family-b']);
  prefetchHarmonyInstruments(['new-family']);
  const selected = preloadInstruments(['selected-now']);
  assert.equal(instances.at(-1)!.name, 'selected-now');
  instances.at(-1)!.resolve();
  await selected;
  instances.find((inst) => inst.name === 'family-a')!.resolve();
  await flush();
  assert.equal(instances.at(-1)!.name, 'new-family');
  assert.equal(instances.some((inst) => inst.name === 'stale-c'), false);
  instances.find((inst) => inst.name === 'family-b')!.resolve();
  instances.find((inst) => inst.name === 'new-family')!.resolve();
  await flush();
});

test('a stalled load rejects before playback and can become ready later', async () => {
  await assert.rejects(preloadInstruments(['slow'], 1), /still loading/);
  assert.equal(isInstrumentReady('slow'), false);
  assert.equal(playChordOnInstrument('slow', [60], 0, 1), false);
  instances.find((inst) => inst.name === 'slow')!.resolve();
  await flush();
  assert.equal(isInstrumentReady('slow'), true);
});

test('failed background decoding frees the next slot and foreground retries the instrument', async (t) => {
  t.mock.method(console, 'warn', () => {});
  prefetchHarmonyInstruments(['failed', 'finishes', 'next']);
  instances.find((inst) => inst.name === 'failed')!.reject(new Error('network'));
  await flush();
  assert.equal(instances.at(-1)!.name, 'next');
  const retry = preloadInstruments(['failed']);
  assert.equal(isInstrumentReady('failed'), false);
  instances.at(-1)!.reject(new Error('decode error'));
  await assert.rejects(retry, /decode error/);
  instances.find((inst) => inst.name === 'finishes')!.resolve();
  instances.find((inst) => inst.name === 'next')!.resolve();
  await flush();
});
