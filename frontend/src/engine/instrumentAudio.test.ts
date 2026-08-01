import { test } from 'node:test';
import assert from 'node:assert/strict';
import { checkedInstrumentContext, instrumentSampleUrl, instrumentStorage } from './instrumentAudio.ts';

test('CP80 uses pinned lossless source with escaped sharp notes preserved', () => {
  const source = instrumentSampleUrl('https://smpldsnds.github.io/sfzinstruments-greg-sullivan-e-pianos/cp80/samples/027-D%231-FF.m4a');
  assert.match(source, /8c3e581acda3594b553948ff0222d4f84a698376\/CP80\/Samples\/027-D%231-FF.flac$/);
  assert.equal(instrumentSampleUrl('/drums/kick.flac'), '/drums/kick.flac');
});

test('partial decoder failure cannot be reported as a ready instrument', async () => {
  const raw = { decodeAudioData: async () => { throw new Error('Decoding failed'); }, createGain() { assert.equal(this, raw); } };
  const checked = checkedInstrumentContext(raw as unknown as AudioContext);
  checked.context.createGain();
  await assert.rejects(checked.context.decodeAudioData(new ArrayBuffer(1)), /Decoding failed/);
  assert.throws(() => checked.assertReady(), /could not be decoded/);
  assert.doesNotThrow(() => checkedInstrumentContext(raw as unknown as AudioContext).assertReady());
});

test('HTTP sample failure is rejected before smplr can swallow it', async (t) => {
  t.mock.method(globalThis, 'fetch', async () => new Response('', { status: 404 }));
  await assert.rejects(instrumentStorage.fetch('/missing.flac'), /404/);
});
