import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createAudioMixer } from './audioMixer.ts';

/** A connected source remains connected while gains change, like a sustained
 * audio voice. This tests routing and live levels, not browser audio fidelity. */
class FakeNode {
  gain = { value: 1 };
  outputs: FakeNode[] = [];
  connect(destination: FakeNode): void { this.outputs.push(destination); }
  amplitude(): number {
    return this.gain.value * (this.outputs.length === 0
      ? 1
      : this.outputs.reduce((sum, destination) => sum + destination.amplitude(), 0));
  }
}

function fixture() {
  let opened = 0;
  const destination = new FakeNode();
  const nodes: FakeNode[] = [];
  const mixer = createAudioMixer(() => {
    opened++;
    return {
      destination,
      createGain: () => {
        const node = new FakeNode();
        nodes.push(node);
        return node;
      },
    } as unknown as BaseAudioContext;
  });
  const input = (bus: 'master' | 'harmony' | 'bass') => mixer.input(bus) as unknown as FakeNode;
  return { mixer, input, destination, nodes, opened: () => opened };
}

test('restored mixer levels do not open audio, and the first source uses them', () => {
  const f = fixture();
  f.mixer.setVolumes({ masterVolume: 0.5, harmonyVolume: 0.4, bassVolume: 0.8 });
  assert.equal(f.opened(), 0);
  assert.equal(f.input('harmony').amplitude(), 0.2);
  assert.equal(f.input('bass').amplitude(), 0.4);
  assert.equal(f.opened(), 1);
  assert.equal(f.nodes.length, 3);
});

test('default buses preserve levels and route every output through one master', () => {
  const f = fixture();
  const harmony = f.input('harmony');
  const bass = f.input('bass');
  const master = f.input('master');
  assert.notEqual(harmony, bass);
  assert.deepEqual(harmony.outputs, [master]);
  assert.deepEqual(bass.outputs, [master]);
  assert.deepEqual(master.outputs, [f.destination]);
  assert.equal(harmony.amplitude(), 1);
  assert.equal(bass.amplitude(), 1);
  assert.equal(f.input('harmony'), harmony);
});

test('independent layer gains affect already connected voices; master mutes all layers', () => {
  const f = fixture();
  const harmonyVoice = new FakeNode();
  const bassVoice = new FakeNode();
  const processedDrums = new FakeNode();
  harmonyVoice.connect(f.input('harmony'));
  bassVoice.connect(f.input('bass'));
  processedDrums.connect(f.input('master'));
  f.mixer.setVolumes({ masterVolume: 0.5, harmonyVolume: 0, bassVolume: 0.8 });
  assert.equal(harmonyVoice.amplitude(), 0);
  assert.equal(bassVoice.amplitude(), 0.4);
  assert.equal(processedDrums.amplitude(), 0.5);
  f.mixer.setVolumes({ masterVolume: 0, harmonyVolume: 1, bassVolume: 1 });
  assert.equal(harmonyVoice.amplitude(), 0);
  assert.equal(bassVoice.amplitude(), 0);
  assert.equal(processedDrums.amplitude(), 0);
});

test('invalid or out-of-range mixer levels cannot amplify or poison output gains', () => {
  const f = fixture();
  f.mixer.setVolumes({ masterVolume: 4, harmonyVolume: -1, bassVolume: Number.NaN });
  assert.equal(f.input('master').gain.value, 1);
  assert.equal(f.input('harmony').gain.value, 0);
  assert.equal(f.input('bass').gain.value, 1);
});
