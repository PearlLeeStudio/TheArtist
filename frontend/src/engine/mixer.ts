import * as Tone from 'tone';
import { createAudioMixer } from './audioMixer';
import type { InstrumentLayer, MixerVolumes } from './audioMixer';

const mixer = createAudioMixer(() => Tone.getContext().rawContext as AudioContext);

/** Safe at startup: these levels are remembered without creating audio nodes. */
export function setMixerVolumes(volumes: MixerVolumes): void {
  mixer.setVolumes(volumes);
}

export function getInstrumentOutput(layer: InstrumentLayer): GainNode {
  return mixer.input(layer);
}

/** The existing drum gain/filter/compressor chain joins the master here. */
export function getMasterInput(): GainNode {
  return mixer.input('master');
}
