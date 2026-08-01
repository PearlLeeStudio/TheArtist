/** Native gain buses shared by Tone and smplr. No context is opened until
 * an audio source asks for its output; restoring settings only stores levels. */
export interface MixerVolumes {
  masterVolume: number;
  harmonyVolume: number;
  bassVolume: number;
}

export type InstrumentLayer = 'harmony' | 'bass';
type MixerBus = InstrumentLayer | 'master';

const DEFAULT_VOLUMES: MixerVolumes = {
  masterVolume: 1,
  harmonyVolume: 1,
  bassVolume: 1,
};

function level(value: number, fallback: number): number {
  return Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : fallback;
}

/** Drums retain their existing gain before the filter/compressor. Their
 * processed output connects to `master`, alongside the two pitched buses. */
export function createAudioMixer(getContext: () => BaseAudioContext) {
  let volumes = { ...DEFAULT_VOLUMES };
  let buses: Record<MixerBus, GainNode> | null = null;

  const apply = () => {
    if (!buses) return;
    // Change actual output gains, including sustained notes already playing.
    // Setting zero directly guarantees mute rather than an asymptotic fade.
    buses.master.gain.value = volumes.masterVolume;
    buses.harmony.gain.value = volumes.harmonyVolume;
    buses.bass.gain.value = volumes.bassVolume;
  };

  return {
    setVolumes(next: MixerVolumes): void {
      volumes = {
        masterVolume: level(next.masterVolume, DEFAULT_VOLUMES.masterVolume),
        harmonyVolume: level(next.harmonyVolume, DEFAULT_VOLUMES.harmonyVolume),
        bassVolume: level(next.bassVolume, DEFAULT_VOLUMES.bassVolume),
      };
      apply();
    },

    input(bus: MixerBus): GainNode {
      if (!buses) {
        const context = getContext();
        buses = {
          master: context.createGain(),
          harmony: context.createGain(),
          bass: context.createGain(),
        };
        apply();
        buses.harmony.connect(buses.master);
        buses.bass.connect(buses.master);
        buses.master.connect(context.destination);
      }
      return buses[bus];
    },
  };
}
