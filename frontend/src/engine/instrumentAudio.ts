import type { Storage } from 'smplr';

// The derived CP80 m4a/ogg files contain damaged container data. Keep the
// same instrument and velocity layers, using Greg Sullivan's lossless source.
const CP80_SOURCE = 'https://raw.githubusercontent.com/sfzinstruments/GregSullivan.E-Pianos/8c3e581acda3594b553948ff0222d4f84a698376/CP80/Samples/';
const CP80_DERIVED = 'https://smpldsnds.github.io/sfzinstruments-greg-sullivan-e-pianos/cp80/samples/';

export function instrumentSampleUrl(url: string): string {
  return url.startsWith(CP80_DERIVED)
    ? CP80_SOURCE + url.slice(CP80_DERIVED.length).replace(/\.(m4a|ogg)$/, '.flac')
    : url;
}

export const instrumentStorage: Storage = {
  async fetch(url) {
    const response = await fetch(instrumentSampleUrl(url), { signal: AbortSignal.timeout(30_000) });
    if (!response.ok) throw new Error(`Instrument sample download failed (${response.status}).`);
    return response;
  },
};

/** smplr resolves load even after decode failures. Track them per instrument
 * without replacing the shared context's decoder or affecting other layers. */
export function checkedInstrumentContext(context: AudioContext) {
  let failure: unknown;
  const checked = new Proxy(context, {
    get(target, key) {
      if (key === 'decodeAudioData') return async (data: ArrayBuffer) => {
        try { return await target.decodeAudioData(data); }
        catch (error) { failure = error; throw error; }
      };
      const value = Reflect.get(target, key, target);
      return typeof value === 'function' ? value.bind(target) : value;
    },
  });
  return {
    context: checked,
    assertReady() {
      if (failure) throw new Error('Some instrument samples could not be decoded. Reload the instrument and try again.', { cause: failure });
    },
  };
}
