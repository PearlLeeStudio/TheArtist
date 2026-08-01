/**
 * Genre-aware instrument registry — Pearl 2026-05-09 Phase 2b.
 *
 * Lazy-loads smplr Soundfont instruments (FluidR3 GM samples on CDN) so each
 * 13-genre vocab entry can play its harmony layer with a genre-appropriate
 * instrument. Replaces the previous piano/guitar/vocal-only palette.
 *
 * Architecture:
 *   - One Soundfont instance per layer and GM instrument name, lazy-loaded from
 *     gleitz/midi-js-soundfonts CDN (~100KB-1MB per instrument).
 *   - Shares Tone's AudioContext for unified scheduling with the rest of
 *     playback (drums, MIDI input, etc.).
 *   - Cold-load grace: if instrument not ready, callers should fall back
 *     to the existing Tone.js piano/guitar/vocal so playback never silences.
 *
 * Boundary (revised 2026-05-09):
 *   Artist owns the genre→instrument decision. Voyager only sends genre.
 *   GENRE_HARMONY_INSTRUMENT / GENRE_BASS_INSTRUMENT (genreTables.ts) are
 *   the authoritative maps for the harmony and bass layers; playSettings.ts
 *   resolves a song's genre through them.
 */
import * as Tone from 'tone';
import { Soundfont, SplendidGrandPiano, ElectricPiano } from 'smplr';
import { getInstrumentOutput } from './mixer';
import type { InstrumentLayer } from './audioMixer';
import { beginTiming } from './experienceTiming.ts';
import { checkedInstrumentContext, instrumentStorage } from './instrumentAudio.ts';

/**
 * Pearl 2026-05-09 — Logic Pro 퀄리티 요청. Tier-1 (curated, high quality):
 *   acoustic_grand_piano → smplr.SplendidGrandPiano (Yamaha samples, Logic
 *                          Pro Steinway/Yamaha-class)
 *   electric_piano_1     → smplr.ElectricPiano (Rhodes samples)
 * Tier-2 (FluidR3_GM Soundfont — solid GM but not Logic Pro level):
 *   everything else (acoustic_guitar_steel, drawbar_organ, sax, etc.)
 *
 * For drums see drums.ts (DrumGizmo Aasimonster v2.1 acoustic samples, GPL-3.0;
 * its smplr drum machines are picked by no built-in pattern).
 */
type InstrumentLike = {
  load: Promise<unknown>;
  start: (args: { note: number | string; velocity?: number; time?: number; duration?: number }) => unknown;
};
const registry: Record<string, { sf: InstrumentLike; ready: boolean; failed: boolean }> = {};

/** Each layer gets its own instance of an instrument, wired to that layer's gain. */
function registryKey(name: string, layer: InstrumentLayer): string {
  return `${layer}:${name}`;
}

function getAudioContext(): AudioContext {
  return Tone.getContext().rawContext as AudioContext;
}

/** Curated smplr classes for high-quality samples (Logic Pro-equivalent
 *  free options). Falls through to FluidR3 GM Soundfont for everything else.
 *
 *  Both smplr and Tone sources use native layer gains in the same context.
 *  A layer owns its instance, so changing harmony gain never changes bass.
 */
function buildCuratedInstrument(name: string, destination: AudioNode, ctx: AudioContext): InstrumentLike | null {
  if (name === 'acoustic_grand_piano') {
    return new SplendidGrandPiano(ctx, { destination, storage: instrumentStorage }) as unknown as InstrumentLike;
  }
  if (name === 'electric_piano_1') {
    // smplr's ElectricPiano accepts 'CP80' (no hyphen) — 'CP-80' throws.
    // Other valid names: PianetT, WurlitzerEP200, TX81Z.
    return new ElectricPiano(ctx, { instrument: 'CP80', destination, storage: instrumentStorage }) as unknown as InstrumentLike;
  }
  return null;
}

/**
 * Load (or return cached) smplr Soundfont for a GM instrument name.
 * Names follow gleitz/midi-js-soundfonts convention: snake_case English
 * (e.g. "acoustic_grand_piano", "electric_piano_1", "alto_sax",
 * "acoustic_guitar_steel", "drawbar_organ"). A name outside FluidR3_GM's
 * 128-entry list 404s on the CDN — validate additions against
 * https://gleitz.github.io/midi-js-soundfonts/FluidR3_GM/names.json.
 */
export function getInstrument(name: string, layer: InstrumentLayer = 'harmony'): InstrumentLike | null {
  const key = registryKey(name, layer);
  if (!registry[key]) {
    try {
      // Try curated high-quality class first (SplendidGrandPiano,
      // ElectricPiano CP-80 etc. — closest free thing to Logic Pro
      // built-in samples). Falls back to FluidR3_GM Soundfont.
      const destination = getInstrumentOutput(layer);
      const checked = checkedInstrumentContext(getAudioContext());
      let inst = buildCuratedInstrument(name, destination, checked.context);
      if (inst === null) {
        inst = new Soundfont(checked.context, {
          instrument: name,
          kit: 'FluidR3_GM',
          destination,
          storage: instrumentStorage,
        }) as unknown as InstrumentLike;
      }
      const source = inst;
      const entry = { sf: { start: source.start.bind(source), load: source.load.then(() => checked.assertReady()) }, ready: false, failed: false };
      registry[key] = entry;
      entry.sf.load.then(() => { entry.ready = true; })
        .catch((err) => { entry.failed = true; console.warn(`Instrument '${name}' failed:`, err); });
    } catch (err) {
      console.warn(`Instrument '${name}' init failed:`, err);
      return null;
    }
  }
  return registry[key].sf;
}

export function isInstrumentReady(name: string, layer: InstrumentLayer = 'harmony'): boolean {
  return registry[registryKey(name, layer)]?.ready ?? false;
}

/**
 * Warm the CDN samples for `names` and resolve once they are playable.
 *
 * Without this, playback starts the Transport immediately while the
 * Soundfont is still downloading, so `playChordOnInstrument` reports
 * not-ready and the opening bars fall through to the Tone.js synth —
 * audible as the first measures sounding thin or wrong before the real
 * instrument cuts in. Awaiting here makes the first bar sound like the
 * rest of the take.
 *
 * Reject on failure or timeout, so playback cannot silently substitute a
 * different instrument or start with missing samples. A later Play retries.
 */
export function preloadInstruments(
  names: (string | null | undefined)[],
  timeoutMs = 30_000,
  layer: InstrumentLayer = 'harmony',
): Promise<void> {
  const wanted = [...new Set(names.filter((n): n is string => !!n))];
  if (wanted.length === 0) return Promise.resolve();
  const finish = beginTiming('instrument.ready', wanted.every((name) => isInstrumentReady(name, layer)) ? 'warm' : 'cold');
  const loads = wanted.map((name) => {
    const key = registryKey(name, layer);
    if (registry[key]?.failed) delete registry[key];
    const inst = getInstrument(name, layer);
    return inst ? inst.load : Promise.reject(new Error(`Could not prepare ${name.replaceAll('_', ' ')}.`));
  });
  return new Promise<void>((resolve, reject) => {
    const timeout = setTimeout(() => {
      finish('timeout');
      reject(new Error('Instrument samples are still loading. Please try Play again.'));
    }, timeoutMs);
    void Promise.all(loads).then(() => {
      clearTimeout(timeout);
      finish(wanted.every((name) => isInstrumentReady(name, layer)) ? 'ok' : 'error');
      resolve();
    }, (error: unknown) => {
      clearTimeout(timeout);
      finish('error');
      reject(error);
    });
  });
}

let prefetchQueue: string[] = [];
let prefetchActive = 0;
const PREFETCH_CONCURRENCY = 2;

/** Warm the family the user is browsing. Replace stale queued choices and
 * cap background decoding; direct selections still bypass this queue. */
export function prefetchHarmonyInstruments(names: string[]): void {
  prefetchQueue = [...new Set(names)].filter((name) => !registry[registryKey(name, 'harmony')]);
  drainPrefetchQueue();
}

function drainPrefetchQueue(): void {
  while (prefetchActive < PREFETCH_CONCURRENCY && prefetchQueue.length) {
    const name = prefetchQueue.shift()!;
    // A hover or selection may have started this load since it was queued.
    if (registry[registryKey(name, 'harmony')]) continue;
    const inst = getInstrument(name);
    if (!inst) continue;
    prefetchActive++;
    void inst.load.catch(() => undefined).finally(() => {
      prefetchActive--;
      drainPrefetchQueue();
    });
  }
}

/**
 * Trigger a chord (one or more MIDI note numbers) on a specific GM instrument.
 * Returns true if the trigger fired through smplr; false if the instrument
 * isn't ready (caller should fall back to Tone.js synth so audio never silences).
 *
 * `time` is AudioContext seconds (matches what Tone.Transport.schedule passes
 * to its callback's first arg). `duration` is in seconds. `velocity` is 0-1.
 */
export function playChordOnInstrument(
  instrumentName: string,
  midiNotes: number[],
  time: number,
  duration: number,
  velocity: number = 1.0,
  layer: InstrumentLayer = 'harmony',
): boolean {
  const inst = getInstrument(instrumentName, layer);
  if (!inst || !isInstrumentReady(instrumentName, layer)) return false;
  const vel = Math.round(Math.max(0, Math.min(1, velocity)) * 127);
  for (const note of midiNotes) {
    inst.start({ note, velocity: vel, time, duration });
  }
  return true;
}
