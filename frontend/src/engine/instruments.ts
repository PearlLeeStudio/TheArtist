/**
 * Genre-aware instrument registry — Pearl 2026-05-09 Phase 2b.
 *
 * Lazy-loads smplr Soundfont instruments (FluidR3 GM samples on CDN) so each
 * 13-genre vocab entry can play its harmony layer with a genre-appropriate
 * instrument. Replaces the previous piano/guitar/vocal-only palette.
 *
 * Architecture:
 *   - One Soundfont instance per GM instrument name, lazy-loaded from
 *     gleitz/midi-js-soundfonts CDN (~100KB-1MB per instrument).
 *   - Shares Tone's AudioContext for unified scheduling with the rest of
 *     playback (drums, MIDI input, etc.).
 *   - Cold-load grace: if instrument not ready, callers should fall back
 *     to the existing Tone.js piano/guitar/vocal so playback never silences.
 *
 * Boundary (revised 2026-05-09):
 *   Artist owns the genre→instrument decision. Voyager only sends genre.
 *   GENRE_HARMONY_INSTRUMENT below is the authoritative map for the
 *   harmony layer — when R4 multi-track lands, equivalent maps for bass +
 *   melody join.
 */
import * as Tone from 'tone';
import { Soundfont, SplendidGrandPiano, ElectricPiano } from 'smplr';

/**
 * Pearl 2026-05-09 — Logic Pro 퀄리티 요청. Tier-1 (curated, high quality):
 *   acoustic_grand_piano → smplr.SplendidGrandPiano (Yamaha samples, Logic
 *                          Pro Steinway/Yamaha-class)
 *   electric_piano_1     → smplr.ElectricPiano (Rhodes samples)
 * Tier-2 (FluidR3_GM Soundfont — solid GM but not Logic Pro level):
 *   everything else (acoustic_guitar_steel, hammond_organ, sax, etc.)
 *
 * For drums see drums.ts (smplr DrumMachine TR-808 + MuldjordKit acoustic).
 */
type InstrumentLike = {
  load: Promise<unknown>;
  start: (args: { note: number | string; velocity?: number; time?: number; duration?: number }) => unknown;
};
const registry: Record<string, { sf: InstrumentLike; ready: boolean }> = {};

function getAudioContext(): AudioContext {
  return Tone.getContext().rawContext as AudioContext;
}

/** Curated smplr classes for high-quality samples (Logic Pro-equivalent
 *  free options). Falls through to FluidR3 GM Soundfont for everything else.
 *
 *  Connects to audioCtx.destination by default (not Tone.getDestination().input
 *  which can throw "value with key not found" on some Tone v15 versions).
 */
function buildCuratedInstrument(name: string): InstrumentLike | null {
  const ctx = getAudioContext();
  if (name === 'acoustic_grand_piano') {
    return new SplendidGrandPiano(ctx) as unknown as InstrumentLike;
  }
  if (name === 'electric_piano_1') {
    // smplr's ElectricPiano accepts 'CP80' (no hyphen) — 'CP-80' throws.
    // Other valid names: PianetT, WurlitzerEP200, TX81Z.
    return new ElectricPiano(ctx, { instrument: 'CP80' }) as unknown as InstrumentLike;
  }
  return null;
}

/**
 * Load (or return cached) smplr Soundfont for a GM instrument name.
 * Names follow gleitz/midi-js-soundfonts convention: snake_case English
 * (e.g. "acoustic_grand_piano", "electric_piano_1", "alto_sax",
 * "acoustic_guitar_steel", "hammond_organ").
 */
export function getInstrument(name: string): InstrumentLike | null {
  if (!registry[name]) {
    try {
      // Try curated high-quality class first (SplendidGrandPiano,
      // ElectricPiano CP-80 etc. — closest free thing to Logic Pro
      // built-in samples). Falls back to FluidR3_GM Soundfont.
      let inst = buildCuratedInstrument(name);
      if (inst === null) {
        inst = new Soundfont(getAudioContext(), {
          instrument: name,
          kit: 'FluidR3_GM',
        }) as unknown as InstrumentLike;
      }
      registry[name] = { sf: inst, ready: false };
      inst.load.then(() => { registry[name].ready = true; })
        .catch((err) => console.warn(`Instrument '${name}' failed:`, err));
    } catch (err) {
      console.warn(`Instrument '${name}' init failed:`, err);
      return null;
    }
  }
  return registry[name].sf;
}

export function isInstrumentReady(name: string): boolean {
  return registry[name]?.ready ?? false;
}

/**
 * 13-genre → harmony-layer GM instrument (Pearl 2026-05-09).
 * Phase 2b — covers harmony only. Bass + melody maps land with R4 multi-track.
 */
export const GENRE_HARMONY_INSTRUMENT: Record<string, string> = {
  jazz:       'electric_piano_1',     // Rhodes — jazz comping standard
  pop:        'acoustic_grand_piano',
  rock:       'overdriven_guitar',
  blues:      'electric_guitar_clean',
  bossa:      'acoustic_guitar_nylon', // Brazilian classical-style nylon
  classical:  'acoustic_grand_piano',
  country:    'acoustic_guitar_steel',
  rnb_soul:   'electric_piano_1',     // Rhodes (R&B/neo-soul standard)
  hip_hop:    'electric_piano_1',     // Soulful chord pads under hip-hop
  electronic: 'pad_2_warm',           // Synth pad
  funk:       'electric_guitar_clean', // Funky comping rhythm guitar
  folk:       'acoustic_guitar_steel',
  gospel:     'hammond_organ',        // Church organ
};

export function harmonyInstrumentForGenre(genre: string | null | undefined): string | null {
  if (!genre) return null;
  return GENRE_HARMONY_INSTRUMENT[genre.toLowerCase()] ?? null;
}

/**
 * 13-genre → bass-layer GM instrument (Pearl 2026-05-10 R4 MVP).
 * Pearl scoped to two organic options: orchestral contrabass (upright) OR
 * electric bass (P-bass/J-bass fingerstyle). Synth-bass / slap-bass are
 * achievable later via instrument swap on the same map. Choices follow
 * recording-convention norms for each genre.
 */
export const GENRE_BASS_INSTRUMENT: Record<string, string> = {
  jazz:       'contrabass',           // upright = jazz ensemble standard
  pop:        'electric_bass_finger', // modern pop = J-bass fingerstyle
  rock:       'electric_bass_finger', // P/J-bass, drives the rhythm section
  blues:      'electric_bass_finger', // modern blues bands
  bossa:      'contrabass',           // bossa nova = nylon + upright (Jobim)
  classical:  'contrabass',           // orchestral double bass
  country:    'electric_bass_finger', // modern country = electric (older = upright)
  rnb_soul:   'electric_bass_finger', // R&B / neo-soul fingerstyle
  hip_hop:    'electric_bass_finger', // bass synth conceptually but staying organic
  electronic: 'electric_bass_finger', // synth-bass would be ideal; staying organic per Pearl scope
  funk:       'electric_bass_finger', // funk = slap eventually; for now fingerstyle root
  folk:       'contrabass',           // bluegrass / traditional folk = upright
  gospel:     'electric_bass_finger', // modern gospel = electric
};

export function bassInstrumentForGenre(genre: string | null | undefined): string | null {
  if (!genre) return null;
  return GENRE_BASS_INSTRUMENT[genre.toLowerCase()] ?? null;
}

// Melody layer dropped 2026-05-10 — see arrangement.ts header for
// rationale. Harmony's instrument-override grid (VoicingViewer) absorbs
// the variety role melody used to provide.

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
): boolean {
  const inst = getInstrument(instrumentName);
  if (!inst || !isInstrumentReady(instrumentName)) return false;
  const vel = Math.round(Math.max(0, Math.min(1, velocity)) * 127);
  for (const note of midiNotes) {
    inst.start({ note, velocity: vel, time, duration });
  }
  return true;
}
