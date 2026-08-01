/**
 * Drum playback (Tone.js).
 *
 * Primary path: Tone.Sampler instances loaded from `/drums/*.flac` —
 * samples are MuldjordKit (FreePats edition, CC BY 4.0). One Sampler per voice
 * so each can have its own volume balance and release tail (long for cymbals,
 * short for shells). Attribution + sample provenance: see public/drums/LICENSE.txt.
 *
 * Fallback path: the existing synthesised voices below stay live and are used
 * before samples finish loading, or if loading fails. The `playDrum()` API is
 * unchanged so callers don't care which path is taken.
 */
import * as Tone from 'tone';

// ---------- Synthesized drum voices (fallback while samples load) ----------

let kickSynth: Tone.MembraneSynth | null = null;
let snareSynth: Tone.NoiseSynth | null = null;
let snareTone: Tone.MembraneSynth | null = null;  // body
let hihatSynth: Tone.MetalSynth | null = null;
let openHihatSynth: Tone.MetalSynth | null = null;
let rideSynth: Tone.MetalSynth | null = null;
let crashSynth: Tone.MetalSynth | null = null;
let crashNoise: Tone.NoiseSynth | null = null;
let tomHi: Tone.MembraneSynth | null = null;
let tomMid: Tone.MembraneSynth | null = null;
let tomLo: Tone.MembraneSynth | null = null;
let drumBus: Tone.Gain | null = null;

// ---------- Electronic / hip-hop voices (smplr real-sample drum machines) ----
// Pearl feedback timeline:
//   round 1 (synth 808)   → too thin
//   round 2 (smplr TR-808) → kept for hip-hop (classic boom-bap)
//   round 3 (LM-2 for EDM) → still missing 비트감
//   round 4 (this commit)  → smplr only ships 5 vintage machines (no TR-909!).
//     Switched electronic LM-2 → MFB-512 (raw analog Berlin synth, much
//     punchier kick + crisper hat than the dry Linn LM-2). Kept LM-2/CR-8000
//     in the dispatch for future variety.
// Tone.js synth versions stay live as fallback while CDN samples decode.
import { DrumMachine } from 'smplr';

// Per-machine smplr DrumMachine instances (lazy-loaded on first request).
const drumMachines: Record<string, { dm: DrumMachine; ready: boolean }> = {};

// Map our voice convention → smplr drum-machine sample names. Roland machines
// (TR-808, TR-909) and Linn (LM-2) all use the same smplr aliases.
const DRUM_MACHINE_VOICE_MAP: Partial<Record<DrumVoice, string>> = {
  K: 'kick',
  S: 'snare',
  X: 'cowbell',
  H: 'hihat-closed',
  O: 'hihat-open',
  C: 'clap',
  R: 'cymbal',
  T: 'tom-high',
  M: 'tom-mid',
  L: 'tom-low',
};

function getDrumMachine(name: string): DrumMachine | null {
  if (!drumMachines[name]) {
    try {
      const audioCtx = Tone.getContext().rawContext as AudioContext;
      const dm = new DrumMachine(audioCtx, { instrument: name, volume: 100 });
      drumMachines[name] = { dm, ready: false };
      dm.load.then(() => { drumMachines[name].ready = true; })
        .catch((err) => console.warn(`smplr ${name} failed:`, err));
    } catch (err) {
      console.warn(`smplr DrumMachine '${name}' init failed:`, err);
      return null;
    }
  }
  return drumMachines[name].dm;
}

function isDrumMachineReady(name: string): boolean {
  return drumMachines[name]?.ready ?? false;
}

function ensureSynths(): void {
  if (drumBus) return;

  // Output chain: drum sources → busGain → HPF (removes sub rumble below 50 Hz)
  // → Compressor (tight 4:1 glue for punch without being louder) → destination.
  drumBus = new Tone.Gain(0.6);
  const hpf = new Tone.Filter({ type: 'highpass', frequency: 50, Q: 0.7 });
  const glue = new Tone.Compressor({
    threshold: -18,
    ratio: 4,
    attack: 0.003,
    release: 0.12,
    knee: 6,
  });
  drumBus.chain(hpf, glue, Tone.getDestination());

  // Kick: snappy transient, less boom, shorter tail.
  kickSynth = new Tone.MembraneSynth({
    pitchDecay: 0.03,
    octaves: 5,
    envelope: { attack: 0.001, decay: 0.18, sustain: 0, release: 0.05 },
  }).connect(drumBus);

  // Snare noise: crisper top, shorter decay.
  snareSynth = new Tone.NoiseSynth({
    noise: { type: 'white' },
    envelope: { attack: 0.001, decay: 0.10, sustain: 0 },
  }).connect(drumBus);

  // Snare body: tight body knock.
  snareTone = new Tone.MembraneSynth({
    pitchDecay: 0.015,
    octaves: 2,
    envelope: { attack: 0.001, decay: 0.05, sustain: 0 },
  }).connect(drumBus);

  // Closed hi-hat: already tight; slightly tighter decay.
  hihatSynth = new Tone.MetalSynth({
    envelope: { attack: 0.001, decay: 0.035, release: 0.01 },
    harmonicity: 5.1,
    modulationIndex: 32,
    resonance: 4000,
    octaves: 1.5,
  }).connect(drumBus);
  hihatSynth.volume.value = -18;

  // Open hi-hat: shorter decay so it doesn't smear into next beat.
  openHihatSynth = new Tone.MetalSynth({
    envelope: { attack: 0.001, decay: 0.18, release: 0.05 },
    harmonicity: 5.1,
    modulationIndex: 32,
    resonance: 4000,
    octaves: 1.5,
  }).connect(drumBus);
  openHihatSynth.volume.value = -20;

  // Ride: drier, articulated.
  rideSynth = new Tone.MetalSynth({
    envelope: { attack: 0.001, decay: 0.25, release: 0.10 },
    harmonicity: 8,
    modulationIndex: 22,
    resonance: 7000,
    octaves: 1,
  }).connect(drumBus);
  rideSynth.volume.value = -22;

  // Crash: keep metallic shimmer, trim noise wash so the tail is clean.
  crashSynth = new Tone.MetalSynth({
    envelope: { attack: 0.001, decay: 1.2, release: 0.4 },
    harmonicity: 5.1,
    modulationIndex: 64,
    resonance: 5000,
    octaves: 2.5,
  }).connect(drumBus);
  crashSynth.volume.value = -14;

  crashNoise = new Tone.NoiseSynth({
    noise: { type: 'white' },
    envelope: { attack: 0.002, decay: 0.6, sustain: 0.02, release: 0.5 },
  }).connect(drumBus);
  crashNoise.volume.value = -24;

  // Toms: tighter heads, less boom.
  tomHi = new Tone.MembraneSynth({
    pitchDecay: 0.02,
    octaves: 3,
    envelope: { attack: 0.001, decay: 0.18, sustain: 0, release: 0.1 },
  }).connect(drumBus);
  tomMid = new Tone.MembraneSynth({
    pitchDecay: 0.025,
    octaves: 3,
    envelope: { attack: 0.001, decay: 0.22, sustain: 0, release: 0.12 },
  }).connect(drumBus);
  tomLo = new Tone.MembraneSynth({
    pitchDecay: 0.03,
    octaves: 4,
    envelope: { attack: 0.001, decay: 0.28, sustain: 0, release: 0.15 },
  }).connect(drumBus);

  // ---------- smplr drum machines (Pearl 2026-05-09) ----------
  // TR-808 (hip-hop) + LM-2 (electronic, Linn drum machine — crisper modern
  // beat character per Pearl 비트감 feedback) lazy-load on first pattern
  // play via getDrumMachine().

  // ---------- Real samples (MuldjordKit, CC BY 4.0) ----------
  // One Sampler per voice so we can tune per-voice mix volume and release.
  // Each sample is mapped to a single note (C2) — we always trigger that note,
  // so no pitch shifting happens.
  const SAMPLE_RELEASE: Record<DrumVoice, number> = {
    K: 0.05, S: 0.05, X: 0.02, H: 0.02, O: 0.4,
    R: 3.0,  C: 4.0,  T: 0.3,  M: 0.3,  L: 0.3,
  };
  const SAMPLE_VOL_DB: Record<DrumVoice, number> = {
    K: -2,   S: -3,   X: -8,   H: -14,  O: -16,
    R: -16,  C: -12,  T: -6,   M: -6,   L: -6,
  };
  // Pearl 2026-05-09: reverted from multi-velocity (felt worse) back to the
  // single-velocity 80-95% high-vel state, which Pearl
  // described as the "real-sounding" baseline. Tone.js velocity arg in
  // triggerAttackRelease handles soft/loud amplitude scaling.
  const SAMPLE_FILE: Record<DrumVoice, string> = {
    K: 'kick.flac',          S: 'snare.flac',     X: 'sidestick.flac',
    H: 'hihat-closed.flac',  O: 'hihat-open.flac',
    R: 'ride.flac',          C: 'crash.flac',
    T: 'tom-hi.flac',        M: 'tom-mid.flac',   L: 'tom-lo.flac',
  };

  for (const v of ['K','S','X','H','O','R','C','T','M','L'] as DrumVoice[]) {
    const sampler = new Tone.Sampler({
      urls: { C2: SAMPLE_FILE[v] },
      baseUrl: '/drums/',
      release: SAMPLE_RELEASE[v],
    }).connect(drumBus);
    sampler.volume.value = SAMPLE_VOL_DB[v];
    samplers[v] = sampler;
  }

  // Mark ready once every Sampler has decoded its file. If decoding fails
  // (e.g. files missing in deployment), `samplesReady` stays false and we
  // keep using the synth fallback below — playback never errors.
  Tone.loaded()
    .then(() => { samplesReady = true; })
    .catch((err) => { console.warn('Drum samples failed to load — using synth fallback:', err); });
}

// Voices: Kick, Snare, Hihat (closed), Open hh, Ride, siX-stick (rim),
//         Crash cymbal, Tom-high, tom-Mid, tom-Low (floor)
type DrumVoice = 'K' | 'S' | 'H' | 'O' | 'R' | 'X' | 'C' | 'T' | 'M' | 'L';

// ---------- Real-sample voices ----------
const samplers: Partial<Record<DrumVoice, Tone.Sampler>> = {};
let samplesReady = false;

// How long to hold each voice before release. Shells get short holds (decay is
// driven by the sample); cymbals get long holds so the natural ring isn't
// truncated.
const VOICE_HOLD: Record<DrumVoice, string> = {
  K: '4n',  S: '4n',  X: '8n',  H: '16n', O: '4n',
  R: '1m',  C: '1m',  T: '4n',  M: '4n',  L: '4n',
};

export function playDrum(
  voice: DrumVoice,
  time: number,
  velocity = 1.0,
  kit: 'acoustic' | '808' | 'MFB' | 'LM-2' = 'acoustic',
): void {
  ensureSynths();

  // smplr drum machines (real samples from smpldsnds.github.io CDN):
  //   '808'  → TR-808 (hip-hop classic, sub kick + click snare)
  //   'MFB'  → MFB-512 (Berlin analog, punchy kick + crisp hat — EDM default)
  //   'LM-2' → Linn LM-2 (vintage drier electro, kept for variety)
  // Pattern.kit selects. Velocity jitter ±8% per trigger avoids machine-gun
  // repetition (same trick we use on the acoustic Sampler kit). Falls through
  // to synth fallback below if the chosen machine isn't loaded yet.
  if (kit === '808' || kit === 'MFB' || kit === 'LM-2') {
    const machineName = kit === '808' ? 'TR-808' : kit === 'MFB' ? 'MFB-512' : 'LM-2';
    const dm = getDrumMachine(machineName);
    if (dm && isDrumMachineReady(machineName)) {
      const sampleName = DRUM_MACHINE_VOICE_MAP[voice];
      if (sampleName) {
        const jittered = velocity * (0.92 + Math.random() * 0.16);
        try {
          dm.start({
            note: sampleName,
            velocity: Math.round(Math.max(0, Math.min(1, jittered)) * 127),
            time,
          });
        } catch (err) {
          console.warn(`smplr ${machineName} trigger failed (${voice}):`, err);
        }
        return;
      }
    }
    // not ready or voice unmapped → fall through to acoustic synth
    // fallback below (avoids silence on cold load).
  }

  // Acoustic kit (default): prefer real samples once decoded, synth fallback
  // until then so the user never hears silence on early playback.
  // Pearl 2026-05-09: per-trigger pitch detune (±15 cents) + velocity jitter
  // (±8%) for natural variation — avoids the "machine-gun" repetition feel
  // of single-sample drums. Cymbals/ride get smaller detune (more sensitive
  // to pitch shift). Real drum kits show ±15-25 cents per hit naturally.
  if (samplesReady) {
    const sampler = samplers[voice];
    if (sampler) {
      // Per-trigger pitch variation (Pearl 2026-05-09) avoids the machine-gun
      // feel of a single repeated sample. Tone v15 Sampler doesn't expose a
      // .detune Signal, so we shift the trigger pitch directly: each Sampler
      // is mapped to C2 (~65.41 Hz), and triggering at a frequency offset by
      // ±N cents gives that pitch shift via Tone's playback-rate math.
      const detuneCents =
        voice === 'C' || voice === 'R' || voice === 'O'
          ? (Math.random() - 0.5) * 12   // cymbals: ±6 cents (subtle)
          : (Math.random() - 0.5) * 30;  // shells: ±15 cents
      const velJitter = velocity * (0.92 + Math.random() * 0.16); // ±8%
      const detunedHz = 65.4064 * Math.pow(2, detuneCents / 1200);
      try {
        sampler.triggerAttackRelease(detunedHz, VOICE_HOLD[voice], time, velJitter);
      } catch (err) {
        console.warn(`drum trigger failed (${voice}):`, err);
      }
      return;
    }
  }
  switch (voice) {
    case 'K':
      kickSynth?.triggerAttackRelease('C1', '8n', time, velocity);
      break;
    case 'S':
      snareSynth?.triggerAttackRelease('16n', time, 0.6 * velocity);
      snareTone?.triggerAttackRelease('A2', '32n', time, 0.4 * velocity);
      break;
    case 'X':
      // Side-stick / rim: short woody click, no noise burst
      snareTone?.triggerAttackRelease('D3', '64n', time, 0.55 * velocity);
      break;
    case 'H':
      hihatSynth?.triggerAttackRelease('32n', time, 0.5 * velocity);
      break;
    case 'O':
      openHihatSynth?.triggerAttackRelease('8n', time, 0.5 * velocity);
      break;
    case 'R':
      rideSynth?.triggerAttackRelease('8n', time, 0.4 * velocity);
      break;
    case 'C':
      crashSynth?.triggerAttackRelease('2n', time, 0.6 * velocity);
      crashNoise?.triggerAttackRelease('2n', time, 0.5 * velocity);
      break;
    case 'T':
      tomHi?.triggerAttackRelease('E3', '8n', time, velocity);
      break;
    case 'M':
      tomMid?.triggerAttackRelease('B2', '8n', time, velocity);
      break;
    case 'L':
      tomLo?.triggerAttackRelease('F2', '8n', time, velocity);
      break;
  }
}

// ---------- Patterns ----------
// Each pattern is keyed by 16th-note grid position within a 4/4 measure (0..15).
// For 3/4 we use the first 12 positions; for 6/8 the first 12.

/**
 * Hit: a 16th-position plus optional velocity.
 * - number → hit at that position with default velocity (1.0)
 * - [pos, vel] → hit with a custom velocity, e.g. [2, 0.3] = ghost
 */
export type DrumHit = number | [number, number];

export interface DrumPattern {
  id: number;
  name: string;
  emoji: string;
  voices: Partial<Record<DrumVoice, DrumHit[]>>;
  swing?: boolean;  // delays off-8ths slightly
  /** Pearl 2026-05-09: each pattern is the canonical drum for one genre
   *  in the 13-genre vocabulary. getDrumPatternForGenre() resolves by
   *  this field. Variants (e.g. Funk Show, Samba, Rock Anthem) share a
   *  genre with another pattern; the helper picks the lower-id primary. */
  genre?: string;
  /** Drum kit dispatch (Pearl 2026-05-09).
   *  - 'acoustic' (default): Aasimonster multi-mic samples via Tone.Sampler
   *  - '808':  smplr TR-808 (Roland classic, hip-hop boom-bap default)
   *  - 'MFB':  smplr MFB-512 (Berlin analog, electronic default — punchy)
   *  - 'LM-2': smplr LM-2 (Linn vintage electro, available alternate) */
  kit?: 'acoustic' | '808' | 'MFB' | 'LM-2';
}

// Helper to unpack a hit.
function hitPos(h: DrumHit): number {
  return typeof h === 'number' ? h : h[0];
}
function hitVel(h: DrumHit): number {
  return typeof h === 'number' ? 1.0 : h[1];
}

export const DRUM_PATTERNS: DrumPattern[] = [
  {
    // Pop/Rock — straight 8th hats, "boom-ba boom-bap" kick, heavy ghost cloud
    // Identity: syncopated kick on "and-of-3", driving 8ths, many ghost snares
    id: 1,
    name: 'Pop',
    emoji: '🥁',
    genre: 'pop',
    voices: {
      C: [[0, 0.5]],
      K: [[0, 1.0], [6, 0.9], [10, 0.9]],
      S: [[4, 1.0], [12, 1.0],
          [2, 0.22], [7, 0.28], [11, 0.28], [14, 0.32]],
      H: [[0, 0.95], [2, 0.55], [4, 0.85], [6, 0.55],
          [8, 0.95], [10, 0.55], [12, 0.85]],
      O: [[14, 0.65]],
    },
  },
  {
    // Half-time HIP-HOP — Aasimonster acoustic kit (Pearl 2026-05-10 round 5:
    // smplr drum machines all rejected — TR-808 thin, MFB-512 "띡띡" clicky,
    // LM-2 dry. Falling back to Aasimonster which Pearl has consistently
    // liked. Pattern keeps the boom-bap shape: snare ONLY on beat 3 (pos 8),
    // sparse kick, ghost cloud — bar feels twice as long because backbeat
    // is singular.
    id: 2,
    name: 'Hip-hop',
    emoji: '🎤',
    genre: 'hip_hop',
    voices: {
      K: [[0, 1.0], [11, 0.9]],
      S: [[8, 1.0],
          [2, 0.3], [3, 0.22], [6, 0.28],
          [10, 0.35], [11, 0.22], [13, 0.28], [14, 0.35], [15, 0.28]],
      H: [[0, 0.9], [2, 0.5], [4, 0.7], [6, 0.5],
          [8, 0.9], [10, 0.5], [12, 0.7], [14, 0.5]],
    },
  },
  {
    // Modern EDM beat — Aasimonster acoustic kit (Pearl 2026-05-10 round 5:
    // MFB-512 sounded "띡띡" clicky-thin, LM-2 dry, no TR-909 in smplr).
    // Falling back to Aasimonster which Pearl has consistently liked. The
    // electronic feel comes from the *pattern*, not the kit color: 4-on-
    // floor kick on every quarter + ghost "and-of-1", 16th hats with trap-
    // style accents, ghost snares for groove.
    id: 3,
    name: 'Electronic',
    emoji: '⚡',
    genre: 'electronic',
    voices: {
      C: [[0, 0.6]],                                     // crash on 1
      K: [[0, 1.0], [4, 1.0], [8, 1.0], [12, 1.0],
          [3, 0.6]],                                     // ghost kick "and-of-1"
      S: [[4, 1.0], [12, 1.0],
          [10, 0.45], [14, 0.5]],                        // ghost snares for groove
      // 16th hat with extra accents at 7, 11, 15 (trap-style hat roll feel)
      H: [[0, 0.95], [1, 0.5], [2, 0.7], [3, 0.5],
          [4, 0.95], [5, 0.5], [6, 0.7], [7, 0.85],
          [8, 0.95], [9, 0.5], [10, 0.7], [11, 0.85],
          [12, 0.95], [13, 0.5], [14, 0.7], [15, 0.85]],
      O: [[2, 0.7], [6, 0.7], [10, 0.7], [14, 0.7]],     // open hat offbeats
      X: [[7, 0.6], [15, 0.6]],                          // cowbell syncopation
    },
  },
  {
    // Jazz ride: ding — ding-a ding — ding-a on swung 8ths, hat on 2&4
    id: 4,
    name: 'Jazz Swing',
    emoji: '🎷',
    genre: 'jazz',
    voices: {
      R: [[0, 1.0], [4, 0.9], [6, 0.7], [8, 1.0], [12, 0.9], [14, 0.7]],
      H: [[4, 0.55], [12, 0.55]],   // hi-hat foot on 2 & 4
      S: [[2, 0.18], [10, 0.22]],   // light brush ghosts
      K: [[0, 0.55], [8, 0.5]],     // feathered kick
    },
    swing: true,
  },
  {
    // Classic bossa: bass drum dotted feel, cross-stick clave, flowing 8th hats
    id: 5,
    name: 'Bossa',
    emoji: '🌴',
    genre: 'bossa',
    voices: {
      K: [[0, 1.0], [6, 0.85], [8, 1.0], [14, 0.85]],
      X: [[3, 0.9], [6, 0.9], [10, 0.9], [12, 0.9]], // 3-2 cross-stick clave
      H: [[0, 0.75], [2, 0.5], [4, 0.7], [6, 0.5],
          [8, 0.75], [10, 0.5], [12, 0.7], [14, 0.5]],
    },
  },
  {
    // Slow ballad with brushed side-stick and sparse quarter-note hat
    // (Doubles as folk pattern — gentle backbone, brushes feel)
    id: 6,
    name: 'Folk / Ballad',
    emoji: '🎻',
    genre: 'folk',
    voices: {
      K: [[0, 0.9], [10, 0.7]],
      X: [[8, 0.85]],  // side-stick on 3 for softness
      H: [[0, 0.65], [4, 0.65], [8, 0.65], [12, 0.65]],
    },
  },
  {
    // 16th-note funk hat, syncopated kick, crisp backbeat with ghosts
    id: 7,
    name: 'Funk 16',
    emoji: '🪩',
    genre: 'funk',
    voices: {
      K: [[0, 1.0], [3, 0.85], [7, 0.8], [10, 0.9]],
      S: [[4, 1.0], [12, 1.0],
          [2, 0.22], [6, 0.22], [11, 0.22], [14, 0.22]],
      H: [[0, 0.85], [1, 0.4], [2, 0.65], [3, 0.4],
          [4, 0.85], [5, 0.4], [6, 0.65], [7, 0.4],
          [8, 0.85], [9, 0.4], [10, 0.65], [11, 0.4],
          [12, 0.85], [13, 0.4], [14, 0.65], [15, 0.4]],
    },
  },
  {
    // Shuffle (triplet feel via swing) — bluesy 8th shuffle groove
    id: 8,
    name: 'Blues Shuffle',
    emoji: '🎸',
    genre: 'blues',
    voices: {
      K: [[0, 1.0], [8, 1.0], [10, 0.7]],
      S: [[4, 1.0], [12, 1.0]],
      H: [[0, 0.85], [2, 0.55], [4, 0.85], [6, 0.55],
          [8, 0.85], [10, 0.55], [12, 0.85], [14, 0.55]],
    },
    swing: true,
  },
  {
    // Stadium rock anthem: huge crash on 1, driving 8ths, tom fill on beat 4
    id: 9,
    name: 'Rock Anthem',
    emoji: '🎸',
    genre: 'rock',
    voices: {
      C: [[0, 1.0]],
      K: [[0, 1.0], [3, 0.9], [8, 1.0], [11, 0.9]],
      S: [[4, 1.0], [12, 1.0], [6, 0.25]],
      H: [[0, 0.95], [2, 0.6], [4, 0.9], [6, 0.6],
          [8, 0.95], [10, 0.6]],
      // Tom fill across beat 4 (hi → mid → low → floor)
      T: [[12, 1.0]],
      M: [[13, 0.95]],
      L: [[14, 0.95], [15, 1.0]],
    },
  },
  {
    // Funk showcase: all-16th hats, syncopated kicks, ghost cloud, snare accents
    // Variant of Funk 16 (id 7) — kept for live performance variation.
    id: 10,
    name: 'Funk Show',
    emoji: '🔥',
    voices: {
      C: [[0, 0.55]],
      K: [[0, 1.0], [3, 0.9], [6, 0.75], [7, 0.8],
          [10, 0.9], [13, 0.75]],
      S: [[4, 1.0], [12, 1.0],
          [2, 0.25], [6, 0.22], [9, 0.28], [11, 0.25], [14, 0.28], [15, 0.3]],
      H: [[0, 0.95], [1, 0.45], [2, 0.7], [3, 0.45],
          [4, 0.9],  [5, 0.45], [6, 0.7], [7, 0.45],
          [8, 0.95], [9, 0.45], [10, 0.7], [11, 0.45],
          [12, 0.9], [13, 0.45], [14, 0.7]],
      O: [[15, 0.7]],
    },
  },
  {
    // Latin Samba: surdo low, busy hats, tom accents, crash, side-stick clave
    // Variant of Bossa (id 5) — busier latin feel.
    id: 11,
    name: 'Samba',
    emoji: '💃',
    voices: {
      C: [[0, 0.65]],
      K: [[0, 1.0], [3, 0.7], [8, 1.0], [11, 0.7]],
      L: [[4, 0.9], [12, 0.9]],     // surdo on 2 & 4
      X: [[3, 0.85], [6, 0.85], [10, 0.85], [12, 0.85], [14, 0.85]], // clave
      H: [[0, 0.8], [1, 0.5], [2, 0.7], [3, 0.5],
          [4, 0.8], [5, 0.5], [6, 0.7], [7, 0.5],
          [8, 0.8], [9, 0.5], [10, 0.7], [11, 0.5],
          [12, 0.8], [13, 0.5], [14, 0.7], [15, 0.5]],
    },
  },
  // ---- R4 13-genre additions (Pearl 2026-05-09) ----
  {
    // Country 2-step shuffle — kick on 1&3, snare on 2&4, brushes on offbeats
    // Identity: train-beat / honky-tonk feel. Brush stroke (cross-stick) gives
    // the country swing without full shuffle commitment.
    id: 12,
    name: 'Country',
    emoji: '🤠',
    genre: 'country',
    voices: {
      K: [[0, 1.0], [8, 1.0]],
      S: [[4, 1.0], [12, 1.0]],
      X: [[2, 0.55], [6, 0.55], [10, 0.55], [14, 0.55]],
      H: [[0, 0.7], [4, 0.7], [8, 0.7], [12, 0.7]],
    },
    swing: true,
  },
  {
    // R&B / Soul groove — tight pocket, deep ghost cloud around the main
    // backbeat, kick syncopation. Slower than pop, more breathing room.
    id: 13,
    name: 'R&B / Soul',
    emoji: '🎶',
    genre: 'rnb_soul',
    voices: {
      K: [[0, 1.0], [10, 0.85]],
      S: [[4, 1.0], [12, 1.0],
          [2, 0.22], [3, 0.18], [6, 0.24], [10, 0.26], [14, 0.26], [15, 0.18]],
      H: [[0, 0.85], [2, 0.5], [4, 0.75], [6, 0.5],
          [8, 0.85], [10, 0.5], [12, 0.75], [14, 0.5]],
    },
  },
  {
    // Gospel — backbeat anchor + tambourine-feel via open hat on offbeats
    // (no dedicated tambourine voice). Soft cymbal swell on 1.
    id: 14,
    name: 'Gospel',
    emoji: '🙏',
    genre: 'gospel',
    voices: {
      C: [[0, 0.45]],
      K: [[0, 1.0], [8, 1.0]],
      S: [[4, 1.0], [12, 1.0]],
      H: [[0, 0.8], [4, 0.8], [8, 0.8], [12, 0.8]],
      O: [[2, 0.55], [6, 0.55], [10, 0.55], [14, 0.55]], // tambourine substitute
    },
  },
  {
    // Classical — minimal kit, timpani-feel via floor tom on 1, very soft
    // cymbal swell. Most classical pieces have NO drum kit — this is the
    // gentlest representation when the user wants a hint of pulse.
    id: 15,
    name: 'Classical',
    emoji: '🎻',
    genre: 'classical',
    voices: {
      L: [[0, 0.6]],   // floor tom on 1 (timpani feel)
      C: [[0, 0.3]],   // soft cymbal swell on 1
    },
  },
];

// ---- Genre → drum pattern dispatch (Pearl 2026-05-09 R4) ----
// 13-genre vocabulary mapped 1:1 to canonical drum pattern. Variants
// (Funk Show id 10, Samba id 11) are NOT in this map — they're alt
// patterns the user can pick manually but not the auto-route.
const GENRE_TO_DRUM_PATTERN_ID: Record<string, number> = {
  pop: 1,
  hip_hop: 2,
  electronic: 3,
  jazz: 4,
  bossa: 5,
  folk: 6,
  funk: 7,
  blues: 8,
  rock: 9,
  country: 12,
  rnb_soul: 13,
  gospel: 14,
  classical: 15,
};

/**
 * Resolve the canonical drum pattern for a 13-genre vocab entry.
 * Used when the chord generator's genre selection should auto-update the
 * drum pattern. Returns null for unknown genre or when genre is null.
 */
export function getDrumPatternForGenre(genre: string | null | undefined): DrumPattern | null {
  if (!genre) return null;
  const id = GENRE_TO_DRUM_PATTERN_ID[genre];
  if (id === undefined) return null;
  return DRUM_PATTERNS.find(p => p.id === id) ?? null;
}

// ---------- Live pattern selection ----------
// `currentPatternId` is read at *play* time so the user can toggle drum styles
// (1..6) or turn drums off (null) mid-playback without rescheduling.
let currentPatternId: number | null = null;

export function setCurrentDrumPattern(id: number | null): void {
  currentPatternId = id;
}

export function getCurrentDrumPattern(): number | null {
  return currentPatternId;
}

function getPatternById(id: number | null): DrumPattern | null {
  if (id === null) return null;
  return DRUM_PATTERNS.find(p => p.id === id) ?? null;
}

// ---------- Scheduling ----------

let drumScheduledIds: number[] = [];

export function clearDrumSchedule(): void {
  for (const id of drumScheduledIds) Tone.getTransport().clear(id);
  drumScheduledIds = [];
}

/**
 * Schedule one 16th-note grid worth of "drum slots" for a single measure.
 * Each scheduled tick reads the *current* drumPatternId at fire time, so
 * toggling pattern mid-playback takes effect on the next 16th.
 *
 * For 4/4: 16 ticks per measure. For 3/4: 12 ticks. etc.
 */
export function scheduleDrumMeasure(
  measureStart: number,
  secondsPerBeat: number,
  beatsPerMeasure: number,
): void {
  ensureSynths();
  const sixteenthDur = secondsPerBeat / 4;
  const totalSixteenths = beatsPerMeasure * 4;

  for (let pos = 0; pos < totalSixteenths; pos++) {
    const baseTime = measureStart + pos * sixteenthDur;
    const id = Tone.getTransport().schedule((t) => {
      const pattern = getPatternById(currentPatternId);
      if (!pattern) return;  // drums off
      const swingOffset = pattern.swing && pos % 2 === 1 ? sixteenthDur * 0.4 : 0;
      const when = t + swingOffset;
      const kit = pattern.kit ?? 'acoustic';
      for (const [voice, hits] of Object.entries(pattern.voices) as [DrumVoice, DrumHit[]][]) {
        if (!hits) continue;
        for (const hit of hits) {
          if (hitPos(hit) === pos) playDrum(voice, when, hitVel(hit), kit);
        }
      }
    }, baseTime);
    drumScheduledIds.push(id);
  }
}

export function setDrumVolume(linear01: number): void {
  ensureSynths();
  if (drumBus) drumBus.gain.value = linear01;
}
