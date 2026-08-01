/**
 * Drum playback (Tone.js).
 *
 * Primary path: Tone.Sampler instances loaded from `/drums/*.flac` —
 * samples are DrumGizmo Aasimonster v2.1 (GPL-3.0). One Sampler per voice
 * so each can have its own volume balance and release tail (long for cymbals,
 * short for shells). Attribution + sample provenance: see public/drums/LICENSE.txt.
 *
 * Fallback path: the existing synthesised voices below stay live and are used
 * before samples finish loading, or if loading fails. The `playDrum()` API is
 * unchanged so callers don't care which path is taken.
 *
 * The patterns themselves live in drumPatterns.ts (pure data).
 */
import * as Tone from 'tone';
import { drumPatternById, hitPos, hitVel } from './drumPatterns';
import type { DrumHit, DrumVoice } from './drumPatterns';
import { getMasterInput } from './mixer';
import { setPlaybackStatus } from './playbackStatus.ts';

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
let drumBus: GainNode | null = null;
/** Drum bus gain; kept here so it can be set before the bus exists. */
let drumVolume = 0.6;

// ---------- Electronic / hip-hop voices (smplr real-sample drum machines) ----
// Pearl feedback timeline:
//   round 1 (synth 808)   → too thin
//   round 2 (smplr TR-808) → kept for hip-hop (classic boom-bap)
//   round 3 (LM-2 for EDM) → still missing 비트감
//   round 4 (MFB for EDM)  → smplr only ships 5 vintage machines (no TR-909!).
//     Switched electronic LM-2 → MFB-512 (raw analog Berlin synth, much
//     punchier kick + crisper hat than the dry Linn LM-2). Kept LM-2/CR-8000
//     in the dispatch for future variety.
//   round 5 (2026-05-10)   → all rejected (TR-808 thin, MFB-512 clicky, LM-2
//     dry): every built-in pattern plays the Aasimonster acoustic kit, and
//     none selects a machine (DrumPattern.kit).
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
      ensureSynths();
      const dm = new DrumMachine(audioCtx, { instrument: name, volume: 100, destination: drumBus! });
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
  // → Compressor (tight 4:1 glue for punch without being louder) → master.
  // A native input also routes smplr through the same gain and effects.
  drumBus = (Tone.getContext().rawContext as AudioContext).createGain();
  drumBus.gain.value = drumVolume;
  const hpf = new Tone.Filter({ type: 'highpass', frequency: 50, Q: 0.7 });
  const glue = new Tone.Compressor({
    threshold: -18,
    ratio: 4,
    attack: 0.003,
    release: 0.12,
    knee: 6,
  });
  Tone.connect(drumBus, hpf);
  hpf.chain(glue, getMasterInput());

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
  // TR-808, MFB-512 and LM-2 lazy-load via getDrumMachine() on the first hit
  // of a pattern whose `kit` selects one; no built-in pattern does.

  // ---------- Real samples (DrumGizmo Aasimonster v2.1, GPL-3.0) ----------
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

  const loads: Promise<void>[] = [];
  for (const v of ['K','S','X','H','O','R','C','T','M','L'] as DrumVoice[]) {
    let loaded!: () => void;
    let failed!: (error: Error) => void;
    loads.push(new Promise<void>((resolve, reject) => { loaded = resolve; failed = reject; }));
    const sampler = new Tone.Sampler({
      urls: { C2: SAMPLE_FILE[v] },
      baseUrl: '/drums/',
      release: SAMPLE_RELEASE[v],
      onload: loaded,
      onerror: () => failed(new Error(`The ${SAMPLE_FILE[v].replace('.flac', '')} drum sample could not be loaded. Please reload and try again.`)),
    }).connect(drumBus);
    sampler.volume.value = SAMPLE_VOL_DB[v];
    samplers[v] = sampler;
  }

  // Mark ready once every Sampler has decoded its file. If decoding fails
  // (e.g. files missing in deployment), `samplesReady` stays false and we
  // keep using the synth fallback below — playback never errors.
  samplesLoad = Promise.all(loads)
    .then(() => { samplesReady = true; })
  void samplesLoad.catch((err) => { console.warn('Drum samples failed to load:', err); });
}

// ---------- Real-sample voices ----------
const samplers: Partial<Record<DrumVoice, Tone.Sampler>> = {};
let samplesReady = false;
let samplesLoad: Promise<void> | null = null;

export async function preloadDrums(timeoutMs = 30_000): Promise<void> {
  if (currentPatternId === null || samplesReady) return;
  ensureSynths();
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    await Promise.race([
      samplesLoad,
      new Promise<never>((_resolve, reject) => { timer = setTimeout(() => reject(new Error('Drum samples are still loading. Please try Play again.')), timeoutMs); }),
    ]);
  } finally { clearTimeout(timer); }
}

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
  //   'MFB'  → MFB-512 (Berlin analog, punchy kick + crisp hat)
  //   'LM-2' → Linn LM-2 (vintage drier electro, kept for variety)
  // Pattern.kit selects (no built-in pattern does). Velocity jitter ±8% per
  // trigger avoids machine-gun repetition (same trick we use on the acoustic
  // Sampler kit). Falls through to synth fallback below if the chosen machine
  // isn't loaded yet.
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

// ---------- Live pattern selection ----------
// `currentPatternId` is read at *play* time so the user can toggle drum styles
// (1..6) or turn drums off (null) mid-playback without rescheduling.
let currentPatternId: number | null = null;

export function setCurrentDrumPattern(id: number | null): void {
  currentPatternId = id;
  if (drumBus && id !== null && !samplesReady) {
    setPlaybackStatus({ loading: 'Loading drum samples…', error: null });
    void preloadDrums().then(() => setPlaybackStatus({ loading: null, error: null }), (error: unknown) => {
      setPlaybackStatus({ loading: null, error: error instanceof Error ? error.message : 'Drum samples could not be loaded.' });
    });
  }
}

/**
 * Seconds to delay a note at `beatInBar` while a swing pattern plays: the
 * off-beat 8ths (the "&"s) of every layer — drums, harmony, bass — land
 * together the pattern's swing late (a triplet 8th for Jazz Swing and Blues
 * Shuffle, where the Shuffle comping rhythm strikes; less for Country).
 * Read when the note fires, so a pattern switch or the BPM knob applies
 * from the next note.
 */
export function swingDelay(beatInBar: number, beatUnit = 1): number {
  const swing = drumPatternById(currentPatternId)?.swing ?? 0;
  if (swing <= 0) return 0;
  const offBeat = Math.abs(beatInBar - Math.floor(beatInBar) - 0.5) < 1e-9;
  return offBeat ? (swing * 60 * beatUnit) / Tone.getTransport().bpm.value : 0;
}

// ---------- Scheduling ----------

/** Drum steps per beat of the song's meter: 16ths in x/4, 32nds in x/8. */
export const DRUM_STEPS_PER_BEAT = 4;

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
 * For 4/4: 16 ticks per measure. For 3/4: 12 ticks. etc. A bar entered on
 * its second chord slot schedules from `startStep` on, the first of those
 * steps at `measureStart`.
 */
export function scheduleDrumMeasure(
  measureStart: number,
  secondsPerBeat: number,
  beatsPerMeasure: number,
  startStep = 0,
  beatUnit = 1,
): void {
  ensureSynths();
  const sixteenthDur = secondsPerBeat / DRUM_STEPS_PER_BEAT;
  const totalSixteenths = beatsPerMeasure * DRUM_STEPS_PER_BEAT;

  for (let pos = startStep; pos < totalSixteenths; pos++) {
    const baseTime = measureStart + (pos - startStep) * sixteenthDur;
    const id = Tone.getTransport().schedule((t) => {
      const pattern = drumPatternById(currentPatternId);
      if (!pattern) return;  // drums off
      const when = t + swingDelay(pos / DRUM_STEPS_PER_BEAT, beatUnit);
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

/** Drum bus gain 0..1. Live while playing; before the first play it is only
 *  remembered, so restoring a saved setting at startup creates no audio
 *  nodes ahead of the user's first gesture. */
export function setDrumVolume(linear01: number): void {
  drumVolume = Math.max(0, Math.min(1, linear01));
  if (drumBus) drumBus.gain.value = drumVolume;
}
