import * as Tone from 'tone';
import type { Song, SlotAddress, Chord, Arrangement } from '../models/types';
import { generatePianoVoicing, generateGuitarVoicing } from './voicingEngine';
import { scheduleDrumMeasure, clearDrumSchedule } from './drums';
import { harmonyInstrumentForGenre, playChordOnInstrument } from './instruments';
import { scheduleArrangement, GENRE_HARMONY_RHYTHM, expandPatternPerSlot } from './arrangement';
import type { Instrument } from '../store/songStore';

/**
 * User-picked GM instrument that replaces the genre's auto-resolved
 * harmony default. `null` falls back to harmonyInstrumentForGenre().
 * Set via setHarmonyOverride() — VoicingViewer mirrors the songStore
 * value into here so the in-flight Transport callback picks the
 * latest at note-trigger time.
 */
let harmonyOverride: string | null = null;
export function setHarmonyOverride(name: string | null): void {
  harmonyOverride = name;
}

// Piano uses real Salamander Grand Piano samples (Yamaha C5, recorded by
// Alexander Holm, CC BY 3.0) hosted on tonejs.github.io. Sounds like an
// acoustic Steinway-class grand instead of the previous sine-wave
// PolySynth, which read as a thin EP. Sampler shares the same
// triggerAttackRelease/releaseAll API as PolySynth, so the rest of the
// pipeline doesn't need to know which one is in play.
let pianoSampler: Tone.Sampler | null = null;
let guitarSynth: Tone.PolySynth | null = null;
let currentInstrument: Instrument = 'piano';
let scheduledEvents: number[] = [];
let isPlaying = false;
let isPaused = false;
let loopEnabled = false;
let currentMeasureIndex = 0;
let scheduledSlots: SlotAddress[] = [];

type PositionCallback = (measureIndex: number, slotIndex: 0 | 1, chord: Chord | null) => void;
let positionCallback: PositionCallback | null = null;

function getPianoSynth(): Tone.Sampler {
  if (!pianoSampler) {
    // One sample every ~minor-third spans the keyboard; Tone.Sampler
    // pitch-shifts to fill the gaps. mp3 is small enough for fast first
    // decode while sounding broadly indistinguishable from the full set.
    pianoSampler = new Tone.Sampler({
      urls: {
        A0: 'A0.mp3',
        C1: 'C1.mp3',
        'D#1': 'Ds1.mp3',
        'F#1': 'Fs1.mp3',
        A1: 'A1.mp3',
        C2: 'C2.mp3',
        'D#2': 'Ds2.mp3',
        'F#2': 'Fs2.mp3',
        A2: 'A2.mp3',
        C3: 'C3.mp3',
        'D#3': 'Ds3.mp3',
        'F#3': 'Fs3.mp3',
        A3: 'A3.mp3',
        C4: 'C4.mp3',
        'D#4': 'Ds4.mp3',
        'F#4': 'Fs4.mp3',
        A4: 'A4.mp3',
        C5: 'C5.mp3',
        'D#5': 'Ds5.mp3',
        'F#5': 'Fs5.mp3',
        A5: 'A5.mp3',
        C6: 'C6.mp3',
        'D#6': 'Ds6.mp3',
        'F#6': 'Fs6.mp3',
        A6: 'A6.mp3',
        C7: 'C7.mp3',
        'D#7': 'Ds7.mp3',
        'F#7': 'Fs7.mp3',
        A7: 'A7.mp3',
        C8: 'C8.mp3',
      },
      baseUrl: 'https://tonejs.github.io/audio/salamander/',
      release: 1,
    }).toDestination();
    pianoSampler.volume.value = -6;
  }
  return pianoSampler;
}

function getGuitarSynth(): Tone.PolySynth {
  if (!guitarSynth) {
    guitarSynth = new Tone.PolySynth(Tone.FMSynth, {
      harmonicity: 3.01,
      modulationIndex: 1.5,
      oscillator: { type: 'triangle' } as any,
      modulation: { type: 'square' } as any,
      envelope: {
        attack: 0.002,
        decay: 1.2,
        sustain: 0.0,
        release: 0.8,
      },
      modulationEnvelope: {
        attack: 0.002,
        decay: 0.4,
        sustain: 0.0,
        release: 0.5,
      },
    }).toDestination();
    guitarSynth.volume.value = -8;
  }
  return guitarSynth;
}

function getSynth(): Tone.Sampler | Tone.PolySynth {
  return currentInstrument === 'guitar' ? getGuitarSynth() : getPianoSynth();
}

function midiToFreq(midi: number): number {
  return 440 * Math.pow(2, (midi - 69) / 12);
}

function clearSchedule(): void {
  scheduledEvents.forEach(id => Tone.getTransport().clear(id));
  scheduledEvents = [];
}

/**
 * Schedule playback for specific measure ranges.
 * If slots is provided, only those slots play. Otherwise plays all measures.
 */
// Pearl 2026-05-09: callback fires when playback ends naturally (song
// completes without loop). Lets the React store sync isPlaying back to
// false so the play/pause button icon reflects reality. Without this,
// audio stops but the button still shows "pause" until user interacts.
let endCallback: (() => void) | null = null;

export function schedulePlayback(
  song: Song,
  onPosition: PositionCallback,
  slots?: SlotAddress[],
  instrument?: Instrument,
  onEnd?: () => void,
  /** Pre-computed multi-track arrangement (1-bar pattern + instruments).
   *  When provided, the bass + melody layers schedule on top of harmony
   *  + drum. Pearl 2026-05-10 round 3 — used by MyPage's Play button.
   *  Editor's regular play leaves this undefined → harmony + drum only. */
  arrangement?: Arrangement,
): void {
  currentInstrument = instrument ?? 'piano';
  stopPlayback();
  positionCallback = onPosition;
  endCallback = onEnd ?? null;

  const bpm = song.bpm || 80;
  Tone.getTransport().bpm.value = bpm;

  const beatsPerMeasure = song.timeSignature[0];
  const beatsPerSlot = beatsPerMeasure / 2;
  const secondsPerBeat = 60 / bpm;
  const slotDuration = beatsPerSlot * secondsPerBeat;

  // Determine which slots to play
  if (slots && slots.length > 0) {
    scheduledSlots = slots;
  } else {
    scheduledSlots = [];
    for (let mi = 0; mi < song.measures.length; mi++) {
      scheduledSlots.push({ measureIndex: mi, slotIndex: 0 });
      scheduledSlots.push({ measureIndex: mi, slotIndex: 1 });
    }
  }

  // Capture genre at schedule time. Reading song.genre directly (instead
  // of a module-level global like the old `playbackGenre`) keeps the
  // rhythm pattern lookup correct across Vite HMR, which would otherwise
  // reset the global to null and silently route every genre to the
  // [0, 2] fallback until the user re-picked the genre.
  const songGenre = (song.genre ?? '').toLowerCase();

  let time = 0;
  const scheduledDrumMeasures = new Set<number>();

  for (const slot of scheduledSlots) {
    const measure = song.measures[slot.measureIndex];
    if (!measure) continue;
    const chord = measure.chords[slot.slotIndex];
    const slotTime = time;

    // Schedule drums at the start of each distinct measure we encounter
    if (!scheduledDrumMeasures.has(slot.measureIndex) && slot.slotIndex === 0) {
      scheduleDrumMeasure(slotTime, secondsPerBeat, beatsPerMeasure);
      scheduledDrumMeasures.add(slot.measureIndex);
    }

    // Position callback
    const eventId = Tone.getTransport().schedule(() => {
      currentMeasureIndex = slot.measureIndex;
      positionCallback?.(slot.measureIndex, slot.slotIndex, chord);
    }, slotTime);
    scheduledEvents.push(eventId);

    // Play chord — Pearl 2026-05-10: harmony follows the genre's comping
    // rhythm pattern (was dumb block-on-slot before). Pattern is expanded
    // per-slot so sparse 1-bar patterns (jazz Charleston `[0, 1.5]`,
    // blues `[0]`) still strike both chord slots by mirroring slot 0's
    // sub-pattern into slot 1. Strike duration auto-scales with density.
    if (chord) {
      const chordRef = chord;
      const harmonyPattern = GENRE_HARMONY_RHYTHM[songGenre] ?? [0, 2];
      const expanded = expandPatternPerSlot(harmonyPattern, beatsPerSlot, beatsPerMeasure);
      const positionsForThisSlot = expanded.filter(e => e.slotIdx === slot.slotIndex);
      const slotStartBeat = slot.slotIndex * beatsPerSlot;
      const strikeDur = (beatsPerMeasure / Math.max(1, expanded.length)) * secondsPerBeat * 0.85;
      for (const { pos } of positionsForThisSlot) {
        const triggerTime = slotTime + (pos - slotStartBeat) * secondsPerBeat;
        const noteEventId = Tone.getTransport().schedule((t) => {
          const OPEN_STRINGS = [40, 45, 50, 55, 59, 64];
          let voicing: number[];
          if (currentInstrument === 'guitar') {
            const frets = generateGuitarVoicing(chordRef);
            voicing = frets
              .map((f, i) => f >= 0 ? OPEN_STRINGS[i] + f : -1)
              .filter(m => m >= 0);
          } else {
            voicing = chordRef.voicing.length > 0 ? chordRef.voicing : generatePianoVoicing(chordRef);
          }
          const sfInstrument = harmonyOverride ?? harmonyInstrumentForGenre(songGenre);
          if (sfInstrument && playChordOnInstrument(sfInstrument, voicing, t, strikeDur, 0.7)) {
            // smplr fired
          } else {
            const synth = getSynth();
            const isSampler = (synth as { loaded?: boolean }).loaded !== undefined;
            if (isSampler && !(synth as { loaded?: boolean }).loaded) {
              return;
            }
            const notes = voicing.map(m => midiToFreq(m));
            try {
              synth.triggerAttackRelease(notes, strikeDur, t);
            } catch (err) {
              console.warn('chord trigger failed:', err);
            }
          }
        }, triggerTime);
        scheduledEvents.push(noteEventId);
      }
    }

    time += slotDuration;
  }

  // Multi-track arrangement (bass + melody) — Pearl 2026-05-10. Only
  // schedules when the caller explicitly passes events (MyPage Play
  // button). Editor's regular play stays harmony + drum only — Pearl
  // can opt the editor in later if desired.
  if (arrangement) {
    scheduleArrangement(arrangement, song, 0, secondsPerBeat, scheduledEvents);
  }

  // End event — Tone v15 warns if scheduled callbacks call schedule-affecting
  // APIs (e.g. Transport.stop) without forwarding the passed-in `t`. Use the
  // precise scheduled time for the stop so timing stays sample-accurate.
  const endId = Tone.getTransport().schedule((t) => {
    if (loopEnabled) {
      Tone.getTransport().seconds = 0;
    } else {
      stopPlayback(t);
      if (scheduledSlots.length > 0) {
        positionCallback?.(scheduledSlots[0].measureIndex, 0, null);
      }
      // Notify caller that playback ended naturally (button icon sync).
      endCallback?.();
    }
  }, time);
  scheduledEvents.push(endId);
}

export async function startPlayback(): Promise<void> {
  await Tone.start();
  Tone.getTransport().start();
  isPlaying = true;
  isPaused = false;
}

export function pausePlayback(): void {
  Tone.getTransport().pause();
  isPlaying = false;
  isPaused = true;
}

export function resumePlayback(): void {
  Tone.getTransport().start();
  isPlaying = true;
  isPaused = false;
}

export function stopPlayback(time?: number): void {
  // When called from inside a Transport callback, `time` should be the
  // scheduled-event time so Tone v15 doesn't warn about inaccurate timing.
  // External callers (UI buttons) pass nothing → undefined defers to "now".
  Tone.getTransport().stop(time);
  Tone.getTransport().seconds = 0;
  clearSchedule();
  clearDrumSchedule();
  isPlaying = false;
  isPaused = false;
  currentMeasureIndex = 0;
}

export function getIsPlaying(): boolean {
  return isPlaying;
}

export function getIsPaused(): boolean {
  return isPaused;
}

export function setLoopEnabled(enabled: boolean): void {
  loopEnabled = enabled;
}

export function getLoopEnabled(): boolean {
  return loopEnabled;
}

export function getCurrentMeasureIndex(): number {
  return currentMeasureIndex;
}

export function setCurrentInstrument(instrument: Instrument): void {
  currentInstrument = instrument;
}

/** Update transport BPM live (takes effect for both playing and paused transport). */
export function setLiveBpm(bpm: number): void {
  Tone.getTransport().bpm.value = bpm;
}
