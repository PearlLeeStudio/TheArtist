/**
 * Play-panel settings resolved against a song's genre: which harmony
 * instrument and comping rhythm, keyboard or guitar voicing, which drum
 * pattern, bass on or off, and master/part levels. Pure — the store, the
 * playback engine, MyPage and the node tests all resolve through here,
 * which is what makes a saved session replay exactly like the editor.
 */
import type {
  Chord,
  DrumMode,
  PlaySettings,
  SessionGenerationContext,
  VisualMode,
} from '../models/types';
import { DRUM_PATTERNS } from './drumPatterns.ts';
import {
  DEFAULT_BASS_INSTRUMENT,
  DEFAULT_BASS_RHYTHM,
  DEFAULT_HARMONY_RHYTHM_ID,
  GENRE_BASS_INSTRUMENT,
  GENRE_BASS_RHYTHM,
  GENRE_HARMONY_INSTRUMENT,
  GENRE_TO_DRUM_PATTERN_ID,
  GUITAR_FAMILY,
  HARMONY_INSTRUMENTS,
  HARMONY_RHYTHM_PRESETS,
} from './genreTables.ts';
import type { HarmonyRhythmPreset } from './genreTables.ts';

export type { DrumMode, PlaySettings, VisualMode } from '../models/types';

/** How the harmony layer voices chords (and how the Visual pane draws them). */
export type VoicingMode = 'piano' | 'guitar';

export const DEFAULT_PLAY_SETTINGS: PlaySettings = {
  harmonyOverride: null,
  harmonyRhythmId: null,
  drumMode: 'auto',
  drumPatternId: null,
  bassEnabled: true,
  visualMode: 'auto',
  masterVolume: 1,
  harmonyVolume: 1,
  bassVolume: 1,
  drumVolume: 0.6,
};

/** Canonical genre key: lower-case; the legacy 'bossa nova' becomes 'bossa'. */
export function normalizeGenre(genre: string | null | undefined): string | null {
  const g = (genre ?? '').trim().toLowerCase();
  if (g === 'bossa nova') return 'bossa';
  return g || null;
}

/** A per-genre table's entry for `genre`, looked up by its canonical key;
 *  undefined without a genre or for one the table lacks. */
function genreEntry<T>(table: Record<string, T>, genre: string | null | undefined): T | undefined {
  const g = normalizeGenre(genre);
  return g ? table[g] : undefined;
}

export function rhythmPresetById(id: string | null | undefined): HarmonyRhythmPreset | null {
  if (!id) return null;
  return HARMONY_RHYTHM_PRESETS.find((p) => p.id === id) ?? null;
}

const BLOCK_PRESET = rhythmPresetById(DEFAULT_HARMONY_RHYTHM_ID) as HarmonyRhythmPreset;

/** The rhythm Auto plays for a genre: the genre's preset, or Block. */
export function autoRhythmPreset(genre: string | null | undefined): HarmonyRhythmPreset {
  const g = normalizeGenre(genre);
  return HARMONY_RHYTHM_PRESETS.find((p) => g !== null && p.genres.includes(g)) ?? BLOCK_PRESET;
}

/** The comping rhythm the harmony layer plays: the picked preset, else Auto. */
export function resolveHarmonyRhythm(settings: PlaySettings, genre: string | null | undefined): HarmonyRhythmPreset {
  return rhythmPresetById(settings.harmonyRhythmId) ?? autoRhythmPreset(genre);
}

/** The genre's default harmony instrument; null without a genre (the engine
 *  then plays its built-in piano). */
export function genreHarmonyInstrument(genre: string | null | undefined): string | null {
  return genreEntry(GENRE_HARMONY_INSTRUMENT, genre) ?? null;
}

/** The harmony instrument that plays: the pick, else the genre default. */
export function resolveHarmonyInstrument(settings: PlaySettings, genre: string | null | undefined): string | null {
  return settings.harmonyOverride ?? genreHarmonyInstrument(genre);
}

/** Keyboard or guitar shapes. The harmony layer voices chords the way the
 *  Visual pane draws them, so what you see is what you hear. */
export function resolveVoicingMode(settings: PlaySettings, genre: string | null | undefined): VoicingMode {
  if (settings.visualMode !== 'auto') return settings.visualMode;
  return GUITAR_FAMILY.has(resolveHarmonyInstrument(settings, genre) ?? '') ? 'guitar' : 'piano';
}

/** The genre's drum pattern id; null without a genre. */
export function genreDrumPatternId(genre: string | null | undefined): number | null {
  return genreEntry(GENRE_TO_DRUM_PATTERN_ID, genre) ?? null;
}

/** The drum pattern that plays; null = no drums. */
export function resolveDrumPatternId(settings: PlaySettings, genre: string | null | undefined): number | null {
  return settings.drumMode === 'manual' ? settings.drumPatternId : genreDrumPatternId(genre);
}

/** The genre's 1-bar bass rhythm (the bass follows the genre; no override). */
export function bassPatternFor(genre: string | null | undefined): number[] {
  return genreEntry(GENRE_BASS_RHYTHM, genre) ?? DEFAULT_BASS_RHYTHM;
}

export function bassInstrumentFor(genre: string | null | undefined): string {
  return genreEntry(GENRE_BASS_INSTRUMENT, genre) ?? DEFAULT_BASS_INSTRUMENT;
}

/** What the playback engine reads when a note fires (playback.ts
 *  applyPlayConfig). The two overrides stay null on Auto so the engine falls
 *  back to the genre of whichever song it is playing. */
export interface EnginePlayConfig {
  harmonyOverride: string | null;
  harmonyRhythmOverride: number[] | null;
  voicingMode: VoicingMode;
  drumPatternId: number | null;
  bassEnabled: boolean;
  masterVolume: number;
  harmonyVolume: number;
  bassVolume: number;
  drumVolume: number;
}

export function enginePlayConfig(settings: PlaySettings, genre: string | null | undefined): EnginePlayConfig {
  return {
    harmonyOverride: settings.harmonyOverride,
    harmonyRhythmOverride: rhythmPresetById(settings.harmonyRhythmId)?.pattern ?? null,
    voicingMode: resolveVoicingMode(settings, genre),
    drumPatternId: resolveDrumPatternId(settings, genre),
    bassEnabled: settings.bassEnabled,
    masterVolume: settings.masterVolume,
    harmonyVolume: settings.harmonyVolume,
    bassVolume: settings.bassVolume,
    drumVolume: settings.drumVolume,
  };
}

/** One strike of a 1-bar rhythm: where it falls and how long it lasts, both in beats. */
export interface HarmonyStrike {
  pos: number;
  dur: number;
}

/**
 * The strikes of a 1-bar rhythm with their lengths. A strike lasts 0.85 ×
 * the pattern's average spacing (sparse patterns sustain, dense ones stab),
 * clamped to 0.85 × the gap before the next strike — wrapping into the next
 * bar — so a chord never runs into its own re-strike. Positions outside the
 * bar are dropped.
 */
export function harmonyStrikes(pattern: number[], beatsPerMeasure: number): HarmonyStrike[] {
  const positions = [...new Set(pattern)]
    .filter((p) => p >= 0 && p < beatsPerMeasure)
    .sort((a, b) => a - b);
  if (positions.length === 0) return [];
  const average = beatsPerMeasure / positions.length;
  return positions.map((pos, i) => {
    const next = i + 1 < positions.length ? positions[i + 1] : positions[0] + beatsPerMeasure;
    return { pos, dur: 0.85 * Math.min(average, next - pos) };
  });
}

/** What a chord slot sounds: root, quality, slash bass and extensions; '' when empty. */
function chordIdentity(chord: Chord | null | undefined): string {
  return chord ? [chord.root, chord.quality, chord.bass ?? '', ...chord.extensions].join('|') : '';
}

/** Sparse presets keep their feel, but never omit a different second chord. */
export function harmonyStrikesForChords(
  pattern: number[], beatsPerMeasure: number, chords: readonly (Chord | null)[],
): HarmonyStrike[] {
  const positions = meterRhythm(pattern, beatsPerMeasure).filter((p) => p >= 0 && p < beatsPerMeasure);
  if (!positions.length) return [];
  const secondSlotStart = beatsPerMeasure / 2;
  const secondChordDiffers = chordIdentity(chords[0]) !== chordIdentity(chords[1]);
  // A slot whose chord the rhythm misses is struck on its first beat; the
  // second slot only for a new chord (a repeated one keeps ringing).
  if (chords[0] && !positions.some((p) => p < secondSlotStart)) positions.push(0);
  if (chords[1] && secondChordDiffers && !positions.some((p) => p >= secondSlotStart)) positions.push(secondSlotStart);
  // A first-slot strike is cut at a chord change.
  return harmonyStrikes(positions, beatsPerMeasure).map((strike) => ({
    ...strike,
    dur: secondChordDiffers && strike.pos < secondSlotStart ? Math.min(strike.dur, secondSlotStart - strike.pos) : strike.dur,
  }));
}

/** BPM is quarter notes/minute; pattern positions use the written meter unit. */
export function secondsPerMeterBeat(bpm: number, denominator: number): number {
  return (60 / bpm) * (4 / denominator);
}

/** Where the second chord slot starts in a 4/4 bar. */
const HALF_BAR_4_4 = 2;

/**
 * A 1-bar rhythm laid on a bar of `beatsPerMeasure` beats. Positions keep
 * their 4/4 beats (harmonyStrikes drops those past a short bar), except in
 * a rhythm that strikes only where the 4/4 chord slots start — Block's
 * [0, 2], the boom-chick bass: those strikes follow the chord slots, so
 * each chord is still struck once in 2/4, 3/4, 5/4, 6/8 and 7/8. 4/4 is
 * unchanged.
 */
export function meterRhythm(pattern: number[], beatsPerMeasure: number): number[] {
  if (pattern.length === 0 || !pattern.every((p) => p === 0 || p === HALF_BAR_4_4)) return pattern;
  return pattern.map((p) => (p === 0 ? 0 : beatsPerMeasure / 2));
}

/** Every beat position any rhythm preset strikes, ascending. The engine
 *  schedules one harmony event per grid position and asks the live rhythm
 *  whether to strike when it fires, so a rhythm switch mid-playback needs no
 *  reschedule (the drum grid works the same way). */
export const HARMONY_GRID: number[] = [
  ...new Set(HARMONY_RHYTHM_PRESETS.flatMap((p) => p.pattern)),
].sort((a, b) => a - b);

/** A rhythm drawn on `stepsPerBeat` cells per beat (16 for a 4/4 bar);
 *  triplet positions light the nearest cell. */
export function patternSteps(pattern: number[], beatsPerMeasure: number, stepsPerBeat = 4): boolean[] {
  const steps = Math.max(1, Math.round(beatsPerMeasure * stepsPerBeat));
  const cells = new Array<boolean>(steps).fill(false);
  for (const p of pattern) {
    if (p < 0 || p >= beatsPerMeasure) continue;
    cells[Math.min(steps - 1, Math.round(p * stepsPerBeat))] = true;
  }
  return cells;
}

const VISUAL_MODES: VisualMode[] = ['auto', 'piano', 'guitar'];
const DRUM_MODES: DrumMode[] = ['auto', 'manual'];
const HARMONY_NAMES = new Set(HARMONY_INSTRUMENTS.map((i) => i.name));
const DRUM_IDS = new Set(DRUM_PATTERNS.map((p) => p.id));

/** Preserve a deliberate mute (0); old or invalid values use the previous mix. */
export function sanitizeVolume(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value)
    ? Math.max(0, Math.min(1, value))
    : fallback;
}

/** `value` when it is one of `allowed`, else `fallback`. */
function oneOf<T>(value: unknown, allowed: readonly T[], fallback: T): T {
  return allowed.includes(value as T) ? (value as T) : fallback;
}

/** PlaySettings from untrusted JSON (localStorage, an imported session file):
 *  every missing, unknown or malformed field falls back to its default. */
export function sanitizePlaySettings(raw: unknown): PlaySettings {
  const saved = (typeof raw === 'object' && raw !== null ? raw : {}) as Record<string, unknown>;
  const defaults = DEFAULT_PLAY_SETTINGS;
  return {
    harmonyOverride: typeof saved.harmonyOverride === 'string' && HARMONY_NAMES.has(saved.harmonyOverride)
      ? saved.harmonyOverride
      : defaults.harmonyOverride,
    harmonyRhythmId: rhythmPresetById(typeof saved.harmonyRhythmId === 'string' ? saved.harmonyRhythmId : null)?.id
      ?? defaults.harmonyRhythmId,
    drumMode: oneOf(saved.drumMode, DRUM_MODES, defaults.drumMode),
    drumPatternId: typeof saved.drumPatternId === 'number' && DRUM_IDS.has(saved.drumPatternId)
      ? saved.drumPatternId
      : defaults.drumPatternId,
    bassEnabled: typeof saved.bassEnabled === 'boolean' ? saved.bassEnabled : defaults.bassEnabled,
    visualMode: oneOf(saved.visualMode, VISUAL_MODES, defaults.visualMode),
    masterVolume: sanitizeVolume(saved.masterVolume, defaults.masterVolume),
    harmonyVolume: sanitizeVolume(saved.harmonyVolume, defaults.harmonyVolume),
    bassVolume: sanitizeVolume(saved.bassVolume, defaults.bassVolume),
    drumVolume: sanitizeVolume(saved.drumVolume, defaults.drumVolume),
  };
}

/**
 * The play settings a saved session replays with. Sessions saved before the
 * Play panel have no drumMode: they replay the way MyPage played them then —
 * their saved drum pattern, else the genre's — with the bass on.
 */
export function playSettingsFromSession(generation: SessionGenerationContext): PlaySettings {
  const settings = sanitizePlaySettings(generation);
  if (generation.drumMode !== undefined) return settings;
  return { ...settings, drumMode: settings.drumPatternId === null ? 'auto' : 'manual' };
}
