/**
 * Pure Play-state transitions behind the Harmony and Drums on/off toggles.
 * Turning a part off keeps what it played (lastHarmonyVolume, resumeDrum),
 * so turning it on again restores that instead of a default. playStore
 * applies them to its state.
 */
import type { PlaySettings } from '../models/types.ts';
import { DRUM_PATTERNS } from '../engine/drumPatterns.ts';
import { DEFAULT_PLAY_SETTINGS, sanitizeVolume } from '../engine/playSettings.ts';

export type ResumeDrum = Pick<PlaySettings, 'drumMode' | 'drumPatternId'>;

/** UI-only restore values; they are not part of the saved song's audio settings. */
export interface PlayResumeState {
  lastHarmonyVolume: number;
  resumeDrum: ResumeDrum;
}

export interface PlayMuteState extends PlayResumeState {
  settings: PlaySettings;
}

const DRUM_IDS = new Set(DRUM_PATTERNS.map((pattern) => pattern.id));
const DEFAULT_DRUM: ResumeDrum = { drumMode: 'auto', drumPatternId: null };
/** Manual mode without a pattern is how PlaySettings say "no drums". */
const DRUMS_OFF: ResumeDrum = { drumMode: 'manual', drumPatternId: null };

/** Untrusted JSON as a record; anything that is not an object reads as {}. */
function asRecord(raw: unknown): Record<string, unknown> {
  return (typeof raw === 'object' && raw !== null ? raw : {}) as Record<string, unknown>;
}

/** A Harmony level worth restoring: silence or an invalid value becomes the default. */
function audibleVolume(value: unknown): number {
  const volume = sanitizeVolume(value, DEFAULT_PLAY_SETTINGS.harmonyVolume);
  return volume > 0 ? volume : DEFAULT_PLAY_SETTINGS.harmonyVolume;
}

function sanitizeResumeDrum(raw: unknown): ResumeDrum {
  const { drumMode, drumPatternId } = asRecord(raw);
  const validId = typeof drumPatternId === 'number' && DRUM_IDS.has(drumPatternId);
  if (drumMode === 'auto' && (drumPatternId === null || validId)) {
    return { drumMode: 'auto', drumPatternId: validId ? drumPatternId : null };
  }
  if (drumMode === 'manual' && validId) {
    return { drumMode: 'manual', drumPatternId };
  }
  return { ...DEFAULT_DRUM };
}

/** Harmony off is a zero gain; its instrument and rhythm stay picked. */
function harmonyEnabled(settings: PlaySettings): boolean {
  return settings.harmonyVolume > 0;
}

function drumsEnabled(settings: PlaySettings): boolean {
  return settings.drumMode !== 'manual' || settings.drumPatternId !== null;
}

/** Play `choice` now and keep it as the choice Drums on restores. */
function withDrumChoice(state: PlayMuteState, choice: ResumeDrum): PlayMuteState {
  return { ...state, resumeDrum: choice, settings: { ...state.settings, ...choice } };
}

/** A project replacement passes no raw backup, so restore values cannot leak
 *  from the previously open project. Browser reload may restore validated prefs. */
export function playResumeState(settings: PlaySettings, raw?: unknown): PlayResumeState {
  const saved = asRecord(raw);
  return {
    lastHarmonyVolume: harmonyEnabled(settings)
      ? settings.harmonyVolume : audibleVolume(saved.lastHarmonyVolume),
    resumeDrum: sanitizeResumeDrum(drumsEnabled(settings) ? settings : saved.resumeDrum),
  };
}

export function setHarmonyEnabledState(state: PlayMuteState, on: boolean): PlayMuteState {
  if (on === harmonyEnabled(state.settings)) return state;
  const lastHarmonyVolume = audibleVolume(on ? state.lastHarmonyVolume : state.settings.harmonyVolume);
  return {
    ...state, lastHarmonyVolume,
    settings: { ...state.settings, harmonyVolume: on ? lastHarmonyVolume : 0 },
  };
}

/** The Harmony fader; a nonzero level also becomes the one Harmony on restores. */
export function setHarmonyVolumeState(state: PlayMuteState, volume: number): PlayMuteState {
  const harmonyVolume = sanitizeVolume(volume, DEFAULT_PLAY_SETTINGS.harmonyVolume);
  if (harmonyVolume === state.settings.harmonyVolume) return state;
  return {
    ...state,
    lastHarmonyVolume: harmonyVolume > 0 ? harmonyVolume : state.lastHarmonyVolume,
    settings: { ...state.settings, harmonyVolume },
  };
}

export function setDrumsEnabledState(state: PlayMuteState, on: boolean): PlayMuteState {
  if (on === drumsEnabled(state.settings)) return state;
  const resumeDrum = sanitizeResumeDrum(on ? state.resumeDrum : state.settings);
  return {
    ...state, resumeDrum,
    settings: { ...state.settings, ...(on ? resumeDrum : DRUMS_OFF) },
  };
}

export function setDrumAutoState(state: PlayMuteState): PlayMuteState {
  return withDrumChoice(state, sanitizeResumeDrum({ drumMode: 'auto', drumPatternId: state.settings.drumPatternId }));
}

/** A fixed pattern, or null for Drums off; an unknown id changes nothing. */
export function setDrumPatternState(state: PlayMuteState, id: number | null): PlayMuteState {
  if (id === null) return setDrumsEnabledState(state, false);
  if (!DRUM_IDS.has(id)) return state;
  return withDrumChoice(state, { drumMode: 'manual', drumPatternId: id });
}
