import { test } from 'node:test';
import assert from 'node:assert/strict';
import type { PlaySettings } from '../models/types.ts';
import { DEFAULT_PLAY_SETTINGS } from '../engine/playSettings.ts';
import {
  playResumeState,
  setDrumAutoState,
  setDrumPatternState,
  setDrumsEnabledState,
  setHarmonyEnabledState,
  setHarmonyVolumeState,
} from './playMuteState.ts';
import type { PlayMuteState } from './playMuteState.ts';

function state(overrides: Partial<PlaySettings> = {}): PlayMuteState {
  const settings = { ...DEFAULT_PLAY_SETTINGS, ...overrides };
  return { settings, ...playResumeState(settings) };
}

test('Harmony off/on restores the audible level while retaining every other choice', () => {
  const initial = state({
    harmonyVolume: 0.37, harmonyOverride: 'trumpet', harmonyRhythmId: 'shuffle',
    masterVolume: 0.8, bassEnabled: false, bassVolume: 0.2, drumVolume: 0.4,
    drumMode: 'manual', drumPatternId: 2, visualMode: 'guitar',
  });
  const original = structuredClone(initial);
  const muted = setHarmonyEnabledState(initial, false);
  assert.deepEqual(muted.settings, { ...initial.settings, harmonyVolume: 0 });
  assert.equal(muted.lastHarmonyVolume, 0.37);
  const mutedAgain = setHarmonyEnabledState(muted, false);
  assert.equal(mutedAgain.lastHarmonyVolume, 0.37);
  const resumed = setHarmonyEnabledState(mutedAgain, true);
  assert.deepEqual(resumed.settings, initial.settings);
  assert.deepEqual(setHarmonyEnabledState(resumed, true), resumed);
  assert.deepEqual(initial, original);
});

test('nonzero fader changes refresh Harmony resume volume and zero retains it', () => {
  let current = setHarmonyEnabledState(state({ harmonyVolume: 0.61 }), false);
  current = setHarmonyVolumeState(current, 0.24);
  assert.equal(current.settings.harmonyVolume, 0.24);
  assert.equal(current.lastHarmonyVolume, 0.24);
  current = setHarmonyVolumeState(current, 0);
  assert.equal(current.settings.harmonyVolume, 0);
  assert.equal(current.lastHarmonyVolume, 0.24);
  assert.equal(setHarmonyEnabledState(current, true).settings.harmonyVolume, 0.24);
  assert.equal(setHarmonyVolumeState(current, 4).settings.harmonyVolume, 1);
  assert.equal(setHarmonyVolumeState(current, -2).settings.harmonyVolume, 0);
  for (const value of [Number.NaN, Number.POSITIVE_INFINITY]) {
    const sanitized = setHarmonyVolumeState(current, value);
    assert.equal(sanitized.settings.harmonyVolume, DEFAULT_PLAY_SETTINGS.harmonyVolume);
    assert.equal(sanitized.lastHarmonyVolume, DEFAULT_PLAY_SETTINGS.harmonyVolume);
  }
});

test('muted browser preferences roundtrip the prior volume and explicit drum pick', () => {
  let saved = state({ harmonyVolume: 0.43, drumMode: 'manual', drumPatternId: 2, drumVolume: 0.29 });
  saved = setHarmonyEnabledState(setDrumsEnabledState(saved, false), false);
  const raw = JSON.parse(JSON.stringify(saved));
  const reloaded = { settings: raw.settings as PlaySettings, ...playResumeState(raw.settings, raw) };
  assert.equal(setHarmonyEnabledState(reloaded, true).settings.harmonyVolume, 0.43);
  const resumed = setDrumsEnabledState(reloaded, true);
  assert.equal(resumed.settings.drumMode, 'manual');
  assert.equal(resumed.settings.drumPatternId, 2);
  assert.equal(resumed.settings.drumVolume, 0.29);
});

test('live settings are authoritative over stale persisted resume preferences', () => {
  const settings = state({ harmonyVolume: 0.72, drumMode: 'manual', drumPatternId: 1 }).settings;
  assert.deepEqual(playResumeState(settings, {
    lastHarmonyVolume: 0.21, resumeDrum: { drumMode: 'manual', drumPatternId: 2 },
  }), {
    lastHarmonyVolume: 0.72, resumeDrum: { drumMode: 'manual', drumPatternId: 1 },
  });
});

test('invalid muted resume preferences cannot restore silence or an unknown drum id', () => {
  const settings = state({ harmonyVolume: 0, drumMode: 'manual', drumPatternId: null }).settings;
  for (const value of [0, -0.3, Number.NaN, Number.POSITIVE_INFINITY, '0.5', null]) {
    assert.equal(playResumeState(settings, { lastHarmonyVolume: value }).lastHarmonyVolume,
      DEFAULT_PLAY_SETTINGS.harmonyVolume);
  }
  assert.equal(playResumeState(settings, { lastHarmonyVolume: 8 }).lastHarmonyVolume, 1);
  for (const resumeDrum of [
    null, [], 'auto', {},
    { drumMode: 'manual', drumPatternId: null },
    { drumMode: 'manual', drumPatternId: 999 },
    { drumMode: 'auto', drumPatternId: 999 },
    { drumMode: 'manual', drumPatternId: '2' },
    { drumMode: 'other', drumPatternId: 2 },
  ]) {
    assert.deepEqual(playResumeState(settings, { resumeDrum }).resumeDrum,
      { drumMode: 'auto', drumPatternId: null });
  }
});

test('Drums off/on restores Auto or manual choice without changing output gains', () => {
  for (const choice of [
    { drumMode: 'auto' as const, drumPatternId: null },
    { drumMode: 'auto' as const, drumPatternId: 2 },
    { drumMode: 'manual' as const, drumPatternId: 1 },
  ]) {
    const initial = state({ ...choice, harmonyVolume: 0.38, drumVolume: 0.27 });
    const muted = setDrumsEnabledState(initial, false);
    assert.deepEqual(muted.settings, { ...initial.settings, drumMode: 'manual', drumPatternId: null });
    assert.deepEqual(muted.resumeDrum, choice);
    const mutedAgain = setDrumsEnabledState(muted, false);
    assert.deepEqual(mutedAgain.resumeDrum, choice);
    const resumed = setDrumsEnabledState(mutedAgain, true);
    assert.deepEqual(resumed.settings, initial.settings);
    assert.deepEqual(setDrumsEnabledState(resumed, true), resumed);
  }
});

test('explicit Off backs up the previous drum selection and new picks turn Drums on', () => {
  const initial = state({ drumMode: 'manual', drumPatternId: 1, drumVolume: 0.33 });
  const off = setDrumPatternState(initial, null);
  assert.deepEqual(off.resumeDrum, { drumMode: 'manual', drumPatternId: 1 });
  assert.deepEqual(setDrumsEnabledState(off, true).settings, initial.settings);
  const picked = setDrumPatternState(off, 2);
  assert.equal(picked.settings.drumMode, 'manual');
  assert.equal(picked.settings.drumPatternId, 2);
  assert.deepEqual(picked.resumeDrum, { drumMode: 'manual', drumPatternId: 2 });
  const automatic = setDrumAutoState(off);
  assert.equal(automatic.settings.drumMode, 'auto');
  assert.deepEqual(automatic.resumeDrum, { drumMode: 'auto', drumPatternId: null });
  assert.equal(automatic.settings.drumVolume, 0.33);
});

test('unknown drum picks cannot replace the last valid selection', () => {
  const initial = state({ drumMode: 'manual', drumPatternId: 2 });
  for (const id of [0, -1, 999, 1.5, Number.NaN, Number.POSITIVE_INFINITY]) {
    assert.deepEqual(setDrumPatternState(initial, id), initial);
  }
});

test('replacement projects derive fresh resume defaults instead of prior project backups', () => {
  const oldProject = setHarmonyEnabledState(setDrumsEnabledState(state({
    harmonyVolume: 0.18, drumMode: 'manual', drumPatternId: 2,
  }), false), false);
  const newSettings = state({ harmonyVolume: 0, drumMode: 'manual', drumPatternId: null }).settings;
  const replacement = { settings: newSettings, ...playResumeState(newSettings) };
  assert.equal(setHarmonyEnabledState(replacement, true).settings.harmonyVolume, DEFAULT_PLAY_SETTINGS.harmonyVolume);
  assert.deepEqual(setDrumsEnabledState(replacement, true).resumeDrum,
    { drumMode: 'auto', drumPatternId: null });
  assert.equal(oldProject.lastHarmonyVolume, 0.18);
  assert.deepEqual(oldProject.resumeDrum, { drumMode: 'manual', drumPatternId: 2 });
});

test('Harmony gain mute and Drums scheduling mute remain independent', () => {
  const initial = state({ harmonyVolume: 0.55, drumMode: 'manual', drumPatternId: 2 });
  const bothOff = setDrumsEnabledState(setHarmonyEnabledState(initial, false), false);
  const harmonyOn = setHarmonyEnabledState(bothOff, true);
  assert.equal(harmonyOn.settings.harmonyVolume, 0.55);
  assert.equal(harmonyOn.settings.drumMode, 'manual');
  assert.equal(harmonyOn.settings.drumPatternId, null);
  const drumsOn = setDrumsEnabledState(bothOff, true);
  assert.equal(drumsOn.settings.harmonyVolume, 0);
  assert.equal(drumsOn.settings.drumPatternId, 2);
});
