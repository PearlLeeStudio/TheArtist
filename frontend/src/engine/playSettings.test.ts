import { test } from 'node:test';
import assert from 'node:assert/strict';
import type { PlaySettings } from '../models/types.ts';
import {
  GENRE_HARMONY_RHYTHM,
  GENRE_TO_DRUM_PATTERN_ID,
  HARMONY_INSTRUMENTS,
  HARMONY_RHYTHM_PRESETS,
} from './genreTables.ts';
import { DRUM_PATTERNS, drumGrid, drumPatternById, getDrumPatternForGenre } from './drumPatterns.ts';
import {
  DEFAULT_PLAY_SETTINGS,
  HARMONY_GRID,
  autoRhythmPreset,
  enginePlayConfig,
  harmonyStrikes,
  meterRhythm,
  normalizeGenre,
  patternSteps,
  playSettingsFromSession,
  rhythmPresetById,
  resolveDrumPatternId,
  resolveHarmonyInstrument,
  resolveHarmonyRhythm,
  resolveVoicingMode,
  sanitizePlaySettings,
} from './playSettings.ts';

const GENRES = [
  'jazz', 'pop', 'rock', 'blues', 'bossa', 'classical', 'country',
  'rnb_soul', 'hip_hop', 'electronic', 'funk', 'folk', 'gospel',
];

const settings = (overrides: Partial<PlaySettings> = {}): PlaySettings => ({
  ...DEFAULT_PLAY_SETTINGS,
  ...overrides,
});

test('the 13 rhythm presets have unique ids and keep every genre rhythm', () => {
  assert.equal(HARMONY_RHYTHM_PRESETS.length, 13);
  assert.equal(new Set(HARMONY_RHYTHM_PRESETS.map((p) => p.id)).size, 13);
  // Each genre belongs to exactly one preset.
  for (const genre of GENRES) {
    assert.equal(HARMONY_RHYTHM_PRESETS.filter((p) => p.genres.includes(genre)).length, 1, genre);
  }
  // The comping rhythms the genres played before the presets existed.
  assert.deepEqual(GENRE_HARMONY_RHYTHM, {
    jazz: [0, 1.5],
    pop: [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5],
    rock: [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5],
    country: [1, 3],
    gospel: [1, 3],
    blues: [0, 2 / 3, 1, 5 / 3, 2, 8 / 3, 3, 11 / 3],
    bossa: [0, 1.5, 3],
    classical: [0, 1, 2, 3],
    folk: [0, 2, 2.5, 3, 3.5],
    rnb_soul: [1.5, 2.5, 3.5],
    hip_hop: [0, 2.5],
    electronic: [0.5, 1.5, 2.5, 3.5],
    funk: [0, 0.5, 2, 2.5],
  });
});

test('Auto rhythm follows the genre, Block without one', () => {
  assert.equal(autoRhythmPreset('jazz').id, 'charleston');
  assert.equal(autoRhythmPreset('Hip_Hop').id, 'boom_bap');
  assert.equal(autoRhythmPreset('bossa nova').id, 'bossa');
  assert.equal(autoRhythmPreset(null).id, 'block');
  assert.equal(autoRhythmPreset('polka').id, 'block');
  assert.equal(normalizeGenre('  '), null);
});

test('a picked rhythm wins over the genre; an unknown id falls back to Auto', () => {
  assert.equal(resolveHarmonyRhythm(settings({ harmonyRhythmId: 'shuffle' }), 'pop').id, 'shuffle');
  assert.equal(resolveHarmonyRhythm(settings({ harmonyRhythmId: 'nope' }), 'pop').id, 'eighth_pulse');
  assert.equal(resolveHarmonyRhythm(settings(), 'funk').id, 'funky_drummer');
});

test('harmony instrument: the pick, else the genre default, else none', () => {
  assert.equal(resolveHarmonyInstrument(settings(), 'jazz'), 'electric_piano_1');
  assert.equal(resolveHarmonyInstrument(settings({ harmonyOverride: 'trumpet' }), 'jazz'), 'trumpet');
  assert.equal(resolveHarmonyInstrument(settings(), null), null);
});

test('voicing follows the instrument family on auto and the pick otherwise', () => {
  assert.equal(resolveVoicingMode(settings(), 'rock'), 'guitar');        // overdriven_guitar
  assert.equal(resolveVoicingMode(settings(), 'pop'), 'piano');
  assert.equal(resolveVoicingMode(settings({ harmonyOverride: 'trumpet' }), 'rock'), 'piano');
  assert.equal(resolveVoicingMode(settings({ harmonyOverride: 'acoustic_guitar_nylon' }), 'pop'), 'guitar');
  assert.equal(resolveVoicingMode(settings({ visualMode: 'piano' }), 'rock'), 'piano');
  assert.equal(resolveVoicingMode(settings({ visualMode: 'guitar' }), 'pop'), 'guitar');
});

test('drums: auto follows the genre, manual keeps the pick or stays off', () => {
  assert.equal(resolveDrumPatternId(settings(), 'pop'), 1);
  assert.equal(resolveDrumPatternId(settings(), 'country'), 12);
  assert.equal(resolveDrumPatternId(settings(), null), null);
  assert.equal(resolveDrumPatternId(settings({ drumMode: 'manual', drumPatternId: 11 }), 'pop'), 11);
  assert.equal(resolveDrumPatternId(settings({ drumMode: 'manual', drumPatternId: null }), 'pop'), null);
  // A manual pick survives a genre change.
  const picked = settings({ drumMode: 'manual', drumPatternId: 4 });
  assert.equal(resolveDrumPatternId(picked, 'rock'), resolveDrumPatternId(picked, 'gospel'));
});

test('the engine config keeps Auto overrides null so the played song decides', () => {
  assert.deepEqual(enginePlayConfig(settings(), 'rock'), {
    harmonyOverride: null,
    harmonyRhythmOverride: null,
    voicingMode: 'guitar',
    drumPatternId: 9,
    bassEnabled: true,
    masterVolume: 1,
    harmonyVolume: 1,
    bassVolume: 1,
    drumVolume: 0.6,
  });
  const cfg = enginePlayConfig(
    settings({ harmonyRhythmId: 'half', harmonyOverride: 'cello', drumMode: 'manual', bassEnabled: false }),
    'rock',
  );
  assert.deepEqual(cfg.harmonyRhythmOverride, [0]);
  assert.equal(cfg.harmonyOverride, 'cello');
  assert.equal(cfg.voicingMode, 'piano');
  assert.equal(cfg.drumPatternId, null);
  assert.equal(cfg.bassEnabled, false);
});

test('strike lengths scale with density and never reach the next strike', () => {
  const close = (a: number, b: number) => Math.abs(a - b) < 1e-9;
  assert.ok(harmonyStrikes(GENRE_HARMONY_RHYTHM.pop, 4).every((s) => close(s.dur, 0.425)));
  const bap = harmonyStrikes([0, 2.5], 4);
  assert.deepEqual(bap.map((s) => s.pos), [0, 2.5]);
  assert.ok(close(bap[0].dur, 1.7) && close(bap[1].dur, 1.275));
  for (const beats of [4, 3, 2, 6]) {
    for (const preset of HARMONY_RHYTHM_PRESETS) {
      const strikes = harmonyStrikes(preset.pattern, beats);
      strikes.forEach((s, i) => {
        const next = i + 1 < strikes.length ? strikes[i + 1].pos : strikes[0].pos + beats;
        assert.ok(s.pos + s.dur < next, `${preset.id} in ${beats} beats overlaps at ${s.pos}`);
      });
    }
  }
  // Positions past a 3/4 bar are dropped; unsorted, duplicated input is fine.
  assert.equal(harmonyStrikes(GENRE_HARMONY_RHYTHM.pop, 3).length, 6);
  assert.deepEqual(harmonyStrikes([2, 0, 2], 4).map((s) => s.pos), [0, 2]);
  assert.deepEqual(harmonyStrikes([], 4), []);
});

test('Block strikes each chord slot once in every meter; beat rhythms keep their 4/4 beats', () => {
  const block = rhythmPresetById('block')?.pattern ?? [];
  // The meters SongSettings offers: 4/4, 3/4, 6/8, 2/4, 5/4, 7/8.
  for (const beats of [4, 3, 6, 2, 5, 7]) {
    const strikes = harmonyStrikes(meterRhythm(block, beats), beats);
    assert.deepEqual(strikes.map((s) => s.pos), [0, beats / 2], `${beats} beats`);
    strikes.forEach((s, i) => assert.ok(s.pos + s.dur < (strikes[i + 1]?.pos ?? beats), `${beats} beats`));
  }
  for (const preset of HARMONY_RHYTHM_PRESETS) {
    assert.deepEqual(meterRhythm(preset.pattern, 4), preset.pattern, `${preset.id} in 4/4`);
    if (preset.id !== 'block') assert.deepEqual(meterRhythm(preset.pattern, 6), preset.pattern, preset.id);
  }
  // Half stays once per bar; the boom-chick bass follows the chord slots too.
  assert.deepEqual(meterRhythm([0], 7), [0]);
  assert.deepEqual(meterRhythm([0, 2], 3), [0, 1.5]);
  assert.deepEqual(meterRhythm([], 6), []);
  // The Rhythm pane draws Block on the second chord's first step.
  const lit = (cells: boolean[]) => cells.flatMap((on, i) => (on ? [i] : []));
  assert.deepEqual(lit(patternSteps(meterRhythm(block, 6), 6)), [0, 12]);
});

test('the harmony grid holds every preset position once, ascending', () => {
  for (const preset of HARMONY_RHYTHM_PRESETS) {
    for (const pos of preset.pattern) assert.ok(HARMONY_GRID.includes(pos), `${preset.id} ${pos}`);
  }
  assert.deepEqual([...HARMONY_GRID].sort((a, b) => a - b), HARMONY_GRID);
  assert.equal(new Set(HARMONY_GRID).size, HARMONY_GRID.length);
});

test('rhythms draw on 16 cells per 4/4 bar, triplets on the nearest cell', () => {
  const lit = (cells: boolean[]) => cells.flatMap((on, i) => (on ? [i] : []));
  assert.deepEqual(lit(patternSteps(GENRE_HARMONY_RHYTHM.blues, 4)), [0, 3, 4, 7, 8, 11, 12, 15]);
  assert.deepEqual(lit(patternSteps([0, 1.5], 4)), [0, 6]);
  assert.equal(patternSteps([0], 3).length, 12);
  assert.deepEqual(lit(patternSteps(GENRE_HARMONY_RHYTHM.pop, 3)), [0, 2, 4, 6, 8, 10]);
});

test('sanitize keeps valid settings and defaults everything else', () => {
  const valid = settings({
    harmonyOverride: 'tenor_sax',
    harmonyRhythmId: 'bossa',
    drumMode: 'manual',
    drumPatternId: 10,
    bassEnabled: false,
    visualMode: 'guitar',
    drumVolume: 0.3,
  });
  assert.deepEqual(sanitizePlaySettings(JSON.parse(JSON.stringify(valid))), valid);
  for (const junk of [null, undefined, 42, 'x', []]) {
    assert.deepEqual(sanitizePlaySettings(junk), DEFAULT_PLAY_SETTINGS);
  }
  assert.deepEqual(
    sanitizePlaySettings({
      harmonyOverride: 'kazoo',
      harmonyRhythmId: 'polka',
      drumMode: 'robot',
      drumPatternId: 99,
      bassEnabled: 'yes',
      visualMode: 'banjo',
      drumVolume: 7,
    }),
    { ...DEFAULT_PLAY_SETTINGS, drumVolume: 1 },
  );
});

test('sessions saved before the Play panel replay the way MyPage played them', () => {
  const legacy = { model: 'ft_f1', genre: 'pop' };
  assert.deepEqual(playSettingsFromSession({ ...legacy, drumPatternId: 7 }), settings({ drumMode: 'manual', drumPatternId: 7 }));
  assert.deepEqual(playSettingsFromSession({ ...legacy, drumPatternId: null }), settings());
  assert.deepEqual(playSettingsFromSession(legacy), settings());
  assert.equal(playSettingsFromSession({ ...legacy, harmonyOverride: 'cello' }).harmonyOverride, 'cello');
  // A current session keeps drums off.
  const off = settings({ drumMode: 'manual', drumPatternId: null, bassEnabled: false, visualMode: 'piano' });
  assert.deepEqual(playSettingsFromSession({ ...legacy, ...off }), off);
});

test('controller levels keep saved mutes and migrate old mixes without changing their volume', () => {
  const oldMix = { model: 'ft_f1', genre: 'pop', drumMode: 'manual' as const, drumPatternId: 4, drumVolume: 0.35 };
  const restored = playSettingsFromSession(oldMix);
  assert.deepEqual(
    [restored.masterVolume, restored.harmonyVolume, restored.bassVolume, restored.drumVolume],
    [1, 1, 1, 0.35],
  );
  const mix = settings({ masterVolume: 0.7, harmonyVolume: 0, bassVolume: 0.4, drumVolume: 0.2 });
  const saved = JSON.parse(JSON.stringify({ ...oldMix, ...mix }));
  assert.deepEqual(playSettingsFromSession(saved), mix);
  for (const genre of ['pop', 'jazz', null]) {
    const config = enginePlayConfig(playSettingsFromSession(saved), genre);
    assert.deepEqual(
      [config.masterVolume, config.harmonyVolume, config.bassVolume, config.drumVolume],
      [0.7, 0, 0.4, 0.2],
    );
  }
  const muted = sanitizePlaySettings({ masterVolume: 0, harmonyVolume: 0, bassVolume: 0, drumVolume: 0 });
  assert.deepEqual([muted.masterVolume, muted.harmonyVolume, muted.bassVolume, muted.drumVolume], [0, 0, 0, 0]);
});

test('controller levels clamp imported values and reject non-finite or non-number values', () => {
  const normalized = sanitizePlaySettings({ masterVolume: -1, harmonyVolume: 2, bassVolume: NaN, drumVolume: Infinity });
  assert.deepEqual(
    [normalized.masterVolume, normalized.harmonyVolume, normalized.bassVolume, normalized.drumVolume],
    [0, 1, 1, 0.6],
  );
  const invalid = sanitizePlaySettings({ masterVolume: '0', harmonyVolume: null, bassVolume: false, drumVolume: {} });
  assert.deepEqual(invalid, DEFAULT_PLAY_SETTINGS);
});

test('every harmony instrument is unique and every genre default is pickable', () => {
  const names = HARMONY_INSTRUMENTS.map((i) => i.name);
  assert.equal(names.length, 31);
  assert.equal(new Set(names).size, 31);
  for (const genre of GENRES) {
    assert.ok(names.includes(resolveHarmonyInstrument(settings(), genre) ?? ''), genre);
  }
});

test('15 drum patterns; every genre routes to its own pattern', () => {
  assert.deepEqual(DRUM_PATTERNS.map((p) => p.id), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
  for (const genre of GENRES) {
    assert.equal(getDrumPatternForGenre(genre)?.genre, genre);
    assert.equal(getDrumPatternForGenre(genre)?.id, GENRE_TO_DRUM_PATTERN_ID[genre]);
  }
  assert.equal(drumPatternById(null), null);
  assert.equal(drumPatternById(99), null);
});

test('the drum grid lights K / S / H with ghosts and alternate voices', () => {
  const [k, s, h] = drumGrid(drumPatternById(1));
  const lit = (row: typeof k) => row.cells.flatMap((c, i) => (c.level > 0 ? [i] : []));
  assert.deepEqual(lit(k), [0, 6, 10]);
  assert.deepEqual(lit(s), [2, 4, 7, 11, 12, 14]);
  assert.equal(s.cells[2].level, 0.22);
  assert.deepEqual(h.cells[14], { level: 0.65, alt: true });   // open hat
  // Classical has only a floor tom and a crash: both still show.
  const [ck, cs, ch] = drumGrid(drumPatternById(15));
  assert.deepEqual(ck.cells[0], { level: 0.6, alt: true });
  assert.deepEqual(ch.cells[0], { level: 0.3, alt: true });
  assert.equal(lit(cs).length, 0);
  // Off, and a 3/4 bar.
  assert.ok(drumGrid(null).every((row) => row.cells.every((c) => c.level === 0)));
  assert.equal(drumGrid(drumPatternById(1), 12)[0].cells.length, 12);
});
