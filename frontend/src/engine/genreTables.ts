/**
 * Per-genre playback tables and the Play-panel presets. Pure data (no
 * Tone/smplr) so the store, the playback engine, MyPage and the node tests
 * all read one copy.
 *
 * Mirrored in backend/app/services/genre_tables.py, which renders the same
 * tracks for /api/generate/song; backend/tests/test_genre_tables_parity.py
 * fails when the two sides drift.
 */

/** A named 1-bar comping rhythm for the harmony layer. Positions are beats
 *  in a 4/4 bar; positions at or past the bar length are dropped (3/4,
 *  2/4). A chord slot with no position landing in it is not struck. Block,
 *  which strikes only where chord slots start, follows the chord slots in
 *  other meters (playSettings.ts meterRhythm). */
export interface HarmonyRhythmPreset {
  id: string;
  label: string;
  /** Key-cap label in the Play panel */
  short: string;
  pattern: number[];
  /** Genres whose Auto rhythm this is */
  genres: string[];
}

/**
 * The genre comping rhythms by name, plus Block and Half. Harmony absorbed
 * the rhythmic-variety role when the melody track was dropped (2026-05-10):
 * each pattern is the most-recognized comping rhythm for its genres. Strike
 * length follows the pattern (playSettings.ts harmonyStrikes).
 */
export const HARMONY_RHYTHM_PRESETS: HarmonyRhythmPreset[] = [
  // Count Basie / Red Garland — beat 1 + "& of 2".
  { id: 'charleston', label: 'Charleston', short: 'Charl', pattern: [0, 1.5], genres: ['jazz'] },
  // Beatles / Coldplay strum, AC/DC / Foo Fighters driving 8ths. Checked
  // 2026-05-10 against POP909 (909 songs, 73,290 bars): the top 8 hit
  // positions matched this pattern at 0.67–0.93; 16th positions clustered
  // at ~0.21, under the 0.30 threshold. See ai/training/rhythm/extract_pop909.py.
  { id: 'eighth_pulse', label: '8th pulse', short: '8ths', pattern: [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5], genres: ['pop', 'rock'] },
  // Country boom-chick / gospel organ backbeat — chord on 2 & 4 (bass on 1 & 3).
  { id: 'boom_chick', label: 'Boom-chick', short: 'Chick', pattern: [1, 3], genres: ['country', 'gospel'] },
  // 12-bar shuffle (Stevie Ray Vaughan / Albert King) — triplet stab on every
  // beat + "& of the triplet"; chosen over the slow-blues whole-bar sustain
  // so the genre is audible against the harmony stabs.
  { id: 'shuffle', label: 'Shuffle', short: 'Shufl', pattern: [0, 2 / 3, 1, 5 / 3, 2, 8 / 3, 3, 11 / 3], genres: ['blues'] },
  // Bossa nova — 1-bar approximation of the 3-2 clave's first half.
  { id: 'bossa', label: 'Bossa', short: 'Bossa', pattern: [0, 1.5, 3], genres: ['bossa'] },
  // Chamber / hymn style — chord on every beat (4-part chorale).
  { id: 'chorale', label: 'Chorale', short: 'Chorl', pattern: [0, 1, 2, 3], genres: ['classical'] },
  // Joni Mitchell / Bill Withers pocket — half + 4 eighths.
  { id: 'folk_pocket', label: 'Folk pocket', short: 'Folk', pattern: [0, 2, 2.5, 3, 3.5], genres: ['folk'] },
  // D'Angelo / J Dilla — offbeat-only, behind-the-beat feel.
  { id: 'neo_soul_offbeat', label: 'Neo-soul offbeat', short: 'Neo', pattern: [1.5, 2.5, 3.5], genres: ['rnb_soul'] },
  // Boom-bap / trap drop — beat 1 + "& of 3".
  { id: 'boom_bap', label: 'Boom-bap', short: 'Bap', pattern: [0, 2.5], genres: ['hip_hop'] },
  // House / EDM sidechain pump — offbeats only (drop the 1).
  { id: 'sidechain_offbeat', label: 'Sidechain offbeat', short: 'Pump', pattern: [0.5, 1.5, 2.5, 3.5], genres: ['electronic'] },
  // JB "Funky Drummer" anticipation — two 8th clusters around beats 1 and 3.
  { id: 'funky_drummer', label: 'Funky drummer', short: 'Funky', pattern: [0, 0.5, 2, 2.5], genres: ['funk'] },
  // One held strike per chord slot — what plays when no genre is set.
  { id: 'block', label: 'Block', short: 'Block', pattern: [0, 2], genres: [] },
  // Half the strike rate of Block: one strike per bar on beat 1, so a bar's
  // second chord is not struck.
  { id: 'half', label: 'Half time', short: 'Half', pattern: [0], genres: [] },
];

/** Rhythm used when the song has no genre (or one outside the 13). */
export const DEFAULT_HARMONY_RHYTHM_ID = 'block';

/** Genre → Auto comping rhythm, derived from the presets above. */
export const GENRE_HARMONY_RHYTHM: Record<string, number[]> = Object.fromEntries(
  HARMONY_RHYTHM_PRESETS.flatMap((p) => p.genres.map((g) => [g, p.pattern])),
);

/** Genre → 1-bar bass pattern (beat positions where the bass plays the root). */
export const GENRE_BASS_RHYTHM: Record<string, number[]> = {
  pop:        [0, 1, 2, 3],
  country:    [0, 2],          // boom-chick: bass on 1 & 3
  rock:       [0, 1, 2, 3],
  blues:      [0, 1, 2, 3],
  bossa:      [0, 1.5],        // 1-bar bossa bass (was 2-bar [0, 1.5, 4, 5.5])
  classical:  [0],
  folk:       [0, 1, 2, 3],
  rnb_soul:   [0, 1, 2, 3],
  hip_hop:    [0, 3],
  electronic: [0, 1, 2, 3],    // 4-on-floor sub
  funk:       [0, 0.5, 2, 2.5],
  gospel:     [0, 2],          // boom-chick: bass on 1 & 3
  jazz:       [0, 1, 2, 3],    // walking deferred to v2; root for now
};

/** Bass pattern when the song has no genre. */
export const DEFAULT_BASS_RHYTHM: number[] = [0, 1, 2, 3];

/**
 * 13-genre → harmony-layer GM instrument (2026-05-09). Artist owns the
 * genre → instrument decision; Voyager only sends the genre.
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
  gospel:     'drawbar_organ',        // Hammond-style drawbar organ (GM 16)
};

/**
 * 13-genre → bass-layer GM instrument (2026-05-10 R4). Two organic options:
 * orchestral contrabass (upright) or electric bass (fingerstyle), following
 * each genre's recording convention. The bass has no user override.
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
  electronic: 'electric_bass_finger', // synth-bass would be ideal; staying organic
  funk:       'electric_bass_finger', // funk = slap eventually; for now fingerstyle root
  folk:       'contrabass',           // bluegrass / traditional folk = upright
  gospel:     'electric_bass_finger', // modern gospel = electric
};

/** Bass instrument when the song has no genre. */
export const DEFAULT_BASS_INSTRUMENT = 'electric_bass_finger';

/**
 * 13-genre → canonical drum pattern id (drumPatterns.ts). The variants
 * Funk Show (10) and Samba (11) are pick-only, never an Auto route.
 */
export const GENRE_TO_DRUM_PATTERN_ID: Record<string, number> = {
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

export type InstrumentFamily = 'keys' | 'guitar' | 'strings' | 'winds' | 'reeds' | 'brass';

export interface HarmonyInstrument {
  /** FluidR3_GM Soundfont key (instruments.ts loads it on first use) */
  name: string;
  label: string;
  family: InstrumentFamily;
}

/**
 * The 31 harmony instruments the user can pick instead of the genre default.
 * Polyphonic-friendly ones (piano, guitar, strings, brass section, choir)
 * sound natural; monophonic ones (solo violin, clarinet, …) play chords too —
 * fine for sketching, odd as a final mix. Every genre default is in here.
 */
export const HARMONY_INSTRUMENTS: HarmonyInstrument[] = [
  { name: 'acoustic_grand_piano',  label: 'Piano',    family: 'keys' },
  { name: 'electric_piano_1',      label: 'E.Piano',  family: 'keys' },
  { name: 'drawbar_organ',         label: 'Organ',    family: 'keys' },
  { name: 'clavinet',              label: 'Clav',     family: 'keys' },
  { name: 'pad_2_warm',            label: 'Pad',      family: 'keys' },
  { name: 'choir_aahs',            label: 'Choir',    family: 'keys' },
  { name: 'acoustic_guitar_nylon', label: 'Nylon',    family: 'guitar' },
  { name: 'acoustic_guitar_steel', label: 'Steel',    family: 'guitar' },
  { name: 'electric_guitar_clean', label: 'Clean',    family: 'guitar' },
  { name: 'overdriven_guitar',     label: 'Drive',    family: 'guitar' },
  { name: 'distortion_guitar',     label: 'Dist',     family: 'guitar' },
  { name: 'violin',                label: 'Violin',   family: 'strings' },
  { name: 'viola',                 label: 'Viola',    family: 'strings' },
  { name: 'cello',                 label: 'Cello',    family: 'strings' },
  { name: 'string_ensemble_1',     label: 'Strings',  family: 'strings' },
  { name: 'pizzicato_strings',     label: 'Pizz',     family: 'strings' },
  { name: 'orchestral_harp',       label: 'Harp',     family: 'strings' },
  { name: 'flute',                 label: 'Flute',    family: 'winds' },
  { name: 'clarinet',              label: 'Clarinet', family: 'winds' },
  { name: 'oboe',                  label: 'Oboe',     family: 'winds' },
  { name: 'bassoon',               label: 'Bassoon',  family: 'winds' },
  { name: 'soprano_sax',           label: 'S.Sax',    family: 'reeds' },
  { name: 'alto_sax',              label: 'A.Sax',    family: 'reeds' },
  { name: 'tenor_sax',             label: 'T.Sax',    family: 'reeds' },
  { name: 'baritone_sax',          label: 'B.Sax',    family: 'reeds' },
  { name: 'trumpet',               label: 'Trumpet',  family: 'brass' },
  { name: 'trombone',              label: 'Trombone', family: 'brass' },
  { name: 'french_horn',           label: 'F.Horn',   family: 'brass' },
  { name: 'tuba',                  label: 'Tuba',     family: 'brass' },
  { name: 'brass_section',         label: 'Brass',    family: 'brass' },
  { name: 'muted_trumpet',         label: 'MutedTpt', family: 'brass' },
];

export const BASS_INSTRUMENT_LABELS: Record<string, string> = {
  contrabass: 'Upright',
  electric_bass_finger: 'E.Bass',
};

/** Harmony instruments whose chords are voiced and drawn as guitar shapes. */
export const GUITAR_FAMILY = new Set<string>([
  'acoustic_guitar_nylon',
  'acoustic_guitar_steel',
  'electric_guitar_jazz',
  'electric_guitar_clean',
  'electric_guitar_muted',
  'overdriven_guitar',
  'distortion_guitar',
  'guitar_harmonics',
]);

/** Display label for a harmony or bass GM name; the raw name when unknown. */
export function instrumentLabel(name: string | null | undefined): string {
  if (!name) return '—';
  return HARMONY_INSTRUMENTS.find((i) => i.name === name)?.label
    ?? BASS_INSTRUMENT_LABELS[name]
    ?? name;
}
