export interface Chord {
  root: string;
  quality: string;
  bass?: string;        // slash chord bass note, e.g. "G" in C/G
  romanNumeral: string;
  voicing: number[];
  extensions: string[];
}

export interface Measure {
  id: string;
  index: number;
  chords: [Chord | null, Chord | null];
}

export interface Song {
  id: string;
  title: string;
  bpm: number;
  key: string;
  genre: string | null;
  timeSignature: [number, number];
  measures: Measure[];
}

export interface SlotAddress {
  measureIndex: number;
  slotIndex: 0 | 1;
}

/**
 * Generation-time settings remembered alongside a saved session — what the
 * user had configured on the Generate panel when they pressed Save.
 * Re-applying these is what makes a saved session function as a "preset"
 * for Favorite Generate.
 */
export interface SessionGenerationContext {
  model: string;            // checkpoint key, e.g. 'ft_f1' (default 2026-05-09)
  genre: string | null;
  /** Drum pattern selection at save time. null = drums off. Pearl 2026-05-09:
   *  bundled so MyPage music-gen reproduces the exact playback config. */
  drumPatternId?: number | null;
  /** User-picked GM harmony instrument name captured at Save time
   *  (Pearl 2026-05-10). null = use the genre default table. Bass has
   *  no override (Pearl owns it); melody layer was dropped. Voyager API
   *  never reads this; only affects MyPage local Play. */
  harmonyOverride?: string | null;
  creativity?: 'conservative' | 'balanced' | 'creative';
  alpha?: number;           // R1 (Sethares) rerank weight, optional override
  beta?: number;            // R2 (Berklee) rerank weight, optional override
}

/**
 * A snapshot saved on the user's MyPage tab. Stored in localStorage; the
 * full song is embedded so loading restores both the score and its
 * generation context. Multi-star is allowed; starred sessions are the
 * pool Favorite Generate draws from.
 */
export interface SavedSession {
  id: string;
  name: string;
  createdAt: number;        // ms epoch, for sort order
  song: Song;
  generation: SessionGenerationContext;
  starred: boolean;
  /** Pre-computed multi-track arrangement (bass + melody) — populated
   *  at save time so MyPage Play schedules instantly without any per-play
   *  computation. Pearl 2026-05-10 round 3: stored as a *1-bar pattern*
   *  (positions in beats) per layer, repeated across all measures at
   *  play time per the song's time signature. Was a flat ArrangementEvent[]
   *  in round 2 — too verbose. Now ~5 fields total per session. */
  arrangement?: Arrangement;
}

/** 1-bar arrangement pattern, repeated per measure at play time —
 *  stored per bar so it repeats to fit the time signature.
 *  Beat positions in `0..beatsPerMeasure`; positions ≥
 *  beatsPerMeasure are silently dropped at expand (handles odd time
 *  signatures gracefully). Melody field dropped 2026-05-10 — 3-track
 *  (harmony + bass + drum), see arrangement.ts header for rationale. */
export interface Arrangement {
  /** 1-bar bass pattern: beat positions where bass strikes the chord
   *  root. e.g. `[0, 1, 2, 3]` (root on every beat) or `[0, 2]`
   *  (boom-chick — bass on 1, 3). */
  bassPattern: number[];
  /** GM instrument resolved from GENRE_BASS_INSTRUMENT at save time.
   *  e.g. "contrabass", "electric_bass_finger". */
  bassInstrument: string;
  /** Time signature numerator at save time. Pattern positions ≥ this
   *  value are dropped at expand. Stored as a sanity check. */
  beatsPerMeasure: number;
}
