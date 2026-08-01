/**
 * Drum patterns — pure data (no Tone/smplr), so the Play panel's step grid,
 * the play settings and the node tests read it without loading audio.
 * drums.ts schedules and plays them. Mirrored in
 * backend/app/services/genre_tables.py (DRUM_PATTERNS).
 */
import { GENRE_TO_DRUM_PATTERN_ID } from './genreTables.ts';

// Voices: Kick, Snare, Hihat (closed), Open hh, Ride, siX-stick (rim),
//         Crash cymbal, Tom-high, tom-Mid, tom-Low (floor)
export type DrumVoice = 'K' | 'S' | 'H' | 'O' | 'R' | 'X' | 'C' | 'T' | 'M' | 'L';

// ---------- Patterns ----------
// Each pattern is keyed by 16th-note grid position within a 4/4 measure (0..15).
// drums.ts plays 4 steps per beat of the song's meter, so a 3/4 bar plays the
// first 12 positions and a 6/8 bar (24 steps) all 16.

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
  /** Beats the off-beat 8ths of every layer land late (drums.ts swingDelay):
   *  1/6 = a triplet 8th, a full shuffle. Omitted = straight. */
  swing?: number;
  /** The genre whose Auto pattern this is (GENRE_TO_DRUM_PATTERN_ID routes
   *  each of the 13 genres to one pattern). The pick-only variants Funk Show
   *  and Samba have none. */
  genre?: string;
  /** Drum kit dispatch (Pearl 2026-05-09).
   *  - 'acoustic' (default, used by every built-in pattern): Aasimonster
   *    multi-mic samples via Tone.Sampler
   *  - '808':  smplr TR-808 (Roland classic)
   *  - 'MFB':  smplr MFB-512 (Berlin analog, punchy)
   *  - 'LM-2': smplr LM-2 (Linn vintage electro)
   *  drums.ts can play the three drum machines, but no pattern picks one. */
  kit?: 'acoustic' | '808' | 'MFB' | 'LM-2';
}

// Helpers to unpack a hit.
export function hitPos(h: DrumHit): number {
  return typeof h === 'number' ? h : h[0];
}
export function hitVel(h: DrumHit): number {
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
    // Half-time HIP-HOP on the Aasimonster acoustic kit (the smplr drum
    // machines sounded thin, clicky or dry here: TR-808, MFB-512, LM-2).
    // Boom-bap shape: snare ONLY on beat 3 (pos 8), sparse kick, ghost
    // cloud — bar feels twice as long because backbeat is singular.
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
    // Modern EDM beat on the Aasimonster acoustic kit (MFB-512 sounded
    // clicky-thin, LM-2 dry, and smplr has no TR-909). The electronic feel
    // comes from the *pattern*, not the kit color: 4-on-floor kick on every
    // quarter + ghost "and-of-1", 16th hats with trap-style accents, ghost
    // snares for groove.
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
    swing: 1 / 6,
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
    swing: 1 / 6,
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
    swing: 0.1,   // a light swing, not a full shuffle
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


export function drumPatternById(id: number | null | undefined): DrumPattern | null {
  if (id === null || id === undefined) return null;
  return DRUM_PATTERNS.find(p => p.id === id) ?? null;
}

/**
 * Resolve the canonical drum pattern for a 13-genre vocab entry
 * (GENRE_TO_DRUM_PATTERN_ID). Returns null for an unknown or null genre.
 */
export function getDrumPatternForGenre(genre: string | null | undefined): DrumPattern | null {
  if (!genre) return null;
  return drumPatternById(GENRE_TO_DRUM_PATTERN_ID[genre]);
}

/** One LED of the Play panel's drum grid: the loudest hit's velocity
 *  (0 = off) and whether that hit is one of the row's secondary voices. */
export interface DrumGridCell {
  level: number;
  alt: boolean;
}

export interface DrumGridRow {
  label: 'K' | 'S' | 'H';
  cells: DrumGridCell[];
}

/** Which voices each of the three grid rows shows: main voice first, the
 *  rest light as alternates, so every voice of every pattern shows up
 *  (Classical has only floor tom + crash). */
const GRID_ROWS: { label: DrumGridRow['label']; voices: DrumVoice[] }[] = [
  { label: 'K', voices: ['K', 'L'] },
  { label: 'S', voices: ['S', 'X', 'T', 'M'] },
  { label: 'H', voices: ['H', 'O', 'R', 'C'] },
];

/** K / S / H rows of `steps` 16th cells for the Drums pane (null = off). */
export function drumGrid(pattern: DrumPattern | null, steps = 16): DrumGridRow[] {
  return GRID_ROWS.map(({ label, voices }) => {
    const cells: DrumGridCell[] = Array.from({ length: steps }, () => ({ level: 0, alt: false }));
    voices.forEach((voice, vi) => {
      for (const hit of pattern?.voices[voice] ?? []) {
        const pos = hitPos(hit);
        const level = hitVel(hit);
        if (pos < 0 || pos >= steps || level <= cells[pos].level) continue;
        cells[pos] = { level, alt: vi > 0 };
      }
    });
    return { label, cells };
  });
}
