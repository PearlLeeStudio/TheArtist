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

/** Score import (a PDF or image read into a song; see Song.importSource).
 *  'chord_symbols' transcribes printed chord symbols only; 'infer_harmony'
 *  may also propose chords from the notes. */
export type ScoreImportMode = 'chord_symbols' | 'infer_harmony';
/** Where a bar's chords came from: printed symbols, a proposal read from
 *  the notes, or nothing readable. */
export type ImportedChordBasis = 'printed' | 'inferred' | 'unreadable';
export type ImportConfidence = 'high' | 'medium' | 'low';
/** Who read the score. An import writes 'local' (this app's own reader); 'meta'
 *  and 'mixed' remain only in projects saved by earlier builds. */
export type ScoreAnalysisSource = 'local' | 'meta' | 'mixed';
/** Chords and minimal reading provenance; source pitches and prose are discarded. */
export interface ImportMeasureSource {
  chordBasis: ImportedChordBasis;
  confidence: ImportConfidence;
  /** Legacy compatibility field, cleared when an import enters the editor/storage. */
  evidence: string;
  /** Initial recognition/proposal, retained after the user edits the score. */
  originalChords: string[];
  /** Application-written review status, without source note evidence. */
  editorReview?: string;
  /** The bar's position in the source's written order. */
  sourceMeasureIndex?: number;
  /** Who read this bar. An import writes 'local', a model of null (or 'ft_f1' for
   *  own-model sparse harmonization) and a null requestId; 'meta', other model
   *  names and request ids remain only in projects saved by earlier builds. */
  analysisSource?: 'local' | 'meta';
  model?: string | null;
  requestId?: string | null;
  /** Legacy fields accepted on load only so they can be removed during migration. */
  notes?: string[];
  writtenNotes?: string[];
  bassNotes?: string[];
  /** The source bar itself is a rest or N.C. */
  originalNoChord?: boolean;
  /** The user explicitly entered N.C. for a bar that had no source rest/N.C. */
  userNoChord?: boolean;
  timeSignature?: [number, number] | null;
}

/** Song-form section ids. Mirrors backend/app/models/sections.py; labels,
 *  dropdown groups and default chord counts live in engine/songForm.ts. */
export type SectionType =
  | 'intro' | 'verse' | 'pre_chorus' | 'chorus' | 'post_chorus' | 'bridge' | 'outro'
  | 'interlude' | 'solo' | 'breakdown' | 'drop' | 'vamp' | 'turnaround' | 'tag'
  | 'head' | 'a_section' | 'b_section';

/** A song-form section laid over a run of measures. */
export interface SongSection {
  id: string;
  /** null keeps an uncertain imported region without inventing Verse/Chorus. */
  type: SectionType | null;
  label?: string;
  origin?: 'explicit' | 'inferred' | 'manual';
  startMeasure: number;  // index of its first measure
  length: number;        // measures (bars)
  /** Set while "New from Song Form" is still generating the section, which
   *  it places before the request: how many measures were appended for it.
   *  A song loaded with a pending section drops it and those measures
   *  (withSongFormDefaults in songForm.ts). */
  pending?: { addedMeasures: number };
}

export interface Song {
  id: string;
  title: string;
  bpm: number;
  key: string;
  genre: string | null;
  timeSignature: [number, number];
  measures: Measure[];
  /** Song-form sections in measure order. Missing on songs saved before
   *  sections existed; songStore fills in []. */
  sections?: SongSection[];
  /** true while the title follows the sections (see autoTitle in
   *  songForm.ts); editing the title turns it off. Missing on older songs,
   *  where a title of '' or 'Untitled' counts as automatic. */
  autoTitle?: boolean;
  /** Chord reading provenance only; source filenames, bytes and note evidence are discarded. */
  importSource?: {
    filename: string; // Empty; retained for compatibility with older project files.
    model: string | null;
    requestId: string | null; // Null for every import; retained for the same compatibility.
    mode?: ScoreImportMode;
    analysisSource?: ScoreAnalysisSource;
    /** Source-bar order at import time; later score edits do not rewrite it. */
    measures?: ImportMeasureSource[];
  };
}

export interface SlotAddress {
  measureIndex: number;
  slotIndex: 0 | 1;
}

/** Keyboard or guitar shapes in the Play panel's Visual pane; auto follows
 *  the harmony instrument (guitars → guitar). */
export type VisualMode = 'auto' | 'piano' | 'guitar';

/** auto: the song genre's drum pattern; manual: drumPatternId (null = off). */
export type DrumMode = 'auto' | 'manual';

/**
 * The Play panel's sound settings, persisted under `theartist-play` and
 * saved with every session so MyPage replays a session the way the editor
 * played it. Resolution against a genre lives in engine/playSettings.ts.
 */
export interface PlaySettings {
  /** User-picked GM harmony instrument (one of the 31 in genreTables.ts);
   *  null = the genre default. The bass has no override. */
  harmonyOverride: string | null;
  /** HARMONY_RHYTHM_PRESETS id; null = the genre's comping rhythm (Auto). */
  harmonyRhythmId: string | null;
  drumMode: DrumMode;
  /** Pattern id (drumPatterns.ts) used while drumMode is manual; null = off. */
  drumPatternId: number | null;
  /** Bass layer (chord root on the genre's bass rhythm) on or off. */
  bassEnabled: boolean;
  visualMode: VisualMode;
  /** Overall output gain 0..1, applied after the individual parts. */
  masterVolume: number;
  /** Harmony and bass gains 0..1, independent of instrument and rhythm. */
  harmonyVolume: number;
  bassVolume: number;
  /** Drum bus gain 0..1. */
  drumVolume: number;
}

/**
 * Generation-time settings remembered alongside a saved session — what the
 * user had configured on the Generate panel when they pressed Save, plus
 * the Play panel settings. Re-applying these is what makes a saved session
 * function as a "preset" for Favorite Generate. Every PlaySettings field is
 * optional: sessions saved before the Play panel carry only drumPatternId
 * and harmonyOverride (see playSettingsFromSession).
 */
export interface SessionGenerationContext extends Partial<PlaySettings> {
  model: string;            // checkpoint key, e.g. 'ft_f1' (default 2026-05-09)
  genre: string | null;
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
