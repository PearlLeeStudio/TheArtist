/**
 * Storage adapter for saved sessions. The whole MyPage feature talks to
 * this module, never to localStorage directly — so when v2 wants
 * IndexedDB / Supabase, only this file changes.
 *
 * Schema versioning: a top-level `{ version, sessions }` envelope so a
 * future migration can detect the old shape and upgrade in place.
 */
import type { SavedSession, Song } from '../models/types';
import { CHORDS_PER_BAR, isSectionType } from '../engine/songForm.ts';
import { withoutImportedContent } from '../engine/importRetention.ts';
import { useImportCleanupStore } from './importCleanupStore.ts';

const STORAGE_KEY = 'theartist-sessions';
export const SCHEMA_VERSION = 1;

interface Envelope {
  version: number;
  sessions: SavedSession[];
}

export function loadAll(): SavedSession[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) { useImportCleanupStore.getState().complete('saved projects'); return []; }
    const parsed = JSON.parse(raw) as Envelope;
    if (!parsed || parsed.version !== SCHEMA_VERSION || !Array.isArray(parsed.sessions)) {
      return [];
    }
    // Migrate source-bearing projects without dropping their chords, settings or names.
    const cleaned = parsed.sessions.map((session: unknown) => {
      if (!isRecord(session) || !isRecord(session.song)) return session;
      return { ...session, song: withoutImportedContent(session.song as unknown as Song) };
    });
    const replacement = JSON.stringify({ ...parsed, sessions: cleaned });
    if (replacement !== raw) {
      // A denied storage write must not hide the user's recovered musical work.
      try {
        localStorage.setItem(STORAGE_KEY, replacement);
        useImportCleanupStore.getState().complete('saved projects');
      } catch { useImportCleanupStore.getState().failed('saved projects'); }
    } else useImportCleanupStore.getState().complete('saved projects');
    return cleaned.filter(isValidSession).map(withValidArrangement);
  } catch {
    return [];
  }
}

export function saveAll(sessions: SavedSession[]): void {
  const envelope: Envelope = { version: SCHEMA_VERSION,
    sessions: sessions.map((session) => ({ ...session, song: withoutImportedContent(session.song) })) };
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(envelope));
    useImportCleanupStore.getState().complete('saved projects');
  } catch {
    throw new Error('This browser could not save the project. Storage may be full or disabled. Your current score is still open.');
  }
}

/**
 * Persist one deletion transaction before publishing its replacement list.
 * When no id matches, nothing is written and `sessions` itself comes back.
 */
export function removeSavedSessions(sessions: SavedSession[], ids: readonly string[]): SavedSession[] {
  const removedIds = new Set(ids);
  const remaining = sessions.filter((session) => !removedIds.has(session.id));
  if (remaining.length === sessions.length) return sessions;
  saveAll(remaining);
  return remaining;
}

// Shape limits of a saved project, checked on load.
const MIN_BPM = 20;
const MAX_BPM = 300;
const MAX_SONG_MEASURES = 4096;
/** The largest time-signature numerator; bass-pattern beats fall within such a bar. */
const MAX_BEATS_PER_MEASURE = 32;
const TIME_SIGNATURE_DENOMINATORS = [1, 2, 4, 8, 16, 32];
const MAX_VOICING_NOTES = 32;
const MAX_MIDI_NOTE = 127;

// Song.importSource: how a score was read, and how each bar was read. An import
// writes analysisSource 'local', a model of null or 'ft_f1' and a null requestId.
// Projects saved by earlier builds also hold 'meta' and 'mixed' sources (or no
// analysisSource at all), other model names and request ids. All of those stay
// accepted here, so that no saved project is dropped on load.
const ANALYSIS_SOURCES = ['local', 'meta', 'mixed'];
const MEASURE_ANALYSIS_SOURCES = ['local', 'meta'];
const IMPORT_MODES = ['chord_symbols', 'infer_harmony'];
const CHORD_BASES = ['printed', 'inferred', 'unreadable'];
const CONFIDENCE_LEVELS = ['high', 'medium', 'low'];
const MAX_FILENAME_LENGTH = 512;
/** Measures in an imported score: the same cap as engine/scoreImport.ts. */
const MAX_SCORE_IMPORT_MEASURES = 512;
/** Model names and request ids. */
const MAX_ID_LENGTH = 200;
/** Evidence and editor-review text. */
const MAX_EVIDENCE_LENGTH = 400;
const MAX_MEASURE_CHORDS = 16;
const MAX_CHORD_SYMBOL_LENGTH = 120;
const MAX_MEASURE_NOTES = 128;
const MAX_NOTE_NAME_LENGTH = 80;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isTextUpTo(value: unknown, maxLength: number): value is string {
  return typeof value === 'string' && value.length <= maxLength;
}

function isTextList(value: unknown, maxItems: number, maxLength: number): value is string[] {
  return Array.isArray(value) && value.length <= maxItems && value.every((item) => isTextUpTo(item, maxLength));
}

function isOneOf(value: unknown, options: readonly unknown[]): boolean {
  return options.includes(value);
}

function isIntegerInRange(value: unknown, min: number, max: number): value is number {
  return typeof value === 'number' && Number.isInteger(value) && value >= min && value <= max;
}

/** 1–32 beats over a power-of-two note value. */
function isTimeSignature(value: unknown): boolean {
  return Array.isArray(value) && value.length === 2 &&
    isIntegerInRange(value[0], 1, MAX_BEATS_PER_MEASURE) && isOneOf(value[1], TIME_SIGNATURE_DENOMINATORS);
}

function isOptionalNoteList(value: unknown): boolean {
  return value === undefined || isTextList(value, MAX_MEASURE_NOTES, MAX_NOTE_NAME_LENGTH);
}

function isLocalImportModel(model: unknown): boolean {
  return model === null || model === 'ft_f1';
}

/**
 * Every bar of a 'mixed' import (earlier builds only) names its own analysis
 * source, 'local' or 'meta'; a bar of any other import may only repeat the
 * import's. Local deterministic reading names no model; local sparse
 * harmonization names F1. Neither names an API request. A 'meta' bar of an
 * earlier build keeps its model name and may keep its request id.
 */
function hasValidProvenance(measure: Record<string, unknown>, importAnalysisSource: unknown): boolean {
  const namesValidSource = measure.analysisSource === undefined
    ? importAnalysisSource !== 'mixed'
    : isOneOf(measure.analysisSource, MEASURE_ANALYSIS_SOURCES) &&
      (importAnalysisSource === 'mixed' || measure.analysisSource === importAnalysisSource);
  if (!namesValidSource) return false;
  const analyzedLocally = (measure.analysisSource ?? importAnalysisSource) === 'local';
  if (measure.model !== undefined &&
      !(analyzedLocally ? isLocalImportModel(measure.model) : isTextUpTo(measure.model, MAX_ID_LENGTH))) return false;
  return measure.requestId === undefined || measure.requestId === null ||
    (isTextUpTo(measure.requestId, MAX_ID_LENGTH) && !analyzedLocally);
}

/** One source bar of an import, read against the import's analysis source and mode. */
function isValidImportedMeasure(measure: unknown, importAnalysisSource: unknown, mode: unknown): boolean {
  return isRecord(measure) &&
    isOneOf(measure.chordBasis, CHORD_BASES) &&
    isOneOf(measure.confidence, CONFIDENCE_LEVELS) &&
    isTextUpTo(measure.evidence, MAX_EVIDENCE_LENGTH) &&
    (measure.editorReview === undefined || isTextUpTo(measure.editorReview, MAX_EVIDENCE_LENGTH)) &&
    isTextList(measure.originalChords, MAX_MEASURE_CHORDS, MAX_CHORD_SYMBOL_LENGTH) &&
    (measure.sourceMeasureIndex === undefined ||
      isIntegerInRange(measure.sourceMeasureIndex, 0, MAX_SCORE_IMPORT_MEASURES - 1)) &&
    hasValidProvenance(measure, importAnalysisSource) &&
    isOptionalNoteList(measure.notes) &&
    isOptionalNoteList(measure.writtenNotes) &&
    isOptionalNoteList(measure.bassNotes) &&
    // A source rest or N.C. holds no chords; the user's own N.C. marks only
    // a bar the source did not.
    (measure.originalNoChord === undefined || (typeof measure.originalNoChord === 'boolean' &&
      (!measure.originalNoChord || measure.originalChords.length === 0))) &&
    (measure.userNoChord === undefined || (typeof measure.userNoChord === 'boolean' &&
      !(measure.userNoChord && measure.originalNoChord))) &&
    (measure.timeSignature === undefined || measure.timeSignature === null || isTimeSignature(measure.timeSignature)) &&
    // Inferred chords need the infer-harmony mode; source evidence is discarded. An
    // unreadable bar is low confidence with no chords.
    (measure.chordBasis !== 'inferred' || mode === 'infer_harmony') &&
    (measure.chordBasis !== 'unreadable' || (measure.confidence === 'low' && measure.originalChords.length === 0));
}

function isValidImportSource(importSource: unknown): boolean {
  if (!isRecord(importSource)) return false;
  // Sources saved before the analysisSource field existed have none; they are read as 'meta'.
  const analysisSource = importSource.analysisSource ?? 'meta';
  // A local analysis can name F1 for sparse harmonization, never an API request;
  // a 'meta' or 'mixed' one (earlier builds only) names its model and may name its request.
  const modelMatchesSource = analysisSource === 'local'
    ? isLocalImportModel(importSource.model)
    : isTextUpTo(importSource.model, MAX_ID_LENGTH);
  const requestIdMatchesSource = importSource.requestId === null ||
    (isTextUpTo(importSource.requestId, MAX_ID_LENGTH) && analysisSource !== 'local');
  if (!isTextUpTo(importSource.filename, MAX_FILENAME_LENGTH) || !isOneOf(analysisSource, ANALYSIS_SOURCES) ||
      !modelMatchesSource || !requestIdMatchesSource ||
      (importSource.mode !== undefined && !isOneOf(importSource.mode, IMPORT_MODES))) return false;
  // Inferred harmony retains each bar's confidence and reading basis.
  if (importSource.measures === undefined) return importSource.mode !== 'infer_harmony';
  if (!Array.isArray(importSource.measures) || importSource.measures.length > MAX_SCORE_IMPORT_MEASURES) return false;
  return importSource.measures.every((measure) => isValidImportedMeasure(measure, analysisSource, importSource.mode));
}

/** A filled chord slot; an empty slot is null. */
function isValidChord(chord: unknown): boolean {
  return isRecord(chord) && typeof chord.root === 'string' && typeof chord.quality === 'string' &&
    typeof chord.romanNumeral === 'string' && (chord.bass === undefined || typeof chord.bass === 'string') &&
    Array.isArray(chord.voicing) && chord.voicing.length <= MAX_VOICING_NOTES &&
    chord.voicing.every((note) => isIntegerInRange(note, 0, MAX_MIDI_NOTE)) &&
    Array.isArray(chord.extensions) && chord.extensions.every((extension) => typeof extension === 'string');
}

/** Measures have unique ids and CHORDS_PER_BAR slots, each empty or a chord. */
function areValidMeasures(measures: unknown[]): boolean {
  const ids = new Set<string>();
  for (const measure of measures) {
    if (!isRecord(measure) || typeof measure.id !== 'string' || ids.has(measure.id) ||
        !Number.isInteger(measure.index) || !Array.isArray(measure.chords) ||
        measure.chords.length !== CHORDS_PER_BAR) return false;
    ids.add(measure.id);
    for (const chord of measure.chords) {
      if (chord !== null && !isValidChord(chord)) return false;
    }
  }
  return true;
}

/** Sections have unique ids, lie within the song's measures and never overlap. */
function areValidSections(sections: unknown, measureCount: number): boolean {
  if (!Array.isArray(sections)) return false;
  const ranges: Array<{ start: number; end: number }> = [];
  const ids = new Set<string>();
  for (const section of sections) {
    if (!isRecord(section) || typeof section.id !== 'string' || ids.has(section.id) ||
        !(section.type === null || isSectionType(section.type)) ||
        typeof section.startMeasure !== 'number' || !Number.isInteger(section.startMeasure) || section.startMeasure < 0 ||
        typeof section.length !== 'number' || !Number.isInteger(section.length) || section.length < 1 ||
        section.startMeasure + section.length > measureCount ||
        (section.label !== undefined && typeof section.label !== 'string')) return false;
    ids.add(section.id);
    ranges.push({ start: section.startMeasure, end: section.startMeasure + section.length });
  }
  ranges.sort((a, b) => a.start - b.start);
  return !ranges.some((range, i) => i > 0 && range.start < ranges[i - 1].end);
}

function isValidSong(song: unknown): boolean {
  if (!isRecord(song) || typeof song.id !== 'string' || typeof song.title !== 'string' ||
      typeof song.key !== 'string' || !(song.genre === null || typeof song.genre === 'string') ||
      typeof song.bpm !== 'number' || !Number.isFinite(song.bpm) || song.bpm < MIN_BPM || song.bpm > MAX_BPM) return false;
  if (song.importSource !== undefined && !isValidImportSource(song.importSource)) return false;
  if (!isTimeSignature(song.timeSignature)) return false;
  if (!Array.isArray(song.measures) || song.measures.length > MAX_SONG_MEASURES || !areValidMeasures(song.measures)) return false;
  return song.sections === undefined || areValidSections(song.sections, song.measures.length);
}

/** The bar-long bass pattern computeArrangement caches with a session. */
function isValidArrangement(arrangement: unknown): boolean {
  return isRecord(arrangement) && Array.isArray(arrangement.bassPattern) &&
    arrangement.bassPattern.every((beat) =>
      typeof beat === 'number' && Number.isFinite(beat) && beat >= 0 && beat < MAX_BEATS_PER_MEASURE) &&
    typeof arrangement.bassInstrument === 'string' &&
    isIntegerInRange(arrangement.beatsPerMeasure, 1, MAX_BEATS_PER_MEASURE);
}

/** Arrangement is a derived cache; a stale cache must not break a valid score. */
function withValidArrangement(session: SavedSession): SavedSession {
  if (session.arrangement === undefined || isValidArrangement(session.arrangement)) return session;
  const repaired = { ...session };
  delete repaired.arrangement;
  return repaired;
}

/** Invalid saved entries are skipped without hiding neighboring valid scores. */
function isValidSession(session: unknown): session is SavedSession {
  return isRecord(session) &&
    typeof session.id === 'string' &&
    typeof session.name === 'string' &&
    typeof session.createdAt === 'number' && Number.isFinite(session.createdAt) &&
    typeof session.starred === 'boolean' &&
    isValidSong(session.song) &&
    isRecord(session.generation) && typeof session.generation.model === 'string' &&
    (session.generation.genre === null || typeof session.generation.genre === 'string');
}
