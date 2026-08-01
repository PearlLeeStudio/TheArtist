/**
 * Client of POST /api/plan/song/stream ("Complete the form"): the request
 * built from the song as the staff shows it and the typed SSE events of a
 * run.
 *
 * Shapes mirror backend/app/models/plan.py. The sketch sent is the song up
 * to its last used measure (a chord or a section), the same place the
 * section flow appends to, so the planned sections start where Apply puts
 * them. Errors reject like chordGenerator.ts: an unreachable server, the
 * HTTP error detail, or an SSE error frame without a trace. A run that
 * failed server-side resolves with its trace (status 'error') so its
 * failure stays visible.
 */
import type { Chord, SectionType, Song } from '../models/types';
import { chordToString } from './chordParser.ts';
import type { SectionGenerateResponse } from './chordGenerator.ts';
import { calculateRomanNumeral } from './romanNumeral.ts';
import { sectionPlacement } from './songForm.ts';
import { readSSEFrames } from './sseStream.ts';
import { formatErrorDetail, postJson } from './httpClient.ts';

export const TARGET_BARS = [16, 32, 48, 64] as const;
export type TargetBars = (typeof TARGET_BARS)[number];
/** A planned section has at least this many bars (backend MIN_PLAN_SECTION_BARS). */
export const MIN_PLAN_SECTION_BARS = 2;
/** The backend reads at most this many sketch bars (backend MAX_SECTION_BARS). */
const MAX_SKETCH_BARS = 64;

export type PlanContrast = 'low' | 'medium' | 'high';
export type ChorusMode = 'reuse' | 'new';
export const CONTRASTS: readonly PlanContrast[] = ['low', 'medium', 'high'];

export interface PlanOptions {
  targetBars: TargetBars;
  wantOutro: boolean;
  contrast: PlanContrast;
  /** Checkpoint for every section, as the section flow sends it. */
  modelKey?: string | null;
  chorusMode?: ChorusMode;
}

export interface SketchMeasure {
  index: number;
  chords: [string | null, string | null];
  /** Roman numeral per slot as the staff shows it; null for an empty slot. */
  roman: [string | null, string | null];
}

export interface LockedSection {
  type: SectionType | null;
  start: number;
  length: number;
  label: string | null;
}

/** Body of POST /api/plan/song(/stream). */
export interface PlanRequest {
  genre: string;
  key: string;
  timeSignature: [number, number];
  bpm: number;
  measures: SketchMeasure[];
  sections: LockedSection[];
  targetBars: TargetBars;
  wantOutro: boolean;
  contrast: PlanContrast;
  modelKey: string | null;
  chorusMode?: ChorusMode;
}

// --- The trace (backend PlanTrace) ---

export interface PlanIssue {
  code: string;
  section: number | null;
  message: string;
}

export interface PlanCite {
  bar: number;
  roman: string;
}

export interface PlannedSection {
  type: string;
  bars: number;
  cadence: string;
  reason: string;
  cites: PlanCite[];
}

export interface SongPlan {
  sections: PlannedSection[];
}

export interface ValidationAttempt {
  attempt: number;
  ok: boolean;
  errors: PlanIssue[];
}

/** A planned section as /api/generate/section returned it, plus the
 *  planner's cadence and the slots the song-form rules wrote. */
export interface PlanSectionResult extends Omit<SectionGenerateResponse, 'source'> {
  source: 'ai' | 'rules' | 'reuse';
  reuse?: {
    origin: 'sketch' | 'section';
    sectionIndex: number | null;
    startMeasure: number;
    source: 'sketch' | 'ai' | 'rules';
    modelUsed: string | null;
    retainedSlots: number[];
    changedSlots: number[];
  } | null;
  cadence: string;
  /** Section-relative chord slot indices written by the rules, not the model. */
  rulesSlots: number[];
  reason: string;
  cites: PlanCite[];
  regenerated: boolean;
}

export type PlanErrorKind = 'timeout' | 'invalid' | 'generation';

export interface PlanError {
  kind: PlanErrorKind;
  message: string;
}

export interface PlanTrace {
  status: 'ok' | 'error';
  genre: string;
  key: string;
  bpm: number;
  timeSignature: [number, number];
  plan: SongPlan | null;
  validation: ValidationAttempt[];
  sections: PlanSectionResult[];
  elapsedMs: number;
  generationMs: number;
  error: PlanError | null;
}

// --- The stream's events, in run order ---

export type PlanEvent =
  | { event: 'progress'; data: { label: string } }
  | { event: 'plan'; data: { attempt: number; source: 'rules'; plan: SongPlan; t_ms: number } }
  | { event: 'validate'; data: { attempt: number; ok: boolean; errors: PlanIssue[]; t_ms: number } }
  | { event: 'generate_section'; data: PlanSectionResult & { index: number; t_ms: number } }
  | { event: 'complete' | 'error'; data: PlanTrace };

/** The song's genre as the backend names it, or null when none is set.
 *  Songs saved before the genre keys were canonical may hold 'bossa nova'. */
export function planGenre(song: Pick<Song, 'genre'>): string | null {
  if (!song.genre || song.genre === 'none') return null;
  return song.genre === 'bossa nova' ? 'bossa' : song.genre;
}

/** Bars of the sketch: up to the last measure with a chord or a section,
 *  where the section flow (and Apply) appends the next section. */
export function sketchBars(song: Pick<Song, 'measures' | 'sections'>): number {
  return sectionPlacement(song, 1).startMeasure;
}

/** The lengths a song of `bars` sketch bars can still be planned to. */
export function availableTargetBars(bars: number): TargetBars[] {
  return TARGET_BARS.filter((target) => bars + MIN_PLAN_SECTION_BARS <= target);
}

/** Undo or another section can change the sketch while the dialog stays open. */
export function resolveTargetBars(bars: number, preferred: TargetBars): TargetBars | null {
  const available = availableTargetBars(bars);
  return available.includes(preferred) ? preferred : (available[0] ?? null);
}

const NO_GENRE = 'Pick a genre first: the form follows the genre.';

/** Why the song cannot be planned yet, or null. */
export function planBlocker(song: Pick<Song, 'genre' | 'measures' | 'sections'>): string | null {
  if (!planGenre(song)) return NO_GENRE;
  const bars = sketchBars(song);
  if (bars > MAX_SKETCH_BARS - MIN_PLAN_SECTION_BARS) {
    return `The sketch already uses ${bars} bars; a form has at most ${MAX_SKETCH_BARS}.`;
  }
  return null;
}

function slotSymbol(chord: Chord | null): string | null {
  return chord ? chordToString(chord) : null;
}

/** The PlanRequest for `song`: throws the planBlocker, or when the target
 *  length leaves no room for a section after the sketch. */
export function buildPlanRequest(song: Song, options: PlanOptions): PlanRequest {
  const genre = planGenre(song);
  const blocker = planBlocker(song);
  if (!genre || blocker) throw new Error(blocker ?? NO_GENRE);
  const bars = sketchBars(song);
  if (!availableTargetBars(bars).includes(options.targetBars)) {
    throw new Error(`The sketch has ${bars} bars; pick a length of at least ${bars + MIN_PLAN_SECTION_BARS} bars.`);
  }
  const measures = song.measures.slice(0, bars).map((measure, index): SketchMeasure => {
    const [first, second] = measure.chords;
    return {
      index,
      chords: [slotSymbol(first), slotSymbol(second)],
      // The same call analyzeChord makes when a chord lands on the staff.
      roman: [
        first ? calculateRomanNumeral(first, song.key) : null,
        second ? calculateRomanNumeral(second, song.key) : null,
      ],
    };
  });
  const sections = [...(song.sections ?? [])]
    .sort((a, b) => a.startMeasure - b.startMeasure)
    .map((section): LockedSection => ({
      type: section.type,
      start: section.startMeasure,
      length: section.length,
      label: section.label?.trim().slice(0, 80) || null,  // the backend caps labels at 80 characters
    }));
  return {
    genre,
    key: song.key,
    timeSignature: [song.timeSignature[0], song.timeSignature[1]],
    // bpm is an int on the backend, and SongSettings accepts decimals.
    bpm: Math.round(song.bpm),
    measures,
    sections,
    targetBars: options.targetBars,
    wantOutro: options.wantOutro,
    contrast: options.contrast,
    modelKey: options.modelKey ?? null,
    ...(options.chorusMode ? { chorusMode: options.chorusMode } : {}),
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/** The terminal frames carry the whole trace; a route crash sends only `detail`. */
function isPlanTrace(value: unknown): value is PlanTrace {
  if (!isRecord(value)) return false;
  return (value.status === 'ok' || value.status === 'error') &&
    Array.isArray(value.sections) && Array.isArray(value.validation);
}

/** The frame as a typed event; null for an event name this client does not know. */
function toPlanEvent(event: string, data: unknown): PlanEvent | null {
  if (!isRecord(data)) throw new Error(`The server sent an invalid ${event} event`);
  switch (event) {
    case 'progress':
      if (typeof data.label !== 'string') throw new Error('The server sent an invalid progress event');
      return { event, data: { label: data.label } };
    case 'plan':
      if (!isRecord(data.plan) || !Array.isArray(data.plan.sections)) throw new Error(`The server sent an invalid ${event} event`);
      return { event, data: data as unknown as Extract<PlanEvent, { event: 'plan' }>['data'] };
    case 'validate':
      if (typeof data.ok !== 'boolean' || !Array.isArray(data.errors)) throw new Error('The server sent an invalid validate event');
      return { event, data: data as unknown as Extract<PlanEvent, { event: 'validate' }>['data'] };
    case 'generate_section':
      if (typeof data.index !== 'number' || !Array.isArray(data.suggestions) || !Array.isArray(data.rulesSlots)) {
        throw new Error('The server sent an invalid section result');
      }
      return { event, data: data as unknown as Extract<PlanEvent, { event: 'generate_section' }>['data'] };
    case 'complete':
      if (!isPlanTrace(data)) throw new Error('The server sent an invalid plan result');
      return { event, data };
    case 'error':
      if (isPlanTrace(data)) return { event, data };
      throw new Error(formatErrorDetail(data.detail) ?? 'The server reported a planning error');
    default:
      return null;
  }
}

/**
 * POST the request to /api/plan/song/stream, hand every typed event to
 * `onEvent` as it arrives, and resolve with the trace of the terminal
 * `complete` or `error` frame (status 'error' for a run that failed on the
 * server). Rejects when the server cannot be reached, answers an HTTP
 * error (FastAPI 422 included), sends an error frame without a trace, or
 * ends the stream without a terminal frame; aborting `signal` rejects
 * with the AbortError.
 */
export async function runPlan(
  request: PlanRequest,
  options: { signal?: AbortSignal; onEvent?: (event: PlanEvent) => void } = {},
): Promise<PlanTrace> {
  const res = await postJson('/api/plan/song/stream', request, { signal: options.signal });

  let trace: PlanTrace | null = null;
  for await (const { event, data } of readSSEFrames<unknown>(res)) {
    const typed = toPlanEvent(event, data);
    if (!typed) continue;
    options.onEvent?.(typed);
    if (typed.event === 'complete' || typed.event === 'error') trace = typed.data;
  }
  if (!trace) throw new Error('The plan stream ended without a result. Please try again.');
  return trace;
}
