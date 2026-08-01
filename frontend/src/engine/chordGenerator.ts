import type { Chord, Measure, Song, SlotAddress, SectionType } from '../models/types';
import { parseChordName, chordToString } from './chordParser.ts';
import { readSSEFrames } from './sseStream.ts';
import { formatErrorDetail, postJson } from './httpClient.ts';
import { timeOperation } from './experienceTiming.ts';

export interface TheoryExplanation {
  chord_a: string;
  chord_b: string;
  concept: string;
  explanation: string;
  chapter: string;
  section: string;
  page_start: number;
  page_end: number;
  kind?: 'generated' | 'source_excerpt';
  citation?: {
    source: 'omt' | 'berklee';
    title: string;
    url: string;
    authors: string[];
    editors?: string[];
    license: string;
    license_url: string;
    copyright: string;
    changes: string;
    passage_id: string;
  } | null;
}

export interface Suggestion {
  label: string;
  chords: Record<string, [string, string]>;
  explanations?: TheoryExplanation[] | null;
}

/** Who answered: 'rules' = the model was unavailable and the backend's
 *  explicitly identified rule-based generator answered. */
export type SuggestionSource = 'ai' | 'rules' | 'ai+rules';

export interface GenerateResult {
  suggestions: Suggestion[];
  source: SuggestionSource | null;
}

/** Body of POST /api/generate/section(/stream). */
export interface SectionGenerateParams {
  song: Song;
  sectionType: SectionType;
  /** Even, 2..128 (two chord slots per measure). */
  chordCount: number;
  /** First measure of the new section; the measures before it go as context. */
  startMeasure: number;
  modelKey?: string | null;
}

/** Result of /api/generate/section(/stream). `suggestions[i].chords` is keyed
 *  by absolute measure index, startMeasure .. startMeasure + bars - 1. */
export interface SectionGenerateResponse {
  sectionType: SectionType;
  title: string;
  startMeasure: number;
  bars: number;
  chordCount: number;
  suggestions: Suggestion[];
  source: 'ai' | 'rules';
  modelUsed: string | null;
  elapsed_ms?: number;
}

/** Live progress update from the streaming generate endpoint. */
export type GeneratePhase =
  | { name: 'model_load'; label: string }
  | { name: 'composing'; label: string; step: number; total: number; creativity: string }
  | { name: 'ranking'; label: string }
  | { name: 'voicing'; label: string }
  | { name: 'explaining'; label: string }
  | { name: 'complete'; label: string };

function _contextMeasures(measures: Measure[]) {
  return measures.map((m, i) => ({
    measure: i,
    chords: m.chords.map(c => c ? chordToString(c) : null),
  }));
}

/** Request fields shared by every generate endpoint. bpm is an int on the
 *  backend, and SongSettings accepts decimals. */
function _songFields(song: Song) {
  return {
    key: song.key,
    bpm: Math.round(song.bpm),
    genre: song.genre,
    timeSignature: song.timeSignature,
  };
}

function _buildBody(
  song: Song,
  selectedSlots: SlotAddress[],
  sectionType: SectionType | null,
  modelKey?: string,
) {
  const selectedMeasures = [...new Set(selectedSlots.map(s => s.measureIndex))];
  return {
    ..._songFields(song),
    context: _contextMeasures(song.measures),
    selectedMeasures,
    sectionType,
    modelKey,
  };
}

function _toSource(value: unknown): SuggestionSource | null {
  return value === 'ai' || value === 'rules' || value === 'ai+rules' ? value : null;
}

const _isChordSymbol = (value: unknown): boolean => typeof value === 'string' && value.trim() !== '';

/** A labelled candidate whose chords map measure indexes to two chord symbols. */
function _isSuggestion(value: unknown): value is Suggestion {
  if (!value || typeof value !== 'object') return false;
  const { label, chords } = value as Partial<Suggestion>;
  if (typeof label !== 'string' || label.trim() === '') return false;
  if (!chords || typeof chords !== 'object' || Array.isArray(chords) || Object.keys(chords).length === 0) return false;
  return Object.entries(chords).every(([measure, pair]) =>
    /^\d+$/.test(measure) && Array.isArray(pair) && pair.length === 2 && pair.every(_isChordSymbol));
}

/**
 * Call backend /api/generate/stream and stream progress events to `onPhase`
 * while yielding the final suggestions. HTTP, network and invalid/incomplete
 * stream responses reject so the caller can show the failure and keep the score.
 * When `onPhase` is omitted, progress events are discarded.
 *
 * `sectionType` is sent as given — null means no song-form rules.
 */
export async function generateChords(
  song: Song,
  selectedSlots: SlotAddress[],
  sectionType: SectionType | null,
  modelKey?: string,
  onPhase?: (phase: GeneratePhase) => void,
): Promise<GenerateResult> {
  return timeOperation('generate.result', 'default', async () => {
    const body = _buildBody(song, selectedSlots, sectionType, modelKey);
    const res = await postJson('/api/generate/stream', body);

    let final: GenerateResult | null = null;
    for await (const { event, data } of readSSEFrames<Record<string, unknown>>(res)) {
      if (event === 'result') {
        const suggestions = data?.suggestions;
        if (!Array.isArray(suggestions) || suggestions.length === 0 || !suggestions.every(_isSuggestion)) {
          throw new Error('The server sent an invalid generation result');
        }
        final = { suggestions, source: _toSource(data.source) };
      } else if (event === 'error') {
        throw new Error(formatErrorDetail(data?.detail) || 'The server reported a generation error');
      } else if (onPhase) {
        onPhase({ name: event, ...data } as GeneratePhase);
      }
    }

    if (!final) throw new Error('The generation stream ended without a result. Please try again.');
    return final;
  });
}

/**
 * Generate a new song-form section: POST /api/generate/section/stream with
 * the measures before `startMeasure` as context, forward phase events to
 * `onPhase`, and resolve with the server's SectionGenerateResponse.
 *
 * No offline fallback here: a non-2xx response (FastAPI 422 included), an
 * 'error' event or a stream that ends without a result rejects with the
 * server's message. Aborting `signal` rejects with the AbortError.
 */
export async function generateSection(
  params: SectionGenerateParams,
  onPhase?: (phase: GeneratePhase) => void,
  signal?: AbortSignal,
): Promise<SectionGenerateResponse> {
  return timeOperation('section.result', 'default', async () => {
    const { song, sectionType, chordCount, startMeasure, modelKey } = params;
    const body = {
      ..._songFields(song),
      sectionType,
      chordCount,
      context: _contextMeasures(song.measures.slice(0, startMeasure)),
      modelKey: modelKey ?? null,
    };
    const res = await postJson('/api/generate/section/stream', body, { signal, unreachableMessage: 'Could not reach the server' });

    let result: SectionGenerateResponse | null = null;
    for await (const { event, data } of readSSEFrames<Record<string, unknown>>(res)) {
      if (event === 'result') {
        if (!Array.isArray(data.suggestions)) throw new Error('The server sent an invalid section result');
        result = data as unknown as SectionGenerateResponse;
      } else if (event === 'error') {
        throw new Error(formatErrorDetail(data.detail) ?? 'The server reported an error');
      } else if (onPhase) {
        onPhase({ name: event, ...data } as GeneratePhase);
      }
    }
    if (!result) throw new Error('The stream ended without a result');
    return result;
  });
}

/**
 * Parse a suggestion's chord names into Chord objects and apply to slots.
 */
export function parseSuggestionChords(
  suggestion: Suggestion,
): Map<string, Chord> {
  const result = new Map<string, Chord>();
  for (const [mIdx, [name0, name1]] of Object.entries(suggestion.chords)) {
    const c0 = parseChordName(name0);
    const c1 = parseChordName(name1);
    if (c0) result.set(`${mIdx}-0`, c0);
    if (c1) result.set(`${mIdx}-1`, c1);
  }
  return result;
}
