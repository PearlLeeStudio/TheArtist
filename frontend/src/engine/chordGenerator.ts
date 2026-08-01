import type { Chord, Song, SlotAddress } from '../models/types';
import { NOTE_VALUES, VALUE_TO_FLAT_NAME, MAJOR_SCALE_INTERVALS, MINOR_SCALE_INTERVALS } from './constants';
import { parseChordName, chordToString } from './chordParser';
import { readSSEFrames } from './sseStream';

export interface TheoryExplanation {
  chord_a: string;
  chord_b: string;
  concept: string;
  explanation: string;
  chapter: string;
  section: string;
  page_start: number;
  page_end: number;
}

export interface Suggestion {
  label: string;
  chords: Record<string, [string, string]>;
  explanations?: TheoryExplanation[] | null;
}

const GENRE_QUALITIES: Record<string, string[]> = {
  jazz:         ['maj7', 'm7', '7', 'm7b5', 'dim7', '9', 'm9', '13', '6', 'mMaj7', 'sus4', '11'],
  pop:          ['maj', 'min', '7', 'sus4', 'sus2', 'add9', 'maj7', 'm7'],
  blues:        ['7', '9', 'm7', '13', 'dim', 'maj'],
  bossa:        ['maj7', 'm7', '7', '6', 'm6', 'dim7', '9', 'maj9'],
  'bossa nova': ['maj7', 'm7', '7', '6', 'm6', 'dim7', '9', 'maj9'],
  default:      ['maj', 'min', '7', 'maj7', 'm7', 'dim', 'aug', 'sus4'],
};

function pickRandom<T>(arr: T[]): T {
  return arr[Math.floor(Math.random() * arr.length)];
}

function getDiatonicRoots(keyRoot: string, isMinor: boolean): string[] {
  const rootVal = NOTE_VALUES[keyRoot] ?? 0;
  const intervals = isMinor ? MINOR_SCALE_INTERVALS : MAJOR_SCALE_INTERVALS;
  return intervals.map(i => VALUE_TO_FLAT_NAME[(rootVal + i) % 12]);
}

/**
 * Local random fallback — used when backend is unreachable.
 */
export function generateRandomChords(
  slots: SlotAddress[],
  key: string,
  genre: string | null,
): Map<string, Chord> {
  const qualities = GENRE_QUALITIES[genre ?? ''] ?? GENRE_QUALITIES.default;
  const keyRoot = key.split(' ')[0];
  const diatonicRoots = getDiatonicRoots(keyRoot, key.includes('minor'));

  const result = new Map<string, Chord>();
  for (const slot of slots) {
    const root = Math.random() < 0.7 ? pickRandom(diatonicRoots) : pickRandom(VALUE_TO_FLAT_NAME);
    result.set(`${slot.measureIndex}-${slot.slotIndex}`, {
      root,
      quality: pickRandom(qualities),
      romanNumeral: '',
      voicing: [],
      extensions: [],
    });
  }
  return result;
}

/** Live progress update from the streaming generate endpoint. */
export type GeneratePhase =
  | { name: 'model_load'; label: string }
  | { name: 'composing'; label: string; step: number; total: number; creativity: string }
  | { name: 'ranking'; label: string }
  | { name: 'voicing'; label: string }
  | { name: 'explaining'; label: string }
  | { name: 'complete'; label: string };

function _inferSectionType(song: Song, selectedMeasures: number[]): string {
  const totalMeasures = song.measures.length;
  const maxSelected = Math.max(...selectedMeasures);
  const minSelected = Math.min(...selectedMeasures);
  const endsAtLast = maxSelected >= totalMeasures - 1;
  const startsAtFirst = minSelected === 0;
  const coversMost = selectedMeasures.length >= totalMeasures * 0.7;
  const endRatio = totalMeasures > 0 ? (maxSelected + 1) / totalMeasures : 1;
  if (coversMost || (startsAtFirst && endsAtLast)) return 'chorus';
  if (startsAtFirst && endRatio <= 0.2) return 'intro';
  if (endsAtLast) return 'outro';
  if (endRatio >= 0.7) return 'chorus';
  if (endRatio >= 0.45) return 'bridge';
  return 'verse';
}

function _buildBody(song: Song, selectedSlots: SlotAddress[], modelKey?: string) {
  const selectedMeasures = [...new Set(selectedSlots.map(s => s.measureIndex))];
  return {
    key: song.key,
    bpm: song.bpm,
    genre: song.genre,
    timeSignature: song.timeSignature,
    context: song.measures.map((m, i) => ({
      measure: i,
      chords: m.chords.map(c => c ? chordToString(c) : null),
    })),
    selectedMeasures,
    sectionType: _inferSectionType(song, selectedMeasures),
    modelKey,
  };
}

function _offlineFallback(song: Song, selectedSlots: SlotAddress[]): Suggestion[] {
  const selectedMeasures = [...new Set(selectedSlots.map(s => s.measureIndex))];
  const fallback = generateRandomChords(selectedSlots, song.key, song.genre);
  const chords: Record<string, [string, string]> = {};
  for (const mIdx of selectedMeasures) {
    const s0 = fallback.get(`${mIdx}-0`);
    const s1 = fallback.get(`${mIdx}-1`);
    chords[String(mIdx)] = [
      s0 ? chordToString(s0) : 'Cmaj7',
      s1 ? chordToString(s1) : 'Cmaj7',
    ];
  }
  return [{ label: 'Random (offline)', chords }];
}

/**
 * Call backend /api/generate/stream and stream progress events to `onPhase`
 * while yielding the final suggestions. Falls back to local random generation
 * on network / parse error. When `onPhase` is omitted, behaves identically
 * to the non-streaming path (just discards events).
 */
export async function generateChords(
  song: Song,
  selectedSlots: SlotAddress[],
  modelKey?: string,
  onPhase?: (phase: GeneratePhase) => void,
): Promise<Suggestion[]> {
  const body = _buildBody(song, selectedSlots, modelKey);
  try {
    const res = await fetch('/api/generate/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`API error: ${res.status}`);

    let finalSuggestions: Suggestion[] | null = null;
    for await (const { event, data } of readSSEFrames<Record<string, unknown>>(res)) {
      if (event === 'result') {
        const sugs = (data as { suggestions?: unknown }).suggestions;
        if (!Array.isArray(sugs)) throw new Error('Invalid result payload');
        finalSuggestions = sugs as Suggestion[];
      } else if (event === 'error') {
        throw new Error((data as { detail?: string }).detail || 'stream error');
      } else if (onPhase) {
        onPhase({ name: event, ...data } as GeneratePhase);
      }
    }

    if (!finalSuggestions) throw new Error('Stream closed without result event');
    return finalSuggestions;
  } catch (err) {
    console.warn('Chord generation failed, using fallback:', err);
    return _offlineFallback(song, selectedSlots);
  }
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
