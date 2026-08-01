/**
 * ReviewStore — the "Second ear" over the A/B/C list: one POST
 * /api/review/candidates, then its result or its failure.
 *
 * The request is the list as the comparison shows it: every candidate as
 * Apply would write it over the captured Original (compareCandidate), the
 * score's own chords in the slots of those measures the run did not target,
 * and, when the bars sit in a section, the section before it from the same
 * offset. The server scores the candidates with the selected Music
 * Transformer and computes the facts; nothing here touches the score. New
 * suggestions or a replaced song drop the review, and an answer for an older
 * list is ignored.
 */
import { create } from 'zustand';
import type { Chord, SectionType, SlotAddress, Song, SongSection } from '../models/types';
import { parseSuggestionChords, type Suggestion } from '../engine/chordGenerator';
import { chordToString } from '../engine/chordParser';
import { formatChordSymbol } from '../engine/chordDisplay';
import { compareCandidate, slotKey, type GenerationComparison } from '../engine/generationComparison';
import { planGenre } from '../engine/planClient';
import { postJson } from '../engine/httpClient';
import { autoSection, sectionContaining } from '../engine/songForm';
import { useGenerationStore } from './generationStore';
import { useSongStore } from './songStore';

/** The backend's chord symbol limit (models/review.py). */
const MAX_SYMBOL_CHARS = 32;

type Bars = Array<[string | null, string | null]>;

/** Body of POST /api/review/candidates (backend/app/models/review.py). */
export interface ReviewRequest {
  key: string;
  modelKey?: string;
  timeSignature?: [number, number];
  genre: string | null;
  sectionType: SectionType | null;
  original: Bars;
  candidates: Array<{ label: string; chords: Bars }>;
  previousSection: Bars | null;
}

export type ReviewFactName = 'outOfKeyRatio' | 'changedSlots' | 'repetitionVsPrevious' | 'r1Roughness' | 'modelScore';

/** What the analyzer computed about one candidate. */
export interface CandidateFacts {
  label: string;
  modelScore?: number | null;
  outOfKeyRatio: number | null;
  changedSlots: number;
  repetitionVsPrevious: number | null;
  r1Roughness: number | null;
  /** Slots (0-based over the bars) the song-form rules write for the section type. */
  rulesSlots: number[];
}

export interface ReviewReason {
  /** '<factName>=<value>'. */
  fact: string;
  factName: ReviewFactName;
  value: number | null;
  claim: string;
  /** The analyzer's table holds that value for the chosen candidate. */
  confirmed: boolean;
  analyzerValue: number | null;
}

export interface ReviewResult {
  choice: string;
  reasons: ReviewReason[];
  facts: CandidateFacts[];
  model: string;
  latencyMs: number;
}

interface ReviewState {
  loading: boolean;
  /** The body of the newest review, as sent. */
  request: ReviewRequest | null;
  result: ReviewResult | null;
  error: string | null;
  /** Id of the newest review; answers of older ones are ignored. */
  runId: number;
  /** Review the A/B/C list on screen. */
  runReview: () => Promise<void>;
  /** Drop the review, aborting one in flight. */
  clear: () => void;
}

/** The chord as the review sends it: the staff's symbol with its extensions, e.g. 'C7(b9)/E'. */
function reviewSymbol(chord: Chord | null): string | null {
  if (!chord) return null;
  const full = formatChordSymbol(chord);
  return full.length <= MAX_SYMBOL_CHARS ? full : chordToString(chord).slice(0, MAX_SYMBOL_CHARS);
}

/** The bars of `measures` with each slot's chord from `chordAt`. */
function barsOf(measures: number[], chordAt: (address: SlotAddress) => Chord | null): Bars {
  return measures.map((measureIndex) => [
    reviewSymbol(chordAt({ measureIndex, slotIndex: 0 })),
    reviewSymbol(chordAt({ measureIndex, slotIndex: 1 })),
  ]);
}

/** The section before `section` in measure order, from the same offset and for at most `count` bars;
 *  null when there is none or it is too short to reach that offset. */
function previousSectionBars(song: Song, section: SongSection | null, offset: number, count: number): Bars | null {
  if (!section) return null;
  const previous = (song.sections ?? [])
    .filter((s) => s.startMeasure + s.length <= section.startMeasure)
    .reduce<SongSection | null>((latest, s) => (!latest || s.startMeasure > latest.startMeasure ? s : latest), null);
  if (!previous || offset >= previous.length) return null;
  const start = previous.startMeasure + offset;
  const measures = song.measures.slice(start, start + Math.min(count, previous.length - offset));
  if (measures.length === 0) return null;
  return measures.map((measure) => [reviewSymbol(measure.chords[0]), reviewSymbol(measure.chords[1])]);
}

/** The review's body for the A/B/C list (see the module comment). `sectionChoice` is the
 *  Section dropdown Generate read: 'auto' is what autoSection gives for the bars (their
 *  section, else their place in the whole song). */
export function buildReviewRequest(
  song: Song,
  suggestions: readonly Suggestion[],
  comparison: GenerationComparison,
  sectionChoice: SectionType | 'auto',
): ReviewRequest {
  const measures = [...new Set(comparison.original.map((slot) => slot.address.measureIndex))].sort((a, b) => a - b);
  if (measures.length === 0) throw new Error('There are no bars to compare');
  const scoreChord = (address: SlotAddress) => song.measures[address.measureIndex]?.chords[address.slotIndex] ?? null;
  const captured = new Map(comparison.original.map((slot) => [slotKey(slot.address), slot.chord]));
  const candidates = suggestions.map((suggestion, index) => {
    const after = new Map(compareCandidate(comparison, parseSuggestionChords(suggestion))
      .map((row) => [slotKey(row.address), row.after]));
    const chordAt = (address: SlotAddress) => after.has(slotKey(address)) ? after.get(slotKey(address))! : scoreChord(address);
    return { label: String.fromCharCode(65 + index), chords: barsOf(measures, chordAt) };
  });
  const section = sectionContaining(song, measures);
  return {
    key: song.key,
    genre: planGenre(song),
    // A New-from-Song-Form list was generated for its placed section, whatever the Generate dropdown says;
    // a drag-path run reviews with the section it sent, even if bars were added since; an unrecorded run
    // resolves the dropdown as Generate does (Auto: the bars' section, else their place in the whole song)
    sectionType: comparison.kind === 'section' ? section?.type ?? null
      : comparison.sectionType !== undefined ? comparison.sectionType
        : sectionChoice === 'auto' ? autoSection(song, measures).type : sectionChoice,
    original: barsOf(measures, (address) =>
      captured.has(slotKey(address)) ? captured.get(slotKey(address))! : scoreChord(address)),
    candidates,
    previousSection: previousSectionBars(song, section, section ? measures[0] - section.startMeasure : 0, measures.length),
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isReviewResult(value: unknown): value is ReviewResult {
  return isRecord(value) && typeof value.choice === 'string' &&
    Array.isArray(value.reasons) && Array.isArray(value.facts) && typeof value.model === 'string';
}

/** POST the review. Rejects with the server's detail (or the HTTP status when it sends none),
 *  an unreachable server, or an answer that is not a review. */
export async function requestReview(request: ReviewRequest, signal?: AbortSignal): Promise<ReviewResult> {
  const res = await postJson('/api/review/candidates', request, { signal });
  const body: unknown = await res.json().catch(() => null);
  if (!isReviewResult(body)) throw new Error('The server sent an invalid review');
  return body;
}

const FACT_LABELS: Record<ReviewFactName, string> = {
  modelScore: 'model likelihood',
  outOfKeyRatio: 'out-of-key',
  changedSlots: 'changed slots',
  repetitionVsPrevious: 'repeats previous',
  r1Roughness: 'roughness',
};

/** 'lowest' or 'highest' (for slots: 'fewest' or 'most') when the chosen candidate alone holds that extreme. */
function factRank(result: ReviewResult, name: ReviewFactName): string | null {
  const rows = result.facts.filter((row) => row[name] != null);
  const chosen = rows.find((row) => row.label === result.choice);
  if (!chosen || rows.length < 2) return null;
  const value = chosen[name] as number;
  const others = rows.filter((row) => row !== chosen).map((row) => row[name] as number);
  if (others.every((other) => value < other)) return name === 'changedSlots' ? 'fewest' : 'lowest';
  if (others.every((other) => value > other)) return name === 'changedSlots' ? 'most' : 'highest';
  return null;
}

/** A reason's fact as the line shows it: 'model likelihood -1.2, highest' or 'out-of-key 0'. A confirmed
 *  fact shows the analyzer's value and its rank among the candidates; any other the reason's own value. */
export function reasonFact(result: ReviewResult, reason: ReviewReason): string {
  const label = FACT_LABELS[reason.factName] ?? reason.factName;
  const value = reason.confirmed ? reason.analyzerValue : reason.value;
  const text = `${label} ${value === null ? '?' : String(Number(value.toFixed(3)))}`;
  const rank = reason.confirmed ? factRank(result, reason.factName) : null;
  return rank ? `${text}, ${rank}` : text;
}

/** How many slots of the chosen candidate the song-form rules write; 0 when none. */
export function rulesSlotCount(result: ReviewResult): number {
  return result.facts.find((row) => row.label === result.choice)?.rulesSlots.length ?? 0;
}

/** Module-level because it is not UI state. */
let controller: AbortController | null = null;

export const useReviewStore = create<ReviewState>((set, get) => ({
  loading: false,
  request: null,
  result: null,
  error: null,
  runId: 0,

  runReview: async () => {
    const { suggestions, comparison, generateSectionType, loading } = useGenerationStore.getState();
    if (!comparison || suggestions.length < 2 || suggestions.length > 3 || loading) return;
    controller?.abort();
    const current = new AbortController();
    controller = current;
    const runId = get().runId + 1;
    set({ runId, loading: true, request: null, result: null, error: null });
    try {
      const request = buildReviewRequest(useSongStore.getState().song, suggestions, comparison, generateSectionType);
      request.modelKey = useGenerationStore.getState().modelKey;
      request.timeSignature = useSongStore.getState().song.timeSignature;
      set({ request });
      const result = await requestReview(request, current.signal);
      if (get().runId === runId) set({ result, loading: false });
    } catch (err) {
      if (get().runId !== runId || current.signal.aborted) return;
      set({ error: `Second ear failed: ${err instanceof Error ? err.message : String(err)}`, loading: false });
    } finally {
      if (controller === current) controller = null;
    }
  },

  clear: () => {
    controller?.abort();
    controller = null;
    set((state) => ({ loading: false, request: null, result: null, error: null, runId: state.runId + 1 }));
  },
}));

function dropReview(): void {
  const { loading, request, result, error, clear } = useReviewStore.getState();
  if (loading || request || result || error) clear();
}

// New suggestions (a new run, a failed one, Keep current version) or a replaced
// song make the review meaningless. Dropped here, not in a component, so it
// also happens while the Generate panel is not mounted.
useGenerationStore.subscribe((state, prev) => {
  if (state.suggestions !== prev.suggestions || state.comparison !== prev.comparison) dropReview();
});
useSongStore.subscribe((state, prev) => {
  if (state.songEpoch !== prev.songEpoch) dropReview();
});
