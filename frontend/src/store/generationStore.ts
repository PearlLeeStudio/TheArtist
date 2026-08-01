/**
 * GenerationStore — UI-side state for the Generate panel.
 *
 * Lives outside songStore because these settings travel with the
 * generate engine, not with the song.
 *
 * Holds:
 *   - modelKey: persisted to localStorage so non-Generate UI (staff-header
 *     Save button) can bundle the current model into a session preset.
 *   - In-flight generation state (loading / phase / suggestions / etc.) —
 *     hoisted out of ChordActions component-local useState so the request
 *     keeps running and its result lands even when the user navigates
 *     away (Play tab, MyPage, etc.) before generate completes. Pearl
 *     2026-05-10: previously this state died on ChordActions unmount,
 *     losing the result.
 */
import { create } from 'zustand';
import type { Song, SlotAddress } from '../models/types';
import {
  generateChords,
  type GeneratePhase,
  type Suggestion,
} from '../engine/chordGenerator';
import type { SavedSession } from '../models/types';

const STORAGE_KEY = 'theartist-generation';
const DEFAULT_MODEL = 'ft_f1';  // Pearl 2026-05-09 — F1 (pop-preserving baseline)

interface GenerationState {
  // Persisted: model selection
  modelKey: string;
  setModelKey: (key: string) => void;

  // Ephemeral: in-flight generation state
  loading: boolean;
  phase: GeneratePhase | null;
  suggestions: Suggestion[];
  showSuggestions: boolean;
  /** Index of the suggestion currently applied to the score (so the user
   *  can hop A/B/C without re-running). null = none picked yet. */
  activeSuggestionIdx: number | null;

  // Actions on suggestion display
  setSuggestions: (s: Suggestion[]) => void;
  setShowSuggestions: (b: boolean) => void;
  setActiveSuggestionIdx: (i: number | null) => void;
  clearSuggestions: () => void;

  // Generate orchestration — owns the async lifecycle so navigation
  // doesn't kill in-flight generation.
  runGenerate: (
    song: Song,
    selectedSlots: SlotAddress[],
    modelKey: string,
  ) => Promise<void>;
  runFavoriteGenerate: (
    song: Song,
    selectedSlots: SlotAddress[],
    starredSessions: SavedSession[],
  ) => Promise<void>;
}

function load(): { modelKey: string } {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return { modelKey: DEFAULT_MODEL };
    const parsed = JSON.parse(raw);
    return { modelKey: typeof parsed?.modelKey === 'string' ? parsed.modelKey : DEFAULT_MODEL };
  } catch {
    return { modelKey: DEFAULT_MODEL };
  }
}

function persist(state: { modelKey: string }): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch {
    // ignore
  }
}

export const useGenerationStore = create<GenerationState>((set, get) => ({
  modelKey: load().modelKey,
  setModelKey: (key) => {
    set({ modelKey: key });
    persist({ modelKey: key });
  },

  loading: false,
  phase: null,
  suggestions: [],
  showSuggestions: false,
  activeSuggestionIdx: null,

  setSuggestions: (suggestions) => set({ suggestions }),
  setShowSuggestions: (showSuggestions) => set({ showSuggestions }),
  setActiveSuggestionIdx: (activeSuggestionIdx) => set({ activeSuggestionIdx }),
  clearSuggestions: () =>
    set({
      suggestions: [],
      showSuggestions: false,
      activeSuggestionIdx: null,
    }),

  runGenerate: async (song, selectedSlots, modelKey) => {
    if (selectedSlots.length === 0) return;
    // If a generate is already running, ignore — singleton lifecycle.
    if (get().loading) return;
    set({
      loading: true,
      phase: null,
      showSuggestions: false,
      activeSuggestionIdx: null,
    });
    try {
      const results = await generateChords(
        song,
        selectedSlots,
        modelKey,
        (p) => set({ phase: p }),
      );
      set({ suggestions: results, showSuggestions: true });
    } finally {
      set({ loading: false, phase: null });
    }
  },

  runFavoriteGenerate: async (song, selectedSlots, starredSessions) => {
    if (selectedSlots.length === 0 || starredSessions.length === 0) return;
    if (get().loading) return;
    set({
      loading: true,
      phase: null,
      showSuggestions: false,
      activeSuggestionIdx: null,
    });
    try {
      const calls = starredSessions.map((s) =>
        generateChords(song, selectedSlots, s.generation.model).then((sugs) => {
          const top = sugs[0];
          if (!top) return null;
          return { ...top, label: `${s.name} — ${top.label}` };
        }),
      );
      const settled = await Promise.all(calls);
      const merged = settled.filter((s): s is Suggestion => s !== null);
      set({ suggestions: merged, showSuggestions: true });
    } finally {
      set({ loading: false, phase: null });
    }
  },
}));
