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
 *   - targetSlots: the slots the current suggestions were generated for.
 *     applySuggestion writes exactly those, wherever the selection is now.
 *   - requestId: every run takes a new one, and a response that arrives
 *     after a newer run started (or after the song was replaced) is dropped.
 */
import { create } from 'zustand';
import type { SavedSession, SectionType, SlotAddress, Song } from '../models/types';
import {
  generateChords,
  generateSection,
  parseSuggestionChords,
  type GeneratePhase,
  type GenerateResult,
  type Suggestion,
  type SuggestionSource,
} from '../engine/chordGenerator';
import { CHORDS_PER_BAR, MAX_SECTION_CHORDS } from '../engine/songForm';
import {
  canApplyComparison,
  captureGenerationComparison,
  comparisonUpdates,
  matchesCapturedRange,
  type GenerationComparison,
} from '../engine/generationComparison';
import { useSongStore } from './songStore';

const STORAGE_KEY = 'theartist-generation';
const DEFAULT_MODEL = 'ft_f1';  // Pearl 2026-05-09 — F1 (pop-preserving baseline)

/** How a runSectionGenerate call ended, for the dialog that started it. */
export type SectionRunResult =
  | { status: 'done' }
  | { status: 'failed'; message: string }
  | { status: 'cancelled' };

interface GenerationState {
  // Persisted: model selection
  modelKey: string;
  setModelKey: (key: string) => void;

  /** Section sent by the drag-path Generate (not persisted). 'auto' = what
   *  songForm.autoSection gives: the type of the song section that holds the
   *  whole selection, else the part the bars' place in the whole song suggests,
   *  and no song-form rules when the bars cross a section boundary. */
  generateSectionType: SectionType | 'auto';
  setGenerateSectionType: (choice: SectionType | 'auto') => void;

  // Ephemeral: in-flight generation state
  loading: boolean;
  phase: GeneratePhase | null;
  error: string | null;
  suggestions: Suggestion[];
  /** Slots the current suggestions were generated for. */
  targetSlots: SlotAddress[];
  /** Original chords captured before this request; null slots are preserved. */
  comparison: GenerationComparison | null;
  /** The exact applied result, used to clear its active marker after edits. */
  appliedSnapshot: GenerationComparison | null;
  /** Who answered the current suggestions; 'rules' = the model was
   *  unavailable and the backend's rule-based generator answered. */
  source: SuggestionSource | null;
  /** Checkpoint that produced the current suggestions, when the server says. */
  modelUsed: string | null;
  showSuggestions: boolean;
  /** Index of the suggestion currently applied to the score (so the user
   *  can hop A/B/C without re-running). null = none picked yet. */
  activeSuggestionIdx: number | null;
  /** Id of the newest run; results of older runs are ignored. */
  requestId: number;

  // Actions on suggestion display
  setShowSuggestions: (b: boolean) => void;
  clearSuggestions: () => void;
  /** Write suggestion `index` into targetSlots and mark it active. */
  applySuggestion: (index: number) => void;
  /** Restore the request's original chords in its original target range. */
  restoreOriginal: () => void;

  // Generate orchestration — owns the async lifecycle so navigation
  // doesn't kill in-flight generation.
  runGenerate: (
    song: Song,
    selectedSlots: SlotAddress[],
    modelKey: string,
    sectionType: SectionType | null,
  ) => Promise<void>;
  runFavoriteGenerate: (
    song: Song,
    selectedSlots: SlotAddress[],
    starredSessions: SavedSession[],
    sectionType: SectionType | null,
  ) => Promise<void>;
  /** "New from Song Form": place the section after the last used measure,
   *  generate it, keep the A/B/C suggestions, apply A and select the new
   *  measures. On failure or cancel the placement is taken back and the
   *  previous A/B/C list shows again. */
  runSectionGenerate: (request: {
    sectionType: SectionType;
    chordCount: number;
  }) => Promise<SectionRunResult>;
  /** Abort the in-flight section request, undo its placement and show the
   *  A/B/C list it hid again. */
  cancelSectionGenerate: () => void;
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

/** Whether a request's captured range is still in the song as it is now. */
function canApplyToCurrentSong(comparison: GenerationComparison): boolean {
  const { song, songEpoch } = useSongStore.getState();
  return canApplyComparison(comparison, song, songEpoch);
}

/** State fields of an empty A/B/C list: nothing to show, apply or compare.
 *  Fresh arrays on every call: restoreList tells one list from another by
 *  the identity of its targetSlots. */
function clearedSuggestions(): Partial<GenerationState> {
  return {
    suggestions: [],
    targetSlots: [],
    comparison: null,
    appliedSnapshot: null,
    source: null,
    modelUsed: null,
    showSuggestions: false,
    activeSuggestionIdx: null,
  };
}

/** One starred preset's answer to Favorite Generate. */
interface PresetAnswer {
  suggestion: Suggestion;
  source: SuggestionSource | null;
}

/** A starred preset's top suggestion, labelled with the preset's name;
 *  null when it returned none. */
function presetTopSuggestion(
  session: SavedSession,
  song: Song,
  selectedSlots: SlotAddress[],
  sectionType: SectionType | null,
): Promise<PresetAnswer | null> {
  return generateChords(song, selectedSlots, sectionType, session.generation.model).then(({ suggestions, source }) => {
    const top = suggestions[0];
    if (!top) return null;
    return { suggestion: { ...top, label: `${session.name} — ${top.label}` }, source };
  });
}

/** The presets' answers as one A/B/C list. One note for the whole list when
 *  every preset fell back to the rule-based generator; otherwise mark just
 *  the ones that did. */
function mergePresetAnswers(answers: Array<PresetAnswer | null>): GenerateResult {
  const results = answers.filter((answer): answer is PresetAnswer => answer !== null);
  const allRules = results.length > 0 && results.every((r) => r.source === 'rules');
  const suggestions = results.map(({ suggestion, source }) =>
    !allRules && source === 'rules'
      ? { ...suggestion, label: `${suggestion.label} (rule-based)` }
      : suggestion,
  );
  return { suggestions, source: allRules ? 'rules' : null };
}

/** Both slots of every measure in [start, start + count). */
function slotsForMeasures(start: number, count: number): SlotAddress[] {
  const slots: SlotAddress[] = [];
  for (let m = start; m < start + count; m++) {
    slots.push({ measureIndex: m, slotIndex: 0 }, { measureIndex: m, slotIndex: 1 });
  }
  return slots;
}

/** A section request fills whole measures, up to the size limit. */
function isValidSectionChordCount(chordCount: number): boolean {
  return Number.isInteger(chordCount) &&
    chordCount >= CHORDS_PER_BAR &&
    chordCount <= MAX_SECTION_CHORDS &&
    chordCount % CHORDS_PER_BAR === 0;
}

/** The A/B/C list as it was before beginRequest hid it. */
type ListView = Pick<GenerationState,
  'targetSlots' | 'showSuggestions' | 'activeSuggestionIdx' | 'comparison' | 'appliedSnapshot'>;

function listViewOf(state: GenerationState): ListView {
  return {
    targetSlots: state.targetSlots,
    showSuggestions: state.showSuggestions,
    activeSuggestionIdx: state.activeSuggestionIdx,
    comparison: state.comparison,
    appliedSnapshot: state.appliedSnapshot,
  };
}

/** The in-flight section request. */
interface PendingSection {
  /** Aborts its fetch. */
  controller: AbortController;
  /** Takes its placed section back off the staff. */
  rollback: () => void;
  /** The A/B/C list it hid. */
  hiddenList: ListView;
}

/** Module-level because it is not UI state. */
let pendingSection: PendingSection | null = null;

/** Abort the in-flight section request; `rollback` also takes its section
 *  back off the staff (a replaced song has nothing left to take back). */
function dropPendingSection(rollback: boolean): void {
  const pending = pendingSection;
  if (!pending) return;
  pendingSection = null;
  pending.controller.abort();
  if (rollback) pending.rollback();
}

/** Place a pending section on the staff for a section run. Its rollback
 *  removes the section and the measures appended for it, unless the song
 *  has been replaced since. */
function placeSectionForRun(sectionType: SectionType, bars: number) {
  const songState = useSongStore.getState();
  const songEpoch = songState.songEpoch;
  const measuresBefore = songState.song.measures.length;
  const section = songState.placeSection(sectionType, bars);
  const rollback = () => {
    const current = useSongStore.getState();
    if (current.songEpoch === songEpoch) current.removeSection(section.id, measuresBefore);
  };
  return { section, song: useSongStore.getState().song, songEpoch, rollback };
}

export const useGenerationStore = create<GenerationState>((set, get) => {
  /** Start a run: supersede whatever was in flight and return its id. */
  const beginRequest = (): number => {
    dropPendingSection(true);
    const requestId = get().requestId + 1;
    set({
      requestId,
      loading: true,
      phase: null,
      error: null,
      showSuggestions: false,
      activeSuggestionIdx: null,
      appliedSnapshot: null,
    });
    return requestId;
  };
  const isCurrent = (requestId: number) => get().requestId === requestId;
  const onPhaseFor = (requestId: number) => (phase: GeneratePhase) => {
    if (isCurrent(requestId)) set({ phase });
  };
  const finish = (requestId: number) => {
    if (isCurrent(requestId)) set({ loading: false, phase: null });
  };
  const fail = (requestId: number, error: unknown) => {
    if (!isCurrent(requestId)) return;
    const reason = error instanceof Error ? error.message : 'Please try again.';
    set({ error: `Generation failed. ${reason} Your score has not changed.`, ...clearedSuggestions() });
  };
  /** After a section run that did not land, show the A/B/C list it hid
   *  again, unless those suggestions were replaced meanwhile. */
  const restoreList = (list: ListView) => {
    if (get().targetSlots !== list.targetSlots) return;
    const current = useSongStore.getState();
    const stillApplied = list.appliedSnapshot !== null &&
      matchesCapturedRange(list.appliedSnapshot, current.song, current.songEpoch);
    set({
      showSuggestions: list.showSuggestions,
      activeSuggestionIdx: stillApplied ? list.activeSuggestionIdx : null,
      comparison: list.comparison,
      appliedSnapshot: stillApplied ? list.appliedSnapshot : null,
    });
  };

  return {
    modelKey: load().modelKey,
    setModelKey: (key) => {
      set({ modelKey: key });
      persist({ modelKey: key });
    },

    generateSectionType: 'auto',
    setGenerateSectionType: (generateSectionType) => set({ generateSectionType }),

    loading: false,
    phase: null,
    error: null,
    suggestions: [],
    targetSlots: [],
    comparison: null,
    appliedSnapshot: null,
    source: null,
    modelUsed: null,
    showSuggestions: false,
    activeSuggestionIdx: null,
    requestId: 0,

    setShowSuggestions: (showSuggestions) => set({ showSuggestions }),
    clearSuggestions: () => set({ error: null, ...clearedSuggestions() }),

    applySuggestion: (index) => {
      const { suggestions, comparison, loading } = get();
      const suggestion = suggestions[index];
      if (!suggestion || !comparison || loading) return;
      // Backend works at measure granularity (selectedMeasures = [5] returns
      // both slot 0 and slot 1 for that measure), but a drag selection is
      // slot-level. Write only the slots this generation targeted so other
      // slots (and any chord they already had) stay intact.
      const chordMap = parseSuggestionChords(suggestion);
      const current = useSongStore.getState();
      const updates = comparisonUpdates(comparison, current.song, current.songEpoch, chordMap);
      if (updates.length === 0) return;
      current.setChords(updates);
      // Keep the panel open so the user can A/B/C between candidates.
      // Re-applying the active one is cheap and idempotent.
      const applied = useSongStore.getState();
      set({
        activeSuggestionIdx: index,
        appliedSnapshot: captureGenerationComparison(applied.song, updates.map((u) => u.address), applied.songEpoch),
      });
    },

    restoreOriginal: () => {
      const { comparison, loading } = get();
      if (!comparison || loading) return;
      const current = useSongStore.getState();
      const updates = comparisonUpdates(comparison, current.song, current.songEpoch);
      if (updates.length === 0) return;
      current.setChords(updates, { preserveAnalysis: true });
      set({ activeSuggestionIdx: null, appliedSnapshot: null });
    },

    runGenerate: async (song, selectedSlots, modelKey, sectionType) => {
      if (selectedSlots.length === 0) return;
      const requestId = beginRequest();
      const targetSlots = selectedSlots.map((s) => ({ ...s }));
      try {
        // Inside the try: a song the capture cannot read ends the run like
        // a failed request instead of leaving it loading with no error.
        const comparison = { ...captureGenerationComparison(song, targetSlots, useSongStore.getState().songEpoch), sectionType };
        const { suggestions, source } = await generateChords(
          song,
          selectedSlots,
          sectionType,
          modelKey,
          onPhaseFor(requestId),
        );
        if (!isCurrent(requestId) || !canApplyToCurrentSong(comparison)) return;
        set({ suggestions, source, modelUsed: null, targetSlots, comparison, appliedSnapshot: null, showSuggestions: true });
      } catch (error) {
        fail(requestId, error);
      } finally {
        finish(requestId);
      }
    },

    runFavoriteGenerate: async (song, selectedSlots, starredSessions, sectionType) => {
      if (selectedSlots.length === 0 || starredSessions.length === 0) return;
      const requestId = beginRequest();
      const targetSlots = selectedSlots.map((s) => ({ ...s }));
      try {
        const comparison = { ...captureGenerationComparison(song, targetSlots, useSongStore.getState().songEpoch), sectionType };
        const answers = await Promise.all(starredSessions.map((session) =>
          presetTopSuggestion(session, song, selectedSlots, sectionType)));
        if (!isCurrent(requestId) || !canApplyToCurrentSong(comparison)) return;
        const { suggestions, source } = mergePresetAnswers(answers);
        set({ suggestions, source, modelUsed: null, targetSlots, comparison, appliedSnapshot: null, showSuggestions: true });
      } catch (error) {
        fail(requestId, error);
      } finally {
        finish(requestId);
      }
    },

    runSectionGenerate: async ({ sectionType, chordCount }) => {
      if (!isValidSectionChordCount(chordCount)) {
        return {
          status: 'failed',
          message: `Use an even number of chords from ${CHORDS_PER_BAR} to ${MAX_SECTION_CHORDS}`,
        };
      }
      // beginRequest hides the A/B/C list; remember it so a run that does
      // not land can show it again (restoreList).
      const hiddenList = listViewOf(get());
      const requestId = beginRequest();
      const { section, song, songEpoch, rollback } = placeSectionForRun(sectionType, chordCount / CHORDS_PER_BAR);
      const targetSlots = slotsForMeasures(section.startMeasure, section.length);
      const pending: PendingSection = { controller: new AbortController(), rollback, hiddenList };
      pendingSection = pending;
      /** The run did not land: take its section back, end the request and
       *  show the hidden list again. */
      const abandonRun = () => {
        rollback();
        finish(requestId);
        restoreList(hiddenList);
      };

      let message: string;
      try {
        const comparison = captureGenerationComparison(song, targetSlots, songEpoch, 'section');
        const res = await generateSection(
          {
            song,
            sectionType,
            chordCount,
            startMeasure: section.startMeasure,
            modelKey: get().modelKey,
          },
          onPhaseFor(requestId),
          pending.controller.signal,
        );
        // Cancelled, superseded or the song was replaced meanwhile: whoever
        // did that already took the placement back (or the song is gone).
        if (pendingSection !== pending || !isCurrent(requestId)) return { status: 'cancelled' };
        pendingSection = null;
        if (!canApplyToCurrentSong(comparison)) {
          abandonRun();
          return { status: 'cancelled' };
        }
        if (res.suggestions.length > 0) {
          // Its chords land now, so a reload keeps the section.
          useSongStore.getState().commitSection(section.id);
          set({
            suggestions: res.suggestions,
            targetSlots,
            comparison,
            appliedSnapshot: null,
            source: res.source ?? null,
            modelUsed: res.modelUsed ?? null,
            showSuggestions: true,
            loading: false,
            phase: null,
          });
          get().applySuggestion(0);
          useSongStore.getState().selectSlotRange(targetSlots[0], targetSlots[targetSlots.length - 1]);
          return { status: 'done' };
        }
        message = 'The server returned no suggestions';
      } catch (err) {
        if (pendingSection !== pending) return { status: 'cancelled' };
        pendingSection = null;
        message = err instanceof Error ? err.message : String(err);
      }
      abandonRun();
      return { status: 'failed', message };
    },

    cancelSectionGenerate: () => {
      const pending = pendingSection;
      if (!pending) return;
      dropPendingSection(true);
      set((state) => ({ requestId: state.requestId + 1, loading: false, phase: null }));
      restoreList(pending.hiddenList);
    },
  };
});

// A replaced song (Initialize, MyPage Load) makes the suggestions and any
// in-flight request meaningless. Drop them here rather than in a component,
// so it also happens while the Generate panel is not mounted. Keyed on
// songEpoch, not song.id: a loaded session keeps the id of the song it was
// saved from, and its chords are not the ones the suggestions were made for.
useSongStore.subscribe((state, prev) => {
  if (state.songEpoch !== prev.songEpoch) {
    dropPendingSection(false);
    useGenerationStore.setState((s) => ({
      requestId: s.requestId + 1,
      loading: false,
      phase: null,
      error: null,
      ...clearedSuggestions(),
    }));
  } else if (state.song.measures !== prev.song.measures) {
    // An edit that changed the applied chords leaves no suggestion applied.
    const { appliedSnapshot } = useGenerationStore.getState();
    if (appliedSnapshot && !matchesCapturedRange(appliedSnapshot, state.song, state.songEpoch)) {
      useGenerationStore.setState({ activeSuggestionIdx: null, appliedSnapshot: null });
    }
  }
});
