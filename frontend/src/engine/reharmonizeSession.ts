import type { Song, SlotAddress } from '../models/types.ts';
import { captureGenerationComparison, matchesCapturedRange } from './generationComparison.ts';
import type { GenerationComparison, TargetChord } from './generationComparison.ts';
import { reharmFingerprint, reharmSelectionError, reharmTargets, reharmUpdates, requestReharmonization, toggleReharmSelection } from './reharmonize.ts';
import type { ReharmChangeLimit, ReharmMode, ReharmResult } from './reharmonize.ts';
import { previewRollback } from './chordPreview.ts';

export interface ReharmState {
  loading: boolean;
  error: string | null;
  result: ReharmResult | null;
  comparison: GenerationComparison | null;
  applied: GenerationComparison | null;
  selected: number[];
  committed: GenerationComparison | null;
  committedSelection: number[];
  originalKey: string | null;
  expectedFingerprint: string | null;
  /** What the last Apply kept, shown in the controls after the cards close. */
  notice: string | null;
  /** Wall time of the last successful search, for Run details. */
  elapsedMs: number | null;
  run: (modelKey: string, maxChanges: ReharmChangeLimit, mode: ReharmMode) => Promise<void>;
  preview: (index: number | null) => void;
  apply: () => void;
  close: () => void;
}
export const emptyReharmState = {
  loading: false, error: null, result: null, comparison: null, applied: null,
  selected: [], committed: null, committedSelection: [], originalKey: null, expectedFingerprint: null,
  notice: null, elapsedMs: null,
};
interface Host {
  read: () => { song: Song; songEpoch: number; selectedSlots: SlotAddress[] };
  stop: () => void;
  write: (updates: TargetChord[], previewOriginal?: GenerationComparison) => void;
  commit: () => void;
  clearGeneration: () => void;
  /** Keep a song for the editor's undo (Ctrl/Cmd+Z) before Apply changes it for good. */
  remember?: (song: Song) => void;
}

/** The song with the given slots replaced; chords are copied. */
export function withUpdates(song: Song, updates: TargetChord[]): Song {
  const byAddress = new Map(updates.map(u => [`${u.address.measureIndex}-${u.address.slotIndex}`, u.chord]));
  return { ...song, measures: song.measures.map((measure, measureIndex) => ({ ...measure,
    chords: measure.chords.map((chord, slotIndex) => {
      const key = `${measureIndex}-${slotIndex}`;
      return byAddress.has(key) ? structuredClone(byAddress.get(key)!) : chord;
    }) as typeof measure.chords,
  })) };
}

/** 'M3' or 'M3–M4' for the compared passage. */
function passageLabel(comparison: GenerationComparison): string {
  const measures = comparison.original.map(c => c.address.measureIndex + 1);
  const first = Math.min(...measures);
  const last = Math.max(...measures);
  return first === last ? `M${first}` : `M${first}–M${last}`;
}

export function hasPendingReharmSelection(state: Pick<ReharmState, 'selected' | 'committedSelection'>): boolean {
  return state.selected.join(',') !== state.committedSelection.join(',');
}

/** Editor-independent lifecycle, shared by the UI store and transition tests. */
export function reharmActions(host: Host, get: () => ReharmState, set: (patch: Partial<ReharmState>) => void,
  requestCandidates = requestReharmonization): Pick<ReharmState, 'run' | 'preview' | 'apply' | 'close'> {
  let request: AbortController | null = null;
  const close = () => {
    request?.abort();
    request = null;
    const state = get();
    const { song, songEpoch } = host.read();
    if (state.committed && state.applied && state.originalKey && hasPendingReharmSelection(state)) {
      const updates = previewRollback(song, songEpoch, {
        original: state.committed, shown: state.applied, key: state.originalKey,
      });
      if (updates.length) { host.stop(); host.write(updates); }
    }
    host.commit();
    set(emptyReharmState);
  };
  return {
    run: async (modelKey, maxChanges, mode) => {
      // A new search starts from the last Apply, never from a pending preview.
      close();
      const { song, songEpoch, selectedSlots } = host.read();
      const problem = reharmSelectionError(song, selectedSlots);
      if (problem) { set({ error: problem }); return; }
      const targets = reharmTargets(song, selectedSlots);
      request?.abort();
      const controller = new AbortController();
      request = controller;
      const comparison = captureGenerationComparison(song, targets, songEpoch);
      const fingerprint = reharmFingerprint(song);
      host.clearGeneration();
      set({ ...emptyReharmState, loading: true, comparison, applied: comparison, committed: comparison,
        originalKey: song.key, expectedFingerprint: fingerprint });
      const started = performance.now();
      try {
        const result = await requestCandidates(song, targets, maxChanges, mode, modelKey, controller.signal);
        if (request !== controller) return;
        const current = host.read();
        if (current.songEpoch !== songEpoch || reharmFingerprint(current.song) !== fingerprint) {
          throw new Error('The score changed during the search. Generate fresh candidates for this passage.');
        }
        set({ result, loading: false, elapsedMs: Math.round(performance.now() - started) });
      } catch (error) {
        if (request === controller) set({ loading: false, error: error instanceof Error ? error.message : 'Reharmonization failed.' });
      }
    },
    preview: index => {
      const state = get();
      const { song, songEpoch } = host.read();
      if (!state.comparison || !state.applied || !state.committed || state.loading || (index !== null && !state.result?.candidates[index])) return;
      if (!matchesCapturedRange(state.applied, song, songEpoch)) {
        set({ error: 'This passage was edited. Generate fresh candidates to keep those edits.' });
        return;
      }
      const stale = reharmFingerprint(song) !== state.expectedFingerprint;
      if (index !== null && stale) {
        set({ error: 'The surrounding score changed. Restore the original or generate fresh candidates.' });
        return;
      }
      if (song.key !== state.originalKey) {
        set({ error: 'The key changed. Generate fresh candidates for the new key.' });
        return;
      }
      const selected = index === null ? [] : toggleReharmSelection(state.selected, index, state.result!.candidates);
      host.stop();
      host.write(reharmUpdates(state.comparison, song, songEpoch,
        selected.map(i => state.result!.candidates[i])), state.committed);
      const current = host.read();
      set({ selected, error: stale ? 'Origin restored. Generate again to use the changed context.' : null,
        applied: captureGenerationComparison(current.song, state.comparison.original.map(c => c.address), current.songEpoch),
        expectedFingerprint: stale ? state.expectedFingerprint : reharmFingerprint(current.song) });
    },
    apply: () => {
      const state = get();
      const { song, songEpoch } = host.read();
      if (!state.result || !state.applied || !state.committed || !state.comparison || state.loading || !hasPendingReharmSelection(state)) return;
      if (song.key !== state.originalKey || !matchesCapturedRange(state.applied, song, songEpoch) ||
          reharmFingerprint(song) !== state.expectedFingerprint) {
        set({ error: 'The score changed. Close this comparison and search again to keep your edits.' });
        return;
      }
      // The passage as it stood before this Apply goes to the editor's undo.
      const before = previewRollback(song, songEpoch, { original: state.committed, shown: state.applied, key: state.originalKey! });
      if (before.length) host.remember?.(withUpdates(song, before));
      const kept = state.selected.length ? state.selected.map(i => `Alter ${i + 1}`).join(' + ') : 'Origin';
      const notice = `${kept} applied to ${passageLabel(state.comparison)}. Undo with ⌘Z / Ctrl+Z.`;
      host.commit();
      set({ committed: state.applied, committedSelection: [...state.selected], error: null });
      // Applying is the end of the comparison: the cards close and the chords stay.
      close();
      set({ notice });
    },
    close,
  };
}
