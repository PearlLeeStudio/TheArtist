/**
 * PlanStore — one "Complete the form" run at a time: its chips as the
 * stream reports them, the planned sections and the trace, then
 * Apply / Undo on the staff.
 *
 * Nothing touches the song until applyPlan: a run that fails or is
 * cancelled leaves the score as it was. Apply appends every planned
 * section's first suggestion through the song-form section APIs the
 * section flow uses (placeSection, setChords, commitSection), after a
 * snapshot that undoPlan restores with setSong.
 */
import { create } from 'zustand';
import type { Song } from '../models/types';
import { parseSuggestionChords } from '../engine/chordGenerator';
import type { TargetChord } from '../engine/generationComparison';
import {
  buildPlanRequest,
  runPlan,
  type PlanEvent,
  type PlanOptions,
  type PlanRequest,
  type PlanSectionResult,
  type PlanTrace,
} from '../engine/planClient';
import { CHORDS_PER_BAR, sectionLabel } from '../engine/songForm';
import { useSongStore } from './songStore';
import { beginTiming } from '../engine/experienceTiming.ts';

export type PlanStatus = 'idle' | 'running' | 'done' | 'failed';
export type PlanChipStep = 'plan' | 'validate' | 'generate_section' | 'error';
/** Who did the step: the author's chord model, the rules, or an earlier chorus. */
export type PlanChipBadge = 'own model' | 'rules' | 'reused chorus';

/** One step of the run as the band shows it, in run order. */
export interface PlanChip {
  step: PlanChipStep;
  ok: boolean;
  label: string;
  badge: PlanChipBadge | null;
  /** 'cadence: rules (n)' for a section with slots the song-form rules wrote. */
  note: string | null;
  /** Details for the tooltip: the plan, the validator's issues, a reason, the failure. */
  message: string | null;
}

interface PlanState {
  progress: string;
  status: PlanStatus;
  chips: PlanChip[];
  /** The generated sections by plan index. */
  sections: PlanSectionResult[];
  error: string | null;
  /** The terminal frame's trace (also of a failed run), for Export trace. */
  trace: PlanTrace | null;
  /** The sketch and settings the result was made for; checked again before Apply. */
  request: PlanRequest | null;
  applied: boolean;
  /** The song before Apply; undoPlan restores it. */
  snapshot: Song | null;
  /** Id of the newest run; events and results of older runs are ignored. */
  runId: number;

  /** Start a run for the current song; a run in flight is aborted first. */
  startPlan: (options: PlanOptions) => Promise<void>;
  /** Abort the run in flight and hide the band; the song is untouched. */
  cancelPlan: () => void;
  /** Append every planned section's first suggestion to the song, in order. */
  applyPlan: () => void;
  /** Restore the song from before Apply. */
  undoPlan: () => void;
  /** The trace as JSON, or null before a terminal frame arrived. */
  exportTrace: () => string | null;
  /** Hide the band (cancelling a run in flight); applied sections stay. */
  dismiss: () => void;
}

/** Store fields of no run at all. */
function idleState(): Pick<PlanState, 'progress' | 'status' | 'chips' | 'sections' | 'error' | 'trace' | 'request' | 'applied' | 'snapshot'> {
  return {
    progress: 'Preparing the song form…', status: 'idle', chips: [], sections: [], error: null, trace: null, request: null,
    applied: false, snapshot: null,
  };
}

/** 'chorus 8 (authentic) · verse 8 (half)': the plan as one line. */
function planSummary(event: Extract<PlanEvent, { event: 'plan' }>['data']): string {
  return event.plan.sections.map((s) => `${s.type} ${s.bars} (${s.cadence})`).join(' · ');
}

function sectionChip(section: PlanSectionResult): PlanChip {
  const rules = section.rulesSlots.length;
  return {
    step: 'generate_section',
    ok: true,
    label: `${sectionLabel(section.sectionType)} · ${section.bars} bars`,
    // The backend's rule-based generator answers as 'rules' when the model is unavailable.
    badge: section.source === 'reuse' ? 'reused chorus' : section.source === 'ai' ? 'own model' : 'rules',
    note: section.reuse
      ? `from bar ${section.reuse.startMeasure + 1} · ${section.reuse.changedSlots.length} ending slots changed`
      : rules > 0 ? `cadence: rules (${rules})` : null,
    message: [section.modelUsed ? `model ${section.modelUsed}` : null, section.cadence ? `cadence ${section.cadence}` : null, section.reason || null]
      .filter(Boolean).join(' · ') || null,
  };
}

/** The chip for one stream event, or null for the terminal frames. */
function chipFor(event: PlanEvent): PlanChip | null {
  switch (event.event) {
    case 'plan':
      return {
        step: 'plan',
        ok: true,
        label: 'plan',
        badge: 'rules',
        note: null,
        message: planSummary(event.data),
      };
    case 'validate': {
      const { ok, errors } = event.data;
      return {
        step: 'validate',
        ok,
        label: ok ? 'validate ok' : `validate failed (${errors.length})`,
        badge: null,
        note: null,
        message: errors.length ? errors.map((issue) => `${issue.code}: ${issue.message}`).join('; ') : null,
      };
    }
    case 'generate_section':
      return sectionChip(event.data);
    default:
      return null;
  }
}

/** The red chip of a failed run. */
function errorChip(label: string, message: string): PlanChip {
  return { step: 'error', ok: false, label, badge: null, note: null, message };
}

/** The section's chords, re-keyed from the server's absolute measure
 *  indices to where placeSection put the section. */
function sectionUpdates(section: PlanSectionResult, startMeasure: number): TargetChord[] {
  const suggestion = section.suggestions[0];
  if (!suggestion) return [];
  const chords = parseSuggestionChords(suggestion);
  const updates: TargetChord[] = [];
  for (let bar = 0; bar < section.bars; bar++) {
    for (let slot = 0; slot < CHORDS_PER_BAR; slot++) {
      const chord = chords.get(`${section.startMeasure + bar}-${slot}`);
      if (chord) updates.push({ address: { measureIndex: startMeasure + bar, slotIndex: slot as 0 | 1 }, chord });
    }
  }
  return updates;
}

/** Module-level because it is not UI state. */
let controller: AbortController | null = null;
/** True while undoPlan replaces the song: that epoch bump is the plan's own. */
let restoringSnapshot = false;

export const usePlanStore = create<PlanState>((set, get) => {
  const isCurrent = (runId: number) => get().runId === runId;

  const onEventFor = (runId: number) => (event: PlanEvent) => {
    if (!isCurrent(runId)) return;
    if (event.event === 'progress') { set({ progress: event.data.label }); return; }
    const chip = chipFor(event);
    if (!chip) return;
    set((state) => {
      const sections = [...state.sections];
      if (event.event === 'generate_section') sections[event.data.index] = event.data;
      return { chips: [...state.chips, chip], sections };
    });
  };

  return {
    ...idleState(),
    runId: 0,

    startPlan: async (options) => {
      controller?.abort();
      const current = new AbortController();
      controller = current;
      const runId = get().runId + 1;
      set({ ...idleState(), runId, status: 'running' });
      // 'rules' is the timing condition the genre-default form has always been recorded under
      const finish = beginTiming('form.result', 'rules');
      try {
        const request = buildPlanRequest(useSongStore.getState().song, options);
        set({ request });
        const trace = await runPlan(request, { signal: current.signal, onEvent: onEventFor(runId) });
        finish(trace.status === 'ok' ? 'ok' : 'error');
        if (!isCurrent(runId)) return;
        if (trace.status === 'ok') {
          set({ status: 'done', sections: trace.sections, trace });
        } else {
          const failure = trace.error;
          const label = failure ? `failed: ${failure.kind}` : 'failed';
          const message = failure?.message ?? 'The run failed';
          set((state) => ({
            status: 'failed', error: message, trace,
            chips: [...state.chips, errorChip(label, message)],
          }));
        }
      } catch (err) {
        finish(current.signal.aborted ? 'cancelled' : 'error');
        if (!isCurrent(runId) || current.signal.aborted) return;
        const message = err instanceof Error ? err.message : String(err);
        set((state) => ({ status: 'failed', error: message, chips: [...state.chips, errorChip('failed', message)] }));
      } finally {
        if (controller === current) controller = null;
      }
    },

    cancelPlan: () => {
      controller?.abort();
      controller = null;
      set((state) => ({ ...idleState(), runId: state.runId + 1 }));
    },

    applyPlan: () => {
      const { status, sections, applied, request } = get();
      if (status !== 'done' || applied || sections.length === 0 || !request) return;
      const { song, placeSection, setChords, commitSection } = useSongStore.getState();
      let matches = false;
      try {
        matches = JSON.stringify(buildPlanRequest(song, request)) === JSON.stringify(request);
      } catch {
        // An edit may also make the old request invalid (no genre or no room left).
      }
      if (!matches) {
        const message = 'The score changed after this plan started. Start a new plan before applying it.';
        set((state) => ({
          status: 'failed', error: message,
          chips: [...state.chips, errorChip('score changed', message)],
        }));
        return;
      }
      const snapshot = structuredClone(song);
      for (const section of sections) {
        // After the last used measure, so the sections follow each other.
        const placed = placeSection(section.sectionType, section.bars);
        const updates = sectionUpdates(section, placed.startMeasure);
        if (updates.length > 0) setChords(updates);
        // Its chords landed, so a reload keeps the section.
        commitSection(placed.id);
      }
      set({ applied: true, snapshot });
    },

    undoPlan: () => {
      const { applied, snapshot } = get();
      if (!applied || !snapshot) return;
      // setSong replaces the whole song; the import undo it would clear is unrelated to this plan
      const { previousImportSong } = useSongStore.getState();
      restoringSnapshot = true;
      try {
        useSongStore.getState().setSong(snapshot);
      } finally {
        restoringSnapshot = false;
      }
      useSongStore.setState({ previousImportSong });
      set({ applied: false, snapshot: null });
    },

    exportTrace: () => {
      const trace = get().trace;
      return trace ? JSON.stringify(trace, null, 2) : null;
    },

    dismiss: () => {
      if (get().status === 'running') {
        get().cancelPlan();
        return;
      }
      set(idleState());
    },
  };
});

// A replaced song (Initialize, Load, Import) is not the sketch the plan was
// made for: drop the run, in flight or done, as generationStore drops its
// suggestions. undoPlan's own replacement is the exception.
useSongStore.subscribe((state, prev) => {
  if (state.songEpoch === prev.songEpoch || restoringSnapshot) return;
  if (usePlanStore.getState().status !== 'idle') usePlanStore.getState().dismiss();
});
