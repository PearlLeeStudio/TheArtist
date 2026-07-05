import { useEffect, useCallback, useRef, useState } from 'react';
import { initMidi, disconnectMidi } from '../../engine/midiInput';
import { useSongStore } from '../../store/songStore';
import { rankCandidatesByContext } from '../../engine/romanNumeral';
import { nextSlotAfter, chordAtPrevSlot, chordAtNextSlot } from '../../engine/slotNav';
import type { Chord } from '../../models/types';

const HOLD_MS = 3000;
const TICK_MS = 50;

export interface ConfirmState {
  candidates: Chord[];
  activeIdx: number;
}

export type ChordView =
  | { source: 'live'; candidates: Chord[]; activeIdx: number }
  | { source: 'confirm'; candidates: Chord[]; activeIdx: number }
  | null;

export interface UseMidiCaptureResult {
  // Live state
  enabled: boolean;
  connected: boolean;
  holdProgress: number;
  confirming: ConfirmState | null;
  // Derived
  canEnable: boolean;
  view: ChordView;
  active: Chord | null;
  prevSlotChord: Chord | null;
  nextSlotChord: Chord | null;
  songKey: string;
  songMeasureCount: number;
  // Actions
  toggleMidi: () => Promise<void>;
  setActiveIdx: (i: number) => void;
  setConfirmActiveIdx: (i: number) => void;
  handleConfirm: () => void;
  handleCancel: () => void;
}

/**
 * MIDI capture state machine — the entire chord-detect / hold-timer /
 * confirm-flow logic for the MIDI input panel, lifted out of the
 * presenter component so each piece is independently legible.
 *
 * Two paths drive the state:
 *
 *   Fast path (≥3 s hold)   — `autoCommit` fires and writes the
 *                             active interpretation to the selected
 *                             slot, then advances selection.
 *   Slow path (release < 3s) — full release transitions into Confirm
 *                              mode, where the user can swap root
 *                              interpretations on the chip row before
 *                              clicking Confirm / Cancel.
 *
 * Two refs guard against MIDI's real-world physics:
 *
 *   `lastCandidatesRef`  — most recent non-empty candidates. Read by
 *                          the auto-commit timer at the moment it
 *                          fires.
 *   `peakCandidatesRef`  — candidates from the moment with the *most*
 *                          held notes during the press cycle. Humans
 *                          don't strike or release 4 keys
 *                          simultaneously; without this, the last
 *                          state before the release event would always
 *                          be smaller than the chord the user
 *                          intended. The Confirm flow reads peak.
 */
export function useMidiCapture(
  onMidiNotes: (notes: number[]) => void,
): UseMidiCaptureResult {
  const [enabled, setEnabled] = useState(false);
  const [connected, setConnected] = useState(false);
  const [activeIdx, setActiveIdx] = useState(0);
  const [holdProgress, setHoldProgress] = useState(0);
  const [confirming, setConfirming] = useState<ConfirmState | null>(null);

  const candidates = useSongStore((s) => s.midiCandidates);
  const setMidiCandidates = useSongStore((s) => s.setMidiCandidates);
  const songKey = useSongStore((s) => s.song.key);
  const selectedSlots = useSongStore((s) => s.selectedSlots);
  const setChord = useSongStore((s) => s.setChord);
  const selectSlot = useSongStore((s) => s.selectSlot);
  const song = useSongStore((s) => s.song);

  const canEnable = selectedSlots.length === 1;

  // Refs the once-registered MIDI callbacks read through.
  const slotsRef = useRef(selectedSlots);
  useEffect(() => { slotsRef.current = selectedSlots; }, [selectedSlots]);
  const songRef = useRef(song);
  useEffect(() => { songRef.current = song; }, [song]);
  const onMidiNotesRef = useRef(onMidiNotes);
  useEffect(() => { onMidiNotesRef.current = onMidiNotes; }, [onMidiNotes]);
  const activeIdxRef = useRef(0);
  useEffect(() => { activeIdxRef.current = activeIdx; }, [activeIdx]);

  const lastCandidatesRef = useRef<Chord[]>([]);
  const peakCandidatesRef = useRef<Chord[]>([]);
  useEffect(() => {
    if (candidates.length === 0) return;
    lastCandidatesRef.current = candidates;
    const newSize = candidates[0]?.voicing?.length ?? 0;
    const peakSize = peakCandidatesRef.current[0]?.voicing?.length ?? 0;
    if (newSize >= peakSize) {
      peakCandidatesRef.current = candidates;
    }
  }, [candidates]);
  // Set when the 3 s hold timer fires; tells the release handler to
  // skip Confirm because commit already happened.
  const committedViaTimerRef = useRef(false);

  /** Reset all chord-press transient refs (after commit / song change /
   *  MIDI disable). */
  const resetCaptureRefs = () => {
    lastCandidatesRef.current = [];
    peakCandidatesRef.current = [];
    committedViaTimerRef.current = false;
  };

  // Drop transient state when the song is replaced (Initialize / Clear).
  useEffect(() => {
    setActiveIdx(0);
    setHoldProgress(0);
    setConfirming(null);
    resetCaptureRefs();
  }, [song.id]);

  /** Write `chord` into the currently selected slot and advance. */
  const commitChord = useCallback(
    (chord: Chord) => {
      const slots = slotsRef.current;
      if (slots.length !== 1) return;
      setChord(slots[0], chord);
      const next = nextSlotAfter(slots[0], songRef.current.measures.length);
      if (next) selectSlot(next);
    },
    [setChord, selectSlot],
  );

  // Auto-commit when the 3 s hold timer fires. Uses peak candidates
  // (largest-cardinality reading during the hold).
  const autoCommit = useCallback(() => {
    const cands = peakCandidatesRef.current.length > 0
      ? peakCandidatesRef.current
      : lastCandidatesRef.current;
    if (cands.length === 0) return;
    const idx = activeIdxRef.current;
    const chord = cands[Math.min(idx, cands.length - 1)] ?? cands[0];
    commitChord(chord);
    setMidiCandidates([]);
    setActiveIdx(0);
    setHoldProgress(0);
    lastCandidatesRef.current = [];
    peakCandidatesRef.current = [];
    committedViaTimerRef.current = true;
  }, [commitChord, setMidiCandidates]);

  // Detected-chord callback: rank by context and reset hold timer for
  // a fresh 3 s window. Implicitly cancels Confirm if active.
  const handleChordDetected = useCallback(
    (chords: Chord[]) => {
      if (chords.length === 0) return;
      setMidiCandidates(rankCandidatesByContext(chords, songKey));
      setActiveIdx(0);
      setConfirming(null);
      committedViaTimerRef.current = false;
    },
    [setMidiCandidates, songKey],
  );

  // Note-state callback: forward live notes for VoicingViewer; on full
  // release either skip (already auto-committed) or enter Confirm.
  // Partial-release events do NOT touch `committedViaTimerRef` —
  // dropping fingers one at a time would otherwise clear it and the
  // Confirm card would pop after a 3 s hold.
  const handleNotes = useCallback(
    (notes: number[]) => {
      onMidiNotesRef.current(notes);
      if (notes.length > 0) return;
      if (committedViaTimerRef.current) {
        committedViaTimerRef.current = false;
        peakCandidatesRef.current = [];
        lastCandidatesRef.current = [];
        return;
      }
      const cands = peakCandidatesRef.current;
      if (cands.length > 0) {
        setConfirming({ candidates: cands, activeIdx: activeIdxRef.current });
      }
      setMidiCandidates([]);
      setActiveIdx(0);
      setHoldProgress(0);
      lastCandidatesRef.current = [];
      peakCandidatesRef.current = [];
    },
    [setMidiCandidates],
  );

  const toggleMidi = useCallback(async () => {
    if (enabled) {
      disconnectMidi();
      setEnabled(false);
      setConnected(false);
      onMidiNotesRef.current([]);
      setMidiCandidates([]);
      setActiveIdx(0);
      setHoldProgress(0);
      setConfirming(null);
      resetCaptureRefs();
    } else {
      const success = await initMidi(handleNotes, handleChordDetected);
      setConnected(success);
      setEnabled(true);
    }
  }, [enabled, handleNotes, handleChordDetected, setMidiCandidates]);

  useEffect(() => () => disconnectMidi(), []);

  // Auto-disable MIDI when the slot count breaks the 1-slot invariant
  // (commit-on-release needs exactly one target slot).
  useEffect(() => {
    if (enabled && !canEnable) {
      disconnectMidi();
      setEnabled(false);
      setConnected(false);
      onMidiNotesRef.current([]);
      setMidiCandidates([]);
      setActiveIdx(0);
      setHoldProgress(0);
      setConfirming(null);
      resetCaptureRefs();
    }
  }, [enabled, canEnable, setMidiCandidates]);

  // Hold timer: while a chord is held, fill 0→1 over HOLD_MS, then
  // auto-commit. Resets on chord change or Confirm-mode entry.
  const holdStartRef = useRef<number | null>(null);
  const primarySig = candidates[0]
    ? `${candidates[0].root}${candidates[0].quality}`
    : '';
  useEffect(() => {
    if (!enabled || !canEnable || !primarySig || confirming) {
      holdStartRef.current = null;
      setHoldProgress(0);
      return;
    }
    holdStartRef.current = Date.now();
    setHoldProgress(0);

    const tickId = window.setInterval(() => {
      if (holdStartRef.current === null) return;
      const elapsed = Date.now() - holdStartRef.current;
      const p = Math.min(1, elapsed / HOLD_MS);
      setHoldProgress(p);
      if (p >= 1) {
        window.clearInterval(tickId);
        autoCommit();
      }
    }, TICK_MS);
    return () => window.clearInterval(tickId);
  }, [primarySig, enabled, canEnable, confirming, autoCommit]);

  // ─── Confirm actions ────────────────────────────────────────────────

  const handleConfirm = () => {
    if (!confirming) return;
    const chord = confirming.candidates[confirming.activeIdx]
      ?? confirming.candidates[0];
    if (chord) commitChord(chord);
    setConfirming(null);
  };

  const handleCancel = () => setConfirming(null);

  const setConfirmActiveIdx = (i: number) => {
    if (!confirming) return;
    setConfirming({ ...confirming, activeIdx: i });
  };

  // ─── Derived view state ─────────────────────────────────────────────

  const view: ChordView = confirming
    ? { source: 'confirm', candidates: confirming.candidates, activeIdx: confirming.activeIdx }
    : candidates.length > 0
      ? { source: 'live', candidates, activeIdx }
      : null;

  const active = view ? (view.candidates[view.activeIdx] ?? view.candidates[0] ?? null) : null;
  const sel = selectedSlots[0] ?? null;
  const prevSlotChord = chordAtPrevSlot(song.measures, sel);
  const nextSlotChord = chordAtNextSlot(song.measures, sel);

  return {
    enabled,
    connected,
    holdProgress,
    confirming,
    canEnable,
    view,
    active,
    prevSlotChord,
    nextSlotChord,
    songKey,
    songMeasureCount: song.measures.length,
    toggleMidi,
    setActiveIdx,
    setConfirmActiveIdx,
    handleConfirm,
    handleCancel,
  };
}

export const HOLD_DURATION_MS = HOLD_MS;
export const PROGRESS_TICK_MS = TICK_MS;
