import { useEffect, useLayoutEffect, useCallback, useRef, useState, useSyncExternalStore } from 'react';
import { initMidi, disconnectMidi, resetMidiNotes } from '../../engine/midiInput';
import type { MidiTransport } from '../../engine/midiInput';
import { createMidiCaptureState } from '../../engine/midiCaptureState';
import type { ConfirmState } from '../../engine/midiCaptureState';
import { useSongStore } from '../../store/songStore';
import { rankCandidatesByContext } from '../../engine/romanNumeral';
import { nextSlotAfter, chordAtPrevSlot, chordAtNextSlot } from '../../engine/slotNav';
import type { Chord, SlotAddress } from '../../models/types';

export { HOLD_DURATION_MS, PROGRESS_TICK_MS } from '../../engine/midiCaptureState';
export type { ConfirmState } from '../../engine/midiCaptureState';

export type ChordView =
  | { source: 'live'; candidates: Chord[]; activeIdx: number }
  | { source: 'confirm'; candidates: Chord[]; activeIdx: number }
  | null;

interface MidiConnection {
  devices: string[];
  enabled: boolean;
  connected: boolean;
  connecting: boolean;
  error: string | null;
}

export interface UseMidiCaptureResult extends MidiConnection {
  holdProgress: number;
  confirming: ConfirmState | null;
  awaitingRelease: boolean;
  canEnable: boolean;
  view: ChordView;
  active: Chord | null;
  prevSlotChord: Chord | null;
  nextSlotChord: Chord | null;
  songKey: string;
  songMeasureCount: number;
  toggleMidi: (transport?: MidiTransport) => Promise<void>;
  setActiveIdx: (i: number) => void;
  setConfirmActiveIdx: (i: number) => void;
  handleConfirm: () => void;
  handleCancel: () => void;
}

/** MIDI turned off, with the reason when connecting failed. */
function disconnected(error: string | null = null): MidiConnection {
  return { enabled: false, connected: false, connecting: false, error, devices: [] };
}

/** Capture target: the one selected slot, or null when zero or several are selected. */
function targetKey(slots: SlotAddress[]): string | null {
  return slots.length === 1 ? `${slots[0].measureIndex}:${slots[0].slotIndex}` : null;
}

/** The connection belongs to the editor. Capture is only armed while the
 *  MIDI input tab is visible; every other editor view only monitors notes. */
export function useMidiCapture(onMidiNotes: (notes: number[]) => void, captureActive: boolean): UseMidiCaptureResult {
  const [connection, setConnection] = useState(disconnected());
  const mountedRef = useRef(false);
  const enabledRef = useRef(false);
  const connectingRef = useRef(false);
  const requestRef = useRef(0);
  const captureActiveRef = useRef(captureActive);
  const onMidiNotesRef = useRef(onMidiNotes);
  const song = useSongStore((state) => state.song);
  const selectedSlots = useSongStore((state) => state.selectedSlots);

  const [capture] = useState(() => createMidiCaptureState({
    clock: {
      now: () => performance.now(),
      setInterval: (callback, delay) => window.setInterval(callback, delay),
      clearInterval: (id) => window.clearInterval(id),
    },
    onCandidates: (chords) => useSongStore.getState().setMidiCandidates(chords),
    // Write into the one selected slot and advance the selection. Refused
    // after unmount, with MIDI off, or while capture is not armed.
    onCommit: (chord) => {
      const state = useSongStore.getState();
      if (!mountedRef.current || !enabledRef.current || !captureActiveRef.current || state.selectedSlots.length !== 1) return false;
      const target = state.selectedSlots[0];
      state.setChord(target, chord);
      const next = nextSlotAfter(target, state.song.measures.length);
      if (next) state.selectSlot(next);
      return true;
    },
  }));
  const snapshot = useSyncExternalStore(capture.subscribe, capture.getSnapshot, capture.getSnapshot);

  useLayoutEffect(() => {
    onMidiNotesRef.current = onMidiNotes;
    captureActiveRef.current = captureActive;
    capture.setContext(captureActive && enabledRef.current, targetKey(useSongStore.getState().selectedSlots));
  }, [captureActive, onMidiNotes, capture]);

  useEffect(() => {
    mountedRef.current = true;
    requestRef.current++;
    const unsubscribe = useSongStore.subscribe((state, previous) => {
      const songReplaced = state.songEpoch !== previous.songEpoch;
      if (songReplaced) {
        capture.reset();
        resetMidiNotes();
        onMidiNotesRef.current([]);
      }
      const target = targetKey(state.selectedSlots);
      if (songReplaced || target !== targetKey(previous.selectedSlots)) {
        capture.setContext(captureActiveRef.current && enabledRef.current, target);
      }
    });
    return () => {
      mountedRef.current = false;
      enabledRef.current = false;
      connectingRef.current = false;
      unsubscribe();
      disconnectMidi();
      capture.dispose();
      useSongStore.getState().setMidiCandidates([]);
      onMidiNotesRef.current([]);
    };
  }, [capture]);

  const handleNotes = useCallback((notes: number[]) => {
    if (!mountedRef.current || !enabledRef.current) return;
    onMidiNotesRef.current(notes);
    capture.notes(notes);
  }, [capture]);

  const handleDetected = useCallback((chords: Chord[]) => {
    if (!mountedRef.current || !enabledRef.current) return;
    capture.detected(rankCandidatesByContext(chords, useSongStore.getState().song.key));
  }, [capture]);

  const toggleMidi = useCallback(async (transport: MidiTransport = 'auto') => {
    const request = ++requestRef.current;
    // A later toggle or an unmount supersedes this request.
    const isCurrent = () => mountedRef.current && request === requestRef.current;
    const disable = (error: string | null) => {
      enabledRef.current = false;
      connectingRef.current = false;
      disconnectMidi();
      capture.setContext(false, null);
      capture.reset();
      onMidiNotesRef.current([]);
      setConnection(disconnected(error));
    };

    if (enabledRef.current || connectingRef.current) {
      disable(null);
      return;
    }
    const target = targetKey(useSongStore.getState().selectedSlots);
    enabledRef.current = true;
    connectingRef.current = true;
    capture.setContext(captureActiveRef.current, target);
    setConnection({ enabled: true, connected: false, connecting: true, error: null, devices: [] });
    try {
      const connected = await initMidi(handleNotes, handleDetected, (value, devices) => {
        if (!isCurrent()) return;
        // Port changes can lose note-off events. Discard that press
        // cycle instead of offering it as a released chord to confirm.
        capture.reset();
        onMidiNotesRef.current([]);
        setConnection((state) => ({ ...state, connected: value, devices }));
      }, transport);
      if (!isCurrent()) return;
      connectingRef.current = false;
      setConnection((state) => ({ ...state, enabled: true, connected, connecting: false, error: null }));
    } catch (error) {
      if (!isCurrent()) return;
      disable(error instanceof Error ? error.message : 'MIDI could not connect. Please try again.');
    }
  }, [capture, handleNotes, handleDetected]);

  const { candidates, activeIdx, confirming } = snapshot;
  // A released chord under review takes precedence over the chord being held.
  const view: ChordView = confirming
    ? { source: 'confirm', candidates: confirming.candidates, activeIdx: confirming.activeIdx }
    : candidates.length ? { source: 'live', candidates, activeIdx } : null;
  const active = view ? view.candidates[view.activeIdx] ?? view.candidates[0] ?? null : null;
  const selection = selectedSlots[0] ?? null;

  return {
    ...connection,
    holdProgress: snapshot.holdProgress,
    confirming,
    awaitingRelease: snapshot.awaitingRelease,
    // Connection/monitoring is always available; writing needs one target.
    canEnable: !captureActive || selectedSlots.length === 1,
    view,
    active,
    prevSlotChord: chordAtPrevSlot(song.measures, selection),
    nextSlotChord: chordAtNextSlot(song.measures, selection),
    songKey: song.key,
    songMeasureCount: song.measures.length,
    toggleMidi,
    setActiveIdx: capture.choose,
    setConfirmActiveIdx: capture.chooseConfirm,
    handleConfirm: capture.confirm,
    handleCancel: capture.cancel,
  };
}
