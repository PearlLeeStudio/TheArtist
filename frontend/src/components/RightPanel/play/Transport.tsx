/**
 * Play-panel transport: stop and go to start, back 4
 * measures, play/pause, forward 4 measures, loop, play the whole song.
 * Play follows the staff selection: it pauses while playing, resumes a
 * paused schedule while the selection and the score are unchanged, and
 * otherwise schedules the selection (or the whole song) afresh. Every
 * schedule carries the Play-panel settings, voicing and bass layer.
 */
import { useCallback, useState, useSyncExternalStore } from 'react';
import OperationProgress from '../../OperationProgress';
import { getPlaybackStatus, subscribePlaybackStatus } from '../../../engine/playbackStatus';
import type { ReactNode } from 'react';
import type { SlotAddress, Song } from '../../../models/types';
import { useSongStore } from '../../../store/songStore';
import { applyEditorPlayConfig, usePlayStore } from '../../../store/playStore';
import { computeArrangement } from '../../../engine/arrangement';
import { resolveVoicingMode } from '../../../engine/playSettings';
import {
  getIsPaused,
  getIsPlaying,
  getLoopEnabled,
  pausePlayback,
  resumePlayback,
  schedulePlayback,
  setLiveBpm,
  setLoopEnabled,
  startPlayback,
  stopPlayback,
} from '../../../engine/playback';
import KeyButton from './ui/KeyButton';
import './Transport.css';

/** Measures the back and forward keys move. */
const JUMP = 4;
/** Six key columns; play/pause, the third, is the widest. */
const COLUMNS = 'repeat(2, minmax(0, 1fr)) minmax(0, 1.65fr) repeat(3, minmax(0, 1fr))';
const KEY_STYLE = { height: 42 };

// Monochrome 16×16 glyphs in currentColor; play/pause draws a little larger.
const ICON = 16;
const MAIN_ICON = 18;

const STOP_ICON = (
  <svg width={ICON} height={ICON} viewBox="0 0 16 16" fill="currentColor" aria-hidden>
    <rect x="3" y="3" width="10" height="10" rx="0.5" />
  </svg>
);
const BACK_ICON = (
  <svg width={ICON} height={ICON} viewBox="0 0 16 16" fill="currentColor" aria-hidden>
    <path d="M9 3v10L1 8l8-5z" />
    <path d="M15 3v10L7 8l8-5z" />
  </svg>
);
const PLAY_ICON = (
  <svg width={MAIN_ICON} height={MAIN_ICON} viewBox="0 0 16 16" fill="currentColor" aria-hidden>
    <path d="M3.5 2.2v11.6L13.7 8 3.5 2.2z" />
  </svg>
);
const PAUSE_ICON = (
  <svg width={MAIN_ICON} height={MAIN_ICON} viewBox="0 0 16 16" fill="currentColor" aria-hidden>
    <rect x="3.5" y="2" width="3" height="12" rx="0.5" />
    <rect x="9.5" y="2" width="3" height="12" rx="0.5" />
  </svg>
);
const FORWARD_ICON = (
  <svg width={ICON} height={ICON} viewBox="0 0 16 16" fill="currentColor" aria-hidden>
    <path d="M1 3v10l8-5-8-5z" />
    <path d="M7 3v10l8-5-8-5z" />
  </svg>
);
const LOOP_ICON = (
  <svg
    width={ICON}
    height={ICON}
    viewBox="0 0 16 16"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.6"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden
  >
    <path d="M13.2 8a5.2 5.2 0 1 1-1.4-3.6" />
    <polyline points="13.5,2 13.5,5 10.5,5" />
  </svg>
);
// Start bar + right triangle: plays from bar 1 (stop goes there silently).
const PLAY_FULL_ICON = (
  <svg width={ICON} height={ICON} viewBox="0 0 16 16" fill="currentColor" aria-hidden>
    <rect x="2" y="3" width="2" height="10" rx="0.5" />
    <path d="M5 3v10l8-5z" />
  </svg>
);

function songHasAnyChord(song: Song): boolean {
  return song.measures.some((m) => m.chords.some((c) => c !== null));
}

/** The staff selection in schedule form: undefined (the whole song) when
 *  nothing is selected. */
function selectionSlots(selectedSlots: SlotAddress[]): SlotAddress[] | undefined {
  return selectedSlots.length > 0 ? selectedSlots : undefined;
}

/** A selection as a comparable string; 'all' for the whole song. */
function slotsKey(slots: SlotAddress[] | undefined): string {
  return slots ? slots.map((s) => `${s.measureIndex}-${s.slotIndex}`).join(',') : 'all';
}

/** Every slot from `measureIndex` to the end of the song. */
function slotsFrom(song: Song, measureIndex: number): SlotAddress[] {
  const slots: SlotAddress[] = [];
  for (let mi = measureIndex; mi < song.measures.length; mi++) {
    slots.push({ measureIndex: mi, slotIndex: 0 }, { measureIndex: mi, slotIndex: 1 });
  }
  return slots;
}

/** A paused schedule still plays this score: same chords, genre and meter.
 *  Tempo is left out because the transport retimes it live. */
function sameScore(a: Song, b: Song): boolean {
  return a.measures === b.measures && a.genre === b.genre && a.timeSignature === b.timeSignature;
}

/** The selection key and song the running or paused schedule was built
 *  from. Module state like the engine's schedule, so a paused schedule
 *  still resumes after the Play panel unmounts and mounts again. */
let scheduled: { key: string; song: Song | null } = { key: '', song: null };

/** Play can resume the paused schedule: it was built from the selection
 *  keyed `key` and still plays this score. */
function canResume(key: string, song: Song): boolean {
  const last = scheduled;
  return getIsPaused() && key === last.key && last.song !== null && sameScore(last.song, song);
}

/** The play key's tooltip and accessible name. */
function playKeyTitle(playing: boolean, selectedCount: number): string {
  if (playing) return 'Pause';
  if (selectedCount === 0) return 'Play full song';
  return `Play selection (${selectedCount} slot${selectedCount > 1 ? 's' : ''})`;
}

interface TransportKeyProps {
  /** Accessible name, and the tooltip unless `title` differs. */
  label: string;
  title?: string;
  onClick: () => void;
  active?: boolean;
  /** The play/pause key, lit while the song plays. */
  play?: boolean;
  children: ReactNode;
}

/** One transport key: an md key at the transport's height, with an icon. */
function TransportKey({ label, title = label, onClick, active, play, children }: TransportKeyProps) {
  const className = play ? 'te-transport__key te-transport__key--play' : 'te-transport__key';
  return (
    <KeyButton className={className} size="md" style={KEY_STYLE} onClick={onClick} title={title} ariaLabel={label} active={active}>
      {children}
    </KeyButton>
  );
}

export default function Transport() {
  const audio = useSyncExternalStore(subscribePlaybackStatus, getPlaybackStatus);
  const song = useSongStore((s) => s.song);
  const selectedSlots = useSongStore((s) => s.selectedSlots);
  const playing = useSongStore((s) => s.isPlaying);
  const setPlaying = useSongStore((s) => s.setIsPlaying);
  const setPlaybackPosition = useSongStore((s) => s.setPlaybackPosition);
  const selectSlot = useSongStore((s) => s.selectSlot);
  const settings = usePlayStore((s) => s.settings);
  // The loop flag lives in the engine, which outlives this component.
  const [loop, setLoop] = useState(getLoopEnabled);

  const voicing = resolveVoicingMode(settings, song.genre);

  const onPosition = useCallback((measureIndex: number, slotIndex: 0 | 1) => {
    setPlaybackPosition({ measureIndex, slotIndex });
  }, [setPlaybackPosition]);

  // The end leaves no playhead, so the Visual pane and the back and
  // forward keys go back to the selection.
  const onEnd = useCallback(() => {
    setPlaying(false);
    setPlaybackPosition(null);
  }, [setPlaying, setPlaybackPosition]);

  /** Schedule `slots` (undefined = the whole song) without starting; `key`
   *  is the selection the main play key compares against to resume. */
  const schedule = useCallback((slots: SlotAddress[] | undefined, key: string) => {
    scheduled = { key, song };
    applyEditorPlayConfig();
    schedulePlayback(song, onPosition, slots, voicing, onEnd, computeArrangement(song));
  }, [song, onPosition, voicing, onEnd]);

  /** Schedule and start. A start called off while samples load (stop, pause,
   *  a new schedule or an edit) leaves the key showing play. */
  const play = useCallback((slots: SlotAddress[] | undefined, key: string) => {
    schedule(slots, key);
    startPlayback()
      .then((started) => { if (started) setPlaying(true); })
      .catch(() => setPlaying(false));
  }, [schedule, setPlaying]);

  // playing → pause; paused, nothing changed → resume; otherwise a fresh
  // schedule of the selection, or the whole song without one.
  const handlePlay = useCallback(() => {
    if (getIsPlaying()) {
      pausePlayback();
      setPlaying(false);
      return;
    }
    const slots = selectionSlots(selectedSlots);
    const key = slotsKey(slots);
    if (canResume(key, song)) {
      applyEditorPlayConfig();
      setLiveBpm(song.bpm || 80);
      resumePlayback();
      setPlaying(true);
      return;
    }
    stopPlayback();
    if (!songHasAnyChord(song)) {
      setPlaying(false);
      setPlaybackPosition(null);
      return;
    }
    play(slots, key);
  }, [selectedSlots, song, play, setPlaying, setPlaybackPosition]);

  // Always from bar 1 of the whole song, whatever is selected. Keyed to the
  // selection as it stands, so pause and play resume the song.
  const handlePlayFullSong = useCallback(() => {
    stopPlayback();
    setPlaying(false);
    if (!songHasAnyChord(song)) {
      setPlaybackPosition(null);
      return;
    }
    play(undefined, slotsKey(selectionSlots(selectedSlots)));
  }, [song, selectedSlots, play, setPlaying, setPlaybackPosition]);

  const handleStop = useCallback(() => {
    stopPlayback();
    setPlaying(false);
    setPlaybackPosition(null);
  }, [setPlaying, setPlaybackPosition]);

  // Move the cursor `bars` measures from the playhead while one is playing
  // or paused, else from the selection. Playing: carry on from the new
  // measure. Paused: wait there, so play carries on from it to the end.
  const jump = useCallback((bars: number) => {
    const live = getIsPlaying() || getIsPaused();
    const playhead = live ? useSongStore.getState().playbackPosition : null;
    const from = playhead?.measureIndex ?? selectedSlots[0]?.measureIndex ?? 0;
    const target = Math.max(0, Math.min(song.measures.length - 1, from + bars));
    const cursor: SlotAddress = { measureIndex: target, slotIndex: 0 };
    selectSlot(cursor);
    if (!live) return;
    // Keyed to the cursor selection, so pause and play resume this schedule.
    if (getIsPlaying()) {
      stopPlayback();
      setPlaying(false);
      play(slotsFrom(song, target), slotsKey([cursor]));
      return;
    }
    schedule(slotsFrom(song, target), slotsKey([cursor]));
    pausePlayback();
    setPlaybackPosition(cursor);
  }, [selectedSlots, song, selectSlot, play, schedule, setPlaying, setPlaybackPosition]);

  const handleBack = useCallback(() => jump(-JUMP), [jump]);
  const handleForward = useCallback(() => jump(JUMP), [jump]);

  const handleLoop = useCallback(() => {
    const next = !loop;
    setLoop(next);
    setLoopEnabled(next);
  }, [loop]);

  const playTitle = playKeyTitle(playing, selectedSlots.length);

  return (
    <div
      role="group"
      aria-label="Transport"
      className="te-transport"
      data-playing={playing ? 'true' : 'false'}
    >
      <div className="te-transport__keys" style={{ gridTemplateColumns: COLUMNS }}>
        <TransportKey label="Stop and go to start" onClick={handleStop}>{STOP_ICON}</TransportKey>
        <TransportKey label={`Back ${JUMP} measures`} onClick={handleBack}>{BACK_ICON}</TransportKey>
        <TransportKey play label={playTitle} onClick={handlePlay}>
          {playing ? PAUSE_ICON : PLAY_ICON}
        </TransportKey>
        <TransportKey label={`Forward ${JUMP} measures`} onClick={handleForward}>{FORWARD_ICON}</TransportKey>
        <TransportKey label="Loop" title={loop ? 'Loop on' : 'Loop off'} onClick={handleLoop} active={loop}>
          {LOOP_ICON}
        </TransportKey>
        <TransportKey label="Play full song from the start" title="Play full song (ignore selection)" onClick={handlePlayFullSong}>
          {PLAY_FULL_ICON}
        </TransportKey>
      </div>
      <div className="te-transport__disc" aria-hidden="true">💿</div>
      {audio.loading && <OperationProgress label={audio.loading} />}
      {audio.error && <p role="alert" className="text-xs" style={{ color: 'var(--text-primary)' }}>{audio.error}</p>}
    </div>
  );
}
