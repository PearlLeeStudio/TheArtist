import { useState, useCallback, useRef } from 'react';
import { useSongStore } from '../../store/songStore';
import type { Chord, SlotAddress } from '../../models/types';
import {
  schedulePlayback,
  startPlayback,
  stopPlayback,
  pausePlayback,
  resumePlayback,
  getIsPlaying,
  getIsPaused,
  setLoopEnabled,
} from '../../engine/playback';

function songHasAnyChord(measures: { chords: (unknown | null)[] }[]): boolean {
  return measures.some(m => m.chords.some(c => c !== null));
}

export default function PlaybackControls() {
  const song = useSongStore((s) => s.song);
  const selectedSlots = useSongStore((s) => s.selectedSlots);
  const instrument = useSongStore((s) => s.instrument);
  const setPlaybackPosition = useSongStore((s) => s.setPlaybackPosition);
  const selectSlot = useSongStore((s) => s.selectSlot);
  const playing = useSongStore((s) => s.isPlaying);
  const setPlaying = useSongStore((s) => s.setIsPlaying);
  const [loop, setLoop] = useState(false);

  const onPosition = useCallback((measureIndex: number, slotIndex: 0 | 1, _chord: Chord | null) => {
    setPlaybackPosition({ measureIndex, slotIndex });
  }, [setPlaybackPosition]);

  // Track which selection the most-recent schedule was built against, so
  // we can tell whether the user changed it during pause. Serialised to a
  // stable key for cheap comparison.
  const lastScheduledKey = useRef<string>('');
  const slotsKey = (slots?: SlotAddress[]): string =>
    slots ? slots.map(s => `${s.measureIndex}-${s.slotIndex}`).join(',') : 'all';

  // Main play button: selection-aware with pause/resume that honours
  // selection changes. Behaviour:
  //   playing                                 → pause
  //   paused, selection unchanged             → resume from paused position
  //   paused, selection changed since pause   → stop + reschedule for new selection
  //   idle                                    → new schedule from current selection
  const handlePlay = useCallback(() => {
    const slots = selectedSlots.length > 0 ? selectedSlots : undefined;
    const key = slotsKey(slots);

    if (getIsPlaying()) {
      pausePlayback();
      setPlaying(false);
      return;
    }
    if (getIsPaused() && key === lastScheduledKey.current) {
      resumePlayback();
      setPlaying(true);
      return;
    }
    // Idle, or paused with a different selection — fresh schedule.
    if (getIsPaused()) stopPlayback();
    lastScheduledKey.current = key;
    if (!songHasAnyChord(song.measures)) {
      stopPlayback();
      setPlaying(false);
      setPlaybackPosition(null);
      return;
    }
    schedulePlayback(song, onPosition, slots, instrument, () => setPlaying(false));
    startPlayback().then(() => setPlaying(true)).catch(() => setPlaying(false));
  }, [selectedSlots, song, onPosition, instrument, setPlaying, setPlaybackPosition]);

  // Secondary play button: always start from the top of the whole song,
  // ignoring whatever selection is active. Stops any current playback
  // first so the schedule is rebuilt from bar 1, and updates the
  // last-scheduled key so the main play button's resume logic stays
  // consistent.
  const handlePlayFullSong = useCallback(() => {
    stopPlayback();
    setPlaying(false);
    lastScheduledKey.current = slotsKey(undefined);
    if (!songHasAnyChord(song.measures)) {
      setPlaybackPosition(null);
      return;
    }
    schedulePlayback(song, onPosition, undefined, instrument, () => setPlaying(false));
    startPlayback().then(() => setPlaying(true)).catch(() => setPlaying(false));
  }, [song, onPosition, instrument, setPlaying, setPlaybackPosition]);

  const handleStop = useCallback(() => {
    stopPlayback();
    setPlaying(false);
    setPlaybackPosition(null);
  }, [setPlaybackPosition]);

  const handleRewind4 = useCallback(async () => {
    const current = useSongStore.getState().playbackPosition;
    const mi = current ? current.measureIndex : 0;
    const newMi = Math.max(0, mi - 4);
    selectSlot({ measureIndex: newMi, slotIndex: 0 });
    if (getIsPlaying()) {
      stopPlayback();
      setPlaying(false);
      const slotsFrom: SlotAddress[] = [];
      for (let i = newMi; i < song.measures.length; i++) {
        slotsFrom.push({ measureIndex: i, slotIndex: 0 });
        slotsFrom.push({ measureIndex: i, slotIndex: 1 });
      }
      schedulePlayback(song, onPosition, slotsFrom, undefined, () => setPlaying(false));
      await startPlayback();
      setPlaying(true);
    }
  }, [song, onPosition, selectSlot]);

  const handleForward4 = useCallback(async () => {
    const current = useSongStore.getState().playbackPosition;
    const mi = current ? current.measureIndex : 0;
    const newMi = Math.min(song.measures.length - 1, mi + 4);
    selectSlot({ measureIndex: newMi, slotIndex: 0 });
    if (getIsPlaying()) {
      stopPlayback();
      setPlaying(false);
      const slotsFrom: SlotAddress[] = [];
      for (let i = newMi; i < song.measures.length; i++) {
        slotsFrom.push({ measureIndex: i, slotIndex: 0 });
        slotsFrom.push({ measureIndex: i, slotIndex: 1 });
      }
      schedulePlayback(song, onPosition, slotsFrom, undefined, () => setPlaying(false));
      await startPlayback();
      setPlaying(true);
    }
  }, [song, onPosition, selectSlot, instrument]);

  const handleLoop = useCallback(() => {
    const newLoop = !loop;
    setLoop(newLoop);
    setLoopEnabled(newLoop);
  }, [loop]);

  // Inline monochrome SVG icons — they inherit `color` so the glyph
  // tone always matches the surrounding text instead of the OS's
  // colour-emoji rendering of U+23EE / U+1F501 etc. All icons share a
  // 16×16 viewBox so they sit at the same optical size; the side
  // buttons render at 18px and the primary play/pause renders at 22px
  // so it reads as the dominant action.
  const SIDE = 20;
  const MAIN = 26;
  const goStart = (
    <svg width={SIDE} height={SIDE} viewBox="0 0 16 16" fill="currentColor" aria-hidden>
      <rect x="2.5" y="3" width="2" height="10" rx="0.5" />
      <path d="M14 3v10L6 8l8-5z" />
    </svg>
  );
  const rewind = (
    <svg width={SIDE} height={SIDE} viewBox="0 0 16 16" fill="currentColor" aria-hidden>
      <path d="M9 3v10L1 8l8-5z" />
      <path d="M15 3v10L7 8l8-5z" />
    </svg>
  );
  const playGlyph = (
    <svg width={MAIN} height={MAIN} viewBox="0 0 16 16" fill="currentColor" aria-hidden>
      <path d="M3.5 2.2v11.6L13.7 8 3.5 2.2z" />
    </svg>
  );
  const pauseGlyph = (
    <svg width={MAIN} height={MAIN} viewBox="0 0 16 16" fill="currentColor" aria-hidden>
      <rect x="3.5" y="2" width="3" height="12" rx="0.5" />
      <rect x="9.5" y="2" width="3" height="12" rx="0.5" />
    </svg>
  );
  const forward = (
    <svg width={SIDE} height={SIDE} viewBox="0 0 16 16" fill="currentColor" aria-hidden>
      <path d="M1 3v10l8-5-8-5z" />
      <path d="M7 3v10l8-5-8-5z" />
    </svg>
  );
  // Single circular arrow — the universal "repeat" mark. Easier to parse
  // at-a-glance than the previous two-arc shape.
  const loopIcon = (
    <svg width={SIDE} height={SIDE} viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M13.2 8a5.2 5.2 0 1 1-1.4-3.6" />
      <polyline points="13.5,2 13.5,5 10.5,5" />
    </svg>
  );
  // "Play from start" — bar (the start marker) + right-facing triangle.
  // Mirrors the goStart icon (bar + left-facing triangle = "go to start"),
  // so the pair reads as: ⏮ goes to bar 1 silently, this plays from bar 1.
  const playFull = (
    <svg width={SIDE} height={SIDE} viewBox="0 0 16 16" fill="currentColor" aria-hidden>
      <rect x="2" y="3" width="2" height="10" rx="0.5" />
      <path d="M5 3v10l8-5z" />
    </svg>
  );

  const ghostBtn: React.CSSProperties = { background: 'var(--btn-bg)', color: 'var(--text-primary)' };
  const filledBtn: React.CSSProperties = { background: 'var(--text-heading)', color: 'var(--bg-primary)' };

  return (
    <div className="space-y-1.5">
      <h3 className="text-xs uppercase tracking-wide font-semibold" style={{ color: 'var(--text-heading)' }}>Playback</h3>
      <div className="flex items-center gap-1.5">
        <button onClick={handleStop}    className="px-3 py-2.5  flex items-center justify-center" style={ghostBtn} title="Go to start">{goStart}</button>
        <button onClick={handleRewind4} className="px-3 py-2.5  flex items-center justify-center" style={ghostBtn} title="Back 4 measures">{rewind}</button>
        <button
          onClick={handlePlay}
          className="flex-1 px-4 py-2.5  flex items-center justify-center"
          style={filledBtn}
          title={playing
            ? 'Pause'
            : selectedSlots.length > 0
              ? `Play selection (${selectedSlots.length} slot${selectedSlots.length > 1 ? 's' : ''})`
              : 'Play full song'}
        >
          {playing ? pauseGlyph : playGlyph}
        </button>
        <button onClick={handleForward4} className="px-3 py-2.5  flex items-center justify-center" style={ghostBtn} title="Forward 4 measures">{forward}</button>
        <button
          onClick={handleLoop}
          className="px-3 py-2.5  flex items-center justify-center"
          style={loop ? filledBtn : ghostBtn}
          title="Loop"
        >
          {loopIcon}
        </button>
        <button
          onClick={handlePlayFullSong}
          className="px-3 py-2.5  flex items-center justify-center"
          style={ghostBtn}
          title="Play full song (ignore selection)"
        >
          {playFull}
        </button>
      </div>
    </div>
  );
}
