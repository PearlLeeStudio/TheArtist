/**
 * MyPage — full-page view of saved sessions.
 *
 * Each session renders as a read-only staff (SessionStaff) so the user
 * sees the actual progression at a glance, not just metadata. Per-session
 * actions: Play (full multi-track arrangement), Load (back to editor),
 * MIDI export (DAW handoff), Delete. Top bar: bulk JSON export/import for
 * cross-device transfer.
 */
import { useRef, useState, useCallback } from 'react';
import { useSessionStore } from '../../store/sessionStore';
import { useSongStore } from '../../store/songStore';
import { useGenerationStore } from '../../store/generationStore';
import { downloadSongAsMidi } from '../../engine/midiExport';
import {
  schedulePlayback,
  startPlayback,
  stopPlayback,
  pausePlayback,
  resumePlayback,
  setHarmonyOverride,
} from '../../engine/playback';
import { setCurrentDrumPattern, getDrumPatternForGenre } from '../../engine/drums';
import { computeArrangement } from '../../engine/arrangement';
import type { SavedSession } from '../../models/types';
import SessionStaff from './SessionStaff';

function formatRelativeDate(ts: number): string {
  return new Date(ts).toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

interface SessionRowProps {
  session: SavedSession;
  onAfterLoad: () => void;
  activeSessionId: string | null;
  isPaused: boolean;
  onTogglePlay: (session: SavedSession) => void;
}

function SessionRow({ session, onAfterLoad, activeSessionId, isPaused, onTogglePlay }: SessionRowProps) {
  const setSong = useSongStore((s) => s.setSong);
  const setModelKey = useGenerationStore((s) => s.setModelKey);
  const setDrumPatternId = useSongStore((s) => s.setDrumPatternId);
  const setHarmonyOverrideStore = useSongStore((s) => s.setHarmonyOverride);
  const remove = useSessionStore((s) => s.remove);
  const toggleStar = useSessionStore((s) => s.toggleStar);

  const isActive = activeSessionId === session.id;
  // Three states: idle (not active) | playing (active + not paused) | paused (active + paused)
  const buttonLabel = !isActive ? '▶ Play' : isPaused ? '▶ Resume' : '⏸ Pause';
  const buttonTitle = !isActive
    ? 'Play with full multi-track arrangement (chord + bass + melody + drum)'
    : isPaused
      ? 'Resume from paused position'
      : 'Pause (Tone.Transport pause — schedule retained for instant resume)';

  const handleLoad = () => {
    setSong(session.song);
    setModelKey(session.generation.model);
    // Pearl 2026-05-09: restore drum selection bundled with the session.
    // Pearl 2026-05-10: also restore harmony override so the editor
    // reflects the saved instrument. Melody was dropped — leftover
    // melodyOverride from old sessions ignored by everything downstream.
    setDrumPatternId(session.generation.drumPatternId ?? null);
    setHarmonyOverrideStore(session.generation.harmonyOverride ?? null);
    onAfterLoad();
  };

  const handleDelete = () => {
    if (window.confirm(`Delete "${session.name}"?`)) {
      remove(session.id);
    }
  };

  const meta = [
    session.song.key,
    session.generation.genre,
    session.generation.model,
  ]
    .filter(Boolean)
    .join(' · ');

  return (
    <div
      className="p-4 space-y-3"
      style={{
        background: 'var(--bg-surface)',
        border: '1px solid var(--border)',
        borderLeft: session.starred
          ? '3px solid var(--brand-yellow)'
          : '1px solid var(--border)',
      }}
    >
      <div className="flex items-start gap-3">
        <button
          onClick={() => toggleStar(session.id)}
          className="text-lg leading-none shrink-0 mt-0.5"
          style={{
            color: session.starred ? 'var(--brand-yellow)' : 'var(--text-muted)',
            background: 'transparent',
          }}
          title={session.starred ? 'Unstar' : 'Star (add to Favorite Generate pool)'}
        >
          {session.starred ? '★' : '☆'}
        </button>

        <div className="flex-1 min-w-0">
          <div className="font-display text-xl truncate" style={{ color: 'var(--text-heading)', fontWeight: 600 }}>
            {session.name}
          </div>
          <div className="text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>
            {meta} · {formatRelativeDate(session.createdAt)}
          </div>
        </div>

        <div className="flex gap-2 shrink-0">
          <button
            onClick={() => onTogglePlay(session)}
            className="px-3 py-1.5 text-xs font-bold"
            style={{
              background: isActive ? 'var(--text-heading)' : 'var(--btn-bg)',
              color: isActive ? 'var(--bg-primary)' : 'var(--text-primary)',
              border: '1px solid var(--border)',
            }}
            title={buttonTitle}
          >
            {buttonLabel}
          </button>
          <button
            onClick={handleLoad}
            className="px-3 py-1.5 text-xs font-bold"
            style={{ background: 'var(--text-heading)', color: 'var(--bg-primary)' }}
            title="Replace the current song with this saved session and return to the editor"
          >
            Load
          </button>
          <button
            onClick={() => downloadSongAsMidi(session.song, session.name)}
            className="px-3 py-1.5 text-xs"
            style={{ background: 'var(--btn-bg)', color: 'var(--text-primary)' }}
            title="Export to MIDI (.mid) — drag into Logic Pro or any DAW"
          >
            MIDI
          </button>
          <button
            onClick={handleDelete}
            className="px-3 py-1.5 text-xs"
            style={{ background: 'var(--btn-bg)', color: 'var(--text-muted)' }}
          >
            Delete
          </button>
        </div>
      </div>

      <div style={{ background: 'var(--bg-primary)', border: '1px solid var(--border)' }}>
        <SessionStaff song={session.song} />
      </div>
    </div>
  );
}

interface MyPageProps {
  onExitToEditor: () => void;
}

export default function MyPage({ onExitToEditor }: MyPageProps) {
  const sessions = useSessionStore((s) => s.sessions);
  const exportAll = useSessionStore((s) => s.exportAll);
  const importJson = useSessionStore((s) => s.importJson);
  const starredCount = sessions.filter((s) => s.starred).length;
  const fileInputRef = useRef<HTMLInputElement>(null);

  /** Track which session (if any) is currently the *active* playback —
   *  could be playing or paused. Pearl 2026-05-10 round 2: replaced
   *  the simple play/stop with a 3-state play/pause/resume toggle. */
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [isPaused, setIsPaused] = useState(false);

  const handleTogglePlay = useCallback((session: SavedSession) => {
    // Same session is currently active → toggle pause/resume (Tone.Transport
    // pause/start, schedule retained for instant resume).
    if (activeSessionId === session.id) {
      if (isPaused) {
        resumePlayback();
        setIsPaused(false);
      } else {
        pausePlayback();
        setIsPaused(true);
      }
      return;
    }

    // Different session (or first play) → full swap. stopPlayback is idempotent.
    stopPlayback();

    // Resolve effective genre: prefer the saved generation context, fall
    // back to the song's own genre tag. Used for drum + instrument routing.
    const effectiveGenre = session.generation.genre ?? session.song.genre ?? null;

    // Restore the user-picked harmony override saved with the session
    // so the chord track plays back on the exact GM instrument the user
    // A/B'd at Save time. null = fall through to genre default. The
    // playback engine reads song.genre directly (effectiveGenre is
    // already baked into songForPlay below), so no separate genre setter.
    setHarmonyOverride(session.generation.harmonyOverride ?? null);

    // Drum: Pearl 2026-05-10 — honour the saved drumPatternId first
    // (the user's last play-view selection bundled with the session).
    // Falls through to the genre's signature pattern if the session
    // predates that field.
    const savedDrum = session.generation.drumPatternId;
    if (savedDrum !== undefined && savedDrum !== null) {
      setCurrentDrumPattern(savedDrum);
    } else {
      const genreDrumPattern = effectiveGenre ? getDrumPatternForGenre(effectiveGenre) : null;
      setCurrentDrumPattern(genreDrumPattern?.id ?? null);
    }

    // Make sure the song passed to schedulePlayback / arrangement carries
    // the same effective genre — the bass + melody rhythm tables key on
    // song.genre, and the arrangement was computed against this same
    // resolution at save time.
    const songForPlay = { ...session.song, genre: effectiveGenre };
    // Round 3 (2026-05-10) reshaped Arrangement from ArrangementEvent[] to
    // a 1-bar pattern object. Detect legacy array-format and ignore so old
    // sessions saved between rounds 2 and 3 fall back to live compute
    // rather than crashing on `.bassPattern` access.
    const stored = session.arrangement;
    const arrangement = (stored && !Array.isArray(stored) && 'bassPattern' in stored)
      ? stored
      : computeArrangement(songForPlay);

    schedulePlayback(
      songForPlay,
      () => {/* no editor highlight on MyPage */},
      undefined,
      'piano',
      () => { setActiveSessionId(null); setIsPaused(false); },  // natural end
      arrangement,
    );
    void startPlayback();
    setActiveSessionId(session.id);
    setIsPaused(false);
  }, [activeSessionId, isPaused]);

  const handleExport = () => {
    const json = exportAll();
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    const dateTag = new Date().toISOString().slice(0, 10);
    anchor.href = url;
    anchor.download = `theartist-sessions-${dateTag}.json`;
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
    setTimeout(() => URL.revokeObjectURL(url), 0);
  };

  const handleImportFile = async (file: File) => {
    try {
      const text = await file.text();
      const result = importJson(text);
      const parts = [
        result.added > 0 ? `${result.added} added` : null,
        result.renamed > 0 ? `${result.renamed} re-keyed (id collision)` : null,
        result.skipped > 0 ? `${result.skipped} skipped (invalid)` : null,
      ].filter(Boolean);
      window.alert(parts.length > 0 ? `Import complete: ${parts.join(', ')}.` : 'No sessions imported.');
    } catch (err) {
      window.alert(`Import failed: ${err instanceof Error ? err.message : 'unknown error'}`);
    }
  };

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="max-w-5xl mx-auto p-6 space-y-4">
        <div className="flex items-baseline justify-between flex-wrap gap-2">
          <div>
            <h2 className="font-display text-3xl" style={{ color: 'var(--text-heading)', fontWeight: 700 }}>
              My Sessions
            </h2>
            {sessions.length > 0 && (
              <div className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
                {starredCount > 0 ? `★ ${starredCount} starred · ${sessions.length} total` : `${sessions.length} saved`}
              </div>
            )}
          </div>

          <div className="flex gap-2">
            <button
              onClick={handleExport}
              disabled={sessions.length === 0}
              className="px-3 py-1.5 text-xs"
              style={{
                background: 'var(--btn-bg)',
                color: 'var(--text-primary)',
                border: '1px solid var(--border)',
                opacity: sessions.length === 0 ? 0.5 : 1,
              }}
              title="Download all sessions as a JSON file"
            >
              Export JSON…
            </button>
            <button
              onClick={() => fileInputRef.current?.click()}
              className="px-3 py-1.5 text-xs"
              style={{
                background: 'var(--btn-bg)',
                color: 'var(--text-primary)',
                border: '1px solid var(--border)',
              }}
              title="Import sessions from a previously-exported JSON file"
            >
              Import JSON…
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept="application/json,.json"
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) void handleImportFile(file);
                e.target.value = '';
              }}
            />
          </div>
        </div>

        {sessions.length === 0 ? (
          <p className="text-sm italic py-12 text-center" style={{ color: 'var(--text-muted)' }}>
            No saved sessions yet. From the editor, press Save (top right of the staff) to capture the current song and generation settings.
          </p>
        ) : (
          <div className="space-y-4">
            {sessions.map((s) => (
              <SessionRow
                key={s.id}
                session={s}
                onAfterLoad={onExitToEditor}
                activeSessionId={activeSessionId}
                isPaused={isPaused}
                onTogglePlay={handleTogglePlay}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
