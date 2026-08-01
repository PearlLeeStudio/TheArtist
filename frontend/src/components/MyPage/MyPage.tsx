import OperationProgress from '../OperationProgress';
import { useEffect, useRef, useState } from 'react';
import { useSessionStore } from '../../store/sessionStore';
import { useSongStore } from '../../store/songStore';
import { useGenerationStore } from '../../store/generationStore';
import { usePlayStore } from '../../store/playStore';
import {
  buildExportSpec, downloadExportBlob, requestMidiExport, safeExportFilename, snapshotSessionExport,
} from '../../engine/exportSpec';
import {
  applyPlayConfig, schedulePlayback, startPlayback, stopPlayback, pausePlayback, resumePlayback,
} from '../../engine/playback';
import { computeArrangement } from '../../engine/arrangement';
import { enginePlayConfig, playSettingsFromSession, resolveVoicingMode } from '../../engine/playSettings';
import type { PlaySettings, SavedSession, Song } from '../../models/types';
import SessionStaff from './SessionStaff';
import ExportMenu from '../ScoreEditor/ExportMenu';

const buttonStyle = {
  background: 'var(--btn-bg)', color: 'var(--text-primary)', border: '1px solid var(--border)',
};
const primaryButtonStyle = { background: 'var(--text-heading)', color: 'var(--bg-primary)' };
const mutedStyle = { color: 'var(--text-muted)' };

const PLAYBACK_FAILED = 'Playback could not start. Please try again.';

interface Notice { error: boolean; message: string }

/** The thrown Error's own message, or the fallback for anything else. */
function errorMessage(error: unknown, fallback: string): string {
  return error instanceof Error ? error.message : fallback;
}

function errorNotice(error: unknown, fallback: string): Notice {
  return { error: true, message: errorMessage(error, fallback) };
}

/** The genre the session was generated with, else the song's own. */
function sessionGenre(session: SavedSession): string | null {
  return session.generation.genre ?? session.song.genre ?? null;
}

/** The saved song with the session's genre, as it opens, plays and exports. */
function songWithSessionGenre(session: SavedSession): Song {
  return { ...session.song, genre: sessionGenre(session) };
}

/** Key · genre · tempo · meter · length, for a project card. */
function sessionSummary(session: SavedSession): string {
  const { song } = session;
  return [
    song.key, sessionGenre(session)?.replaceAll('_', ' '), `${song.bpm} BPM`,
    song.timeSignature.join('/'), `${song.measures.length} bars`,
  ].filter(Boolean).join(' · ');
}

/**
 * Schedule one pass of a project with its saved instruments and levels. The
 * saved bass pattern is used as is (loadAll drops an invalid cache); a
 * session without one gets a fresh pattern.
 */
function scheduleSession(session: SavedSession, onEnd: () => void): void {
  const song = songWithSessionGenre(session);
  const genre = song.genre;
  const settings = playSettingsFromSession(session.generation);
  const arrangement = session.arrangement ?? computeArrangement(song);
  applyPlayConfig(enginePlayConfig(settings, genre));
  schedulePlayback(song, () => {}, undefined, resolveVoicingMode(settings, genre), onEnd, arrangement, false);
}

/** True when a saved project already holds this song, model and Play settings.
 *  components/Layout.tsx has the same check (also isAlreadySaved) before an
 *  import replaces the score; keep the two in step. */
function isAlreadySaved(song: Song, model: string, settings: PlaySettings): boolean {
  const songJson = JSON.stringify(song);
  const settingsJson = JSON.stringify(settings);
  return useSessionStore.getState().sessions.some((saved) =>
    JSON.stringify(saved.song) === songJson && saved.generation.model === model &&
    JSON.stringify(playSettingsFromSession(saved.generation)) === settingsJson);
}

/** Replace the editor's song, model and Play settings with the project's. */
function openInEditor(session: SavedSession): void {
  useSongStore.getState().setSong(songWithSessionGenre(session));
  useGenerationStore.getState().setModelKey(session.generation.model);
  usePlayStore.getState().replaceSettings(playSettingsFromSession(session.generation));
}

interface SessionRowProps {
  session: SavedSession;
  activeSessionId: string | null;
  loadingSessionId: string | null;
  exportingId: string | null;
  isPaused: boolean;
  selected: boolean;
  onSelect: (id: string) => void;
  onTogglePlay: (session: SavedSession) => void;
  onOpen: (session: SavedSession) => void;
  onDelete: (session: SavedSession) => void;
  onFavorite: (session: SavedSession) => void;
  onExportMidi: (session: SavedSession) => void;
}

function SessionRow({
  session, activeSessionId, loadingSessionId, exportingId, isPaused, selected,
  onSelect, onTogglePlay, onOpen, onDelete, onFavorite, onExportMidi,
}: SessionRowProps) {
  const isActive = activeSessionId === session.id;
  const isLoading = loadingSessionId === session.id;
  const buttonLabel = isLoading ? 'Loading…' : !isActive ? '▶ Play' : isPaused ? '▶ Resume' : '⏸ Pause';

  return (
    <article className="p-4 space-y-3" style={{
      background: 'var(--bg-surface)', border: '1px solid var(--border)',
      borderLeft: session.starred ? '3px solid var(--brand-yellow)' : '1px solid var(--border)',
      outline: selected ? '2px solid var(--text-heading)' : undefined,
      outlineOffset: selected ? '2px' : undefined,
    }}>
      <div className="flex items-start gap-3 flex-wrap">
        <input type="checkbox" checked={selected} onChange={() => onSelect(session.id)}
          className="mt-1.5 h-4 w-4 shrink-0 cursor-pointer" aria-label={`Select ${session.name}`} />
        <button onClick={() => onFavorite(session)} className="text-xl leading-none shrink-0 mt-1"
          style={{ color: session.starred ? 'var(--brand-yellow)' : 'var(--text-muted)', background: 'transparent' }}
          title={session.starred ? 'Remove from favorites' : 'Favorite for future inspiration'}
          aria-label={`${session.starred ? 'Unfavorite' : 'Favorite'} ${session.name}`} aria-pressed={session.starred}>
          {session.starred ? '★' : '☆'}
        </button>
        <div className="flex-1 min-w-40">
          <h3 className="font-display text-xl break-words" style={{ color: 'var(--text-heading)', fontWeight: 600 }}>
            {session.name}
          </h3>
          <div className="text-xs mt-1" style={mutedStyle}>{sessionSummary(session)}</div>
        </div>
        <div className="flex flex-wrap gap-2">
          <button onClick={() => onTogglePlay(session)} disabled={loadingSessionId !== null}
            className="px-3 py-2 text-xs font-bold disabled:opacity-50"
            style={isActive ? primaryButtonStyle : buttonStyle}
            title={!isActive ? 'Play with the instruments and levels saved in this project' : isPaused ? 'Resume playback' : 'Pause playback'}>
            {buttonLabel}
          </button>
          <button onClick={() => onOpen(session)} className="px-3 py-2 text-xs font-bold" style={primaryButtonStyle}
            title="Edit in the editor; current work is preserved in Projects first">Edit</button>
          <button onClick={() => onExportMidi(session)} disabled={exportingId !== null}
            className="px-3 py-2 text-xs disabled:opacity-50" style={buttonStyle}
            title="Download harmony, bass and drums as a MIDI file for your DAW">
            {exportingId === session.id ? 'Exporting…' : 'MIDI'}
          </button>
          <button onClick={() => onDelete(session)} className="px-3 py-2 text-xs" style={buttonStyle}>Delete</button>
          <ExportMenu session={session} />
        </div>
      </div>
      {isLoading && <OperationProgress label="Loading the project's instrument and drum samples…" />}
      {exportingId === session.id && <OperationProgress label="Rendering the project's MIDI tracks…" />}
      <div className="overflow-x-auto" style={{ background: 'var(--bg-primary)', border: '1px solid var(--border)' }}>
        <SessionStaff song={session.song} />
      </div>
    </article>
  );
}

interface MyPageProps { onExitToEditor: () => void }

export default function MyPage({ onExitToEditor }: MyPageProps) {
  const sessions = useSessionStore((state) => state.sessions);
  const mountedRef = useRef(true);
  // Bumped by every play request, stop and mount; an async play step only
  // applies while its request is still the latest.
  const playSequence = useRef(0);
  const exportController = useRef<AbortController | null>(null);
  const selectAllRef = useRef<HTMLInputElement>(null);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [loadingSessionId, setLoadingSessionId] = useState<string | null>(null);
  const [isPaused, setIsPaused] = useState(false);
  const [exportingId, setExportingId] = useState<string | null>(null);
  const [notice, setNotice] = useState<Notice | null>(null);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(() => new Set());
  const starredCount = sessions.filter((session) => session.starred).length;
  const selectedSessions = sessions.filter((session) => selectedIds.has(session.id));
  const selectedCount = selectedSessions.length;
  const allSelected = sessions.length > 0 && selectedCount === sessions.length;

  useEffect(() => {
    if (selectAllRef.current) {
      selectAllRef.current.indeterminate = selectedCount > 0 && !allSelected;
    }
  }, [allSelected, selectedCount]);

  useEffect(() => {
    mountedRef.current = true;
    playSequence.current++;
    return () => {
      mountedRef.current = false;
      exportController.current?.abort();
      stopPlayback();
    };
  }, []);

  const stopProjectPlayback = () => {
    playSequence.current++;
    stopPlayback();
    setActiveSessionId(null);
    setLoadingSessionId(null);
    setIsPaused(false);
  };

  /** Deleting the playing or loading project stops it; other playback continues. */
  const stopPlaybackIfDeleted = (deletedIds: readonly string[]) => {
    if (deletedIds.some((id) => id === activeSessionId || id === loadingSessionId)) stopProjectPlayback();
  };

  const pauseOrResume = () => {
    try {
      if (isPaused) resumePlayback(); else pausePlayback();
      setIsPaused(!isPaused);
    } catch (error) {
      stopProjectPlayback();
      setNotice(errorNotice(error, PLAYBACK_FAILED));
    }
  };

  const handleTogglePlay = async (session: SavedSession) => {
    setNotice(null);
    if (activeSessionId === session.id && !loadingSessionId) {
      pauseOrResume();
      return;
    }
    const sequence = ++playSequence.current;
    const isLatestRequest = () => mountedRef.current && sequence === playSequence.current;
    stopPlayback();
    setActiveSessionId(session.id);
    setLoadingSessionId(session.id);
    setIsPaused(false);
    try {
      scheduleSession(session, () => {
        if (isLatestRequest()) {
          setActiveSessionId(null);
          setIsPaused(false);
        }
      });
      const started = await startPlayback();
      if (isLatestRequest() && !started) {
        setActiveSessionId(null);
        setNotice({ error: true, message: 'Playback could not start. Press Play to try again.' });
      }
    } catch (error) {
      if (isLatestRequest()) {
        stopPlayback();
        setActiveSessionId(null);
        setNotice(errorNotice(error, PLAYBACK_FAILED));
      }
    } finally {
      if (isLatestRequest()) setLoadingSessionId(null);
    }
  };

  const handleOpen = (session: SavedSession) => {
    // Keep the editor snapshot before replacing it. A failed browser save
    // throws before setSong, so the current score remains available.
    const current = useSongStore.getState().song;
    const model = useGenerationStore.getState().modelKey;
    const settings = usePlayStore.getState().settings;
    const alreadyKept = isAlreadySaved(current, model, settings);
    try {
      if (!alreadyKept) {
        useSessionStore.getState().add(current, { model, genre: current.genre, ...settings }, current.title.trim() || 'Untitled');
      }
    } catch (error) {
      const reason = errorMessage(error, 'Check browser storage, then try again.');
      setNotice({ error: true, message: `Your current work could not be saved, so this project was not opened. ${reason}` });
      return;
    }
    stopProjectPlayback();
    openInEditor(session);
    onExitToEditor();
  };

  const handleDelete = (session: SavedSession) => {
    if (!window.confirm(`Delete "${session.name}" from this browser?`)) return;
    try {
      useSessionStore.getState().remove(session.id);
      stopPlaybackIfDeleted([session.id]);
      setSelectedIds((previous) => {
        const remaining = new Set(previous);
        remaining.delete(session.id);
        return remaining;
      });
      setNotice({ error: false, message: `Deleted “${session.name}” from this browser.` });
    } catch (error) {
      setNotice(errorNotice(error, 'The project could not be deleted. Please try again.'));
    }
  };

  const handleDeleteSelected = () => {
    if (!selectedCount) return;
    const description = `${selectedCount} selected project${selectedCount === 1 ? '' : 's'}`;
    if (!window.confirm(`Delete ${description} from this browser? This cannot be undone.`)) return;
    try {
      const ids = selectedSessions.map((session) => session.id);
      useSessionStore.getState().removeMany(ids);
      stopPlaybackIfDeleted(ids);
      setSelectedIds(new Set());
      setNotice({ error: false, message: `Deleted ${description} from this browser.` });
    } catch (error) {
      setNotice(errorNotice(error, 'The selected projects could not be deleted. Please try again.'));
    }
  };

  const handleSelect = (id: string) => {
    setSelectedIds((previous) => {
      const next = new Set(previous);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  };

  const handleFavorite = (session: SavedSession) => {
    try {
      useSessionStore.getState().toggleStar(session.id);
      setNotice(null);
    } catch (error) {
      setNotice(errorNotice(error, 'This change could not be saved. Please try again.'));
    }
  };

  const handleExportMidi = async (session: SavedSession) => {
    if (exportController.current) return;
    const controller = new AbortController();
    exportController.current = controller;
    setExportingId(session.id);
    setNotice(null);
    try {
      const { song, settings } = snapshotSessionExport(session);
      const spec = buildExportSpec(song, settings);
      const blob = await requestMidiExport(spec, controller.signal);
      if (!mountedRef.current || controller.signal.aborted) return;
      downloadExportBlob(blob, safeExportFilename(session.name, 'mid'));
      setNotice({ error: false, message: `MIDI download started for “${session.name}”.` });
    } catch (error) {
      if (mountedRef.current && !controller.signal.aborted) {
        setNotice(errorNotice(error, 'MIDI export failed. Please try again.'));
      }
    } finally {
      if (exportController.current === controller) exportController.current = null;
      if (mountedRef.current) setExportingId(null);
    }
  };

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="max-w-5xl mx-auto p-6 space-y-5">
        <div className="flex items-start justify-between flex-wrap gap-4">
          <div className="max-w-xl">
            <h2 className="font-display text-3xl" style={{ color: 'var(--text-heading)', fontWeight: 700 }}>Projects</h2>
          </div>
        </div>
        {notice && (
          <p role={notice.error ? 'alert' : 'status'} className="text-sm px-3 py-2" style={{
            color: 'var(--text-primary)',
            background: 'var(--bg-surface)', border: '1px solid var(--border)',
            borderLeft: notice.error ? '3px solid var(--brand-yellow)' : undefined,
          }}>{notice.message}</p>
        )}
        {sessions.length > 0 ? (
          <>
            <div className="flex justify-between flex-wrap gap-2 text-xs" style={mutedStyle}>
              <span>{sessions.length} saved{starredCount > 0 ? ` · ★ ${starredCount} favorites` : ''}</span>
            </div>
            <div className="flex items-center flex-wrap gap-3 text-xs" style={{ color: 'var(--text-primary)' }}>
              <label className="flex items-center gap-2 cursor-pointer">
                <input ref={selectAllRef} type="checkbox" checked={allSelected}
                  aria-checked={selectedCount > 0 && !allSelected ? 'mixed' : allSelected}
                  onChange={(event) => setSelectedIds(event.target.checked ? new Set(sessions.map((session) => session.id)) : new Set())}
                  className="h-4 w-4" />
                Select all saved projects
              </label>
              <span aria-live="polite" style={mutedStyle}>{selectedCount} selected</span>
              <button onClick={() => setSelectedIds(new Set())} disabled={!selectedCount}
                className="px-3 py-2 disabled:opacity-50" style={buttonStyle}>Clear selection</button>
              <button onClick={handleDeleteSelected} disabled={!selectedCount}
                className="px-3 py-2 font-bold disabled:opacity-50" style={buttonStyle}>Delete selected</button>
            </div>
            <div className="space-y-3">
              {sessions.map((session) => (
                <SessionRow key={session.id} session={session} activeSessionId={activeSessionId}
                  loadingSessionId={loadingSessionId} exportingId={exportingId} isPaused={isPaused}
                  selected={selectedIds.has(session.id)} onSelect={handleSelect}
                  onTogglePlay={(item) => void handleTogglePlay(item)} onOpen={handleOpen}
                  onDelete={handleDelete} onFavorite={handleFavorite} onExportMidi={(item) => void handleExportMidi(item)} />
              ))}
            </div>
          </>
        ) : (
          <div className="py-16 text-center space-y-3" style={mutedStyle}>
            <p className="font-display text-2xl" style={{ color: 'var(--text-heading)' }}>Your next idea starts here.</p>
            <p className="text-sm">Save a sketch from the editor to keep it here.</p>
            <button onClick={onExitToEditor} className="px-4 py-2 text-xs font-bold" style={primaryButtonStyle}>Back to editor</button>
          </div>
        )}
      </div>
    </div>
  );
}
