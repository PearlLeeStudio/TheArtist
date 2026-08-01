import { useState, useCallback, useRef } from 'react';
import './Layout.css';
import Staff from './ScoreEditor/Staff';
import SongFormDialog from './ScoreEditor/SongFormDialog';
import ImportScoreDialog from './ScoreEditor/ImportScoreDialog';
import SectionsDialog from './ScoreEditor/SectionsDialog';
import ExportMenu from './ScoreEditor/ExportMenu';
import PlayPanel from './RightPanel/play/PlayPanel';
import SongSettings from './RightPanel/SongSettings';
import ChordActions from './RightPanel/ChordActions';
import MidiStatus from './RightPanel/MidiStatus';
import MyPage from './MyPage/MyPage';
import ThemeToggle from './RightPanel/ThemeToggle';
import { useSongStore } from '../store/songStore';
import { useSessionStore } from '../store/sessionStore';
import { useGenerationStore } from '../store/generationStore';
import { usePlayStore } from '../store/playStore';
import { useImportCleanupStore } from '../store/importCleanupStore';
import { startNewSongFromForm } from '../store/newSong';
import { playSettingsFromSession } from '../engine/playSettings';
import type { PlaySettings, SessionGenerationContext, Song } from '../models/types';

type InputMode = 'generate' | 'reharmonize' | 'midi' | 'manual';
const INPUT_TABS: { id: InputMode; label: string }[] = [
  { id: 'generate', label: 'Generate' },
  { id: 'reharmonize', label: 'Reharmonize' },
];
const ADDITIONAL_INPUTS: { id: InputMode; label: string }[] = [
  { id: 'midi', label: 'MIDI' },
  { id: 'manual', label: 'Set manually' },
];

/** The Play panel needs this much width for its rail and one pane. */
const MIN_PANEL_WIDTH = 300;
const MAX_PANEL_WIDTH = 640;

/** What a saved project keeps next to its song: the model, the song's genre
 *  and the Play settings. */
function generationContext(song: Song, model: string, settings: PlaySettings): SessionGenerationContext {
  return { model, genre: song.genre, ...settings };
}

/** True when a saved project already holds this song, model and Play settings.
 *  MyPage/MyPage.tsx has the same check (also isAlreadySaved) before opening a
 *  project; keep the two in step. */
function isAlreadySaved(song: Song, model: string, settings: PlaySettings): boolean {
  const songJson = JSON.stringify(song);
  const settingsJson = JSON.stringify(settings);
  return useSessionStore.getState().sessions.some((saved) =>
    JSON.stringify(saved.song) === songJson && saved.generation.model === model &&
    JSON.stringify(playSettingsFromSession(saved.generation)) === settingsJson);
}

export default function Layout() {
  const song = useSongStore((s) => s.song);
  const setTitle = useSongStore((s) => s.setTitle);
  const clearSong = useSongStore((s) => s.clearSong);
  const selectedSlots = useSongStore((s) => s.selectedSlots);
  const clearSlots = useSongStore((s) => s.clearSlots);
  const addSession = useSessionStore((s) => s.add);
  const sessionCount = useSessionStore((s) => s.sessions.length);
  const modelKey = useGenerationStore((s) => s.modelKey);
  const playSettings = usePlayStore((s) => s.settings);
  const resetPlayback = useSongStore((s) => s.resetPlayback);
  const importSong = useSongStore((s) => s.importSong);
  const previousImportSong = useSongStore((s) => s.previousImportSong);
  const undoImport = useSongStore((s) => s.undoImport);
  const [saveMessage, setSaveMessage] = useState('');
  const importCleanupFailed = useImportCleanupStore((state) => state.failedTargets.length > 0);

  // Bundle the Play-panel settings with the saved score, so MyPage replays
  // the session the way the editor played it at Save time (bpm + genre are
  // already inside `song`).
  const handleSaveSession = useCallback(() => {
    try {
      addSession(song, generationContext(song, modelKey, playSettings));
      setSaveMessage('');
    } catch (error) {
      setSaveMessage(error instanceof Error ? error.message : 'The project could not be saved.');
    }
  }, [addSession, song, modelKey, playSettings]);

  const [midiActiveNotes, setMidiActiveNotes] = useState<number[]>([]);
  const [panelOpen, setPanelOpen] = useState(() => !window.matchMedia('(max-width: 760px), (orientation: portrait)').matches);
  const [panelWidth, setPanelWidth] = useState(320);
  const [panelMode, setPanelMode] = useState<'play' | 'input'>('input');
  const [inputMode, setInputMode] = useState<InputMode>('generate');
  const inputButtons = useRef<Partial<Record<InputMode, HTMLButtonElement | null>>>({});
  // Top-level view: editor (staff + right panel) vs full-page MyPage.
  // MyPage is its own page, not a panel mode — sessions render as
  // read-only staves there, which needs the whole canvas width.
  const [view, setView] = useState<'editor' | 'my'>('editor');
  const [songFormOpen, setSongFormOpen] = useState(false);
  const [importOpen, setImportOpen] = useState(false);
  const [sectionsOpen, setSectionsOpen] = useState(false);
  const resizingRef = useRef(false);

  const openScoreControls = useCallback((mode: 'generate' | 'reharmonize') => {
    setPanelOpen(true);
    setPanelMode('input');
    setInputMode(mode);
    requestAnimationFrame(() => inputButtons.current[mode]?.focus());
  }, []);

  // After New from Song Form lands, show Input → Generate so the A/B/C
  // suggestions for the new section are right there to swap.
  const handleSongFormGenerated = useCallback(() => {
    setPanelOpen(true);
    setPanelMode('input');
    setInputMode('generate');
  }, []);

  // After an import, show Input → Set Chord, where imported bars are finished.
  const handleImported = useCallback((imported: Song) => {
    const current = useSongStore.getState().song;
    const model = useGenerationStore.getState().modelKey;
    const settings = usePlayStore.getState().settings;
    // Keep the current work as a project before the import replaces it. A
    // storage failure throws here, which leaves the reviewed import open and
    // the current score intact. Undo remains available after a successful import.
    if (!isAlreadySaved(current, model, settings)) {
      useSessionStore.getState().add(current, generationContext(current, model, settings), current.title.trim() || 'Untitled');
    }
    importSong(imported);
    setPanelOpen(true);
    setPanelMode('input');
    setInputMode('manual');
  }, [importSong]);

  const startResize = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    resizingRef.current = true;
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
    const onMove = (ev: MouseEvent) => {
      if (!resizingRef.current) return;
      const next = window.innerWidth - ev.clientX - 20;
      setPanelWidth(Math.min(MAX_PANEL_WIDTH, Math.max(MIN_PANEL_WIDTH, next)));
    };
    const onUp = () => {
      resizingRef.current = false;
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
  }, []);

  const handleClear = useCallback(() => {
    if (window.confirm('Reset the entire song? This cannot be undone.')) {
      clearSong();
    }
  }, [clearSong]);

  const handleClearSelection = useCallback(() => {
    clearSlots(selectedSlots);
  }, [selectedSlots, clearSlots]);

  const handleMidiNotes = useCallback((notes: number[]) => {
    setMidiActiveNotes(notes);
  }, []);

  const toggleProjectsView = () => {
    // Whatever is playing belongs to the view being left.
    resetPlayback();
    setView(view === 'my' ? 'editor' : 'my');
  };

  const handleUndoImport = () => {
    if (window.confirm('Restore the song from before the import? Changes made after importing will be replaced.')) undoImport();
  };

  const playPaneShown = panelOpen && panelMode === 'play';
  const inputPaneShown = panelOpen && panelMode === 'input';
  const dialogOpen = songFormOpen || importOpen || sectionsOpen;

  return (
    <div className="artist-layout flex flex-col h-screen" style={{ background: 'var(--bg-primary)', color: 'var(--text-primary)' }}>
      {/* Top header — editorial brand lockup, compact so the right panel
          can fit a full-screen viewport without scrolling. */}
      <header className="artist-header shrink-0 py-4 px-4 relative text-center" style={{ borderBottom: '1px solid var(--border)' }}>
        <h1
          className="font-display text-4xl leading-none"
          style={{ color: 'var(--text-heading)', fontWeight: 700 }}
        >
          PearlLeeStudio
        </h1>
        <p
          className="font-display text-base mt-1"
          style={{ color: 'var(--text-muted)', fontStyle: 'italic', fontWeight: 400 }}
        >
          TheArtist
        </p>
        <div className="header-actions absolute top-3 right-4 flex items-center gap-2">
          <ThemeToggle />
          <button
            onClick={toggleProjectsView}
            className="px-3 py-1.5 text-sm btn-hover"
            style={{
              background: view === 'my' ? 'var(--text-heading)' : 'var(--btn-bg)',
              color: view === 'my' ? 'var(--bg-primary)' : 'var(--text-primary)',
            }}
            title={view === 'my' ? 'Back to editor' : 'Open saved projects'}
          >
            {view === 'my' ? 'Editor' : 'Projects'}
          </button>
        </div>
      </header>

      {importCleanupFailed && <p role="alert" className="shrink-0 px-4 py-2 text-xs" style={{ color: 'var(--text-primary)', background: 'var(--bg-surface)', borderBottom: '1px solid var(--border)' }}>
        Old source details could not be removed from browser storage. Your recovered chord work is available. Allow browser storage, then reload or save again.
      </p>}

      {view === 'my' ? (
        <MyPage onExitToEditor={() => setView('editor')} />
      ) : (
      <div className="editor-shell flex flex-1 min-h-0">
        {/* Left Panel - Score Editor */}
        <div className="score-pane flex-1 overflow-y-auto p-4 min-w-0">
          {/* Top bar: import and song-form entry left, title centered,
              export/save/clear right-aligned */}
          <div className="editor-toolbar flex flex-wrap items-center mb-4 gap-3">
            {/* Left group — flex-1 like the right group, so the title
                stays centered */}
            <div className="flex-1 flex items-center justify-start gap-2">
              <button
                onClick={() => setImportOpen(true)}
                className="px-3 py-1 text-xs font-bold whitespace-nowrap btn-hover"
                style={{ background: 'var(--btn-bg)', color: 'var(--text-heading)', border: '1px solid var(--border)' }}
                title="Import and review a PDF score"
                aria-haspopup="dialog"
              >
                Import Score
              </button>
              <button
                onClick={() => setSongFormOpen(true)}
                className="px-3 py-1 text-xs font-bold whitespace-nowrap btn-hover"
                style={{ background: 'var(--brand-yellow)', color: '#1a1a1a' }}
                title="Pick a song-form section and how many chords to generate"
                aria-haspopup="dialog"
              >
                New from Song Form
              </button>
            </div>

            {/* Song title — centered, editorial. Typing here turns the
                automatic section-based title off (songStore.setTitle). */}
            <input
              type="text"
              value={song.title}
              onChange={(e) => setTitle(e.target.value)}
              className="font-display text-2xl bg-transparent outline-none text-center shrink min-w-0"
              style={{ color: 'var(--text-heading)', fontStyle: 'italic', fontWeight: 500 }}
              placeholder="Untitled"
              aria-label="Song title"
            />

            {/* Export / Save / Clear — right. "Clear" (the selected slots)
                only appears when the user has slots selected; it lives next
                to Initialize (the whole song) so the two destructive actions
                are co-located. */}
            <div className="flex-1 flex items-center justify-end gap-2">
              {previousImportSong && (
                <button onClick={handleUndoImport} className="px-2 py-1 text-xs btn-hover" style={{ background: 'var(--btn-bg)' }}>
                  Undo import
                </button>
              )}
              <ExportMenu />
              <button
                onClick={handleSaveSession}
                className="px-3 py-1  text-xs font-bold btn-hover"
                style={{ background: 'var(--text-heading)', color: 'var(--bg-primary)' }}
                title={`Save current song + settings as a session (${sessionCount} saved)`}
              >
                Save
              </button>
              {selectedSlots.length > 0 && (
                <button
                  onClick={handleClearSelection}
                  className="px-3 py-1  text-xs btn-hover"
                  style={{ background: 'var(--btn-bg)', color: 'var(--text-primary)' }}
                  title="Reset chords in the selected slots (Delete; ⌘Z / Ctrl+Z undoes it)"
                >
                  Clear
                </button>
              )}
              <button
                onClick={handleClear}
                className="px-3 py-1  text-xs btn-hover"
                style={{ background: 'var(--btn-bg)', color: 'var(--text-muted)' }}
                title="Reset the entire song to its initial state"
              >
                Initialize
              </button>
            </div>
          </div>

          {saveMessage && <p role="alert" className="text-xs mb-3" style={{ color: 'var(--text-soft)' }}>{saveMessage}</p>}

          {/* Inline song settings — sits above the staff so the user can
              click straight on BPM / Time / Key / Genre to change them
              instead of hunting in a side panel. */}
          <div className="mb-3 pb-3" style={{ borderBottom: '1px solid var(--border)' }}>
            <SongSettings />
          </div>

          <Staff onOpenControls={openScoreControls} />
        </div>

        {/* Toggle column — short yellow chevron button at the top
            (sized to roughly match the Play/Input tab row), plus a
            transparent resize handle filling the rest when the panel
            is open. Drag the handle to resize, click the button to
            collapse. */}
        <div className="panel-toggle shrink-0 flex flex-col" style={{ width: 'var(--panel-toggle-width, 20px)' }}>
          <button
            onClick={() => setPanelOpen(!panelOpen)}
            className="flex items-center justify-center btn-hover"
            style={{
              background: 'var(--brand-yellow)',
              color: '#1a1a1a',
              height: '52px',
              fontSize: '14px',
              fontWeight: 700,
            }}
            title={panelOpen ? 'Hide panel' : 'Show panel'}
            aria-expanded={panelOpen}
            aria-controls="artist-side-panel"
          >
            {panelOpen ? '▶' : '◀'}
          </button>
          {panelOpen && (
            <div
              onMouseDown={startResize}
              className="panel-resize flex-1 cursor-col-resize"
              title="Drag to resize panel"
            />
          )}
        </div>

        {/* Right Panel — collapsible. A column: the Play/Input tabs, then
            the mode's content filling the rest. Play never scrolls (one
            pane at a time); Input scrolls inside its own area. */}
        <div
          id="artist-side-panel"
          className="side-panel shrink-0 flex flex-col min-h-0 p-3 gap-3"
          style={{
            width: `${panelWidth}px`,
            borderLeft: '1px solid var(--border)',
            background: 'var(--bg-surface)',
            display: panelOpen ? undefined : 'none',
          }}
        >
          {/* Top-level mode tabs — Play / Input. Saved sessions live on
              their own full-page route (header "Projects" button) so they
              get room to render each session as a real staff. */}
          <div
            className="shrink-0 flex items-end justify-between"
            style={{ borderBottom: '1px solid var(--border)' }}
          >
            {(['play', 'input'] as const).map((mode) => (
              <button
                key={mode}
                onClick={() => setPanelMode(mode)}
                className="font-display text-2xl leading-none px-1 pt-1 pb-2 transition-colors btn-hover"
                style={{
                  color: panelMode === mode ? 'var(--brand-yellow)' : 'var(--text-muted)',
                  borderTop: panelMode === mode ? '3px solid var(--brand-yellow)' : '3px solid transparent',
                  fontWeight: 700,
                  background: 'transparent',
                }}
              >
                {mode === 'play' ? 'Play' : 'Input'}
              </button>
            ))}
          </div>

          {/* Mode content */}
          {playPaneShown && (
            <div className="flex-1 min-h-0">
              <PlayPanel midiActiveNotes={midiActiveNotes} />
            </div>
          )}

          <div className="flex-1 min-h-0 overflow-y-auto space-y-3" style={{ display: panelMode === 'input' ? undefined : 'none' }}>
            {/* Composition is primary; MIDI and manual entry remain compact utilities. */}
            <div role="tablist" aria-label="Composition tools" className="flex" style={{ borderBottom: '1px solid var(--border)' }}>
              {INPUT_TABS.map((t) => {
                const active = inputMode === t.id;
                return (
                  <button
                    key={t.id}
                    ref={element => { inputButtons.current[t.id] = element; }}
                    id={`input-tool-${t.id}`}
                    role="tab"
                    aria-selected={active}
                    aria-controls="input-tool-content"
                    tabIndex={active || (inputMode !== 'generate' && inputMode !== 'reharmonize' && t.id === 'generate') ? 0 : -1}
                    onClick={() => setInputMode(t.id)}
                    onKeyDown={event => {
                      if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
                      event.preventDefault();
                      const next = event.key === 'Home' ? 'generate' : event.key === 'End' ? 'reharmonize'
                        : t.id === 'generate' ? 'reharmonize' : 'generate';
                      setInputMode(next);
                      inputButtons.current[next]?.focus();
                    }}
                    className="flex-1 min-w-0 px-2 py-2 text-xs font-semibold transition-colors btn-hover"
                    style={{
                      color: active ? 'var(--text-heading)' : 'var(--text-muted)',
                      borderTop: active ? '2px solid var(--brand-yellow)' : '2px solid transparent',
                      background: 'transparent',
                    }}
                  >
                    {t.label}
                  </button>
                );
              })}
            </div>

            <div role="group" aria-label="Additional input tools" className="flex justify-end gap-2">
              {ADDITIONAL_INPUTS.map(t => <button key={t.id} id={`input-tool-${t.id}`}
                aria-pressed={inputMode === t.id} aria-controls="input-tool-content"
                onClick={() => setInputMode(t.id)} className="px-2 py-1 text-[11px] rounded btn-hover"
                style={{ background: 'var(--btn-bg)', color: inputMode === t.id ? 'var(--text-heading)' : 'var(--text-muted)',
                  border: `1px solid ${inputMode === t.id ? 'var(--brand-yellow)' : 'var(--border)'}` }}>
                {t.label}
              </button>)}
            </div>

            <div id="input-tool-content" role={inputMode === 'generate' || inputMode === 'reharmonize' ? 'tabpanel' : 'region'}
              aria-labelledby={`input-tool-${inputMode}`}>
              {/* Hidden rather than unmounted, like this pane and the panel
                  itself, so MIDI monitoring stays connected while the panel is
                  closed or on Play. Played chords are captured only while the
                  MIDI tab is in front and no dialog is open. */}
              <div hidden={inputMode !== 'midi'}>
                <MidiStatus onMidiNotes={handleMidiNotes}
                  captureActive={inputPaneShown && inputMode === 'midi' && !dialogOpen} />
              </div>
              {inputPaneShown && inputMode === 'generate' && <ChordActions view="generate" />}
              {inputPaneShown && inputMode === 'reharmonize' && <ChordActions view="reharmonize" />}
              {inputPaneShown && inputMode === 'manual' && <ChordActions view="manual" />}
            </div>
          </div>
        </div>

      </div>
      )}

      {/* Mounted only while open; the dialog opens itself as a modal. */}
      {songFormOpen && (
        <SongFormDialog
          onClose={() => setSongFormOpen(false)}
          onGenerated={handleSongFormGenerated}
          onNewSong={() => {
            const current = useSongStore.getState().song;
            startNewSongFromForm(generationContext(current,
              useGenerationStore.getState().modelKey, usePlayStore.getState().settings));
          }}
          onEditSections={() => {
            setSongFormOpen(false);
            setSectionsOpen(true);
          }}
        />
      )}
      {importOpen && <ImportScoreDialog onClose={() => setImportOpen(false)} onImported={handleImported} />}
      {sectionsOpen && <SectionsDialog onClose={() => setSectionsOpen(false)} />}
    </div>
  );
}
