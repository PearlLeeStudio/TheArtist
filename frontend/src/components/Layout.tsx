import { useState, useCallback, useRef } from 'react';

type InputMode = 'generate' | 'midi' | 'manual';
// Order matters — Generate is the default and the most-used path, so it
// occupies the leftmost tab. MIDI and Set Chord sit in declining-frequency
// order to its right.
const INPUT_TABS: { id: InputMode; label: string }[] = [
  { id: 'generate', label: 'Generate' },
  { id: 'midi', label: 'MIDI' },
  { id: 'manual', label: 'Set Chord' },
];
import Staff from './ScoreEditor/Staff';
import VoicingViewer from './RightPanel/VoicingViewer';
import PlaybackControls from './RightPanel/PlaybackControls';
import SongSettings from './RightPanel/SongSettings';
import DrumSelector from './RightPanel/DrumSelector';
import ChordActions from './RightPanel/ChordActions';
import MidiStatus from './RightPanel/MidiStatus';
import MyPage from './MyPage/MyPage';
import ThemeToggle from './RightPanel/ThemeToggle';
import { useSongStore } from '../store/songStore';
import { useSessionStore } from '../store/sessionStore';
import { useGenerationStore } from '../store/generationStore';

export default function Layout() {
  const song = useSongStore((s) => s.song);
  const setTitle = useSongStore((s) => s.setTitle);
  const clearSong = useSongStore((s) => s.clearSong);
  const selectedSlots = useSongStore((s) => s.selectedSlots);
  const setChord = useSongStore((s) => s.setChord);
  const addSession = useSessionStore((s) => s.add);
  const sessionCount = useSessionStore((s) => s.sessions.length);
  const modelKey = useGenerationStore((s) => s.modelKey);
  const drumPatternId = useSongStore((s) => s.drumPatternId);
  const harmonyOverride = useSongStore((s) => s.harmonyOverride);

  // Pearl 2026-05-09: bundle the right-panel playback config (genre/bpm/drum)
  // with the saved score, so MyPage's "Generate music" can reproduce the
  // intended playback faithfully. Note: bpm + genre are already inside `song`
  // (song.bpm, song.genre); drumPatternId + harmonyOverride live in songStore
  // separately and snapshot here so a saved session replays with the exact
  // harmony instrument the user heard at Save time. Melody dropped 2026-05-10.
  const handleSaveSession = useCallback(() => {
    addSession(song, {
      model: modelKey,
      genre: song.genre,
      drumPatternId,
      harmonyOverride,
    });
  }, [addSession, song, modelKey, drumPatternId, harmonyOverride]);

  const [midiActiveNotes, setMidiActiveNotes] = useState<number[]>([]);
  const [panelOpen, setPanelOpen] = useState(true);
  const [panelWidth, setPanelWidth] = useState(320);
  const [panelMode, setPanelMode] = useState<'play' | 'input'>('input');
  const [inputMode, setInputMode] = useState<InputMode>('generate');
  // Top-level view: editor (staff + right panel) vs full-page MyPage.
  // MyPage is its own page, not a panel mode — sessions render as
  // read-only staves there, which needs the whole canvas width.
  const [view, setView] = useState<'editor' | 'my'>('editor');
  const resizingRef = useRef(false);

  const startResize = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    resizingRef.current = true;
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
    const onMove = (ev: MouseEvent) => {
      if (!resizingRef.current) return;
      const next = window.innerWidth - ev.clientX - 20;
      setPanelWidth(Math.min(640, Math.max(240, next)));
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
    for (const slot of selectedSlots) setChord(slot, null);
  }, [selectedSlots, setChord]);

  const handleMidiNotes = useCallback((notes: number[]) => {
    setMidiActiveNotes(notes);
  }, []);

  return (
    <div className="flex flex-col h-screen" style={{ background: 'var(--bg-primary)', color: 'var(--text-primary)' }}>
      {/* Top header — editorial brand lockup, compact so the right panel
          can fit a full-screen viewport without scrolling. */}
      <header className="shrink-0 py-4 px-4 relative text-center" style={{ borderBottom: '1px solid var(--border)' }}>
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
        <div className="absolute top-3 right-4 flex items-center gap-2">
          <ThemeToggle />
          <button
            onClick={() => setView(view === 'my' ? 'editor' : 'my')}
            className="px-3 py-1.5 text-sm btn-hover"
            style={{
              background: view === 'my' ? 'var(--text-heading)' : 'var(--btn-bg)',
              color: view === 'my' ? 'var(--bg-primary)' : 'var(--text-primary)',
            }}
            title={view === 'my' ? 'Back to editor' : 'Open MyPage'}
          >
            MyPage
          </button>
        </div>
      </header>

      {view === 'my' ? (
        <MyPage onExitToEditor={() => setView('editor')} />
      ) : (
      <div className="flex flex-1 min-h-0">
        {/* Left Panel - Score Editor */}
        <div className="flex-1 overflow-y-auto p-4 min-w-0">
          {/* Top bar: title centered, Save/Clear right-aligned */}
          <div className="flex items-center mb-4 gap-3">
            {/* Spacer for centering */}
            <div className="flex-1" />

            {/* Song title — centered, editorial */}
            <input
              type="text"
              value={song.title}
              onChange={(e) => setTitle(e.target.value)}
              className="font-display text-2xl bg-transparent outline-none text-center shrink min-w-0"
              style={{ color: 'var(--text-heading)', fontStyle: 'italic', fontWeight: 500 }}
              placeholder="Untitled"
            />

            {/* Save / Clear — right. "Clear selection" only appears when
                the user has slots selected; it lives next to the global
                Clear so the two destructive actions are co-located. */}
            <div className="flex-1 flex items-center justify-end gap-2">
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
                  title="Reset chords in the selected slots"
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

          {/* Inline song settings — sits above the staff so the user can
              click straight on BPM / Time / Key / Genre to change them
              instead of hunting in a side panel. */}
          <div className="mb-3 pb-3" style={{ borderBottom: '1px solid var(--border)' }}>
            <SongSettings />
          </div>

          <Staff />
        </div>

        {/* Toggle column — short yellow chevron button at the top
            (sized to roughly match the Play/Input tab row), plus a
            transparent resize handle filling the rest when the panel
            is open. Drag the handle to resize, click the button to
            collapse. */}
        <div className="shrink-0 flex flex-col" style={{ width: '20px' }}>
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
          >
            {panelOpen ? '▶' : '◀'}
          </button>
          {panelOpen && (
            <div
              onMouseDown={startResize}
              className="flex-1 cursor-col-resize"
              title="Drag to resize panel"
            />
          )}
        </div>

        {/* Right Panel — collapsible. Tight spacing so the whole stack
            fits in a 1080p viewport without scrolling. */}
        {panelOpen && (
          <div
            className="shrink-0 overflow-y-auto p-3 space-y-3"
            style={{
              width: `${panelWidth}px`,
              borderLeft: '1px solid var(--border)',
              background: 'var(--bg-surface)',
            }}
          >
            {/* Top-level mode tabs — Play / Input. Saved sessions live on
                their own full-page route (header "My" button) so they get
                room to render each session as a real staff. */}
            <div
              className="flex items-end justify-between"
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
            {panelMode === 'play' && (
              <div className="space-y-3">
                <PlaybackControls />
                <DrumSelector />
                {/* Voicing lives at the bottom of Play mode — the user
                    naturally wants to see the chord shape while they're
                    listening back, but it's noise during chord input. */}
                <div className="pt-3" style={{ borderTop: '1px solid var(--border)' }}>
                  <VoicingViewer midiActiveNotes={midiActiveNotes} />
                </div>
              </div>
            )}

            {panelMode === 'input' && (
              <div className="space-y-3">
                {/* Sub-tab bar — three input methods. */}
                <div className="flex" style={{ borderBottom: '1px solid var(--border)' }}>
                  {INPUT_TABS.map((t) => {
                    const active = inputMode === t.id;
                    return (
                      <button
                        key={t.id}
                        onClick={() => setInputMode(t.id)}
                        className="flex-1 px-3 py-2 text-xs font-semibold uppercase tracking-wide transition-colors btn-hover"
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

                {inputMode === 'midi' && <MidiStatus onMidiNotes={handleMidiNotes} />}
                {inputMode === 'generate' && <ChordActions view="generate" />}
                {inputMode === 'manual' && <ChordActions view="manual" />}
              </div>
            )}
          </div>
        )}

      </div>
      )}
    </div>
  );
}
