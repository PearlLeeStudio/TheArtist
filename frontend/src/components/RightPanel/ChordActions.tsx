import { useState, useCallback, useEffect } from 'react';
import { useShallow } from 'zustand/react/shallow';
import { useSongStore } from '../../store/songStore';
import { useGenerationStore } from '../../store/generationStore';
import { useSessionStore, selectStarredSessions } from '../../store/sessionStore';
import { parseSuggestionChords } from '../../engine/chordGenerator';
import type { Suggestion } from '../../engine/chordGenerator';
import GenerateProgress from './GenerateProgress';
import { getChordDisplayParts } from '../../engine/chordDisplay';
import type { Chord } from '../../models/types';

const ROOTS = ['C', 'C#', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B'];

/**
 * Infer the song's genre from the picked model. Model pick always syncs
 * song.genre — Pearl 2026-05-10 found "first-time-only" sync confusing
 * (e.g. switching from blues LoRA to F4 left genre=blues, not jazz).
 * If the user wants a different genre than what the model implies, they
 * can override via SongSettings after picking the model.
 *
 * Mapping:
 *   ft_f1_lora_X        → X (LoRA carries genre tag in its key)
 *   ft_f1 / phase0      → pop  (F1/Phase0 are pop-leaning)
 *   ft_f2..f5 → jazz (jazz fine-tunes with varying pop mix)
 */
function inferGenreForModelPick(modelKey: string): string | null {
  const lora = modelKey.match(/^ft_f1_lora_(.+)$/);
  if (lora) return lora[1];
  if (modelKey === 'ft_f1' || modelKey === 'phase0') return 'pop';
  if (modelKey.startsWith('ft_f')) return 'jazz';
  return null;
}

const BASE_QUALITIES = [
  { label: 'Maj', value: 'maj' },
  { label: 'Min', value: 'min' },
  { label: 'Dim', value: 'dim' },
  { label: 'Aug', value: 'aug' },
  { label: '5', value: '5' },
];

const EXTENSIONS = [
  { label: '-', value: '' },
  { label: '6', value: '6' },
  { label: '7', value: '7' },
  { label: 'M7', value: 'maj7' },
  { label: '9', value: '9' },
  { label: 'M9', value: 'maj9' },
  { label: '11', value: '11' },
  { label: '13', value: '13' },
];

const ADD_OPTIONS = [
  { label: '-', value: '' },
  { label: 'add2', value: 'add2' },
  { label: 'add4', value: 'add4' },
  { label: 'add9', value: 'add9' },
  { label: 'add11', value: 'add11' },
];

const SUS_OPTIONS = [
  { label: '-', value: '' },
  { label: 'sus2', value: 'sus2' },
  { label: 'sus4', value: 'sus4' },
];

const ALTERATIONS = [
  { label: 'b5', value: 'b5' },
  { label: '#5', value: '#5' },
  { label: 'b9', value: 'b9' },
  { label: '#9', value: '#9' },
  { label: '#11', value: '#11' },
  { label: 'b13', value: 'b13' },
];

function resolveQuality(base: string, ext: string, add: string, sus: string, _alts: string[]): string {
  if (sus) return ext ? `${ext}${sus}` : sus;
  if (add) return base === 'min' ? `m(${add})` : add;
  if (base === '5') return ext === '7' ? '7(no3)' : '5';

  if (base === 'min') {
    if (ext === 'maj7') return 'mMaj7';
    if (ext === '7') return 'm7';
    if (ext === '9') return 'm9';
    if (ext === '11') return 'm11';
    if (ext === '13') return 'm13';
    if (ext === '6') return 'm6';
    if (ext === '') return 'min';
    return `m${ext}`;
  }
  if (base === 'dim') {
    if (ext === '7') return 'dim7';
    if (ext === '') return 'dim';
    return 'm7b5';
  }
  if (base === 'aug') return ext === '7' ? 'aug7' : 'aug';

  if (ext === '7') return '7';
  if (ext) return ext;
  return 'maj';
}

const btnStyle = (active: boolean) => ({
  background: active ? 'var(--text-heading)' : 'var(--btn-bg)',
  color: active ? 'var(--bg-primary)' : 'var(--text-primary)',
});

interface ChordActionsProps {
  /**
   * Which slice of the chord-actions UI to render.
   * - 'all'      → Generate + manual builder (legacy default).
   * - 'generate' → Model selector + Generate button + Suggestions only.
   * - 'manual'   → Manual chord builder forced open (no toggle button).
   * Used by Layout to back the right-panel Input tabs.
   */
  view?: 'all' | 'generate' | 'manual';
}

export default function ChordActions({ view = 'all' }: ChordActionsProps) {
  const selectedSlots = useSongStore((s) => s.selectedSlots);
  const song = useSongStore((s) => s.song);
  const setGenre = useSongStore((s) => s.setGenre);
  const setChord = useSongStore((s) => s.setChord);
  const resetPlayback = useSongStore((s) => s.resetPlayback);

  // In tab-mode 'manual', the builder is the entire view, so always open.
  const [expanded, setExpanded] = useState(view === 'manual');
  const [root, setRoot] = useState('C');
  const [base, setBase] = useState('maj');
  const [ext, setExt] = useState('');
  const [add, setAdd] = useState('');
  const [sus, setSus] = useState('');
  const [alts, setAlts] = useState<string[]>([]);
  const [bass, setBass] = useState('');

  // AI suggestion state — hoisted to generationStore (Pearl 2026-05-10) so
  // the in-flight generate keeps running and its result lands even if the
  // user navigates away (Play tab / MyPage) before completion. Previously
  // this state lived in component-local useState and died on unmount,
  // losing the result.
  const suggestions = useGenerationStore((s) => s.suggestions);
  const loading = useGenerationStore((s) => s.loading);
  const phase = useGenerationStore((s) => s.phase);
  const showSuggestions = useGenerationStore((s) => s.showSuggestions);
  const activeSuggestionIdx = useGenerationStore((s) => s.activeSuggestionIdx);
  const setActiveSuggestionIdx = useGenerationStore((s) => s.setActiveSuggestionIdx);
  const clearSuggestions = useGenerationStore((s) => s.clearSuggestions);
  const runGenerate = useGenerationStore((s) => s.runGenerate);
  const runFavoriteGenerate = useGenerationStore((s) => s.runFavoriteGenerate);

  // Model selection — same store, persists to localStorage.
  const modelKey = useGenerationStore((s) => s.modelKey);
  const setModelKey = useGenerationStore((s) => s.setModelKey);
  const [modelList, setModelList] = useState<Array<{ key: string; label: string; available: boolean }>>([]);

  // Starred sessions feed Favorite Generate — one suggestion per preset.
  // useShallow keeps the snapshot reference stable across renders;
  // without it, selectStarredSessions returns a fresh array each call
  // and useSyncExternalStore enters an infinite re-render loop.
  const starredSessions = useSessionStore(useShallow(selectStarredSessions));

  // Fetch available models once on mount. Backend returns a flat `models`
  // list combining paper F-series + R4 LoRA adapters; both visible in one
  // dropdown (Pearl 2026-05-09 — single-selector preference).
  useEffect(() => {
    fetch('/api/models/list')
      .then(r => r.ok ? r.json() : null)
      .then(data => {
        if (data?.models) {
          // Only surface usable checkpoints — hide any the backend reports
          // unavailable on this install so the dropdown
          // lists only models the user can actually pick.
          setModelList(data.models.filter((m: { available: boolean }) => m.available));
          // Migrate stale ft_f3 default → backend default (now ft_f1).
          if (data.default && useGenerationStore.getState().modelKey === 'ft_f3') {
            setModelKey(data.default);
          }
        }
      })
      .catch(() => { /* fallback: model list unavailable */ });
  }, [setModelKey]);

  // Reset transient AI state whenever the song is replaced (e.g. Clear) —
  // the suggestions tied to the old song would be stale and confusing if
  // they stayed on screen against an empty staff.
  useEffect(() => {
    clearSuggestions();
    setExpanded(false);
  }, [song.id, clearSuggestions]);

  // Mount-time genre sync. Every reload clears localStorage (main.tsx)
  // so song.genre starts null; modelKey hydrates from its store default.
  // Without this nudge the harmony rhythm would route to the fallback
  // pattern until the user manually picked a genre in SongSettings.
  useEffect(() => {
    if (!song.genre) {
      const inferred = inferGenreForModelPick(modelKey);
      if (inferred) setGenre(inferred);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const firstSlot = selectedSlots[0] ?? null;
  const quality = resolveQuality(base, ext, add, sus, alts);
  const previewParts = getChordDisplayParts({ root, quality, bass: bass || undefined, romanNumeral: '', voicing: [], extensions: alts });

  const handleSet = useCallback(() => {
    if (!firstSlot) return;
    const chord: Chord = {
      root, quality,
      bass: bass || undefined,
      romanNumeral: '', voicing: [], extensions: alts,
    };
    setChord(firstSlot, chord);
    setExpanded(false);
  }, [firstSlot, root, quality, bass, alts, setChord]);

  const toggleAlt = (alt: string) => {
    setAlts(prev => prev.includes(alt) ? prev.filter(a => a !== alt) : [...prev, alt]);
  };

  const handleGenerate = () => {
    if (selectedSlots.length === 0) return;
    // Stop any ongoing playback — the user is reshaping the song; what's
    // playing won't reflect what they're about to see.
    resetPlayback();
    // Fire-and-forget: store owns the async lifecycle so navigation away
    // from this panel during generate doesn't kill the request.
    void runGenerate(song, selectedSlots, modelKey);
  };

  // Favorite Generate — runs the generate pipeline once per starred
  // session, taking the top-ranked suggestion from each. Result count
  // equals starred-session count, so the user gets one option per
  // preset they marked. Calls run in parallel so total time is bounded
  // by the slowest single generate, not the sum.
  const handleFavoriteGenerate = () => {
    if (selectedSlots.length === 0 || starredSessions.length === 0) return;
    resetPlayback();
    void runFavoriteGenerate(song, selectedSlots, starredSessions);
  };

  const handleSelectSuggestion = (suggestion: Suggestion, idx: number) => {
    const chordMap = parseSuggestionChords(suggestion);
    // Backend works at measure granularity (selectedMeasures = [5] returns
    // both slot 0 and slot 1 for that measure), but the user's selection
    // is slot-level. Apply only the slots the user actually picked so
    // unselected slots (and any chord they already had) stay intact.
    const selectedKeys = new Set(
      selectedSlots.map(s => `${s.measureIndex}-${s.slotIndex}`)
    );
    for (const [key, chord] of chordMap) {
      if (!selectedKeys.has(key)) continue;
      const [mIdx, sIdx] = key.split('-').map(Number);
      if (Number.isNaN(mIdx) || (sIdx !== 0 && sIdx !== 1)) continue;
      setChord({ measureIndex: mIdx, slotIndex: sIdx as 0 | 1 }, chord);
    }
    // Keep the panel open so the user can A/B/C between candidates.
    // Re-clicking the active one re-applies (cheap, idempotent).
    setActiveSuggestionIdx(idx);
  };

  const slotLabel = selectedSlots.length === 1
    ? `M${firstSlot!.measureIndex + 1} Beat ${firstSlot!.slotIndex === 0 ? 1 : 3}`
    : `${selectedSlots.length} slots`;

  if (!firstSlot) {
    return (
      <div className="space-y-2">
        <h3 className="text-xs uppercase tracking-wide font-semibold" style={{ color: 'var(--text-heading)' }}>Chords</h3>
        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Select a slot on the staff</p>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <h3 className="text-xs uppercase tracking-wide font-semibold" style={{ color: 'var(--text-heading)' }}>
        Chords <span className="font-normal normal-case">— {slotLabel}</span>
      </h3>

      {/* ---------- Generate (model) UI ---------- */}
      {(view === 'all' || view === 'generate') && (
      <>
      {/* Model selector (single dropdown — paper F-series + R4 LoRA adapters
          all visible together, Pearl 2026-05-09 preference). Picking a LoRA
          also auto-syncs song.genre (Pearl 2026-05-10) so the harmony rhythm
          pattern in playback.ts routes correctly without a second click. */}
      {modelList.length > 0 && (
        <div className="space-y-1">
          <label className="text-[10px] uppercase tracking-wide font-semibold" style={{ color: 'var(--text-muted)' }}>
            Model
          </label>
          <select
            value={modelKey}
            onChange={(e) => {
              const newKey = e.target.value;
              setModelKey(newKey);
              const inferred = inferGenreForModelPick(newKey);
              if (inferred) setGenre(inferred);
            }}
            className="w-full px-2 py-1.5  text-xs"
            style={{ background: 'var(--btn-bg)', color: 'var(--text-primary)', border: '1px solid var(--border)' }}
          >
            {modelList.map(m => (
              <option key={m.key} value={m.key}>{m.label}</option>
            ))}
          </select>
        </div>
      )}

      <button
        onClick={handleGenerate}
        disabled={loading}
        className="w-full px-3 py-2  text-sm font-bold"
        style={{ background: 'var(--text-heading)', color: 'var(--bg-primary)', opacity: loading ? 0.6 : 1 }}
      >
        {loading ? 'Generating...' : `Generate ${selectedSlots.length > 1 ? `(${selectedSlots.length} slots)` : 'Chord'}`}
      </button>

      {/* Favorite Generate — runs once per starred session, returning one
          suggestion per preset. Hidden until the user has at least one
          starred session on MyPage. */}
      {starredSessions.length > 0 && (
        <button
          onClick={handleFavoriteGenerate}
          disabled={loading}
          className="w-full px-3 py-2 text-sm font-bold"
          style={{
            background: 'var(--brand-yellow)',
            color: '#1a1a1a',
            opacity: loading ? 0.6 : 1,
          }}
          title={`Generate one option per starred preset (${starredSessions.length})`}
        >
          ★ Favorite Generate ({starredSessions.length})
        </button>
      )}

      {/* Live progress — driven by SSE phase events from /api/generate/stream.
          Each pipeline stage gets a chip; the active one is highlighted.
          Only shown for regular Generate; Favorite Generate runs N parallel
          calls and the chip flow would race across them. */}
      {loading && phase && <GenerateProgress phase={phase} />}

      {/* AI Suggestions */}
      {showSuggestions && suggestions.length > 0 && (
        <div className="space-y-1.5" style={{ borderTop: '1px solid var(--border)', paddingTop: '8px' }}>
          <p className="text-[10px] uppercase tracking-wide font-semibold" style={{ color: 'var(--text-muted)' }}>
            {activeSuggestionIdx === null ? 'Pick a suggestion' : 'Tap to swap'}
          </p>
          {suggestions.map((s, idx) => {
            const chordPreview = Object.entries(s.chords)
              .sort(([a], [b]) => Number(a) - Number(b))
              .map(([, pair]) => pair.join(' | '))
              .join('  ');
            const isActive = idx === activeSuggestionIdx;
            return (
              <div
                key={idx}
                className=" text-sm"
                style={{
                  background: isActive ? 'var(--text-heading)' : 'var(--btn-bg)',
                  color: isActive ? 'var(--bg-primary)' : 'var(--text-primary)',
                  border: isActive ? '1px solid var(--text-heading)' : '1px solid transparent',
                }}
              >
                <button
                  onClick={() => handleSelectSuggestion(s, idx)}
                  className="w-full text-left px-3 py-2"
                >
                  <div
                    className="font-semibold text-xs truncate"
                    style={{ color: isActive ? 'var(--bg-primary)' : 'var(--text-heading)' }}
                    title={s.label}
                  >
                    {isActive ? '▸ ' : ''}{s.label}
                  </div>
                  <div
                    className="text-xs mt-0.5 truncate"
                    style={{
                      color: isActive ? 'var(--bg-primary)' : 'var(--text-muted)',
                      opacity: isActive ? 0.85 : 1,
                      fontFamily: 'monospace',
                    }}
                    title={chordPreview}
                  >
                    {chordPreview}
                  </div>
                </button>
                {s.explanations && s.explanations.length > 0 && (
                  <details className="px-3 pb-2">
                    <summary
                      className="cursor-pointer text-[10px] uppercase tracking-wide"
                      style={{ color: 'var(--text-muted)' }}
                    >
                      Theory ({s.explanations.length} transitions)
                    </summary>
                    <ul className="mt-1 space-y-1.5 text-[11px]">
                      {s.explanations.map((e, eIdx) => (
                        <li
                          key={eIdx}
                          style={{ color: 'var(--text-primary)', borderLeft: '2px solid var(--border)', paddingLeft: '6px' }}
                        >
                          <div>
                            <span style={{ fontFamily: 'monospace', color: 'var(--text-heading)' }}>
                              {e.chord_a} → {e.chord_b}
                            </span>
                            <span style={{ color: 'var(--text-muted)', marginLeft: '6px' }}>[{e.concept}]</span>
                            {e.page_start > 0 && (
                              <span style={{ color: 'var(--text-muted)', marginLeft: '6px' }}>
                                p{e.page_start}{e.page_end > e.page_start ? `-${e.page_end}` : ''}
                              </span>
                            )}
                          </div>
                          {e.explanation && (
                            <div className="mt-0.5" style={{ color: 'var(--text-muted)', lineHeight: 1.4 }}>
                              {e.explanation}
                            </div>
                          )}
                        </li>
                      ))}
                    </ul>
                  </details>
                )}
              </div>
            );
          })}
          <button
            onClick={() => clearSuggestions()}
            className="w-full px-3 py-1  text-xs"
            style={{ background: 'var(--btn-bg)', color: 'var(--text-muted)' }}
          >
            Dismiss
          </button>
        </div>
      )}
      </>
      )}
      {/* ---------- end Generate UI ---------- */}

      {/* Set Chord toggle (legacy 'all' layout only). Clear-in-selection
          lives in the top toolbar so destructive actions are co-located. */}
      {view === 'all' && (
        <button
          onClick={() => setExpanded(!expanded)}
          className="w-full px-3 py-1.5  text-sm"
          style={{ background: expanded ? 'var(--text-heading)' : 'var(--btn-bg)', color: expanded ? 'var(--bg-primary)' : 'var(--text-primary)' }}
        >
          {expanded ? 'Cancel' : 'Set Chord'}
        </button>
      )}

      {/* ---------- Manual chord builder ----------
          Always shown in 'manual' view; collapsible in 'all'; hidden in 'generate'. */}
      {(view === 'manual' || (view === 'all' && expanded)) && (
        <div className="space-y-2 pt-1" style={{ borderTop: '1px solid var(--border)' }}>
          {/* Preview */}
          <div className="flex items-center justify-center py-1.5 " style={{ background: 'var(--btn-bg)' }}>
            <span className="text-2xl font-bold" style={{ color: 'var(--text-heading)' }}>{previewParts.root}</span>
            <span className="flex flex-col ml-0.5 leading-tight" style={{ color: 'var(--text-heading)' }}>
              <span className="text-xs font-semibold -mb-0.5">{previewParts.sup || '\u00A0'}</span>
              <span className="text-xs">{previewParts.sub || '\u00A0'}</span>
            </span>
            {previewParts.bass && (
              <span className="text-lg font-bold ml-0.5" style={{ color: 'var(--text-heading)' }}>/{previewParts.bass}</span>
            )}
          </div>

          {/* Root */}
          <div>
            <label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Root</label>
            <div className="flex flex-wrap gap-0.5">
              {ROOTS.map(r => (
                <button key={r} onClick={() => setRoot(r)} className="px-1.5 py-0.5  text-[11px] font-bold" style={btnStyle(r === root)}>{r}</button>
              ))}
            </div>
          </div>

          {/* Base + Sus in one row */}
          <div className="flex gap-2">
            <div className="flex-1">
              <label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Base</label>
              <div className="flex gap-0.5">
                {BASE_QUALITIES.map(q => (
                  <button key={q.value} onClick={() => { setBase(q.value); setSus(''); }} className="flex-1 px-1 py-0.5  text-[11px]" style={btnStyle(q.value === base && !sus)}>{q.label}</button>
                ))}
              </div>
            </div>
            <div className="w-24">
              <label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Sus</label>
              <div className="flex gap-0.5">
                {SUS_OPTIONS.map(s => (
                  <button key={s.value || 'n'} onClick={() => setSus(s.value)} className="flex-1 px-1 py-0.5  text-[10px]" style={btnStyle(s.value === sus)}>{s.label}</button>
                ))}
              </div>
            </div>
          </div>

          {/* Extension */}
          <div>
            <label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Ext</label>
            <div className="flex flex-wrap gap-0.5">
              {EXTENSIONS.map(e => (
                <button key={e.value || 'n'} onClick={() => setExt(e.value)} className="px-1.5 py-0.5  text-[11px]" style={btnStyle(e.value === ext)}>{e.label}</button>
              ))}
            </div>
          </div>

          {/* Add */}
          <div>
            <label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Add</label>
            <div className="flex gap-0.5">
              {ADD_OPTIONS.map(a => (
                <button key={a.value || 'n'} onClick={() => setAdd(a.value)} className="px-1.5 py-0.5  text-[11px]" style={btnStyle(a.value === add)}>{a.label}</button>
              ))}
            </div>
          </div>

          {/* Alterations */}
          <div>
            <label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Alt</label>
            <div className="flex gap-0.5">
              {ALTERATIONS.map(a => (
                <button key={a.value} onClick={() => toggleAlt(a.value)} className="px-1.5 py-0.5  text-[11px]"
                  style={alts.includes(a.value) ? { background: 'var(--text-accent)', color: 'var(--bg-primary)' } : { background: 'var(--btn-bg)', color: 'var(--text-primary)' }}
                >{a.label}</button>
              ))}
            </div>
          </div>

          {/* Bass (slash chord) */}
          <div>
            <label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Bass /</label>
            <div className="flex flex-wrap gap-0.5">
              <button onClick={() => setBass('')} className="px-1.5 py-0.5  text-[11px]" style={btnStyle(!bass)}>-</button>
              {ROOTS.map(r => (
                <button key={r} onClick={() => setBass(r)} className="px-1.5 py-0.5  text-[11px]" style={btnStyle(r === bass)}>{r}</button>
              ))}
            </div>
          </div>

          <button onClick={handleSet} className="w-full px-3 py-2  text-sm font-bold" style={{ background: 'var(--text-heading)', color: 'var(--bg-primary)' }}>
            Apply
          </button>
        </div>
      )}
    </div>
  );
}
