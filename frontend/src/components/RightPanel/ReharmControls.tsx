import { useState } from 'react';
import OperationProgress from '../OperationProgress';
import { useSongStore } from '../../store/songStore';
import { useReharmonizeStore } from '../../store/reharmonizeStore';
import { useGenerationStore } from '../../store/generationStore';
import { REHARM_CHANGE_LIMITS, REHARM_MODES, reharmOutcome, reharmRunDetails, reharmSelectionError, reharmTargets } from '../../engine/reharmonize';
import type { ReharmChangeLimit, ReharmMode } from '../../engine/reharmonize';
import { hasPendingReharmSelection } from '../../engine/reharmonizeSession';

/** The outcome of a search: which model scored the alternatives. */
function OutcomeBox({ title, detail }: { title: string; detail: string }) {
  return <p role="status" className="px-3 py-2 text-xs" style={{ border: '1px solid var(--border)', color: 'var(--text-primary)' }}>
    <strong className="block text-sm">{title}</strong>
    {detail}
  </p>;
}

export default function ReharmControls({ modelKey }: { modelKey: string }) {
  const song = useSongStore(s => s.song);
  const slots = useSongStore(s => s.selectedSlots);
  const loading = useReharmonizeStore(s => s.loading);
  const error = useReharmonizeStore(s => s.error);
  const comparison = useReharmonizeStore(s => s.comparison);
  const result = useReharmonizeStore(s => s.result);
  const generating = useGenerationStore(s => s.loading);
  const run = useReharmonizeStore(s => s.run);
  const apply = useReharmonizeStore(s => s.apply);
  const pending = useReharmonizeStore(hasPendingReharmSelection);
  const selected = useReharmonizeStore(s => s.selected);
  const hasApplied = useReharmonizeStore(s => s.committed !== s.comparison);
  const notice = useReharmonizeStore(s => s.notice);
  const elapsedMs = useReharmonizeStore(s => s.elapsedMs);
  const [maxChanges, setMaxChanges] = useState<ReharmChangeLimit>(1);
  const [mode, setMode] = useState<ReharmMode>('auto');
  const problem = reharmSelectionError(song, slots);
  const targetCount = reharmTargets(song, slots).length;
  const disabled = loading || generating || !!problem;
  const applyDisabled = loading || generating || !result || !pending || !!error;
  const applied = !!result && hasApplied && !pending && !error;
  const selectionLabel = selected.length ? selected.map(index => `Alter ${index + 1}`).join(' + ') : 'Origin';
  const selectStyle = { background: 'var(--btn-bg)', color: 'var(--text-primary)', border: '1px solid var(--border)' };
  return <div className="space-y-3">
    <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
      Select alternatives together and use Play to compare. Overlapping choices replace each other. Apply keeps your selection and closes the cards.
    </p>
    <label className="flex items-center justify-between gap-2 text-xs">
      Change up to
      <select aria-label="Maximum changed chords per candidate" value={maxChanges}
        onChange={e => {
          const value = REHARM_CHANGE_LIMITS.find(limit => limit === Number(e.target.value));
          if (value) setMaxChanges(value);
        }} className="px-2 py-1" style={selectStyle}>
        {REHARM_CHANGE_LIMITS.map(limit => <option key={limit} value={limit}>
          {limit} {limit === 1 ? 'chord' : 'chords'}
        </option>)}
      </select>
    </label>
    <label className="flex items-center justify-between gap-2 text-xs">
      Mode
      <select aria-label="Mode preference for substitutes" value={mode}
        onChange={e => setMode(e.target.value as ReharmMode)} className="px-2 py-1 capitalize" style={selectStyle}>
        {REHARM_MODES.map(m => <option key={m} value={m}>{m === 'auto' ? 'Auto · In context' : m}</option>)}
      </select>
    </label>
    <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
      {mode === 'auto' ? 'Considers familiar substitutions and how they connect to surrounding chords.'
        : 'Only substitutes compatible with this parallel mode.'}
    </p>
    {problem
      ? <p role="status" className="text-xs font-semibold" style={{ color: 'var(--text-primary)' }}>{problem}</p>
      : <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
        {targetCount} {targetCount === 1 ? 'chord' : 'chords'} selected (a passage of up to 8 bars). Empty slots stay unchanged.
      </p>}
    <button disabled={disabled} onClick={() => void run(modelKey, maxChanges, mode)}
      className="w-full px-3 py-2 text-sm font-bold"
      style={{ background: 'var(--brand-yellow)', color: '#1a1a1a', opacity: disabled ? .55 : 1 }}>
      {loading ? 'Exploring substitutes…' : 'Find 3 alternatives'}
    </button>
    <button disabled={applyDisabled} onClick={apply} className="w-full px-3 py-2 text-sm font-bold"
      style={{ background: applyDisabled ? 'var(--btn-bg)' : 'var(--brand-yellow)',
        color: applyDisabled ? 'var(--text-muted)' : '#1a1a1a', border: '1px solid var(--border)' }}>
      {applied ? 'Applied ✓' : 'Apply'}
    </button>
    {result && !loading && !error && <p role="status" className="text-xs" style={{ color: 'var(--text-muted)' }}>
      {applied ? `${selectionLabel} applied. Choose a different combination to apply again.`
        : pending ? `${selectionLabel} selected. Apply to keep ${selected.length ? 'these changes' : 'the original chords'}.`
          : 'Choose an alternative on the score to enable Apply.'}
    </p>}
    {result && !loading && !error && <OutcomeBox {...reharmOutcome(result)} />}
    {result && !loading && !error && <details className="text-xs" style={{ color: 'var(--text-muted)' }}>
      <summary className="cursor-pointer">Run details</summary>
      <ul className="mt-1 space-y-0.5 break-words">
        {reharmRunDetails(result, elapsedMs).map(line => <li key={line}>{line}</li>)}
      </ul>
    </details>}
    {!result && !loading && notice && <p role="status" className="text-xs" style={{ color: 'var(--text-muted)' }}>{notice}</p>}
    {loading && <OperationProgress label="Comparing substitutions against the surrounding chords…" />}
    {error && <p role="alert" className="px-3 py-2 text-xs" style={{ border: '1px solid var(--border)', color: 'var(--text-primary)' }}>
      <strong className="block text-sm">Search failed</strong>{error}
    </p>}
    {comparison && <button className="text-xs underline" onClick={() => {
      document.getElementById('reharmonization-comparison')?.scrollIntoView({ block: 'center', behavior: 'smooth' });
    }}>{result ? 'View 3 alternatives on the score' : 'View search on the score'}</button>}
  </div>;
}
