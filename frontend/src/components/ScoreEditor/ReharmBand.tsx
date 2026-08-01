import { useReharmonizeStore } from '../../store/reharmonizeStore';
import { reharmSymbol } from '../../engine/reharmonize';

const FAMILY = { common_tones: 'Shared tones', modal: 'Modal color', dominant: 'Dominant approach' };

/** This remains on the score while the user switches to the existing Play controls. */
export default function ReharmBand() {
  const { comparison, result, loading, error, selected, preview, close } = useReharmonizeStore();
  if (!comparison) return null;
  const measures = comparison.original.map(c => c.address.measureIndex + 1);
  const range = Math.min(...measures) === Math.max(...measures) ? `M${measures[0]}` : `M${Math.min(...measures)}–${Math.max(...measures)}`;
  return <section id="reharmonization-comparison" aria-label="Reharmonization comparison" className="p-3 space-y-2 rounded-md"
    style={{ color: 'var(--text-primary)', background: 'var(--bg-secondary)', border: '1px solid var(--border)' }}>
    <div className="flex flex-wrap items-center justify-between gap-2">
      <div className="text-xs font-semibold">Reharmonize · {range}</div>
      <button onClick={close} className="w-6 h-6 rounded text-base leading-none"
        style={{ border: '1px solid var(--border)' }}
        aria-label="Close reharmonization" title="Close · discard unapplied changes">
        <span aria-hidden="true">×</span>
      </button>
    </div>
    {loading && <p role="status" className="text-xs">Exploring substitutions…</p>}
    {error && <p role="alert" className="text-xs">{error}</p>}
    {result && <>
      <div className="grid grid-cols-4 items-stretch gap-2">
        <button aria-pressed={selected.length === 0} aria-label="Preview Origin on the score" onClick={() => preview(null)}
          className="min-w-0 flex flex-col items-start text-left px-2 py-2 rounded space-y-1"
          style={{ background: selected.length === 0 ? 'var(--brand-yellow)' : 'var(--btn-bg)',
            color: selected.length === 0 ? '#1a1a1a' : 'var(--text-primary)', border: '1px solid var(--border)' }}>
          <div className="text-xs font-semibold">Origin{selected.length === 0 ? ' ✓' : ''}</div>
          <div className="text-xs break-words">{comparison.original.map(c => c.chord ? reharmSymbol(c.chord) : '—').join(' · ')}</div>
        </button>
        {result.candidates.map((candidate, index) => <button key={candidate.id} aria-pressed={selected.includes(index)}
          aria-label={`Select Alter ${index + 1}`} onClick={() => preview(index)}
          className="min-w-0 flex flex-col items-start text-left px-2 py-2 rounded space-y-1"
          style={{ background: selected.includes(index) ? 'var(--brand-yellow)' : 'var(--btn-bg)',
            color: selected.includes(index) ? '#1a1a1a' : 'var(--text-primary)', border: '1px solid var(--border)' }}>
          <div className="flex flex-wrap items-center gap-x-2 text-xs font-semibold">
            <span>Alter {index + 1}{selected.includes(index) ? ' ✓' : ''}</span>
            <span className="text-[10px] font-normal">{FAMILY[candidate.family]}</span>
          </div>
          {candidate.changes.map(c => <div key={`${c.address.measureIndex}-${c.address.slotIndex}`} className="text-xs break-words">
            <span className="text-[10px]">M{c.address.measureIndex + 1}.{c.address.slotIndex + 1} </span>
            <span className="opacity-60">{c.before}</span> → <strong>{c.after}</strong>
          </div>)}
          <div className="text-[10px] opacity-75">
            {[...new Set(candidate.changes.flatMap(c => c.conventions ?? []))].join(' · ') ||
              candidate.changes.map(c => `${c.commonTones} shared tones`).join(' · ')}
          </div>
        </button>)}
      </div>
      {!!result.modelApproximations?.length && <details className="text-[11px]" style={{ color: 'var(--text-muted)' }}>
        <summary className="cursor-pointer">Some chord colors have limited model scoring</summary>
        <p>Full chord tones are used for substitution matching and comparison. The music model uses these simpler symbols for scoring only:</p>
        <ul>{result.modelApproximations.map(a => <li key={`${a.address.measureIndex}.${a.address.slotIndex}`}>
          M{a.address.measureIndex + 1}.{a.address.slotIndex + 1}: {a.original} → {a.scoredAs}
        </li>)}</ul>
      </details>}
    </>}
  </section>;
}
