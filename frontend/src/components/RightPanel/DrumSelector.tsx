import { useSongStore } from '../../store/songStore';
import { DRUM_PATTERNS } from '../../engine/drums';

export default function DrumSelector() {
  const drumPatternId = useSongStore((s) => s.drumPatternId);
  const setDrumPatternId = useSongStore((s) => s.setDrumPatternId);

  const btnStyle = (active: boolean) => active
    ? { background: 'var(--text-heading)', color: 'var(--bg-primary)' }
    : { background: 'var(--btn-bg)', color: 'var(--text-primary)' };

  return (
    <div className="space-y-1.5">
      <h3 className="text-xs uppercase tracking-wide font-semibold" style={{ color: 'var(--text-heading)' }}>Drums</h3>
      <div className="grid grid-cols-4 gap-1">
        <button
          onClick={() => setDrumPatternId(null)}
          className="px-1.5 py-1 text-[10px] truncate"
          style={btnStyle(drumPatternId === null)}
          title="No drums"
        >
          OFF
        </button>
        {DRUM_PATTERNS.map((p) => (
          <button
            key={p.id}
            onClick={() => setDrumPatternId(p.id)}
            className="px-1.5 py-1 text-[10px] truncate"
            style={btnStyle(drumPatternId === p.id)}
            title={p.name}
          >
            {p.name}
          </button>
        ))}
      </div>
    </div>
  );
}
