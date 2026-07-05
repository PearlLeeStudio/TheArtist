import { useState, useEffect } from 'react';
import { useSongStore } from '../../store/songStore';

const ALL_KEYS = [
  'C major', 'G major', 'D major', 'A major', 'E major', 'B major',
  'F# major', 'Gb major', 'Db major', 'Ab major', 'Eb major', 'Bb major', 'F major',
  'A minor', 'E minor', 'B minor', 'F# minor', 'C# minor', 'G# minor',
  'Eb minor', 'Bb minor', 'F minor', 'C minor', 'G minor', 'D minor',
];

const TIME_SIGNATURES: [number, number][] = [
  [4, 4], [3, 4], [6, 8], [2, 4], [5, 4], [7, 8],
];

/** Full 13-genre vocabulary (R4 LoRA expansion). `value` is the canonical
 *  key shared by the backend dispatch (`GENRE_MODEL_DISPATCH`) and the
 *  frontend arrangement/instrument/drum tables; `label` is display-only. */
const GENRE_OPTIONS: { value: string; label: string }[] = [
  { value: 'none',       label: 'none' },
  { value: 'jazz',       label: 'jazz' },
  { value: 'pop',        label: 'pop' },
  { value: 'rock',       label: 'rock' },
  { value: 'blues',      label: 'blues' },
  { value: 'bossa',      label: 'bossa nova' },
  { value: 'classical',  label: 'classical' },
  { value: 'country',    label: 'country' },
  { value: 'rnb_soul',   label: 'R&B / soul' },
  { value: 'hip_hop',    label: 'hip-hop' },
  { value: 'electronic', label: 'electronic' },
  { value: 'funk',       label: 'funk' },
  { value: 'folk',       label: 'folk' },
  { value: 'gospel',     label: 'gospel' },
];

// Use --btn-bg (not transparent) so the dropdown options surface stays
// readable in both light (#f6f6f6) and dark (#1a1a1a) themes — a
// transparent select on a dark background would render dark-on-dark
// for the option list on some browsers.
const fieldStyle: React.CSSProperties = {
  background: 'var(--btn-bg)',
  border: '1px solid var(--border)',
  color: 'var(--text-primary)',
};

const labelStyle: React.CSSProperties = {
  color: 'var(--text-muted)',
};

/**
 * Inline horizontal settings bar — sits above the staff so the user can
 * click straight on BPM / Time / Key / Genre to change them, instead of
 * hunting in a side panel. Replaces the old vertical SongSettings card
 * that lived in the right panel.
 */
export default function SongSettings() {
  const song = useSongStore((s) => s.song);
  const setBpm = useSongStore((s) => s.setBpm);
  const setKey = useSongStore((s) => s.setKey);
  const setTimeSignature = useSongStore((s) => s.setTimeSignature);
  const setGenre = useSongStore((s) => s.setGenre);

  const [bpmDraft, setBpmDraft] = useState(String(song.bpm));
  useEffect(() => { setBpmDraft(String(song.bpm)); }, [song.bpm]);

  const commitBpm = () => {
    const n = Number(bpmDraft);
    if (Number.isFinite(n) && n > 0) setBpm(n);
    else setBpmDraft(String(song.bpm));
  };

  return (
    <div className="flex items-center justify-center flex-wrap gap-x-5 gap-y-1.5 text-sm">
      <label className="flex items-center gap-1.5">
        <span className="text-[10px] uppercase tracking-wide font-semibold" style={labelStyle}>BPM</span>
        <input
          type="number"
          value={bpmDraft}
          onChange={(e) => setBpmDraft(e.target.value)}
          onBlur={commitBpm}
          onKeyDown={(e) => {
            if (e.key === 'Enter') {
              commitBpm();
              (e.target as HTMLInputElement).blur();
            }
          }}
          className="tabular w-14  px-1.5 py-0.5 text-sm text-center"
          style={fieldStyle}
          min={20}
          max={300}
        />
      </label>
      <label className="flex items-center gap-1.5">
        <span className="text-[10px] uppercase tracking-wide font-semibold" style={labelStyle}>Time</span>
        <select
          value={`${song.timeSignature[0]}/${song.timeSignature[1]}`}
          onChange={(e) => {
            const [n, d] = e.target.value.split('/').map(Number);
            setTimeSignature([n, d]);
          }}
          className="tabular  px-1.5 py-0.5 text-sm"
          style={fieldStyle}
        >
          {TIME_SIGNATURES.map(([n, d]) => (
            <option key={`${n}/${d}`} value={`${n}/${d}`}>{n}/{d}</option>
          ))}
        </select>
      </label>
      <label className="flex items-center gap-1.5">
        <span className="text-[10px] uppercase tracking-wide font-semibold" style={labelStyle}>Key</span>
        <select
          value={song.key}
          onChange={(e) => setKey(e.target.value)}
          className=" px-1.5 py-0.5 text-sm"
          style={fieldStyle}
        >
          {ALL_KEYS.map((k) => (
            <option key={k} value={k}>{k}</option>
          ))}
        </select>
      </label>
      <label className="flex items-center gap-1.5">
        <span className="text-[10px] uppercase tracking-wide font-semibold" style={labelStyle}>Genre</span>
        <select
          // legacy persisted value 'bossa nova' maps onto the canonical key
          value={(song.genre === 'bossa nova' ? 'bossa' : song.genre) || 'none'}
          onChange={(e) => setGenre(e.target.value === 'none' ? null : e.target.value)}
          className=" px-1.5 py-0.5 text-sm"
          style={fieldStyle}
        >
          {GENRE_OPTIONS.map((g) => (
            <option key={g.value} value={g.value}>{g.label}</option>
          ))}
        </select>
      </label>
    </div>
  );
}
