import { useState } from 'react';
import { useSongStore } from '../../store/songStore';
import { STANDARD_KEYS } from '../../engine/keySignatures';
import { GENRE_OPTIONS, genreOptionValue } from '../../engine/genres';

const TIME_SIGNATURES: [number, number][] = [
  [4, 4], [3, 4], [6, 8], [2, 4], [5, 4], [7, 8],
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

  // BPM text while it is being typed; null means the field shows song.bpm.
  // Committing keeps a positive number and discards anything else.
  const [bpmDraft, setBpmDraft] = useState<string | null>(null);

  const commitBpm = () => {
    const bpm = Number(bpmDraft ?? song.bpm);
    if (Number.isFinite(bpm) && bpm > 0) setBpm(bpm);
    setBpmDraft(null);
  };

  return (
    <div className="flex items-center justify-center flex-wrap gap-x-5 gap-y-1.5 text-sm">
      <label className="flex items-center gap-1.5">
        <span className="text-[10px] uppercase tracking-wide font-semibold" style={labelStyle}>BPM</span>
        <input
          type="number"
          value={bpmDraft ?? String(song.bpm)}
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
          {/* An unsupported song key stays listed (disabled), so the select
              does not display a key the song does not have. */}
          {!STANDARD_KEYS.includes(song.key) && (
            <option value={song.key} disabled>{song.key} — choose a supported key</option>
          )}
          {STANDARD_KEYS.map((k) => (
            <option key={k} value={k}>{k}</option>
          ))}
        </select>
      </label>
      <label className="flex items-center gap-1.5">
        <span className="text-[10px] uppercase tracking-wide font-semibold" style={labelStyle}>Genre</span>
        <select
          // legacy persisted value 'bossa nova' maps onto the canonical key
          value={genreOptionValue(song.genre)}
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
