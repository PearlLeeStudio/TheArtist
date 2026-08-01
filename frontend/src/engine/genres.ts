/**
 * The song's genre as the editor offers it, and the genre a picked model implies.
 * Shared by the song settings above the staff, Input → Generate and New from Song Form.
 */

/** Full 13-genre vocabulary (R4 LoRA expansion). `value` is the canonical
 *  key shared by the backend dispatch (`GENRE_MODEL_DISPATCH`) and the
 *  frontend arrangement/instrument/drum tables; `label` is display-only. */
export const GENRE_OPTIONS: { value: string; label: string }[] = [
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

/** The select value for a song genre: songs saved with the old 'bossa nova' key read as bossa. */
export function genreOptionValue(genre: string | null | undefined): string {
  return (genre === 'bossa nova' ? 'bossa' : genre) || 'none';
}

/**
 * The genre a picked model implies. Model pick always syncs song.genre —
 * Pearl 2026-05-10 found "first-time-only" sync confusing (e.g. switching
 * from blues LoRA to F4 left genre=blues, not jazz).
 *
 * Mapping:
 *   ft_f1_lora_X        → X (LoRA carries genre tag in its key)
 *   ft_f1 / phase0      → pop  (F1/Phase0 are pop-leaning)
 *   any other ft_f* key → jazz (ft_f2..ft_f5, ft_f1_v2: jazz fine-tunes)
 */
export function genreForModel(modelKey: string): string | null {
  const lora = modelKey.match(/^ft_f1_lora_(.+)$/);
  if (lora) return lora[1];
  if (modelKey === 'ft_f1' || modelKey === 'phase0') return 'pop';
  if (modelKey.startsWith('ft_f')) return 'jazz';
  return null;
}
