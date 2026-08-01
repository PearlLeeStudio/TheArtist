import { normalizeGenre } from '../../../engine/playSettings';

/** The song's genre as the pane screens and tooltips name it. */
export function genreTag(genre: string | null): string {
  const g = normalizeGenre(genre);
  if (!g) return 'no genre';
  return g === 'rnb_soul' ? 'r&b/soul' : g.replace('_', '-');
}
