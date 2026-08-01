import type { Song } from '../models/types.ts';
import { matchesCapturedRange, slotKey } from './generationComparison.ts';
import type { GenerationComparison, TargetChord } from './generationComparison.ts';

export interface ChordPreview {
  original: GenerationComparison;
  shown: GenerationComparison;
  key: string;
}

/** Restore only our preview: keep later manual edits, moved bars and replacement songs. */
export function previewRollback(song: Song, epoch: number, preview: ChordPreview): TargetChord[] {
  if (song.key !== preview.key) return [];
  const shown = new Map(preview.shown.original.map(item => [slotKey(item.address), item]));
  return preview.original.original.filter(item => {
    const expected = shown.get(slotKey(item.address));
    return expected && matchesCapturedRange({ ...preview.shown, original: [expected] }, song, epoch);
  }).map(({ address, chord }) => ({ address, chord }));
}

/** Reload recovers the last Apply plus unrelated edits, never an unconfirmed preview. */
export function withoutChordPreview(song: Song, epoch: number, preview: ChordPreview | null): Song {
  if (!preview) return song;
  const updates = new Map(previewRollback(song, epoch, preview).map(u => [slotKey(u.address), u.chord]));
  return { ...song, measures: song.measures.map((measure, measureIndex) => ({ ...measure,
    chords: measure.chords.map((chord, slotIndex) => updates.has(`${measureIndex}-${slotIndex}`)
      ? structuredClone(updates.get(`${measureIndex}-${slotIndex}`)!) : chord) as typeof measure.chords,
  })) };
}
