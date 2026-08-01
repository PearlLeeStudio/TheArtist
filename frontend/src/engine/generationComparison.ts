import type { Chord, SectionType, SlotAddress, Song } from '../models/types.ts';

export interface TargetChord {
  address: SlotAddress;
  chord: Chord | null;
}

/** A request's original range, independent of the selection made afterwards. */
export interface GenerationComparison {
  songEpoch: number;
  songId: string;
  kind: 'selection' | 'section';
  original: Array<TargetChord & { measureId: string }>;
  /** The section type the drag-path Generate sent (null = no song-form rules), so a
   *  later review reads the run as it was, even after bars are added. Absent when
   *  the run did not record one. */
  sectionType?: SectionType | null;
}

export function slotKey(address: SlotAddress): string {
  return `${address.measureIndex}-${address.slotIndex}`;
}

export function captureGenerationComparison(
  song: Song,
  slots: readonly SlotAddress[],
  songEpoch: number,
  kind: GenerationComparison['kind'] = 'selection',
): GenerationComparison {
  const seen = new Set<string>();
  const original: GenerationComparison['original'] = [];
  for (const address of slots) {
    const measure = song.measures[address.measureIndex];
    if (!measure || (address.slotIndex !== 0 && address.slotIndex !== 1) || seen.has(slotKey(address))) continue;
    seen.add(slotKey(address));
    original.push({
      address: { ...address },
      measureId: measure.id,
      chord: structuredClone(measure.chords[address.slotIndex]),
    });
  }
  original.sort((a, b) => a.address.measureIndex - b.address.measureIndex || a.address.slotIndex - b.address.slotIndex);
  return { songEpoch, songId: song.id, kind, original };
}

/** Epoch catches loading a saved copy with the same song id. Measure ids also
 * reject a range that was removed or moved while its request was running. */
export function canApplyComparison(comparison: GenerationComparison, song: Song, songEpoch: number): boolean {
  return comparison.songEpoch === songEpoch && comparison.songId === song.id && comparison.original.length > 0 &&
    comparison.original.every(({ address, measureId }) => song.measures[address.measureIndex]?.id === measureId);
}

/** The candidate's chord for `address`, or `original` when the candidate has none there. */
function candidateChordAt(
  candidate: ReadonlyMap<string, Chord | null> | undefined,
  address: SlotAddress,
  original: Chord | null,
): Chord | null {
  const key = slotKey(address);
  return candidate?.has(key) ? candidate.get(key)! : original;
}

/** Apply each candidate over the captured original, never over another
 * candidate. Missing/invalid response slots keep their original chord;
 * unselected slots in the server's whole-measure response are ignored. */
export function comparisonUpdates(
  comparison: GenerationComparison,
  song: Song,
  songEpoch: number,
  candidate?: ReadonlyMap<string, Chord | null>,
): TargetChord[] {
  if (!canApplyComparison(comparison, song, songEpoch)) return [];
  return comparison.original.map(({ address, chord }) => ({
    address: { ...address },
    chord: structuredClone(candidateChordAt(candidate, address, chord)),
  }));
}

/** Roman numerals are derived from the key; they do not change the chord. */
function sameChord(a: Chord | null, b: Chord | null, compareVoicing: boolean): boolean {
  if (a === null || b === null) return a === b;
  return a.root === b.root && a.quality === b.quality && (a.bass ?? '') === (b.bass ?? '') &&
    JSON.stringify([...a.extensions].sort()) === JSON.stringify([...b.extensions].sort()) &&
    (!compareVoicing || JSON.stringify(a.voicing) === JSON.stringify(b.voicing));
}

export function matchesCapturedRange(comparison: GenerationComparison, song: Song, songEpoch: number): boolean {
  return canApplyComparison(comparison, song, songEpoch) && comparison.original.every(({ address, chord }) =>
    sameChord(chord, song.measures[address.measureIndex].chords[address.slotIndex], true));
}

/** Chord-symbol differences for the preview. Candidate voicings are resolved
 * when applied, so empty voicings from the parser are not counted as edits. */
export function compareCandidate(comparison: GenerationComparison, candidate: ReadonlyMap<string, Chord | null>) {
  return comparison.original.map(({ address, chord }) => {
    const key = slotKey(address);
    const after = candidate.has(key) ? candidate.get(key)! : chord;
    return { address, before: chord, after, changed: !sameChord(chord, after, false) };
  });
}
