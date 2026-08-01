/**
 * State of an imported bar, shared by the score markers (Staff) and the
 * selected bar's source details (ImportedMeasureReview) so both agree on
 * what counts as N.C., unresolved or worth reviewing.
 */
import type { ImportMeasureSource, Measure } from '../../models/types';

/** At least one chord slot of the bar is empty. */
export function hasEmptySlot(measure: Measure): boolean {
  return measure.chords.some((chord) => chord === null);
}

/** Every slot is empty and the source, or the user at import, marked the bar N.C. */
export function isExplicitNoChord(measure: Measure, source: ImportMeasureSource): boolean {
  return measure.chords.every((chord) => chord === null) && !!(source.originalNoChord || source.userNoChord);
}

/** An empty slot that still needs a chord: the bar is not an explicit N.C. */
export function hasUnresolvedSlots(measure: Measure, source: ImportMeasureSource): boolean {
  return !isExplicitNoChord(measure, source) && hasEmptySlot(measure);
}

/** The chords were not read with full confidence from printed symbols, or import left a note. */
export function needsSourceReview(source: ImportMeasureSource): boolean {
  return !!source.editorReview || source.chordBasis !== 'printed' || source.confidence !== 'high';
}
