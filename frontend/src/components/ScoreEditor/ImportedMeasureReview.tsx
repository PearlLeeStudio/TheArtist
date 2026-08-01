import type { ImportMeasureSource, Measure, Song, SlotAddress } from '../../models/types';
import { hasUnresolvedSlots, isExplicitNoChord } from './importedBars';

interface ImportedMeasureReviewProps {
  song: Song;
  selectedSlots: SlotAddress[];
}

/** Reading status stays separate from the editable chords. Only the selected
 *  bar is shown, so a long imported score does not become another analysis table. */
export default function ImportedMeasureReview({ song, selectedSlots }: ImportedMeasureReviewProps) {
  const sources = song.importSource?.measures;
  if (!sources?.length) return null;
  const index = selectedSlots[0]?.measureIndex;
  const source = index === undefined ? undefined : sources[index];
  const measure = index === undefined ? undefined : song.measures[index];
  const selectedCount = new Set(selectedSlots.map((slot) => slot.measureIndex)).size;

  return (
    <div className="mb-3 space-y-1 text-xs" style={{ color: 'var(--text-muted)' }}>
      {/* Keyed by song and bar, so the details start collapsed for every newly selected bar. */}
      {source && measure && index !== undefined && (
        <SourceDetails key={`${song.id}-${index}`} index={index} source={source} measure={measure} selectedCount={selectedCount} />
      )}
    </div>
  );
}

interface SourceDetailsProps {
  /** The selected bar (the first one when several are selected). */
  index: number;
  source: ImportMeasureSource;
  measure: Measure;
  selectedCount: number;
}

/** How the selected bar was read at import, next to what the editor now holds. */
function SourceDetails({ index, source, measure, selectedCount }: SourceDetailsProps) {
  const explicitNoChord = isExplicitNoChord(measure, source);
  const hasUnresolved = hasUnresolvedSlots(measure, source);
  const inferred = source.chordBasis === 'inferred';
  const status = hasUnresolved ? 'Unresolved' : explicitNoChord ? 'N.C.' : inferred ? 'Inferred harmony' : 'Imported chords';
  const sourceChords = source.originalChords.join(' · ') || (source.originalNoChord ? 'N.C.' : 'No chord identified');

  return (
    <details style={{ borderLeft: '2px solid var(--border)', paddingLeft: 10 }}>
      <summary className="cursor-pointer" style={{ color: 'var(--text-soft)' }}>
        Bar {index + 1} · {status} · Source details{selectedCount > 1 ? ` · first of ${selectedCount} selected bars` : ''}
      </summary>
      <div className="mt-2 space-y-1 break-words leading-relaxed">
        <p>Chord reading status retains the import order. Source files and extracted note details are not kept.</p>
        {source.editorReview && <p role="status">Import note: {source.editorReview}</p>}
        <p>
          Original bar {(source.sourceMeasureIndex ?? index) + 1} · {inferred ? 'Proposed' : 'Read'}: {sourceChords} · {source.confidence} confidence.
        </p>
        {source.userNoChord && <p>You chose N.C. at import; the original did not state N.C.</p>}
        {hasUnresolved && <p>Unresolved slots have no chord assigned. They are not marked N.C.; choose their harmony in the editor.</p>}
        {source.timeSignature && <p>Source meter: {source.timeSignature.join('/')}</p>}
      </div>
    </details>
  );
}
