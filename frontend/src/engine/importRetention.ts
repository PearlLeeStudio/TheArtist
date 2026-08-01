import type { ImportMeasureSource, Song } from '../models/types';

/** Keep the arrangement and minimal reading provenance, never source pitches or prose. */
export function withoutImportedContent(song: Song): Song {
  const source = song.importSource;
  if (!source || typeof source !== 'object' || Array.isArray(source)) return song;
  return {
    ...song,
    importSource: {
      filename: '',
      model: source.model,
      requestId: source.requestId,
      ...(source.mode !== undefined ? { mode: source.mode } : {}),
      ...(source.analysisSource !== undefined ? { analysisSource: source.analysisSource } : {}),
      ...(Array.isArray(source.measures) ? { measures: source.measures.map(withoutMeasureContent) } : {}),
    },
  };
}

function withoutMeasureContent(source: ImportMeasureSource): ImportMeasureSource {
  if (!source || typeof source !== 'object' || Array.isArray(source)) return source;
  return {
    chordBasis: source.chordBasis,
    confidence: source.confidence,
    evidence: '',
    originalChords: Array.isArray(source.originalChords) ? [...source.originalChords] : source.originalChords,
    ...(source.editorReview !== undefined ? { editorReview: retainedReviewMessage(source.editorReview) } : {}),
    ...(source.sourceMeasureIndex !== undefined ? { sourceMeasureIndex: source.sourceMeasureIndex } : {}),
    ...(source.analysisSource !== undefined ? { analysisSource: source.analysisSource } : {}),
    ...(source.model !== undefined ? { model: source.model } : {}),
    ...(source.requestId !== undefined ? { requestId: source.requestId } : {}),
    ...(source.originalNoChord !== undefined ? { originalNoChord: source.originalNoChord } : {}),
    ...(source.userNoChord !== undefined ? { userNoChord: source.userNoChord } : {}),
    ...(source.timeSignature !== undefined ? { timeSignature: source.timeSignature } : {}),
  };
}

/** Application-written review statuses are useful; old reader prose may contain score content. */
function retainedReviewMessage(message: string): string {
  const statuses = [
    'No chord was determined. Choose a chord in the score view.',
    'Some source chord symbols need correction in the score view.',
    // Written only by earlier builds; kept so projects saved by them keep their wording.
    'Proposed from the original after import; the local reader had none.',
    'Check against the original: the readings differ.',
    'Review imported chords against your source.',
  ];
  if (statuses.includes(message) || /^\d+ source changes are preserved\. Place the desired changes in the editor's two chord slots\.$/.test(message)) return message;
  return 'Review imported chords against your source.';
}
