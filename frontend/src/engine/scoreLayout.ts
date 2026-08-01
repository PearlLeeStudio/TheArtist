/** Shared score geometry for the editor, saved projects and PDF pages. */
export const SCORE_MEASURES_PER_ROW = 4;
export const SCORE_ROW_HEIGHT = 170;
export const SCORE_TOP = 18;

const PADDING_X = 10;
const MIN_MEASURE_WIDTH = 120;
/** Extra width of the first measure in a row for its clef and key signature.
 *  The measures after it shift right by the same amount, so every measure
 *  keeps the same chord area. */
const CLEF_OFFSET = 60;
/** Space before the first chord of a measure, after its barline or after the
 *  clef and key signature. */
const CHORD_MARGIN = 10;

export interface ScoreMeasureLayout {
  x: number;
  y: number;
  width: number;
  isFirstInRow: boolean;
  contentX: number;
}

export function scoreMeasureLayout(
  index: number,
  width: number,
  options: { top?: number; rowHeight?: number } = {},
): ScoreMeasureLayout {
  const col = index % SCORE_MEASURES_PER_ROW;
  const row = Math.floor(index / SCORE_MEASURES_PER_ROW);
  const baseWidth = Math.max(MIN_MEASURE_WIDTH, (width - PADDING_X * 2 - CLEF_OFFSET) / SCORE_MEASURES_PER_ROW);
  const x = PADDING_X + col * baseWidth + (col ? CLEF_OFFSET : 0);
  const isFirstInRow = col === 0;
  return {
    x, y: (options.top ?? SCORE_TOP) + row * (options.rowHeight ?? SCORE_ROW_HEIGHT),
    width: baseWidth + (isFirstInRow ? CLEF_OFFSET : 0), isFirstInRow,
    // Chords start after the clef and key signature, or after the barline.
    contentX: x + (isFirstInRow ? CLEF_OFFSET + CHORD_MARGIN : CHORD_MARGIN),
  };
}
