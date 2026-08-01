import { SCORE_MEASURES_PER_ROW, scoreMeasureLayout } from './scoreLayout.ts';

/** Page canvas size, in the proportions of the 186 × 273 mm image that
 *  pdf.ts places on each A4 page. */
export const PDF_WIDTH = 960;
export const PDF_HEIGHT = 1409;
const PDF_ROWS_PER_PAGE = 7;
export const PDF_BARS_PER_PAGE = SCORE_MEASURES_PER_ROW * PDF_ROWS_PER_PAGE;
/** Room above the first row for the title and the song details. */
export const PDF_SCORE_TOP = 110;

export function pdfPageRanges(measureCount: number): { start: number; end: number }[] {
  if (!Number.isInteger(measureCount) || measureCount < 1 || measureCount > 512) {
    throw new Error('PDF export supports 1–512 bars.');
  }
  return Array.from({ length: Math.ceil(measureCount / PDF_BARS_PER_PAGE) }, (_, page) => ({
    start: page * PDF_BARS_PER_PAGE, end: Math.min(measureCount, (page + 1) * PDF_BARS_PER_PAGE),
  }));
}

export function pdfMeasureLayout(indexOnPage: number) {
  return scoreMeasureLayout(indexOnPage, PDF_WIDTH, { top: PDF_SCORE_TOP });
}

const CREATION_DATE_PREFIX = Array.from('/CreationDate (', (char) => char.charCodeAt(0));
const CLOSING_PAREN = 0x29;
const SPACE = 0x20;

/** Remove jsPDF's automatic work timestamp without moving any PDF xref offsets. */
export function withoutPdfCreationDate(buffer: ArrayBuffer): Uint8Array<ArrayBuffer> {
  const bytes = new Uint8Array(buffer.slice(0));
  for (let start = 0; start <= bytes.length - CREATION_DATE_PREFIX.length; start++) {
    if (bytes[start] !== CREATION_DATE_PREFIX[0] ||
        !CREATION_DATE_PREFIX.every((byte, offset) => bytes[start + offset] === byte)) continue;
    let end = start + CREATION_DATE_PREFIX.length;
    while (end < bytes.length && bytes[end] !== CLOSING_PAREN) end++;
    // Blank the whole entry with spaces, so every later byte keeps its offset.
    if (end < bytes.length) bytes.fill(SPACE, start, end + 1);
  }
  return bytes;
}
