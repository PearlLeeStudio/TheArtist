import { CanvasContext, Renderer, Stave } from 'vexflow/bravura';
import { jsPDF } from 'jspdf';
import type { Chord, Song } from '../models/types';
import { getChordDisplayParts } from './chordDisplay';
import { DEFAULT_SONG_TITLE, sectionAt, sectionColor, sectionDisplayLabel } from './songForm';
import { keySignatureFor } from './keySignatures';
import {
  PDF_WIDTH, PDF_HEIGHT, pdfPageRanges, pdfMeasureLayout, withoutPdfCreationDate,
} from './pdfLayout';

export type PdfTheme = 'current' | 'light' | 'dark';
const FONT = '-apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo", "Noto Sans KR", sans-serif';
const PALETTES = {
  light: { background: '#ffffff', ink: '#171717', chord: '#404040', muted: '#525252', sectionAlpha: 0.12 },
  dark: { background: '#0a0a0a', ink: '#eaeaea', chord: '#d6d6d6', muted: '#9e9e9e', sectionAlpha: 0.17 },
};
type Palette = typeof PALETTES.light;
// VexFlow supplies its own OffscreenCanvas type. Use its actual context contract.
type DrawingContext = CanvasContext['context2D'];

// A chord symbol is measured and then drawn with the same three fonts.
const CHORD_ROOT_FONT = `600 24px ${FONT}`;
const CHORD_ANNOTATION_FONT = `12px ${FONT}`;
const CHORD_BASS_FONT = `20px ${FONT}`;
/** Space between a chord root and its superscript and subscript. */
const ANNOTATION_GAP = 2;
const MUSIC_FONT = '16px Bravura';
/** Canvas pixels per layout pixel, for a sharp page image. */
const RENDER_SCALE = 3;
// Each page image fills A4 (210 × 297 mm) inside 12 mm margins.
const PAGE_MARGIN_MM = 12;
const PAGE_IMAGE_WIDTH_MM = 186;
const PAGE_IMAGE_HEIGHT_MM = 273;

function fittedText(ctx: DrawingContext, text: string, x: number, y: number, width: number): void {
  ctx.save();
  ctx.translate(x, y);
  ctx.scale(Math.min(1, width / Math.max(1, ctx.measureText(text).width)), 1);
  ctx.fillText(text, 0, 0);
  ctx.restore();
}

/** A chord symbol centered on (cx, y), shrunk to fit `width`, with its Roman numeral below. */
function drawChord(ctx: DrawingContext, chord: Chord, cx: number, y: number, width: number, palette: Palette) {
  const parts = getChordDisplayParts(chord);
  const sup = parts.sup + (chord.extensions.length ? `(${chord.extensions.join(',')})` : '');
  ctx.font = CHORD_ROOT_FONT;
  const rootWidth = ctx.measureText(parts.root).width;
  ctx.font = CHORD_ANNOTATION_FONT;
  const annotationWidth = Math.max(ctx.measureText(sup).width, ctx.measureText(parts.sub).width);
  ctx.font = CHORD_BASS_FONT;
  const bass = parts.bass ? `/${parts.bass}` : '';
  const bassWidth = ctx.measureText(bass).width;
  const gapIfAnnotated = annotationWidth ? ANNOTATION_GAP : 0;
  const total = rootWidth + annotationWidth + bassWidth + gapIfAnnotated;
  const left = -total / 2;
  const scale = Math.min(1, Math.max(10, width - 14) / Math.max(total, 1));
  ctx.save();
  ctx.translate(cx, y);
  ctx.scale(scale, scale);
  // Mask the staff lines behind the symbol.
  ctx.fillStyle = palette.background;
  ctx.fillRect(left - 5, -22, total + 10, 40);
  ctx.fillStyle = palette.chord;
  ctx.font = CHORD_ROOT_FONT;
  ctx.fillText(parts.root, left, 8);
  ctx.font = CHORD_ANNOTATION_FONT;
  ctx.fillText(sup, left + rootWidth + ANNOTATION_GAP, -7);
  ctx.fillText(parts.sub, left + rootWidth + ANNOTATION_GAP, 14);
  ctx.font = CHORD_BASS_FONT;
  ctx.fillText(bass, left + rootWidth + annotationWidth + gapIfAnnotated, 8);
  ctx.restore();
  ctx.fillStyle = palette.muted;
  ctx.font = `17px ${FONT}`;
  ctx.save();
  ctx.textAlign = 'center';
  fittedText(ctx, chord.romanNumeral, cx, y + 54, width - 14);
  ctx.restore();
}

function resolveTheme(theme: PdfTheme): 'light' | 'dark' {
  if (theme !== 'current') return theme;
  return document.documentElement.classList.contains('light') ? 'light' : 'dark';
}

async function waitForMusicFont(): Promise<void> {
  await document.fonts.load(MUSIC_FONT);
  await document.fonts.ready;
  if (!document.fonts.check(MUSIC_FONT)) throw new Error('The music font is not ready. Refresh the page and try exporting again.');
}

/** A hidden canvas in the document to render the pages on. */
function mountScratchCanvas(): { scratch: HTMLDivElement; canvas: HTMLCanvasElement } {
  const scratch = document.createElement('div');
  scratch.setAttribute('aria-hidden', 'true');
  scratch.style.cssText = 'position:fixed;left:-100000px;top:0;pointer-events:none;';
  const canvas = document.createElement('canvas');
  scratch.append(canvas);
  document.body.append(scratch);
  return { scratch, canvas };
}

/** Page background, title, and the key · tempo · meter · genre line. */
function drawPageHeader(ctx: DrawingContext, song: Song, palette: Palette) {
  ctx.fillStyle = palette.background;
  ctx.fillRect(0, 0, PDF_WIDTH, PDF_HEIGHT);
  ctx.fillStyle = palette.ink;
  ctx.font = `600 30px ${FONT}`;
  fittedText(ctx, song.title.trim() || DEFAULT_SONG_TITLE, 10, 42, PDF_WIDTH - 20);
  ctx.fillStyle = palette.muted;
  ctx.font = `16px ${FONT}`;
  const details = [song.key, `${song.bpm} BPM`, song.timeSignature.join('/'), song.genre].filter(Boolean).join('  ·  ');
  fittedText(ctx, details, 10, 75, PDF_WIDTH - 20);
}

/** The stave, section band and label, bar number and chords of one measure. */
function drawMeasure(context: CanvasContext, song: Song, index: number, indexOnPage: number, palette: Palette) {
  const ctx = context.context2D;
  const layout = pdfMeasureLayout(indexOnPage);
  const stave = new Stave(layout.x, layout.y, layout.width);
  if (layout.isFirstInRow) {
    stave.addClef('treble');
    const key = keySignatureFor(song.key);
    if (!key) throw new Error(`Unsupported PDF key signature: ${song.key}`);
    stave.addKeySignature(key);
  }
  if (index === 0) stave.addTimeSignature(song.timeSignature.join('/'));
  context.setFillStyle(palette.ink);
  context.setStrokeStyle(palette.ink);
  stave.setContext(context).draw();
  const contentX = Math.max(layout.contentX, stave.getNoteStartX() + 5);
  const right = layout.x + layout.width;
  const section = sectionAt(song, index);
  if (section) {
    ctx.fillStyle = sectionColor(section.type);
    ctx.globalAlpha = palette.sectionAlpha;
    ctx.fillRect(contentX, layout.y + 1, right - contentX, 23);
    ctx.globalAlpha = 1;
    // Name the section where it starts and again at the start of each row.
    if (index === section.startMeasure || layout.isFirstInRow) {
      ctx.fillStyle = palette.ink;
      ctx.font = `600 13px ${FONT}`;
      fittedText(ctx, sectionDisplayLabel(section), contentX + 4, layout.y + 17, right - contentX - 9);
    }
  }
  ctx.fillStyle = palette.muted;
  ctx.font = `11px ${FONT}`;
  ctx.fillText(String(index + 1), layout.x + 3, layout.y + 32);
  const slotWidth = (right - contentX - 5) / 2;
  song.measures[index].chords.forEach((chord, slot) => {
    if (chord) drawChord(ctx, chord, contentX + slotWidth * (slot + 0.5), stave.getYForLine(2), slotWidth, palette);
  });
}

function drawPageFooter(ctx: DrawingContext, pageIndex: number, pageCount: number, palette: Palette) {
  ctx.fillStyle = palette.muted;
  ctx.font = `13px ${FONT}`;
  ctx.fillText('TheArtist', 10, PDF_HEIGHT - 20);
  ctx.textAlign = 'right';
  ctx.fillText(`${pageIndex + 1} / ${pageCount}`, PDF_WIDTH - 10, PDF_HEIGHT - 20);
  ctx.textAlign = 'left';
}

/** Render a score snapshot, independent of the editor viewport, selection and theme changes. */
export async function createSongPdf(song: Song, theme: PdfTheme = 'current'): Promise<Blob> {
  const snapshot = structuredClone(song);
  const ranges = pdfPageRanges(snapshot.measures.length);
  if (snapshot.sections?.some((section) => section.pending)) throw new Error('Wait for section generation to finish before exporting.');
  const palette = PALETTES[resolveTheme(theme)];
  await waitForMusicFont();
  const { scratch, canvas } = mountScratchCanvas();
  try {
    const renderer = new Renderer(canvas, Renderer.Backends.CANVAS);
    const context = renderer.getContext() as CanvasContext;
    const pdf = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4', compress: true });
    for (const [pageIndex, range] of ranges.entries()) {
      context.resize(PDF_WIDTH, PDF_HEIGHT, RENDER_SCALE);
      const ctx = context.context2D;
      drawPageHeader(ctx, snapshot, palette);
      for (let index = range.start; index < range.end; index++) {
        drawMeasure(context, snapshot, index, index - range.start, palette);
      }
      drawPageFooter(ctx, pageIndex, ranges.length, palette);
      if (pageIndex) pdf.addPage();
      pdf.addImage(canvas.toDataURL('image/png'), 'PNG', PAGE_MARGIN_MM, PAGE_MARGIN_MM,
        PAGE_IMAGE_WIDTH_MM, PAGE_IMAGE_HEIGHT_MM, undefined, 'FAST');
      // Let the page paint its busy state while longer scores render.
      await new Promise<void>((resolve) => setTimeout(resolve, 0));
    }
    return new Blob([withoutPdfCreationDate(pdf.output('arraybuffer'))], { type: 'application/pdf' });
  } finally {
    scratch.remove();
    canvas.width = 0;
    canvas.height = 0;
  }
}
