import type { Chord, SectionType, Song, ScoreImportMode, ImportedChordBasis, ImportConfidence, ImportMeasureSource } from '../models/types.ts';
import { parseChordName } from './chordParser.ts';
import { normalizeKeySignature } from './keySignatures.ts';
import { calculateRomanNumeral } from './romanNumeral.ts';
import { CHORDS_PER_BAR, isSectionType } from './songForm.ts';
import { generatePianoVoicing, QUALITY_INTERVALS } from './voicingEngine.ts';

export const MAX_SCORE_IMPORT_BYTES = 8 * 1024 * 1024;
export const MAX_SCORE_IMPORT_MEASURES = 512;
export const IMPORT_TIME_SIGNATURES = ['4/4', '3/4', '6/8', '2/4', '5/4', '7/8'] as const;
export type ScoreMime = 'image/png' | 'image/jpeg' | 'image/webp' | 'application/pdf';

// Size limits of a recognition response; anything larger is rejected, never truncated.
const MAX_IMPORT_SECTIONS = 64;
const MAX_IMPORT_WARNINGS = 32;
const MAX_MEASURE_CHORDS = 16;
const MAX_MEASURE_NOTES = 128;

/** One measure as the recognizer reports it. */
interface RecognizedMeasure {
  chords: string[];
  sourceLabel: string;
  needsReview: boolean;
  note: string;
  chordBasis?: ImportedChordBasis;
  confidence?: ImportConfidence;
  noChord?: boolean;
  notes?: string[];
  writtenNotes?: string[];
  bassNotes?: string[];
  timeSignature?: [number, number] | null;
  analysisSource?: 'local' | null;
  model?: string | null;
}

interface RecognizedSection {
  section: SectionType | null;
  label: string;
  startMeasure: number;
  length: number;
  basis: 'explicit' | 'inferred';
  note: string;
}

export interface ScoreImportResponse {
  mode?: ScoreImportMode;
  analysisSource?: 'local';
  title: string;
  key: string | null;
  timeSignature: [number, number] | null;
  measures: RecognizedMeasure[];
  sections: RecognizedSection[];
  warnings: string[];
  model: string | null;
}

export interface ImportMeasureDraft {
  /** Kept even when the editable chord text is corrected. */
  originalChords: string[];
  chordText: string;
  sourceLabel: string;
  needsReview: boolean;
  note: string;
  chordBasis: ImportedChordBasis;
  confidence: ImportConfidence;
  sourceMeasureIndex: number;
  analysisSource: 'local';
  model: string | null;
  notes: string[];
  writtenNotes: string[];
  bassNotes: string[];
  originalNoChord: boolean;
  timeSignature: [number, number] | null;
}

export interface ImportSectionDraft {
  type: SectionType | null;
  label: string;
  /** Inclusive, one-based bar numbers as editable text. */
  startBar: string;
  endBar: string;
  origin: 'explicit' | 'inferred' | 'manual';
  note: string;
}

export interface ScoreImportDraft {
  mode: ScoreImportMode;
  analysisSource: 'local';
  filename: string;
  title: string;
  key: string;
  timeSignature: string;
  bpm: string;
  measures: ImportMeasureDraft[];
  sections: ImportSectionDraft[];
  warnings: string[];
  model: string | null;
}

const EXTENSION_MIMES: Record<string, ScoreMime> = {
  png: 'image/png', jpg: 'image/jpeg', jpeg: 'image/jpeg', webp: 'image/webp', pdf: 'application/pdf',
};

// Leading bytes ("magic numbers") of each accepted file type.
const PNG_SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
const JPEG_SIGNATURE = [0xff, 0xd8, 0xff];
const PDF_SIGNATURE = [0x25, 0x50, 0x44, 0x46, 0x2d]; // "%PDF-"
const RIFF_SIGNATURE = [0x52, 0x49, 0x46, 0x46]; // "RIFF", a 4-byte size, then the form type
const WEBP_FORM_TYPE = [0x57, 0x45, 0x42, 0x50]; // "WEBP"
const RIFF_FORM_TYPE_OFFSET = 8;

function hasBytesAt(header: Uint8Array, bytes: readonly number[], offset = 0): boolean {
  return bytes.every((byte, i) => header[offset + i] === byte);
}

/** The accepted file type the header bytes identify, or null. */
function sniffScoreMime(header: Uint8Array): ScoreMime | null {
  if (hasBytesAt(header, PNG_SIGNATURE)) return 'image/png';
  if (hasBytesAt(header, JPEG_SIGNATURE)) return 'image/jpeg';
  if (hasBytesAt(header, PDF_SIGNATURE)) return 'application/pdf';
  if (hasBytesAt(header, RIFF_SIGNATURE) && hasBytesAt(header, WEBP_FORM_TYPE, RIFF_FORM_TYPE_OFFSET)) return 'image/webp';
  return null;
}

/** Check extension, reported MIME and magic bytes before previewing or sending. */
export function checkScoreFile(
  file: { name: string; type: string; size: number },
  header: Uint8Array,
): { ok: true; mimeType: ScoreMime } | { ok: false; message: string } {
  if (file.size <= 0) return { ok: false, message: 'The selected file is empty.' };
  if (file.size > MAX_SCORE_IMPORT_BYTES) return { ok: false, message: 'Choose a file no larger than 8 MiB.' };
  const extension = file.name.split('.').at(-1)?.toLowerCase() ?? '';
  const expected = EXTENSION_MIMES[extension];
  if (!expected) return { ok: false, message: 'Choose a PNG, JPEG, WebP or PDF file.' };
  if (file.type && file.type.toLowerCase() !== expected) {
    return { ok: false, message: 'The file type and filename extension do not match.' };
  }
  const actual = sniffScoreMime(header);
  if (actual !== expected) return { ok: false, message: 'The file contents do not match its image or PDF type.' };
  return { ok: true, mimeType: actual };
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);
const isTextUpTo = (value: unknown, maxLength: number): value is string =>
  typeof value === 'string' && value.length <= maxLength;
/** Text that is not blank. */
const isFilledTextUpTo = (value: unknown, maxLength: number): value is string =>
  isTextUpTo(value, maxLength) && value.trim() !== '';
const isListUpTo = (value: unknown, maxItems: number): value is unknown[] =>
  Array.isArray(value) && value.length <= maxItems;
const isTextList = (value: unknown, maxItems: number, maxLength: number): value is string[] =>
  isListUpTo(value, maxItems) && value.every((item) => isTextUpTo(item, maxLength));
const isOneOf = <T extends string>(value: unknown, options: readonly T[]): value is T =>
  (options as readonly unknown[]).includes(value);
/** 1–32 beats over a power-of-two note value, or null when the meter is unknown. */
const isValidMeter = (meter: unknown): boolean => meter === null || (Array.isArray(meter) && meter.length === 2 &&
  Number.isInteger(meter[0]) && meter[0] >= 1 && meter[0] <= 32 && [1, 2, 4, 8, 16, 32].includes(meter[1]));

/** The JSON body of one import request; the server refuses any other field. */
export function scoreImportRequestBody(filename: string, mimeType: ScoreMime, dataBase64: string, mode: ScoreImportMode) {
  return { filename, mimeType, dataBase64, mode };
}

/** The status of a score that was read but whose project fields still need correction. */
export function scoreImportReadyStatus(): string {
  return 'Ready to review.';
}

export function isNoChordText(value: string): boolean {
  return /^N\.?C\.?$/i.test(value.trim());
}

function invalidResponse(): never {
  throw new Error('The recognition response was incomplete or invalid. Please try again.');
}

/** Local deterministic analysis has no model; own-model suggestions name F1.
 *  Local work never names an API request: the field is absent, or null. */
function hasValidAttribution(model: unknown, requestId: unknown): boolean {
  return (model === null || model === 'ft_f1') && (requestId === undefined || requestId === null);
}

/** The measure with its optional fields filled in, or null when a field is
 *  malformed or contradicts the measure's provenance. */
function readMeasure(measure: unknown, mode: ScoreImportMode): RecognizedMeasure | null {
  if (!isRecord(measure) || !isListUpTo(measure.chords, MAX_MEASURE_CHORDS) ||
      !measure.chords.every((chord) => isFilledTextUpTo(chord, 120)) || !isTextUpTo(measure.sourceLabel, 80) ||
      typeof measure.needsReview !== 'boolean' || !isTextUpTo(measure.note, 400)) return null;
  const chordBasis = measure.chordBasis ?? 'printed';
  const confidence = measure.confidence ?? 'high';
  const noChord = measure.noChord ?? false;
  const notes = measure.notes ?? [];
  const writtenNotes = measure.writtenNotes ?? [];
  const bassNotes = measure.bassNotes ?? [];
  const timeSignature = measure.timeSignature ?? null;
  // A measure may leave its source out; it can only repeat the response's.
  const analysisSource = measure.analysisSource ?? 'local';
  const model = measure.model ?? null;

  // Note analysis states how it arrived at every chord.
  if (mode === 'infer_harmony' && (measure.chordBasis === undefined || measure.confidence === undefined)) return null;
  if (!isOneOf(chordBasis, ['printed', 'inferred', 'unreadable']) || !isOneOf(confidence, ['high', 'medium', 'low'])) return null;
  if (typeof noChord !== 'boolean' || (noChord && measure.chords.length !== 0)) return null;
  if (!isTextList(notes, MAX_MEASURE_NOTES, 80) || !isTextList(writtenNotes, MAX_MEASURE_NOTES, 80) ||
      !isTextList(bassNotes, MAX_MEASURE_NOTES, 80) || !isValidMeter(timeSignature)) return null;
  if (analysisSource !== 'local') return null;
  if (!hasValidAttribution(model, measure.requestId)) return null;
  // Inferred chords come only from note analysis, with evidence, and always need review.
  if (chordBasis === 'inferred' && (mode !== 'infer_harmony' || !measure.needsReview || !measure.note.trim())) return null;
  if (confidence === 'low' && !measure.needsReview) return null;
  // An unreadable bar has no chords and no N.C., and is low confidence pending review.
  if (chordBasis === 'unreadable' && (noChord || measure.chords.length !== 0 || confidence !== 'low' || !measure.needsReview)) return null;
  return { ...measure, chordBasis, confidence, noChord, notes, writtenNotes, bassNotes, timeSignature, analysisSource, model } as RecognizedMeasure;
}

function isRecognizedSection(section: unknown): boolean {
  return isRecord(section) && (section.section === null || isSectionType(section.section)) &&
    isTextUpTo(section.label, 80) && isTextUpTo(section.note, 400) &&
    Number.isInteger(section.startMeasure) && Number.isInteger(section.length) &&
    isOneOf(section.basis, ['explicit', 'inferred']);
}

/** An unexpected API response is an error, never a partly successful import. */
export function readScoreImportResponse(raw: unknown, expectedMode: ScoreImportMode = 'chord_symbols'): ScoreImportResponse {
  if (!isRecord(raw)) return invalidResponse();
  const mode = raw.mode ?? 'chord_symbols';
  if (!isOneOf(mode, ['chord_symbols', 'infer_harmony']) || mode !== expectedMode) return invalidResponse();
  // The response states its reader; only this app's own reader is accepted.
  const analysisSource = raw.analysisSource;
  if (analysisSource !== 'local') return invalidResponse();
  if (!isTextUpTo(raw.title, 120) || !(raw.key === null || isTextUpTo(raw.key, 40)) || !isValidMeter(raw.timeSignature) ||
      !hasValidAttribution(raw.model, raw.requestId) ||
      !isListUpTo(raw.measures, MAX_SCORE_IMPORT_MEASURES) || !isListUpTo(raw.sections, MAX_IMPORT_SECTIONS) ||
      !isTextList(raw.warnings, MAX_IMPORT_WARNINGS, 400)) return invalidResponse();
  const measures: RecognizedMeasure[] = [];
  for (const measure of raw.measures) {
    const recognized = readMeasure(measure, mode);
    if (!recognized) return invalidResponse();
    measures.push(recognized);
  }
  for (const section of raw.sections) {
    if (!isRecognizedSection(section)) return invalidResponse();
  }
  return { ...raw, mode, analysisSource, measures } as unknown as ScoreImportResponse;
}

const copyMeter = (meter: [number, number] | null | undefined): [number, number] | null =>
  meter ? [...meter] as [number, number] : null;

function measureDraft(measure: RecognizedMeasure, index: number): ImportMeasureDraft {
  return {
    needsReview: measure.needsReview,
    // Recognition is complete. Only its chords and reading status enter the draft.
    sourceLabel: '',
    note: '',
    chordBasis: measure.chordBasis ?? 'printed',
    confidence: measure.confidence ?? 'high',
    sourceMeasureIndex: index,
    analysisSource: 'local',
    model: measure.model ?? null,
    notes: [],
    writtenNotes: [],
    bassNotes: [],
    originalNoChord: measure.noChord ?? false,
    timeSignature: copyMeter(measure.timeSignature),
    originalChords: [...measure.chords],
    chordText: measure.noChord ? 'N.C.' : measure.chords.join(' '),
  };
}

export function createScoreImportDraft(response: ScoreImportResponse, filename: string): ScoreImportDraft {
  // Older callers pass a filename; it is no longer kept as project content.
  void filename;
  return {
    mode: response.mode ?? 'chord_symbols',
    analysisSource: 'local',
    filename: '',
    title: response.title || 'Imported harmony',
    key: response.key ?? '',
    timeSignature: response.timeSignature?.join('/') ?? '',
    bpm: '80',
    measures: response.measures.map((measure, index) => measureDraft(measure, index)),
    sections: response.sections.map((section) => ({
      type: section.section,
      label: section.label,
      startBar: String(section.startMeasure + 1),
      endBar: String(section.startMeasure + section.length),
      origin: section.basis,
      note: '',
    })),
    warnings: [],
    model: response.model,
  };
}

/** A root, a suffix without spaces, slashes or parentheses, and an optional slash bass. */
const IMPORTED_CHORD_SYMBOL = /^[A-G](?:#|b)?(?:m?6\/9|[^\s/()]*)(?:\/[A-G](?:#|b)?)?$/;

/** Deliberately stricter than manual chord entry: no unknown suffix may
 *  silently fall back to a major triad, and parenthesized extensions are
 *  not supported by the voicing engine. The original text stays in review. */
export function parseImportedChord(symbol: string, key: string): Chord | null {
  const normalized = symbol.trim().replaceAll('♯', '#').replaceAll('♭', 'b');
  if (!IMPORTED_CHORD_SYMBOL.test(normalized)) return null;
  const chord = parseChordName(normalized);
  if (!chord || chord.extensions.length !== 0 || !Object.hasOwn(QUALITY_INTERVALS, chord.quality)) return null;
  return { ...chord, voicing: generatePianoVoicing(chord), romanNumeral: calculateRomanNumeral(chord, key) };
}

/** The chord symbols of a measure's editable text. */
function chordSymbols(chordText: string): string[] {
  return chordText.trim().split(/\s+/).filter(Boolean);
}

/** Analyze the edited symbols against the selected review key, without transposing them. */
export function importMeasureRomanPreview(chordText: string, keyText: string): string | null {
  const key = normalizeKeySignature(keyText);
  if (!key) return null;
  if (isNoChordText(chordText)) return 'N.C.';
  const symbols = chordSymbols(chordText);
  if (symbols.length < 1 || symbols.length > MAX_MEASURE_CHORDS) return null;
  const chords = symbols.map((symbol) => parseImportedChord(symbol, key));
  if (chords.some((chord) => chord === null)) return null;
  return chords.map((chord) => chord!.romanNumeral).join(' · ');
}

export interface ScoreImportValidation {
  valid: boolean;
  errors: string[];
  measureErrors: string[];
  sectionErrors: string[];
  singleChordBars: number;
}

const isWholeNumberText = (text: string): boolean => /^\d+$/.test(text);

/** Problems with the project fields, in the order the review lists them. */
function projectFieldErrors(draft: ScoreImportDraft, key: string | null): string[] {
  const errors: string[] = [];
  const title = draft.title.trim();
  if (!title || title.length > 120) errors.push('Enter a title of 1–120 characters.');
  if (!key) errors.push('Choose one of the 30 standard major/minor keys (up to seven sharps or flats). Correct unsupported keys manually before importing.');
  if (!(IMPORT_TIME_SIGNATURES as readonly string[]).includes(draft.timeSignature)) errors.push('Choose one of the time signatures supported by the editor.');
  if (!isWholeNumberText(draft.bpm) || Number(draft.bpm) < 20 || Number(draft.bpm) > 300) errors.push('Playback tempo must be a whole number from 20 to 300 BPM.');
  if (draft.measures.length === 0 || draft.measures.length > MAX_SCORE_IMPORT_MEASURES) errors.push('The score must contain 1–512 measures.');
  if (draft.sections.length > MAX_IMPORT_SECTIONS) errors.push('The score can contain at most 64 sections.');
  return errors;
}

/** Why a measure's chord text cannot fill the editor's chord slots, or ''. */
function measureChordError(measure: ImportMeasureDraft, key: string): string {
  if (isNoChordText(measure.chordText)) return '';
  const symbols = chordSymbols(measure.chordText);
  if (symbols.length > CHORDS_PER_BAR) return `${symbols.length} changes were kept in the analysis. Choose at most two chords for the editor's equal slots; no changes are dropped automatically.`;
  if (symbols.length === 0) return 'Enter one or two chords, or explicitly enter N.C. to leave this bar without harmony.';
  const unsupported = symbols.filter((symbol) => !parseImportedChord(symbol, key));
  if (unsupported.length) return `Unsupported or unreadable chord: ${unsupported.join(', ')}. Correct it before importing.`;
  return '';
}

function sectionFieldError(section: ImportSectionDraft, measureCount: number): string {
  const label = section.label.trim();
  if (!label || label.length > 80) return 'Enter a section name of 1–80 characters.';
  if (section.type !== null && !isSectionType(section.type)) return 'Choose a section type or Unknown.';
  if (!isWholeNumberText(section.startBar) || !isWholeNumberText(section.endBar)) return 'Enter whole bar numbers.';
  const start = Number(section.startBar);
  const end = Number(section.endBar);
  if (start < 1 || end < start || end > measureCount) return `Use a range within bars 1–${measureCount}.`;
  return '';
}

/** One message per section ('' when it is fine). Sections whose own fields
 *  are valid are then checked for overlaps, and both sides of an overlap
 *  are marked. */
function sectionErrorsFor(sections: ImportSectionDraft[], measureCount: number): string[] {
  const errors = sections.map((section) => sectionFieldError(section, measureCount));
  const sectionByBar = new Map<number, number>();
  sections.forEach((section, i) => {
    if (errors[i]) return;
    for (let bar = Number(section.startBar); bar <= Number(section.endBar); bar++) {
      const other = sectionByBar.get(bar);
      if (other !== undefined) {
        errors[i] = `Overlaps section ${other + 1} at bar ${bar}.`;
        errors[other] = `Overlaps section ${i + 1} at bar ${bar}.`;
      } else sectionByBar.set(bar, i);
    }
  });
  return errors;
}

function validateScoreImport(draft: ScoreImportDraft, reviewChords: boolean): ScoreImportValidation {
  const key = normalizeKeySignature(draft.key);
  const errors = projectFieldErrors(draft, key);
  // The key only affects the Roman numerals, so any key can check that the symbols parse.
  const measureErrors = reviewChords ? draft.measures.map((measure) => measureChordError(measure, key ?? 'C major')) : [];
  // Checked bars with exactly one chord (N.C. bars excluded); 0 when chords are not reviewed.
  const singleChordBars = reviewChords
    ? draft.measures.filter((measure, i) =>
      !measureErrors[i] && !isNoChordText(measure.chordText) && chordSymbols(measure.chordText).length === 1).length
    : 0;
  const sectionErrors = sectionErrorsFor(draft.sections, draft.measures.length);
  return { valid: errors.length === 0 && !measureErrors.some(Boolean) && !sectionErrors.some(Boolean), errors, measureErrors, sectionErrors, singleChordBars };
}

/** Chord correction belongs to the score view. Only project metadata blocks
 *  opening a draft there; unresolved source material remains attached. */
export function validateScoreImportForEditor(draft: ScoreImportDraft): ScoreImportValidation {
  return validateScoreImport(draft, false);
}

export function validateScoreImportDraft(draft: ScoreImportDraft): ScoreImportValidation {
  return validateScoreImport(draft, true);
}

function pendingMeasureReview(measure: ImportMeasureDraft, key: string): string | undefined {
  if (isNoChordText(measure.chordText)) return undefined;
  const symbols = chordSymbols(measure.chordText);
  if (!symbols.length) return 'No chord was determined. Choose a chord in the score view.';
  if (symbols.length > CHORDS_PER_BAR) return `${symbols.length} source changes are preserved. Place the desired changes in the editor's two chord slots.`;
  if (symbols.some((symbol) => !parseImportedChord(symbol, key))) return 'Some source chord symbols need correction in the score view.';
  return undefined;
}

function importedMeasureChords(measure: ImportMeasureDraft, key: string): [Chord | null, Chord | null] {
  const symbols = chordSymbols(measure.chordText);
  // Do not turn an unresolved bar into N.C., invent a chord, or silently choose
  // two out of a longer progression. All source symbols stay in importSource.
  if (isNoChordText(measure.chordText) || symbols.length === 0 || symbols.length > CHORDS_PER_BAR) return [null, null];
  // A single chord fills both slots.
  return [parseImportedChord(symbols[0], key), parseImportedChord(symbols[1] ?? symbols[0], key)];
}

function importedMeasureSource(measure: ImportMeasureDraft, key: string, deferChordReview: boolean): ImportMeasureSource {
  const editorReview = (deferChordReview ? pendingMeasureReview(measure, key) : undefined)
    ?? (measure.needsReview ? 'Review imported chords against your source.' : undefined);
  return {
    chordBasis: measure.chordBasis, confidence: measure.confidence,
    evidence: '', originalChords: [...measure.originalChords],
    // requestId is always null; the field stays so the stored project shape is unchanged.
    sourceMeasureIndex: measure.sourceMeasureIndex, analysisSource: measure.analysisSource, model: measure.model, requestId: null,
    originalNoChord: measure.originalNoChord,
    userNoChord: isNoChordText(measure.chordText) && !measure.originalNoChord,
    timeSignature: copyMeter(measure.timeSignature),
    ...(editorReview ? { editorReview } : {}),
  };
}

/** Strict conversion remains available for callers that finish chord review. */
export function buildImportedSong(draft: ScoreImportDraft, makeId: () => string = () => crypto.randomUUID()): Song {
  return buildScoreImport(draft, makeId, false);
}

/** Open the analysis as an editable score, retaining unfinished chord reading. */
export function buildScoreImportForEditor(draft: ScoreImportDraft, makeId: () => string = () => crypto.randomUUID()): Song {
  return buildScoreImport(draft, makeId, true);
}

/** Source bytes and preview URLs are never included in the saved Song. */
function buildScoreImport(draft: ScoreImportDraft, makeId: () => string, deferChordReview: boolean): Song {
  const validation = deferChordReview ? validateScoreImportForEditor(draft) : validateScoreImportDraft(draft);
  if (!validation.valid) throw new Error(deferChordReview
    ? 'Correct the highlighted project fields and sections before importing.'
    : 'Correct the highlighted measures and sections before importing.');
  const key = normalizeKeySignature(draft.key)!;
  const meter = draft.timeSignature.split('/').map(Number) as [number, number];
  return {
    id: makeId(),
    title: draft.title.trim(),
    key,
    bpm: Number(draft.bpm),
    genre: null,
    timeSignature: meter,
    autoTitle: false,
    measures: draft.measures.map((measure, index) => ({
      id: makeId(),
      index,
      chords: importedMeasureChords(measure, key),
    })),
    sections: draft.sections.map((section) => ({
      id: makeId(),
      type: section.type,
      label: section.label.trim(),
      startMeasure: Number(section.startBar) - 1,
      length: Number(section.endBar) - Number(section.startBar) + 1,
      origin: section.origin,
    })).sort((a, b) => a.startMeasure - b.startMeasure),
    // filename ('') and requestId (null) carry nothing; saved-project validation
    // (store/sessionStorage.ts, also in earlier builds) requires both fields.
    importSource: {
      filename: '', model: draft.model, requestId: null, mode: draft.mode,
      analysisSource: draft.analysisSource,
      measures: draft.measures.map((measure) => importedMeasureSource(measure, key, deferChordReview)),
    },
  };
}
