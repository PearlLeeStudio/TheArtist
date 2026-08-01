import type { Chord, PlaySettings, SavedSession, Song, SongSection } from '../models/types';
import { formatChordSymbol } from './chordDisplay.ts';
import { playSettingsFromSession, resolveVoicingMode, sanitizePlaySettings } from './playSettings.ts';
import type { VoicingMode } from './playSettings.ts';
import { guitarVoicingNotes, resolvePianoVoicing } from './voicingEngine.ts';
import { checkSectionRanges, sectionDisplayLabel } from './songForm.ts';

export interface ExportChord { symbol: string; voicing: number[] }
export interface ExportSpec {
  title: string;
  bpm: number;
  key: string;
  timeSignature: [number, number];
  genre: string | null;
  measures: { chords: [ExportChord | null, ExportChord | null] }[];
  sections: { label: string; startMeasure: number; length: number }[];
  play: Omit<PlaySettings, 'visualMode'>;
}

/** A saved card's complete export input, independent of the active editor and later edits. */
export function snapshotSessionExport(session: SavedSession): { song: Song; settings: PlaySettings } {
  const song = structuredClone(session.song);
  song.genre = session.generation.genre ?? song.genre ?? null;
  // A renamed project should keep its visible name in both the file and its contents.
  song.title = session.name.trim() || song.title;
  return { song, settings: playSettingsFromSession(session.generation) };
}

// The backend's ExportSpec limits (backend/app/models/export.py).
const MAX_EXPORT_BARS = 512;
const MAX_EXPORT_SECTIONS = 128;
const MAX_VOICING_NOTES = 16;
const MAX_MIDI_NOTE = 127;

/** How long the backend may take to render an export file. */
const EXPORT_TIMEOUT_MS = 60_000;
/** The chunk id every Standard MIDI File starts with. */
const MIDI_FILE_HEADER = 'MThd';
/** The media types a JavaScript file is served under: the current one and the
 *  older registration servers and proxies still send. */
const JAVASCRIPT_MEDIA_TYPES = new Set(['text/javascript', 'application/javascript']);

const WINDOWS_RESERVED_NAME = /^(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)/i;
const MAX_FILENAME_STEM_CHARS = 100;
const LOOPBACK_NAMES = new Set(['localhost', '::1', '[::1]']);

/** The backend takes 1–16 whole MIDI note numbers per chord. */
function isExportableVoicing(voicing: number[]): boolean {
  return voicing.length > 0 && voicing.length <= MAX_VOICING_NOTES &&
    voicing.every((note) => Number.isInteger(note) && note >= 0 && note <= MAX_MIDI_NOTE);
}

/** One chord slot with the notes the editor plays for it in `voicingMode`. */
function exportChord(chord: Chord | null, voicingMode: VoicingMode): ExportChord | null {
  if (!chord) return null;
  // A copy: the piano voicing may be the chord's own stored array.
  const voicing = voicingMode === 'guitar' ? guitarVoicingNotes(chord) : [...resolvePianoVoicing(chord)];
  if (!isExportableVoicing(voicing)) {
    throw new Error(`Cannot export the voicing for ${formatChordSymbol(chord)}.`);
  }
  return { symbol: formatChordSymbol(chord), voicing };
}

/** Throws the reason the song cannot be exported (yet), if there is one. */
function assertExportable(song: Song, sections: SongSection[]): void {
  if (!song.measures.length || song.measures.length > MAX_EXPORT_BARS) {
    throw new Error(`Export supports 1–${MAX_EXPORT_BARS} bars.`);
  }
  if (sections.some((section) => section.pending)) throw new Error('Wait for section generation to finish before exporting.');
  if (sections.length > MAX_EXPORT_SECTIONS) throw new Error(`Export supports up to ${MAX_EXPORT_SECTIONS} sections.`);
  const sectionError = checkSectionRanges(sections, song.measures.length);
  if (sectionError) throw new Error(sectionError);
}

/** The Play settings sent with the export. visualMode is not part of the
 *  wire format: it only decides the voicings. */
function exportPlaySettings(settings: PlaySettings): ExportSpec['play'] {
  return {
    harmonyOverride: settings.harmonyOverride,
    harmonyRhythmId: settings.harmonyRhythmId,
    bassEnabled: settings.bassEnabled,
    drumMode: settings.drumMode,
    drumPatternId: settings.drumPatternId,
    masterVolume: settings.masterVolume,
    harmonyVolume: settings.harmonyVolume,
    bassVolume: settings.bassVolume,
    drumVolume: settings.drumVolume,
  };
}

/** Snapshot the editor's actual voicings; later edits cannot change an in-flight export. */
export function buildExportSpec(song: Song, settings: PlaySettings): ExportSpec {
  const sections = song.sections ?? [];
  assertExportable(song, sections);
  const playSettings = sanitizePlaySettings(settings);
  const voicingMode = resolveVoicingMode(playSettings, song.genre);
  return {
    title: song.title.trim() || 'Untitled',
    bpm: song.bpm,
    key: song.key,
    timeSignature: [...song.timeSignature],
    genre: song.genre,
    measures: song.measures.map((measure) => ({
      chords: [exportChord(measure.chords[0], voicingMode), exportChord(measure.chords[1], voicingMode)],
    })),
    sections: [...sections]
      .sort((a, b) => a.startMeasure - b.startMeasure)
      .map((section) => ({ label: sectionDisplayLabel(section), startMeasure: section.startMeasure, length: section.length })),
    play: exportPlaySettings(playSettings),
  };
}

/** Not an ASCII control character (U+0000–U+001F, U+007F). */
function isNotAsciiControl(char: string): boolean {
  const code = char.codePointAt(0)!;
  return code >= 32 && code !== 127;
}

export function safeExportFilename(title: string, extension: 'mid' | 'pdf' | 'js'): string {
  // Preserve readable Unicode titles while excluding path separators and OS-reserved names.
  const cleaned = Array.from(title.normalize('NFKC'))
    .filter(isNotAsciiControl)
    .join('')
    .replace(/[\\/:*?"<>|]/g, '-')
    .replace(/\s+/g, ' ')
    .replace(/^[. ]+|[. ]+$/g, '');
  // Cut by code point so no surrogate pair is split, then trim the new end.
  let stem = Array.from(cleaned).slice(0, MAX_FILENAME_STEM_CHARS).join('').replace(/[. ]+$/g, '') || 'Untitled';
  if (WINDOWS_RESERVED_NAME.test(stem)) stem = `_${stem}`;
  // The Scripter script lands among a Downloads folder's other .js files; the marker says whose it is.
  return extension === 'js' ? `${stem}.theartist.js` : `${stem}.${extension}`;
}

/** 127.x.y.z with every part at most 255. */
function isIpv4Loopback(host: string): boolean {
  return /^127(?:\.(?:\d{1,3})){3}$/.test(host) && host.split('.').every((part) => Number(part) <= 255);
}

export function isLoopbackHostname(hostname: string): boolean {
  const host = hostname.toLowerCase();
  return LOOPBACK_NAMES.has(host) || isIpv4Loopback(host);
}

/** The message in a FastAPI error body: `detail` itself, or the joined
 *  `msg`s of a validation-error list; null when there is none. */
function errorDetail(body: unknown): string | null {
  if (typeof body !== 'object' || body === null || !('detail' in body)) return null;
  const detail = body.detail;
  if (typeof detail === 'string') return detail;
  if (!Array.isArray(detail)) return null;
  const reasons = detail.flatMap((item) => typeof item === 'object' && item !== null &&
    'msg' in item && typeof item.msg === 'string' ? [item.msg] : []);
  return reasons.length ? reasons.join('; ') : null;
}

export async function exportResponseError(response: Response): Promise<Error> {
  let message = `Export failed (${response.status}).`;
  try {
    message = errorDetail(await response.json()) ?? message;
  } catch { /* A non-JSON server error still gets a visible status. */ }
  return new Error(message);
}

function isMidiFileHeader(bytes: Uint8Array): boolean {
  return bytes.length === MIDI_FILE_HEADER.length && String.fromCharCode(...bytes) === MIDI_FILE_HEADER;
}

/** Whether a Content-Type header names a JavaScript file, its parameters aside. */
function isJavaScriptContentType(contentType: string | null): boolean {
  return JAVASCRIPT_MEDIA_TYPES.has(contentType?.split(';', 1)[0].trim().toLowerCase() ?? '');
}

/** POST the spec to an export route and hand the response to `readFile`. One
 *  timer bounds the exchange body included, so a stalled download fails like a
 *  stalled render; `signal` cancels either. */
async function requestExportFile(path: string, spec: ExportSpec, signal: AbortSignal | undefined,
  timeoutMessage: string, readFile: (response: Response) => Promise<Blob>): Promise<Blob> {
  const controller = new AbortController();
  const cancel = () => controller.abort(signal?.reason);
  if (signal?.aborted) cancel();
  else signal?.addEventListener('abort', cancel, { once: true });
  let timedOut = false;
  const timer = setTimeout(() => { timedOut = true; controller.abort(); }, EXPORT_TIMEOUT_MS);
  try {
    const response = await fetch(path, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(spec), signal: controller.signal,
    });
    if (!response.ok) throw await exportResponseError(response);
    return await readFile(response);
  } catch (error) {
    if (timedOut) throw new Error(timeoutMessage, { cause: error });
    throw error;
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener('abort', cancel);
  }
}

async function readMidiFile(response: Response): Promise<Blob> {
  const blob = await response.blob();
  const header = new Uint8Array(await blob.slice(0, MIDI_FILE_HEADER.length).arrayBuffer());
  if (!isMidiFileHeader(header)) throw new Error('The server did not return a MIDI file.');
  return blob;
}

export function requestMidiExport(spec: ExportSpec, signal?: AbortSignal): Promise<Blob> {
  return requestExportFile('/api/export/midi', spec, signal, 'MIDI export timed out. Please try again.', readMidiFile);
}

/** A script has no magic bytes, so the declared type is the check: a proxy's
 *  HTML error page or a JSON body must not be saved as a .js file. */
async function readScripterFile(response: Response): Promise<Blob> {
  if (!isJavaScriptContentType(response.headers.get('Content-Type'))) {
    throw new Error('The server did not return a Scripter script.');
  }
  const blob = await response.blob();
  if (!blob.size) throw new Error('The server did not return a Scripter script.');
  return blob;
}

/** The Logic Pro Scripter script of the song, rendered from the same spec as the MIDI file. */
export function requestScripterExport(spec: ExportSpec, signal?: AbortSignal): Promise<Blob> {
  return requestExportFile('/api/export/scripter', spec, signal, 'Scripter export timed out. Please try again.', readScripterFile);
}

export function downloadExportBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  document.body.append(anchor);
  try { anchor.click(); } finally {
    anchor.remove();
    // Safari needs the URL to remain alive until it starts the download.
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
}
