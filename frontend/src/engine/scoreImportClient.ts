import { readSSEFrames } from './sseStream.ts';
import { readScoreImportResponse, scoreImportRequestBody } from './scoreImport.ts';
import type { ScoreImportResponse, ScoreMime } from './scoreImport.ts';
import type { ScoreImportMode } from '../models/types';

/**
 * One score analysis request over the streaming import endpoint, for the
 * Import dialog, which opens the result.
 */

/** Bytes per String.fromCharCode call, so the spread stays within argument limits. */
const BASE64_CHUNK_BYTES = 0x8000;

export class ScoreAnalysisError extends Error {
  /** The HTTP status the server answered, 0 when the stream ended without a result. */
  readonly status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = 'ScoreAnalysisError';
    this.status = status;
  }
}

export async function fileBase64(file: Blob): Promise<string> {
  const bytes = new Uint8Array(await file.arrayBuffer());
  const chunks: string[] = [];
  for (let start = 0; start < bytes.length; start += BASE64_CHUNK_BYTES) {
    chunks.push(String.fromCharCode(...bytes.subarray(start, start + BASE64_CHUNK_BYTES)));
  }
  return btoa(chunks.join(''));
}

/** The API's `detail` (a message or a list of validation errors), else a generic failure. */
export function recognitionErrorMessage(body: unknown, status: number): string {
  if (body && typeof body === 'object' && 'detail' in body) {
    const detail = (body as { detail: unknown }).detail;
    if (typeof detail === 'string') return detail;
    if (Array.isArray(detail)) {
      const items: unknown[] = detail;
      const messages = items.flatMap((item) =>
        item && typeof item === 'object' && 'msg' in item && typeof (item as { msg: unknown }).msg === 'string'
          ? [(item as { msg: string }).msg] : []);
      if (messages.length) return messages.join('; ');
    }
  }
  return `Score recognition failed (${status}). Please try again.`;
}

export interface AnalyzeScoreOptions {
  signal?: AbortSignal;
  /** Reader stages as the server reports them. */
  onStatus?: (label: string) => void;
  fetcher?: typeof fetch;
}

/** Send the file once and return the validated reading; server failures carry their HTTP status. */
export async function analyzeScoreFile(file: File, mimeType: ScoreMime, mode: ScoreImportMode,
  options: AnalyzeScoreOptions = {}): Promise<ScoreImportResponse> {
  const { signal, onStatus } = options;
  const fetcher = options.fetcher ?? fetch;
  const dataBase64 = await fileBase64(file);
  if (signal?.aborted) throw new DOMException('Recognition was interrupted.', 'AbortError');
  onStatus?.('Uploading the score for analysis…');
  const response = await fetcher('/api/import/score/stream', {
    method: 'POST',
    cache: 'no-store',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(scoreImportRequestBody(file.name, mimeType, dataBase64, mode)),
    signal,
  });
  if (!response.ok) {
    throw new ScoreAnalysisError(recognitionErrorMessage(await response.json().catch(() => null), response.status), response.status);
  }
  let body: unknown;
  for await (const frame of readSSEFrames<Record<string, unknown>>(response)) {
    if (frame.event === 'progress' && typeof frame.data.label === 'string') onStatus?.(frame.data.label);
    if (frame.event === 'error') {
      const status = Number(frame.data.status) || 500;
      throw new ScoreAnalysisError(recognitionErrorMessage(frame.data, status), status);
    }
    if (frame.event === 'result') { body = frame.data; break; }
  }
  if (!body) throw new ScoreAnalysisError('The score connection ended before analysis finished. Please try again.', 0);
  return readScoreImportResponse(body, mode);
}
