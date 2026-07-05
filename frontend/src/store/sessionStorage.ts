/**
 * Storage adapter for saved sessions. The whole MyPage feature talks to
 * this module, never to localStorage directly — so when v2 wants
 * IndexedDB / Supabase, only this file changes.
 *
 * Schema versioning: a top-level `{ version, sessions }` envelope so a
 * future migration can detect the old shape and upgrade in place. The
 * same envelope shape is used for the JSON file format produced by
 * Export / consumed by Import — no separate schema there.
 */
import type { SavedSession } from '../models/types';

const STORAGE_KEY = 'theartist-sessions';
export const SCHEMA_VERSION = 1;

export interface Envelope {
  version: number;
  sessions: SavedSession[];
  exportedAt?: number;  // present only on file exports, ignored when reading from localStorage
}

export function loadAll(): SavedSession[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as Envelope;
    if (!parsed || parsed.version !== SCHEMA_VERSION || !Array.isArray(parsed.sessions)) {
      return [];
    }
    return parsed.sessions;
  } catch {
    return [];
  }
}

export function saveAll(sessions: SavedSession[]): void {
  const envelope: Envelope = { version: SCHEMA_VERSION, sessions };
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(envelope));
  } catch (err) {
    // localStorage quota or disabled storage — fail loud in console but
    // don't crash the app. User keeps their in-memory state.
    console.error('[sessionStorage] saveAll failed', err);
  }
}

/** Validate a parsed object as a SavedSession. Reject silently rather
 *  than throwing — bad entries from a foreign export are skipped, not
 *  fatal. */
function isValidSession(x: unknown): x is SavedSession {
  if (typeof x !== 'object' || x === null) return false;
  const s = x as Record<string, unknown>;
  return (
    typeof s.id === 'string' &&
    typeof s.name === 'string' &&
    typeof s.createdAt === 'number' &&
    typeof s.starred === 'boolean' &&
    typeof s.song === 'object' && s.song !== null &&
    typeof s.generation === 'object' && s.generation !== null
  );
}

/** Serialize all sessions as a JSON string for file download. Wraps the
 *  envelope with `exportedAt` so the receiving side can show provenance. */
export function serializeForExport(sessions: SavedSession[]): string {
  const envelope: Envelope = {
    version: SCHEMA_VERSION,
    sessions,
    exportedAt: Date.now(),
  };
  return JSON.stringify(envelope, null, 2);
}

export interface ImportResult {
  added: number;       // freshly imported (no id collision)
  renamed: number;     // imported with a regenerated id (existing id collided)
  skipped: number;     // failed validation
}

/** Parse a JSON file's text content and merge its sessions into the
 *  existing list. Collisions on `id` get a fresh UUID so both copies
 *  survive — users can prune duplicates manually. Returns counts so
 *  the UI can show a confirmation. Throws on parse / version errors. */
export function mergeImport(
  existing: SavedSession[],
  jsonText: string,
): { merged: SavedSession[]; result: ImportResult } {
  const parsed = JSON.parse(jsonText) as Envelope;
  if (!parsed || parsed.version !== SCHEMA_VERSION || !Array.isArray(parsed.sessions)) {
    throw new Error(`Unsupported file format (expected version ${SCHEMA_VERSION})`);
  }

  const existingIds = new Set(existing.map((s) => s.id));
  const result: ImportResult = { added: 0, renamed: 0, skipped: 0 };
  const incoming: SavedSession[] = [];

  for (const raw of parsed.sessions) {
    if (!isValidSession(raw)) {
      result.skipped += 1;
      continue;
    }
    if (existingIds.has(raw.id)) {
      incoming.push({ ...raw, id: crypto.randomUUID() });
      result.renamed += 1;
    } else {
      incoming.push(raw);
      result.added += 1;
    }
  }

  // Newest-first: imports go on top, but preserve their internal order.
  return { merged: [...incoming, ...existing], result };
}
