/** Shared transport for JSON POSTs; callers own their response schemas and streams. */

/** A backend detail can be a plain message or FastAPI's field-validation list. */
export function formatErrorDetail(detail: unknown): string | null {
  if (typeof detail === 'string') return detail.trim() || null;
  if (!Array.isArray(detail)) return null;
  const parts = detail.map((item) => {
    if (typeof item === 'string') return item;
    if (typeof item !== 'object' || item === null) return '';
    const { loc, msg } = item as { loc?: unknown; msg?: unknown };
    const field = Array.isArray(loc) ? loc.filter((p) => p !== 'body').join('.') : '';
    const text = typeof msg === 'string' ? msg : '';
    return field && text ? `${field}: ${text}` : text;
  }).filter(Boolean);
  return parts.length > 0 ? parts.join('; ') : null;
}

export async function describeHttpError(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: unknown; error?: unknown } | null;
    const detail = formatErrorDetail(body?.detail) ?? formatErrorDetail(body?.error);
    if (detail) return detail;
  } catch {
    // A non-JSON error still gets a visible HTTP status.
  }
  return response.status >= 500
    ? `Server error (HTTP ${response.status})`
    : `Request failed (HTTP ${response.status})`;
}

interface PostJsonOptions {
  signal?: AbortSignal;
  unreachableMessage?: string;
}

/** Preserve abort errors, report HTTP failures, and leave successful bodies unread. */
export async function postJson(url: string, body: unknown, options: PostJsonOptions = {}): Promise<Response> {
  let response: Response;
  try {
    response = await fetch(url, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body), signal: options.signal,
    });
  } catch (error) {
    if (options.signal?.aborted) throw error;
    throw new Error(options.unreachableMessage ??
      'Could not reach the server. Check the backend connection and try again.', { cause: error });
  }
  if (!response.ok) throw new Error(await describeHttpError(response));
  return response;
}
