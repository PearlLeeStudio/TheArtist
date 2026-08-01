/**
 * Tiny Server-Sent Events parser for `fetch`-based POST streams.
 *
 * The browser's built-in `EventSource` only supports GET, but the
 * backend's `/api/generate/stream` is POST (the request body carries
 * the song + selectedMeasures + modelKey). So we read the response
 * body manually and walk the wire format ourselves:
 *
 *     event: <name>\n
 *     data: <json>\n
 *     \n
 *
 * Yielded as `{ event, data }` parsed objects, one per frame.
 *
 * Generic so callers can narrow the data type at the call site.
 */
export interface SSEFrame<T = unknown> {
  event: string;
  data: T;
}

/** Async generator over the SSE frames in a `Response` body. */
export async function* readSSEFrames<T = unknown>(
  response: Response,
): AsyncGenerator<SSEFrame<T>> {
  if (!response.body) {
    throw new Error('SSE response has no body');
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // Each frame ends with a blank line (\n\n).
    let idx: number;
    while ((idx = buffer.indexOf('\n\n')) !== -1) {
      const block = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      let event = '';
      let dataLine = '';
      for (const line of block.split('\n')) {
        if (line.startsWith('event: ')) event = line.slice(7);
        else if (line.startsWith('data: ')) dataLine = line.slice(6);
      }
      if (!event || !dataLine) continue;
      yield { event, data: JSON.parse(dataLine) as T };
    }
  }
}
