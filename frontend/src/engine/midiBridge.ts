/** A loopback helper page owns the native connection; HTTPS never fetches HTTP.
 * Messages are accepted only from the opened window, fixed origin and fresh nonce.
 */
export const MIDI_BRIDGE_ORIGIN = 'http://127.0.0.1:8768';

export function bridgeMessage(event: Pick<MessageEvent, 'origin' | 'source' | 'data'>,
  popup: Window, token: string): { type: 'state'; connected: boolean; devices: string[] } | { type: 'midi'; data: number[]; port: string } | null {
  if (event.origin !== MIDI_BRIDGE_ORIGIN || event.source !== popup) return null;
  const message = event.data;
  if (!message || message.token !== token || message.protocol !== 'theartist-midi-v1') return null;
  if (message.type === 'state' && typeof message.connected === 'boolean') {
    const devices = Array.isArray(message.devices)
      ? message.devices.filter((name: unknown): name is string => typeof name === 'string' && name.length <= 160).slice(0, 64)
      : [];
    return { type: 'state', connected: message.connected, devices };
  }
  if (message.type === 'midi' && typeof message.port === 'string' && message.port.length <= 80 &&
    Array.isArray(message.data) && message.data.length === 3 &&
    message.data.every((n: unknown) => Number.isInteger(n) && (n as number) >= 0 && (n as number) <= 255) &&
    [0x80, 0x90, 0xb0].includes(message.data[0] & 0xf0) && message.data[1] < 128 && message.data[2] < 128) {
    return { type: 'midi', data: message.data, port: message.port };
  }
  return null;
}

export function openMidiBridge(onBytes: (data: number[], port: string) => void,
  onConnection: (connected: boolean, devices: string[]) => void): { ready: Promise<boolean>; close: () => void } {
  const token = crypto.randomUUID();
  const url = `${MIDI_BRIDGE_ORIGIN}/bridge?origin=${encodeURIComponent(window.location.origin)}&token=${token}`;
  const popup = window.open(url, 'theartist-midi-bridge', 'popup,width=420,height=300');
  if (!popup) throw new Error('Allow the MIDI connection window in this browser, then enable MIDI again.');
  let finished = false;
  let resolveReady: (connected: boolean) => void;
  let rejectReady: (error: Error) => void;
  const ready = new Promise<boolean>((resolve, reject) => { resolveReady = resolve; rejectReady = reject; });
  const message = (event: MessageEvent) => {
    const value = bridgeMessage(event, popup, token);
    if (!value) return;
    if (value.type === 'state') {
      onConnection(value.connected, value.devices);
      if (!finished) { finished = true; clearTimeout(timeout); resolveReady(value.connected); }
    } else onBytes(value.data, value.port);
  };
  const close = () => {
    clearTimeout(timeout);
    clearInterval(closed);
    window.removeEventListener('message', message);
    popup.close();
    if (!finished) { finished = true; resolveReady(false); }
  };
  const timeout = setTimeout(() => {
    if (!finished) {
      finished = true;
      rejectReady(new Error('Start TheArtist MIDI Bridge on the Mac connected to your keyboard, then enable MIDI again.'));
      close();
    }
  }, 8000);
  const closed = setInterval(() => {
    if (popup.closed) { onConnection(false, []); close(); }
  }, 500);
  window.addEventListener('message', message);
  return { ready, close };
}
