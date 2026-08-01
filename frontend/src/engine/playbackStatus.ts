export type PlaybackStatus = { loading: string | null; error: string | null };
let status: PlaybackStatus = { loading: null, error: null };
const listeners = new Set<() => void>();
export const getPlaybackStatus = () => status;
export function subscribePlaybackStatus(listener: () => void) {
  listeners.add(listener);
  return () => { listeners.delete(listener); };
}
export function setPlaybackStatus(next: PlaybackStatus) {
  status = next;
  listeners.forEach(listener => listener());
}
