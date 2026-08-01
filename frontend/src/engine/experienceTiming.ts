/** Opt-in, in-memory timings. No score content, file names, dates or network telemetry. */
export type TimingName = 'import.review' | 'generate.result' | 'section.result' | 'form.result'
  | 'instrument.ready' | 'play.transport';
export type TimingOutcome = 'ok' | 'error' | 'cancelled' | 'timeout';
export interface TimingSample {
  name: TimingName;
  condition: string;
  durationMs: number;
  outcome: TimingOutcome;
}
const samples: TimingSample[] = [];
let enabled = false;
export function setTimingEnabled(value: boolean): void { enabled = value; }
export function clearTimings(): void { samples.length = 0; }
export function timingSamples(): TimingSample[] { return samples.map((sample) => ({ ...sample })); }

export function beginTiming(name: TimingName, condition = 'default'): (outcome?: TimingOutcome) => void {
  if (!enabled) return () => {};
  const start = performance.now();
  let finished = false;
  return (outcome = 'ok') => {
    if (finished) return;
    finished = true;
    samples.push({ name, condition, durationMs: Math.max(0, performance.now() - start), outcome });
    if (samples.length > 200) samples.shift();
  };
}

export async function timeOperation<T>(name: TimingName, condition: string, operation: () => Promise<T>): Promise<T> {
  const finish = beginTiming(name, condition);
  try {
    const value = await operation();
    finish();
    return value;
  } catch (error) {
    finish(error instanceof Error && error.name === 'AbortError' ? 'cancelled' : 'error');
    throw error;
  }
}

export function timingSummary() {
  const groups = new Map<string, TimingSample[]>();
  for (const sample of samples) {
    const key = `${sample.name}/${sample.condition}`;
    groups.set(key, [...(groups.get(key) ?? []), sample]);
  }
  return [...groups].map(([name, group]) => {
    const values = group.filter((sample) => sample.outcome === 'ok').map((sample) => sample.durationMs).sort((a, b) => a - b);
    const percentile = (p: number) => values.length ? values[Math.ceil(p * values.length) - 1] : null;
    return { name, count: values.length, failures: group.length - values.length, p50Ms: percentile(0.5), p95Ms: percentile(0.95) };
  });
}

declare global {
  interface Window {
    theArtistPerformance?: { samples: typeof timingSamples; summary: typeof timingSummary; clear: typeof clearTimings };
  }
}

export function installTimingConsole(): void {
  if (new URLSearchParams(window.location.search).get('measurePerformance') !== '1') return;
  setTimingEnabled(true);
  window.theArtistPerformance = { samples: timingSamples, summary: timingSummary, clear: clearTimings };
}
