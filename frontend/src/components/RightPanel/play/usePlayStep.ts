/**
 * The step of the bar playing now — a 16th by default — for the Play
 * panel's LED playheads. Polled once per animation frame while the editor
 * transport plays; null while stopped or paused.
 */
import { useEffect, useState } from 'react';
import { useSongStore } from '../../../store/songStore';
import { getPlaybackStep } from '../../../engine/playback';

export function usePlayStep(stepsPerBeat = 4): number | null {
  const isPlaying = useSongStore((s) => s.isPlaying);
  const [step, setStep] = useState<number | null>(null);

  useEffect(() => {
    if (!isPlaying) return;
    let frame = 0;
    // -1 is never a step, so the first frame always publishes.
    let last: number | null = -1;
    const tick = () => {
      const next = getPlaybackStep(stepsPerBeat);
      if (next !== last) {
        last = next;
        setStep(next);
      }
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(frame);
      // The next run starts with no playhead rather than this run's last step.
      setStep(null);
    };
  }, [isPlaying, stepsPerBeat]);

  return isPlaying ? step : null;
}
