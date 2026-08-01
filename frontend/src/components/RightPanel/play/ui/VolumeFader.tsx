import type { KeyboardEvent } from 'react';
import './VolumeFader.css';

interface VolumeFaderProps {
  label: string;
  value: number;
  defaultValue: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  ariaLabel?: string;
}

/** Snap `value` to the step grid that starts at `min`, then clamp it to [min, max]. */
const bounded = (value: number, min: number, max: number, step: number): number =>
  Math.min(max, Math.max(min, min + Math.round((value - min) / step) * step));

/** Native vertical range behavior with a quiet, recessed hardware face. */
export default function VolumeFader({ label, value, defaultValue, onChange,
  min = 0, max = 100, step = 1, ariaLabel }: VolumeFaderProps) {
  const snap = (raw: number) => bounded(raw, min, max, step);
  const current = snap(value);
  const percent = (current - min) / (max - min) * 100;

  const handleKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    let next: number;
    if (event.key === 'ArrowUp') next = current + step;
    else if (event.key === 'ArrowDown') next = current - step;
    else if (event.key === 'Home') next = min;
    else if (event.key === 'End') next = max;
    else return;
    event.preventDefault();
    onChange(snap(next));
  };

  return (
    <label className="te-volume-fader">
      <span className="te-volume-fader-travel">
        <input
          type="range"
          className="te-volume-fader-input"
          min={min}
          max={max}
          step={step}
          value={current}
          aria-label={ariaLabel ?? `${label} volume`}
          aria-orientation="vertical"
          aria-valuemin={min}
          aria-valuemax={max}
          aria-valuenow={current}
          onChange={(event) => onChange(snap(Number(event.currentTarget.value)))}
          onKeyDown={handleKeyDown}
          onDoubleClick={() => onChange(snap(defaultValue))}
        />
        <span className="te-volume-fader-groove" aria-hidden="true" />
        <span
          className="te-volume-fader-thumb"
          aria-hidden="true"
          style={{ bottom: `${percent}%`, transform: `translateY(${percent}%)` }}
        />
      </span>
      <span className="te-volume-fader-label">{label}</span>
    </label>
  );
}
