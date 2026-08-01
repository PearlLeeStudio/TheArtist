/**
 * Shared by the Instrument, Rhythm and Drums panes: the type of their
 * worded keys, and keyboard movement through a group of choice keys.
 */
import type { CSSProperties, KeyboardEvent } from 'react';

/** Type of a pane key that carries a word or two: a part toggle, Auto. */
export const KEY_TEXT_STYLE: CSSProperties = { fontSize: 12, lineHeight: '14px', letterSpacing: 'normal' };

/** A full-width key as tall as an sm key (26px): the Rhythm and Drums
 *  part toggles and their Auto keys. */
export const WIDE_KEY_STYLE: CSSProperties = { height: 26, minHeight: 26, ...KEY_TEXT_STYLE };

/** The picked key of a narrow grid reads heavier at the same width: bold would widen
 *  its label and wrap it past the key's height, a hairline stroke does not. */
export function pickedLabelStyle(picked: boolean): CSSProperties {
  return { fontWeight: 400, WebkitTextStroke: picked ? '0.45px currentColor' : undefined };
}

/** Where an arrow, Home or End key moves from choice `index` of `count`
 *  (the arrows wrap around); null for any other key. */
function nextChoiceIndex(key: string, index: number, count: number): number | null {
  switch (key) {
    case 'ArrowRight':
    case 'ArrowDown':
      return (index + 1) % count;
    case 'ArrowLeft':
    case 'ArrowUp':
      return (index + count - 1) % count;
    case 'Home':
      return 0;
    case 'End':
      return count - 1;
    default:
      return null;
  }
}

/**
 * Roving choice for a group's onKeyDown: from a focused key matching
 * `selector`, the arrows, Home and End click and focus another one.
 */
export function moveChoiceByKey(event: KeyboardEvent<HTMLElement>, selector: string): void {
  const choices = Array.from(event.currentTarget.querySelectorAll<HTMLButtonElement>(selector));
  const index = choices.indexOf(event.target as HTMLButtonElement);
  if (index < 0) return;
  const next = nextChoiceIndex(event.key, index, choices.length);
  if (next === null) return;
  event.preventDefault();
  choices[next]?.click();
  choices[next]?.focus();
}
