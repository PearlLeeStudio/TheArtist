import type { Chord, Measure, SlotAddress } from '../models/types';
import type { TargetChord } from './generationComparison';
import { chordToString, parseChordName } from './chordParser.ts';

/**
 * Cut / copy / paste over the score's chord slots.
 *
 * A copied passage travels as a structured payload (chords with their
 * voicings, two slots per bar) under CHORD_CLIPBOARD_TYPE and as chord
 * text under text/plain, e.g. "Dm7 G7 | Cmaj7 | -". Text pasted from
 * elsewhere is parsed with the same bar-line convention; one symbol in a
 * bar fills both of its slots, as it does on import.
 *
 * Paste lands on the selection: a clipboard that divides the selected
 * slots exactly is tiled over them (copy two bars, select eight, paste);
 * anything else is written forward from the first selected slot, adding
 * bars at the end of the score as needed. Whole bars stay whole: a copy
 * that began on a bar line and holds complete bars is pasted from the
 * start of the first selected bar, whichever half of it was clicked, so
 * it never straddles a bar line or spills a half-bar onto the next row.
 */

export const CHORD_CLIPBOARD_TYPE = 'application/x-theartist-chords';
const CLIPBOARD_FORMAT = 'theartist-chords';
const SLOTS_PER_BAR = 2;
/** Text for an empty slot, and the tokens read back as one. */
const EMPTY_SLOT_TEXT = '-';
const EMPTY_TOKENS = new Set(['-', '–', '—', '.', '_', 'nc', 'n.c', 'n.c.', 'x']);
/** Chart shorthand for "repeat the previous bar". */
const REPEAT_BAR_TOKEN = '%';

export interface ChordClipboard {
  format: typeof CLIPBOARD_FORMAT;
  version: 1;
  /** Copied slots in score order; null is an empty slot. */
  slots: (Chord | null)[];
  /** Slot index of the first copied slot, so the text keeps its bar lines. */
  firstSlotIndex: 0 | 1;
}

export interface PastePlan {
  targets: TargetChord[];
  /** Bars to append before writing, when the paste runs past the score. */
  measuresToAdd: number;
  /** The pasted range, selected afterwards. */
  select: [SlotAddress, SlotAddress];
}

export interface PasteOptions {
  /** The clipboard holds complete bars from a bar line: paste bar-aligned. */
  wholeBars?: boolean;
}

/** Whether a copied passage is complete bars starting on a bar line. */
export function holdsWholeBars(clip: Pick<ChordClipboard, 'slots' | 'firstSlotIndex'>): boolean {
  return clip.firstSlotIndex === 0 && clip.slots.length % SLOTS_PER_BAR === 0;
}

function slotToLinear(slot: SlotAddress): number {
  return slot.measureIndex * SLOTS_PER_BAR + slot.slotIndex;
}

function linearToSlot(linear: number): SlotAddress {
  return { measureIndex: Math.floor(linear / SLOTS_PER_BAR), slotIndex: (linear % SLOTS_PER_BAR) as 0 | 1 };
}

/** The selection in score order, each slot once. */
export function orderedSlots(selection: SlotAddress[]): SlotAddress[] {
  const linear = [...new Set(selection.map(slotToLinear))].sort((a, b) => a - b);
  return linear.map(linearToSlot);
}

function cloneChord(chord: Chord): Chord {
  return { ...chord, voicing: [...chord.voicing], extensions: [...chord.extensions] };
}

/** The selected slots' chords, or null when nothing is selected. */
export function copySelection(measures: Measure[], selection: SlotAddress[]): ChordClipboard | null {
  const slots = orderedSlots(selection).filter((slot) => slot.measureIndex < measures.length);
  if (slots.length === 0) return null;
  return {
    format: CLIPBOARD_FORMAT,
    version: 1,
    slots: slots.map((slot) => {
      const chord = measures[slot.measureIndex].chords[slot.slotIndex];
      return chord ? cloneChord(chord) : null;
    }),
    firstSlotIndex: slots[0].slotIndex,
  };
}

/** Whether any selected slot holds a chord (what Cut and Clear would remove). */
export function selectionHasChords(measures: Measure[], selection: SlotAddress[]): boolean {
  return selection.some((slot) => measures[slot.measureIndex]?.chords[slot.slotIndex]);
}

function slotText(chord: Chord | null): string {
  return chord ? chordToString(chord) : EMPTY_SLOT_TEXT;
}

/** Chord text with bar lines: "Dm7 G7 | Cmaj7 | -". A bar whose two slots
 *  hold the same symbol is written once. */
export function clipboardText(clip: ChordClipboard): string {
  const bars: string[] = [];
  let bar: (Chord | null)[] = [];
  let slotIndex: number = clip.firstSlotIndex;
  const flush = () => {
    if (bar.length === 0) return;
    const texts = bar.map(slotText);
    bars.push(texts.length === SLOTS_PER_BAR && texts[0] === texts[1] ? texts[0] : texts.join(' '));
    bar = [];
  };
  for (const chord of clip.slots) {
    bar.push(chord);
    slotIndex++;
    if (slotIndex === SLOTS_PER_BAR) { flush(); slotIndex = 0; }
  }
  flush();
  return bars.join(' | ');
}

export function serializeClipboard(clip: ChordClipboard): string {
  return JSON.stringify(clip);
}

function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((item) => typeof item === 'string');
}

/** A chord from a payload slot, or undefined when the slot is not one. */
function readChord(value: unknown): Chord | null | undefined {
  if (value === null) return null;
  if (!value || typeof value !== 'object') return undefined;
  const chord = value as Record<string, unknown>;
  if (typeof chord.root !== 'string' || typeof chord.quality !== 'string') return undefined;
  if (chord.bass !== undefined && typeof chord.bass !== 'string') return undefined;
  if (!parseChordName(chord.root + (chord.quality === 'maj' ? '' : chord.quality))) return undefined;
  const voicing = Array.isArray(chord.voicing) ? chord.voicing.filter((note): note is number => Number.isInteger(note)) : [];
  return {
    root: chord.root,
    quality: chord.quality,
    ...(chord.bass ? { bass: chord.bass } : {}),
    romanNumeral: '',
    voicing,
    extensions: isStringArray(chord.extensions) ? [...chord.extensions] : [],
  };
}

/** The structured payload, or null when the data is not one of ours. */
export function readClipboardPayload(json: string): ChordClipboard | null {
  let parsed: unknown;
  try {
    parsed = JSON.parse(json);
  } catch {
    return null;
  }
  if (!parsed || typeof parsed !== 'object') return null;
  const payload = parsed as Record<string, unknown>;
  if (payload.format !== CLIPBOARD_FORMAT || payload.version !== 1 || !Array.isArray(payload.slots)) return null;
  const slots: (Chord | null)[] = [];
  for (const item of payload.slots) {
    const chord = readChord(item);
    if (chord === undefined) return null;
    slots.push(chord);
  }
  if (slots.length === 0) return null;
  return { format: CLIPBOARD_FORMAT, version: 1, slots, firstSlotIndex: payload.firstSlotIndex === 1 ? 1 : 0 };
}

/** One text token as a slot: a chord, or null for an explicit empty slot;
 *  undefined when the token is not chord text. */
function tokenChord(token: string): Chord | null | undefined {
  if (EMPTY_TOKENS.has(token.toLowerCase())) return null;
  return parseChordName(token) ?? undefined;
}

/** Chord text as slots, or null when the text holds no chord symbol at all.
 *  Bar lines ("|" or line breaks) make bars: one symbol fills both slots, two
 *  take one each, a longer list keeps its first and middle symbols, "%"
 *  repeats the previous bar. Without bar lines, each symbol is one slot. */
export function parseClipboardText(text: string): (Chord | null)[] | null {
  const barred = /[|\n\r]/.test(text);
  const bars = barred ? text.split(/[|\n\r]+/) : [text];
  const slots: (Chord | null)[] = [];
  let chords = 0;
  let previousBar: (Chord | null)[] | null = null;
  for (const barText of bars) {
    const tokens = barText.split(/[\s,]+/).filter(Boolean);
    if (tokens.length === 0) continue;
    if (barred && tokens.length === 1 && tokens[0] === REPEAT_BAR_TOKEN) {
      if (previousBar) slots.push(...previousBar.map((chord) => chord ? cloneChord(chord) : null));
      continue;
    }
    const read = tokens.map(tokenChord).filter((chord): chord is Chord | null => chord !== undefined);
    if (read.length === 0) continue;
    chords += read.filter(Boolean).length;
    let bar: (Chord | null)[];
    if (!barred) bar = read;
    else if (read.length === 1) bar = [read[0], read[0] ? cloneChord(read[0]) : null];
    else if (read.length === SLOTS_PER_BAR) bar = read;
    else bar = [read[0], read[Math.floor(read.length / 2)]];
    slots.push(...bar);
    previousBar = bar;
  }
  return chords > 0 ? slots : null;
}

/** Where a paste of `slots` lands for this selection, or null when there is
 *  nothing to paste or no selection to anchor it. */
export function planPaste(selection: SlotAddress[], slots: (Chord | null)[], measureCount: number,
  options: PasteOptions = {}): PastePlan | null {
  let selected = orderedSlots(selection).filter((slot) => slot.measureIndex < measureCount);
  if (slots.length === 0 || selected.length === 0) return null;
  if (options.wholeBars) {
    // Whole bars land on whole bars: widen the selection to the bars it touches.
    const first = selected[0].measureIndex;
    const last = selected[selected.length - 1].measureIndex;
    selected = [];
    for (let measureIndex = first; measureIndex <= last; measureIndex++) {
      selected.push({ measureIndex, slotIndex: 0 }, { measureIndex, slotIndex: 1 });
    }
  }
  const clone = (chord: Chord | null) => chord ? cloneChord(chord) : null;
  if (selected.length > 1 && selected.length % slots.length === 0) {
    return {
      targets: selected.map((address, i) => ({ address, chord: clone(slots[i % slots.length]) })),
      measuresToAdd: 0,
      select: [selected[0], selected[selected.length - 1]],
    };
  }
  const start = slotToLinear(selected[0]);
  const targets = slots.map((chord, i) => ({ address: linearToSlot(start + i), chord: clone(chord) }));
  const lastMeasure = targets[targets.length - 1].address.measureIndex;
  return {
    targets,
    measuresToAdd: Math.max(0, lastMeasure + 1 - measureCount),
    select: [targets[0].address, targets[targets.length - 1].address],
  };
}

export type ScoreEditKey = 'copy' | 'cut' | 'paste' | 'clear' | 'undo';

type EditKeyEvent = Pick<KeyboardEvent, 'key' | 'ctrlKey' | 'metaKey' | 'altKey' | 'shiftKey' | 'isComposing' | 'defaultPrevented'>;

/** The score edit a keystroke asks for, or null. `blocked` is true inside
 *  text controls and open dialogs, which keep their native keys. */
export function scoreEditKey(event: EditKeyEvent, blocked: boolean): ScoreEditKey | null {
  if (blocked || event.defaultPrevented || event.isComposing || event.altKey) return null;
  const modifier = event.ctrlKey || event.metaKey;
  if (!modifier) {
    if (event.shiftKey) return null;
    return event.key === 'Delete' || event.key === 'Backspace' ? 'clear' : null;
  }
  if (event.shiftKey || (event.ctrlKey && event.metaKey)) return null;
  switch (event.key.toLowerCase()) {
    case 'c': return 'copy';
    case 'x': return 'cut';
    case 'v': return 'paste';
    case 'z': return 'undo';
    default: return null;
  }
}
