import type { Chord, Measure, SlotAddress } from '../models/types';

/**
 * Slot-position arithmetic. Each measure has exactly two chord slots
 * (slotIndex 0 / 1), so "next" and "prev" only need to flip the slot
 * index and step the measure when crossing a bar line.
 *
 * Returns null when stepping would go past either edge of the song.
 */

export function nextSlotAfter(
  slot: SlotAddress,
  totalMeasures: number,
): SlotAddress | null {
  const nextSlotIndex: 0 | 1 = slot.slotIndex === 0 ? 1 : 0;
  const nextMeasureIndex =
    slot.slotIndex === 0 ? slot.measureIndex : slot.measureIndex + 1;
  if (nextMeasureIndex >= totalMeasures) return null;
  return { measureIndex: nextMeasureIndex, slotIndex: nextSlotIndex };
}

export function prevSlotBefore(
  slot: SlotAddress,
): SlotAddress | null {
  if (slot.slotIndex === 1) {
    return { measureIndex: slot.measureIndex, slotIndex: 0 };
  }
  if (slot.measureIndex > 0) {
    return { measureIndex: slot.measureIndex - 1, slotIndex: 1 };
  }
  return null;
}

/** Resolve the chord at the slot one step before / after `slot`, or null
 *  if there is none (edge of song, or empty slot at that position). */
export function chordAtPrevSlot(
  measures: Measure[],
  slot: SlotAddress | null,
): Chord | null {
  if (!slot) return null;
  const prev = prevSlotBefore(slot);
  if (!prev) return null;
  return measures[prev.measureIndex]?.chords[prev.slotIndex] ?? null;
}

export function chordAtNextSlot(
  measures: Measure[],
  slot: SlotAddress | null,
): Chord | null {
  if (!slot) return null;
  const next = nextSlotAfter(slot, measures.length);
  if (!next) return null;
  return measures[next.measureIndex]?.chords[next.slotIndex] ?? null;
}
