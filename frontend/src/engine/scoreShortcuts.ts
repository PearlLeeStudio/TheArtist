import type { SlotAddress } from '../models/types';

type SelectAllKey = Pick<KeyboardEvent,
  'key' | 'ctrlKey' | 'metaKey' | 'altKey' | 'shiftKey' | 'isComposing' | 'defaultPrevented' | 'preventDefault'>;

/** The editor owns select-all only outside text controls and open dialogs. */
export function selectAllScore(
  event: SelectAllKey,
  measureCount: number,
  blocked: boolean,
  selectRange: (from: SlotAddress, to: SlotAddress) => void,
): boolean {
  if (blocked || measureCount < 1 || event.defaultPrevented || event.isComposing ||
      event.altKey || event.shiftKey || (!event.ctrlKey && !event.metaKey) || event.key.toLowerCase() !== 'a') return false;
  event.preventDefault();
  selectRange({ measureIndex: 0, slotIndex: 0 }, { measureIndex: measureCount - 1, slotIndex: 1 });
  return true;
}
