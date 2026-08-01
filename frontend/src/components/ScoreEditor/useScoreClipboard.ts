import { useEffect, useSyncExternalStore } from 'react';
import { useSongStore } from '../../store/songStore';
import {
  CHORD_CLIPBOARD_TYPE,
  clipboardText,
  copySelection,
  holdsWholeBars,
  parseClipboardText,
  planPaste,
  readClipboardPayload,
  scoreEditKey,
  selectionHasChords,
  serializeClipboard,
} from '../../engine/scoreClipboard';
import type { ChordClipboard, ScoreEditKey } from '../../engine/scoreClipboard';

/** Keys and clipboard events the editor leaves to text controls and open dialogs. */
export function editorKeysBlocked(target: EventTarget | null): boolean {
  const element = target instanceof Element ? target : null;
  const textControl = element?.closest('input, textarea, select, [contenteditable]:not([contenteditable="false"]), [role="textbox"], [role="combobox"]');
  return Boolean(textControl || document.querySelector('dialog[open]'));
}

// The last copy, kept so the context menu can paste and so a browser that
// withholds the clipboard event from a keystroke still pastes within the editor.
let memory: ChordClipboard | null = null;
let memoryText = '';
const memoryListeners = new Set<() => void>();

function remember(clip: ChordClipboard): void {
  memory = clip;
  memoryText = clipboardText(clip);
  memoryListeners.forEach((listener) => listener());
}

function subscribeMemory(listener: () => void): () => void {
  memoryListeners.add(listener);
  return () => memoryListeners.delete(listener);
}

/** The chords the editor last copied or cut, for the context menu's Paste. */
export function useClipboardMemory(): ChordClipboard | null {
  const read = () => memory;
  // The third reader serves server rendering (the staff's render tests).
  return useSyncExternalStore(subscribeMemory, read, read);
}

/** How long a Cmd/Ctrl+C/X/V keystroke waits for its clipboard event
 *  before the editor handles it from memory. */
const CLIPBOARD_EVENT_GRACE_MS = 120;

/** The score edits behind the keys, the clipboard events and the context
 *  menu. Each returns whether it changed or copied anything. */
export const scoreEdits = {
  copy(event?: ClipboardEvent): boolean {
    const { song, selectedSlots } = useSongStore.getState();
    const clip = copySelection(song.measures, selectedSlots);
    if (!clip) return false;
    remember(clip);
    if (event?.clipboardData) {
      event.clipboardData.setData(CHORD_CLIPBOARD_TYPE, serializeClipboard(clip));
      event.clipboardData.setData('text/plain', memoryText);
      event.preventDefault();
    } else {
      // Best effort outside a clipboard event (context menu); the memory copy is what Paste uses.
      navigator.clipboard?.writeText(memoryText).catch(() => {});
    }
    return true;
  },

  cut(event?: ClipboardEvent): boolean {
    if (!scoreEdits.copy(event)) return false;
    const { selectedSlots, clearSlots } = useSongStore.getState();
    clearSlots(selectedSlots);
    return true;
  },

  paste(event?: ClipboardEvent): boolean {
    const { song, selectedSlots, pasteSlots } = useSongStore.getState();
    const data = event?.clipboardData;
    const payload = data?.getData(CHORD_CLIPBOARD_TYPE) || '';
    const text = data?.getData('text/plain') || '';
    let clip = payload ? readClipboardPayload(payload) : null;
    // The editor's own copy, with voicings, when the browser gives back its text or nothing.
    if (!clip && memory && (!text || text === memoryText)) clip = memory;
    let slots = clip?.slots ?? null;
    // Chord text with bar lines holds whole bars; a bare list of symbols is slot by slot.
    let wholeBars = clip ? holdsWholeBars(clip) : false;
    if (!slots && text) {
      slots = parseClipboardText(text);
      wholeBars = /[|\n\r]/.test(text) && (slots?.length ?? 1) % 2 === 0;
    }
    if (!slots) return false;
    const plan = planPaste(selectedSlots, slots, song.measures.length, { wholeBars });
    if (!plan) return false;
    event?.preventDefault();
    pasteSlots(plan);
    return true;
  },

  clear(): boolean {
    const { song, selectedSlots, clearSlots } = useSongStore.getState();
    if (!selectionHasChords(song.measures, selectedSlots)) return false;
    clearSlots(selectedSlots);
    return true;
  },

  undo(): boolean {
    return useSongStore.getState().undoEdit();
  },
};

/** Cut, copy, paste, Delete and undo over the selected slots while the
 *  score is on screen. Text controls, open dialogs and a text selection
 *  elsewhere on the page keep the browser's own behavior. */
export function useScoreClipboard(): void {
  useEffect(() => {
    const controller = new AbortController();
    const { signal } = controller;
    let pending: { action: ScoreEditKey; timer: number } | null = null;
    const settle = () => {
      if (pending) window.clearTimeout(pending.timer);
      pending = null;
    };
    const textSelected = () => Boolean(window.getSelection()?.toString());
    const ours = (event: Event) => !editorKeysBlocked(event.target) && !textSelected();

    // Safari enables Copy, Cut and Paste over non-editable content only when
    // the before* event is cancelled; the clipboard event then follows.
    const before = (event: Event) => { if (ours(event)) event.preventDefault(); };
    for (const type of ['beforecopy', 'beforecut', 'beforepaste']) document.addEventListener(type, before, { signal });

    const clipboard = (action: 'copy' | 'cut' | 'paste') => (event: ClipboardEvent) => {
      if (!ours(event)) return;
      settle();
      scoreEdits[action](event);
    };
    document.addEventListener('copy', clipboard('copy'), { signal });
    document.addEventListener('cut', clipboard('cut'), { signal });
    document.addEventListener('paste', clipboard('paste'), { signal });

    const keydown = (event: KeyboardEvent) => {
      const action = scoreEditKey(event, editorKeysBlocked(event.target));
      if (!action) return;
      if (action === 'clear' || action === 'undo') {
        if (scoreEdits[action]()) event.preventDefault();
        return;
      }
      if (textSelected()) return;
      // The keystroke's clipboard event carries the data; when none arrives,
      // the editor's memory copy stands in.
      settle();
      pending = {
        action,
        timer: window.setTimeout(() => { pending = null; scoreEdits[action](); }, CLIPBOARD_EVENT_GRACE_MS),
      };
    };
    window.addEventListener('keydown', keydown, { signal });

    return () => {
      settle();
      controller.abort();
    };
  }, []);
}
