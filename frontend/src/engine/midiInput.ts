import type { Chord } from '../models/types';
import { NOTE_NAMES } from './constants.ts';
import { openMidiBridge } from './midiBridge.ts';

export type MidiTransport = 'auto' | 'bridge';

const heldNotes = new Set<number>();

type MidiCallback = (notes: number[]) => void;
type ChordCallback = (chords: Chord[]) => void;

let noteCallback: MidiCallback | null = null;
let chordCallback: ChordCallback | null = null;
let midiAccess: MIDIAccess | null = null;
let requestGeneration = 0;
const attachedInputs = new Set<MIDIInput>();
let bridge: ReturnType<typeof openMidiBridge> | null = null;
const heldVoices = new Map<string, number>();

function onMidiMessage(event: Pick<MIDIMessageEvent, 'data'>, port = 'native') {
  const data = event.data;
  if (!data || data.length < 3) return;

  const status = data[0] & 0xf0;
  const note = data[1];
  const velocity = data[2];
  const voice = `${port}:${data[0] & 0x0f}:${note}`;

  if (status === 0x90 && velocity > 0) {
    heldVoices.set(voice, note);
  } else if (status === 0x80 || (status === 0x90 && velocity === 0)) {
    heldVoices.delete(voice);
  } else if (status === 0xb0 && (note === 120 || note === 123)) {
    const channel = `${port}:${data[0] & 0x0f}:`;
    for (const key of heldVoices.keys()) if (key.startsWith(channel)) heldVoices.delete(key);
  } else return;

  heldNotes.clear();
  for (const pitch of heldVoices.values()) heldNotes.add(pitch);

  const notes = Array.from(heldNotes).sort((a, b) => a - b);
  noteCallback?.(notes);

  if (notes.length >= 3) {
    const candidates = detectChordCandidates(notes);
    chordCallback?.(candidates);
  }
}

const CHORD_PATTERNS: [string, number[]][] = [
  ['maj',    [0, 4, 7]],
  ['min',    [0, 3, 7]],
  ['7',      [0, 4, 7, 10]],
  ['maj7',   [0, 4, 7, 11]],
  ['m7',     [0, 3, 7, 10]],
  ['dim',    [0, 3, 6]],
  ['dim7',   [0, 3, 6, 9]],
  ['m7b5',   [0, 3, 6, 10]],
  ['aug',    [0, 4, 8]],
  ['sus4',   [0, 5, 7]],
  ['sus2',   [0, 2, 7]],
  ['6',      [0, 4, 7, 9]],
  ['m6',     [0, 3, 7, 9]],
  ['9',      [0, 4, 7, 10, 2]],
  ['m9',     [0, 3, 7, 10, 2]],
  ['maj9',   [0, 4, 7, 11, 2]],
  ['add9',   [0, 4, 7, 2]],
  ['mMaj7',  [0, 3, 7, 11]],
  ['11',     [0, 4, 7, 10, 2, 5]],
  ['13',     [0, 4, 7, 10, 2, 9]],
];

export function detectChordCandidates(midiNotes: number[], maxResults = 3): Chord[] {
  if (midiNotes.length < 3) return [];

  // Lowest sounding pitch class — when it differs from the matched root,
  // we emit a slash chord (e.g. F-G-B-D with F at bottom → G7/F).
  const lowestMidi = Math.min(...midiNotes);
  const bassPC = lowestMidi % 12;

  const pitchClasses = [...new Set(midiNotes.map(n => n % 12))].sort((a, b) => a - b);
  const allMatches: { root: number; quality: string; score: number }[] = [];

  for (const root of pitchClasses) {
    const intervals = pitchClasses.map(pc => (pc - root + 12) % 12).sort((a, b) => a - b);
    for (const [quality, pattern] of CHORD_PATTERNS) {
      const patternSet = new Set(pattern);
      let matches = 0;
      let misses = 0;
      for (const i of intervals) {
        if (patternSet.has(i)) matches++;
        else misses++;
      }
      const score = matches * 2 - misses;
      if (matches >= 3 || (matches >= 2 && pattern.length <= 3)) {
        allMatches.push({ root, quality, score });
      }
    }
  }

  if (allMatches.length === 0) return [];

  // Sort by score desc, deduplicate by root+quality
  allMatches.sort((a, b) => b.score - a.score);
  const seen = new Set<string>();
  const unique: typeof allMatches = [];
  for (const m of allMatches) {
    const key = `${m.root}-${m.quality}`;
    if (!seen.has(key)) {
      seen.add(key);
      unique.push(m);
    }
  }

  return unique.slice(0, maxResults).map(m => {
    const isInversion = m.root !== bassPC;
    return {
      root: NOTE_NAMES[m.root],
      quality: m.quality,
      bass: isInversion ? NOTE_NAMES[bassPC] : undefined,
      romanNumeral: '',
      voicing: midiNotes,
      extensions: [],
    };
  });
}

export async function initMidi(
  onNotes: MidiCallback,
  onChord: ChordCallback,
  onConnection?: (connected: boolean, devices: string[]) => void,
  transport: MidiTransport = 'auto',
): Promise<boolean> {
  const request = ++requestGeneration;
  detachAccess();
  resetMidiNotes();
  noteCallback = null;
  chordCallback = null;

  if (transport === 'bridge' || !navigator.requestMIDIAccess) {
    noteCallback = onNotes;
    chordCallback = onChord;
    bridge = openMidiBridge((data, port) => {
      if (request === requestGeneration) onMidiMessage({ data: new Uint8Array(data) }, port);
    }, (connected, devices) => {
      if (request !== requestGeneration) return;
      resetMidiNotes();
      noteCallback?.([]);
      onConnection?.(connected, devices);
    });
    try { return await bridge.ready; }
    catch (error) { if (request === requestGeneration) disconnectMidi(); throw error; }
  }

  try {
    const access = await navigator.requestMIDIAccess({ sysex: false });
    // A cancelled permission request must never reconnect after the user
    // leaves the editor or starts a newer connection request.
    if (request !== requestGeneration) return false;
    midiAccess = access;
    noteCallback = onNotes;
    chordCallback = onChord;
    // These handlers outlive this call. Each one acts only while its
    // connection is still the current one.
    const isCurrent = () => request === requestGeneration && midiAccess === access;
    const connectInputs = (): boolean => {
      if (!isCurrent()) return false;
      detachInputs();
      for (const input of access.inputs.values()) {
        if (input.state !== 'connected') continue;
        input.onmidimessage = (event) => {
          if (isCurrent()) onMidiMessage(event, input.id);
        };
        attachedInputs.add(input);
      }
      const connected = attachedInputs.size > 0;
      onConnection?.(connected, Array.from(attachedInputs, (input) => input.name || 'MIDI input'));
      return connected;
    };
    const connected = connectInputs();
    access.onstatechange = () => {
      if (request !== requestGeneration) return;
      resetMidiNotes();
      noteCallback?.([]);
      connectInputs();
    };
    return connected;
  } catch (error) {
    if (request !== requestGeneration) return false;
    disconnectMidi();
    const reason = error instanceof Error ? error.name : '';
    throw new Error(reason === 'NotAllowedError' || reason === 'SecurityError'
      ? 'Allow MIDI in this site’s browser permissions, then enable MIDI again. You can also choose Local bridge on Mac.'
      : 'The browser could not open MIDI. Check the controller connection, then retry or choose Local bridge on Mac.', { cause: error });
  }
}

function detachInputs(): void {
  for (const input of attachedInputs) input.onmidimessage = null;
  attachedInputs.clear();
}

function detachAccess(): void {
  bridge?.close();
  bridge = null;
  detachInputs();
  if (midiAccess) midiAccess.onstatechange = null;
  midiAccess = null;
}

/** Drop notes from the previous song while keeping the controller connected. */
export function resetMidiNotes(): void {
  heldNotes.clear();
  heldVoices.clear();
}

export function disconnectMidi(): void {
  requestGeneration++;
  detachAccess();
  resetMidiNotes();
  noteCallback = null;
  chordCallback = null;
}
