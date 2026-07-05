import type { Chord } from '../models/types';
import { NOTE_NAMES } from './constants';

const heldNotes = new Set<number>();

type MidiCallback = (notes: number[]) => void;
type ChordCallback = (chords: Chord[]) => void;

let noteCallback: MidiCallback | null = null;
let chordCallback: ChordCallback | null = null;
let midiAccess: MIDIAccess | null = null;
let connected = false;

function onMidiMessage(event: MIDIMessageEvent) {
  const data = event.data;
  if (!data || data.length < 3) return;

  const status = data[0] & 0xf0;
  const note = data[1];
  const velocity = data[2];

  if (status === 0x90 && velocity > 0) {
    heldNotes.add(note);
  } else if (status === 0x80 || (status === 0x90 && velocity === 0)) {
    heldNotes.delete(note);
  }

  const notes = Array.from(heldNotes).sort((a, b) => a - b);
  noteCallback?.(notes);

  if (notes.length >= 3) {
    const candidates = detectChordCandidates(notes);
    if (candidates.length > 0) chordCallback?.(candidates);
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

export function detectChord(midiNotes: number[]): Chord | null {
  const candidates = detectChordCandidates(midiNotes, 1);
  return candidates[0] ?? null;
}

export async function initMidi(
  onNotes: MidiCallback,
  onChord: ChordCallback,
): Promise<boolean> {
  noteCallback = onNotes;
  chordCallback = onChord;

  if (!navigator.requestMIDIAccess) {
    console.warn('Web MIDI API not supported');
    return false;
  }

  try {
    midiAccess = await navigator.requestMIDIAccess();
    const connectInputs = () => {
      if (!midiAccess) return;
      for (const input of midiAccess.inputs.values()) {
        input.onmidimessage = onMidiMessage;
      }
      connected = midiAccess.inputs.size > 0;
    };
    connectInputs();
    midiAccess.onstatechange = () => connectInputs();
    return connected;
  } catch (err) {
    console.warn('MIDI access denied:', err);
    return false;
  }
}

export function disconnectMidi(): void {
  if (midiAccess) {
    for (const input of midiAccess.inputs.values()) {
      input.onmidimessage = null;
    }
  }
  heldNotes.clear();
  noteCallback = null;
  chordCallback = null;
  connected = false;
}
