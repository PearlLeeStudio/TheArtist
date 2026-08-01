// The chord readout and drawing remain above every selection tab.
import { usePlayStore } from '../../../store/playStore';
import type { KeyboardEvent } from 'react';
import { useSongStore } from '../../../store/songStore';
import { resolveHarmonyInstrument } from '../../../engine/playSettings';
import type { VisualDisplayMode } from '../../../store/playStore';
import { instrumentLabel } from '../../../engine/genreTables';
import { generateGuitarVoicing, midiToNoteName, resolvePianoVoicing } from '../../../engine/voicingEngine';
import { chordToString } from '../../../engine/chordParser';
import type { Chord, Measure, SlotAddress } from '../../../models/types';
import PianoKeyboard from '../PianoKeyboard';
import GuitarFretboard from '../GuitarFretboard';
import KeyButton from './ui/KeyButton';
import Screen from './ui/Screen';

/** The drawing frame's widest size, which caps the pane's height; a
 *  fretboard's frame stays narrower. */
const DRAWING_MAX_WIDTH = 320;
const FRETBOARD_MAX_WIDTH = 220;
const MUTED_FRETS = [-1, -1, -1, -1, -1, -1];

/** Where the drawn chord comes from, highest priority first. */
type Source = 'midi-chord' | 'midi-raw' | 'playback' | 'selection' | 'score';

const SOURCE_TAG: Record<Source, string> = {
  'midi-chord': 'MIDI',
  'midi-raw': 'MIDI',
  playback: 'PLAY',
  selection: 'SEL',
  score: 'CHORD',
};

/** Key highlight per source: green for MIDI, the playback yellow, the selection yellow. */
const SOURCE_COLOR: Record<Source, string> = {
  'midi-chord': 'var(--text-accent)',
  'midi-raw': 'var(--text-accent)',
  playback: 'var(--playback-bar)',
  selection: 'var(--brand-yellow)',
  score: 'var(--brand-yellow)',
};

const MODES: { mode: VisualDisplayMode; label: string; emoji: string; title: string }[] = [
  { mode: 'piano', label: 'Piano', emoji: '🎹', title: 'Show the chord on a piano keyboard' },
  { mode: 'guitar', label: 'Guitar', emoji: '🎸', title: 'Show the chord on a guitar fretboard' },
];

function chordAt(measures: Measure[], slot: SlotAddress): Chord | null {
  return measures[slot.measureIndex]?.chords[slot.slotIndex] ?? null;
}

/** MIDI > playback > selected slot > the score's first readable chord. */
function pickSource(
  midiChord: Chord | null,
  midiNotes: number[],
  playing: SlotAddress | null,
  selected: SlotAddress | null,
  measures: Measure[],
): { source: Source | null; chord: Chord | null } {
  if (midiChord) return { source: 'midi-chord', chord: midiChord };
  if (midiNotes.length > 0) return { source: 'midi-raw', chord: null };
  if (playing) return { source: 'playback', chord: chordAt(measures, playing) };
  if (selected) return { source: 'selection', chord: chordAt(measures, selected) };
  for (const measure of measures) {
    for (const chord of measure.chords) {
      if (chord) return { source: 'score', chord };
    }
  }
  return { source: null, chord: null };
}

function noteNames(notes: number[]): string {
  return [...new Set(notes)].sort((a, b) => a - b).map(midiToNoteName).join(' ');
}

/** Piano/Guitar keys: the arrows, Home and End pick one (no wrap-around). */
function handleDisplayKey(event: KeyboardEvent<HTMLDivElement>) {
  const radios = Array.from(event.currentTarget.querySelectorAll<HTMLButtonElement>('[role="radio"]'));
  if (!radios.includes(event.target as HTMLButtonElement)) return;
  let next: number;
  if (event.key === 'ArrowDown' || event.key === 'ArrowRight' || event.key === 'End') next = 1;
  else if (event.key === 'ArrowUp' || event.key === 'ArrowLeft' || event.key === 'Home') next = 0;
  else return;
  event.preventDefault();
  radios[next]?.click();
  radios[next]?.focus();
}

interface ChordReadoutProps {
  showHint: boolean;
  source: Source | null;
  chordName: string | null;
  midiNotes: number[];
}

/** The screen's main line: a prompt, the held MIDI notes, the chord name,
 *  or a dash for an empty slot under the playhead. */
function ChordReadout({ showHint, source, chordName, midiNotes }: ChordReadoutProps) {
  if (showHint) return <span style={{ color: 'var(--screen-text-dim)' }}>Select a chord</span>;
  if (source === 'midi-raw') {
    return <span className="min-w-0 flex-1 truncate" style={{ fontSize: 13 }}>{noteNames(midiNotes)}</span>;
  }
  if (chordName) {
    return (
      <span className="min-w-0 flex-1 truncate" style={{ fontSize: 24, lineHeight: '32px', fontWeight: 600 }}>
        {chordName}
      </span>
    );
  }
  return <span style={{ fontSize: 22, lineHeight: '28px', color: 'var(--screen-text-dim)' }}>—</span>;
}

/** The active chord on a keyboard or fretboard, and the shape it is voiced in. */
export default function VisualPane({ midiActiveNotes }: { midiActiveNotes: number[] }) {
  const measures = useSongStore((s) => s.song.measures);
  const genre = useSongStore((s) => s.song.genre);
  const selectedSlots = useSongStore((s) => s.selectedSlots);
  const playbackPosition = useSongStore((s) => s.playbackPosition);
  const midiCandidates = useSongStore((s) => s.midiCandidates);
  const settings = usePlayStore((s) => s.settings);
  const shape = usePlayStore((s) => s.displayMode);
  const setDisplayMode = usePlayStore((s) => s.setDisplayMode);

  // This switch chooses the drawing independently of the playback sound.
  const instrument = resolveHarmonyInstrument(settings, genre);
  const instrumentName = instrument ? instrumentLabel(instrument) : 'Built-in piano';

  const { source, chord } = pickSource(
    midiCandidates[0] ?? null,
    midiActiveNotes,
    playbackPosition,
    selectedSlots[0] ?? null,
    measures,
  );
  const showHint = source === null || (source === 'selection' && chord === null);
  const highlight = source ? SOURCE_COLOR[source] : 'var(--brand-yellow)';
  const chordName = chord ? chordToString(chord) : null;

  const pianoNotes = source === 'midi-raw' ? midiActiveNotes : chord ? resolvePianoVoicing(chord) : [];
  const guitarFrets = chord ? generateGuitarVoicing(chord) : MUTED_FRETS;

  return (
    <div role="group" aria-label="Visual voicing" className="flex flex-col min-w-0 gap-2">
      <Screen style={{ height: 48, padding: '8px' }}>
        <div className="flex items-center gap-2" style={{ height: 32 }}>
          <span className="shrink-0 text-[11px] uppercase" style={{ color: highlight }}>{source ? SOURCE_TAG[source] : 'Visual'}</span>
          <ChordReadout showHint={showHint} source={source} chordName={chordName} midiNotes={midiActiveNotes} />
          <span className="ml-auto min-w-0 truncate" style={{ maxWidth: 78, fontSize: 12, color: 'var(--screen-text-dim)' }} title={instrumentName}>{instrumentName}</span>
        </div>
      </Screen>

      <div
        className="grid min-w-0 items-center gap-1.5"
        style={{ gridTemplateColumns: 'minmax(0, 1fr) 32px', height: shape === 'piano' ? 72 : 'clamp(72px, 8dvh, 132px)' }}
      >
        <div
          className="te-visual-selector grid gap-[3px] grid-cols-1"
          role="radiogroup"
          aria-label="Piano or guitar display"
          onKeyDown={handleDisplayKey}
          style={{ gridColumn: 2, gridRow: 1 }}
        >
          {MODES.map(({ mode, label, emoji, title }) => {
            const chosen = shape === mode;
            return (
              <KeyButton
                key={mode}
                label={label}
                ariaLabel={label}
                active={chosen}
                role="radio"
                tabIndex={chosen ? 0 : -1}
                onClick={() => setDisplayMode(mode)}
                className="te-visual-key"
                title={title}
              ><span aria-hidden="true" style={{ fontSize: 17, lineHeight: '20px' }}>{emoji}</span></KeyButton>
            );
          })}
        </div>

        <div
          className="te-voicing-drawing w-full mx-auto"
          role="img"
          aria-label={`${shape === 'guitar' ? 'Guitar shape' : 'Keyboard voicing'}${chordName ? ` of ${chordName}` : ''}`}
          style={{
            maxWidth: shape === 'guitar' ? FRETBOARD_MAX_WIDTH : DRAWING_MAX_WIDTH,
            padding: 4,
            gridColumn: 1,
            gridRow: 1,
            background: 'transparent',
          }}
        >
          {shape === 'guitar' ? (
            <GuitarFretboard frets={guitarFrets} highlightColor={highlight} />
          ) : (
            <PianoKeyboard activeNotes={pianoNotes} highlightColor={highlight} />
          )}
        </div>
      </div>
    </div>
  );
}
