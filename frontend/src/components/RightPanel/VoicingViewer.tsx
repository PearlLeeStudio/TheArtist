import { useEffect } from 'react';
import { useSongStore } from '../../store/songStore';
import type { Instrument } from '../../store/songStore';
import { generatePianoVoicing, generateGuitarVoicing } from '../../engine/voicingEngine';
import {
  setCurrentInstrument,
  setHarmonyOverride as setHarmonyOverrideEngine,
} from '../../engine/playback';
import { harmonyInstrumentForGenre } from '../../engine/instruments';
import PianoKeyboard from './PianoKeyboard';
import GuitarFretboard from './GuitarFretboard';
import { chordToString } from '../../engine/chordParser';
import type { Chord } from '../../models/types';

interface VoicingViewerProps {
  midiActiveNotes?: number[];
}

const GUITAR_FAMILY = new Set<string>([
  'acoustic_guitar_nylon',
  'acoustic_guitar_steel',
  'electric_guitar_jazz',
  'electric_guitar_clean',
  'electric_guitar_muted',
  'overdriven_guitar',
  'distortion_guitar',
  'guitar_harmonics',
]);

function displayModeFor(harmonyInst: string | null): 'piano' | 'guitar' {
  if (!harmonyInst) return 'piano';
  return GUITAR_FAMILY.has(harmonyInst) ? 'guitar' : 'piano';
}

// Pearl 2026-05-10 — melody track dropped, harmony absorbs the variety
// role via this instrument-override grid + the per-genre rhythm pattern
// in arrangement.ts (GENRE_HARMONY_RHYTHM). Grouped into orchestra / big-
// band / band families so the user can browse by section. All names are
// FluidR3_GM Soundfont keys (instruments.ts loads them lazily on first
// click). Polyphonic-friendly instruments (piano, guitar, strings,
// brass section, choir) sound natural; monophonic ones (solo violin,
// clarinet, etc.) play polyphonically here for chord-stab effect — fine
// for sketching, weird as a final mix.
type HarmonyOption = { name: string; label: string };
const HARMONY_GROUPS: { title: string; options: HarmonyOption[] }[] = [
  {
    title: 'Keys / Voice',
    options: [
      { name: 'acoustic_grand_piano', label: 'Piano' },
      { name: 'electric_piano_1',     label: 'E.Piano' },
      { name: 'hammond_organ',        label: 'Organ' },
      { name: 'clavinet',             label: 'Clav' },
      { name: 'pad_2_warm',           label: 'Pad' },
      { name: 'choir_aahs',           label: 'Choir' },
    ],
  },
  {
    title: 'Guitar',
    options: [
      { name: 'acoustic_guitar_nylon', label: 'Nylon' },
      { name: 'acoustic_guitar_steel', label: 'Steel' },
      { name: 'electric_guitar_clean', label: 'Clean' },
      { name: 'overdriven_guitar',     label: 'Drive' },
      { name: 'distortion_guitar',     label: 'Dist' },
    ],
  },
  {
    title: 'Strings',
    options: [
      { name: 'violin',             label: 'Violin' },
      { name: 'viola',              label: 'Viola' },
      { name: 'cello',              label: 'Cello' },
      { name: 'string_ensemble_1',  label: 'Strings' },
      { name: 'pizzicato_strings',  label: 'Pizz' },
      { name: 'orchestral_harp',    label: 'Harp' },
    ],
  },
  {
    title: 'Winds',
    options: [
      { name: 'flute',    label: 'Flute' },
      { name: 'clarinet', label: 'Clarinet' },
      { name: 'oboe',     label: 'Oboe' },
      { name: 'bassoon',  label: 'Bassoon' },
    ],
  },
  {
    title: 'Reeds',
    options: [
      { name: 'soprano_sax',  label: 'S.Sax' },
      { name: 'alto_sax',     label: 'A.Sax' },
      { name: 'tenor_sax',    label: 'T.Sax' },
      { name: 'baritone_sax', label: 'B.Sax' },
    ],
  },
  {
    title: 'Brass',
    options: [
      { name: 'trumpet',       label: 'Trumpet' },
      { name: 'trombone',      label: 'Trombone' },
      { name: 'french_horn',   label: 'F.Horn' },
      { name: 'tuba',          label: 'Tuba' },
      { name: 'brass_section', label: 'Brass' },
      { name: 'muted_trumpet', label: 'MutedTpt' },
    ],
  },
];

export default function VoicingViewer({ midiActiveNotes }: VoicingViewerProps) {
  const song = useSongStore((s) => s.song);
  const setInstrumentStore = useSongStore((s) => s.setInstrument);
  const selectedSlots = useSongStore((s) => s.selectedSlots);
  const playbackPosition = useSongStore((s) => s.playbackPosition);
  const midiCandidates = useSongStore((s) => s.midiCandidates);
  const harmonyOverride = useSongStore((s) => s.harmonyOverride);
  const setHarmonyOverride = useSongStore((s) => s.setHarmonyOverride);

  const genreHarmony = harmonyInstrumentForGenre(song.genre) ?? 'acoustic_grand_piano';
  const effectiveHarmony = harmonyOverride ?? genreHarmony;
  const displayMode = displayModeFor(effectiveHarmony);

  // Mirror override into the playback engine so an in-flight Transport
  // callback picks the latest at note-trigger time. Live swap, no restart.
  useEffect(() => {
    setHarmonyOverrideEngine(harmonyOverride);
  }, [harmonyOverride]);

  // Sync display mode (keyboard vs fretboard) into store + playback
  // engine — chord-voicing math depends on this and must match what
  // the user sees.
  useEffect(() => {
    setInstrumentStore(displayMode as Instrument);
    setCurrentInstrument(displayMode as Instrument);
  }, [displayMode, setInstrumentStore]);

  // Priority: MIDI detected chord > MIDI raw notes > playback > selection
  const firstSlot = selectedSlots[0] ?? null;
  const midiDetectedChord = midiCandidates[0] ?? null;

  let activeChord: Chord | null = null;
  let source: 'midi-chord' | 'midi-raw' | 'playback' | 'selection' | null = null;

  if (midiDetectedChord) {
    activeChord = midiDetectedChord;
    source = 'midi-chord';
  } else if (midiActiveNotes && midiActiveNotes.length > 0) {
    source = 'midi-raw';
  } else if (playbackPosition) {
    activeChord = song.measures[playbackPosition.measureIndex]?.chords[playbackPosition.slotIndex] ?? null;
    source = 'playback';
  } else if (firstSlot) {
    activeChord = song.measures[firstSlot.measureIndex]?.chords[firstSlot.slotIndex] ?? null;
    source = 'selection';
  }

  const pianoNotes = source === 'midi-raw'
    ? (midiActiveNotes ?? [])
    : activeChord
      ? (activeChord.voicing && activeChord.voicing.length > 0
          ? activeChord.voicing
          : generatePianoVoicing(activeChord))
      : [];

  const guitarFrets = activeChord
    ? generateGuitarVoicing(activeChord)
    : [-1, -1, -1, -1, -1, -1];

  const accent = 'var(--text-accent)';
  const highlightColor = source === 'midi-chord' ? accent
    : source === 'midi-raw' ? accent
    : source === 'playback' ? 'var(--playback-bar)'
    : '#facc15';

  const btnStyle = (active: boolean): React.CSSProperties => active
    ? { background: 'var(--text-heading)', color: 'var(--bg-primary)' }
    : { background: 'var(--btn-bg)', color: 'var(--text-primary)' };

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <h3 className="text-xs uppercase tracking-wide font-semibold" style={{ color: 'var(--text-heading)' }}>Voicing</h3>
        <span className="text-[10px] tabular" style={{ color: 'var(--text-muted)' }}>
          {effectiveHarmony}
        </span>
      </div>

      {activeChord && (
        <p className="font-display text-center text-xl" style={{ color: 'var(--text-heading)', fontWeight: 600 }}>
          {chordToString(activeChord)}
          {source === 'midi-chord' && (
            <span className="ml-2 text-[10px] font-normal" style={{ color: 'var(--text-accent)', fontFamily: 'var(--font-serif)' }}>
              (from MIDI)
            </span>
          )}
        </p>
      )}

      <div className="p-2" style={{ background: 'var(--btn-bg)', border: '1px solid var(--border)' }}>
        {displayMode === 'guitar' ? (
          <GuitarFretboard frets={guitarFrets} highlightColor={highlightColor} />
        ) : (
          <PianoKeyboard activeNotes={pianoNotes} highlightColor={highlightColor} />
        )}
      </div>

      {!activeChord && source !== 'midi-raw' && (
        <p className="text-xs text-center italic" style={{ color: 'var(--text-muted)' }}>Select a chord to see voicing</p>
      )}

      {/* Harmony override grid — Auto + per-section instrument groups
          (Keys / Guitar / Strings / Winds / Reeds / Brass). Click sets
          the override; if playback is running, the next chord trigger
          picks it up immediately (live swap via setHarmonyOverrideEngine).
          Pairs with GENRE_HARMONY_RHYTHM for genre-distinct comping
          rhythm even when the user keeps Auto. */}
      <div className="space-y-1.5 pt-1">
        <div className="flex items-baseline justify-between">
          <span className="text-[10px] uppercase tracking-wide font-semibold" style={{ color: 'var(--text-muted)' }}>Harmony</span>
          {harmonyOverride && (
            <span className="text-[9px] tabular" style={{ color: 'var(--text-muted)' }}>genre default: {genreHarmony}</span>
          )}
        </div>
        <div className="grid grid-cols-4 gap-1">
          <button
            onClick={() => setHarmonyOverride(null)}
            className="px-1.5 py-1 text-[10px] truncate btn-hover"
            style={btnStyle(harmonyOverride === null)}
            title={`Auto (genre default: ${genreHarmony})`}
          >
            Auto
          </button>
        </div>
        {HARMONY_GROUPS.map((group) => (
          <div key={group.title} className="space-y-0.5">
            <span className="text-[9px] uppercase tracking-wider" style={{ color: 'var(--text-muted)' }}>
              {group.title}
            </span>
            <div className="grid grid-cols-4 gap-1">
              {group.options.map((opt) => (
                <button
                  key={opt.name}
                  onClick={() => setHarmonyOverride(opt.name)}
                  className="px-1.5 py-1 text-[10px] truncate btn-hover"
                  style={btnStyle(harmonyOverride === opt.name)}
                  title={opt.name}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
