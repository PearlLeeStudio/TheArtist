// Family buttons keep every sound reachable without a scrolling list.
import { useState } from 'react';
import { usePlayStore } from '../../../store/playStore';
import { useSongStore } from '../../../store/songStore';
import { genreHarmonyInstrument } from '../../../engine/playSettings';
import { prepareHarmonyFamily, prepareHarmonyInstrument } from '../../../engine/playback';
import { HARMONY_INSTRUMENTS, instrumentLabel } from '../../../engine/genreTables';
import type { InstrumentFamily } from '../../../engine/genreTables';
import { genreTag } from './genreTag';
import { KEY_TEXT_STYLE, moveChoiceByKey, pickedLabelStyle } from './paneKeys';

interface FamilyGroup {
  id: string;
  emoji: string;
  label: string;
  /** Shorter key-cap text where the label is too long for the key. */
  keyLabel?: string;
  families: InstrumentFamily[];
}

const GROUPS: FamilyGroup[] = [
  { id: 'keys', emoji: '🎹', label: 'Keys', families: ['keys'] },
  { id: 'guitar', emoji: '🎸', label: 'Guitar', families: ['guitar'] },
  { id: 'strings', emoji: '🎻', label: 'Strings', families: ['strings'] },
  { id: 'woodwinds', emoji: '🪈', label: 'Woodwinds', keyLabel: 'Winds', families: ['winds', 'reeds'] },
  { id: 'brass', emoji: '🎺', label: 'Brass', families: ['brass'] },
];

/** Full, readable names for the compact labels used elsewhere on the device. */
const NAMES: Record<string, string> = {
  electric_piano_1: 'Electric piano',
  clavinet: 'Clavinet',
  pad_2_warm: 'Warm pad',
  acoustic_guitar_nylon: 'Nylon guitar',
  acoustic_guitar_steel: 'Steel guitar',
  electric_guitar_clean: 'Clean guitar',
  overdriven_guitar: 'Overdrive',
  distortion_guitar: 'Distortion',
  pizzicato_strings: 'Pizzicato',
  soprano_sax: 'Soprano sax',
  alto_sax: 'Alto sax',
  tenor_sax: 'Tenor sax',
  baritone_sax: 'Baritone sax',
  french_horn: 'French horn',
  brass_section: 'Brass section',
  muted_trumpet: 'Muted trumpet',
};

/** The group holding the harmony sound `name`; Keys for no known sound. */
function groupIdOf(name: string | null): string {
  const sound = HARMONY_INSTRUMENTS.find((instrument) => instrument.name === name);
  if (!sound) return 'keys';
  return GROUPS.find((group) => group.families.includes(sound.family))?.id ?? 'keys';
}

/** Every harmony sound remains available through its family. The bass follows the genre. */
export default function InstrumentPane() {
  const override = usePlayStore((s) => s.settings.harmonyOverride);
  const harmonyOn = usePlayStore((s) => s.settings.harmonyVolume > 0);
  const setHarmonyEnabled = usePlayStore((s) => s.setHarmonyEnabled);
  const setHarmonyOverride = usePlayStore((s) => s.setHarmonyOverride);
  const genre = useSongStore((s) => s.song.genre);

  const genreDefault = genreHarmonyInstrument(genre);
  const tag = genreTag(genre);
  // Opens on the family of the sound playing now.
  const [openGroupId, setOpenGroupId] = useState(() => groupIdOf(override ?? genreDefault));
  const openGroup = GROUPS.find((item) => item.id === openGroupId) ?? GROUPS[0];
  const instruments = HARMONY_INSTRUMENTS.filter((instrument) => openGroup.families.includes(instrument.family));

  const isAuto = override === null;
  const autoTitle = genreDefault
    ? `Auto: the ${tag} default (${instrumentLabel(genreDefault)})`
    : 'Auto: no genre set, so the built-in piano';

  return (
    <div className="te-instrument-list min-w-0 h-full p-[3px]"
      style={{ display: 'grid', gridTemplateRows: '26px 26px 36px 1fr', gap: 4 }} role="group" aria-label="Harmony instrument"
      onKeyDown={(event) => moveChoiceByKey(event, '[data-instrument-choice]')}>
      <button type="button" aria-pressed={harmonyOn} data-active={harmonyOn ? '' : undefined}
        onClick={() => setHarmonyEnabled(!harmonyOn)}
        className="te-key btn-hover w-full px-2"
        style={{ ...KEY_TEXT_STYLE, fontWeight: harmonyOn ? 700 : 400 }}
        title="Turn the harmony part on or off without changing the selected instrument">
        Instrument {harmonyOn ? 'on' : 'off'}
      </button>
      <button type="button" aria-pressed={isAuto} data-instrument-choice
        data-active={isAuto ? '' : undefined} onClick={() => setHarmonyOverride(null)} title={autoTitle}
        onPointerEnter={() => prepareHarmonyInstrument(genreDefault)} onFocus={() => prepareHarmonyInstrument(genreDefault)}
        className="te-key btn-hover flex w-full items-center justify-between gap-2 px-2 text-left"
        style={{ ...KEY_TEXT_STYLE, fontWeight: isAuto ? 700 : 400 }}>
        <span>Auto</span>
        <span style={{ color: 'var(--key-text-dim)', fontWeight: 400 }}>Genre default</span>
      </button>

      <div className="grid grid-cols-5 gap-[3px]" role="group" aria-label="Instrument families">
        {GROUPS.map((familyGroup) => {
          const isOpen = familyGroup.id === openGroupId;
          const prepareFamily = () => prepareHarmonyFamily(HARMONY_INSTRUMENTS
            .filter((sound) => familyGroup.families.includes(sound.family)).map((sound) => sound.name));
          return (
            <button key={familyGroup.id} type="button" className="te-key btn-hover flex min-w-0 flex-col items-center justify-center"
              aria-pressed={isOpen} data-active={isOpen ? '' : undefined}
              onPointerEnter={prepareFamily} onFocus={prepareFamily}
              onClick={() => setOpenGroupId(familyGroup.id)} title={familyGroup.label} style={{ letterSpacing: 'normal' }}>
              <span aria-hidden="true" style={{ fontSize: 14, lineHeight: '16px' }}>{familyGroup.emoji}</span>
              <span style={{ fontSize: 10, lineHeight: '12px' }}>{familyGroup.keyLabel ?? familyGroup.label}</span>
            </button>
          );
        })}
      </div>
      <div className="grid min-h-0 grid-cols-5 content-start gap-[3px]"
        style={{ gridAutoRows: 36 }}
        role="group" aria-label={`${openGroup.label} instruments`}>
        {instruments.map((instrument) => {
          const selected = override === instrument.name;
          const label = NAMES[instrument.name] ?? instrument.label;
          return (
            <button key={instrument.name} type="button" aria-pressed={selected} data-instrument-choice
              aria-label={`${label}, ${openGroup.label}`}
              data-active={selected ? '' : undefined} onClick={() => setHarmonyOverride(instrument.name)}
              onPointerEnter={() => prepareHarmonyInstrument(instrument.name)} onFocus={() => prepareHarmonyInstrument(instrument.name)}
              title={`${openGroup.label} · ${label}`} className="te-key btn-hover min-w-0 px-1 py-1 text-left"
              style={{ minHeight: 28, fontSize: 11, lineHeight: '12px', letterSpacing: 'normal',
                whiteSpace: 'normal', overflowWrap: 'anywhere', ...pickedLabelStyle(selected) }}>
              {label}
            </button>
          );
        })}
      </div>
    </div>
  );
}
