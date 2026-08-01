// Rhythm choices sit below the permanent playback status windows.
import { HARMONY_RHYTHM_PRESETS, instrumentLabel } from '../../../engine/genreTables';
import {
  autoRhythmPreset,
  bassInstrumentFor,
  harmonyStrikes,
  meterRhythm,
  normalizeGenre,
} from '../../../engine/playSettings';
import { usePlayStore } from '../../../store/playStore';
import { useSongStore } from '../../../store/songStore';
import ChoiceKeyGrid from './ChoiceKeyGrid';
import { genreTag } from './genreTag';
import { moveChoiceByKey, WIDE_KEY_STYLE } from './paneKeys';

const SUBDIVISIONS: [number, string][] = [
  [0, ''], [0.25, 'e'], [1 / 3, '⅓'], [0.5, '&'], [2 / 3, '⅔'], [0.75, 'a'],
];

/** A beat position as counted in the bar: 0 → 1, 1.5 → 2&, 2/3 → 1⅔. */
function countLabel(pos: number): string {
  const beat = Math.floor(pos + 1e-6);
  const name = SUBDIVISIONS.find(([fraction]) => Math.abs(pos - beat - fraction) < 1e-3)?.[1];
  return name === undefined ? (pos + 1).toFixed(2) : `${beat + 1}${name}`;
}

/** Where a 1-bar rhythm strikes in a bar of `beats`, as counted: "1, 2&, 4". */
function strikeCounts(pattern: number[], beats: number): string {
  return harmonyStrikes(meterRhythm(pattern, beats), beats).map((strike) => countLabel(strike.pos)).join(', ');
}

/** Harmony comping rhythm (Auto or one of the presets) and the bass layer. */
export default function RhythmPane() {
  const settings = usePlayStore((s) => s.settings);
  const setHarmonyRhythmId = usePlayStore((s) => s.setHarmonyRhythmId);
  const setBassEnabled = usePlayStore((s) => s.setBassEnabled);
  const genre = useSongStore((s) => s.song.genre);
  const beats = useSongStore((s) => s.song.timeSignature[0]);

  const isAuto = settings.harmonyRhythmId === null;
  const auto = autoRhythmPreset(genre);
  const bassOn = settings.bassEnabled;
  const bassName = instrumentLabel(bassInstrumentFor(genre));
  const hasGenre = normalizeGenre(genre) !== null;
  const tag = genreTag(genre);
  const autoTitle = hasGenre
    ? `Auto: the ${tag} comping rhythm (${auto.label})`
    : `Auto: no genre set, so ${auto.label}`;
  const choices = HARMONY_RHYTHM_PRESETS.map((preset) => {
    const autoNote = isAuto && preset.id === auto.id ? ` · Auto plays this for ${tag}` : '';
    return {
      id: preset.id,
      label: preset.label,
      checked: settings.harmonyRhythmId === preset.id,
      onSelect: () => setHarmonyRhythmId(preset.id),
      title: `${preset.label}: strikes on ${strikeCounts(preset.pattern, beats)}${autoNote}`,
    };
  });

  return (
    <div className="te-choice-grid flex h-full min-h-0 min-w-0 flex-col gap-1 p-[3px]">
      <button type="button" aria-pressed={bassOn} data-active={bassOn ? '' : undefined}
        onClick={() => setBassEnabled(!bassOn)}
        className="te-key btn-hover w-full shrink-0 px-2"
        style={{ ...WIDE_KEY_STYLE, whiteSpace: 'nowrap', fontWeight: bassOn ? 700 : 400 }}
        title={`Bass ${bassOn ? 'on' : 'off'}: ${bassName} plays each chord's root on the ${hasGenre ? tag : 'default'} bass rhythm`}
      >
        Bass {bassOn ? 'on' : 'off'} · {bassName}
      </button>

      <div className="flex min-h-0 flex-1 flex-col gap-1" role="radiogroup" aria-label="Harmony rhythm"
        onKeyDown={(event) => moveChoiceByKey(event, '[role="radio"]')}>
        <button type="button" role="radio" aria-checked={isAuto} tabIndex={isAuto ? 0 : -1}
          data-active={isAuto ? '' : undefined} onClick={() => setHarmonyRhythmId(null)} title={autoTitle}
          className="te-key btn-hover w-full shrink-0 px-2"
          style={{ ...WIDE_KEY_STYLE, fontWeight: isAuto ? 700 : 400 }}>
          Auto
        </button>
        <ChoiceKeyGrid choices={choices} />
      </div>
    </div>
  );
}
