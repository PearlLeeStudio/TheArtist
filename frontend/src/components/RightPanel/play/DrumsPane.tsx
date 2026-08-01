// The shared playback status stays visible above this selection grid.
import { DRUM_PATTERNS, drumPatternById } from '../../../engine/drumPatterns';
import { genreDrumPatternId } from '../../../engine/playSettings';
import { usePlayStore } from '../../../store/playStore';
import { useSongStore } from '../../../store/songStore';
import ChoiceKeyGrid from './ChoiceKeyGrid';
import { genreTag } from './genreTag';
import { moveChoiceByKey, WIDE_KEY_STYLE } from './paneKeys';

/** Drums: follow the genre (Auto), off, or one of the 15 patterns. A pick
 *  stays when the genre changes. */
export default function DrumsPane() {
  const settings = usePlayStore((s) => s.settings);
  const setDrumAuto = usePlayStore((s) => s.setDrumAuto);
  const setDrumPattern = usePlayStore((s) => s.setDrumPattern);
  const setDrumsEnabled = usePlayStore((s) => s.setDrumsEnabled);
  const genre = useSongStore((s) => s.song.genre);

  const isAuto = settings.drumMode === 'auto';
  const isOff = !isAuto && settings.drumPatternId === null;
  const drumsOn = !isOff;
  // With drums off no pattern is checked, so Auto keeps the group's tab stop.
  const autoTabStop = isAuto || isOff;
  const genreId = genreDrumPatternId(genre);
  const genrePattern = drumPatternById(genreId);
  const tag = genreTag(genre);
  const autoTitle = genrePattern
    ? `Auto: drums follow the genre (${genrePattern.name} for ${tag})`
    : 'Auto: no genre set, so no drums';
  const choices = DRUM_PATTERNS.map((pattern) => {
    const swingNote = pattern.swing ? ' · swing' : '';
    const autoNote = isAuto && pattern.id === genreId ? ` · Auto plays this for ${tag}` : '';
    return {
      id: pattern.id,
      label: pattern.name,
      ariaLabel: pattern.name,
      checked: !isAuto && settings.drumPatternId === pattern.id,
      onSelect: () => setDrumPattern(pattern.id),
      title: `${pattern.name}${swingNote}${autoNote}`,
    };
  });

  return (
    <div className="te-choice-grid flex h-full min-h-0 min-w-0 flex-col gap-1 p-[3px]">
      <button type="button" aria-pressed={drumsOn} data-active={drumsOn ? '' : undefined}
        onClick={() => setDrumsEnabled(!drumsOn)} className="te-key btn-hover w-full shrink-0 px-2"
        style={{ ...WIDE_KEY_STYLE, fontWeight: drumsOn ? 700 : 400 }}
        title={drumsOn ? 'Turn drums off while keeping the current pattern.' : 'Turn drums on and restore the previous pattern.'}>
        Drums {drumsOn ? 'on' : 'off'}
      </button>

      <div role="radiogroup" aria-label="Drum pattern" onKeyDown={(event) => moveChoiceByKey(event, '[role="radio"]')}
        className="flex min-h-0 flex-1 flex-col gap-1">
        <button type="button" role="radio" aria-label="Auto drum pattern" aria-checked={isAuto}
          tabIndex={autoTabStop ? 0 : -1} data-active={isAuto ? '' : undefined}
          onClick={setDrumAuto} title={autoTitle} className="te-key btn-hover w-full shrink-0 px-2"
          style={{ ...WIDE_KEY_STYLE, fontWeight: isAuto ? 700 : 400 }}>
          Auto
        </button>
        <ChoiceKeyGrid fullWidth choices={choices} />
      </div>
    </div>
  );
}
