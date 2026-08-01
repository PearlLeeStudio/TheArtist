import { drumGrid, drumPatternById } from '../../../engine/drumPatterns';
import { bassPatternFor, meterRhythm, patternSteps, resolveDrumPatternId, resolveHarmonyRhythm } from '../../../engine/playSettings';
import { usePlayStore } from '../../../store/playStore';
import { useSongStore } from '../../../store/songStore';
import StepRow from './ui/StepRow';
import { usePlayStep } from './usePlayStep';

/** LED cells per beat (16ths); the playhead counts in the same steps. */
const STEPS_PER_BEAT = 4;

/** A 1-bar rhythm as one LED cell per 16th of the bar: full brightness
 *  where it strikes, off elsewhere. */
function rhythmCells(pattern: number[], beats: number): { level: number }[] {
  return patternSteps(meterRhythm(pattern, beats), beats, STEPS_PER_BEAT).map((on) => ({ level: on ? 1 : 0 }));
}

/** Playback indicators are permanent, independently of the selected control tab. */
export default function PlaybackStatus() {
  const settings = usePlayStore((state) => state.settings);
  const genre = useSongStore((state) => state.song.genre);
  const beats = useSongStore((state) => state.song.timeSignature[0]);
  const step = usePlayStep(STEPS_PER_BEAT);
  const rhythm = resolveHarmonyRhythm(settings, genre);
  const harmony = rhythmCells(rhythm.pattern, beats);
  const bass = rhythmCells(bassPatternFor(genre), beats);
  const drum = drumPatternById(resolveDrumPatternId(settings, genre));
  const rows = drumGrid(drum, beats * STEPS_PER_BEAT);

  return (
    <div className="te-playback-status shrink-0 grid grid-cols-1 gap-1" aria-label="Live rhythm and drum status">
      <section className="te-status-window" aria-label="Rhythm playback">
        <p className="te-status-title"><span className="te-status-type">Rhythm</span><span className="min-w-0 truncate">{rhythm.label}</span></p>
        <div className="flex flex-col gap-[2px]">
          <StepRow label="HARM" cells={harmony} playhead={step} dim={settings.harmonyVolume === 0} />
          <StepRow label="BASS" cells={bass} playhead={step} dim={!settings.bassEnabled} />
        </div>
      </section>
      <section className="te-status-window" aria-label="Drum playback">
        <p className="te-status-title" title={drum?.name ?? 'Drums off'}>
          <span className="te-status-type">Drums</span><span className="min-w-0 truncate">{drum?.name ?? 'Off'}</span>
        </p>
        <div className="flex flex-col gap-[2px]">
          {rows.map((row) => <StepRow key={row.label} label={row.label} cells={row.cells} playhead={drum ? step : null} dim={!drum} />)}
        </div>
      </section>
    </div>
  );
}
