import { DEFAULT_PLAY_SETTINGS } from '../../../engine/playSettings';
import { usePlayStore } from '../../../store/playStore';
import { useSongStore } from '../../../store/songStore';
import VolumeFader from './ui/VolumeFader';

/** The tempo range the song store clamps to, and a new song's tempo, which
 *  a double-click on the BPM fader restores. */
const MIN_BPM = 20;
const MAX_BPM = 300;
const DEFAULT_BPM = 80;

/** Tempo and four independent mix controls; the master scales the complete mix. */
export default function ControllerPane() {
  const bpm = useSongStore((s) => s.song.bpm);
  const setBpmLive = useSongStore((s) => s.setBpmLive);
  const settings = usePlayStore((s) => s.settings);
  const setMasterVolume = usePlayStore((s) => s.setMasterVolume);
  const setHarmonyVolume = usePlayStore((s) => s.setHarmonyVolume);
  const setBassVolume = usePlayStore((s) => s.setBassVolume);
  const setDrumVolume = usePlayStore((s) => s.setDrumVolume);
  // Gains are 0..1 in the store and 0..100 on the faders.
  const channels = [
    { id: 'master', label: 'Master', value: settings.masterVolume, initial: DEFAULT_PLAY_SETTINGS.masterVolume, set: setMasterVolume },
    { id: 'harmony', label: 'Harmony', value: settings.harmonyVolume, initial: DEFAULT_PLAY_SETTINGS.harmonyVolume, set: setHarmonyVolume },
    { id: 'bass', label: 'Bass', value: settings.bassVolume, initial: DEFAULT_PLAY_SETTINGS.bassVolume, set: setBassVolume },
    { id: 'drums', label: 'Drums', value: settings.drumVolume, initial: DEFAULT_PLAY_SETTINGS.drumVolume, set: setDrumVolume },
  ];
  return (
    <div className="h-full min-h-0 overflow-y-auto overscroll-contain">
      <div role="group" aria-label="Controller view" className="te-volume-bank">
        <VolumeFader label="BPM" ariaLabel="Tempo in beats per minute" value={bpm}
          min={MIN_BPM} max={MAX_BPM} step={1} defaultValue={DEFAULT_BPM} onChange={setBpmLive} />
        {channels.map((channel) => (
          <VolumeFader
            key={channel.id}
            label={channel.label}
            value={Math.round(channel.value * 100)}
            defaultValue={Math.round(channel.initial * 100)}
            onChange={(value) => channel.set(value / 100)}
          />
        ))}
      </div>
    </div>
  );
}
