import type { SessionGenerationContext, Song } from '../models/types';
import { DEFAULT_SONG_TITLE } from '../engine/songForm';
import { playSettingsFromSession } from '../engine/playSettings';
import { useSessionStore } from './sessionStore';
import { useSongStore } from './songStore';

/** Save the current work before replacing it. A storage failure leaves it active. */
export function startNewSongFromForm(generation: SessionGenerationContext): void {
  const current = useSongStore.getState().song;
  const fresh: Song = {
    id: crypto.randomUUID(), title: DEFAULT_SONG_TITLE, autoTitle: true,
    key: current.key, genre: current.genre ?? 'pop', bpm: current.bpm,
    timeSignature: [...current.timeSignature], sections: [],
    measures: Array.from({ length: 4 }, (_, index) => ({
      id: crypto.randomUUID(), index, chords: [null, null],
    })),
  };
  const settings = JSON.stringify(playSettingsFromSession(generation));
  const alreadySaved = useSessionStore.getState().sessions.some(saved =>
    JSON.stringify(saved.song) === JSON.stringify(current) &&
    saved.generation.model === generation.model &&
    saved.generation.genre === generation.genre &&
    JSON.stringify(playSettingsFromSession(saved.generation)) === settings);
  if (!alreadySaved) useSessionStore.getState().add(current, generation, current.title);
  useSongStore.getState().setSong(fresh);
}
