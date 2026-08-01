/**
 * SessionStore — saved snapshots of song + generation context.
 *
 * Each session is a full Song + the Generate-panel settings the user
 * had at save time. Sessions can be loaded back into the active
 * songStore (replacing the working song), starred (multi-star OK), and
 * deleted. Starred sessions form the pool that Favorite Generate
 * iterates over.
 *
 * Persistence lives behind `sessionStorage.ts` so a future swap to
 * IndexedDB / cloud needs no changes here.
 */
import { create } from 'zustand';
import type { SavedSession, SessionGenerationContext, Song } from '../models/types';
import { computeArrangement } from '../engine/arrangement';
import { loadAll, removeSavedSessions, saveAll } from './sessionStorage';
import { withoutImportedContent } from '../engine/importRetention';

interface SessionState {
  sessions: SavedSession[];

  /** Add a new session from a song + generation context. Returns the
   *  freshly created entry (caller may want its id for UI feedback). */
  add: (song: Song, generation: SessionGenerationContext, name?: string) => SavedSession;
  remove: (id: string) => void;
  /** Delete several sessions with one storage write, or none when no id
   *  matches. The list changes only after that write succeeds; a failed
   *  write throws and leaves it as is. */
  removeMany: (ids: readonly string[]) => void;
  rename: (id: string, name: string) => void;
  toggleStar: (id: string) => void;
  /** Replace a session's song + generation snapshot (used when the user
   *  loads a session, edits, and re-saves into the same slot). */
  update: (id: string, patch: { song?: Song; generation?: SessionGenerationContext }) => void;
}

function autoName(song: Song, generation: SessionGenerationContext): string {
  const genrePart = generation.genre ? ` · ${generation.genre}` : '';
  // Pearl 2026-05-10: keep the song title literal — including "Untitled" —
  // instead of substituting song.key. Users want to see "Untitled" as-is so
  // they can tell the title was never set vs. assuming the key was the title.
  const titlePart = song.title || 'Untitled';
  return `${titlePart}${genrePart}`;
}

function persist(sessions: SavedSession[]): SavedSession[] {
  const cleaned = sessions.map((session) => ({ ...session, song: withoutImportedContent(session.song) }));
  saveAll(cleaned);
  return cleaned;
}

export const useSessionStore = create<SessionState>((set, get) => ({
  sessions: loadAll(),

  add: (song, generation, name) => {
    // Pearl 2026-05-10: pre-compute multi-track arrangement (bass + melody)
    // at save time so MyPage Play button can schedule instantly without any
    // per-play computation. Drum & harmony stay rule-based at play time
    // (rhythm patterns + chord block) — only bass + melody are pre-computed
    // here because they depend on the song's chord progression.
    const songCopy = withoutImportedContent(structuredClone(song));
    // Genre on the song drives the arrangement rhythm tables; in case the
    // user's session.generation.genre differs, prefer that for arrangement.
    const songWithGenre = { ...songCopy, genre: generation.genre ?? songCopy.genre };
    // Pearl 2026-05-10: melody track dropped — arrangement now bundles bass
    // only. Harmony override travels in generation.harmonyOverride and is
    // applied at MyPage Play time via setHarmonyOverride.
    const arrangement = computeArrangement(songWithGenre);
    const session: SavedSession = {
      id: crypto.randomUUID(),
      name: name ?? autoName(song, generation),
      createdAt: Date.now(),
      song: songCopy,
      generation: { ...generation },
      starred: false,
      arrangement,
    };
    set({ sessions: persist([session, ...get().sessions]) });
    return session;
  },

  remove: (id) => get().removeMany([id]),

  removeMany: (ids) => {
    const current = get().sessions;
    const remaining = removeSavedSessions(current, ids);
    if (remaining !== current) set({ sessions: remaining });
  },

  rename: (id, name) => {
    set({
      sessions: persist(
        get().sessions.map((s) => (s.id === id ? { ...s, name } : s)),
      ),
    });
  },

  toggleStar: (id) => {
    set({
      sessions: persist(
        get().sessions.map((s) =>
          s.id === id ? { ...s, starred: !s.starred } : s,
        ),
      ),
    });
  },

  update: (id, patch) => {
    set({
      sessions: persist(
        get().sessions.map((s) => {
          if (s.id !== id) return s;
          const newSong = patch.song ? withoutImportedContent(structuredClone(patch.song)) : s.song;
          const newGeneration = patch.generation ? { ...patch.generation } : s.generation;
          // Re-compute arrangement when song or generation changed (genre
          // drives the rhythm pattern). Cheap (~ms for typical 16-measure song).
          const arrangement = (patch.song || patch.generation)
            ? computeArrangement({ ...newSong, genre: newGeneration.genre ?? newSong.genre })
            : s.arrangement;
          return { ...s, song: newSong, generation: newGeneration, arrangement };
        }),
      ),
    });
  },
}));

/** Selector helper: just the starred sessions, in saved order. */
export const selectStarredSessions = (state: SessionState): SavedSession[] =>
  state.sessions.filter((s) => s.starred);
