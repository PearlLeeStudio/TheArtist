import type { Chord } from '../models/types';

export interface ManualChordDraft {
  root: string;
  base: string;
  ext: string;
  add: string;
  sus: string;
  alts: string[];
  bass: string;
  /** Preserve spellings the builder cannot express until a quality control changes. */
  originalQuality?: string;
}

export function resolveQuality(base: string, ext: string, add: string, sus: string): string {
  if (sus) return ext ? `${ext}${sus}` : sus;
  if (add) return base === 'min' ? `m(${add})` : add;
  if (base === '5') return ext === '7' ? '7(no3)' : '5';

  if (base === 'min') {
    if (ext === 'maj7') return 'mMaj7';
    if (ext === '7') return 'm7';
    if (ext === '9') return 'm9';
    if (ext === '11') return 'm11';
    if (ext === '13') return 'm13';
    if (ext === '6') return 'm6';
    if (ext === '') return 'min';
    return `m${ext}`;
  }
  if (base === 'dim') {
    if (ext === '7') return 'dim7';
    if (ext === '') return 'dim';
    return 'm7b5';
  }
  if (base === 'aug') return ext === '7' ? 'aug7' : 'aug';

  if (ext === '7') return '7';
  if (ext) return ext;
  return 'maj';
}

/** Read the selected symbol without substituting C major or losing its modifiers. */
export function manualChordDraft(chord: Chord | null): ManualChordDraft {
  const draft: ManualChordDraft = {
    root: chord?.root ?? 'C', base: chord ? '' : 'maj', ext: '', add: '', sus: '',
    alts: [...(chord?.extensions ?? [])], bass: chord?.bass ?? '',
    originalQuality: chord?.quality,
  };
  if (!chord) return draft;

  for (const base of ['maj', 'min', 'dim', 'aug', '5']) {
    const extensions = base === 'maj' || base === 'min'
      ? ['', '6', '7', 'maj7', '9', 'maj9', '11', '13'] : ['', '7'];
    for (const ext of extensions) {
      if (resolveQuality(base, ext, '', '') === chord.quality) return { ...draft, base, ext };
    }
  }
  for (const add of ['add2', 'add4', 'add9', 'add11']) {
    for (const base of ['maj', 'min']) {
      if (resolveQuality(base, '', add, '') === chord.quality) return { ...draft, base, add };
    }
  }
  for (const sus of ['sus2', 'sus4']) {
    for (const ext of ['', '6', '7', 'maj7', '9', 'maj9', '11', '13']) {
      if (resolveQuality('maj', ext, '', sus) === chord.quality) return { ...draft, base: 'maj', ext, sus };
    }
  }
  return draft;
}

export function manualDraftChord(draft: ManualChordDraft): Chord {
  return {
    root: draft.root,
    quality: draft.originalQuality ?? resolveQuality(draft.base, draft.ext, draft.add, draft.sus),
    bass: draft.bass || undefined,
    extensions: [...draft.alts], romanNumeral: '', voicing: [],
  };
}
