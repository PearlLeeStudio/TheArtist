/**
 * Song-form sections: the vocabulary shared with the backend
 * (backend/app/models/sections.py — keep the two lists identical) plus the
 * pure helpers the stores and the score view use to title, place and look
 * up sections.
 *
 * Type-only imports on purpose: `node --test` runs songForm.test.ts against
 * this file directly, without a bundler.
 */
import type { Song, SongSection, SectionType } from '../models/types';

export type SectionGroup = 'common' | 'instrumental' | 'jazz';

export interface SectionInfo {
  label: string;
  group: SectionGroup;
  /** Chord count the "Build from Song Form" dialog starts from. */
  defaultChords: number;
}

export const SECTION_INFO: Record<SectionType, SectionInfo> = {
  intro:       { label: 'Intro',       group: 'common',       defaultChords: 8 },
  verse:       { label: 'Verse',       group: 'common',       defaultChords: 16 },
  pre_chorus:  { label: 'Pre-Chorus',  group: 'common',       defaultChords: 8 },
  chorus:      { label: 'Chorus',      group: 'common',       defaultChords: 16 },
  post_chorus: { label: 'Post-Chorus', group: 'common',       defaultChords: 8 },
  bridge:      { label: 'Bridge',      group: 'common',       defaultChords: 16 },
  outro:       { label: 'Outro',       group: 'common',       defaultChords: 8 },
  interlude:   { label: 'Interlude',   group: 'instrumental', defaultChords: 8 },
  solo:        { label: 'Solo',        group: 'instrumental', defaultChords: 16 },
  breakdown:   { label: 'Breakdown',   group: 'instrumental', defaultChords: 16 },
  drop:        { label: 'Drop',        group: 'instrumental', defaultChords: 16 },
  vamp:        { label: 'Vamp',        group: 'instrumental', defaultChords: 4 },
  turnaround:  { label: 'Turnaround',  group: 'instrumental', defaultChords: 4 },
  tag:         { label: 'Tag',         group: 'instrumental', defaultChords: 8 },
  head:        { label: 'Head',        group: 'jazz',         defaultChords: 64 },
  a_section:   { label: 'A Section',   group: 'jazz',         defaultChords: 16 },
  b_section:   { label: 'B Section',   group: 'jazz',         defaultChords: 16 },
};

/** Every section id, in dropdown order. */
export const SECTION_TYPES = Object.keys(SECTION_INFO) as SectionType[];

/** Dropdown <optgroup>s, in order. */
export const SECTION_GROUPS: ReadonlyArray<{ id: SectionGroup; label: string }> = [
  { id: 'common', label: 'Common' },
  { id: 'instrumental', label: 'Instrumental' },
  { id: 'jazz', label: 'Jazz' },
];

/** The editor has two chord slots per measure. */
export const CHORDS_PER_BAR = 2;
/** Cap on one generate request on every path: 128 chords = 64 bars. */
export const MAX_SECTION_CHORDS = 128;
export const MAX_SECTION_BARS = MAX_SECTION_CHORDS / CHORDS_PER_BAR;

/** Title of a song with no sections that nobody has named. */
export const DEFAULT_SONG_TITLE = 'Untitled';
/** autoTitle lists at most this many section labels, then an ellipsis. */
const AUTO_TITLE_MAX_LABELS = 6;

export function isSectionType(value: unknown): value is SectionType {
  return typeof value === 'string' && Object.hasOwn(SECTION_INFO, value);
}

export function sectionLabel(type: SectionType | null): string {
  return type === null ? 'Section' : SECTION_INFO[type].label;
}

export function sectionDisplayLabel(section: SongSection): string {
  return section.label?.trim() || sectionLabel(section.type);
}

/** Index of the first measure after `section`. */
function sectionEnd(section: SongSection): number {
  return section.startMeasure + section.length;
}

/** Keep imported regions editable without pretending their musical role is known. */
export function checkSectionRanges(sections: readonly SongSection[], measureCount: number): string | null {
  const ids = new Set<string>();
  let previousEnd = 0;
  for (const section of [...sections].sort((a, b) => a.startMeasure - b.startMeasure)) {
    if (!isValidSection(section) || ids.has(section.id)) return 'Each section needs a valid, unique range.';
    ids.add(section.id);
    if (sectionEnd(section) > measureCount) return 'A section extends past the last bar.';
    if (section.startMeasure < previousEnd) return 'Sections must not overlap.';
    if (section.label !== undefined && (typeof section.label !== 'string' || section.label.length > 80)) {
      return 'Section names must be 80 characters or fewer.';
    }
    previousEnd = sectionEnd(section);
  }
  return null;
}

export function sectionColor(type: SectionType | null): string {
  if (type === 'chorus' || type === 'post_chorus' || type === 'drop') return '#d29b45';
  if (type === 'verse' || type === 'head' || type === 'a_section') return '#6a9fc2';
  if (type === 'bridge' || type === 'b_section' || type === 'solo') return '#a28abc';
  if (type === 'pre_chorus' || type === 'breakdown' || type === 'turnaround') return '#b58077';
  if (type === null) return '#929292';
  return '#79a58a';
}

export function sectionsInGroup(group: SectionGroup): SectionType[] {
  return SECTION_TYPES.filter((type) => SECTION_INFO[type].group === group);
}

export function barsForChords(chordCount: number): number {
  return Math.ceil(chordCount / CHORDS_PER_BAR);
}

/** '1 bar', '8 bars'. */
function barsLabel(bars: number): string {
  return `${bars} ${bars === 1 ? 'bar' : 'bars'}`;
}

/** Single-section title, e.g. 'Chorus — 8 bars' (em dash, as the backend writes it). */
export function sectionTitle(type: SectionType, bars: number): string {
  return `${sectionLabel(type)} — ${barsLabel(bars)}`;
}

/**
 * Title of a song built from sections. One section gives its sectionTitle;
 * several give their labels in song order joined by en dashes, without
 * spaces ('Intro–Verse–Chorus'), at most six labels and then an ellipsis.
 * No sections gives DEFAULT_SONG_TITLE.
 */
export function autoTitle(sections: readonly SongSection[]): string {
  if (sections.length === 0) return DEFAULT_SONG_TITLE;
  const ordered = [...sections].sort((a, b) => a.startMeasure - b.startMeasure);
  if (ordered.length === 1) {
    const only = ordered[0];
    return `${sectionDisplayLabel(only)} — ${barsLabel(only.length)}`;
  }
  const labels = ordered.slice(0, AUTO_TITLE_MAX_LABELS).map(sectionDisplayLabel);
  // U+2013 en dash between labels, U+2026 ellipsis when some are left out.
  return labels.join('–') + (ordered.length > AUTO_TITLE_MAX_LABELS ? '…' : '');
}

/** While the song's title is automatic, recompute it from its sections. */
export function withAutoTitle(song: Song): Song {
  return song.autoTitle ? { ...song, title: autoTitle(song.sections ?? []) } : song;
}

/**
 * Where a new section of `bars` measures goes: right after the last measure
 * that has any chord or belongs to a section (measure 0 on an empty song),
 * and how many measures must be appended for it to fit.
 */
export function sectionPlacement(
  song: Pick<Song, 'measures' | 'sections'>,
  bars: number,
): { startMeasure: number; measuresToAdd: number } {
  let lastUsed = -1;
  song.measures.forEach((measure, i) => {
    if (measure.chords.some(Boolean)) lastUsed = i;
  });
  for (const section of song.sections ?? []) {
    lastUsed = Math.max(lastUsed, sectionEnd(section) - 1);
  }
  const startMeasure = lastUsed + 1;
  return {
    startMeasure,
    measuresToAdd: Math.max(0, startMeasure + bars - song.measures.length),
  };
}

/** The section covering `measureIndex`, or null. */
export function sectionAt(
  song: Pick<Song, 'sections'>,
  measureIndex: number,
): SongSection | null {
  return (song.sections ?? []).find(
    (s) => measureIndex >= s.startMeasure && measureIndex < sectionEnd(s),
  ) ?? null;
}

/** The section that contains every one of `measureIndices`, or null. */
export function sectionContaining(
  song: Pick<Song, 'sections'>,
  measureIndices: readonly number[],
): SongSection | null {
  if (measureIndices.length === 0) return null;
  const section = sectionAt(song, Math.min(...measureIndices));
  if (!section) return null;
  return Math.max(...measureIndices) < sectionEnd(section) ? section : null;
}

/** The section type a passage's place in the whole song suggests: the song is a
 *  time series of `totalMeasures` bars, and where the passage starts and ends in
 *  it decides its role. This is the editor's original rule (in use until the
 *  2026-11-04 merge), unchanged: bars 5 to 12 of 16 read as a chorus. */
export function sectionTypeByPosition(
  totalMeasures: number,
  measureIndices: readonly number[],
): SectionType | null {
  const measures = [...new Set(measureIndices)];
  if (measures.length === 0 || totalMeasures <= 0) return null;
  const maxSelected = Math.max(...measures);
  const minSelected = Math.min(...measures);
  const endsAtLast = maxSelected >= totalMeasures - 1;
  const startsAtFirst = minSelected === 0;
  const coversMost = measures.length >= totalMeasures * 0.7;
  const endRatio = (maxSelected + 1) / totalMeasures;
  if (coversMost || (startsAtFirst && endsAtLast)) return 'chorus';
  if (startsAtFirst && endRatio <= 0.2) return 'intro';
  if (endsAtLast) return 'outro';
  if (endRatio >= 0.7) return 'chorus';
  if (endRatio >= 0.45) return 'bridge';
  return 'verse';
}

/** What the Generate dropdown's Auto resolves to, and where it came from. */
export interface AutoSection {
  type: SectionType | null;
  /** 'section': the song's section holding the bars; 'position': their place in
   *  the whole song; null: no song-form rules. */
  source: 'section' | 'position' | null;
}

/** Auto: the type of the section holding every selected bar; when no section
 *  covers them, or the one that does has no role yet (an imported region), their
 *  place in the whole song. Bars that cross a section boundary (into another
 *  section or out into uncovered bars) get no rules. */
export function autoSection(
  song: Pick<Song, 'sections' | 'measures'>,
  measureIndices: readonly number[],
): AutoSection {
  if (measureIndices.length === 0) return { type: null, source: null };
  const containing = sectionContaining(song, measureIndices);
  if (containing?.type) return { type: containing.type, source: 'section' };
  const overlaps = (song.sections ?? []).some(
    (s) => measureIndices.some((i) => i >= s.startMeasure && i < sectionEnd(s)),
  );
  if (overlaps && !containing) return { type: null, source: null };
  const type = sectionTypeByPosition(song.measures.length, measureIndices);
  return { type, source: type ? 'position' : null };
}

function isValidSection(value: unknown): value is SongSection {
  if (typeof value !== 'object' || value === null) return false;
  const s = value as Record<string, unknown>;
  return (
    typeof s.id === 'string' &&
    (s.type === null || isSectionType(s.type)) &&
    Number.isInteger(s.startMeasure) && (s.startMeasure as number) >= 0 &&
    Number.isInteger(s.length) && (s.length as number) >= 1
  );
}

/** Measures appended for a pending section; 0 when the record is unreadable. */
function addedMeasures(section: SongSection): number {
  const count = section.pending?.addedMeasures;
  return typeof count === 'number' && count > 0 ? Math.floor(count) : 0;
}

/**
 * Take back a placement whose generation never finished (the page was
 * reloaded or closed while "New from Song Form" ran): drop the pending
 * sections and the measures appended only for them, i.e. at most
 * `addedMeasures` trailing measures that have no chord and no other
 * section. An automatic title then stops naming them.
 */
function withoutPendingSections(song: Song): Song {
  const all = song.sections ?? [];
  const sections = all.filter((s) => !s.pending);
  if (sections.length === all.length) return song;
  const added = all.reduce((sum, s) => sum + addedMeasures(s), 0);
  const floor = Math.max(0, song.measures.length - added);
  let end = song.measures.length;
  while (
    end > floor &&
    !song.measures[end - 1].chords.some(Boolean) &&
    !sectionAt({ sections }, end - 1)
  ) {
    end--;
  }
  return withAutoTitle({ ...song, measures: song.measures.slice(0, end), sections });
}

/**
 * Fill in the song-form fields that songs saved before sections existed
 * lack: no sections, and an automatic title only when the title was never
 * set ('' or 'Untitled'). A section still pending when the song was saved
 * is taken back (see withoutPendingSections).
 */
export function withSongFormDefaults(song: Song): Song {
  const sections = Array.isArray(song.sections) ? song.sections.filter(isValidSection) : [];
  const title = (song.title ?? '').trim();
  const auto = typeof song.autoTitle === 'boolean'
    ? song.autoTitle
    : title === '' || title === DEFAULT_SONG_TITLE;
  return withoutPendingSections({ ...song, sections, autoTitle: auto });
}

/** `count` is the chords the section API takes, two per bar. */
export type BarCountCheck =
  | { ok: true; bars: number; count: number }
  | { ok: false; message: string };

/** Bars a section type starts with in the dialog: its default chord count, two per bar. */
export function defaultSectionBars(type: SectionType): number {
  return barsForChords(SECTION_INFO[type].defaultChords);
}

/** Validate the dialog's bar-count text: a whole number of bars, 1..64. */
export function checkBarCount(text: string): BarCountCheck {
  const trimmed = text.trim();
  if (!/^\d+$/.test(trimmed)) {
    return { ok: false, message: `Enter a number of bars from 1 to ${MAX_SECTION_BARS}` };
  }
  const bars = Number(trimmed);
  if (bars < 1 || bars > MAX_SECTION_BARS) {
    return { ok: false, message: `Use 1 to ${MAX_SECTION_BARS} bars` };
  }
  return { ok: true, bars, count: bars * CHORDS_PER_BAR };
}

/** Live hint for a bar count, e.g. '16 chords, two per bar'. */
export function barCountSummary(bars: number): string {
  return `${bars * CHORDS_PER_BAR} chords, two per bar`;
}
