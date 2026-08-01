import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import type { Measure, Song, SongSection } from '../models/types.ts';
import {
  SECTION_INFO,
  SECTION_TYPES,
  MAX_SECTION_BARS,
  autoSection,
  autoTitle,
  barsForChords,
  barCountSummary,
  checkBarCount,
  checkSectionRanges,
  defaultSectionBars,
  sectionAt,
  sectionContaining,
  sectionPlacement,
  sectionTypeByPosition,
  sectionTitle,
  sectionDisplayLabel,
  sectionsInGroup,
  withAutoTitle,
  withSongFormDefaults,
} from './songForm.ts';

function measures(count: number, withChords: number[] = []): Measure[] {
  return Array.from({ length: count }, (_, i) => ({
    id: `m${i}`,
    index: i,
    chords: withChords.includes(i)
      ? [{ root: 'C', quality: 'maj7', romanNumeral: '', voicing: [], extensions: [] }, null]
      : [null, null],
  }));
}

function song(overrides: Partial<Song> = {}): Song {
  return {
    id: 'song',
    title: 'Untitled',
    bpm: 80,
    key: 'C major',
    genre: null,
    timeSignature: [4, 4],
    measures: measures(4),
    sections: [],
    autoTitle: true,
    ...overrides,
  };
}

function section(type: SongSection['type'], startMeasure: number, length: number): SongSection {
  return { id: `${type}-${startMeasure}`, type, startMeasure, length };
}

test('imported uncertain sections retain their label and uncertainty after loading', () => {
  const region: SongSection = { id: 'region', type: null, label: 'Repeated region', origin: 'inferred', startMeasure: 0, length: 4 };
  const loaded = withSongFormDefaults(song({ sections: [region], title: 'Imported', autoTitle: false }));
  assert.deepEqual(loaded.sections, [region]);
  assert.equal(sectionDisplayLabel(region), 'Repeated region');
  assert.equal(sectionContaining(loaded, [0, 3])?.type, null);
  assert.equal(loaded.title, 'Imported');
});

test('editing section boundaries rejects overlaps, fractions and out-of-score ranges', () => {
  assert.equal(checkSectionRanges([section('verse', 0, 2), section('chorus', 2, 2)], 4), null);
  assert.match(checkSectionRanges([section('verse', 0, 3), section('chorus', 2, 2)], 4)!, /overlap/);
  assert.match(checkSectionRanges([section('verse', 2, 3)], 4)!, /last bar/);
  assert.match(checkSectionRanges([section('verse', 0.5, 2)], 4)!, /valid/);
  assert.match(checkSectionRanges([section('verse', 0, 0)], 4)!, /valid/);
});

test('vocabulary: 17 ids in three groups', () => {
  assert.equal(SECTION_TYPES.length, 17);
  assert.deepEqual(sectionsInGroup('common'), ['intro', 'verse', 'pre_chorus', 'chorus', 'post_chorus', 'bridge', 'outro']);
  assert.deepEqual(sectionsInGroup('instrumental'), ['interlude', 'solo', 'breakdown', 'drop', 'vamp', 'turnaround', 'tag']);
  assert.deepEqual(sectionsInGroup('jazz'), ['head', 'a_section', 'b_section']);
  assert.equal(MAX_SECTION_BARS, 64);
});

test('vocabulary matches backend/app/models/sections.py', (t) => {
  const pyPath = new URL('../../../backend/app/models/sections.py', import.meta.url);
  if (!existsSync(pyPath)) {
    t.skip('backend checkout not present');
    return;
  }
  const source = readFileSync(pyPath, 'utf8');
  const entries = [...source.matchAll(/^\s*"(\w+)": \("([^"]+)", "(\w+)", (\d+)\),$/gm)];
  const backend = Object.fromEntries(
    entries.map(([, id, label, group, chords]) => [id, { label, group, defaultChords: Number(chords) }]),
  );
  assert.deepEqual(backend, SECTION_INFO);
  assert.deepEqual(entries.map(([, id]) => id), SECTION_TYPES);
});

test('barsForChords: two chords per bar, rounded up', () => {
  assert.equal(barsForChords(16), 8);
  assert.equal(barsForChords(15), 8);
  assert.equal(barsForChords(2), 1);
  assert.equal(barsForChords(128), 64);
});

test('sectionTitle: em dash with spaces, singular bar', () => {
  assert.equal(sectionTitle('chorus', 8), 'Chorus — 8 bars');
  assert.equal(sectionTitle('pre_chorus', 4), 'Pre-Chorus — 4 bars');
  assert.equal(sectionTitle('vamp', 1), 'Vamp — 1 bar');
});

test('autoTitle: none, one, several, and truncation after six labels', () => {
  assert.equal(autoTitle([]), 'Untitled');
  assert.equal(autoTitle([section('chorus', 0, 8)]), 'Chorus — 8 bars');
  assert.equal(
    autoTitle([section('intro', 0, 4), section('verse', 4, 8), section('chorus', 12, 8)]),
    'Intro–Verse–Chorus',
  );
  // Labels follow song order, not insertion order.
  assert.equal(autoTitle([section('chorus', 8, 8), section('verse', 0, 8)]), 'Verse–Chorus');
  const seven = ['intro', 'verse', 'pre_chorus', 'chorus', 'verse', 'chorus', 'outro'] as const;
  const sections = seven.map((type, i) => section(type, i * 8, 8));
  assert.equal(autoTitle(sections), 'Intro–Verse–Pre-Chorus–Chorus–Verse–Chorus…');
  assert.equal(autoTitle(sections.slice(0, 6)), 'Intro–Verse–Pre-Chorus–Chorus–Verse–Chorus');
});

test('withAutoTitle only rewrites automatic titles', () => {
  const sections = [section('bridge', 0, 8)];
  assert.equal(withAutoTitle(song({ sections, autoTitle: true })).title, 'Bridge — 8 bars');
  assert.equal(withAutoTitle(song({ sections, title: 'Mine', autoTitle: false })).title, 'Mine');
});

test('sectionPlacement: empty 4-bar song starts at 0 and grows as needed', () => {
  assert.deepEqual(sectionPlacement(song(), 8), { startMeasure: 0, measuresToAdd: 4 });
  assert.deepEqual(sectionPlacement(song(), 4), { startMeasure: 0, measuresToAdd: 0 });
  assert.deepEqual(sectionPlacement(song(), 2), { startMeasure: 0, measuresToAdd: 0 });
});

test('sectionPlacement: starts after the last measure with a chord', () => {
  const s = song({ measures: measures(4, [0, 1]) });
  assert.deepEqual(sectionPlacement(s, 2), { startMeasure: 2, measuresToAdd: 0 });
  assert.deepEqual(sectionPlacement(s, 4), { startMeasure: 2, measuresToAdd: 2 });
  // A gap before the last chord is not reused.
  assert.deepEqual(sectionPlacement(song({ measures: measures(4, [2]) }), 1), { startMeasure: 3, measuresToAdd: 0 });
});

test('sectionPlacement: an existing section counts as used even without chords', () => {
  const s = song({ measures: measures(8), sections: [section('verse', 0, 8)] });
  assert.deepEqual(sectionPlacement(s, 8), { startMeasure: 8, measuresToAdd: 8 });
  const withChordsAfter = song({ measures: measures(12, [9]), sections: [section('verse', 0, 8)] });
  assert.deepEqual(sectionPlacement(withChordsAfter, 4), { startMeasure: 10, measuresToAdd: 2 });
});

test('sectionAt and sectionContaining', () => {
  const s = song({ measures: measures(16), sections: [section('intro', 0, 4), section('chorus', 4, 8)] });
  assert.equal(sectionAt(s, 0)?.type, 'intro');
  assert.equal(sectionAt(s, 3)?.type, 'intro');
  assert.equal(sectionAt(s, 4)?.type, 'chorus');
  assert.equal(sectionAt(s, 11)?.type, 'chorus');
  assert.equal(sectionAt(s, 12), null);
  assert.equal(sectionAt(song({ sections: undefined }), 0), null);
  assert.equal(sectionContaining(s, [5, 6, 7])?.type, 'chorus');
  assert.equal(sectionContaining(s, [3, 4]), null);
  assert.equal(sectionContaining(s, [10, 12]), null);
  assert.equal(sectionContaining(s, []), null);
});

const range = (from: number, to: number) => Array.from({ length: to - from + 1 }, (_, i) => from + i);

test('sectionTypeByPosition: where the bars sit in the whole song gives their role', () => {
  // The LBD frame: bars 5-12 of 16 end at 12/16 of the song, a chorus
  assert.equal(sectionTypeByPosition(16, range(4, 11)), 'chorus');
  assert.equal(sectionTypeByPosition(16, range(0, 15)), 'chorus');       // the whole song
  assert.equal(sectionTypeByPosition(16, range(0, 2)), 'intro');         // opens and ends by 3/16
  assert.equal(sectionTypeByPosition(16, range(0, 3)), 'verse');         // opens but runs past a fifth
  assert.equal(sectionTypeByPosition(16, range(12, 15)), 'outro');       // reaches the last bar
  assert.equal(sectionTypeByPosition(16, range(6, 8)), 'bridge');        // ends at 9/16
  assert.equal(sectionTypeByPosition(16, range(2, 4)), 'verse');         // ends at 5/16
  assert.equal(sectionTypeByPosition(10, range(1, 8)), 'chorus');        // eight of ten bars: most of the song
  assert.equal(sectionTypeByPosition(16, [11, 4, 4, 7]), 'chorus');      // order and repeats do not matter
  assert.equal(sectionTypeByPosition(16, []), null);
  assert.equal(sectionTypeByPosition(0, [0]), null);
});

test('autoSection: the section holding the bars, else their place in the whole song', () => {
  const sectioned = song({ measures: measures(16), sections: [section('intro', 0, 4), section('verse', 4, 8)] });
  assert.deepEqual(autoSection(sectioned, range(5, 7)), { type: 'verse', source: 'section' });
  // Bars that cross a section boundary get no rules: into the next section, or out into uncovered bars
  assert.deepEqual(autoSection(sectioned, range(3, 4)), { type: null, source: null });
  assert.deepEqual(autoSection(sectioned, range(10, 13)), { type: null, source: null });
  // Bars no section covers, in a song that has sections elsewhere: their place in the song (13-16 of 16)
  assert.deepEqual(autoSection(sectioned, range(12, 15)), { type: 'outro', source: 'position' });
  // A song without sections: the LBD frame's middle reads as a chorus
  assert.deepEqual(autoSection(song({ measures: measures(16) }), range(4, 11)), { type: 'chorus', source: 'position' });
  // An imported region whose role is unknown is read by position too
  const region: SongSection = { id: 'region', type: null, startMeasure: 0, length: 16 };
  assert.deepEqual(autoSection(song({ measures: measures(16), sections: [region] }), range(4, 11)),
    { type: 'chorus', source: 'position' });
  assert.deepEqual(autoSection(sectioned, []), { type: null, source: null });
});

test('withSongFormDefaults: old songs load with no sections and a derived autoTitle', () => {
  const legacy = (title: string) => {
    const s = song({ title });
    delete s.sections;
    delete s.autoTitle;
    return withSongFormDefaults(s);
  };
  assert.deepEqual(legacy('Untitled').sections, []);
  assert.equal(legacy('Untitled').autoTitle, true);
  assert.equal(legacy('').autoTitle, true);
  assert.equal(legacy('Blue Bossa').autoTitle, false);
  // Stored values win; malformed sections are dropped.
  assert.equal(withSongFormDefaults(song({ title: 'Untitled', autoTitle: false })).autoTitle, false);
  const bad = [section('verse', 0, 8), { id: 'x', type: 'nope', startMeasure: 0, length: 4 }] as SongSection[];
  assert.deepEqual(withSongFormDefaults(song({ sections: bad })).sections, [section('verse', 0, 8)]);
});

test('withSongFormDefaults: a section still pending when saved goes, with the measures added for it', () => {
  const pendingSection = (
    type: SongSection['type'], startMeasure: number, length: number, addedMeasures: number,
  ): SongSection => ({ ...section(type, startMeasure, length), pending: { addedMeasures } });

  // The empty 4-bar song right after New from Song Form placed an 8-bar
  // Verse (4 measures appended), autosaved before the chords came back.
  const onEmpty = withSongFormDefaults(song({
    title: 'Verse — 8 bars',
    measures: measures(8),
    sections: [pendingSection('verse', 0, 8, 4)],
  }));
  assert.deepEqual(onEmpty.sections, []);
  assert.equal(onEmpty.measures.length, 4);
  assert.equal(onEmpty.title, 'Untitled');

  // A kept section stays, and the automatic title stops naming the pending one.
  const intro = section('intro', 0, 4);
  const afterIntro = withSongFormDefaults(song({
    title: 'Intro–Chorus',
    measures: measures(12, [0, 1, 2, 3]),
    sections: [intro, pendingSection('chorus', 4, 8, 8)],
  }));
  assert.deepEqual(afterIntro.sections, [intro]);
  assert.equal(afterIntro.measures.length, 4);
  assert.equal(afterIntro.title, 'Intro — 4 bars');

  // Placed over blank measures the song already had: those stay.
  const intoBlank = withSongFormDefaults(song({
    title: 'Verse — 4 bars',
    measures: measures(8, [0, 1]),
    sections: [pendingSection('verse', 2, 4, 0)],
  }));
  assert.deepEqual(intoBlank.sections, []);
  assert.equal(intoBlank.measures.length, 8);
  assert.equal(intoBlank.title, 'Untitled');

  // A typed title is kept, and a measure with a chord is never dropped.
  const typed = withSongFormDefaults(song({
    title: 'Blue Bossa',
    autoTitle: false,
    measures: measures(8, [6]),
    sections: [pendingSection('verse', 0, 8, 4)],
  }));
  assert.equal(typed.title, 'Blue Bossa');
  assert.equal(typed.measures.length, 7);

  // An unreadable pending record still drops the section, but no measures.
  const unreadable = { ...section('verse', 0, 8), pending: true } as unknown as SongSection;
  const odd = withSongFormDefaults(song({ measures: measures(8), sections: [unreadable] }));
  assert.deepEqual(odd.sections, []);
  assert.equal(odd.measures.length, 8);

  // Nothing pending: the song loads as saved, trailing blank measures and a
  // never-set '' title included.
  const asSaved = withSongFormDefaults(song({ title: '', measures: measures(12) }));
  assert.equal(asSaved.measures.length, 12);
  assert.equal(asSaved.title, '');
});

test('checkBarCount: whole numbers of bars from 1 to 64, two chords each', () => {
  assert.deepEqual(checkBarCount('8'), { ok: true, bars: 8, count: 16 });
  assert.deepEqual(checkBarCount(' 1 '), { ok: true, bars: 1, count: 2 });
  assert.deepEqual(checkBarCount('64'), { ok: true, bars: 64, count: 128 });
  for (const bad of ['', 'abc', '0', '65', '1.5', '-4', '4 4']) {
    assert.equal(checkBarCount(bad).ok, false, `expected '${bad}' to be rejected`);
  }
});

test('barCountSummary and the default bars of a section', () => {
  assert.equal(barCountSummary(8), '16 chords, two per bar');
  assert.equal(defaultSectionBars('verse'), 8);
  assert.equal(defaultSectionBars('vamp'), 2);
  assert.equal(defaultSectionBars('head'), 32);
});
