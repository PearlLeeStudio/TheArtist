import { test } from 'node:test';
import assert from 'node:assert/strict';
import type { SavedSession } from '../models/types.ts';
import { DEFAULT_PLAY_SETTINGS, enginePlayConfig, playSettingsFromSession } from '../engine/playSettings.ts';
import { saveAll, loadAll, removeSavedSessions } from './sessionStorage.ts';
import { readScoreImportResponse, createScoreImportDraft, buildImportedSong, buildScoreImportForEditor } from '../engine/scoreImport.ts';
import { withoutImportedContent } from '../engine/importRetention.ts';
import { useImportCleanupStore } from './importCleanupStore.ts';

interface StorageMemory { stored: string; writes: number; failWrites: boolean }

function withStoredProjects<T>(sessions: unknown[], run: (memory: StorageMemory) => T): T {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, 'localStorage');
  const memory: StorageMemory = { stored: JSON.stringify({ version: 1, sessions }), writes: 0, failWrites: false };
  Object.defineProperty(globalThis, 'localStorage', { configurable: true, value: {
    setItem(_key: string, value: string) {
      memory.writes++;
      if (memory.failWrites) throw new Error('quota');
      memory.stored = value;
    },
    getItem() { return memory.stored; },
  } });
  try { return run(memory); }
  finally {
    if (descriptor) Object.defineProperty(globalThis, 'localStorage', descriptor);
    else Reflect.deleteProperty(globalThis, 'localStorage');
  }
}

function reloadProjects(sessions: unknown[]): SavedSession[] {
  return withStoredProjects(sessions, () => loadAll());
}

function roundTripProjects(sessions: SavedSession[]): SavedSession[] {
  return withStoredProjects([], () => {
    saveAll(sessions);
    return loadAll();
  });
}

test('invalid saved projects are skipped while valid neighboring projects remain available', () => {
  const malformed = {
    id: 'bad', name: 'Bad', createdAt: 0, starred: false,
    song: {}, generation: { model: 'ft_f1', genre: null },
  };
  assert.deepEqual(reloadProjects([malformed, importedProject]), [importedProject]);
});

test('storage failure is reported to the caller instead of pretending the project was saved', () => {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, 'localStorage');
  Object.defineProperty(globalThis, 'localStorage', { configurable: true, value: { setItem() { throw new Error('quota'); } } });
  try { assert.throws(() => saveAll([]), /could not save/); }
  finally {
    if (descriptor) Object.defineProperty(globalThis, 'localStorage', descriptor);
    else Reflect.deleteProperty(globalThis, 'localStorage');
  }
});

/** A project as an earlier build saved it: its import source has no analysisSource
 *  and names that build's model and request. Such projects must stay loadable. */
const importedProject: SavedSession = {
  id: 'imported-project', name: '봄의 변형', createdAt: 0, starred: true,
  song: {
    id: 'imported-score', title: '봄의 변형', key: 'C major', genre: 'jazz', bpm: 90, timeSignature: [6, 8], autoTitle: false,
    measures: [{ id: 'bar-1', index: 0, chords: [
      { root: 'F', quality: 'maj', bass: 'C', romanNumeral: 'IV', voicing: [48, 53, 57], extensions: [] }, null,
    ] }],
    sections: [{ id: 'region-1', type: null, label: 'Repeated region', origin: 'inferred', startMeasure: 0, length: 1 }],
    importSource: { filename: '', model: 'muse-spark-1.3', requestId: 'sample-request' },
  },
  generation: { model: 'ft_f1', genre: 'jazz', ...DEFAULT_PLAY_SETTINGS, bassVolume: 0 },
  arrangement: { bassPattern: [0, 2], bassInstrument: 'contrabass', beatsPerMeasure: 6 },
};

test('browser save and reload retain real chords, uncertain sections, source and a muted part', () => {
  assert.deepEqual(roundTripProjects([importedProject])[0], importedProject);
});

test('bulk deletion saves once, ignores stale and duplicate selections, and preserves unselected projects', () => {
  const sessions = [importedProject, { ...structuredClone(importedProject), id: 'second' },
    { ...structuredClone(importedProject), id: 'third' }];
  withStoredProjects(sessions, (memory) => {
    const remaining = removeSavedSessions(sessions, ['second', 'stale-id', 'second']);
    assert.equal(memory.writes, 1);
    assert.deepEqual(remaining, [sessions[0], sessions[2]]);
    assert.equal(remaining[0], sessions[0]);
    assert.equal(remaining[1], sessions[2]);
    assert.deepEqual(loadAll(), remaining);
    assert.equal(sessions.length, 3);
    assert.deepEqual(sessions[1].song.importSource, importedProject.song.importSource);
  });
});

test('selecting every saved id persists an empty list in one write', () => {
  const sessions = [importedProject, { ...structuredClone(importedProject), id: 'second' }];
  withStoredProjects(sessions, (memory) => {
    assert.deepEqual(removeSavedSessions(sessions, sessions.map((session) => session.id)), []);
    assert.equal(memory.writes, 1);
    assert.deepEqual(loadAll(), []);
    assert.deepEqual(JSON.parse(memory.stored), { version: 1, sessions: [] });
  });
});

test('empty or entirely stale selections do not write storage or replace the current list', () => {
  const sessions = [importedProject];
  withStoredProjects(sessions, (memory) => {
    assert.equal(removeSavedSessions(sessions, []), sessions);
    assert.equal(removeSavedSessions(sessions, ['missing-id']), sessions);
    assert.equal(memory.writes, 0);
    assert.deepEqual(loadAll(), sessions);
  });
});

test('a failed bulk deletion cannot publish a partial list or lose the previous saved projects', () => {
  const sessions = [importedProject, { ...structuredClone(importedProject), id: 'second' }];
  withStoredProjects(sessions, (memory) => {
    const previousStorage = memory.stored;
    let published = sessions;
    memory.failWrites = true;
    assert.throws(() => { published = removeSavedSessions(sessions, [sessions[0].id]); }, /could not save/);
    assert.equal(published, sessions);
    assert.equal(memory.writes, 1);
    assert.equal(memory.stored, previousStorage);
    assert.deepEqual(loadAll(), sessions);
    assert.equal(sessions.length, 2);
  });
});

test('note-derived chord proposals survive browser save while source evidence is discarded', () => {
  const project = structuredClone(importedProject);
  project.song.importSource = {
    filename: 'notes.pdf', model: 'muse-spark-1.3', requestId: 'notes-request', mode: 'infer_harmony',
    measures: [{ chordBasis: 'inferred', confidence: 'medium', evidence: 'Bass C and notes E G visible.', originalChords: ['C'] }],
  };
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, 'localStorage');
  let stored = '';
  Object.defineProperty(globalThis, 'localStorage', { configurable: true, value: {
    setItem(_key: string, value: string) { stored = value; }, getItem() { return stored; },
  } });
  try {
    saveAll([project]);
    const source = loadAll()[0].song.importSource!;
    assert.equal(source.filename, '');
    assert.equal(source.measures![0].evidence, '');
    assert.deepEqual(source.measures![0].originalChords, ['C']);
    assert.equal(source.measures![0].confidence, 'medium');
    assert.equal(source.requestId, 'notes-request');
    assert.doesNotMatch(stored, /notes\.pdf|Bass C and notes E G visible/);
  } finally {
    if (descriptor) Object.defineProperty(globalThis, 'localStorage', descriptor);
    else Reflect.deleteProperty(globalThis, 'localStorage');
  }
});

test('saved projects with invalid or disguised inferred provenance are skipped', () => {
  const validSource = {
    filename: 'notes.pdf', model: 'muse-spark-1.3', requestId: null, mode: 'infer_harmony',
    measures: [{ chordBasis: 'inferred', confidence: 'medium', evidence: 'A C E visible.', originalChords: ['Am'] }],
  };
  for (const source of [
    { ...validSource, mode: 'unknown' }, { ...validSource, measures: undefined },
    { ...validSource, mode: 'chord_symbols' },
    { ...validSource, measures: [{ ...validSource.measures[0], confidence: 'certain' }] },
    { ...validSource, measures: [{ ...validSource.measures[0], originalChords: [3] }] },
  ]) {
    const project = { ...importedProject, song: { ...importedProject.song, importSource: source } };
    assert.deepEqual(reloadProjects([project]), []);
  }
});

test('local and mixed reading status, rests and explicit empty choices survive without note evidence', () => {
  const project = structuredClone(importedProject);
  project.song.importSource = {
    filename: 'analysis.pdf', mode: 'infer_harmony', analysisSource: 'mixed', model: 'muse-spark-1.3', requestId: 'batch-request',
    measures: [
      { chordBasis: 'inferred', confidence: 'high', evidence: 'C E G over C.', originalChords: ['C'], sourceMeasureIndex: 0, analysisSource: 'local', model: null, requestId: null, notes: ['C4', 'E4', 'G4'], bassNotes: ['C3'], originalNoChord: false, userNoChord: true, timeSignature: [2, 4],
        editorReview: 'Check against the original: the readings differ.' },
      { chordBasis: 'inferred', confidence: 'medium', evidence: 'D F A supports D minor.', originalChords: ['Dm'], sourceMeasureIndex: 1, analysisSource: 'meta', model: 'muse-spark-1.3', requestId: 'passage-request', notes: ['D4', 'F4', 'A4'], bassNotes: ['D3'], originalNoChord: false, userNoChord: false, timeSignature: [4, 4],
        editorReview: 'Proposed from the original after import; the local reader had none.' },
    ],
  };
  const restored = roundTripProjects([project])[0].song.importSource!;
  assert.deepEqual(restored, withoutImportedContent(project.song).importSource);
  // A project saved by an earlier build keeps what that build stored: the sources,
  // model names and request ids of the import and of each bar, and the review wording.
  assert.deepEqual([restored.analysisSource, restored.model, restored.requestId], ['mixed', 'muse-spark-1.3', 'batch-request']);
  assert.deepEqual(restored.measures!.map((measure) => [measure.analysisSource, measure.model, measure.requestId, measure.editorReview]), [
    ['local', null, null, 'Check against the original: the readings differ.'],
    ['meta', 'muse-spark-1.3', 'passage-request', 'Proposed from the original after import; the local reader had none.'],
  ]);
  assert.doesNotMatch(JSON.stringify(restored), /analysis\.pdf|C E G over C|D F A supports|"notes"|"bassNotes"/);
  project.song.importSource.analysisSource = 'local';
  project.song.importSource.model = null;
  project.song.importSource.requestId = null;
  project.song.importSource.measures = [project.song.importSource.measures![0]];
  assert.deepEqual(roundTripProjects([project])[0].song.importSource, withoutImportedContent(project.song).importSource);
});

test('F1 import drafts and safe reading review flags survive save and reload in local and mixed scores', () => {
  const response = readScoreImportResponse({
    title: 'Sparse harmony', key: 'C major', timeSignature: [4, 4], sections: [], warnings: [],
    analysisSource: 'local', model: 'ft_f1', mode: 'infer_harmony',
    measures: [
      { chords: ['C'], sourceLabel: 'Bar 1', needsReview: true, chordBasis: 'printed', confidence: 'high',
        note: 'Private reading ambiguity.', analysisSource: 'local', model: null },
      { chords: ['Dm'], sourceLabel: 'Bar 2', needsReview: true, chordBasis: 'inferred', confidence: 'low',
        note: 'Private note evidence: D4 F4 A4.', notes: ['D4', 'F4', 'A4'],
        analysisSource: 'local', model: 'ft_f1' },
      { chords: [], noChord: true, sourceLabel: 'Bar 3', needsReview: true, chordBasis: 'printed', confidence: 'high',
        note: 'Private rest ambiguity.', analysisSource: 'local', model: null },
      { chords: ['G7'], sourceLabel: 'Bar 4', needsReview: false, chordBasis: 'printed', confidence: 'high',
        note: '', analysisSource: 'local', model: null },
    ],
  }, 'infer_harmony');
  const local = buildScoreImportForEditor(createScoreImportDraft(response, 'private-sparse.pdf'));
  assert.deepEqual([local.importSource?.analysisSource, local.importSource?.model, local.importSource?.requestId], ['local', 'ft_f1', null]);
  // The same score as an earlier build stored it when another reader supplied bar 4.
  const mixed = structuredClone(local);
  mixed.importSource = { ...mixed.importSource!, analysisSource: 'mixed', model: 'muse-spark-1.3', requestId: 'comparison-request',
    measures: mixed.importSource!.measures!.map((measure, index) => index === 3
      ? { ...measure, analysisSource: 'meta' as const, model: 'muse-spark-1.3', requestId: 'comparison-request' } : measure) };
  for (const [source, song] of [['local', local], ['mixed', mixed]] as const) {
    const project = { ...structuredClone(importedProject), song };
    withStoredProjects([], (memory) => {
      saveAll([project]);
      const restored = loadAll();
      assert.equal(restored.length, 1, `${source} F1 project stays available after reload`);
      assert.deepEqual(restored[0].song, JSON.parse(JSON.stringify(song)));
      assert.deepEqual(restored[0].generation, project.generation);
      assert.equal(restored[0].song.importSource?.measures?.[1].model, 'ft_f1');
      assert.equal(restored[0].song.importSource?.measures?.[1].requestId, null);
      assert.deepEqual(restored[0].song.measures.map((measure) => measure.chords[0]?.root), ['C', 'D', undefined, 'G']);
      assert.equal(restored[0].song.importSource?.measures?.[0].editorReview, 'Review imported chords against your source.');
      assert.equal(restored[0].song.importSource?.measures?.[2].editorReview, 'Review imported chords against your source.');
      assert.equal(restored[0].song.importSource?.measures?.[2].originalNoChord, true);
      assert.doesNotMatch(memory.stored, /private-sparse|Private|D4|F4|A4/);
    });
  }
});

test('local F1 provenance still rejects other model names and API request IDs', () => {
  const measure = { chordBasis: 'inferred', confidence: 'low', evidence: '', originalChords: ['Dm'],
    analysisSource: 'local', model: 'ft_f1', requestId: null };
  const source = { filename: '', analysisSource: 'local', model: 'ft_f1', requestId: null,
    mode: 'infer_harmony', measures: [measure] };
  for (const invalid of [
    { ...source, model: 'muse-spark-1.3' },
    { ...source, model: 'another-local-model' },
    { ...source, requestId: 'local-api-request' },
    { ...source, measures: [{ ...measure, model: 'another-local-model' }] },
    { ...source, measures: [{ ...measure, requestId: 'local-api-request' }] },
    { ...source, analysisSource: 'mixed', model: 'muse-spark-1.3', requestId: 'meta-request',
      measures: [{ ...measure, requestId: 'local-api-request' }] },
  ]) {
    const project = { ...structuredClone(importedProject), id: 'invalid-local',
      song: { ...structuredClone(importedProject.song), importSource: invalid } };
    assert.deepEqual(reloadProjects([project, importedProject]), [importedProject]);
  }
});

test('unresolved editor imports retain every source symbol and their review reason through browser reload', () => {
  const response = readScoreImportResponse({
    title: 'Unfinished import', key: 'C major', timeSignature: [4, 4], sections: [], warnings: [],
    analysisSource: 'local', model: null, mode: 'chord_symbols',
    measures: [
      { chords: [], sourceLabel: 'Bar 1', needsReview: true, chordBasis: 'unreadable', confidence: 'low',
        note: 'No supported chord was read.', noChord: false },
      { chords: ['C', 'F', 'G7'], sourceLabel: 'Bar 2', needsReview: true, chordBasis: 'printed', confidence: 'high',
        note: 'Three printed changes in one measure.', noChord: false },
    ],
  }, 'chord_symbols');
  const song = buildScoreImportForEditor(createScoreImportDraft(response, 'unfinished.pdf'));
  const project = { ...structuredClone(importedProject), song };
  const source = song.importSource!;
  assert.ok(source.measures!.every((measure) => measure.editorReview));
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, 'localStorage');
  let stored = '';
  Object.defineProperty(globalThis, 'localStorage', { configurable: true, value: {
    setItem(_key: string, value: string) { stored = value; }, getItem() { return stored; },
  } });
  try {
    saveAll([project]);
    const reloaded = loadAll();
    const restored = reloaded[0];
    assert.deepEqual(restored.song.importSource, source);
    assert.deepEqual(restored.song.measures.map((measure) => measure.chords), [[null, null], [null, null]]);
    assert.deepEqual(restored.song.importSource!.measures!.map((measure) => measure.originalChords), [[], ['C', 'F', 'G7']]);
    assert.ok(restored.song.importSource!.measures!.every((measure) => measure.originalNoChord === false && measure.userNoChord === false));
  } finally {
    if (descriptor) Object.defineProperty(globalThis, 'localStorage', descriptor);
    else Reflect.deleteProperty(globalThis, 'localStorage');
  }
  for (const editorReview of [7, 'x'.repeat(401)]) {
    const invalid = { ...project, song: { ...project.song, importSource: {
      ...source, measures: [{ ...source.measures![0], editorReview }],
    } } };
    const restored = reloadProjects([invalid]);
    assert.equal(restored.length, 1, 'old prose is removed without dropping the musical project');
    assert.equal(restored[0].song.importSource?.measures?.[0].editorReview, 'Review imported chords against your source.');
  }
});

test('ottava source pitches are discarded after their chord has been imported', () => {
  const response = readScoreImportResponse({
    title: 'Octave instruction', key: 'C major', timeSignature: [4, 4], sections: [], warnings: [],
    analysisSource: 'local', model: null, mode: 'infer_harmony',
    measures: [{ chords: ['C'], sourceLabel: 'Bar 1', needsReview: true, chordBasis: 'inferred', confidence: 'medium',
      note: 'A bounded 8va instruction raises the written note by one octave.', writtenNotes: ['C6'], notes: ['C7'], bassNotes: ['C7'] }],
  }, 'infer_harmony');
  const song = buildImportedSong(createScoreImportDraft(response, 'octave.pdf'));
  const project = { ...structuredClone(importedProject), song };
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, 'localStorage');
  let stored = '';
  Object.defineProperty(globalThis, 'localStorage', { configurable: true, value: {
    setItem(_key: string, value: string) { stored = value; }, getItem() { return stored; },
  } });
  try {
    saveAll([project]);
    const source = loadAll()[0].song.importSource?.measures?.[0];
    assert.equal(source?.writtenNotes, undefined);
    assert.equal(source?.notes, undefined);
    assert.equal(source?.bassNotes, undefined);
    assert.deepEqual(source?.originalChords, ['C']);
    assert.doesNotMatch(stored, /C6|C7|8va instruction|octave\.pdf/);
    assert.deepEqual(loadAll()[0].song.importSource, song.importSource);
  } finally {
    if (descriptor) Object.defineProperty(globalThis, 'localStorage', descriptor);
    else Reflect.deleteProperty(globalThis, 'localStorage');
  }
});

test('legacy stored projects are cleaned and rewritten without losing titles, chords, sections or settings', () => {
  const project = structuredClone(importedProject);
  project.song.importSource = {
    filename: 'private-score.pdf', model: 'muse-spark-1.3', requestId: 'request-accounting', mode: 'infer_harmony',
    measures: [{ chordBasis: 'inferred', confidence: 'medium', evidence: 'Private pitch-duration evidence', originalChords: ['F/C'],
      notes: ['C6'], writtenNotes: ['C5'], bassNotes: ['C3'], editorReview: 'Private source prose' }],
  };
  withStoredProjects([project], (memory) => {
    const [restored] = loadAll();
    assert.equal(memory.writes, 1, 'the legacy browser copy is replaced on read');
    assert.equal(restored.name, project.name);
    assert.equal(restored.song.title, project.song.title);
    assert.deepEqual(restored.song.measures, project.song.measures);
    assert.deepEqual(restored.song.sections, project.song.sections);
    assert.deepEqual(restored.generation, project.generation);
    assert.equal(restored.song.importSource?.requestId, 'request-accounting');
    assert.deepEqual(restored.song.importSource?.measures?.[0].originalChords, ['F/C']);
    assert.doesNotMatch(JSON.stringify(restored), /private-score|Private|C6|C5|C3/);
    assert.doesNotMatch(memory.stored, /private-score|Private|C6|C5|C3/);
    loadAll();
    assert.equal(memory.writes, 1, 'a cleaned project does not need another migration');
  });
});

test('a denied migration write still recovers clean projects in memory', () => {
  const project = structuredClone(importedProject);
  project.song.importSource!.filename = 'legacy-original.pdf';
  withStoredProjects([project], (memory) => {
    memory.failWrites = true;
    const [restored] = loadAll();
    assert.equal(restored.song.importSource?.filename, '');
    assert.deepEqual(restored.song.measures, project.song.measures);
    assert.match(memory.stored, /legacy-original\.pdf/, 'a failed write never overwrites storage with a partial list');
    assert.ok(useImportCleanupStore.getState().failedTargets.includes('saved projects'));
    memory.failWrites = false;
    loadAll();
    assert.equal(useImportCleanupStore.getState().failedTargets.includes('saved projects'), false,
      'a successful cleanup clears the error-only status');
    assert.doesNotMatch(memory.stored, /legacy-original\.pdf/);
  });
});

test('an invalid arrangement cache is discarded while the score and settings survive', () => {
  const restored = reloadProjects([
    { ...importedProject, arrangement: { bassPattern: null } },
  ]);
  assert.equal(restored.length, 1);
  assert.equal(restored[0].arrangement, undefined);
  assert.deepEqual(restored[0].song, importedProject.song);
});

test('session browser persistence keeps independent controller levels, including a muted part', () => {
  const session: SavedSession = {
    id: 'controller-session',
    name: 'Quiet bass mix',
    createdAt: 0,
    starred: false,
    song: {
      id: 'controller-song',
      title: 'Quiet bass mix',
      key: 'C major',
      genre: 'pop',
      bpm: 80,
      timeSignature: [4, 4],
      measures: [],
    },
    generation: {
      model: 'ft_f1',
      genre: 'pop',
      ...DEFAULT_PLAY_SETTINGS,
      masterVolume: 0.8,
      harmonyVolume: 0.65,
      bassVolume: 0,
      drumVolume: 0.25,
    },
  };
  const [restored] = roundTripProjects([session]);
  assert.deepEqual(restored.generation, session.generation);
  const config = enginePlayConfig(playSettingsFromSession(restored.generation), restored.song.genre);
  assert.deepEqual(
    [config.masterVolume, config.harmonyVolume, config.bassVolume, config.drumVolume],
    [0.8, 0.65, 0, 0.25],
  );
});
