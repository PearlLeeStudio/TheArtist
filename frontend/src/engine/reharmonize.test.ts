import { test } from 'node:test';
import assert from 'node:assert/strict';
import type { Song, SlotAddress } from '../models/types.ts';
import { parseChordName } from './chordParser.ts';
import { reharmOutcome, reharmRunDetails, reharmSymbol, reharmSelectionError, requestReharmonization, validateReharmResult } from './reharmonize.ts';
import type { ReharmCandidate, ReharmResult } from './reharmonize.ts';
import { emptyReharmState, hasPendingReharmSelection, reharmActions } from './reharmonizeSession.ts';
import type { ReharmState } from './reharmonizeSession.ts';
import { withoutChordPreview } from './chordPreview.ts';
import type { ChordPreview } from './chordPreview.ts';
import { captureGenerationComparison } from './generationComparison.ts';

const slots = [{ measureIndex: 1, slotIndex: 0 as const }, { measureIndex: 1, slotIndex: 1 as const }];
function score(): Song {
  return { id: 'one', title: 'Song', key: 'C major', timeSignature: [4, 4], bpm: 80, genre: null,
    measures: [['Cmaj7', 'Am7'], ['Dm7', 'G7'], ['Cmaj7', 'Cmaj7']].map((chords, index) => ({
      id: `bar-${index}`, index, chords: chords.map(c => ({ ...parseChordName(c)!, voicing: [48, 55, 60], romanNumeral: 'saved' })) as Song['measures'][0]['chords'],
    })) };
}
function candidate(id: string, slotIndex: 0 | 1, after: string): ReharmCandidate {
  return { id, family: 'common_tones', changes: [{ address: { measureIndex: 1, slotIndex },
    before: slotIndex === 0 ? 'Dm7' : 'G7', after, commonTones: 2, modes: ['dorian'] }],
  pitchDistance: .5, originalPitchDistance: .8, bassMotion: 2, originalBassMotion: 3, mathScore: 1, modelScore: -2 };
}
function result(): ReharmResult {
  return { candidates: [candidate('a', 1, 'Db7'), candidate('b', 0, 'Fm7'), candidate('c', 1, 'G9')],
    modelUsed: 'Music Transformer', contextMeasures: [0, 1, 2] };
}
function harness(requester: typeof requestReharmonization = async () => result(), selectedSlots: SlotAddress[] = slots) {
  let song = score();
  let epoch = 1;
  let writes = 0;
  let stops = 0;
  let pendingPreview: ChordPreview | null = null;
  const remembered: Song[] = [];
  let state: ReharmState;
  const actions = reharmActions({
    read: () => ({ song, songEpoch: epoch, selectedSlots }),
    stop: () => { stops++; },
    clearGeneration: () => undefined,
    commit: () => { pendingPreview = null; },
    remember: (snapshot) => { remembered.push(snapshot); },
    write: (updates, previewOriginal) => {
      writes++;
      song = { ...song, measures: song.measures.map((m, index) => {
        const chords = [...m.chords] as typeof m.chords;
        for (const u of updates) if (u.address.measureIndex === index) chords[u.address.slotIndex] = u.chord;
        return { ...m, chords };
      }) };
      pendingPreview = previewOriginal ? { original: previewOriginal,
        shown: captureGenerationComparison(song, previewOriginal.original.map(item => item.address), epoch), key: song.key } : null;
    },
  }, () => state, patch => { state = { ...state, ...patch }; }, requester);
  state = { ...emptyReharmState, ...actions };
  return { actions, state: () => state, song: () => song, writes: () => writes, stops: () => stops, remembered,
    saved: () => withoutChordPreview(song, epoch, pendingPreview),
    edit: (edit: (s: Song) => void) => { song = structuredClone(song); edit(song); },
    replace: () => { song = score(); epoch++; actions.close(); } };
}

test('disjoint alternatives combine and toggle independently; Origin restores the exact score', async () => {
  const h = harness();
  const original = structuredClone(h.song());
  await h.actions.run('ft_f1', 1, 'auto');
  assert.equal(h.writes(), 0);
  const choices = h.state().result;
  h.actions.preview(0);
  assert.equal(reharmSymbol(h.song().measures[1].chords[1]!), 'Db7');
  h.actions.preview(1);
  assert.equal(reharmSymbol(h.song().measures[1].chords[1]!), 'Db7');
  assert.equal(reharmSymbol(h.song().measures[1].chords[0]!), 'Fm7');
  assert.deepEqual(h.state().selected, [0, 1]);
  h.actions.preview(0);
  assert.deepEqual(h.song().measures[1].chords[1], original.measures[1].chords[1]);
  assert.equal(reharmSymbol(h.song().measures[1].chords[0]!), 'Fm7');
  assert.deepEqual(h.state().selected, [1]);
  h.actions.preview(null);
  assert.deepEqual(h.song(), original);
  assert.equal(h.state().result, choices);
  assert.equal(h.state().result?.candidates.length, 3);
  assert.equal(h.stops(), 4);
});

test('the latest overlapping alternative replaces only conflicting candidates', async () => {
  const h = harness();
  await h.actions.run('ft_f1', 1, 'auto');
  h.actions.preview(0);
  h.actions.preview(1);
  h.actions.preview(2);
  assert.deepEqual(h.state().selected, [1, 2]);
  assert.deepEqual(h.song().measures[1].chords.map(c => reharmSymbol(c!)), ['Fm7', 'G9']);
  h.actions.preview(0);
  assert.deepEqual(h.state().selected, [0, 1]);
  assert.deepEqual(h.song().measures[1].chords.map(c => reharmSymbol(c!)), ['Fm7', 'Db7']);
});

test('X discards pending previews; Apply keeps combined choices, closes the cards and leaves an undo step', async () => {
  const h = harness();
  const original = structuredClone(h.song());
  await h.actions.run('ft_f1', 1, 'auto');
  h.actions.preview(2);
  assert.equal(h.state().result?.candidates.length, 3);
  assert.deepEqual(h.saved(), original);
  h.actions.close();
  assert.deepEqual(h.song(), original);
  await h.actions.run('ft_f1', 1, 'auto');
  assert.equal(hasPendingReharmSelection(h.state()), false);
  h.actions.preview(0);
  h.actions.preview(1);
  assert.equal(hasPendingReharmSelection(h.state()), true);
  const chosen = structuredClone(h.song());
  h.actions.apply();
  assert.equal(h.state().result, null, 'the comparison closes on Apply');
  assert.equal(h.state().comparison, null);
  assert.deepEqual(h.song(), chosen);
  assert.deepEqual(h.saved(), chosen);
  assert.match(h.state().notice!, /^Alter 1 \+ Alter 2 applied to M2\./);
  assert.deepEqual(h.remembered, [original], 'the passage before Apply is kept for undo');
  // Nothing is open any more: previews and a second Apply are no-ops.
  h.actions.preview(2);
  h.actions.apply();
  assert.deepEqual(h.song(), chosen);
  await h.actions.run('ft_f1', 1, 'auto');
  assert.equal(h.state().notice, null, 'a new search clears the notice');
});

test('applying Origin after keeping an alternative restores the passage and also closes', async () => {
  const h = harness();
  const original = structuredClone(h.song());
  await h.actions.run('ft_f1', 1, 'auto');
  h.actions.preview(0);
  h.actions.apply();
  const kept = structuredClone(h.song());
  assert.notDeepEqual(kept, original);
  await h.actions.run('ft_f1', 1, 'auto');
  h.actions.preview(1);
  h.actions.preview(null);
  assert.equal(hasPendingReharmSelection(h.state()), false, 'Origin of the new search is the kept passage');
  h.actions.close();
  assert.deepEqual(h.song(), kept);
  assert.deepEqual(h.remembered, [original]);
});

test('a new search discards a pending preview before building its request', async () => {
  const requested: Song[] = [];
  const h = harness(async song => { requested.push(structuredClone(song)); return result(); });
  await h.actions.run('ft_f1', 1, 'auto');
  h.actions.preview(0);
  await h.actions.run('ft_f1', 1, 'auto');
  assert.deepEqual(requested[1], requested[0]);
});

test('Apply rejects stale edits; cancel and autosave preserve manual edits but remove remaining previews', async () => {
  const h = harness();
  const original = structuredClone(h.song());
  await h.actions.run('ft_f1', 1, 'auto');
  h.actions.preview(0);
  h.actions.preview(1);
  h.edit(s => { s.measures[1].chords[1] = parseChordName('E7'); s.measures[2].chords[0] = parseChordName('Ab'); });
  h.actions.apply();
  assert.match(h.state().error!, /score changed/);
  const saved = h.saved();
  assert.deepEqual(saved.measures[1].chords[0], original.measures[1].chords[0]);
  assert.equal(reharmSymbol(saved.measures[1].chords[1]!), 'E7');
  assert.equal(reharmSymbol(saved.measures[2].chords[0]!), 'Ab');
  h.actions.close();
  assert.deepEqual(h.song(), saved);
});

test('closing pending choices after replacing a song does not write into the new song', async () => {
  const h = harness();
  await h.actions.run('ft_f1', 1, 'auto');
  h.actions.preview(0);
  const writes = h.writes();
  h.replace();
  assert.equal(h.writes(), writes);
  assert.deepEqual(h.song(), score());
});

test('four-change requests carry the cap, preview whole alternatives and restore the exact original', async t => {
  const selected: SlotAddress[] = [0, 1].flatMap(measureIndex =>
    ([0, 1] as const).map(slotIndex => ({ measureIndex, slotIndex })));
  const original = score();
  const replacements = [['C6', 'Am9', 'Dm9', 'Db7'], ['Em7', 'C6', 'Fm7'], ['C6', 'Am9']];
  const choices: ReharmResult = { ...result(), candidates: replacements.map((after, i) => ({
    ...candidate(String(i), 0, 'Fm7'),
    changes: after.map((symbol, j) => ({ address: selected[j],
      before: reharmSymbol(original.measures[selected[j].measureIndex].chords[selected[j].slotIndex]!),
      after: symbol, commonTones: 2, modes: [] })),
  })) };
  t.mock.method(globalThis, 'fetch', async (_url, init) => {
    const body = JSON.parse(String(init?.body));
    assert.equal(body.maxChanges, 4);
    assert.deepEqual(body.selectedSlots, selected);
    return Response.json(choices);
  });
  const h = harness(requestReharmonization, selected);
  await h.actions.run('ft_f1', 4, 'auto');
  assert.equal(h.state().error, null);
  assert.equal(h.writes(), 0);
  for (const index of [0, 1, 2, 0]) {
    h.actions.preview(index);
    for (const [j, address] of selected.entries()) {
      assert.equal(reharmSymbol(h.song().measures[address.measureIndex].chords[address.slotIndex]!),
        replacements[index][j] ?? reharmSymbol(original.measures[address.measureIndex].chords[address.slotIndex]!));
    }
    assert.deepEqual(h.song().measures[2], original.measures[2]);
  }
  h.actions.preview(null);
  assert.deepEqual(h.song(), original);
  assert.throws(() => validateReharmResult(choices, original, selected, 3), /invalid reharmonization/);
});

test('outside edits block stale candidates but original restores without touching the outside edit', async () => {
  const h = harness();
  const original = structuredClone(h.song());
  await h.actions.run('ft_f1', 1, 'auto');
  h.actions.preview(0);
  h.edit(s => { s.measures[2].chords[0] = parseChordName('Ab'); });
  h.actions.preview(1);
  assert.equal(h.writes(), 1);
  assert.match(h.state().error!, /surrounding/);
  h.actions.preview(null);
  assert.deepEqual(h.song().measures[1], original.measures[1]);
  assert.equal(reharmSymbol(h.song().measures[2].chords[0]!), 'Ab');
  assert.equal(h.state().result?.candidates.length, 3);
});

test('target edits are never overwritten by a stale comparison', async () => {
  const h = harness();
  await h.actions.run('ft_f1', 1, 'auto');
  h.actions.preview(0);
  h.edit(s => { s.measures[1].chords[1] = parseChordName('E7'); });
  h.actions.preview(null);
  assert.equal(reharmSymbol(h.song().measures[1].chords[1]!), 'E7');
  assert.equal(h.writes(), 1);
});

test('late completion after a replaced song cannot reopen or edit it', async () => {
  let resolve!: (r: ReharmResult) => void;
  const h = harness(() => new Promise(r => { resolve = r; }));
  const pending = h.actions.run('ft_f1', 1, 'auto');
  h.replace();
  resolve(result());
  await pending;
  assert.equal(h.state().result, null);
  assert.equal(h.writes(), 0);
});

test('an edit while a request runs rejects the result without mutating that edit', async () => {
  let resolve!: (r: ReharmResult) => void;
  const h = harness(() => new Promise(r => { resolve = r; }));
  const pending = h.actions.run('ft_f1', 1, 'auto');
  h.edit(s => { s.measures[0].chords[0] = parseChordName('E7'); });
  resolve(result());
  await pending;
  assert.equal(h.state().result, null);
  assert.match(h.state().error!, /score changed/);
  assert.equal(h.writes(), 0);
});

test('reject malformed, duplicate, excess, out-of-selection and mismatched-origin patches as a whole', () => {
  assert.equal(validateReharmResult(result(), score(), slots, 1).candidates.length, 3);
  for (const mutate of [
    (r: ReharmResult) => { r.candidates[0].changes[0].address.measureIndex = 0; },
    (r: ReharmResult) => { r.candidates[0].changes[0].before = 'wrong'; },
    (r: ReharmResult) => { r.candidates[0].changes.push(r.candidates[1].changes[0]); },
    (r: ReharmResult) => { r.candidates[1] = structuredClone(r.candidates[0]); },
    (r: ReharmResult) => { r.candidates[0].modelScore = NaN; },
  ]) {
    const r = result(); mutate(r);
    assert.throws(() => validateReharmResult(r, score(), slots, 1), /invalid reharmonization/);
  }
});

test('manual tension and explicit bass are preserved in the search request symbol', () => {
  assert.equal(reharmSymbol({ ...parseChordName('G7/B')!, extensions: ['b9'] }), 'G7(b9)/B');
  assert.equal(reharmSymbol({ ...parseChordName('G7/B')!, extensions: ['b9', '#11'] }), 'G7(b9,#11)/B');
});

test('augmented-seventh targets survive request, A/B/C preview and Original beside unresolved slots', async t => {
  const choices = { ...result(), modelApproximations: [{ address: slots[0], original: 'Eaug7', scoredAs: 'E7' }],
    candidates: ['E7', 'Cmaj7', 'G#dim7'].map((after, i) => ({ ...candidate(String(i), 0, after),
      changes: [{ ...candidate(String(i), 0, after).changes[0], before: 'Eaug7' }] })) };
  const fetch = t.mock.method(globalThis, 'fetch', async (_url, init) => {
    const body = JSON.parse(String(init?.body));
    assert.deepEqual(body.context[1].chords, ['Eaug7', null]);
    assert.deepEqual(body.selectedSlots, [slots[0]]);
    return Response.json(choices);
  });
  const h = harness(requestReharmonization);
  h.edit(song => { song.measures[1].chords = [parseChordName('Eaug7'), null]; });
  const original = structuredClone(h.song());
  await h.actions.run('ft_f1_lora_classical', 1, 'auto');
  assert.equal(h.state().error, null);
  assert.equal(fetch.mock.callCount(), 1);
  for (const index of [0, 1, 2]) {
    h.actions.preview(index);
    assert.equal(reharmSymbol(h.song().measures[1].chords[0]!), choices.candidates[index].changes[0].after);
    assert.equal(h.song().measures[1].chords[1], null);
    assert.equal(h.state().result?.candidates.length, 3);
  }
  h.actions.preview(null);
  assert.deepEqual(h.song(), original);
});

test('a dragged range with empty half-bars is valid, but empty-only and over-wide selections remain blocked', () => {
  const song = score();
  song.measures[1].chords[1] = null;
  assert.equal(reharmSelectionError(song, slots), null);
  assert.match(reharmSelectionError(song, [slots[1]])!, /no chord symbols/);
  assert.match(reharmSelectionError(song, [])!, /Select a passage/);
  song.measures = Array.from({ length: 9 }, (_, i) => ({ ...song.measures[0], id: `m${i}`, index: i }));
  song.measures[8].chords = [null, null];
  assert.match(reharmSelectionError(song, [{ measureIndex: 0, slotIndex: 0 }, { measureIndex: 8, slotIndex: 1 }])!, /eight measures/);
});

test('mixed selections request and capture only chords; all previews and Original preserve the gaps', async () => {
  const choices = { ...result(), candidates: ['Fm7', 'Em7', 'Bb7'].map((after, i) => candidate(String(i), 0, after)) };
  const h = harness(async (song, targets, maximum, mode) => {
    assert.deepEqual(targets, [slots[0]]);
    assert.equal(song.measures[1].chords[1], null);
    assert.equal(maximum, 1);
    assert.equal(mode, 'auto');
    return choices;
  });
  h.edit(song => { song.measures[1].chords[1] = null; });
  const original = structuredClone(h.song());
  await h.actions.run('ft_f1', 1, 'auto');
  assert.equal(h.state().error, null);
  assert.equal(h.state().comparison?.original.length, 1);
  assert.equal(h.writes(), 0);
  for (const index of [0, 1, 2, null]) {
    h.actions.preview(index);
    assert.equal(h.song().measures[1].chords[1], null);
    assert.equal(h.state().result?.candidates.length, 3);
  }
  assert.deepEqual(h.song(), original);
});

test('HTTP request omits empty targets while retaining full context and auto mode; empty-only makes no call', async t => {
  const song = score();
  song.measures[1].chords[1] = null;
  const choices = { ...result(), candidates: ['Fm7', 'Em7', 'Bb7'].map((after, i) => candidate(String(i), 0, after)) };
  const fetch = t.mock.method(globalThis, 'fetch', async (_url, init) => {
    const body = JSON.parse(String(init?.body));
    assert.deepEqual(body.selectedSlots, [slots[0]]);
    assert.equal(body.context[1].chords[1], null);
    assert.equal(body.context.length, 3);
    assert.equal(body.mode, 'auto');
    return Response.json(choices);
  });
  await requestReharmonization(song, slots, 1, 'auto', 'ft_f1', new AbortController().signal);
  await assert.rejects(requestReharmonization(song, [slots[1]], 1, 'auto', 'ft_f1', new AbortController().signal), /no chord symbols/);
  assert.equal(fetch.mock.callCount(), 1);
});

test('an unresponsive reharmonization request times out visibly and preserves the score', async t => {
  const deadline = new AbortController();
  t.mock.method(AbortSignal, 'timeout', (milliseconds: number) => {
    assert.equal(milliseconds, 110_000);
    return deadline.signal;
  });
  t.mock.method(globalThis, 'fetch', (_url, init) => new Promise<Response>((_resolve, reject) => {
    init!.signal!.addEventListener('abort', () => reject(init!.signal!.reason), { once: true });
  }));
  const h = harness(requestReharmonization);
  const original = structuredClone(h.song());
  const pending = h.actions.run('ft_f1', 1, 'auto');
  assert.equal(h.state().loading, true);
  deadline.abort(new DOMException('Timed out', 'TimeoutError'));
  await pending;
  assert.equal(h.state().loading, false);
  assert.match(h.state().error!, /exceeded 110 seconds/);
  assert.deepEqual(h.song(), original);
  assert.equal(h.writes(), 0);
});


test('a server result shows the scoring model in the outcome box and Run details, and previews like any other', async t => {
  const fetch = t.mock.method(globalThis, 'fetch', async () => Response.json(result()));
  const original = score();
  const h = harness(requestReharmonization);
  await h.actions.run('ft_f1', 1, 'auto');
  assert.equal(fetch.mock.callCount(), 1);
  assert.equal(h.state().loading, false);
  assert.equal(h.state().error, null);
  assert.equal(h.writes(), 0);
  assert.deepEqual(h.song(), original);
  const checked = h.state().result!;
  assert.deepEqual(reharmOutcome(checked), { title: 'Music Transformer alternatives',
    detail: 'Music Transformer scored these substitutions against the surrounding chords.' });
  assert.deepEqual(reharmRunDetails(checked, 1200), [
    'Scored against the surrounding bars by Music Transformer',
    'Three contrasting alternatives ranked by conditional chord likelihood.',
    '1.2 s',
  ]);
  assert.deepEqual(reharmRunDetails(checked, null), [
    'Scored against the surrounding bars by Music Transformer',
    'Three contrasting alternatives ranked by conditional chord likelihood.',
  ]);
  h.actions.preview(0);
  assert.equal(reharmSymbol(h.song().measures[1].chords[1]!), 'Db7');
  h.actions.preview(null);
  assert.deepEqual(h.song(), original);
});
