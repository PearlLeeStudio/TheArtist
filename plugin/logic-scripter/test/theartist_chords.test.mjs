// Drives the Scripter script with the mock host in scripter-host.mjs.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { createHost, notesOn, notesOff } from './scripter-host.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const TEMPLATE = readFileSync(path.join(here, '..', '..', '..', 'backend', 'app', 'services', 'scripter_template.js'), 'utf8');

// Two bars of 4/4: C (bar 0) then G (bar 1), one strike per bar, bass on beats 1 and 3.
const SONG = {
  title: 'Test', key: 'C major', bpm: 120, timeSignature: [4, 4], quartersPerBar: 4, lengthQuarters: 8,
  chords: [
    { bar: 0, slot: 0, symbol: 'C', startQuarter: 0 },
    { bar: 1, slot: 0, symbol: 'G', startQuarter: 4 },
  ],
  notes: [
    { part: 'harmony', pitch: 60, start: 0, end: 3.4, velocity: 0.7 },
    { part: 'harmony', pitch: 64, start: 0, end: 3.4, velocity: 0.7 },
    { part: 'harmony', pitch: 67, start: 0, end: 3.4, velocity: 0.7 },
    { part: 'bass', pitch: 36, start: 0, end: 0.9, velocity: 0.6 },
    { part: 'bass', pitch: 36, start: 2, end: 2.9, velocity: 0.6 },
    { part: 'harmony', pitch: 59, start: 4, end: 7.4, velocity: 0.7 },
    { part: 'harmony', pitch: 62, start: 4, end: 7.4, velocity: 0.7 },
    { part: 'harmony', pitch: 67, start: 4, end: 7.4, velocity: 0.7 },
    { part: 'bass', pitch: 43, start: 4, end: 4.9, velocity: 0.6 },
    { part: 'bass', pitch: 43, start: 6, end: 6.9, velocity: 0.6 },
  ],
};

function script(song = SONG) {
  const marker = '/*__SONG_JSON__*/';
  const at = TEMPLATE.indexOf(marker);
  const lineEnd = TEMPLATE.indexOf(';\n', at);
  return TEMPLATE.slice(0, at) + marker + JSON.stringify(song) + TEMPLATE.slice(lineEnd);
}

function near(a, b) { return Math.abs(a - b) < 1e-6; }

test('every note is struck and released at its beat, bar 1 at host beat 1', () => {
  const host = createHost(script());
  host.play(1, 9.5);
  const on = notesOn(host);
  assert.equal(on.length, SONG.notes.length);
  for (const note of SONG.notes) {
    const hit = on.find((e) => e.pitch === note.pitch && near(e.beat, 1 + note.start));
    assert.ok(hit, `note-on for ${note.pitch} at ${1 + note.start}`);
    const off = notesOff(host).find((e) => e.pitch === note.pitch && near(e.beat, 1 + note.end));
    assert.ok(off, `note-off for ${note.pitch} at ${1 + note.end}`);
  }
  assert.equal(on[0].velocity, Math.round(0.7 * 127));
  assert.deepEqual(host.traces, ['bar 1: C', 'bar 2: G']);
});

test('nothing sounds before the start bar and the song can start later in the ruler', () => {
  const host = createHost(script(), { params: { 'Start bar': 3 } });
  host.play(1, 9);
  assert.equal(notesOn(host).length, 0, 'bars 1-2 are silent');
  host.play(9, 17);
  const on = notesOn(host);
  assert.equal(on.length, SONG.notes.length);
  assert.ok(on.some((e) => e.pitch === 60 && near(e.beat, 9)));
});

test('stopping releases what is sounding, once', () => {
  const host = createHost(script());
  host.play(1, 2);
  const sounding = notesOn(host).length;
  assert.equal(sounding, 4, 'C triad plus one bass note by beat 2');
  host.stop();
  const released = notesOff(host).filter((e) => e.beat === null);
  assert.equal(released.length, 4 - 1, 'bass note already released at 1.9; the triad is released now');
  host.stop();
  assert.equal(notesOff(host).filter((e) => e.beat === null).length, 3, 'a second stop sends nothing');
});

test('locating backwards releases and restarts', () => {
  const host = createHost(script());
  host.play(1, 6);
  const before = notesOn(host).length;
  host.block(1, 1.25);
  const immediate = notesOff(host).filter((e) => e.beat === null);
  assert.ok(immediate.length >= 3, 'the G chord sounding at beat 6 is released on the jump');
  assert.ok(notesOn(host).length > before, 'the first chord is struck again');
});

test('a cycle that wraps inside one block restarts at the left locator', () => {
  const host = createHost(script());
  host.play(1, 4.9, 0.25, { cycling: true, leftCycleBeat: 1, rightCycleBeat: 5 });
  const beforeWrap = notesOn(host).length;
  host.block(4.9, 5.2, { cycling: true, leftCycleBeat: 1, rightCycleBeat: 5 });
  const after = notesOn(host).slice(beforeWrap);
  assert.ok(after.some((e) => e.pitch === 60 && near(e.beat, 1)), 'the C chord restarts at beat 1');
  assert.ok(!after.some((e) => e.pitch === 59), 'the G chord of bar 2 is not struck in the wrapped part');
});

test('without Loop the timeline after the form stays silent', () => {
  const host = createHost(script());
  host.play(1, 25);
  assert.equal(notesOn(host).length, SONG.notes.length, 'the two bars once, nothing in bars 3-6');
  assert.ok(notesOn(host).every((e) => e.beat < 9));
});

test('Loop repeats the arrangement after its last bar', () => {
  const host = createHost(script(), { params: { Loop: 1 } });
  host.play(1, 13);
  const cHits = notesOn(host).filter((e) => e.pitch === 60).map((e) => e.beat);
  assert.deepEqual(cHits.map((b) => Math.round(b * 100) / 100), [1, 9]);
  const gHits = notesOn(host).filter((e) => e.pitch === 59).map((e) => e.beat);
  assert.deepEqual(gHits, [5]);
});

test('Transpose, Velocity % and the part switches apply to new strikes', () => {
  const host = createHost(script(), { params: { Transpose: 2, 'Velocity %': 50, Bass: 0 } });
  host.play(1, 5);
  const on = notesOn(host);
  assert.deepEqual([...new Set(on.map((e) => e.pitch))].sort((a, b) => a - b), [62, 66, 69]);
  assert.ok(on.every((e) => e.velocity === Math.round(0.7 * 127 * 0.5)));
});

test('6/8 maps quarter-note times onto the ruler beats of the meter', () => {
  const song = { ...SONG, timeSignature: [6, 8], quartersPerBar: 3, lengthQuarters: 6,
    chords: [{ bar: 0, slot: 0, symbol: 'C', startQuarter: 0 }],
    notes: [{ part: 'harmony', pitch: 60, start: 1.5, end: 2.5, velocity: 0.7 }] };
  const ruler = createHost(script(song), { meter: [6, 8] });
  ruler.play(1, 7);
  assert.ok(near(notesOn(ruler)[0].beat, 1 + 3), 'eighth-note ruler: 1.5 quarters = 3 ruler beats');
  const quarter = createHost(script(song), { meter: [6, 8], params: { 'Beat unit': 1 } });
  quarter.play(1, 7);
  assert.ok(near(notesOn(quarter)[0].beat, 1 + 1.5));
});

// The same two bars as a song form of two one-bar sections, A (the C bar) and B (the G bar)
const FORM = { ...SONG, sections: [{ label: 'A', startBar: 0, bars: 1 }, { label: 'B', startBar: 1, bars: 1 }] };

test('the Section menu lists the whole form, then each section with its bars', () => {
  const host = createHost(script(FORM));
  const menu = host.context.PluginParameters.find((p) => p.name === 'Section');
  assert.deepEqual([...menu.valueStrings], ['Whole form (2 bars)', 'A (bars 1-1)', 'B (bars 2-2)']);
  assert.equal(menu.defaultValue, 0);
});

test('a chosen section plays alone from the Start bar and is silent after it', () => {
  const host = createHost(script(FORM), { params: { Section: 2, 'Start bar': 3 } });
  host.play(1, 21);
  const on = notesOn(host);
  assert.deepEqual([...new Set(on.map((e) => e.pitch))].sort((a, b) => a - b), [43, 59, 62, 67], 'only section B');
  assert.ok(on.filter((e) => e.pitch === 59).every((e) => near(e.beat, 9)), 'B starts at bar 3 = beat 9');
  assert.equal(on.length, 5, 'once: G triad and its two bass notes');
  assert.deepEqual(host.traces, ['B · bar 2: G']);
});

test('Loop repeats only the chosen section, and its notes end at the section edge', () => {
  const sustained = { ...FORM, notes: [{ part: 'harmony', pitch: 60, start: 0, end: 6, velocity: 0.7 }] };
  const host = createHost(script(sustained), { params: { Section: 1, Loop: 1 } });
  host.play(1, 13);
  assert.deepEqual(notesOn(host).map((e) => e.beat), [1, 5, 9], 'section A every bar');
  assert.ok(notesOff(host).filter((e) => e.beat !== null).every((e) => near((e.beat - 1) % 4, 0)),
    'a note longer than its section is released at the section edge');
});

test('the committed example, generated by the backend, plays every note it carries', () => {
  const example = readFileSync(path.join(here, '..', 'example', 'theartist_chords.example.js'), 'utf8');
  const host = createHost(example, { tempo: 96 });
  const song = host.context.SONG;
  assert.deepEqual([...song.sections.map((s) => s.label)], ['Verse', 'Chorus']);  // copied out of the host's realm
  host.play(1, 1 + song.lengthQuarters + 1);
  assert.equal(notesOn(host).length, song.notes.length);
  assert.equal(notesOff(host).length, song.notes.length, 'every note is released, none left hanging');
  assert.equal(host.traces[0], 'Verse · bar 1: Cmaj7');
  assert.ok(host.traces.includes('Verse · bar 4+: C/E'), 'the second slot of bar 4 is traced as its own chord');
  assert.ok(host.traces.includes('Chorus · bar 5: Fmaj7'));
});

test('incoming notes pass through and Reset releases everything', () => {
  const host = createHost(script());
  host.incoming(72);
  assert.deepEqual(host.sent.at(-1), { type: 'NoteOn', pitch: 72, velocity: 100, beat: null });
  host.play(1, 2);
  host.reset();
  assert.ok(notesOff(host).filter((e) => e.beat === null).length >= 3);
});
