import { test } from 'node:test';
import assert from 'node:assert/strict';
import type { Chord, SavedSession, Song } from '../models/types.ts';
import { DEFAULT_PLAY_SETTINGS } from './playSettings.ts';
import {
  buildExportSpec, isLoopbackHostname, requestMidiExport, requestScripterExport, safeExportFilename, snapshotSessionExport,
} from './exportSpec.ts';

const chord: Chord = { root: 'C', quality: 'maj', romanNumeral: 'I', extensions: [], voicing: [60, 64, 67] };
function song(): Song {
  return {
    id: 'song', title: '한글 스케치', bpm: 92, key: 'C major', genre: 'pop', timeSignature: [6, 8],
    measures: [
      { id: 'm0', index: 0, chords: [structuredClone(chord), { ...chord, bass: 'E', voicing: [] }] },
      { id: 'm1', index: 1, chords: [null, structuredClone(chord)] },
    ],
    sections: [{ id: 'part', type: null, label: 'A part', startMeasure: 0, length: 2 }],
  };
}

test('saved project exports keep the complete saved score, name and playback settings in an isolated snapshot', () => {
  const savedSong = song();
  savedSong.measures = Array.from({ length: 12 }, (_, index) => ({
    id: `m${index}`, index, chords: [structuredClone(chord), null],
  }));
  const session: SavedSession = {
    id: 'saved', name: 'Saved project name', createdAt: 0, starred: false, song: savedSong,
    generation: { model: 'ft_f1', genre: 'jazz', ...DEFAULT_PLAY_SETTINGS,
      harmonyOverride: 'acoustic_guitar_nylon', bassEnabled: false, drumMode: 'manual', drumPatternId: null,
      masterVolume: 0, harmonyVolume: 0.25, bassVolume: 0, drumVolume: 0 },
  };
  const snapshot = snapshotSessionExport(session);
  const exported = buildExportSpec(snapshot.song, snapshot.settings);
  assert.equal(snapshot.song.title, 'Saved project name');
  assert.equal(exported.title, 'Saved project name');
  assert.equal(exported.genre, 'jazz');
  assert.equal(exported.measures.length, 12); // The one-row card preview must not truncate exports.
  assert.equal(exported.play.harmonyOverride, 'acoustic_guitar_nylon');
  assert.equal(exported.play.bassEnabled, false);
  assert.equal(exported.play.drumPatternId, null);
  assert.equal(exported.play.masterVolume, 0);
  assert.equal(exported.play.harmonyVolume, 0.25);
  assert.deepEqual(exported.measures[0].chords[0]?.voicing, [48, 52, 55, 60, 64]);
  session.song.measures[0].chords[0]!.root = 'D';
  session.song.title = 'Changed after export started';
  session.generation.masterVolume = 1;
  assert.equal(snapshot.song.measures[0].chords[0]?.root, 'C');
  assert.equal(snapshot.song.title, 'Saved project name');
  assert.equal(snapshot.settings.masterVolume, 0);
});

test('export preserves two slots, rests, meter, markers and the played voicings in an owned snapshot', () => {
  const original = song();
  const exported = buildExportSpec(original, DEFAULT_PLAY_SETTINGS);
  assert.equal(exported.title, '한글 스케치');
  assert.deepEqual(exported.timeSignature, [6, 8]);
  assert.equal(exported.bpm, 92);
  assert.equal(exported.measures[0].chords[1]?.symbol, 'C/E');
  assert.deepEqual(exported.measures[0].chords[1]?.voicing, [52, 55, 60]);
  assert.equal(exported.measures[1].chords[0], null);
  assert.deepEqual(exported.sections, [{ label: 'A part', startMeasure: 0, length: 2 }]);
  original.measures[0].chords[0]!.voicing[0] = 12;
  original.timeSignature[0] = 3;
  original.sections![0].label = 'Changed';
  assert.deepEqual(exported.measures[0].chords[0]?.voicing, [60, 64, 67]);
  assert.deepEqual(exported.timeSignature, [6, 8]);
  assert.equal(exported.sections[0].label, 'A part');
});

test('guitar visual mode exports played string pitches and drops muted strings', () => {
  const original = song();
  const guitar = buildExportSpec(original, { ...DEFAULT_PLAY_SETTINGS, visualMode: 'guitar' });
  assert.deepEqual(guitar.measures[0].chords[0]?.voicing, [48, 52, 55, 60, 64]);
  assert.deepEqual(guitar.measures[0].chords[1]?.voicing, [40, 48, 52, 55, 60, 64]);
  const automatic = buildExportSpec(original, { ...DEFAULT_PLAY_SETTINGS, harmonyOverride: 'acoustic_guitar_nylon' });
  assert.deepEqual(automatic.measures, guitar.measures);
  assert.equal('visualMode' in guitar.play, false);
});

test('export preserves intentional mute, bass off, drum off and extensions with slash bass', () => {
  const original = song();
  original.measures[0].chords[0] = { ...chord, quality: '7', extensions: ['b9'], bass: 'G' };
  const exported = buildExportSpec(original, {
    ...DEFAULT_PLAY_SETTINGS, bassEnabled: false, drumMode: 'manual', drumPatternId: null,
    masterVolume: 0, harmonyVolume: 0.25, bassVolume: 0, drumVolume: 0,
  });
  assert.equal(exported.measures[0].chords[0]?.symbol, 'C7(b9)/G');
  assert.equal(exported.play.masterVolume, 0);
  assert.equal(exported.play.harmonyVolume, 0.25);
  assert.equal(exported.play.bassEnabled, false);
  assert.equal(exported.play.drumMode, 'manual');
  assert.equal(exported.play.drumPatternId, null);
  assert.equal(exported.play.drumVolume, 0);
});

test('unfinished or overlapping sections cannot silently become a finished export', () => {
  const original = song();
  original.sections![0].pending = { addedMeasures: 2 };
  assert.throws(() => buildExportSpec(original, DEFAULT_PLAY_SETTINGS), /generation/);
  delete original.sections![0].pending;
  original.sections!.push({ id: 'overlap', type: 'chorus', startMeasure: 1, length: 1 });
  assert.throws(() => buildExportSpec(original, DEFAULT_PLAY_SETTINGS), /overlap/);
});

test('filenames preserve Korean titles without paths or reserved names', () => {
  assert.equal(safeExportFilename('  한글/곡: 연습?  ', 'mid'), '한글-곡- 연습-.mid');
  assert.equal(safeExportFilename('../..', 'pdf'), '-.pdf');
  assert.equal(safeExportFilename('CON', 'mid'), '_CON.mid');
  assert.equal(safeExportFilename('\u0000\n . ', 'pdf'), 'Untitled.pdf');
  assert.equal(safeExportFilename('가'.repeat(200), 'pdf'), `${'가'.repeat(100)}.pdf`);
});

test('the Scripter script is named after the title with the TheArtist marker', () => {
  assert.equal(safeExportFilename('  한글/곡: 연습?  ', 'js'), '한글-곡- 연습-.theartist.js');
  assert.equal(safeExportFilename('CON', 'js'), '_CON.theartist.js');
  assert.equal(safeExportFilename('\u0000\n . ', 'js'), 'Untitled.theartist.js');
});

test('automatic DAW handoff is restricted to loopback hosts', () => {
  for (const host of ['localhost', 'LOCALHOST', '127.0.0.1', '127.10.0.3', '[::1]', '::1']) assert.ok(isLoopbackHostname(host));
  for (const host of ['example.com', '192.168.1.2', 'localhost.example.com', '127.999.0.1', '']) assert.equal(isLoopbackHostname(host), false);
});

test('MIDI requests send the complete snapshot and reject successful non-MIDI responses', async (t) => {
  const spec = buildExportSpec(song(), DEFAULT_PLAY_SETTINGS);
  let body = '';
  const mock = t.mock.method(globalThis, 'fetch', async (_url: unknown, init?: RequestInit) => {
    body = String(init?.body);
    return new Response(new Uint8Array([77, 84, 104, 100, 0, 0, 0, 6]), { headers: { 'Content-Type': 'audio/midi' } });
  });
  const blob = await requestMidiExport(spec);
  assert.deepEqual(JSON.parse(body), spec);
  assert.equal(blob.size, 8);
  mock.mock.mockImplementation(async () => new Response('<html>proxy error</html>'));
  await assert.rejects(requestMidiExport(spec), /did not return a MIDI/);
  mock.mock.mockImplementation(async () => new Response(JSON.stringify({ detail: 'Too many notes.' }), { status: 422 }));
  await assert.rejects(requestMidiExport(spec), /Too many notes/);
});

test('Scripter requests send the complete snapshot and accept only a JavaScript response', async (t) => {
  const spec = buildExportSpec(song(), DEFAULT_PLAY_SETTINGS);
  const script = 'var NeedsTimingInfo = true;';
  let body = '';
  const mock = t.mock.method(globalThis, 'fetch', async (url: unknown, init?: RequestInit) => {
    assert.equal(url, '/api/export/scripter');
    body = String(init?.body);
    return new Response(script, { headers: { 'Content-Type': 'text/javascript; charset=utf-8' } });
  });
  assert.equal(await (await requestScripterExport(spec)).text(), script);
  assert.deepEqual(JSON.parse(body), spec);
  mock.mock.mockImplementation(async () => new Response(script, { headers: { 'Content-Type': 'application/javascript' } }));
  assert.equal(await (await requestScripterExport(spec)).text(), script);
  // A proxy's error page, then a body with no declared type at all.
  mock.mock.mockImplementation(async () => new Response('<html>proxy error</html>', { headers: { 'Content-Type': 'text/html' } }));
  await assert.rejects(requestScripterExport(spec), /did not return a Scripter script/);
  mock.mock.mockImplementation(async () => new Response(new TextEncoder().encode(script)));
  await assert.rejects(requestScripterExport(spec), /did not return a Scripter script/);
  mock.mock.mockImplementation(async () => new Response('', { headers: { 'Content-Type': 'text/javascript' } }));
  await assert.rejects(requestScripterExport(spec), /did not return a Scripter script/);
  mock.mock.mockImplementation(async () => new Response(JSON.stringify({ detail: 'Too many notes.' }), { status: 422 }));
  await assert.rejects(requestScripterExport(spec), /Too many notes/);
});

test('a Scripter export that outlives the timeout fails as a timeout', async (t) => {
  const spec = buildExportSpec(song(), DEFAULT_PLAY_SETTINGS);
  t.mock.timers.enable({ apis: ['setTimeout'] });
  // A fetch that ends only when its signal aborts, as a stalled server would.
  t.mock.method(globalThis, 'fetch', (_url: unknown, init?: RequestInit) => new Promise<Response>((_resolve, reject) => {
    const signal = init?.signal;
    signal?.addEventListener('abort', () => reject(signal.reason));
  }));
  const rejection = assert.rejects(requestScripterExport(spec), /Scripter export timed out/);
  t.mock.timers.tick(60_000);
  await rejection;
});
