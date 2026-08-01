import { test } from 'node:test';
import assert from 'node:assert/strict';
import type { Chord } from '../models/types';
import {
  generateGuitarVoicing,
  generatePianoVoicing,
  guitarVoicingNotes,
  QUALITY_INTERVALS,
  resolvePianoVoicing,
  rootPositionVoicing,
} from './voicingEngine.ts';

const chord = (root: string, quality: string, bass?: string, voicing: number[] = []): Chord => ({
  root, quality, bass, romanNumeral: '', voicing, extensions: [],
});

test('piano: a chord-tone bass inverts the chord (F/C = C F A)', () => {
  assert.deepEqual(generatePianoVoicing(chord('F', 'maj', 'C')), [48, 53, 57]);
  assert.deepEqual(generatePianoVoicing(chord('C', 'maj', 'E')), [52, 55, 60]);
  assert.deepEqual(generatePianoVoicing(chord('A', 'min', 'C')), [48, 52, 57]);
  assert.deepEqual(generatePianoVoicing(chord('G', '7', 'B')), [59, 62, 65, 67]);
});

test('piano: a non-chord bass goes under the root-position chord', () => {
  assert.deepEqual(generatePianoVoicing(chord('F', 'maj', 'G')), [43, 53, 57, 60]);
  assert.deepEqual(generatePianoVoicing(chord('D', 'maj', 'C')), [48, 50, 54, 57]);
  assert.deepEqual(generatePianoVoicing(chord('A', 'min', 'G')), [55, 57, 60, 64]);
  // a 4-note chord drops its 5th under a foreign bass
  assert.deepEqual(generatePianoVoicing(chord('C', 'maj7', 'D')), [38, 48, 52, 59]);
});

test('piano: the bass is always the lowest note', () => {
  const roots = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B'];
  for (const root of roots) {
    for (const bass of roots) {
      for (const quality of Object.keys(QUALITY_INTERVALS)) {
        const v = generatePianoVoicing(chord(root, quality, bass));
        assert.equal(Math.min(...v), v[0], `${root}${quality}/${bass}`);
        assert.equal(v[0] % 12, generatePianoVoicing(chord(bass, 'maj'))[0] % 12, `${root}${quality}/${bass}`);
      }
    }
  }
});

test('piano: chords without a slash bass keep the root-position voicing', () => {
  assert.deepEqual(generatePianoVoicing(chord('C', 'maj')), [48, 52, 55]);
  assert.deepEqual(generatePianoVoicing(chord('A', 'm7')), [57, 60, 64, 67]);
  assert.deepEqual(generatePianoVoicing(chord('C', 'maj', 'C')), [48, 52, 55]);
  assert.deepEqual(generatePianoVoicing(chord('F', 'maj', 'C')), [48, 53, 57]);
  assert.deepEqual(rootPositionVoicing(chord('F', 'maj', 'C')), [53, 57, 60]);
});

test('resolvePianoVoicing recomputes only root-position slash voicings saved before the fix', () => {
  assert.deepEqual(resolvePianoVoicing(chord('F', 'maj', 'C', [53, 57, 60])), [48, 53, 57]);
  assert.deepEqual(resolvePianoVoicing(chord('F', 'maj', 'C')), [48, 53, 57]);
  // a voicing played on a MIDI keyboard stays as played
  assert.deepEqual(resolvePianoVoicing(chord('F', 'maj', 'C', [36, 53, 57, 60])), [36, 53, 57, 60]);
  assert.deepEqual(resolvePianoVoicing(chord('C', 'maj', undefined, [60, 64, 67])), [60, 64, 67]);
});

test('guitar: the slash bass is on the lowest sounding string', () => {
  assert.deepEqual(generateGuitarVoicing(chord('F', 'maj', 'C')), [-1, 3, 3, 2, 1, 1]);
  assert.deepEqual(generateGuitarVoicing(chord('C', 'maj', 'E')), [0, 3, 2, 0, 1, 0]);
  assert.deepEqual(generateGuitarVoicing(chord('G', 'maj', 'B')), [-1, 2, 0, 0, 0, 3]);
  assert.deepEqual(generateGuitarVoicing(chord('D', 'maj', 'F#')), [2, 0, 0, 2, 3, 2]);
  assert.deepEqual(generateGuitarVoicing(chord('A', 'min', 'G')), [3, 0, 2, 2, 1, 0]);
});

test('guitar: chords without a slash bass are unchanged', () => {
  assert.deepEqual(generateGuitarVoicing(chord('C', 'maj')), [-1, 3, 2, 0, 1, 0]);
  assert.deepEqual(generateGuitarVoicing(chord('A', 'maj')), [-1, 0, 2, 2, 2, 0]);
  assert.deepEqual(generateGuitarVoicing(chord('G', 'maj')), [3, 2, 0, 0, 0, 3]);
});

// A hand-edited or imported project can name a root or bass after an inherited
// property of NOTE_VALUES. Such a name is not a note: the root has no voicing and
// the bass is no slash bass, exactly as for any other unknown name. Before the
// hasOwn guard these lookups returned a function or object and the piano bass
// search never ended, so a regression here shows as a test run that never finishes.
test('piano: an inherited property name as the bass is not a slash bass', () => {
  for (const bass of ['__proto__', 'constructor']) {
    assert.deepEqual(rootPositionVoicing(chord('C', 'maj', bass)), [48, 52, 55], bass);
    assert.deepEqual(generatePianoVoicing(chord('C', 'maj', bass)), [48, 52, 55], bass);
    assert.deepEqual(generatePianoVoicing(chord('A', 'm7', bass)), [57, 60, 64, 67], bass);
    assert.deepEqual(resolvePianoVoicing(chord('C', 'maj', bass)), [48, 52, 55], bass);
    // a stored root-position voicing is not "saved before slash basses": it stays
    assert.deepEqual(resolvePianoVoicing(chord('C', 'maj', bass, [48, 52, 55])), [48, 52, 55], bass);
    assert.deepEqual(resolvePianoVoicing(chord('C', 'maj', bass, [60, 64, 67])), [60, 64, 67], bass);
  }
});

test('piano: an inherited property name as the root has no voicing', () => {
  assert.deepEqual(rootPositionVoicing(chord('toString', 'maj')), []);
  assert.deepEqual(generatePianoVoicing(chord('toString', 'maj')), []);
  assert.deepEqual(generatePianoVoicing(chord('toString', 'maj', 'C')), []);
  assert.deepEqual(generatePianoVoicing(chord('toString', 'maj', 'E')), []);
  assert.deepEqual(resolvePianoVoicing(chord('toString', 'maj', 'C')), []);
  assert.deepEqual(resolvePianoVoicing(chord('toString', 'maj', 'C', [60, 64, 67])), [60, 64, 67]);
});

test('guitar: an inherited property name as root or bass is not a note', () => {
  assert.deepEqual(generateGuitarVoicing(chord('C', 'maj', '__proto__')), [-1, 3, 2, 0, 1, 0]);
  assert.deepEqual(generateGuitarVoicing(chord('C', 'maj', 'constructor')), [-1, 3, 2, 0, 1, 0]);
  assert.deepEqual(generateGuitarVoicing(chord('toString', 'maj')), [-1, -1, -1, -1, -1, -1]);
  assert.deepEqual(generateGuitarVoicing(chord('toString', 'maj', 'C')), [-1, -1, -1, -1, -1, -1]);
  assert.deepEqual(guitarVoicingNotes(chord('toString', 'maj', 'C')), []);
  assert.deepEqual(guitarVoicingNotes(chord('C', 'maj', '__proto__')), [48, 52, 55, 60, 64]);
});

test('valid chords keep their known voicings through every exported function', () => {
  assert.deepEqual(rootPositionVoicing(chord('F', 'maj', 'C')), [53, 57, 60]);
  assert.deepEqual(generatePianoVoicing(chord('F', 'maj', 'C')), [48, 53, 57]);
  assert.deepEqual(resolvePianoVoicing(chord('F', 'maj', 'C')), [48, 53, 57]);
  assert.deepEqual(generateGuitarVoicing(chord('F', 'maj', 'C')), [-1, 3, 3, 2, 1, 1]);
  assert.deepEqual(guitarVoicingNotes(chord('F', 'maj', 'C')), [48, 53, 57, 60, 65]);
  assert.deepEqual(generatePianoVoicing(chord('Bb', 'maj7', 'A')), [57, 58, 62, 65]);
  assert.deepEqual(generatePianoVoicing(chord('G', '7')), [55, 59, 62, 65]);
  assert.deepEqual(guitarVoicingNotes(chord('G', 'maj')), [43, 47, 50, 55, 59, 67]);
});

test('an inherited property name as the quality falls back to a major triad', () => {
  for (const quality of ['constructor', '__proto__', 'toString']) {
    assert.deepEqual(generatePianoVoicing(chord('C', quality)), [48, 52, 55]);
    assert.deepEqual(generateGuitarVoicing(chord('C', quality)), [-1, 3, 2, 0, 1, 0]);
  }
});
