import { test } from 'node:test';
import assert from 'node:assert/strict';
import { GENRE_OPTIONS, genreForModel, genreOptionValue } from './genres.ts';

test('the genre list is none plus the 13 genres, each once', () => {
  const values = GENRE_OPTIONS.map((option) => option.value);
  assert.equal(values[0], 'none');
  assert.equal(values.length, 14);
  assert.equal(new Set(values).size, 14);
});

test('a song genre reads as its select value, the old bossa nova key as bossa', () => {
  assert.equal(genreOptionValue('rnb_soul'), 'rnb_soul');
  assert.equal(genreOptionValue('bossa nova'), 'bossa');
  assert.equal(genreOptionValue(null), 'none');
  assert.equal(genreOptionValue(''), 'none');
});

test('a picked model implies its genre: LoRA tag, pop for F1 and Phase 0, jazz for the other fine-tunes', () => {
  assert.equal(genreForModel('ft_f1_lora_rnb_soul'), 'rnb_soul');
  assert.equal(genreForModel('ft_f1_lora_hip_hop'), 'hip_hop');
  assert.equal(genreForModel('ft_f1'), 'pop');
  assert.equal(genreForModel('phase0'), 'pop');
  assert.equal(genreForModel('ft_f4'), 'jazz');
  assert.equal(genreForModel('ft_f1_v2'), 'jazz');
  assert.equal(genreForModel('custom'), null);
});
