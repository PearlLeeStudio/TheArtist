import { test } from 'node:test';
import assert from 'node:assert/strict';
import { REHARM_CONTEXT_REACH, reharmContext } from './reharmonize.ts';
import { parseChordName } from './chordParser.ts';
import type { Song } from '../models/types.ts';

function song(bars: number): Song {
  return {
    id: 's', title: 'Long', bpm: 100, key: 'C major', genre: null, timeSignature: [4, 4], sections: [],
    measures: Array.from({ length: bars }, (_, index) => ({
      id: `m${index}`, index, chords: [parseChordName(index % 2 ? 'G7' : 'Cmaj7'), index % 3 ? parseChordName('Am7') : null],
    })),
  };
}

test('a long score sends only the bars around the passage, numbered as in the score', () => {
  const context = reharmContext(song(465), [{ measureIndex: 300, slotIndex: 0 }, { measureIndex: 301, slotIndex: 1 }]);
  assert.equal(context[0].measure, 300 - REHARM_CONTEXT_REACH);
  assert.equal(context[context.length - 1].measure, 301 + REHARM_CONTEXT_REACH);
  assert.equal(context.length, 2 + 2 * REHARM_CONTEXT_REACH);
  assert.ok(context.length <= 512);
  assert.deepEqual(context[16], { measure: 300, chords: ['Cmaj7', null] });
});

test('the window is clipped at the start and end of the score and covers a short score whole', () => {
  const start = reharmContext(song(465), [{ measureIndex: 2, slotIndex: 0 }]);
  assert.equal(start[0].measure, 0);
  assert.equal(start[start.length - 1].measure, 2 + REHARM_CONTEXT_REACH);
  const end = reharmContext(song(465), [{ measureIndex: 464, slotIndex: 0 }]);
  assert.equal(end[end.length - 1].measure, 464);
  const whole = reharmContext(song(12), [{ measureIndex: 5, slotIndex: 0 }]);
  assert.deepEqual(whole.map((bar) => bar.measure), Array.from({ length: 12 }, (_, i) => i));
});
