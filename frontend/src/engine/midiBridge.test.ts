import { test } from 'node:test';
import assert from 'node:assert/strict';
import { bridgeMessage, MIDI_BRIDGE_ORIGIN } from './midiBridge.ts';

test('bridge input requires the opened window, loopback origin and connection nonce', () => {
  const popup = {} as Window;
  const event = { origin: MIDI_BRIDGE_ORIGIN, source: popup,
    data: { protocol: 'theartist-midi-v1', token: 'connection-one', type: 'midi', data: [0x90, 60, 100], port: 'keyboard' } };
  assert.deepEqual(bridgeMessage(event, popup, 'connection-one'), { type: 'midi', data: [0x90, 60, 100], port: 'keyboard' });
  assert.equal(bridgeMessage({ ...event, origin: 'https://unrelated.example' }, popup, 'connection-one'), null);
  assert.equal(bridgeMessage({ ...event, source: {} as Window }, popup, 'connection-one'), null);
  assert.equal(bridgeMessage(event, popup, 'connection-two'), null);
  assert.equal(bridgeMessage({ ...event, data: { ...event.data, data: [0x90, 180, 100] } }, popup, 'connection-one'), null);
  assert.equal(bridgeMessage({ ...event, data: { ...event.data, data: [0xf0, 1, 2] } }, popup, 'connection-one'), null);
});
