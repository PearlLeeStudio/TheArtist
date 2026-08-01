import { test } from 'node:test';
import assert from 'node:assert/strict';
import { disconnectMidi, initMidi, resetMidiNotes } from './midiInput.ts';

function accessFixture() {
  const input = { state: 'connected', onmidimessage: null } as unknown as MIDIInput;
  const access = { inputs: new Map([['keyboard', input]]), onstatechange: null } as unknown as MIDIAccess;
  return {
    input, access,
    note(note: number, on = true) {
      input.onmidimessage?.call(input, { data: new Uint8Array([on ? 0x90 : 0x80, note, on ? 100 : 0]) } as MIDIMessageEvent);
    },
  };
}

async function withMidi(requestMIDIAccess: () => Promise<MIDIAccess>, run: () => Promise<void>) {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  Object.defineProperty(globalThis, 'navigator', { configurable: true, value: { requestMIDIAccess } });
  try { await run(); }
  finally {
    disconnectMidi();
    if (descriptor) Object.defineProperty(globalThis, 'navigator', descriptor);
    else Reflect.deleteProperty(globalThis, 'navigator');
  }
}

test('late MIDI permission results cannot attach after disconnect or replace a newer connection', async () => {
  const old = accessFixture();
  const current = accessFixture();
  let resolveOld!: (access: MIDIAccess) => void;
  let requests = 0;
  const pending = new Promise<MIDIAccess>((resolve) => { resolveOld = resolve; });
  await withMidi(() => ++requests === 1 ? pending : Promise.resolve(current.access), async () => {
    const oldNotes: number[][] = [];
    const currentNotes: number[][] = [];
    const oldRequest = initMidi((notes) => oldNotes.push(notes), () => {});
    disconnectMidi();
    assert.equal(await initMidi((notes) => currentNotes.push(notes), () => {}), true);
    resolveOld(old.access);
    assert.equal(await oldRequest, false);
    assert.equal(old.input.onmidimessage, null);
    current.note(60);
    assert.deepEqual(oldNotes, []);
    assert.deepEqual(currentNotes, [[60]]);
  });
});

test('song replacement clears held notes without disconnecting the controller', async () => {
  const device = accessFixture();
  await withMidi(() => Promise.resolve(device.access), async () => {
    const notes: number[][] = [];
    await initMidi((value) => notes.push(value), () => {});
    device.note(60);
    resetMidiNotes();
    device.note(64);
    assert.deepEqual(notes, [[60], [64]]);
    assert.notEqual(device.input.onmidimessage, null);
    disconnectMidi();
    assert.equal(device.input.onmidimessage, null);
    assert.equal(device.access.onstatechange, null);
  });
});

test('device connection changes update status and clear notes that cannot receive a release', async () => {
  const device = accessFixture();
  await withMidi(() => Promise.resolve(device.access), async () => {
    const notes: number[][] = [];
    const connected: boolean[] = [];
    await initMidi((value) => notes.push(value), () => {}, (value) => connected.push(value));
    device.note(60);
    Object.assign(device.input, { state: 'disconnected' });
    device.access.onstatechange?.call(device.access, {} as MIDIConnectionEvent);
    assert.deepEqual(connected, [true, false]);
    assert.deepEqual(notes, [[60], []]);
    assert.equal(device.input.onmidimessage, null);
  });
});

test('queued messages from an old connection cannot appear in the new session', async () => {
  const device = accessFixture();
  await withMidi(() => Promise.resolve(device.access), async () => {
    await initMidi(() => {}, () => {});
    const oldHandler = device.input.onmidimessage;
    disconnectMidi();
    const notes: number[][] = [];
    await initMidi((value) => notes.push(value), () => {});
    oldHandler?.call(device.input, { data: new Uint8Array([0x90, 60, 100]) } as MIDIMessageEvent);
    device.note(64);
    assert.deepEqual(notes, [[64]]);
  });
});

test('matching notes on different MIDI channels release independently and all-notes-off clears only its channel', async () => {
  const device = accessFixture();
  await withMidi(() => Promise.resolve(device.access), async () => {
    const notes: number[][] = [];
    await initMidi((value) => notes.push(value), () => {});
    const send = (data: number[]) => device.input.onmidimessage?.call(device.input, { data: new Uint8Array(data) } as MIDIMessageEvent);
    send([0x90, 60, 100]); send([0x91, 60, 100]); send([0x80, 60, 0]);
    assert.deepEqual(notes.at(-1), [60]);
    send([0xb1, 123, 0]);
    assert.deepEqual(notes.at(-1), []);
  });
});

test('permission rejection is actionable and leaves no old input connected', async () => {
  await withMidi(() => Promise.reject(Object.assign(new Error('denied'), { name: 'NotAllowedError' })), async () => {
    await assert.rejects(initMidi(() => {}, () => {}), /Allow MIDI.*site.*permissions/);
  });
});
