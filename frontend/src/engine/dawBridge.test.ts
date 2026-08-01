import { test } from 'node:test';
import type { TestContext } from 'node:test';
import assert from 'node:assert/strict';
import { openDawBridge } from './dawBridge.ts';
import { MIDI_BRIDGE_ORIGIN } from './midiBridge.ts';

function browser(context: TestContext) {
  const originalWindow = Object.getOwnPropertyDescriptor(globalThis, 'window');
  const originalLocation = Object.getOwnPropertyDescriptor(globalThis, 'location');
  const listeners = new Set<(event: MessageEvent) => void>();
  const popups: { closed: boolean; token: string; messages: unknown[]; close(): void; postMessage(value: unknown): void }[] = [];
  Object.defineProperty(globalThis, 'location', { configurable: true, value: { origin: 'https://artist.pearlleestudio.co' } });
  Object.defineProperty(globalThis, 'window', { configurable: true, value: {
    open(url: string) {
      const popup = { closed: false, token: new URL(url).searchParams.get('token')!, messages: [] as unknown[],
        close() { this.closed = true; }, postMessage(value: unknown) { this.messages.push(value); } };
      popups.push(popup);
      return popup;
    },
    addEventListener(_: string, listener: (event: MessageEvent) => void) { listeners.add(listener); },
    removeEventListener(_: string, listener: (event: MessageEvent) => void) { listeners.delete(listener); },
  } });
  context.after(() => {
    if (originalWindow) Object.defineProperty(globalThis, 'window', originalWindow); else Reflect.deleteProperty(globalThis, 'window');
    if (originalLocation) Object.defineProperty(globalThis, 'location', originalLocation); else Reflect.deleteProperty(globalThis, 'location');
  });
  return { popups, listeners, emit(index: number, data: object) {
    const popup = popups[index];
    for (const listener of listeners) listener({ origin: MIDI_BRIDGE_ORIGIN, source: popup,
      data: { protocol: 'theartist-daw-v1', token: popup.token, ...data } } as unknown as MessageEvent);
  } };
}

test('an unavailable bridge releases its window and allows a fresh successful retry', async context => {
  const ui = browser(context);
  context.mock.timers.enable({ apis: ['setTimeout', 'setInterval'] });
  const first = openDawBridge(new AbortController().signal);
  const rejected = assert.rejects(first.ready, /automatic startup/);
  context.mock.timers.tick(8000);
  await rejected;
  assert.equal(ui.popups[0].closed, true);
  assert.equal(ui.listeners.size, 0);
  context.mock.timers.reset();
  const retry = openDawBridge(new AbortController().signal);
  try {
    const pending = retry.send('logic', 'Retry', new Blob(['MThd']));
    ui.emit(0, { type: 'ready' });
    await new Promise(resolve => setTimeout(resolve, 0));
    assert.equal(ui.popups[1].messages.length, 0);
    ui.emit(1, { type: 'ready' });
    await new Promise(resolve => setTimeout(resolve, 0));
    assert.equal(ui.popups[1].messages.length, 1);
    ui.emit(1, { type: 'result', action: 'open' });
    assert.equal(await pending, 'open');
    assert.equal(ui.listeners.size, 0);
    assert.equal(ui.popups[1].closed, true);
  } finally { retry.close(); }
});

test('cancel and close settle pending handoffs and leave no listeners', async context => {
  const ui = browser(context);
  const controller = new AbortController();
  const bridge = openDawBridge(controller.signal);
  const pending = bridge.send('logic', 'Cancelled', new Blob(['MThd']));
  controller.abort();
  await assert.rejects(pending, /cancelled/);
  assert.equal(ui.listeners.size, 0);
  assert.equal(ui.popups[0].closed, true);
  assert.throws(() => openDawBridge(controller.signal), /cancelled/);
  const next = openDawBridge(new AbortController().signal);
  next.close();
  await assert.rejects(next.ready, /cancelled/);
  assert.equal(ui.listeners.size, 0);
});

test('DAW handoff accepts only its own popup, origin and nonce, and cleans up', async () => {
  const originalWindow = Object.getOwnPropertyDescriptor(globalThis, 'window');
  const originalLocation = Object.getOwnPropertyDescriptor(globalThis, 'location');
  const listeners = new Set<(event: MessageEvent) => void>();
  const sent: unknown[][] = [];
  let token = '';
  const popup = { closed: false, postMessage: (...args: unknown[]) => sent.push(args), close() { this.closed = true; } };
  Object.defineProperty(globalThis, 'location', { configurable: true, value: { origin: 'https://artist.pearlleestudio.co' } });
  Object.defineProperty(globalThis, 'window', { configurable: true, value: {
    open(url: string) { token = new URL(url).searchParams.get('token')!; return popup; },
    addEventListener(_: string, listener: (event: MessageEvent) => void) { listeners.add(listener); },
    removeEventListener(_: string, listener: (event: MessageEvent) => void) { listeners.delete(listener); },
  } });
  const bridge = openDawBridge(new AbortController().signal);
  const emit = (data: object, origin = MIDI_BRIDGE_ORIGIN, source: unknown = popup) => {
    for (const listener of listeners) listener({ origin, source, data: { protocol: 'theartist-daw-v1', token, ...data } } as MessageEvent);
  };
  try {
    const pending = bridge.send('logic', 'Test score', new Blob(['MThd']));
    emit({type:'ready'}, 'https://attacker.example');
    emit({type:'ready'}, MIDI_BRIDGE_ORIGIN, {});
    emit({type:'ready', token:'wrong'});
    await new Promise(resolve => setTimeout(resolve, 0));
    assert.equal(sent.length, 0);
    emit({type:'ready'});
    await new Promise(resolve => setTimeout(resolve, 0));
    assert.equal(sent.length, 1);
    assert.equal(sent[0][1], MIDI_BRIDGE_ORIGIN);
    emit({type:'result',action:'open'});
    assert.equal(await pending, 'open');
  } finally {
    bridge.close();
    assert.equal(popup.closed, true);
    assert.equal(listeners.size, 0);
    if (originalWindow) Object.defineProperty(globalThis, 'window', originalWindow); else Reflect.deleteProperty(globalThis, 'window');
    if (originalLocation) Object.defineProperty(globalThis, 'location', originalLocation); else Reflect.deleteProperty(globalThis, 'location');
  }
});
