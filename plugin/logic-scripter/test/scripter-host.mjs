// A small stand-in for Logic Pro's Scripter host: the globals a script sees
// (timing info, parameters, NoteOn/NoteOff, Trace, MIDI) plus a driver that
// feeds process blocks. It records every event the script sends, with the
// beat it was scheduled for, so tests can check the arrangement follows the
// transport. It is not Logic; it mirrors the documented Scripter API only.
import vm from 'node:vm';

export function createHost(script, { tempo = 120, meter = [4, 4], params = {} } = {}) {
  const sent = [];
  const traces = [];
  const timing = {
    playing: false, tempo, meterNumerator: meter[0], meterDenominator: meter[1],
    blockStartBeat: 1, blockEndBeat: 1.25, blockLength: 0.25,
    cycling: false, leftCycleBeat: 1, rightCycleBeat: 5,
  };
  let paramDefs = [];
  const values = {};

  class Event {
    constructor(type) { this.type = type; this.pitch = 60; this.velocity = 100; this.channel = 1; }
    send() { sent.push({ type: this.type, pitch: this.pitch, velocity: this.velocity, beat: null }); }
    sendAtBeat(beat) { sent.push({ type: this.type, pitch: this.pitch, velocity: this.velocity, beat }); }
    sendAfterBeats(beats) { this.sendAtBeat(timing.blockStartBeat + beats); }
  }
  class NoteOn extends Event { constructor() { super('NoteOn'); } }
  class NoteOff extends Event { constructor() { super('NoteOff'); } }

  const context = {
    GetTimingInfo: () => ({ ...timing }),
    GetParameter: (name) => (name in values ? values[name] : defaultOf(name)),
    SetParameter: (name, value) => { values[name] = value; },
    NoteOn, NoteOff,
    MIDI: { allNotesOff: () => sent.push({ type: 'AllNotesOff', pitch: null, velocity: null, beat: null }) },
    Trace: (text) => traces.push(String(text)),
    Math, parseInt, parseFloat, JSON, Number, String, Array, Object, undefined,
  };
  vm.createContext(context);
  vm.runInContext(script, context, { filename: 'theartist_chords.js' });
  paramDefs = context.PluginParameters || [];
  for (const [name, value] of Object.entries(params)) values[name] = value;

  function defaultOf(name) {
    const def = paramDefs.find((p) => p.name === name);
    return def ? def.defaultValue : undefined;
  }

  return {
    sent, traces, timing, context,
    setParam(name, value) {
      values[name] = value;
      const index = paramDefs.findIndex((p) => p.name === name);
      if (typeof context.ParameterChanged === 'function') context.ParameterChanged(index, value);
    },
    /** Run blocks of `step` beats from `from` up to `to` while playing. */
    play(from, to, step = 0.25, patch = {}) {
      Object.assign(timing, patch, { playing: true });
      for (let beat = from; beat < to - 1e-9; beat += step) {
        timing.blockStartBeat = beat;
        timing.blockEndBeat = Math.min(beat + step, to);
        timing.blockLength = timing.blockEndBeat - timing.blockStartBeat;
        context.ProcessMIDI();
      }
    },
    /** One block at an explicit position (used for cycle wraps and locates). */
    block(start, end, patch = {}) {
      Object.assign(timing, patch, { playing: true, blockStartBeat: start, blockEndBeat: end, blockLength: end - start });
      context.ProcessMIDI();
    },
    stop() {
      timing.playing = false;
      context.ProcessMIDI();
    },
    reset() { context.Reset(); },
    incoming(pitch) {
      const on = new NoteOn();
      on.pitch = pitch;
      context.HandleMIDI(on);
    },
  };
}

export function notesOn(host) { return host.sent.filter((e) => e.type === 'NoteOn'); }
export function notesOff(host) { return host.sent.filter((e) => e.type === 'NoteOff'); }
