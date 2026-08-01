import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { registerHooks } from 'node:module';
import { transpileModule, JsxEmit, ModuleKind } from 'typescript';
import { manualChordDraft, manualDraftChord } from './manualChordDraft.ts';
import type { Chord, SlotAddress, Song } from '../models/types.ts';

function chord(root: string, quality = 'maj', extensions: string[] = [], bass?: string): Chord {
  return { root, quality, extensions, bass, romanNumeral: '', voicing: [] };
}

test('selected symbols round-trip without losing minor, suspended, altered or slash harmony', () => {
  for (const original of [
    chord('F'), chord('Bb', 'm7', ['b9'], 'D'), chord('G', '7sus4', ['b13']),
    chord('D', 'm7b5'), chord('Ab', 'mMaj7'), chord('C#', 'm(add9)'),
    chord('E', 'maj13(#11)', ['add6'], 'G#'),
  ]) {
    assert.deepEqual(manualDraftChord(manualChordDraft(original)), original);
  }
  assert.deepEqual(manualDraftChord(manualChordDraft(null)), chord('C'));
  assert.equal(manualChordDraft(chord('Bb', 'm7')).base, 'min');
  assert.equal(manualChordDraft(chord('Bb', 'm7')).ext, '7');
  assert.equal(manualChordDraft(chord('G', '7sus4')).sus, 'sus4');
});

test('root and bass edits retain an unsupported quality until its quality is explicitly changed', () => {
  const draft = manualChordDraft(chord('E', 'maj13(#11)', ['add6'], 'G#'));
  assert.deepEqual(manualDraftChord({ ...draft, root: 'F', bass: 'A' }), chord('F', 'maj13(#11)', ['add6'], 'A'));
  assert.equal(manualDraftChord({ ...draft, base: 'min', ext: '7', originalQuality: undefined }).quality, 'm7');
});

// Exercise the actual component's selected-slot input and Apply handler without
// a browser or network. The hook harness retains state between renders and
// re-renders conditional state adjustments before exposing the rendered tree.
const SRC_URL = new URL('../', import.meta.url).href;
const COMPONENT_URL = `${SRC_URL}components/RightPanel/ChordActions.tsx`;
const HOOKS_URL = 'data:text/javascript,' + encodeURIComponent(`
  export const useState = (...args) => globalThis.manualChordTestHooks.useState(...args);
  export const useCallback = callback => callback;
  export const useEffect = () => {};
  export const useId = () => 'manual-test';
`);
registerHooks({
  resolve(specifier, context, nextResolve) {
    if (context.parentURL === COMPONENT_URL && specifier === 'react') {
      return { url: HOOKS_URL, shortCircuit: true };
    }
    if (specifier === 'zustand/react/shallow' && context.parentURL === COMPONENT_URL) {
      return { url: 'data:text/javascript,export const useShallow = selector => selector;', shortCircuit: true };
    }
    if (context.parentURL?.startsWith(SRC_URL) && /^\.\.?\//.test(specifier) && !/\.\w+$/.test(specifier)) {
      specifier += /\/(GenerateProgress|OperationProgress)$/.test(specifier) ? '.tsx' : '.ts';
    }
    return nextResolve(specifier, context);
  },
  load(url, context, nextLoad) {
    if (url.startsWith(SRC_URL) && url.endsWith('.css')) return { format: 'module', shortCircuit: true, source: '' };
    const stores: Record<string, string> = {
      'songStore': 'export const useSongStore = selector => selector(globalThis.manualChordTestSongState);',
      'generationStore': `export const useGenerationStore = selector => selector({
        suggestions: [], loading: false, showSuggestions: false, generateSectionType: 'auto', modelKey: 'test',
      });`,
      'sessionStore': 'export const selectStarredSessions = () => []; export const useSessionStore = selector => selector({});',
      'reviewStore': 'export const useReviewStore = () => {}; export const reasonFact = () => {}; export const rulesSlotCount = () => 0;',
      'reharmonizeStore': 'export const useReharmonizeStore = selector => selector({ loading: false, comparison: null });',
    };
    for (const [name, source] of Object.entries(stores)) {
      if (url === `${SRC_URL}store/${name}.ts`) return { format: 'module', shortCircuit: true, source };
    }
    if (url.startsWith(SRC_URL) && url.endsWith('.tsx')) {
      return {
        format: 'module', shortCircuit: true,
        source: transpileModule(readFileSync(new URL(url), 'utf8'), {
          compilerOptions: { jsx: JsxEmit.ReactJSX, module: ModuleKind.ESNext },
        }).outputText,
      };
    }
    return nextLoad(url, context);
  },
});
const { default: ChordActions } = await import('../components/RightPanel/ChordActions.tsx');

interface ElementNode {
  type?: unknown;
  props?: { children?: unknown; onClick?: () => void; className?: string };
}
function nodes(tree: unknown): ElementNode[] {
  if (Array.isArray(tree)) return tree.flatMap(nodes);
  if (!tree || typeof tree !== 'object') return [];
  const node = tree as ElementNode;
  return [node, ...nodes(node.props?.children)];
}

test('selection, external chord replacement and song replacement hydrate Set Chord before Apply', () => {
  const song: Song = {
    id: 'imported', title: 'Twenty bars', key: 'C major', bpm: 120,
    genre: 'pop', timeSignature: [4, 4], autoTitle: false, sections: [],
    measures: Array.from({ length: 20 }, (_, index) => ({
      id: `bar-${index}`, index,
      chords: [chord(index >= 8 && index < 16 ? 'F' : 'C'), null],
    })),
  };
  const applied: { slot: SlotAddress; chord: Chord }[] = [];
  const state = {
    song, songEpoch: 1, selectedSlots: [{ measureIndex: 0, slotIndex: 0 }],
    setChord: (slot: SlotAddress, value: Chord) => applied.push({ slot, chord: value }),
  };
  let values: unknown[] = [];
  let cursor = 0;
  let changed = false;
  Object.defineProperty(globalThis, 'manualChordTestSongState', { configurable: true, value: state });
  Object.defineProperty(globalThis, 'manualChordTestHooks', {
    configurable: true, value: {
      useState(initial: unknown) {
        const index = cursor++;
        if (!(index in values)) values[index] = typeof initial === 'function' ? initial() : initial;
        return [values[index], (value: unknown) => {
          values[index] = typeof value === 'function' ? value(values[index]) : value;
          changed = true;
        }];
      },
    },
  });
  function render() {
    for (let attempt = 0; attempt < 10; attempt++) {
      cursor = 0;
      changed = false;
      const tree = ChordActions({ view: 'manual' });
      if (!changed) return nodes(tree);
    }
    throw new Error('Repeated state updates during render');
  }
  function click(label: string) {
    const button = render().find(node => node.type === 'button' && node.props?.children === label);
    assert.ok(button?.props?.onClick, `button ${label} exists`);
    button.props.onClick();
  }
  function previewRoot() {
    return render().find(node => node.props?.className === 'text-2xl font-bold')?.props?.children;
  }

  assert.equal(previewRoot(), 'C');
  click('D');
  assert.equal(previewRoot(), 'D');
  state.song = { ...state.song, title: 'Renamed' };
  assert.equal(previewRoot(), 'D', 'unrelated edits retain the unfinished draft');

  state.selectedSlots = [{ measureIndex: 8, slotIndex: 0 }];
  assert.equal(previewRoot(), 'F', 'Review imported bar 9 immediately shows its F chord');
  click('Apply');
  assert.deepEqual(applied.at(-1), { slot: { measureIndex: 8, slotIndex: 0 }, chord: chord('F') });

  const replacement = chord('Bb', 'm7', ['b9'], 'D');
  state.song.measures[8].chords[0] = replacement;
  assert.equal(previewRoot(), 'Bb', 'generation at the same selected slot updates the draft');
  click('Apply');
  assert.deepEqual(applied.at(-1)?.chord, replacement);
  click('A');
  state.songEpoch++;
  assert.equal(previewRoot(), 'Bb', 'opening a song discards an earlier unfinished draft');

  state.selectedSlots = [{ measureIndex: 8, slotIndex: 1 }];
  assert.equal(previewRoot(), 'C', 'an empty slot starts a fresh C major draft');
  state.selectedSlots = [{ measureIndex: 8, slotIndex: 0 }];
  assert.equal(previewRoot(), 'Bb', 'returning to a slot reads its saved chord');
  values = [];
  assert.equal(previewRoot(), 'Bb', 'mounting directly on the selected chord hydrates it');
});
