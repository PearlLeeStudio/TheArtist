import type { Chord } from '../../models/types';
import { generatePianoVoicing } from '../../engine/voicingEngine';
import { NOTE_VALUES } from '../../engine/constants';

interface ChordTransitionViewProps {
  prev: Chord | null;
  current: Chord;
  next: Chord | null;
}

/**
 * Three-chord context view on a single piano. Visual language:
 *
 *   • CURRENT (active interpretation) — full-key fill in brand yellow.
 *     Always rendered last so it sits above everything else.
 *   • PREV — thin purple strip just below the keyboard, in each prev
 *     note's column. Tab-underline aesthetic. No key fill, no arrow.
 *   • NEXT — thin green strip just above the keyboard, in each next
 *     note's column, plus curved arrows from current → next so the
 *     voice-leading direction reads at a glance.
 *
 * The strips sit OUTSIDE the keyboard rectangle so they never compete
 * with the current chord's full-key fill (the user can always read all
 * three chords even when keys are shared between them).
 */

const OCTAVE_START = 48;       // C3
const NUM_OCTAVES = 2;
const TOTAL_KEYS = NUM_OCTAVES * 12;

const PREV_COLOR = '#a78bfa';     // light purple (Tailwind violet-400)
const CURR_COLOR = '#facc15';
const NEXT_COLOR = '#86efac';     // light green (Tailwind green-300)

// Keyboard geometry (mirrors PianoKeyboard.tsx).
const WHITE_W = 20;
const BLACK_W = 13;
const WHITE_H = 64;
const BLACK_H = 40;
const NAMEBOARD = 3;             // rosewood band thickness at the top of the keyboard
const BLACK_OFFSETS: Record<number, number> = { 1: 0.6, 3: 1.6, 6: 3.6, 8: 4.6, 10: 5.6 };

// Strip + arrow band layout (vertical stack, top → bottom):
//   ARROW_BAND px of room for curr→next arrow arcs
//   STRIP_H px green strip row (next markers)
//   STRIP_GAP px gap
//   keyboard (NAMEBOARD + WHITE_H)
//   STRIP_GAP px gap
//   STRIP_H px purple strip row (prev markers)
//   ARROW_BAND_BOTTOM px of room for prev→curr arrow arcs
const ARROW_BAND = 28;
const ARROW_BAND_BOTTOM = 24;
const STRIP_H = 3;
const STRIP_GAP = 2;

const NEXT_STRIP_Y = ARROW_BAND - STRIP_H - 1;      // just above the keyboard top
const KEYBOARD_TOP_Y = ARROW_BAND;                  // SVG y of the keyboard's nameboard
const KEYBOARD_BOTTOM_Y = ARROW_BAND + NAMEBOARD + WHITE_H;
const PREV_STRIP_Y = KEYBOARD_BOTTOM_Y + STRIP_GAP;
const TOTAL_H = PREV_STRIP_Y + STRIP_H + ARROW_BAND_BOTTOM;

const isBlackKey = (n: number) => [1, 3, 6, 8, 10].includes(n % 12);

interface KeyPos { midi: number; x: number; isBlack: boolean }

function buildKeyMap(): Map<number, KeyPos> {
  const map = new Map<number, KeyPos>();
  let whiteIndex = 0;
  for (let i = 0; i < TOTAL_KEYS; i++) {
    const midi = OCTAVE_START + i;
    const noteInOctave = i % 12;
    if (isBlackKey(midi)) {
      const octaveOffset = Math.floor(i / 12);
      const x = (BLACK_OFFSETS[noteInOctave] + octaveOffset * 7) * WHITE_W;
      map.set(midi, { midi, x, isBlack: true });
    } else {
      map.set(midi, { midi, x: whiteIndex * WHITE_W, isBlack: false });
      whiteIndex++;
    }
  }
  return map;
}

const KEY_MAP = buildKeyMap();
const TOTAL_WHITE = [...KEY_MAP.values()].filter(k => !k.isBlack).length;
const TOTAL_W = TOTAL_WHITE * WHITE_W;

/** Fold any MIDI note into the visible 2-octave window (C3..B4) by
 *  octave-shifting. Pitch class is preserved, only the octave changes
 *  — so the key-column visualisation is always populated even when
 *  the player voiced the chord across a wider range than fits on
 *  screen. */
function intoVisibleOctave(midi: number): number {
  let m = midi;
  while (m < OCTAVE_START) m += 12;
  while (m >= OCTAVE_START + TOTAL_KEYS) m -= 12;
  return m;
}

/** All notes of a chord, untouched. Voice-pair sorting reads the
 *  original MIDI values (so octave order is preserved); rendering
 *  uses `intoVisibleOctave` separately. */
function chordNotes(c: Chord | null): number[] {
  if (!c) return [];
  return c.voicing && c.voicing.length > 0 ? c.voicing : generatePianoVoicing(c);
}

/** Pitch-classes (visible-range MIDI numbers) for keyboard rendering.
 *  Dedups so a unison-doubled note doesn't render its column twice. */
function chordVisibleKeys(notes: number[]): number[] {
  return [...new Set(notes.map(intoVisibleOctave))];
}

/** Sorted-by-pitch voice pairing — close-to-optimal for ≤4-note chords.
 *
 *  Cardinality mismatch (e.g. 4-note → 3-note transition) drops the
 *  TOP of the larger chord, intentionally. Musically this matches how
 *  jazz arrangers think: the top note of a 7-chord is the 7th (a
 *  tension / colour tone), not a structural bass-3rd-5th voice — when
 *  voice-leading down to a triad, the tension simply resolves out
 *  rather than mapping to anything. `Math.min(a.length, b.length)`
 *  combined with `sort ascending` produces exactly that: the unpaired
 *  voice is always the highest of whichever chord has more notes.
 */
function pairVoices(a: number[], b: number[]): Array<[number, number]> {
  const n = Math.min(a.length, b.length);
  if (n === 0) return [];
  const sortedA = [...a].sort((x, y) => x - y);
  const sortedB = [...b].sort((x, y) => x - y);
  return Array.from({ length: n }, (_, i) => [sortedA[i], sortedB[i]] as [number, number]);
}

function keyColumn(midi: number): { x: number; w: number } | null {
  const k = KEY_MAP.get(intoVisibleOctave(midi));
  if (!k) return null;
  return k.isBlack
    ? { x: k.x + 0.5, w: BLACK_W }
    : { x: k.x, w: WHITE_W };
}

function keyCenterX(midi: number): number | null {
  const c = keyColumn(midi);
  return c ? c.x + c.w / 2 : null;
}

/** Curved arrow path from current key top → next strip column top. */
function nextArrowPath(fromX: number, toX: number): string {
  const span = Math.abs(toX - fromX);
  const lift = 12 + Math.min(20, span * 0.22);
  const fromY = KEYBOARD_TOP_Y;
  const toY = NEXT_STRIP_Y + STRIP_H / 2;
  const cx = (fromX + toX) / 2;
  const cy = Math.min(fromY, toY) - lift;
  return `M ${fromX} ${fromY} Q ${cx} ${cy} ${toX} ${toY}`;
}

/** Curved arrow path from prev strip column → current key bottom.
 *  Mirrors `nextArrowPath` but routes the arc DOWN through the bottom
 *  band so it doesn't cut through the keyboard rectangle. */
function prevArrowPath(fromX: number, toX: number): string {
  const span = Math.abs(toX - fromX);
  const drop = 12 + Math.min(20, span * 0.22);
  const fromY = PREV_STRIP_Y + STRIP_H / 2;
  const toY = KEYBOARD_BOTTOM_Y;
  const cx = (fromX + toX) / 2;
  const cy = fromY + drop;
  return `M ${fromX} ${fromY} Q ${cx} ${cy} ${toX} ${toY}`;
}

const IVORY = '#f7f7f7';
const IVORY_SHADOW = '#d6d6d6';
const EBONY = '#171410';
const EBONY_HIGHLIGHT = '#3a3329';
const NAMEBOARD_COLOR = '#3b251a';

export default function ChordTransitionView({ prev, current, next }: ChordTransitionViewProps) {
  // INPUT first, no truncation — keyboard rendering is always faithful
  // to what was played. Transposition into the visible window is a
  // RENDERING concern only.
  const prevNotes = chordNotes(prev);
  const currNotes = chordNotes(current);
  const nextNotes = chordNotes(next);

  // Visible-range pitch classes for the keyboard fills + strips.
  const prevVisible = chordVisibleKeys(prevNotes);
  const currVisible = chordVisibleKeys(currNotes);
  const nextVisible = chordVisibleKeys(nextNotes);
  const currSet = new Set(currVisible);

  // Root + bass markers — these change when the user clicks a different
  // candidate chip even though the played pitch classes are identical
  // (C-E-G-A is unambiguously those 4 keys, but C6 vs Am7/C interpret
  // them with different roots/basses). The dot moves to show which note
  // is the "1" of the active interpretation.
  const rootPC = NOTE_VALUES[current.root];
  const bassPC = current.bass ? NOTE_VALUES[current.bass] : undefined;
  const rootKeys = currVisible.filter((m) => m % 12 === rootPC);
  const bassKeys = bassPC !== undefined && bassPC !== rootPC
    ? currVisible.filter((m) => m % 12 === bassPC)
    : [];

  // Voice pairings use the ORIGINAL MIDI values so the lowest voice of
  // prev maps to the lowest voice of curr, etc., even if the player
  // voiced the chord across a range wider than the visible keyboard.
  const prevToCurr = pairVoices(prevNotes, currNotes);
  const currToNext = pairVoices(currNotes, nextNotes);

  const whiteKeys = [...KEY_MAP.values()].filter(k => !k.isBlack);
  const blackKeys = [...KEY_MAP.values()].filter(k => k.isBlack);

  return (
    <svg
      viewBox={`0 0 ${TOTAL_W} ${TOTAL_H}`}
      width="100%"
      preserveAspectRatio="xMidYMid meet"
      style={{ height: 'auto', display: 'block' }}
    >
      {/* NEXT strips — thin green markers above the keyboard. */}
      {nextVisible.map((midi) => {
        const c = keyColumn(midi);
        if (!c) return null;
        return (
          <rect
            key={`next-${midi}`}
            x={c.x}
            y={NEXT_STRIP_Y}
            width={c.w}
            height={STRIP_H}
            fill={NEXT_COLOR}
            rx={1}
          />
        );
      })}

      {/* Keyboard (translated down by ARROW_BAND so arrows have room). */}
      <g transform={`translate(0, ${ARROW_BAND})`}>
        {/* Rosewood nameboard band */}
        <rect x={0} y={0} width={TOTAL_W} height={NAMEBOARD} fill={NAMEBOARD_COLOR} />

        {/* White keys */}
        {whiteKeys.map(({ midi, x }) => {
          const isCurr = currSet.has(midi);
          return (
            <g key={`w-${midi}`}>
              <rect
                x={x}
                y={NAMEBOARD}
                width={WHITE_W}
                height={WHITE_H}
                fill={isCurr ? CURR_COLOR : IVORY}
                stroke={IVORY_SHADOW}
                strokeWidth={0.5}
              />
              <rect
                x={x}
                y={NAMEBOARD + WHITE_H - 2}
                width={WHITE_W}
                height={2}
                fill={IVORY_SHADOW}
                opacity={0.7}
              />
            </g>
          );
        })}

        {/* Black keys — drawn on top of whites so they stack correctly. */}
        {blackKeys.map(({ midi, x }) => {
          const isCurr = currSet.has(midi);
          return (
            <g key={`b-${midi}`}>
              <rect
                x={x + 0.5}
                y={NAMEBOARD}
                width={BLACK_W}
                height={BLACK_H}
                fill={isCurr ? CURR_COLOR : EBONY}
              />
              <rect
                x={x + 0.5}
                y={NAMEBOARD}
                width={BLACK_W}
                height={1}
                fill={isCurr ? CURR_COLOR : EBONY_HIGHLIGHT}
              />
            </g>
          );
        })}

        {/* Root marker — solid black dot near the bottom of the root key.
            Moves when the user picks a different interpretation chip. */}
        {rootKeys.map((midi) => {
          const k = KEY_MAP.get(midi);
          if (!k) return null;
          const cx = k.isBlack ? k.x + 0.5 + BLACK_W / 2 : k.x + WHITE_W / 2;
          const cy = k.isBlack
            ? NAMEBOARD + BLACK_H - 7
            : NAMEBOARD + WHITE_H - 9;
          return (
            <circle
              key={`root-${midi}`}
              cx={cx}
              cy={cy}
              r={4}
              fill="#1a1a1a"
            />
          );
        })}

        {/* Bass marker — open ring on slash-chord bass note (when bass ≠ root). */}
        {bassKeys.map((midi) => {
          const k = KEY_MAP.get(midi);
          if (!k) return null;
          const cx = k.isBlack ? k.x + 0.5 + BLACK_W / 2 : k.x + WHITE_W / 2;
          const cy = k.isBlack
            ? NAMEBOARD + BLACK_H - 7
            : NAMEBOARD + WHITE_H - 9;
          return (
            <circle
              key={`bass-${midi}`}
              cx={cx}
              cy={cy}
              r={4}
              fill="none"
              stroke="#1a1a1a"
              strokeWidth={1.6}
            />
          );
        })}
      </g>

      {/* PREV strips — thin purple markers below the keyboard. */}
      {prevVisible.map((midi) => {
        const c = keyColumn(midi);
        if (!c) return null;
        return (
          <rect
            key={`prev-${midi}`}
            x={c.x}
            y={PREV_STRIP_Y}
            width={c.w}
            height={STRIP_H}
            fill={PREV_COLOR}
            rx={1}
          />
        );
      })}

      {/* prev → current voice-leading arrows (bottom band, arc below) */}
      {prev && prevToCurr.map(([from, to], i) => {
        const fx = keyCenterX(from);
        const tx = keyCenterX(to);
        if (fx == null || tx == null) return null;
        return (
          <path
            key={`pc-${i}`}
            d={prevArrowPath(fx, tx)}
            stroke={PREV_COLOR}
            strokeWidth={1.2}
            fill="none"
            opacity={0.9}
            markerEnd="url(#arrowhead-prev)"
          />
        );
      })}

      {/* current → next voice-leading arrows (top band, arc above) */}
      {next && currToNext.map(([from, to], i) => {
        const fx = keyCenterX(from);
        const tx = keyCenterX(to);
        if (fx == null || tx == null) return null;
        return (
          <path
            key={`cn-${i}`}
            d={nextArrowPath(fx, tx)}
            stroke={NEXT_COLOR}
            strokeWidth={1.2}
            fill="none"
            opacity={0.9}
            markerEnd="url(#arrowhead-next)"
          />
        );
      })}

      {/* Arrow-head defs */}
      <defs>
        <marker id="arrowhead-prev" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
          <path d="M0,0 L5,3 L0,6 Z" fill={PREV_COLOR} />
        </marker>
        <marker id="arrowhead-next" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
          <path d="M0,0 L5,3 L0,6 Z" fill={NEXT_COLOR} />
        </marker>
      </defs>

      {/* Tiny legend, top-left */}
      <g transform="translate(2, 4)">
        <circle cx={3} cy={4} r={3} fill={PREV_COLOR} />
        <text x={9} y={6} fontSize={7} fill="var(--text-muted)" fontFamily="var(--font-serif)">prev</text>
        <circle cx={42} cy={4} r={3} fill="#facc15" />
        <text x={48} y={6} fontSize={7} fill="var(--text-muted)" fontFamily="var(--font-serif)">now</text>
        <circle cx={78} cy={4} r={3} fill={NEXT_COLOR} />
        <text x={84} y={6} fontSize={7} fill="var(--text-muted)" fontFamily="var(--font-serif)">next</text>
      </g>
    </svg>
  );
}
