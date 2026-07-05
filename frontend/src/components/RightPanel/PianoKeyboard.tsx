interface PianoKeyboardProps {
  activeNotes: number[]; // MIDI note numbers
  highlightColor?: string;
}

// Render 2 octaves (C3-B4, MIDI 48-71)
const OCTAVE_START = 48;
const NUM_OCTAVES = 2;
const TOTAL_KEYS = NUM_OCTAVES * 12;

const isBlackKey = (noteInOctave: number) =>
  [1, 3, 6, 8, 10].includes(noteInOctave);

const BLACK_KEY_OFFSETS: Record<number, number> = {
  1: 0.6, 3: 1.6, 6: 3.6, 8: 4.6, 10: 5.6,
};

export default function PianoKeyboard({ activeNotes, highlightColor = 'var(--text-accent)' }: PianoKeyboardProps) {
  // Fold every active note into the visible 2-octave window (C3..B4) by
  // octave shifting so chords whose voicing extends outside (e.g. Bmaj9
  // → MIDI 73, Gmaj11 → MIDI 72) still light up every chord tone — only
  // the displayed octave is lost, not the pitch class. Without this,
  // any note above B4 or below C3 silently disappears from the view.
  const VISIBLE_LOW = OCTAVE_START;
  const VISIBLE_HIGH = OCTAVE_START + TOTAL_KEYS - 1;
  const activeSet = new Set(
    activeNotes.map((n) => {
      let m = n;
      while (m < VISIBLE_LOW) m += 12;
      while (m > VISIBLE_HIGH) m -= 12;
      return m;
    }),
  );

  // Internal layout in SVG units. The SVG renders width=100% so it scales
  // to the right-panel column without ever needing horizontal scroll.
  const whiteKeyWidth = 20;
  const blackKeyWidth = 13;
  const whiteKeyHeight = 64;
  const blackKeyHeight = 40;

  const whiteKeys: { midi: number; x: number }[] = [];
  const blackKeys: { midi: number; x: number }[] = [];
  let whiteIndex = 0;

  for (let i = 0; i < TOTAL_KEYS; i++) {
    const midi = OCTAVE_START + i;
    const noteInOctave = i % 12;
    if (isBlackKey(noteInOctave)) {
      const octaveOffset = Math.floor(i / 12);
      const bx = (BLACK_KEY_OFFSETS[noteInOctave] + octaveOffset * 7) * whiteKeyWidth;
      blackKeys.push({ midi, x: bx });
    } else {
      whiteKeys.push({ midi, x: whiteIndex * whiteKeyWidth });
      whiteIndex++;
    }
  }

  const totalWidth = whiteIndex * whiteKeyWidth;
  const totalHeight = whiteKeyHeight + 6;

  // Palette stays in the ebony / rosewood family so the piano still
  // reads as a piano; gradients (defs below) give the keys a hint of
  // dimension without going full 3-D.
  const IVORY_SHADOW = '#d6d6d6';     // groove between keys, floor shadow
  const EBONY = '#171410';            // ebony shade for side facet
  const EBONY_HIGHLIGHT = '#3a3329';  // top sheen on black keys
  const NAMEBOARD = '#3b251a';        // rosewood band above the keys

  return (
    <svg
      viewBox={`0 0 ${totalWidth} ${totalHeight}`}
      width="100%"
      preserveAspectRatio="xMidYMid meet"
      style={{ height: 'auto', display: 'block' }}
    >
      {/* Subtle gradients to give a hint of dimension without going full
          3-D — light-from-above on white keys, top-sheen on black keys,
          tiny floor shadow under the keyboard. */}
      <defs>
        <linearGradient id="white-key-grad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#ffffff" />
          <stop offset="60%" stopColor="#f7f7f7" />
          <stop offset="100%" stopColor="#e6e6e6" />
        </linearGradient>
        <linearGradient id="black-key-grad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#3a3329" />
          <stop offset="20%" stopColor="#221c15" />
          <stop offset="100%" stopColor="#0c0a08" />
        </linearGradient>
      </defs>

      {/* Rosewood nameboard at the top — visual context, not a key */}
      <rect x={0} y={0} width={totalWidth} height={3} fill={NAMEBOARD} />

      {/* White keys — gradient fill for a top-lit look; floor shadow line
          underneath. Active key swaps the gradient for the highlight
          colour so the chord notes still pop. */}
      {whiteKeys.map(({ midi, x }) => {
        const active = activeSet.has(midi);
        return (
          <g key={midi}>
            <rect
              x={x}
              y={3}
              width={whiteKeyWidth}
              height={whiteKeyHeight}
              fill={active ? highlightColor : 'url(#white-key-grad)'}
              stroke={IVORY_SHADOW}
              strokeWidth={0.5}
            />
            {/* Floor shadow — bottom edge of the key looks recessed. */}
            <rect
              x={x}
              y={3 + whiteKeyHeight - 2}
              width={whiteKeyWidth}
              height={2}
              fill={IVORY_SHADOW}
              opacity={0.7}
            />
          </g>
        );
      })}

      {/* Black keys — gradient + thin top sheen for a glossy ebony look.
          Active key swaps the gradient for the highlight colour. */}
      {blackKeys.map(({ midi, x }) => {
        const active = activeSet.has(midi);
        return (
          <g key={midi}>
            <rect
              x={x + 0.5}
              y={3}
              width={blackKeyWidth}
              height={blackKeyHeight}
              fill={active ? highlightColor : 'url(#black-key-grad)'}
            />
            {/* Top sheen — half-pixel highlight, sells the lit-from-above
                read without going gaudy. */}
            <rect
              x={x + 0.5}
              y={3}
              width={blackKeyWidth}
              height={1}
              fill={active ? highlightColor : EBONY_HIGHLIGHT}
              opacity={active ? 1 : 0.85}
            />
            {/* Side shadow — left edge of black key dropped 1px to add
                a thin facet line. */}
            <rect
              x={x + 0.5}
              y={3}
              width={0.5}
              height={blackKeyHeight}
              fill={EBONY}
              opacity={active ? 0 : 0.5}
            />
          </g>
        );
      })}
    </svg>
  );
}
