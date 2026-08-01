interface GuitarFretboardProps {
  frets: number[]; // 6 values: -1=muted, 0=open, n=fret position
  highlightColor?: string;
}

const STRING_NAMES = ['E', 'A', 'D', 'G', 'B', 'e'];
const NUM_FRETS = 5;

// Classic acoustic-guitar palette: rosewood fretboard, ivory bone nut,
// brass-tinted fretwire, polished steel strings on the trebles + warmer
// bronze on the basses, mother-of-pearl dot inlays.
const ROSEWOOD = '#3a2418';      // dark rosewood body
const ROSEWOOD_GRAIN = '#2b1a10';// faint vertical grain
const NUT = '#ece2c2';           // bone / aged ivory nut
const FRET = '#c0a878';          // warm brass fretwire
const FRET_SHADE = '#8a7240';    // shadow under each fretwire
const STRING_BASS = '#b08c5a';   // wound bronze (E, A, D)
const STRING_TREBLE = '#d4d4d4'; // plain steel (G, B, e)
const INLAY = '#ede5cf';         // pearl dot
const FRET_LABEL = '#806b48';    // muted on the rosewood

export default function GuitarFretboard({ frets, highlightColor = 'var(--text-accent)' }: GuitarFretboardProps) {
  // Internal SVG units; scaled to 100% width via viewBox so the panel
  // never gets a horizontal scrollbar.
  const fretWidth = 40;
  const stringSpacing = 16;
  const padding = 22;
  const nutWidth = 4;
  const width = padding + nutWidth + NUM_FRETS * fretWidth + 10;
  const height = padding * 2 + 5 * stringSpacing;

  // Vertical extents of the rosewood board
  const boardTop = padding - stringSpacing / 2;
  const boardBottom = padding + 5 * stringSpacing + stringSpacing / 2;
  const boardHeight = boardBottom - boardTop;
  const boardLeft = padding + nutWidth;
  const boardRight = padding + nutWidth + NUM_FRETS * fretWidth;
  const boardWidth = boardRight - boardLeft;

  // Pearl dot inlays — fret 3 (single dot) on classical guitar
  // conventions. Fret 5 also gets one. Place at the centerline.
  const dotFrets = [3, 5];
  const centerY = padding + 2.5 * stringSpacing;

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      width="100%"
      preserveAspectRatio="xMidYMid meet"
      style={{ height: 'auto', display: 'block' }}
    >
      {/* Rosewood fretboard background */}
      <rect
        x={boardLeft}
        y={boardTop}
        width={boardWidth}
        height={boardHeight}
        fill={ROSEWOOD}
      />
      {/* A pair of subtle vertical grain streaks for warmth */}
      <rect x={boardLeft + boardWidth * 0.15} y={boardTop} width={0.4} height={boardHeight} fill={ROSEWOOD_GRAIN} opacity={0.6} />
      <rect x={boardLeft + boardWidth * 0.65} y={boardTop} width={0.4} height={boardHeight} fill={ROSEWOOD_GRAIN} opacity={0.5} />

      {/* Open-string labels — pinned to the very left edge so they don't
          collide with the mute / open markers sitting just before the nut. */}
      {STRING_NAMES.map((name, i) => (
        <text
          key={name + i}
          x={5}
          y={padding + i * stringSpacing + 4}
          fill="var(--text-muted)"
          fontSize={10}
          textAnchor="middle"
          fontFamily="var(--font-serif)"
          fontStyle="italic"
        >
          {name}
        </text>
      ))}

      {/* Bone nut */}
      <rect
        x={padding}
        y={boardTop}
        width={nutWidth}
        height={boardHeight}
        fill={NUT}
        stroke="#a89970"
        strokeWidth={0.4}
      />

      {/* Mother-of-pearl dot inlays */}
      {dotFrets.map((f) => (
        <circle
          key={`dot-${f}`}
          cx={padding + nutWidth + (f - 0.5) * fretWidth}
          cy={centerY}
          r={2.6}
          fill={INLAY}
          opacity={0.85}
        />
      ))}

      {/* Fretwire — brass with a 1px shadow underneath */}
      {Array.from({ length: NUM_FRETS }).map((_, i) => {
        const x = padding + nutWidth + (i + 1) * fretWidth;
        return (
          <g key={i}>
            <line x1={x + 0.6} y1={boardTop} x2={x + 0.6} y2={boardBottom} stroke={FRET_SHADE} strokeWidth={1.6} />
            <line x1={x} y1={boardTop} x2={x} y2={boardBottom} stroke={FRET} strokeWidth={1.6} />
          </g>
        );
      })}

      {/* Strings — bottom three (E A D) wound bronze, top three plain steel */}
      {Array.from({ length: 6 }).map((_, i) => {
        const isBass = i < 3;
        return (
          <line
            key={i}
            x1={padding}
            y1={padding + i * stringSpacing}
            x2={boardRight}
            y2={padding + i * stringSpacing}
            stroke={isBass ? STRING_BASS : STRING_TREBLE}
            strokeWidth={isBass ? 1.4 : 0.9}
          />
        );
      })}

      {/* Finger positions / muted / open markers */}
      {frets.map((fret, stringIdx) => {
        const y = padding + stringIdx * stringSpacing;

        if (fret === -1) {
          return (
            <text
              key={stringIdx}
              x={padding - 4}
              y={y + 4}
              fill="var(--text-muted)"
              fontSize={11}
              textAnchor="middle"
              fontWeight="bold"
            >
              ×
            </text>
          );
        }
        if (fret === 0) {
          return (
            <circle
              key={stringIdx}
              cx={padding - 4}
              cy={y}
              r={4}
              fill="none"
              stroke="var(--text-soft)"
              strokeWidth={1.4}
            />
          );
        }
        const x = padding + nutWidth + (fret - 0.5) * fretWidth;
        return (
          <circle
            key={stringIdx}
            cx={x}
            cy={y}
            r={6}
            fill={highlightColor}
            stroke="rgba(0,0,0,0.4)"
            strokeWidth={0.6}
          />
        );
      })}

      {/* Fret numbers — under the board, restrained colour so they read
          as labels not as data. */}
      {Array.from({ length: NUM_FRETS }).map((_, i) => (
        <text
          key={i}
          x={padding + nutWidth + (i + 0.5) * fretWidth}
          y={height - 4}
          fill={FRET_LABEL}
          fontSize={9}
          textAnchor="middle"
          fontFamily="var(--font-serif)"
        >
          {i + 1}
        </text>
      ))}
    </svg>
  );
}
