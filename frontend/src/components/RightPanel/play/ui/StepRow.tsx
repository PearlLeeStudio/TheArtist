import type { CSSProperties } from 'react';

interface StepCell {
  /** 0 = off; 0..1 = on at that brightness (ghost notes dim but visible). */
  level: number;
  /** Alternate articulation, shown in a lighter green. */
  alt?: boolean;
}

interface StepRowProps {
  cells: StepCell[];
  /** Index of the cell playing now; null or out of range = none. */
  playhead?: number | null;
  /** e.g. 'K', 'HARM', 'BASS' — shown in a fixed-width column on the left. */
  label?: string;
  /** Whole row muted (e.g. bass off). */
  dim?: boolean;
}

/** Flat LED row height in pixels. */
const ROW_HEIGHT = 10;
/** Label type size in pixels. */
const LABEL_SIZE = 11;
/** Fits a four-letter label at LABEL_SIZE. */
const LABEL_WIDTH = 42;
const LABEL_GAP = 4;
const CELL_GAP = 2;
/** Cells per group: one beat of 16ths. */
const GROUP = 4;
/** Added to CELL_GAP at each group boundary. */
const GROUP_GAP = 3;

function cellStyle(cell: StepCell, here: boolean, groupStart: boolean): CSSProperties {
  const style: CSSProperties = {
    flex: '1 1 0',
    minWidth: 0,
    borderRadius: 0,
    marginLeft: groupStart ? GROUP_GAP : undefined,
  };
  const level = Math.min(1, Math.max(0, cell.level || 0));
  if (level === 0) {
    style.background = here
      ? '#294b20'
      : '#182015';
    if (here) style.outline = '1px solid #d6ffbc';
    return style;
  }
  const color = cell.alt ? '#c8ff67' : '#8bff4c';
  // The playing step flashes at full brightness whatever its level.
  style.background = here || level >= 1
    ? color
    : `color-mix(in srgb, ${color} ${Math.round(25 + 75 * level)}%, #172211)`;
  if (here) {
    style.outline = '1px solid #e6ffda';
    style.outlineOffset = -1;
  }
  return style;
}

/** Flat green LEDs on a black meter, with the live beat marked. */
export default function StepRow({ cells, playhead = null, label, dim = false }: StepRowProps) {
  const onSteps = cells.flatMap((cell, i) => (cell.level > 0 ? [i + 1] : []));
  const prefix = label ? `${label}: ` : '';
  const steps = onSteps.length > 0 ? `steps ${onSteps.join(' ')}` : 'no steps';
  const name = `${prefix}${steps}${dim ? ', muted' : ''}`;

  return (
    <div
      role="img"
      aria-label={name}
      className="flex w-full min-w-0 items-center"
      style={{
        height: ROW_HEIGHT,
        opacity: dim ? 0.35 : 1,
        filter: dim ? 'saturate(0.4)' : undefined,
        transition: 'opacity 0.15s ease',
      }}
    >
      {label !== undefined && (
        <span
          className="shrink-0 truncate uppercase"
          style={{
            width: LABEL_WIDTH,
            marginRight: LABEL_GAP,
            fontFamily: 'var(--font-device)',
            fontSize: LABEL_SIZE,
            lineHeight: 1,
            letterSpacing: '0.06em',
            color: 'var(--key-text-dim)',
          }}
        >
          {label}
        </span>
      )}
      <div className="te-step-meter flex min-w-0 flex-1 self-stretch" style={{ gap: CELL_GAP }}>
        {cells.map((cell, i) => (
          <span
            key={i}
            style={cellStyle(cell, i === playhead, i > 0 && i % GROUP === 0)}
          />
        ))}
      </div>
    </div>
  );
}
