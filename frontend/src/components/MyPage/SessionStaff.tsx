import { useEffect, useRef, useMemo } from 'react';
import { getChordDisplayParts } from '../../engine/chordDisplay';
import { SCORE_MEASURES_PER_ROW, SCORE_ROW_HEIGHT, SCORE_TOP } from '../../engine/scoreLayout';
import { scoreStaveLayout } from '../../engine/scoreStaveLayout';
import type { Song } from '../../models/types';
import { keySignatureFor } from '../../engine/keySignatures';
import { useStaffViewport, drawScoreStaves } from '../ScoreEditor/scoreSurface';

/** Space below the last row; SCORE_TOP holds the space above the first. */
const BOTTOM_PADDING = 10;

interface SessionStaffProps {
  song: Song;
}

export default function SessionStaff({ song }: SessionStaffProps) {
  const wrapperRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const { containerWidth, theme } = useStaffViewport(wrapperRef);
  // Project cards always show only the first staff row; exports keep the complete song.
  const visibleCount = Math.min(song.measures.length, SCORE_MEASURES_PER_ROW);
  const visibleMeasures = song.measures.slice(0, visibleCount);

  const staveLayouts = useMemo(() => {
    return Array.from({ length: visibleCount }, (_, i) => scoreStaveLayout(i, containerWidth));
  }, [visibleCount, containerWidth]);

  const rows = visibleCount ? 1 : 0;
  const lastLayout = staveLayouts.at(-1);
  // The shared layout keeps bars readable on narrow screens; make that full
  // width scrollable instead of clipping its rightmost bars inside the SVG.
  const totalWidth = Math.max(containerWidth, lastLayout ? lastLayout.x + lastLayout.width + 10 : 0);
  const totalHeight = rows * SCORE_ROW_HEIGHT + SCORE_TOP + BOTTOM_PADDING;

  useEffect(() => {
    drawScoreStaves(containerRef.current, staveLayouts, totalWidth, totalHeight, song.key, song.timeSignature);
  }, [staveLayouts, totalWidth, totalHeight, song.key, song.timeSignature, theme]);

  if (!keySignatureFor(song.key)) {
    return (
      <div ref={wrapperRef} role="alert" className="p-3 text-sm">
        Unsupported key signature: {song.key}. Open the project and choose a standard key to display the score.
      </div>
    );
  }

  return (
    <div ref={wrapperRef} className="relative select-none w-full pointer-events-none">
      <svg
        width={totalWidth}
        height={totalHeight}
        className="absolute top-0 left-0 pointer-events-none"
        style={{ zIndex: 1 }}
      >
        {visibleMeasures.map((measure, mi) => {
          const layout = staveLayouts[mi];
          if (!layout) return null;

          const contentWidth = layout.x + layout.width - layout.contentX;
          const slotWidth = contentWidth / 2;

          return ([0, 1] as const).map((si) => {
            const chord = measure.chords[si];
            const slotX = layout.contentX + si * slotWidth;

            if (!chord) return null;

            const cx = slotX + slotWidth / 2;
            const chordY = layout.midLineY;
            const parts = getChordDisplayParts(chord);
            const rootSize = 20;
            const annoSize = 11;
            const rootW = parts.root.length * rootSize * 0.6;

            return (
              <g key={`${mi}-${si}`}>
                <rect
                  x={cx - slotWidth * 0.45}
                  y={chordY - 16}
                  width={slotWidth * 0.9}
                  height={32}
                  rx={4}
                  fill="var(--bg-primary)"
                  opacity={0.85}
                />
                <text
                  x={cx - (parts.sup || parts.sub ? rootW * 0.15 : 0)}
                  y={chordY}
                  textAnchor="middle"
                  dominantBaseline="central"
                  fill="var(--chord-color)"
                  fontSize={rootSize}
                  fontWeight="bold"
                  fontFamily="system-ui, sans-serif"
                >
                  {parts.root}
                </text>
                {parts.sup && (
                  <text
                    x={cx + rootW * 0.35}
                    y={chordY - 8}
                    textAnchor="start"
                    dominantBaseline="central"
                    fill="var(--chord-color)"
                    fontSize={annoSize}
                    fontFamily="system-ui, sans-serif"
                  >
                    {parts.sup}
                  </text>
                )}
                {parts.sub && (
                  <text
                    x={cx + rootW * 0.35}
                    y={chordY + 8}
                    textAnchor="start"
                    dominantBaseline="central"
                    fill="var(--chord-color)"
                    fontSize={annoSize}
                    fontFamily="system-ui, sans-serif"
                  >
                    {parts.sub}
                  </text>
                )}
                {parts.bass && (
                  <text
                    x={cx + rootW * 0.35 + (parts.sup.length || parts.sub.length) * annoSize * 0.55 + 2}
                    y={chordY}
                    textAnchor="start"
                    dominantBaseline="central"
                    fill="var(--chord-color)"
                    fontSize={rootSize * 0.8}
                    fontWeight="bold"
                    fontFamily="system-ui, sans-serif"
                  >
                    /{parts.bass}
                  </text>
                )}
                {chord.romanNumeral && (
                  <text
                    x={cx}
                    y={layout.y + 100}
                    textAnchor="middle"
                    dominantBaseline="central"
                    fill="var(--text-soft)"
                    fontSize={18}
                    fontWeight="bold"
                    fontFamily="system-ui, sans-serif"
                  >
                    {chord.romanNumeral}
                  </text>
                )}
              </g>
            );
          });
        })}
      </svg>
      <div ref={containerRef} className="w-full vf-staff" />
    </div>
  );
}
