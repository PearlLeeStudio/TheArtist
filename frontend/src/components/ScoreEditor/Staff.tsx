import { useEffect, useRef, useMemo, useCallback, useState } from 'react';
import { Renderer, Stave } from 'vexflow';
import { useSongStore, isSlotInList } from '../../store/songStore';
import { getChordDisplayParts } from '../../engine/chordDisplay';
import type { SlotAddress } from '../../models/types';

const KEY_SIG_MAP: Record<string, string> = {
  'C major': 'C', 'G major': 'G', 'D major': 'D', 'A major': 'A',
  'E major': 'E', 'B major': 'B', 'F# major': 'F#', 'Gb major': 'Gb',
  'Db major': 'Db', 'Ab major': 'Ab', 'Eb major': 'Eb', 'Bb major': 'Bb',
  'F major': 'F',
  'A minor': 'Am', 'E minor': 'Em', 'B minor': 'Bm', 'F# minor': 'F#m',
  'C# minor': 'C#m', 'G# minor': 'G#m', 'Eb minor': 'Ebm', 'Bb minor': 'Bbm',
  'F minor': 'Fm', 'C minor': 'Cm', 'G minor': 'Gm', 'D minor': 'Dm',
};

const MEASURES_PER_ROW = 4;
const CHORD_AREA_HEIGHT = 8;
const STAVE_HEIGHT = 170;
const PADDING_Y = 10;
const PADDING_X = 10;
const ADD_BTN_HEIGHT = 50;

interface StaveLayout {
  x: number;
  y: number;
  width: number;
  isFirstInRow: boolean;
  contentX: number;
  midLineY: number; // actual Y of the 3rd (middle) stave line — set after VexFlow render
}

export default function Staff() {
  const wrapperRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const song = useSongStore((s) => s.song);
  const selectedSlots = useSongStore((s) => s.selectedSlots);
  const playbackPosition = useSongStore((s) => s.playbackPosition);
  const selectSlot = useSongStore((s) => s.selectSlot);
  const selectSlotRange = useSongStore((s) => s.selectSlotRange);
  const addMeasures = useSongStore((s) => s.addMeasures);

  const [dragStart, setDragStart] = useState<SlotAddress | null>(null);
  const [containerWidth, setContainerWidth] = useState(960);
  const [theme, setTheme] = useState(document.documentElement.className);

  // Watch for theme class changes
  useEffect(() => {
    const observer = new MutationObserver(() => {
      setTheme(document.documentElement.className);
    });
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
    return () => observer.disconnect();
  }, []);

  // Track container width
  useEffect(() => {
    const el = wrapperRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setContainerWidth(entry.contentRect.width);
      }
    });
    ro.observe(el);
    setContainerWidth(el.clientWidth);
    return () => ro.disconnect();
  }, []);

  // Pearl 2026-05-09: equal chord-area width across all measures in a row.
  // Previous layout gave every measure the same stave width, but the first
  // measure burns ~60px on clef + key sig + time sig — squeezing its chord
  // slots. Now we add a CLEF_OFFSET (60px) to the first measure's width
  // and shift subsequent measures right by the same amount, so the chord
  // content area (after the leading margin) is identical across m0-m3.
  const CLEF_OFFSET = 60; // = first-in-row content offset (70) - regular margin (10)
  const baseStaveWidth = Math.max(
    120,
    (containerWidth - PADDING_X * 2 - CLEF_OFFSET) / MEASURES_PER_ROW,
  );

  const [staveLayouts, setStaveLayouts] = useState<StaveLayout[]>([]);

  const baseLayouts = useMemo((): Omit<StaveLayout, 'midLineY'>[] => {
    return song.measures.map((_, i) => {
      const row = Math.floor(i / MEASURES_PER_ROW);
      const col = i % MEASURES_PER_ROW;
      const isFirstInRow = col === 0;
      // First-in-row gets baseStaveWidth + CLEF_OFFSET. Subsequent measures
      // start `CLEF_OFFSET` further right (because the first one ate that
      // extra width). Net: all measures share the same effective chord area.
      const width = baseStaveWidth + (isFirstInRow ? CLEF_OFFSET : 0);
      const x = PADDING_X + col * baseStaveWidth + (col > 0 ? CLEF_OFFSET : 0);
      const y = PADDING_Y + CHORD_AREA_HEIGHT + row * STAVE_HEIGHT;
      const contentX = x + (isFirstInRow ? 70 : 10);
      return { x, y, width, isFirstInRow, contentX };
    });
  }, [song.measures.length, baseStaveWidth]);

  const rows = Math.ceil(song.measures.length / MEASURES_PER_ROW);
  const totalWidth = containerWidth;
  const totalHeight = rows * STAVE_HEIGHT + PADDING_Y * 2 + CHORD_AREA_HEIGHT + ADD_BTN_HEIGHT;

  // Drag handlers
  const handleSlotMouseDown = useCallback((slot: SlotAddress) => {
    setDragStart(slot);
    selectSlot(slot);
  }, [selectSlot]);

  const handleSlotMouseEnter = useCallback((slot: SlotAddress) => {
    if (dragStart) {
      selectSlotRange(dragStart, slot);
    }
  }, [dragStart, selectSlotRange]);

  const handleMouseUp = useCallback(() => {
    setDragStart(null);
  }, []);

  useEffect(() => {
    window.addEventListener('mouseup', handleMouseUp);
    return () => window.removeEventListener('mouseup', handleMouseUp);
  }, [handleMouseUp]);

  // Render VexFlow
  useEffect(() => {
    if (!containerRef.current) return;
    const container = containerRef.current;
    container.innerHTML = '';

    const renderer = new Renderer(container, Renderer.Backends.SVG);
    renderer.resize(totalWidth, totalHeight);
    const context = renderer.getContext();
    const keySig = KEY_SIG_MAP[song.key] || 'C';

    // Get stave color from CSS variable
    const staveColor = getComputedStyle(document.documentElement).getPropertyValue('--stave-color').trim() || '#000';
    context.setStrokeStyle(staveColor);
    context.setFillStyle(staveColor);

    const measured: StaveLayout[] = [];
    for (let i = 0; i < song.measures.length; i++) {
      const bl = baseLayouts[i];
      const stave = new Stave(bl.x, bl.y, bl.width);

      if (bl.isFirstInRow) {
        stave.addClef('treble');
        stave.addKeySignature(keySig);
        if (i === 0) {
          stave.addTimeSignature(`${song.timeSignature[0]}/${song.timeSignature[1]}`);
        }
      }

      stave.setContext(context).draw();

      // Get the actual Y of the middle (3rd) stave line
      const midY = stave.getYForLine(2);
      measured.push({ ...bl, midLineY: midY });
    }
    setStaveLayouts(measured);
  }, [song.measures.length, song.key, song.timeSignature, baseLayouts, totalWidth, totalHeight, theme]);

  // Position for add button
  const lastRow = rows - 1;
  const addBtnY = PADDING_Y + CHORD_AREA_HEIGHT + (lastRow + 1) * STAVE_HEIGHT + 8;

  return (
    <div ref={wrapperRef} className="relative select-none w-full">
      {/* SVG overlay for chord labels + click/drag zones + add button */}
      <svg
        width={totalWidth}
        height={totalHeight}
        className="absolute top-0 left-0 pointer-events-none"
        style={{ zIndex: 1 }}
        onContextMenu={(e) => {
          e.preventDefault();
          selectSlot(null);
        }}
      >
        {song.measures.map((measure, mi) => {
          const layout = staveLayouts[mi];
          if (!layout) return null;

          const contentWidth = layout.x + layout.width - layout.contentX;
          const slotWidth = contentWidth / 2;

          return ([0, 1] as const).map((si) => {
            const chord = measure.chords[si];
            const slotX = layout.contentX + si * slotWidth;
            const slot: SlotAddress = { measureIndex: mi, slotIndex: si };
            const isSelected = isSlotInList(slot, selectedSlots);
            const isPlaying = playbackPosition?.measureIndex === mi && playbackPosition?.slotIndex === si;

            // Full stave height for click/highlight zone
            const zoneY = layout.y - CHORD_AREA_HEIGHT;
            const zoneHeight = STAVE_HEIGHT;

            return (
              <g key={`${mi}-${si}`}>
                {/* Click/drag zone covering chord area + entire stave */}
                <rect
                  x={slotX}
                  y={zoneY}
                  width={slotWidth}
                  height={zoneHeight}
                  fill={
                    isPlaying ? 'var(--playback-bg)' :
                    isSelected ? 'var(--selection-bg)' :
                    'transparent'
                  }
                  className="pointer-events-auto cursor-pointer"
                  onMouseDown={() => handleSlotMouseDown(slot)}
                  onMouseEnter={() => handleSlotMouseEnter(slot)}
                />

                {/* Top indicator bar — playback wins over selection so the
                    cursor is always obvious mid-play. */}
                {(isSelected || isPlaying) && (
                  <rect
                    x={slotX}
                    y={zoneY}
                    width={slotWidth}
                    height={2}
                    fill={isPlaying ? 'var(--playback-bar)' : 'var(--selection-bar)'}
                    rx={1}
                  />
                )}

                {chord && (() => {
                  const cx = slotX + slotWidth / 2;
                  const chordY = layout.midLineY; // exact 3rd line from VexFlow
                  const parts = getChordDisplayParts(chord);
                  // Estimate root width for positioning sup/sub
                  const rootSize = 20;
                  const annoSize = 11;
                  const rootW = parts.root.length * rootSize * 0.6;
                  return (
                    <>
                      <rect
                        x={cx - slotWidth * 0.45}
                        y={chordY - 16}
                        width={slotWidth * 0.9}
                        height={32}
                        rx={4}
                        fill="var(--bg-primary)"
                        opacity={0.85}
                        className="pointer-events-none"
                      />
                      {/* Root — large */}
                      <text
                        x={cx - (parts.sup || parts.sub ? rootW * 0.15 : 0)}
                        y={chordY}
                        textAnchor="middle"
                        dominantBaseline="central"
                        fill="var(--chord-color)"
                        fontSize={rootSize}
                        fontWeight="bold"
                        fontFamily="system-ui, sans-serif"
                        className="pointer-events-none"
                      >
                        {parts.root}
                      </text>
                      {/* Superscript — extension (top-right) */}
                      {parts.sup && (
                        <text
                          x={cx + rootW * 0.35}
                          y={chordY - 8}
                          textAnchor="start"
                          dominantBaseline="central"
                          fill="var(--chord-color)"
                          fontSize={annoSize}
                          fontFamily="system-ui, sans-serif"
                          className="pointer-events-none"
                        >
                          {parts.sup}
                        </text>
                      )}
                      {/* Subscript — base modifier (bottom-right) */}
                      {parts.sub && (
                        <text
                          x={cx + rootW * 0.35}
                          y={chordY + 8}
                          textAnchor="start"
                          dominantBaseline="central"
                          fill="var(--chord-color)"
                          fontSize={annoSize}
                          fontFamily="system-ui, sans-serif"
                          className="pointer-events-none"
                        >
                          {parts.sub}
                        </text>
                      )}
                      {/* Bass note — slash chord */}
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
                          className="pointer-events-none"
                        >
                          /{parts.bass}
                        </text>
                      )}
                      {/* Roman numeral — below stave. Theory analysis,
                          read as secondary information, so it sits in
                          the muted-grey tier rather than fighting the
                          chord symbol above for attention. */}
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
                          className="pointer-events-none"
                        >
                          {chord.romanNumeral}
                        </text>
                      )}
                    </>
                  );
                })()}
              </g>
            );
          });
        })}

        {/* Add measures button inside the score area */}
        <g className="pointer-events-auto cursor-pointer" onClick={addMeasures}>
          <rect
            x={PADDING_X}
            y={addBtnY}
            width={totalWidth - PADDING_X * 2}
            height={36}
            rx={6}
            fill="none"
            stroke="var(--border)"
            strokeWidth={1.5}
            strokeDasharray="6 4"
          />
          <text
            x={totalWidth / 2}
            y={addBtnY + 22}
            textAnchor="middle"
            fill="var(--text-muted)"
            fontSize={13}
            fontFamily="system-ui, sans-serif"
          >
            + Add 4 Measures
          </text>
        </g>
      </svg>

      {/* VexFlow renders here */}
      <div ref={containerRef} className="w-full" />
    </div>
  );
}
