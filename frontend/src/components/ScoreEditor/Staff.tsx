import { useEffect, useRef, useMemo, useCallback, useState } from 'react';
import type { KeyboardEvent } from 'react';
import { useSongStore, isSlotInList } from '../../store/songStore';
import { getChordDisplayParts } from '../../engine/chordDisplay';
import { sectionColor, sectionDisplayLabel } from '../../engine/songForm';
import type { Measure, SlotAddress, SongSection } from '../../models/types';
import { scoreStaveLayout } from '../../engine/scoreStaveLayout';
import { SCORE_MEASURES_PER_ROW, SCORE_ROW_HEIGHT } from '../../engine/scoreLayout';
import { selectAllScore } from '../../engine/scoreShortcuts';
import { selectionHasChords } from '../../engine/scoreClipboard';
import { keySignatureFor } from '../../engine/keySignatures';
import { useStaffViewport, drawScoreStaves } from './scoreSurface';
import ImportedMeasureReview from './ImportedMeasureReview';
import ReharmBand from './ReharmBand.tsx';
import ScoreContextMenu from './ScoreContextMenu.tsx';
import type { ContextMenuItem } from './ScoreContextMenu.tsx';
import { editorKeysBlocked, scoreEdits, useClipboardMemory, useScoreClipboard } from './useScoreClipboard';
import { useReharmonizeStore } from '../../store/reharmonizeStore';
import { hasEmptySlot, isExplicitNoChord, needsSourceReview } from './importedBars';

const CHORD_AREA_HEIGHT = 8;
const PADDING_Y = 10;
const PADDING_X = 10;
const ADD_BTN_HEIGHT = 50;
// Song-form section label baseline, below the stave's y. VexFlow puts the
// top stave line 40px below y and chord symbols sit on the middle line, so
// this band holds only the label (and the clef, which the label clears by
// starting at contentX).
const SECTION_LABEL_DY = 16;
/** Longest section name drawn on the score; its tooltip has the full name. */
const SECTION_LABEL_MAX_CHARS = 22;
/** Baseline of an imported bar's Review/Unresolved marker, below the Roman numerals. */
const IMPORT_MARKER_DY = 127;
const APPLE_KEYBOARD = typeof navigator !== 'undefined' && /Mac|iPhone|iPad|iPod/.test(navigator.platform);

/** The shortcut shown in the context menu for a Cmd/Ctrl key. */
function keyHint(key: string): string {
  return APPLE_KEYBOARD ? `⌘${key}` : `Ctrl+${key}`;
}

/** Keyboard activation (Enter or Space) for SVG text that acts as a button. */
function onEnterOrSpace(action: () => void) {
  return (event: KeyboardEvent<SVGTextElement>) => {
    if (event.key !== 'Enter' && event.key !== ' ') return;
    event.preventDefault();
    action();
  };
}

/** The bar's first empty chord slot; slot 0 when both hold chords. */
function firstEmptySlot(measure: Measure): 0 | 1 {
  return measure.chords[0] === null ? 0 : measure.chords[1] === null ? 1 : 0;
}

function sectionOriginHint(origin: SongSection['origin']): string {
  if (origin === 'inferred') return 'Suggested from the score';
  if (origin === 'explicit') return 'Label read from the score';
  return 'Click to select this region';
}

interface ImportPlaceholderProps {
  /** Left edge and width of the box the label is centred in. */
  x: number;
  width: number;
  midLineY: number;
  noChord: boolean;
}

/** Stands in for a missing imported chord: "Unresolved" in one slot, or "N.C."
 *  across a bar the source marked N.C. */
function ImportPlaceholder({ x, width, midLineY, noChord }: ImportPlaceholderProps) {
  return (
    <>
      <rect
        x={x + width * .05}
        y={midLineY - 12}
        width={width * .9}
        height={24}
        rx={3}
        fill="var(--bg-primary)"
        opacity={.85}
        className="pointer-events-none"
      />
      <text
        x={x + width / 2}
        y={midLineY}
        textAnchor="middle"
        dominantBaseline="central"
        fill="var(--text-soft)"
        fontSize={noChord ? 14 : 10}
        fontFamily="system-ui, sans-serif"
        className="pointer-events-none"
      >
        {noChord ? 'N.C.' : 'Unresolved'}
      </text>
    </>
  );
}

interface StaffProps {
  onOpenControls?: (tool: 'generate' | 'reharmonize') => void;
}

export default function Staff({ onOpenControls }: StaffProps = {}) {
  const wrapperRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const reharmRef = useRef<HTMLDivElement>(null);
  const comparison = useReharmonizeStore(s => s.comparison);
  const [bandHeight, setBandHeight] = useState(210);
  const bandAfter = comparison ? Math.max(...comparison.original.map(c => c.address.measureIndex)) : -1;
  const bandRow = bandAfter >= 0 ? Math.floor(bandAfter / SCORE_MEASURES_PER_ROW) : -1;
  const bandSpace = bandRow < 0 ? 0 : bandHeight + 16;
  const song = useSongStore((s) => s.song);
  const songEpoch = useSongStore((s) => s.songEpoch);
  const selectedSlots = useSongStore((s) => s.selectedSlots);
  const playbackPosition = useSongStore((s) => s.playbackPosition);
  const selectSlot = useSongStore((s) => s.selectSlot);
  const selectSlotRange = useSongStore((s) => s.selectSlotRange);
  const addMeasures = useSongStore((s) => s.addMeasures);
  const canUndo = useSongStore((s) => s.editHistory.length > 0);
  const clipboardMemory = useClipboardMemory();

  const [dragStart, setDragStart] = useState<SlotAddress | null>(null);
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number; epoch: number; selection: SlotAddress[] } | null>(null);
  const closeContextMenu = useCallback(() => setContextMenu(null), []);
  const visibleMenu = contextMenu && contextMenu.epoch === songEpoch && contextMenu.selection === selectedSlots ? contextMenu : null;
  const { containerWidth, theme } = useStaffViewport(wrapperRef);

  useEffect(() => {
    if (bandRow < 0 || !reharmRef.current) return;
    const observer = new ResizeObserver(([entry]) => {
      setBandHeight(Math.ceil(entry.target.getBoundingClientRect().height));
    });
    observer.observe(reharmRef.current);
    return () => observer.disconnect();
  }, [bandRow]);

  useEffect(() => {
    const handleKeyDown = (event: globalThis.KeyboardEvent) => {
      selectAllScore(event, useSongStore.getState().song.measures.length, editorKeysBlocked(event.target), selectSlotRange);
    };
    // Staff mounts only in the editor, so Projects keeps its browser shortcuts.
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [selectSlotRange]);
  // Cut / copy / paste / Delete / undo over the selection, likewise editor-only.
  useScoreClipboard();

  const selectionChords = selectionHasChords(song.measures, selectedSlots);
  const contextItems: ContextMenuItem[] = [
    { id: 'cut', label: 'Cut', hint: keyHint('X'), disabled: !selectionChords, title: 'The selected slots are empty.', run: () => scoreEdits.cut() },
    { id: 'copy', label: 'Copy', hint: keyHint('C'), run: () => scoreEdits.copy() },
    { id: 'paste', label: 'Paste', hint: keyHint('V'), disabled: !clipboardMemory, title: 'Copy or cut chords on the score first.', run: () => scoreEdits.paste() },
    { id: 'clear', label: 'Clear', hint: '⌫', disabled: !selectionChords, title: 'The selected slots are empty.', run: () => scoreEdits.clear() },
    { id: 'undo', label: 'Undo edit', hint: keyHint('Z'), disabled: !canUndo, title: 'No cut, paste or clear to undo.', run: () => scoreEdits.undo() },
    { id: 'generate', label: 'Generate', divider: true, run: () => onOpenControls?.('generate') },
    { id: 'reharmonize', label: 'Reharmonize', run: () => onOpenControls?.('reharmonize') },
  ];

  const staveLayouts = useMemo(() => {
    return Array.from({ length: song.measures.length }, (_, i) => {
      // Break the staff after the selected passage so the choices sit between
      // that bar and the next, even when the passage ends mid-row.
      const afterBand = bandAfter >= 0 && i > bandAfter;
      const displayIndex = afterBand ? (bandRow + 1) * SCORE_MEASURES_PER_ROW + i - bandAfter - 1 : i;
      const layout = scoreStaveLayout(displayIndex, containerWidth);
      const offset = afterBand ? bandSpace : 0;
      return { ...layout, y: layout.y + offset, midLineY: layout.midLineY + offset };
    });
  }, [song.measures.length, containerWidth, bandAfter, bandRow, bandSpace]);

  const rows = bandAfter < 0 ? Math.ceil(song.measures.length / SCORE_MEASURES_PER_ROW)
    : bandRow + 1 + Math.ceil(Math.max(0, song.measures.length - bandAfter - 1) / SCORE_MEASURES_PER_ROW);
  const totalWidth = containerWidth;
  const totalHeight = rows * SCORE_ROW_HEIGHT + PADDING_Y * 2 + CHORD_AREA_HEIGHT + ADD_BTN_HEIGHT + bandSpace;

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

  useEffect(() => {
    drawScoreStaves(containerRef.current, staveLayouts, totalWidth, totalHeight, song.key, song.timeSignature);
  }, [staveLayouts, totalWidth, totalHeight, song.key, song.timeSignature, theme]);

  // Position for add button
  const lastRow = rows - 1;
  const addBtnY = PADDING_Y + CHORD_AREA_HEIGHT + (lastRow + 1) * SCORE_ROW_HEIGHT + 8 + bandSpace;

  if (!keySignatureFor(song.key)) {
    return (
      <div ref={wrapperRef} role="alert" className="p-3 text-sm">
        Unsupported key signature: {song.key}. Choose a standard key in the Key control to display the score.
      </div>
    );
  }

  return (
    <>
    <ImportedMeasureReview song={song} selectedSlots={selectedSlots} />
    <div ref={wrapperRef} className="relative select-none w-full">
      {/* SVG overlay for chord labels + click/drag zones + add button */}
      <svg
        width={totalWidth}
        height={totalHeight}
        className="absolute top-0 left-0 pointer-events-none"
        style={{ zIndex: 1 }}
      >
        {/* Low-contrast form bands remain separate from selection and the playhead. */}
        {(song.sections ?? []).flatMap((section) =>
          Array.from({ length: section.length }, (_, offset) => {
            const layout = staveLayouts[section.startMeasure + offset];
            if (!layout) return null;
            return <rect key={`${section.id}-${offset}`} x={layout.contentX} y={layout.y - CHORD_AREA_HEIGHT}
              width={layout.x + layout.width - layout.contentX} height={SCORE_ROW_HEIGHT - 10}
              fill={sectionColor(section.type)} fillOpacity={0.09} className="pointer-events-none" />;
          }),
        )}
        {song.measures.map((measure, mi) => {
          const layout = staveLayouts[mi];
          if (!layout) return null;

          const contentWidth = layout.x + layout.width - layout.contentX;
          const slotWidth = contentWidth / 2;
          const importSource = song.importSource?.measures?.[mi];
          const explicitNoChord = !!importSource && isExplicitNoChord(measure, importSource);

          return ([0, 1] as const).map((si) => {
            const chord = measure.chords[si];
            const slotX = layout.contentX + si * slotWidth;
            const slot: SlotAddress = { measureIndex: mi, slotIndex: si };
            const isSelected = isSlotInList(slot, selectedSlots);
            const isPlaying = playbackPosition?.measureIndex === mi && playbackPosition?.slotIndex === si;

            // Full stave height for click/highlight zone
            const zoneY = layout.y - CHORD_AREA_HEIGHT;
            const zoneHeight = SCORE_ROW_HEIGHT;

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
                  onMouseDown={event => {
                    // A secondary click (including macOS Control-click) must
                    // not collapse the range the user just dragged.
                    if (event.button !== 0 || event.ctrlKey) return;
                    closeContextMenu();
                    handleSlotMouseDown(slot);
                  }}
                  onMouseEnter={() => handleSlotMouseEnter(slot)}
                  onContextMenu={event => {
                    event.preventDefault();
                    event.stopPropagation();
                    setDragStart(null);
                    const selection = useSongStore.getState().selectedSlots;
                    if (!isSlotInList(slot, selection)) selectSlot(slot);
                    const bounds = event.currentTarget.getBoundingClientRect();
                    setContextMenu({
                      x: event.clientX || bounds.left + bounds.width / 2,
                      y: event.clientY || bounds.top + bounds.height / 2,
                      epoch: songEpoch, selection: useSongStore.getState().selectedSlots,
                    });
                  }}
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

                {/* An N.C. bar gets one label across the whole bar, drawn with slot 0. */}
                {!chord && importSource && (!explicitNoChord || si === 0) && (
                  <ImportPlaceholder
                    x={explicitNoChord ? layout.contentX : slotX}
                    width={explicitNoChord ? contentWidth : slotWidth}
                    midLineY={layout.midLineY}
                    noChord={explicitNoChord}
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

        {/* Imported uncertainty is visible on the score, without blocking edits. */}
        {song.importSource?.measures?.map((source, index) => {
          const measure = song.measures[index];
          const layout = staveLayouts[index];
          if (!measure || !layout || isExplicitNoChord(measure, source)) return null;
          const pending = hasEmptySlot(measure);
          if (!pending && !needsSourceReview(source)) return null;
          const select = () => selectSlot({ measureIndex: index, slotIndex: firstEmptySlot(measure) });
          return (
            <text key={`import-review-${index}`} x={layout.x + layout.width - 8} y={layout.y + IMPORT_MARKER_DY}
              textAnchor="end" fill="var(--text-soft)" fontSize={10} fontWeight={600}
              fontFamily="system-ui, sans-serif" className="pointer-events-auto cursor-pointer"
              role="button" tabIndex={0} aria-label={`Review imported bar ${index + 1}${pending ? ', unresolved chord slots' : ''}`}
              onClick={select} onKeyDown={onEnterOrSpace(select)}>
              <title>{source.editorReview || (pending ? 'Choose the unresolved chord slots in the editor.' : 'Compare the imported harmony with its source.')}</title>
              {pending ? 'Unresolved' : 'Review'}
            </text>
          );
        })}

        {/* Labels select the whole region; drag zones still select individual chords. */}
        {(song.sections ?? []).map((section) => {
          const layout = staveLayouts[section.startMeasure];
          if (!layout) return null;
          const label = sectionDisplayLabel(section);
          const selectSection = () => selectSlotRange(
            { measureIndex: section.startMeasure, slotIndex: 0 },
            { measureIndex: section.startMeasure + section.length - 1, slotIndex: 1 },
          );
          return (
            <text
              key={section.id}
              x={layout.contentX}
              y={layout.y + SECTION_LABEL_DY}
              fill="var(--text-soft)"
              fontSize={11}
              fontWeight={700}
              letterSpacing="0.08em"
              fontFamily="system-ui, sans-serif"
              className="pointer-events-auto cursor-pointer"
              role="button"
              tabIndex={0}
              aria-label={`Select ${label}, bars ${section.startMeasure + 1} to ${section.startMeasure + section.length}`}
              onClick={selectSection}
              onKeyDown={onEnterOrSpace(selectSection)}
            >
              <title>{`${label} · ${sectionOriginHint(section.origin)}`}</title>
              {label.slice(0, SECTION_LABEL_MAX_CHARS).toUpperCase()}{section.origin === 'inferred' ? ' ?' : ''}
            </text>
          );
        })}

        {/* Add measures button inside the score area */}
        <g className="pointer-events-auto cursor-pointer" onClick={() => addMeasures()}>
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
      <div ref={containerRef} className="w-full vf-staff" />
      {bandRow >= 0 && <div ref={reharmRef} className="absolute left-2 right-2"
        style={{ top: PADDING_Y + CHORD_AREA_HEIGHT + (bandRow + 1) * SCORE_ROW_HEIGHT, zIndex: 2 }}>
        <ReharmBand />
      </div>}
    </div>
    {visibleMenu && <ScoreContextMenu x={visibleMenu.x} y={visibleMenu.y} items={contextItems} onClose={closeContextMenu}
      onChoose={item => { closeContextMenu(); item.run(); }} />}
    </>
  );
}
