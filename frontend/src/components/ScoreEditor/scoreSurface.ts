import { useEffect, useState } from 'react';
import type { RefObject } from 'react';
import { Renderer, Stave } from 'vexflow/bravura';
import { keySignatureFor } from '../../engine/keySignatures';
import type { ScoreStaveLayout } from '../../engine/scoreStaveLayout';

/** Editor and project previews respond to the same container and theme changes. */
export function useStaffViewport(wrapperRef: RefObject<HTMLDivElement | null>) {
  const [containerWidth, setContainerWidth] = useState(960);
  const [theme, setTheme] = useState(document.documentElement.className);

  useEffect(() => {
    const observer = new MutationObserver(() => {
      setTheme(document.documentElement.className);
    });
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
    return () => observer.disconnect();
  }, []);

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
  }, [wrapperRef]);

  return { containerWidth, theme };
}

/** Draw only the stave layer; each view retains its own overlays and interactions. */
export function drawScoreStaves(
  container: HTMLDivElement | null,
  staveLayouts: readonly ScoreStaveLayout[],
  totalWidth: number,
  totalHeight: number,
  key: string,
  timeSignature: [number, number],
): void {
  if (!container) return;
  container.innerHTML = '';

  const renderer = new Renderer(container, Renderer.Backends.SVG);
  renderer.resize(totalWidth, totalHeight);
  const context = renderer.getContext();
  const keySig = keySignatureFor(key);
  if (!keySig) return;

  // Keep SVG paint bound to the theme even before the redraw effect runs.
  const staveColor = 'var(--stave-color)';
  context.setStrokeStyle(staveColor);
  context.setFillStyle(staveColor);

  for (let i = 0; i < staveLayouts.length; i++) {
    const bl = staveLayouts[i];
    const stave = new Stave(bl.x, bl.y, bl.width);
    if (bl.isFirstInRow) {
      stave.addClef('treble');
      stave.addKeySignature(keySig);
      if (i === 0) stave.addTimeSignature(`${timeSignature[0]}/${timeSignature[1]}`);
    }
    stave.setContext(context).draw();
  }
}
