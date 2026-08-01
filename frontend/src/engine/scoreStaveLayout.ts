import { Stave } from 'vexflow/bravura';
import { scoreMeasureLayout } from './scoreLayout.ts';
import type { ScoreMeasureLayout } from './scoreLayout.ts';

export interface ScoreStaveLayout extends ScoreMeasureLayout {
  midLineY: number;
}

/** VexFlow's line geometry is available before drawing, without an animation frame. */
export function scoreStaveLayout(index: number, width: number): ScoreStaveLayout {
  const layout = scoreMeasureLayout(index, width);
  const stave = new Stave(layout.x, layout.y, layout.width);
  return { ...layout, midLineY: stave.getYForLine(2) };
}
