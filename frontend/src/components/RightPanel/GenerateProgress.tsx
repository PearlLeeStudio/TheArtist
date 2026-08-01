import type { GeneratePhase } from '../../engine/chordGenerator';

interface GenerateProgressProps {
  phase: GeneratePhase | null;
}

/**
 * Stage-by-stage progress indicator driven by SSE phase events from the
 * streaming /api/generate endpoint. Each pipeline stage is a chip in a
 * horizontal row; the active one is highlighted with the brand yellow and
 * earlier stages get a faint check mark.
 *
 * The pipeline shape (and chip labels) is fixed up-front so the UI does
 * not flicker as events arrive — only the active index changes.
 */
const STAGES: Array<{ id: GeneratePhase['name'] | 'starting'; label: string }> = [
  { id: 'starting', label: 'Read' },
  { id: 'model_load', label: 'Load' },
  { id: 'composing', label: 'Compose' },
  { id: 'ranking', label: 'Rank' },
  { id: 'voicing', label: 'Voice' },
  { id: 'explaining', label: 'Theory' },
  { id: 'complete', label: 'Done' },
];

function activeIndex(phase: GeneratePhase | null): number {
  if (phase === null) return 0;  // before first event
  const idx = STAGES.findIndex((s) => s.id === phase.name);
  return idx >= 0 ? idx : 0;
}

export default function GenerateProgress({ phase }: GenerateProgressProps) {
  const idx = activeIndex(phase);
  const detailLabel = phase?.label ?? 'Reading prompt…';

  // For "composing", surface the per-step counter (e.g. 2/3) on the chip
  // itself so users see the inner-loop progress without reading prose.
  const composingStep =
    phase && phase.name === 'composing'
      ? ` ${phase.step}/${phase.total}`
      : '';

  return (
    <div className="space-y-1.5" style={{ background: 'var(--btn-bg)', border: '1px solid var(--border)', padding: '8px' }}>
      {/* Grid (not flex) so each chip occupies a fixed slot — its width does NOT
          depend on its label content. Fixes layout-shift when "✓ " prefix or
          counter " 1/3" is added/removed mid-flight. Border kept at 1px solid
          on every state (active uses transparent so the cell footprint is
          identical to done/pending). */}
      <div className="grid grid-cols-7 gap-1.5 items-center">
        {STAGES.map((stage, i) => {
          const state =
            i < idx ? 'done' :
            i === idx ? 'active' :
            'pending';
          const showCounter = stage.id === 'composing' && state !== 'pending';
          return (
            <div
              key={stage.id}
              className="px-1.5 py-1 text-[10px] font-mono text-center overflow-hidden whitespace-nowrap"
              style={{
                minWidth: 0,
                background:
                  state === 'active' ? 'var(--brand-yellow)' :
                  state === 'done'   ? 'var(--bg-surface)' :
                                       'transparent',
                color:
                  state === 'active' ? '#1a1a1a' :
                  state === 'done'   ? 'var(--text-soft)' :
                                       'var(--text-muted)',
                fontWeight: state === 'active' ? 700 : 400,
                opacity: state === 'pending' ? 0.55 : 1,
                // Always 1px border (transparent on active) so cell footprint
                // is identical across states — no layout shift on transitions.
                border: state === 'active' ? '1px solid transparent' : '1px solid var(--border)',
                transition: 'background 120ms, color 120ms, opacity 120ms',
              }}
              title={stage.label}
            >
              {state === 'done' ? '✓ ' : ''}
              {stage.label}{showCounter ? composingStep : ''}
            </div>
          );
        })}
      </div>

      <div className="text-[11px] flex items-center gap-1.5" style={{ color: 'var(--text-soft)' }}>
        <PulsingDot />
        <span>{detailLabel}</span>
      </div>
    </div>
  );
}

/** Small pulsing dot to make the active state feel alive. */
function PulsingDot() {
  return (
    <span
      className="inline-block"
      style={{
        width: 6,
        height: 6,
        borderRadius: '50%',
        background: 'var(--brand-yellow)',
        animation: 'pearl-pulse 1.2s ease-in-out infinite',
      }}
    />
  );
}
