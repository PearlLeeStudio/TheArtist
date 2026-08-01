/**
 * PlanRunBand — the "Complete the form" details inside the song-form dialog:
 * row of chips in run order (plan, validate, each section), each with its
 * source badge, then the outcome line and Apply / Undo / Export trace /
 * Dismiss.
 *
 * Mounted in the dialog's collapsed details while the plan store is not idle.
 * A failed run ends in a red chip. Successful dialog runs apply
 * automatically; Undo, reapply and trace export remain here.
 */
import { usePlanStore, type PlanChip } from '../../store/planStore';
import { downloadExportBlob } from '../../engine/exportSpec';

/** The error text colour ExportMenu uses; the band adds no colour of its own. */
const ERROR_TEXT_COLOR = '#d97070';

const secondaryStyle: React.CSSProperties = {
  background: 'var(--btn-bg)',
  color: 'var(--text-primary)',
  border: '1px solid var(--border)',
};

function seconds(ms: number): string {
  return `${(ms / 1000).toFixed(1)} s`;
}

function traceFilename(): string {
  return 'song-form-trace.json';
}

/** A chip breaks only between its parts, so a narrow score pane wraps a
 *  long section chip instead of overflowing it. */
function Chip({ chip }: { chip: PlanChip }) {
  const accent = chip.ok ? 'var(--text-heading)' : ERROR_TEXT_COLOR;
  return (
    <span
      className="inline-flex flex-wrap items-center gap-x-1.5 gap-y-0.5 px-1.5 py-0.5 font-mono text-[10px]"
      title={chip.message ?? undefined}
      style={{
        background: 'var(--bg-surface)',
        border: `1px solid ${chip.ok ? 'var(--border)' : ERROR_TEXT_COLOR}`,
        color: 'var(--text-soft)',
      }}
    >
      <span className="whitespace-nowrap" style={{ color: accent, fontWeight: 700 }}>{chip.label}</span>
      {chip.badge && (
        <span className="px-1 uppercase tracking-wide text-[9px] whitespace-nowrap" style={{ border: '1px solid var(--border)', color: 'var(--text-muted)' }}>
          {chip.badge}
        </span>
      )}
      {chip.note && <span className="whitespace-nowrap" style={{ color: 'var(--text-muted)' }}>{chip.note}</span>}
    </span>
  );
}

export default function PlanRunBand() {
  const status = usePlanStore((s) => s.status);
  const chips = usePlanStore((s) => s.chips);
  const sections = usePlanStore((s) => s.sections);
  const error = usePlanStore((s) => s.error);
  const trace = usePlanStore((s) => s.trace);
  const applied = usePlanStore((s) => s.applied);
  const applyPlan = usePlanStore((s) => s.applyPlan);
  const undoPlan = usePlanStore((s) => s.undoPlan);
  const exportTrace = usePlanStore((s) => s.exportTrace);
  const dismiss = usePlanStore((s) => s.dismiss);

  const bars = sections.reduce((sum, section) => sum + section.bars, 0);
  const outcome =
    status === 'running' ? 'Generating the genre form with the selected model…' :
    status === 'failed' ? `${error ?? 'The run failed'} · the staff is unchanged` :
    `Done: ${sections.length} ${sections.length === 1 ? 'section' : 'sections'}, ${bars} bars` +
      `${trace ? ` · ${seconds(trace.elapsedMs)}` : ''} · ${applied ? 'applied to the staff' : 'nothing applied yet'}`;

  const handleExport = () => {
    const json = exportTrace();
    if (json) downloadExportBlob(new Blob([json], { type: 'application/json' }), traceFilename());
  };

  const button = (label: string, onClick: () => void, disabled: boolean, title: string, primary = false) => (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      title={title}
      className={`px-2 py-1 text-xs btn-hover disabled:opacity-40 disabled:cursor-not-allowed${primary ? ' font-bold' : ''}`}
      style={primary ? { background: 'var(--text-heading)', color: 'var(--bg-primary)' } : secondaryStyle}
    >
      {label}
    </button>
  );

  return (
    <section
      aria-label="Complete the form"
      className="mb-3 p-2 space-y-2 text-xs"
      style={{ background: 'var(--btn-bg)', border: '1px solid var(--border)' }}
    >
      <div className="flex flex-wrap items-center gap-1.5">
        <span className="text-[10px] uppercase tracking-wide font-semibold" style={{ color: 'var(--text-muted)' }}>
          Complete the form
        </span>
        {chips.map((chip, index) => <Chip key={`${index}-${chip.step}`} chip={chip} />)}
        {status === 'running' && (
          <span
            className="inline-block shrink-0"
            style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--brand-yellow)', animation: 'pearl-pulse 1.2s ease-in-out infinite' }}
          />
        )}
      </div>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p role="status" className="break-words min-w-0" style={{ color: status === 'failed' ? ERROR_TEXT_COLOR : 'var(--text-soft)' }}>
          {outcome}
        </p>
        <div className="flex flex-wrap gap-1.5">
          {button('Apply', applyPlan, status !== 'done' || applied, 'Append the planned sections to the staff', true)}
          {button('Undo', undoPlan, !applied, 'Restore the score from before Apply')}
          {button('Export trace', handleExport, trace === null, 'Download the form, generated sections and model details as JSON')}
          {button(status === 'running' ? 'Cancel' : 'Dismiss', dismiss, false,
            status === 'running' ? 'Stop the run; the score is unchanged' : 'Hide this run')}
        </div>
      </div>
    </section>
  );
}
