import OperationProgress from '../OperationProgress';
/**
 * SongFormDialog — "New from Song Form". The song's key and genre and the
 * chord model sit at the top for both tabs: they are the same settings as
 * the key and genre above the staff and the model in Input → Generate, so
 * changing one here changes it there (a new key transposes the score; a
 * genre picks the model it is served by; a model sets its genre). Two tabs:
 *
 *   Section    pick a song-form section and how many bars it gets, then
 *              generate it after the last used measure.
 *   Full form  A genre template extends the sketch and the selected Music
 *              Transformer writes its chords. Successful runs apply on close.
 *
 * Native <dialog> opened with showModal(), so Esc, focus trapping and the
 * inert page behind it come from the browser. Mount it to open it; it
 * calls onClose once it has closed (Esc, Cancel, after a successful
 * generation). Closing during a run cancels it. Run details stay inside
 * this dialog, so the staff shows only the score and its section labels.
 */
import { useEffect, useId, useRef, useState } from 'react';
import { useSongStore } from '../../store/songStore';
import { useGenerationStore } from '../../store/generationStore';
import { usePlanStore } from '../../store/planStore';
import GenerateProgress from '../RightPanel/GenerateProgress';
import PlanRunBand from './PlanRunBand';
import { GENRE_OPTIONS, genreForModel, genreOptionValue } from '../../engine/genres';
import { STANDARD_KEYS } from '../../engine/keySignatures';
import {
  SECTION_GROUPS,
  barCountSummary,
  checkBarCount,
  defaultSectionBars,
  isSectionType,
  sectionLabel,
  sectionPlacement,
  sectionsInGroup,
} from '../../engine/songForm';
import {
  CONTRASTS,
  TARGET_BARS,
  availableTargetBars,
  planBlocker,
  resolveTargetBars,
  sketchBars,
  type PlanContrast,
  type ChorusMode,
  type TargetBars,
} from '../../engine/planClient';
import type { SectionType } from '../../models/types';

const DEFAULT_SECTION: SectionType = 'verse';
/** The length the Full form tab starts from when the sketch leaves room for it. */
const DEFAULT_TARGET_BARS: TargetBars = 32;

type FormTab = 'section' | 'full';
const FORM_TABS: { id: FormTab; label: string }[] = [
  { id: 'section', label: 'Section' },
  { id: 'full', label: 'Full form' },
];

const fieldStyle: React.CSSProperties = {
  background: 'var(--btn-bg)',
  border: '1px solid var(--border)',
  color: 'var(--text-primary)',
};

const labelStyle: React.CSSProperties = {
  color: 'var(--text-muted)',
};

const valueStyle: React.CSSProperties = {
  color: 'var(--text-heading)',
};

function barRange(startMeasure: number, bars: number): string {
  return bars === 1
    ? `bar ${startMeasure + 1}`
    : `bars ${startMeasure + 1}–${startMeasure + bars}`;
}

/** A checkpoint as GET /api/models/list reports it. */
interface ModelOption {
  key: string;
  label: string;
  available: boolean;
}

const fieldLabelClass = 'text-[10px] uppercase tracking-wide font-semibold';

interface ChoiceRowProps<T extends string | number | boolean> {
  label: string;
  options: ReadonlyArray<{ value: T; label: string; disabled?: boolean }>;
  value: T;
  onChange: (value: T) => void;
}

/** One labelled row of chips, exactly one of them pressed. */
function ChoiceRow<T extends string | number | boolean>({ label, options, value, onChange }: ChoiceRowProps<T>) {
  return (
    <>
      <span className="text-[10px] uppercase tracking-wide font-semibold" style={labelStyle}>{label}</span>
      <div className="flex flex-wrap gap-1.5">
        {options.map((option) => {
          const pressed = option.value === value;
          return (
            <button
              key={String(option.value)}
              type="button"
              aria-pressed={pressed}
              disabled={option.disabled}
              onClick={() => onChange(option.value)}
              className="px-2.5 py-1 text-xs btn-hover disabled:opacity-40 disabled:cursor-not-allowed"
              style={pressed
                ? { background: 'var(--text-heading)', color: 'var(--bg-primary)', border: '1px solid var(--text-heading)', fontWeight: 700 }
                : fieldStyle}
            >
              {option.label}
            </button>
          );
        })}
      </div>
    </>
  );
}

interface SongFormDialogProps {
  /** Called once the dialog has closed. */
  onClose: () => void;
  /** Called after the new section's chords landed on the staff. */
  onGenerated: () => void;
  onEditSections: () => void;
  /** Saves the current project before opening an empty score; throws if saving fails. */
  onNewSong: () => void;
}

export default function SongFormDialog({ onClose, onGenerated, onEditSections, onNewSong }: SongFormDialogProps) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const sectionRef = useRef<HTMLSelectElement>(null);
  const runningRef = useRef(false);
  const planRunningRef = useRef(false);
  const id = useId();
  const headingId = `${id}-heading`;
  const sectionId = `${id}-section`;
  const barsId = `${id}-bars`;
  const hintId = `${id}-hint`;
  const keyId = `${id}-key`;
  const genreId = `${id}-genre`;
  const modelId = `${id}-model`;

  const song = useSongStore((s) => s.song);
  const setKey = useSongStore((s) => s.setKey);
  const setGenre = useSongStore((s) => s.setGenre);
  const modelKey = useGenerationStore((s) => s.modelKey);
  const setModelKey = useGenerationStore((s) => s.setModelKey);
  const phase = useGenerationStore((s) => s.phase);
  const runSectionGenerate = useGenerationStore((s) => s.runSectionGenerate);
  const cancelSectionGenerate = useGenerationStore((s) => s.cancelSectionGenerate);
  const startPlan = usePlanStore((s) => s.startPlan);
  const planStatus = usePlanStore((s) => s.status);
  const planProgress = usePlanStore((s) => s.progress);
  const planError = usePlanStore((s) => s.error);
  const cancelPlan = usePlanStore((s) => s.cancelPlan);
  const planning = planStatus === 'running';

  const [tab, setTab] = useState<FormTab>(() => planStatus === 'idle' ? 'section' : 'full');
  const [sectionType, setSectionType] = useState<SectionType>(DEFAULT_SECTION);
  const [barsText, setBarsText] = useState(String(defaultSectionBars(DEFAULT_SECTION)));
  // Once the user types a length, picking another section keeps it.
  const [barsTyped, setBarsTyped] = useState(false);
  const [running, setRunning] = useState(false);
  const [runStart, setRunStart] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const busy = running || planning;
  // The models this install has, and the model each genre is served by (GET /api/genres/list)
  const [models, setModels] = useState<ModelOption[]>([]);
  const [genreModels, setGenreModels] = useState<Record<string, string>>({});
  const modelLabel = models.find((m) => m.key === modelKey)?.label ?? null;

  // Full form: the sketch is the song up to its last used measure.
  const sketch = sketchBars(song);
  const targets = availableTargetBars(sketch);
  const [preferredTargetBars, setTargetBars] = useState<TargetBars>(DEFAULT_TARGET_BARS);
  const targetBars = resolveTargetBars(sketch, preferredTargetBars);
  const [wantOutro, setWantOutro] = useState(true);
  const [contrast, setContrast] = useState<PlanContrast>('medium');
  const [chorusMode, setChorusMode] = useState<ChorusMode>('reuse');

  useEffect(() => {
    const dialog = dialogRef.current;
    if (dialog && !dialog.open) dialog.showModal();
    sectionRef.current?.focus();
  }, []);

  // The model picker lists the checkpoints this install has (as Input → Generate does);
  // until the lists answer, it shows the current model's key alone.
  useEffect(() => {
    let alive = true;
    fetch('/api/models/list')
      .then((r) => (r.ok ? r.json() : null))
      .then((data: { models?: ModelOption[] } | null) => {
        if (alive && data?.models) setModels(data.models.filter((m) => m.available));
      })
      .catch(() => { /* the current model stays the only choice */ });
    fetch('/api/genres/list')
      .then((r) => (r.ok ? r.json() : null))
      .then((data: { genres?: Array<{ key: string; served_by: string }> } | null) => {
        if (alive && data?.genres) setGenreModels(Object.fromEntries(data.genres.map((g) => [g.key, g.served_by])));
      })
      .catch(() => { /* a genre pick then keeps the current model */ });
    return () => { alive = false; };
  }, []);

  const check = checkBarCount(barsText);
  // While generating, the song already holds the new section, so a live
  // placement would point past it; show where this run put it instead.
  const startMeasure = running
    ? runStart
    : check.ok ? sectionPlacement(song, check.bars).startMeasure : null;
  const generateDisabled = !check.ok || busy;

  const blocker = planBlocker(song);
  const planDisabled = blocker !== null || targetBars === null || busy;
  const sectionCount = song.sections?.length ?? 0;

  const handleSectionChange = (value: string) => {
    if (!isSectionType(value)) return;
    setSectionType(value);
    if (!barsTyped) setBarsText(String(defaultSectionBars(value)));
  };

  const handleBarsInput = (value: string) => {
    setBarsText(value);
    setBarsTyped(value.trim() !== '');
  };

  // A genre is served by its own model when this install has it, as the backend
  // dispatches; 'none' keeps the model
  const pickGenre = (value: string) => {
    const genre = value === 'none' ? null : value;
    setGenre(genre);
    const served = genre ? genreModels[genre] : undefined;
    if (served && models.some((m) => m.key === served)) setModelKey(served);
  };

  // A model sets the genre it implies, as a pick in Input → Generate does
  const pickModel = (key: string) => {
    setModelKey(key);
    const genre = genreForModel(key);
    if (genre) setGenre(genre);
  };

  const handleGenerate = async () => {
    if (!check.ok || runningRef.current) return;
    runningRef.current = true;
    setRunning(true);
    setRunStart(sectionPlacement(song, check.bars).startMeasure);
    setError(null);
    const result = await runSectionGenerate({ sectionType, chordCount: check.count });
    runningRef.current = false;
    setRunning(false);
    if (result.status === 'done') {
      onGenerated();
      dialogRef.current?.close();
    } else if (result.status === 'failed') {
      setError(result.message);
    }
  };

  const startRun = async () => {
    if (blocker !== null || targetBars === null || busy || planRunningRef.current) return;
    planRunningRef.current = true;
    await startPlan({ targetBars, wantOutro, contrast, chorusMode, modelKey });
    if (!planRunningRef.current) return;
    planRunningRef.current = false;
    const result = usePlanStore.getState();
    if (result.status !== 'done') return;
    result.applyPlan();
    // Applying can reject an obsolete result if the score changed during the run.
    if (usePlanStore.getState().applied) {
      onGenerated();
      dialogRef.current?.close();
    }
  };

  // The one exit path: Esc and close() both end up here via the native
  // 'close' event.
  const handleClose = () => {
    if (planRunningRef.current) {
      planRunningRef.current = false;
      cancelPlan();
    }
    if (runningRef.current) {
      runningRef.current = false;
      cancelSectionGenerate();
    }
    onClose();
  };

  return (
    <dialog
      ref={dialogRef}
      onClose={handleClose}
      aria-labelledby={headingId}
      className="m-auto p-0 backdrop:bg-black/60"
      style={{
        width: 'min(440px, calc(100vw - 32px))',
        background: 'var(--bg-surface)',
        color: 'var(--text-primary)',
        border: '1px solid var(--border)',
        borderTop: '3px solid var(--brand-yellow)',
      }}
    >
      <form
        className="p-5 space-y-4"
        aria-busy={busy}
        onSubmit={(e) => {
          e.preventDefault();
          if (tab === 'section') void handleGenerate();
          else void startRun();
        }}
      >
        <div>
          <h2
            id={headingId}
            className="font-display text-2xl leading-tight"
            style={{ color: 'var(--text-heading)', fontWeight: 700 }}
          >
            New from Song Form
          </h2>
          <button
            type="button"
            disabled={busy}
            onClick={() => {
              dialogRef.current?.close();
              onEditSections();
            }}
            className="mt-2 px-2 py-1 text-xs btn-hover disabled:opacity-40"
            style={{ background: 'var(--btn-bg)', color: 'var(--text-primary)' }}
            aria-haspopup="dialog"
          >
            Edit Song Form
          </button>
        </div>

        {sketch > 0 && (
          <div className="space-y-2 text-xs" style={labelStyle}>
            <p>Generation continues after the current {sketch}-bar score.</p>
            <button
              type="button"
              disabled={busy}
              onClick={() => {
                try {
                  onNewSong();
                  setTargetBars(DEFAULT_TARGET_BARS);
                  setError(null);
                } catch (failure) {
                  setError(failure instanceof Error ? failure.message : 'The current song could not be saved.');
                }
              }}
              className="px-3 py-1.5 text-xs btn-hover disabled:opacity-40"
              style={fieldStyle}
            >
              Start a new song
            </button>
            <p>Your current score is kept in Projects before starting a new song.</p>
          </div>
        )}

        {error && tab === 'full' && <p role="alert" className="text-xs" style={valueStyle}>{error}</p>}

        {/* Section / Full form tabs, in the Input sub-tab style. Locked
            while a section generation runs, so its progress stays in view. */}
        <div role="tablist" className="flex" style={{ borderBottom: '1px solid var(--border)' }}>
          {FORM_TABS.map((t) => {
            const active = tab === t.id;
            return (
              <button
                key={t.id}
                type="button"
                role="tab"
                aria-selected={active}
                disabled={busy}
                onClick={() => setTab(t.id)}
                className="flex-1 px-3 py-2 text-xs font-semibold uppercase tracking-wide transition-colors btn-hover"
                style={{
                  color: active ? 'var(--text-heading)' : 'var(--text-muted)',
                  borderTop: active ? '2px solid var(--brand-yellow)' : '2px solid transparent',
                  background: 'transparent',
                }}
              >
                {t.label}
              </button>
            );
          })}
        </div>

        {/* The song's key and genre and the chord model, shared with the controls
            above the staff and Input → Generate. Locked while a section generates. */}
        <div className="grid grid-cols-[auto_1fr] items-center gap-x-3 gap-y-2.5">
          <label htmlFor={keyId} className={fieldLabelClass} style={labelStyle}>Key</label>
          <select
            id={keyId}
            value={song.key}
            onChange={(e) => setKey(e.target.value)}
            disabled={busy}
            className="w-full px-2 py-1.5 text-sm"
            style={fieldStyle}
            title="The song's key, as the Key above the staff sets it: a new key transposes the score"
          >
            {/* An unsupported song key stays listed (disabled), so the select shows the song's own key */}
            {!STANDARD_KEYS.includes(song.key) && (
              <option value={song.key} disabled>{song.key} — choose a supported key</option>
            )}
            {STANDARD_KEYS.map((k) => (
              <option key={k} value={k}>{k}</option>
            ))}
          </select>

          <label htmlFor={genreId} className={fieldLabelClass} style={labelStyle}>Genre</label>
          <select
            id={genreId}
            value={genreOptionValue(song.genre)}
            onChange={(e) => pickGenre(e.target.value)}
            disabled={busy}
            className="w-full px-2 py-1.5 text-sm"
            style={fieldStyle}
            title="The song's genre: it picks the model it is served by, and Full form follows it"
          >
            {GENRE_OPTIONS.map((g) => (
              <option key={g.value} value={g.value}>{g.label}</option>
            ))}
          </select>

          <label htmlFor={modelId} className={fieldLabelClass} style={labelStyle}>Model</label>
          <select
            id={modelId}
            value={modelKey}
            onChange={(e) => pickModel(e.target.value)}
            disabled={busy}
            className="w-full px-2 py-1.5 text-sm"
            style={fieldStyle}
            title="Chord generation model"
          >
            {!models.some((m) => m.key === modelKey) && <option value={modelKey}>{modelLabel ?? modelKey}</option>}
            {models.map((m) => (
              <option key={m.key} value={m.key}>{m.label}</option>
            ))}
          </select>
        </div>

        {tab === 'section' && (
          <>
        <div className="grid grid-cols-[auto_1fr] items-center gap-x-3 gap-y-2.5">
          <label htmlFor={sectionId} className={fieldLabelClass} style={labelStyle}>
            Section
          </label>
          <select
            id={sectionId}
            ref={sectionRef}
            value={sectionType}
            onChange={(e) => handleSectionChange(e.target.value)}
            disabled={running}
            className="w-full px-2 py-1.5 text-sm"
            style={fieldStyle}
          >
            {SECTION_GROUPS.map((group) => (
              <optgroup key={group.id} label={group.label}>
                {sectionsInGroup(group.id).map((type) => (
                  <option key={type} value={type}>{sectionLabel(type)}</option>
                ))}
              </optgroup>
            ))}
          </select>

          <label htmlFor={barsId} className={fieldLabelClass} style={labelStyle}>
            Bars
          </label>
          <div className="flex items-center gap-2 min-w-0">
            <input
              id={barsId}
              type="text"
              inputMode="numeric"
              autoComplete="off"
              value={barsText}
              onChange={(e) => handleBarsInput(e.target.value)}
              disabled={running}
              aria-invalid={!check.ok}
              aria-describedby={hintId}
              className="tabular w-16 shrink-0 px-2 py-1.5 text-sm text-center"
              style={fieldStyle}
            />
            <span
              id={hintId}
              aria-live="polite"
              className="text-xs"
              style={{ color: check.ok ? 'var(--text-soft)' : 'var(--text-heading)' }}
            >
              {check.ok ? barCountSummary(check.bars) : check.message}
            </span>
          </div>
        </div>

        <p className="text-xs" style={labelStyle}>
          {startMeasure !== null && check.ok
            ? <>Placed at <span style={valueStyle}>{barRange(startMeasure, check.bars)}</span></>
            : 'Placed after the last bar that has chords'}
        </p>

        {running && (
          <>
            <GenerateProgress phase={phase} />
            {/* The chips are visual; screen readers get the phase line. */}
            <p className="sr-only" role="status">{phase?.label ?? 'Generating'}</p>
          </>
        )}

        {error && (
          <p
            role="alert"
            className="text-xs px-3 py-2"
            style={{
              color: 'var(--text-heading)',
              background: 'var(--btn-bg)',
              borderLeft: '3px solid var(--brand-yellow)',
            }}
          >
            {error}
          </p>
        )}

        <div className="flex justify-end gap-2">
          <button
            type="button"
            onClick={() => dialogRef.current?.close()}
            className="px-3 py-1.5 text-xs btn-hover"
            style={{ background: 'var(--btn-bg)', color: 'var(--text-primary)', border: '1px solid var(--border)' }}
            title={running ? 'Stop generating and close' : 'Close without generating'}
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={generateDisabled}
            className="px-4 py-1.5 text-xs font-bold btn-hover"
            style={{
              background: 'var(--text-heading)',
              color: 'var(--bg-primary)',
              // undefined when enabled, so btn-hover's hover dimming applies
              opacity: generateDisabled ? 0.6 : undefined,
            }}
          >
            {running ? 'Generating…' : 'Generate'}
          </button>
        </div>
          </>
        )}

        {tab === 'full' && (
          <>
            <fieldset disabled={planning} className="grid grid-cols-[auto_1fr] items-center gap-x-3 gap-y-2.5">
              <ChoiceRow<TargetBars | 'unavailable'>
                label="Length"
                options={TARGET_BARS.map((bars) => ({ value: bars, label: `${bars} bars`, disabled: !targets.includes(bars) }))}
                value={targetBars ?? 'unavailable'}
                onChange={(value) => { if (value !== 'unavailable') setTargetBars(value); }}
              />
              <ChoiceRow
                label="Outro"
                options={[{ value: true, label: 'yes' }, { value: false, label: 'no' }]}
                value={wantOutro}
                onChange={setWantOutro}
              />
              <ChoiceRow
                label="Contrast"
                options={CONTRASTS.map((level) => ({ value: level, label: level }))}
                value={contrast}
                onChange={setContrast}
              />
              <ChoiceRow<ChorusMode>
                label="Chorus returns"
                options={[{ value: 'reuse', label: 'Keep the theme' }, { value: 'new', label: 'New progression' }]}
                value={chorusMode}
                onChange={setChorusMode}
              />
            </fieldset>

            {chorusMode === 'reuse' && (
              <p className="text-xs" style={labelStyle}>
                Same-length choruses keep the first progression; the ending follows the form.
              </p>
            )}

            {/* The sketch as the planner sees it: the song up to its last used bar. */}
            <p className="text-xs" style={labelStyle}>
              Sketch{' '}
              <span style={valueStyle}>
                {sketch === 0 ? 'empty score' : `${sketch} ${sketch === 1 ? 'bar' : 'bars'}${sectionCount ? `, ${sectionCount} ${sectionCount === 1 ? 'section' : 'sections'}` : ''}`}
                {targetBars === null ? ' · no room to extend' : ` → ${targetBars} bars`}
              </span>
            </p>

            <p className="text-xs" style={labelStyle}>Genre template for the form · selected Music Transformer for chords.</p>
            {planning && <OperationProgress label={planProgress} />}
            {planStatus === 'failed' && (
              <p role="alert" className="text-xs" style={{ color: 'var(--text-heading)' }}>
                {planError ?? 'The song form could not be generated.'}
              </p>
            )}
            {planStatus !== 'idle' && (
              <details className="text-xs" style={labelStyle}>
                <summary className="cursor-pointer">Run details</summary>
                <div className="mt-2"><PlanRunBand /></div>
              </details>
            )}

            {blocker && (
              <p
                role="alert"
                className="text-xs px-3 py-2"
                style={{
                  color: 'var(--text-heading)',
                  background: 'var(--btn-bg)',
                  borderLeft: '3px solid var(--brand-yellow)',
                }}
              >
                {blocker}
              </p>
            )}

            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={() => dialogRef.current?.close()}
                  className="px-3 py-1.5 text-xs btn-hover"
                  style={{ background: 'var(--btn-bg)', color: 'var(--text-primary)', border: '1px solid var(--border)' }}
                  title={planning ? 'Stop generating and close' : 'Close without planning'}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={planDisabled}
                  className="px-4 py-1.5 text-xs font-bold btn-hover"
                  style={{
                    background: 'var(--text-heading)',
                    color: 'var(--bg-primary)',
                    // undefined when enabled, so btn-hover's hover dimming applies
                    opacity: planDisabled ? 0.6 : undefined,
                  }}
                  title="Generate the genre form with the selected Music Transformer"
                >
                  {planning ? 'Generating…' : 'Generate full form'}
                </button>
              </div>
            </div>
          </>
        )}
      </form>
    </dialog>
  );
}
