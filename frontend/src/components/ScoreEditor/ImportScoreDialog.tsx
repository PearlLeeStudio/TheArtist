import { useEffect, useId, useRef, useState } from 'react';
import OperationProgress from '../OperationProgress';
import type { CSSProperties, KeyboardEvent } from 'react';
import type { Song } from '../../models/types';
import { STANDARD_KEYS } from '../../engine/keySignatures';
import { beginTiming } from '../../engine/experienceTiming.ts';
import { SECTION_GROUPS, isSectionType, sectionLabel, sectionsInGroup } from '../../engine/songForm';
import { analyzeScoreFile } from '../../engine/scoreImportClient';
import {
  buildScoreImportForEditor,
  checkScoreFile,
  createScoreImportDraft,
  IMPORT_TIME_SIGNATURES,
  MAX_SCORE_IMPORT_BYTES,
  scoreImportReadyStatus,
  validateScoreImportForEditor,
} from '../../engine/scoreImport';
import type { ImportSectionDraft, ScoreImportDraft, ScoreImportValidation, ScoreMime } from '../../engine/scoreImport';

interface ImportScoreDialogProps {
  onClose: () => void;
  onImported: (song: Song) => void;
}

interface SourceFile {
  file: File;
  mimeType: ScoreMime;
}

type DraftUpdate = (current: ScoreImportDraft) => ScoreImportDraft;

/** Fields and buttons share one look. */
const controlStyle: CSSProperties = {
  background: 'var(--btn-bg)', border: '1px solid var(--border)', color: 'var(--text-primary)',
};
const mutedStyle: CSSProperties = { color: 'var(--text-muted)' };
/** Printed chords stay as written; passages without chords are analyzed from the notes. */
const IMPORT_MODE = 'infer_harmony';
/** Leading bytes read for checkScoreFile's file-signature check. */
const SIGNATURE_BYTES = 16;
const ANALYSIS_TIMEOUT_MS = 180_000;
/** Mirrors the private MAX_IMPORT_SECTIONS in engine/scoreImport.ts, the most
 *  sections draft validation accepts; Add section stops there. Keep the two equal. */
const MAX_IMPORT_SECTIONS = 64;

/** The meters written in the source, each once, e.g. ['4/4', '3/4']. */
function distinctSourceMeters(draft: ScoreImportDraft): string[] {
  return [...new Set(draft.measures.flatMap((measure) => measure.timeSignature ? [measure.timeSignature.join('/')] : []))];
}

function isEditorMeter(meter: string): boolean {
  return (IMPORT_TIME_SIGNATURES as readonly string[]).includes(meter);
}

/** Why the import button is disabled, or '' when it is enabled. `validation`
 *  is null until a draft exists. */
function importBlockReason(reading: boolean, running: boolean, validation: ScoreImportValidation | null): string {
  if (reading) return 'Wait for the file check to finish.';
  if (running) return 'Wait for the score analysis to finish.';
  if (!validation) return 'Choose your file, then select Analyze and import.';
  if (validation.valid) return '';
  const corrections = validation.errors.length + validation.sectionErrors.filter(Boolean).length;
  return `Fix ${corrections} highlighted ${corrections === 1 ? 'field' : 'fields'} to enable import.`;
}

/** Id of the score-fields group in the review; the footer's error button focuses it. */
function scoreFieldsId(idPrefix: string): string {
  return `${idPrefix}-score-fields`;
}

/** Id of one section's fields in the review; the footer's "Fix section" button focuses it. */
function sectionFieldsId(idPrefix: string, index: number): string {
  return `${idPrefix}-section-${index}`;
}

/** Scroll a review field into view and focus it, first opening the collapsed
 *  section list around it, if any. */
function focusReviewField(fieldId: string) {
  const field = document.getElementById(fieldId);
  const details = field?.closest('details');
  if (details) details.open = true;
  field?.scrollIntoView({ block: 'center', behavior: 'smooth' });
  field?.focus({ preventScroll: true });
}

/** Enter in a text field would submit the form; only the import button imports. */
function preventEnterSubmit(event: KeyboardEvent<HTMLFormElement>) {
  if (event.key === 'Enter' && event.target instanceof HTMLInputElement && event.target.type !== 'file') event.preventDefault();
}

function blankSection(measureCount: number): ImportSectionDraft {
  return { type: null, label: 'Section', startBar: '1', endBar: String(measureCount || 1), origin: 'manual', note: '' };
}

/** Original → recognition → the score opens on the staff. The review form
 *  appears only when the recognized project fields (key, meter, tempo,
 *  sections) cannot be imported as read; chords are corrected on the staff.
 *  A selected File is released after analysis, cancellation or close. */
export default function ImportScoreDialog({ onClose, onImported }: ImportScoreDialogProps) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const sourceRef = useRef<SourceFile | null>(null);
  const controllerRef = useRef<AbortController | null>(null);
  // Bumped to invalidate pending file checks and analyses: on mount, a new
  // file, cancel, retry and close.
  const sequenceRef = useRef(0);
  const runningRef = useRef(false);
  const mountedRef = useRef(true);
  const closedRef = useRef(false);
  const id = useId();
  const [source, setSource] = useState<SourceFile | null>(null);
  const [draft, setDraft] = useState<ScoreImportDraft | null>(null);
  const [reading, setReading] = useState(false);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState('');

  useEffect(() => {
    mountedRef.current = true;
    sequenceRef.current++;
    if (dialogRef.current && !dialogRef.current.open) dialogRef.current.showModal();
    fileRef.current?.focus();
    return () => {
      mountedRef.current = false;
      controllerRef.current?.abort();
      controllerRef.current = null;
      sourceRef.current = null;
    };
  }, []);

  const validation = draft ? validateScoreImportForEditor(draft) : null;
  const canImport = !!draft && !!validation?.valid && !running && !reading;
  const blockReason = importBlockReason(reading, running, validation);
  const invalid = !!validation && !validation.valid;
  const firstSectionError = validation?.sectionErrors.findIndex(Boolean) ?? -1;

  /** Whether work started under `sequence` may still update the dialog. */
  const isCurrent = (sequence: number) => mountedRef.current && sequence === sequenceRef.current;

  const cancelRecognition = () => {
    sequenceRef.current++;
    controllerRef.current?.abort();
    controllerRef.current = null;
    sourceRef.current = null;
    if (fileRef.current) fileRef.current.value = '';
    setSource(null);
    runningRef.current = false;
    setRunning(false);
    setReading(false);
    setStatus('Analysis cancelled.');
  };

  // Esc, Close and a successful import all end here via the native 'close' event.
  const handleClose = () => {
    if (closedRef.current) return;
    closedRef.current = true;
    sequenceRef.current++;
    controllerRef.current?.abort();
    controllerRef.current = null;
    sourceRef.current = null;
    if (fileRef.current) fileRef.current.value = '';
    setSource(null);
    setDraft(null);
    setRunning(false);
    setReading(false);
    runningRef.current = false;
    onClose();
  };

  const chooseFile = async (file: File | undefined) => {
    if (!file) return;
    cancelRecognition();
    const sequence = ++sequenceRef.current;
    sourceRef.current = null;
    setSource(null);
    setDraft(null);
    setError(null);
    setStatus('');
    setReading(true);
    try {
      // Reject oversize files before reading any bytes.
      if (file.size > MAX_SCORE_IMPORT_BYTES) throw new Error('Choose a file no larger than 8 MiB.');
      const header = new Uint8Array(await file.slice(0, SIGNATURE_BYTES).arrayBuffer());
      if (!isCurrent(sequence)) return;
      const checked = checkScoreFile(file, header);
      if (!checked.ok) throw new Error(checked.message);
      if (checked.mimeType !== 'application/pdf') throw new Error('Choose a readable vector PDF. Scanned PDFs and images are not supported.');
      const selected = { file, mimeType: checked.mimeType };
      sourceRef.current = selected;
      setSource(selected);
    } catch (caught) {
      if (isCurrent(sequence)) {
        if (fileRef.current) fileRef.current.value = '';
        setError(caught instanceof Error ? caught.message : 'The file could not be read.');
      }
    } finally {
      if (isCurrent(sequence)) setReading(false);
    }
  };

  /** Read the score on the application server using local analysis. */
  const recognize = async (selected: SourceFile, sequence: number) => {
    if (!isCurrent(sequence) || closedRef.current || runningRef.current) return;
    const controller = new AbortController();
    controllerRef.current = controller;
    runningRef.current = true;
    setRunning(true);
    setDraft(null);
    setError(null);
    setStatus('Preparing the score file for analysis…');
    let timedOut = false;
    const timer = window.setTimeout(() => { timedOut = true; controller.abort(); }, ANALYSIS_TIMEOUT_MS);
    const finish = beginTiming('import.review', 'local');
    const report = (label: string) => { if (isCurrent(sequence)) setStatus(label); };
    try {
      const recognized = await analyzeScoreFile(selected.file, selected.mimeType, IMPORT_MODE,
        { signal: controller.signal, onStatus: report });
      if (!isCurrent(sequence)) return;
      const created = createScoreImportDraft(recognized, selected.file.name);
      finish();
      // Open the score as soon as it is read. Reading provenance, unresolved
      // bars and Review marks stay on the staff; only unusable project
      // fields stop here for correction.
      if (validateScoreImportForEditor(created).valid &&
          openImported(created)) return;
      setDraft(created);
      setStatus(`${scoreImportReadyStatus()} Correct the highlighted fields to import.`);
    } catch (caught) {
      if (!isCurrent(sequence)) return;
      setStatus('');
      setError(timedOut ? 'Analysis timed out. Try a smaller file.'
        : controller.signal.aborted ? 'Analysis cancelled.'
          : caught instanceof Error ? caught.message : 'The score could not be analyzed. Try again.');
    } finally {
      window.clearTimeout(timer);
      finish(timedOut ? 'timeout' : controller.signal.aborted ? 'cancelled' : 'error');
      if (isCurrent(sequence)) {
        runningRef.current = false;
        controllerRef.current = null;
        sourceRef.current = null;
        if (fileRef.current) fileRef.current.value = '';
        setSource(null);
        setRunning(false);
      }
    }
  };

  const startAnalysis = () => {
    const selected = sourceRef.current;
    if (!selected || runningRef.current || reading) return;
    void recognize(selected, ++sequenceRef.current);
  };

  const editDraft = (update: DraftUpdate) => {
    setDraft((current) => current ? update(current) : current);
  };

  /** Replace the editor's score with the draft and close; false (with the
   *  error shown) when the editor could not take it. */
  const openImported = (ready: ScoreImportDraft): boolean => {
    try {
      const imported = buildScoreImportForEditor(ready);
      onImported(imported);
      dialogRef.current?.close();
      return true;
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'The reviewed score could not be imported.');
      return false;
    }
  };

  const importDraft = () => {
    if (!canImport || !draft) return;
    openImported(draft);
  };

  return (
    <dialog
      ref={dialogRef}
      onClose={handleClose}
      aria-labelledby={`${id}-heading`}
      className="m-auto p-0 backdrop:bg-black/60"
      style={{
        width: 'min(720px, calc(100vw - 24px))',
        maxHeight: 'calc(100dvh - 24px)',
        overflow: 'hidden',
        background: 'var(--bg-surface)',
        color: 'var(--text-primary)',
        border: '1px solid var(--border)',
        borderTop: '3px solid var(--brand-yellow)',
      }}
    >
      <form className="flex min-h-0 flex-col" style={{ maxHeight: 'calc(100dvh - 28px)' }} onSubmit={(e) => { e.preventDefault(); importDraft(); }} onKeyDown={preventEnterSubmit}>
        <header className="flex shrink-0 items-start justify-between gap-4 px-5 py-4" style={{ borderBottom: '1px solid var(--border)' }}>
          <div>
            <h2 id={`${id}-heading`} className="font-display text-2xl font-bold" style={{ color: 'var(--text-heading)' }}>Import Score</h2>
          </div>
          <button type="button" className="shrink-0 px-3 py-1.5 text-xs btn-hover" style={controlStyle} onClick={() => dialogRef.current?.close()}>Close</button>
        </header>

        <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-5">
          <div className="grid gap-6 sm:grid-cols-[208px_minmax(0,1fr)]">
            <div className="min-w-0">
              <h3 id={`${id}-file-heading`} className="font-display text-xl font-bold" style={{ color: 'var(--text-heading)' }}>Score</h3>
              <p id={`${id}-file-hint`} className="mt-1 text-xs" style={mutedStyle}>Vector PDF · up to 8 MiB</p>
              <label
                className="relative mt-4 flex aspect-square w-full max-w-52 cursor-pointer flex-col items-center justify-center gap-3 border border-dashed p-5 text-center btn-hover focus-within:outline-2 focus-within:outline-offset-4"
                style={{ ...controlStyle, borderStyle: 'dashed', outlineColor: 'var(--brand-yellow)' }}
              >
                {/* Keep the native picker and keyboard focus; the whole square is its target.
                    Cleared after each pick so the same file can be selected again. */}
                <input
                  ref={fileRef}
                  type="file"
                  accept=".pdf,application/pdf"
                  aria-labelledby={`${id}-file-heading ${id}-file-action`}
                  aria-describedby={`${id}-file-hint ${id}-file-name`}
                  className="absolute inset-0 h-full w-full cursor-pointer opacity-0"
                  onChange={(e) => { void chooseFile(e.target.files?.[0]); e.target.value = ''; }}
                />
                <span aria-hidden="true" className="text-4xl font-light" style={{ color: 'var(--brand-yellow)' }}>+</span>
                <span id={`${id}-file-action`} className="text-sm font-semibold">Choose file</span>
                <span id={`${id}-file-name`} className="line-clamp-2 max-w-full break-all text-xs" style={mutedStyle} title={source?.file.name}>
                  {source?.file.name ?? 'No file selected'}
                </span>
              </label>
            </div>
            <div className="min-w-0 space-y-4 border-t pt-6 sm:border-t-0 sm:border-l sm:pt-0 sm:pl-6" style={{ borderColor: 'var(--border)' }}>
              <h3 className="font-display text-xl font-bold" style={{ color: 'var(--text-heading)' }}>TheArtist reader</h3>
              <p className="text-sm">Reads printed chords and notes from vector PDFs. Our Music Transformer helps complete sparse harmony.</p>
              <p className="text-xs" style={mutedStyle}>Scanned PDFs and images are not supported. Inferred chords are marked for review; unreadable passages remain unresolved.</p>
              <p className="text-xs" style={mutedStyle}>The file is analyzed on this app’s server and discarded after analysis. Your chord arrangement is kept.</p>
              <div className="flex flex-wrap gap-2">
                <button type="button" onClick={startAnalysis} disabled={!source || reading || running} className="px-3 py-2 text-xs font-bold btn-hover disabled:opacity-45" style={{ background: 'var(--brand-yellow)', color: '#1a1a1a' }}>
                  {running ? 'Analyzing…' : draft ? 'Analyze again' : 'Analyze and import'}
                </button>
                {(reading || running) && <button type="button" className="px-3 py-2 text-xs btn-hover" style={controlStyle} onClick={cancelRecognition}>Cancel</button>}
              </div>
            </div>
          </div>

          {(reading || status) && <p role="status" className="text-sm" style={mutedStyle}>{reading ? 'Checking the file…' : status}</p>}
          {(reading || running) && <OperationProgress label={reading ? 'Checking the score file…' : status} />}

          {draft && validation && <ScoreReview idPrefix={id} draft={draft} validation={validation} onEdit={editDraft} />}
        </div>

        <footer className="flex shrink-0 flex-col items-stretch gap-3 px-5 py-3 sm:flex-row sm:items-end" style={{ borderTop: '1px solid var(--border)', background: 'var(--bg-surface)' }}>
          <div className="min-w-0 flex-1 text-xs" style={mutedStyle}>
            <p>Import replaces the current score. You can undo the import in the editor.</p>
            {blockReason && !error && <p id={`${id}-import-block`} role="status" className="mt-1">{blockReason}</p>}
            {(!!error || invalid) && <div className="mt-1 max-h-20 overflow-y-auto pr-1">
              {error && <p role="alert" className="border-l-2 px-2 py-1" style={{ background: 'var(--btn-bg)', borderColor: 'var(--brand-yellow)', color: 'var(--text-primary)' }}>{error}</p>}
              {invalid && <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
                {validation.errors.length > 0 && <button type="button" className="text-left underline" onClick={() => focusReviewField(scoreFieldsId(id))}>{validation.errors.join(' ')}</button>}
                {firstSectionError >= 0 && <button type="button" className="text-left underline" onClick={() => focusReviewField(sectionFieldsId(id, firstSectionError))}>Fix section {firstSectionError + 1}: {validation.sectionErrors[firstSectionError]}</button>}
              </div>}
            </div>}
          </div>
          <button type="submit" disabled={!canImport} aria-describedby={blockReason ? `${id}-import-block` : undefined} className="shrink-0 whitespace-nowrap px-4 py-2 text-xs font-bold btn-hover" style={{ background: 'var(--brand-yellow)', color: '#1a1a1a', opacity: !canImport ? 0.45 : undefined }}>Import reviewed score</button>
        </footer>
      </form>
    </dialog>
  );
}

interface ScoreReviewProps {
  /** The dialog's useId(); field ids built from it let the footer focus them. */
  idPrefix: string;
  draft: ScoreImportDraft;
  validation: ScoreImportValidation;
  onEdit: (update: DraftUpdate) => void;
}

/** The recognized score's project fields and song-form sections, editable before import. */
function ScoreReview({ idPrefix, draft, validation, onEdit }: ScoreReviewProps) {
  const meters = distinctSourceMeters(draft);
  const changesMeter = meters.some((meter) => meter !== draft.timeSignature);
  const setFields = (patch: Partial<ScoreImportDraft>) => onEdit((d) => ({ ...d, ...patch }));
  const setSection = (index: number, patch: Partial<ImportSectionDraft>) => onEdit((d) => ({
    ...d, sections: d.sections.map((s, i) => i === index ? { ...s, ...patch } : s),
  }));

  return (
    <div className="min-w-0 space-y-4">
      <div className="space-y-2">
        <h3 className="font-display text-xl font-semibold" style={{ color: 'var(--text-heading)' }}>Review score</h3>
        <p className="text-xs" style={mutedStyle}>The score could not open as read: correct the highlighted fields, then import. Chords are reviewed in the score view.</p>
        {changesMeter && <p className="text-xs" style={mutedStyle}>The original uses {meters.join(', ')}. The editor uses {draft.timeSignature || 'one chosen meter'} throughout.</p>}
      </div>

      <div id={scoreFieldsId(idPrefix)} tabIndex={-1} className="grid grid-cols-2 gap-3">
        <label className="col-span-2 flex flex-col gap-1 text-xs">
          Title
          <input value={draft.title} maxLength={120} onChange={(e) => setFields({ title: e.target.value })} className="w-full px-2 py-1.5 text-sm" style={controlStyle} />
        </label>
        <label className="flex min-w-0 flex-col gap-1 text-xs">
          Key
          <input list={`${idPrefix}-keys`} value={draft.key} placeholder="Choose key" onChange={(e) => setFields({ key: e.target.value })} className="min-w-0 w-full px-2 py-1.5 text-sm" style={controlStyle} />
        </label>
        <datalist id={`${idPrefix}-keys`}>{STANDARD_KEYS.map((key) => <option key={key} value={key} />)}</datalist>
        <label className="flex min-w-0 flex-col gap-1 text-xs">
          Time signature
          <select value={draft.timeSignature} onChange={(e) => setFields({ timeSignature: e.target.value })} className="w-full px-2 py-1.5 text-sm" style={controlStyle}>
            <option value="">Choose meter</option>
            {draft.timeSignature && !isEditorMeter(draft.timeSignature) && <option value={draft.timeSignature}>{draft.timeSignature} — review</option>}
            {IMPORT_TIME_SIGNATURES.map((meter) => <option key={meter} value={meter}>{meter}</option>)}
          </select>
        </label>
        <label className="col-span-2 flex items-center gap-2 text-xs">
          Playback tempo
          <input type="text" inputMode="numeric" value={draft.bpm} onChange={(e) => setFields({ bpm: e.target.value })} className="w-16 px-2 py-1.5 text-center text-sm" style={controlStyle} /> BPM <span style={mutedStyle}>(set here, not read from the score)</span>
        </label>
      </div>
      {validation.errors.length > 0 && <ul className="list-disc space-y-1 pl-4 text-xs" aria-label="Score fields to correct">{validation.errors.map((message) => <li key={message}>{message}</li>)}</ul>}

      <details style={{ border: '1px solid var(--border)' }}>
        <summary className="cursor-pointer px-3 py-2 text-sm font-semibold">Song form · {draft.sections.length ? draft.sections.map((section) => section.label || 'Section').join(' · ') : 'No clear sections'}</summary>
        <div className="space-y-3 px-3 pb-3">
          {draft.sections.map((section, index) => (
            <SectionFields
              key={index}
              idPrefix={idPrefix}
              index={index}
              section={section}
              error={validation.sectionErrors[index]}
              onChange={(patch) => setSection(index, patch)}
              onRemove={() => onEdit((d) => ({ ...d, sections: d.sections.filter((_, i) => i !== index) }))}
            />
          ))}
          <button type="button" disabled={draft.sections.length >= MAX_IMPORT_SECTIONS} onClick={() => onEdit((d) => ({ ...d, sections: [...d.sections, blankSection(d.measures.length)] }))} className="px-3 py-1.5 text-xs btn-hover" style={controlStyle}>Add manually</button>
        </div>
      </details>
    </div>
  );
}

interface SectionFieldsProps {
  idPrefix: string;
  index: number;
  section: ImportSectionDraft;
  /** Validation message for this section, '' when it is valid. */
  error: string;
  onChange: (patch: Partial<ImportSectionDraft>) => void;
  onRemove: () => void;
}

function SectionFields({ idPrefix, index, section, error, onChange, onRemove }: SectionFieldsProps) {
  return (
    <div id={sectionFieldsId(idPrefix, index)} tabIndex={-1} className="space-y-2 pt-2" style={{ borderTop: '1px solid var(--border)' }}>
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs font-semibold">{section.origin === 'inferred' ? 'Inferred' : section.origin === 'explicit' ? 'Written' : 'Manual'}</span>
        <button type="button" onClick={onRemove} className="px-2 py-1 text-xs btn-hover" style={controlStyle} aria-label={`Remove section ${index + 1}`}>Remove</button>
      </div>
      <div className="grid grid-cols-2 gap-2">
        <label className="flex min-w-0 flex-col gap-1 text-xs">
          Name
          <input value={section.label} maxLength={80} onChange={(e) => onChange({ label: e.target.value })} className="w-full min-w-0 px-2 py-1.5 text-sm" style={controlStyle} />
        </label>
        <label className="flex min-w-0 flex-col gap-1 text-xs">
          Type
          <select value={section.type ?? ''} onChange={(e) => { const value = e.target.value; onChange({ type: isSectionType(value) ? value : null }); }} className="min-w-0 w-full px-2 py-1.5 text-sm" style={controlStyle}>
            <option value="">Unknown</option>
            {SECTION_GROUPS.map((group) => <optgroup key={group.id} label={group.label}>{sectionsInGroup(group.id).map((type) => <option key={type} value={type}>{sectionLabel(type)}</option>)}</optgroup>)}
          </select>
        </label>
        <label className="flex items-center gap-2 text-xs">
          From bar
          <input type="text" inputMode="numeric" value={section.startBar} onChange={(e) => onChange({ startBar: e.target.value })} className="w-14 min-w-0 px-2 py-1.5 text-sm" style={controlStyle} />
        </label>
        <label className="flex items-center gap-2 text-xs">
          To bar
          <input type="text" inputMode="numeric" value={section.endBar} onChange={(e) => onChange({ endBar: e.target.value })} className="w-14 min-w-0 px-2 py-1.5 text-sm" style={controlStyle} />
        </label>
      </div>
      {section.note && <details className="text-xs" style={mutedStyle}><summary className="cursor-pointer">Evidence</summary><p className="mt-1">{section.note}</p></details>}
      {error && <p className="text-xs" role="status">{error}</p>}
    </div>
  );
}
