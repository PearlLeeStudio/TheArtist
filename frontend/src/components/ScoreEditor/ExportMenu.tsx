import OperationProgress from '../OperationProgress';
import { useEffect, useId, useRef, useState } from 'react';
import { useSongStore } from '../../store/songStore';
import { usePlayStore } from '../../store/playStore';
import {
  buildExportSpec, downloadExportBlob, exportResponseError,
  isLoopbackHostname, requestMidiExport, requestScripterExport, safeExportFilename,
  snapshotSessionExport,
} from '../../engine/exportSpec';
import type { ExportSpec } from '../../engine/exportSpec';
import type { PdfTheme } from '../../engine/pdf';
import type { SavedSession } from '../../models/types';
import { openDawBridge } from '../../engine/dawBridge';

type DawId = 'logic' | 'ableton';
/** How the server hands the MIDI over: opened in the DAW, or shown in Finder. */
type DawHandoff = 'open' | 'reveal';
interface DawTarget { id: DawId; name: string; available: boolean; action: DawHandoff }
type ExportAction = 'midi' | 'scripter' | PdfTheme | DawId;

const DAW_DETECTION_TIMEOUT_MS = 10_000;
const DAW_HANDOFF_TIMEOUT_MS = 60_000;
const CHECKING_DAWS = 'Checking installed DAWs…';
const ERROR_TEXT_COLOR = '#d97070';

/** The DAWs the local server can hand MIDI to. */
async function fetchDawTargets(signal: AbortSignal): Promise<DawTarget[]> {
  const response = await fetch('/api/daw/targets', { signal });
  if (!response.ok) throw await exportResponseError(response);
  const body: { targets?: DawTarget[] } = await response.json();
  if (!Array.isArray(body.targets)) throw new Error('Could not read installed DAWs.');
  return body.targets;
}

/** Have the local server open the song in a DAW; resolves with how the DAW took it. */
async function handOffToDaw(target: DawId, spec: ExportSpec, controller: AbortController): Promise<DawHandoff> {
  let timedOut = false;
  const timer = setTimeout(() => { timedOut = true; controller.abort(); }, DAW_HANDOFF_TIMEOUT_MS);
  try {
    const response = await fetch('/api/daw/open', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ target, spec }), signal: controller.signal,
    });
    if (!response.ok) throw await exportResponseError(response);
    const result: { action?: string; path?: string } = await response.json();
    if (result.action !== 'open' && result.action !== 'reveal') throw new Error('The DAW did not confirm the handoff.');
    return result.action;
  } catch (error) {
    if (timedOut) throw new Error('DAW handoff timed out. Check the DAW before trying again.', { cause: error });
    throw error;
  } finally { clearTimeout(timer); }
}

/** Tooltip of a DAW button: why it is disabled, or undefined when it can be used. */
function dawUnavailableReason(onLocalhost: boolean, targets: DawTarget[] | null, targetError: string, target: DawTarget | undefined): string | undefined {
  if (!onLocalhost) return 'Use the running TheArtist MIDI Bridge on this Mac to open the MIDI.';
  if (targetError) return targetError;
  if (!targets) return CHECKING_DAWS;
  return target?.available ? undefined : 'This DAW is not installed on the server computer.';
}

/** The note under the DAW buttons. */
function dawHint(onLocalhost: boolean, targets: DawTarget[] | null, targetError: string): string {
  if (!onLocalhost) return 'Direct opening uses TheArtist MIDI Bridge on your Mac. Download MIDI also works without it.';
  if (targetError) return targetError;
  if (!targets) return CHECKING_DAWS;
  return 'Local Mac: Logic opens the MIDI; Ableton uses drag from Finder.';
}

interface ExportMenuProps {
  /** Export this saved project without replacing or reading the editor's current song. */
  session?: SavedSession;
}

export default function ExportMenu({ session }: ExportMenuProps) {
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [feedback, setFeedback] = useState<{ text: string; error: boolean } | null>(null);
  const [targets, setTargets] = useState<DawTarget[] | null>(null);
  const [targetError, setTargetError] = useState('');
  const rootRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const busyRef = useRef(false);
  const mountedRef = useRef(true);
  const requestRef = useRef<AbortController | null>(null);
  const panelId = useId();
  // The server opens the DAW on its own computer, so handoff is offered only on localhost.
  const onLocalhost = isLoopbackHostname(window.location.hostname);

  // Abort a running export when the menu unmounts.
  useEffect(() => {
    mountedRef.current = true;
    return () => { mountedRef.current = false; requestRef.current?.abort(); };
  }, []);

  // While open, a pointer press outside the menu or Escape closes it.
  useEffect(() => {
    if (!open) return;
    const closeOnOutsidePointer = (event: PointerEvent) => {
      if (event.target instanceof Node && !rootRef.current?.contains(event.target)) setOpen(false);
    };
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') { setOpen(false); triggerRef.current?.focus(); }
    };
    document.addEventListener('pointerdown', closeOnOutsidePointer);
    document.addEventListener('keydown', closeOnEscape);
    return () => {
      document.removeEventListener('pointerdown', closeOnOutsidePointer);
      document.removeEventListener('keydown', closeOnEscape);
    };
  }, [open]);

  // Every opening asks the local server again which DAWs are installed.
  useEffect(() => {
    if (!open || !onLocalhost) return;
    const controller = new AbortController();
    let timedOut = false;
    const timer = setTimeout(() => { timedOut = true; controller.abort(); }, DAW_DETECTION_TIMEOUT_MS);
    void (async () => {
      try {
        const found = await fetchDawTargets(controller.signal);
        if (!controller.signal.aborted) setTargets(found);
      } catch (error) {
        if (!controller.signal.aborted) setTargetError(error instanceof Error ? error.message : 'Could not check installed DAWs.');
        else if (timedOut) setTargetError('DAW detection did not finish. Download MIDI is still available.');
      } finally { clearTimeout(timer); }
    })();
    // Closing the menu aborts quietly. Of the two abort sources only the timeout
    // shows a message; the catch checks `timedOut`, so the 'closed' reason is
    // never read. Failures that are not aborts are still shown.
    return () => { clearTimeout(timer); controller.abort('closed'); };
  }, [open, onLocalhost]);

  const runExport = async (action: ExportAction) => {
    if (busyRef.current) return;
    busyRef.current = true;
    setBusy(true);
    setFeedback(null);
    const controller = new AbortController();
    requestRef.current = controller;
    try {
      const { song, settings } = session ? snapshotSessionExport(session) : {
        song: structuredClone(useSongStore.getState().song),
        settings: { ...usePlayStore.getState().settings },
      };
      if (action === 'midi') {
        const blob = await requestMidiExport(buildExportSpec(song, settings), controller.signal);
        if (controller.signal.aborted) return;
        downloadExportBlob(blob, safeExportFilename(song.title, 'mid'));
        setFeedback({ text: 'MIDI downloaded. Import it into a DAW to choose sounds and continue arranging.', error: false });
      } else if (action === 'scripter') {
        const blob = await requestScripterExport(buildExportSpec(song, settings), controller.signal);
        if (controller.signal.aborted) return;
        downloadExportBlob(blob, safeExportFilename(song.title, 'js'));
        setFeedback({ text: 'Scripter script downloaded. In Logic: MIDI FX ▸ Scripter ▸ Open Script in Editor, paste, Run Script.', error: false });
      } else if (action === 'logic' || action === 'ableton') {
        let handoff: DawHandoff;
        if (onLocalhost) handoff = await handOffToDaw(action, buildExportSpec(song, settings), controller);
        else {
          const bridge = openDawBridge(controller.signal);
          try {
            await bridge.ready;
            const blob = await requestMidiExport(buildExportSpec(song, settings), controller.signal);
            handoff = await bridge.send(action, song.title, blob);
          } finally { bridge.close(); }
        }
        if (controller.signal.aborted) return;
        setFeedback({ text: handoff === 'reveal'
          ? 'MIDI shown in Finder. Drag it into Ableton Live to continue.'
          : 'MIDI sent to Logic Pro. Choose or create a project there to continue.', error: false });
      } else {
        const { createSongPdf } = await import('../../engine/pdf');
        const blob = await createSongPdf(song, action);
        if (controller.signal.aborted) return;
        downloadExportBlob(blob, safeExportFilename(song.title, 'pdf'));
        setFeedback({ text: 'PDF downloaded.', error: false });
      }
    } catch (error) {
      if (!mountedRef.current) return;
      setFeedback({ text: error instanceof Error ? error.message : 'Export failed. Please try again.', error: true });
    } finally {
      busyRef.current = false;
      if (mountedRef.current) setBusy(false);
      requestRef.current = null;
    }
  };

  const toggleMenu = () => {
    // Opening forgets the last DAW check; the detection effect runs a new one.
    if (!open) { setTargets(null); setTargetError(''); }
    setOpen(!open);
  };

  const actionButton = (label: string, action: ExportAction, disabled = false, title?: string) => (
    <button type="button" onClick={() => { void runExport(action); }} disabled={busy || disabled} title={title}
      className="block w-full px-3 py-2 text-left text-xs btn-hover disabled:opacity-40 disabled:cursor-not-allowed"
      style={{ color: 'var(--text-primary)' }}>{label}</button>
  );
  const dawButton = (id: DawId, label: string) => {
    const target = targets?.find((item) => item.id === id);
    return actionButton(label, id, onLocalhost && !target?.available, dawUnavailableReason(onLocalhost, targets, targetError, target));
  };

  return (
    <div className="relative" ref={rootRef}>
      <button ref={triggerRef} type="button" aria-expanded={open} aria-controls={panelId}
        aria-label={session ? `Export ${session.name}` : undefined}
        onClick={toggleMenu}
        className={`px-3 ${session ? 'py-2' : 'py-1'} text-xs font-bold whitespace-nowrap btn-hover`}
        style={{ background: 'var(--btn-bg)', color: 'var(--text-heading)', border: '1px solid var(--border)' }}>
        {busy ? 'Exporting…' : 'Export ▾'}
      </button>
      {open && <div id={panelId} className="absolute right-0 top-full mt-2 z-50 py-1 w-72 max-w-[calc(100vw-2rem)] shadow-xl"
        aria-label="Export options" aria-busy={busy}
        style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)' }}>
        {actionButton('Download MIDI', 'midi')}
        <p className="px-3 pb-2 text-[10px]" style={{ color: 'var(--text-muted)' }}>Harmony, bass and drums · DAW sounds may differ.</p>
        {actionButton('Logic Scripter script', 'scripter')}
        <p className="px-3 pb-2 text-[10px]" style={{ color: 'var(--text-muted)' }}>Harmony and bass, played by Logic's Scripter MIDI FX.</p>
        <div style={{ borderTop: '1px solid var(--border)' }}>
          {actionButton('PDF · Current theme', 'current')}
          {actionButton('PDF · Light (print)', 'light')}
          {actionButton('PDF · Dark', 'dark')}
        </div>
        <div style={{ borderTop: '1px solid var(--border)' }}>
          {dawButton('logic', 'Open in Logic Pro')}
          {dawButton('ableton', 'Show MIDI for Ableton Live')}
        </div>
        <p className="px-3 py-2 text-[10px] leading-relaxed" style={{ color: 'var(--text-muted)' }}>
          {dawHint(onLocalhost, targets, targetError)}
          {!onLocalhost && <> <a className="underline" href="https://github.com/PearlLeeStudio/TheArtist#hardware-midi"
            target="_blank" rel="noopener noreferrer">Mac bridge setup · automatic startup</a></>}
        </p>
        {busy && <OperationProgress label="Preparing the export file…" />}
        {feedback && <p role={feedback.error ? 'alert' : 'status'} className="px-3 py-2 text-xs leading-relaxed break-words"
          style={{ color: feedback.error ? ERROR_TEXT_COLOR : 'var(--text-accent)', borderTop: '1px solid var(--border)' }}>{feedback.text}</p>}
      </div>}
    </div>
  );
}
