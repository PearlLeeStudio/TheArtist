import { calculateRomanNumeral } from '../../engine/romanNumeral';
import { chordToString } from '../../engine/chordParser';
import ChordTransitionView from './ChordTransitionView';
import { useMidiCapture, PROGRESS_TICK_MS } from './useMidiCapture';

interface MidiStatusProps {
  onMidiNotes: (notes: number[]) => void;
}

const PITCH_CLASS_NAMES = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B'];

/**
 * MIDI input panel — direct chord entry with two paths:
 *
 *   • Fast path: hold a chord for 3 s → auto-commits the active
 *     interpretation, slot advances. No second click needed.
 *
 *   • Slow path: release before 3 s → enters a Confirm card. User can
 *     still click chips to change which root interpretation commits,
 *     then clicks Confirm (or Cancel to discard). Playing a new chord
 *     during Confirm implicitly cancels.
 *
 * The state machine lives in `useMidiCapture` — this component is a
 * thin presenter that wires the hook's output to UI.
 */
export default function MidiStatus({ onMidiNotes }: MidiStatusProps) {
  const {
    enabled,
    connected,
    holdProgress,
    canEnable,
    view,
    active,
    prevSlotChord,
    nextSlotChord,
    songKey,
    toggleMidi,
    setActiveIdx,
    setConfirmActiveIdx,
    handleConfirm,
    handleCancel,
  } = useMidiCapture(onMidiNotes);

  const activeRoman = active
    ? calculateRomanNumeral({ ...active, romanNumeral: '' }, songKey)
    : '';
  const playedNoteNames = active?.voicing
    ? Array.from(new Set(active.voicing.map((m) => m % 12)))
        .sort((a, b) => a - b)
        .map((pc) => PITCH_CLASS_NAMES[pc])
    : [];

  return (
    <div className="space-y-2">
      <h3 className="text-xs uppercase tracking-wide font-semibold" style={{ color: 'var(--text-heading)' }}>
        MIDI Input
      </h3>

      <button
        onClick={toggleMidi}
        disabled={!enabled && !canEnable}
        className="w-full px-3 py-2 text-sm font-bold"
        style={enabled
          ? { background: 'var(--text-heading)', color: 'var(--bg-primary)' }
          : {
              background: 'var(--btn-bg)',
              color: 'var(--text-primary)',
              opacity: canEnable ? 1 : 0.5,
              cursor: canEnable ? 'pointer' : 'not-allowed',
            }
        }
        title={!enabled && !canEnable ? 'Select exactly one slot on the staff first' : undefined}
      >
        {enabled ? 'Disable MIDI' : 'Enable MIDI'}
      </button>

      <p className="text-[11px] italic leading-snug" style={{ color: 'var(--text-muted)' }}>
        {!canEnable
          ? 'Select one slot on the staff to enable MIDI input.'
          : 'Hold a chord for 3 s to commit, or release early to review and pick the root. The selection auto-advances — build chords one slot at a time.'}
      </p>

      {enabled && (
        <div className="text-xs" style={{ color: 'var(--text-muted)' }}>
          {connected ? (
            <span style={{ color: 'var(--text-heading)' }}>● Controller connected</span>
          ) : (
            <span style={{ color: 'var(--text-soft)' }}>○ No MIDI device detected</span>
          )}
        </div>
      )}

      {/* Detected / Confirm card */}
      {enabled && view && active && (
        <div className="p-3 space-y-2" style={{ background: 'var(--btn-bg)', border: '1px solid var(--border)' }}>
          <div className="flex items-baseline justify-between">
            <span className="text-[10px] uppercase tracking-wide font-semibold" style={{ color: 'var(--text-muted)' }}>
              {view.source === 'confirm' ? 'Review chord' : 'Detected'}
            </span>
            <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
              {view.source === 'confirm' ? 'pick root, then confirm' : 'hold 3s or release to review'}
            </span>
          </div>

          <div className="flex items-baseline gap-3">
            <span className="text-3xl font-bold leading-none" style={{ color: 'var(--text-heading)' }}>
              {chordToString(active)}
            </span>
            {activeRoman && (
              <span className="text-lg font-semibold" style={{ color: 'var(--text-primary)' }}>
                {activeRoman}
              </span>
            )}
            <span className="text-[10px] ml-auto" style={{ color: 'var(--text-muted)' }}>
              in {songKey}
            </span>
          </div>

          {/* Hold progress bar — only visible in 'live' mode */}
          {view.source === 'live' && (
            <div className="h-1.5" style={{ background: 'var(--bg-surface)', overflow: 'hidden', borderRadius: 1 }}>
              <div
                className="h-full"
                style={{
                  width: `${holdProgress * 100}%`,
                  background: holdProgress >= 1 ? 'var(--text-accent)' : 'var(--brand-yellow)',
                  transition: `width ${PROGRESS_TICK_MS}ms linear`,
                }}
              />
            </div>
          )}

          {/* Root-cycling chips — click to swap which interpretation commits */}
          {view.candidates.length > 1 && (
            <div>
              <div className="text-[10px] uppercase tracking-wide font-semibold mb-1" style={{ color: 'var(--text-muted)' }}>
                Other interpretations
              </div>
              <div className="flex flex-wrap gap-1.5">
                {view.candidates.map((c, i) => {
                  const isActive = i === view.activeIdx;
                  const rn = calculateRomanNumeral({ ...c, romanNumeral: '' }, songKey);
                  const label = chordToString(c);
                  return (
                    <button
                      key={`${label}-${i}`}
                      onClick={() =>
                        view.source === 'confirm' ? setConfirmActiveIdx(i) : setActiveIdx(i)
                      }
                      className="px-2 py-0.5 text-xs font-mono"
                      style={{
                        background: isActive ? '#facc15' : 'var(--bg-surface)',
                        color: isActive ? '#1a1a1a' : 'var(--text-primary)',
                        fontWeight: isActive ? 700 : 400,
                        border: '1px solid var(--border)',
                      }}
                      title={`Use ${label} as the interpretation`}
                    >
                      {label}{rn ? ` · ${rn}` : ''}
                    </button>
                  );
                })}
              </div>
            </div>
          )}

          {playedNoteNames.length > 0 && (
            <div className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
              Played: {playedNoteNames.join(' · ')}
            </div>
          )}

          {/* Voice-leading visualisation — show prev → current → next on
              one piano with arrows. Only in Confirm mode (during the live
              hold the user is still playing, so the prev/next context
              isn't yet meaningful). */}
          {view.source === 'confirm' && (
            <div className="pt-1">
              <ChordTransitionView
                prev={prevSlotChord}
                current={active}
                next={nextSlotChord}
              />
            </div>
          )}

          {/* Confirm / Cancel actions — only in confirm mode */}
          {view.source === 'confirm' && (
            <div className="flex gap-2 pt-1">
              <button
                onClick={handleConfirm}
                className="flex-1 px-3 py-1.5 text-xs font-bold"
                style={{ background: 'var(--text-heading)', color: 'var(--bg-primary)' }}
              >
                Confirm
              </button>
              <button
                onClick={handleCancel}
                className="px-3 py-1.5 text-xs"
                style={{ background: 'var(--bg-surface)', color: 'var(--text-primary)' }}
              >
                Cancel
              </button>
            </div>
          )}
        </div>
      )}

      {enabled && !view && (
        <p className="text-xs italic" style={{ color: 'var(--text-muted)' }}>
          Play 3+ notes to detect a chord.
        </p>
      )}
    </div>
  );
}
