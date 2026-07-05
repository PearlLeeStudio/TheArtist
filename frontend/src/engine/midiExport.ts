/**
 * MIDI export — turn a Song into a downloadable .mid file.
 *
 * Each measure has two chord slots; the chord at slot N occupies the
 * first or second half of the bar (e.g. beats 1-2 and 3-4 in 4/4).
 * Voicings already live on each chord (computed at chord-set time by
 * the voicing engine), so we only need to lay them out in time.
 *
 * Empty slots are honoured as rests — no note event, time advances.
 *
 * Output is binary-safe Uint8Array bytes; the caller wraps in a Blob
 * and pipes through a download anchor.
 */
import { Midi } from '@tonejs/midi';
import type { Song } from '../models/types';
import { generatePianoVoicing } from './voicingEngine';

/** Sanitize a free-text name into something safe for a filesystem. */
export function safeMidiFilename(name: string): string {
  const base = name.replace(/[^a-zA-Z0-9._-]+/g, '_').replace(/_+/g, '_').replace(/^_|_$/g, '');
  return (base || 'TheArtist-session') + '.mid';
}

/** Build a .mid byte array for `song`. Note voicings come from each
 *  chord's stored `voicing`; chords without voicing fall back to the
 *  voicing engine so old / partially-built songs still export cleanly. */
export function songToMidiBytes(song: Song): Uint8Array {
  const midi = new Midi();
  midi.header.setTempo(song.bpm);
  midi.header.timeSignatures.push({
    ticks: 0,
    timeSignature: [song.timeSignature[0], song.timeSignature[1]],
  });
  midi.header.name = song.title || 'Untitled';

  const track = midi.addTrack();
  track.name = 'Chords';

  const beatsPerMeasure = song.timeSignature[0];
  const slotBeats = beatsPerMeasure / 2;
  const secondsPerBeat = 60 / Math.max(1, song.bpm);
  const slotDuration = slotBeats * secondsPerBeat;

  let time = 0;
  for (const measure of song.measures) {
    for (const slotIdx of [0, 1] as const) {
      const chord = measure.chords[slotIdx];
      if (chord) {
        const voicing = chord.voicing.length > 0 ? chord.voicing : generatePianoVoicing(chord);
        for (const midiNote of voicing) {
          track.addNote({
            midi: midiNote,
            time,
            duration: slotDuration,
            velocity: 0.75,
          });
        }
      }
      time += slotDuration;
    }
  }

  return midi.toArray();
}

/** Trigger a browser download of `song` as a .mid file named after `name`. */
export function downloadSongAsMidi(song: Song, name: string): void {
  const bytes = songToMidiBytes(song);
  const blob = new Blob([bytes as BlobPart], { type: 'audio/midi' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = safeMidiFilename(name);
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  // Revoke after the click has been processed; Safari needs the URL
  // valid through the click handler dispatch.
  setTimeout(() => URL.revokeObjectURL(url), 0);
}
