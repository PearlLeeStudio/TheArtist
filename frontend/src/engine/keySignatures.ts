/** The 15 conventional signatures (up to seven sharps/flats), in both modes. */
const SIGNATURE_PAIRS = [
  ['C', 'A', 0], ['G', 'E', 1], ['D', 'B', 2], ['A', 'F#', 3],
  ['E', 'C#', 4], ['B', 'G#', 5], ['F#', 'D#', 6], ['C#', 'A#', 7],
  ['Cb', 'Ab', -7], ['Gb', 'Eb', -6], ['Db', 'Bb', -5], ['Ab', 'F', -4],
  ['Eb', 'C', -3], ['Bb', 'G', -2], ['F', 'D', -1],
] as const;

export const STANDARD_KEY_SIGNATURES = [
  ...SIGNATURE_PAIRS.map(([major, , accidentals]) => ({ label: `${major} major`, vexflow: major as string, accidentals })),
  ...SIGNATURE_PAIRS.map(([, minor, accidentals]) => ({ label: `${minor} minor`, vexflow: `${minor}m`, accidentals })),
] as const;

export const STANDARD_KEYS: readonly string[] = STANDARD_KEY_SIGNATURES.map((key) => key.label);

/** Normalize spelling only; a theoretical signature is never substituted silently. */
export function normalizeKeySignature(value: string): string | null {
  const match = value.trim().replaceAll('♯', '#').replaceAll('♭', 'b').match(/^([A-G](?:#|b)?)\s+(major|minor)$/i);
  if (!match) return null;
  const label = `${match[1][0].toUpperCase()}${match[1].slice(1)} ${match[2].toLowerCase()}`;
  return STANDARD_KEYS.includes(label) ? label : null;
}

/** Shared by the interactive score, saved preview and PDF exporter. */
export function keySignatureFor(value: string): string | null {
  const label = normalizeKeySignature(value);
  return STANDARD_KEY_SIGNATURES.find((key) => key.label === label)?.vexflow ?? null;
}
