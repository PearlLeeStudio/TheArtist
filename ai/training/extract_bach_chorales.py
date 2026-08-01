"""Extract chord progressions from Bach chorales (music21 corpus).

Replaces the missing IMSLP pipeline for the classical LoRA. Bach is not the
whole of classical (no Mozart / Beethoven / Romantic), but for *functional
harmony* training Bach chorales are the canonical reference — secondary
dominants, modal mixture, voice leading, all the building blocks F1 base
doesn't reach for by default.

Source: music21.corpus.chorales (~371 BWV-numbered chorales, public domain
because Bach died 1750).

Output: data/processed/songs_bach_classical.jsonl
        + appended to data/splits/{train,val,test}.jsonl with
          source="chordonomicon_classical" (matches LoRA config naming
          convention; 'chordonomicon' prefix is a misnomer here but kept for
          dispatch consistency).

Each chorale becomes a song with 2 chords per measure (matching tokenizer
convention). chordify() reduces 4-part vocal writing to vertical chord
sequence, then we take the chord at the start of beat 1 and start of beat 3.

Usage:
    cd ai
    uv run python training/extract_bach_chorales.py             # extract only
    uv run python training/extract_bach_chorales.py --merge     # extract + append to splits
    uv run python training/extract_bach_chorales.py --dry-run   # report counts
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from music21 import chord as m21_chord
from music21 import corpus

PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
SPLITS_DIR = PROJECT_DIR / "data" / "splits"


def _root_to_symbol(pitch_name: str) -> str:
    """music21 'B-' → 'Bb', preserve sharps."""
    return pitch_name.replace("-", "b")


def _chord_to_symbol(c: m21_chord.Chord) -> str | None:
    """music21 Chord → tokenizer chord symbol (Cmaj, Am7, etc.).

    Returns None for incomplete / ambiguous verticals (passing chords that
    aren't real triads — common in chorale inner voices).
    """
    if c.root() is None:
        return None
    root = _root_to_symbol(c.root().name)
    try:
        if c.isMajorTriad():
            return f"{root}maj"
        if c.isMinorTriad():
            return f"{root}m"
        if c.isDiminishedTriad():
            return f"{root}dim"
        if c.isAugmentedTriad():
            return f"{root}aug"
        if c.isDominantSeventh():
            return f"{root}7"
        if c.isMajorSeventh():
            return f"{root}maj7"
        if c.isMinorSeventh():
            return f"{root}m7"
        if c.isHalfDiminishedSeventh():
            return f"{root}m7b5"
        if c.isDiminishedSeventh():
            return f"{root}dim7"
    except Exception:
        return None
    # Non-triadic vertical (suspension, passing) — fall back to triad guess
    # by including only the lowest 3 unique pitch classes.
    pcs = sorted({p.pitchClass for p in c.pitches})
    if len(pcs) >= 3:
        # Treat as major if the third interval exists, minor otherwise
        third_above = (pcs[0] + 4) % 12
        if third_above in pcs:
            return f"{root}maj"
        else:
            return f"{root}m"
    return None


def extract_chorale(work) -> dict | None:
    """One Bach chorale → song dict (2 chords per measure).

    Skips chorales we can't analyze (parse failures, malformed voicings).
    """
    try:
        chordified = work.chordify()
    except Exception:
        return None

    title = work.metadata.title if work.metadata else "Bach"
    try:
        k = work.analyze("key")
        key_str = f"{k.tonic.name}{'maj' if k.mode == 'major' else 'min'}".replace("-", "b")
    except Exception:
        key_str = "Cmaj"

    bars: list[list[str]] = []
    for measure in chordified.getElementsByClass("Measure"):
        chords_in_bar: list[tuple[float, str]] = []
        for elt in measure.notesAndRests:
            if isinstance(elt, m21_chord.Chord):
                sym = _chord_to_symbol(elt)
                if sym:
                    chords_in_bar.append((float(elt.offset), sym))
        if not chords_in_bar:
            continue
        # Take chord at beat 0 (or earliest) and at beat 2 (or middle).
        chords_in_bar.sort(key=lambda x: x[0])
        first = chords_in_bar[0][1]
        # find chord at offset >= 2 (second half), else use middle index
        mid_chord = first
        for off, sym in chords_in_bar:
            if off >= 2.0:
                mid_chord = sym
                break
        else:
            if len(chords_in_bar) >= 2:
                mid_chord = chords_in_bar[len(chords_in_bar) // 2][1]
        bars.append([first, mid_chord])

    if len(bars) < 2:
        return None

    return {
        "title": str(title),
        "source": "chordonomicon_classical",  # naming convention for dispatch
        "key": key_str,
        "time_signature": "4/4",
        "genre": "classical",
        "bars": bars,
    }


def extract_all() -> list[dict]:
    songs = []
    iterator = corpus.chorales.Iterator()
    total = 0
    failed = 0
    for c in iterator:
        total += 1
        song = extract_chorale(c)
        if song is None:
            failed += 1
            continue
        songs.append(song)
        if total % 50 == 0:
            print(f"  ...processed {total}, kept {len(songs)}", flush=True)
    print(f"Total: {total} chorales, kept {len(songs)}, failed {failed}")
    return songs


def write_jsonl(songs: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in songs:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Wrote {len(songs)} entries to {path}")


def merge_into_splits(songs: list[dict], seed: int = 42) -> None:
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    rng.shuffle(songs)

    n = len(songs)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    chunks = {
        "train": songs[:n_train],
        "val": songs[n_train : n_train + n_val],
        "test": songs[n_train + n_val :],
    }
    for split, chunk in chunks.items():
        split_file = SPLITS_DIR / f"{split}.jsonl"
        with open(split_file, "a", encoding="utf-8") as f:
            for s in chunk:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"  appended {len(chunk):4d} entries to {split_file.name}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--merge", action="store_true", help="Append to splits/")
    p.add_argument("--dry-run", action="store_true", help="No file write")
    args = p.parse_args()

    print("Extracting Bach chorales from music21 corpus...")
    songs = extract_all()

    if args.dry_run:
        print(f"Dry run: would write {len(songs)} chorales")
        return

    out_path = PROCESSED_DIR / "songs_bach_classical.jsonl"
    write_jsonl(songs, out_path)

    if args.merge:
        print("\nMerging into splits/ ...")
        merge_into_splits(songs)
        print("Done. Cache files in data/cache/ are stale — will rebuild on next training run.")
    else:
        print(f"\nReview {out_path} then re-run with --merge to append to splits.")


if __name__ == "__main__":
    main()
