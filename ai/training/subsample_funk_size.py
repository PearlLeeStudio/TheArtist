"""F1 Stage A helper: sub-sample each genre to funk-size for controlled sweep.

Writes a COMBINED funksize_subsample.{train,val}.jsonl that the modified
dataset.py picks up when any source starts with "funksize_subsample_".

Source field is rewritten so source filtering works:
  original: source="chordonomicon_<genre>"
  subsample: source="funksize_subsample_<genre>"

Test split is NOT sub-sampled — Stage A trains on matched-size data but
evaluates on the original full test set so cross-genre numbers remain
comparable.

Deterministic (seed 42). Re-running is safe (overwrites prior output).
"""
from __future__ import annotations
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPLIT_DIR = ROOT / "data" / "splits"

GENRES = ["blues", "bossa", "country", "electronic", "folk", "funk",
          "gospel", "hip_hop", "rnb_soul", "rock"]  # classical excluded (val=37)
SEED = 42


def _load_genre_songs(split: str, genre: str) -> list[dict]:
    f = SPLIT_DIR / f"{split}.jsonl"
    if not f.exists():
        return []
    out = []
    for line in f.read_text().splitlines():
        if not line.strip():
            continue
        try:
            song = json.loads(line)
        except json.JSONDecodeError:
            continue
        # Match either explicit chordonomicon_<genre> source OR genre field
        src = song.get("source", "")
        g = song.get("genre", "").lower()
        if src == f"chordonomicon_{genre}" or g == genre:
            out.append(song)
    return out


def main():
    funk_train = _load_genre_songs("train", "funk")
    funk_val = _load_genre_songs("val", "funk")
    n_train_ref = len(funk_train)
    n_val_ref = len(funk_val)
    print(f"Reference funk size: train={n_train_ref}, val={n_val_ref}")
    if n_train_ref == 0:
        raise SystemExit("FAIL: funk has 0 train songs — splits/train.jsonl missing or no funk genre")

    rng = random.Random(SEED)
    train_out: list[str] = []
    val_out: list[str] = []
    for genre in GENRES:
        for split, ref_n, accumulator in (
            ("train", n_train_ref, train_out),
            ("val", n_val_ref, val_out),
        ):
            full = _load_genre_songs(split, genre)
            if len(full) <= ref_n:
                sampled = full
            else:
                sampled = rng.sample(full, ref_n)
            # Rewrite source field so dataset.py filter matches
            for song in sampled:
                song = dict(song)  # shallow copy
                song["source"] = f"funksize_subsample_{genre}"
                accumulator.append(json.dumps(song))
            print(f"  {genre} {split}: {len(full)} → {len(sampled)} (source=funksize_subsample_{genre})")

    train_file = SPLIT_DIR / "funksize_subsample.train.jsonl"
    val_file = SPLIT_DIR / "funksize_subsample.val.jsonl"
    train_file.write_text("\n".join(train_out) + ("\n" if train_out else ""))
    val_file.write_text("\n".join(val_out) + ("\n" if val_out else ""))
    print(f"\nWrote {train_file} ({len(train_out)} songs)")
    print(f"Wrote {val_file} ({len(val_out)} songs)")


if __name__ == "__main__":
    main()
