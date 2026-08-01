"""Extract per-genre chord-progression subsets from Chordonomicon raw CSV.

Produces JSONL entries usable by ChordSequenceDataset (sources=["chordonomicon_<genre>"]).
Each entry: {title, source: "chordonomicon_<genre>", key, time_signature, genre, bars}.

Output:
- Default: data/processed/songs_genres_extra.jsonl  (review-friendly, separate file)
- With --merge: appends to data/processed/songs.jsonl + recreates splits

Usage:
    cd ai
    uv run python training/extract_genre_subsets.py             # extract only
    uv run python training/extract_genre_subsets.py --merge     # extract + merge into splits
    uv run python training/extract_genre_subsets.py --dry-run   # report counts only

Genre mapping (Pearl 2026-05-09 final 13):
- country: main_genre == "country" (~53K)
- rnb_soul: main_genre == "soul" + genres tag soul/r&b/motown (~7K+)
- hip_hop: main_genre == "rap" + genres tag hip-hop/trap (~11K)
- electronic: main_genre == "electronic" + genres tag house/techno/edm/synth (~5-8K)
- funk: genres tag funk (~2.8K — note: includes funk-rock/funk-metal subtags)
- folk: genres tag folk/acoustic/singer-songwriter
- gospel: genres tag gospel (~3.7K)
- bossa: genres tag bossa/samba/latin/salsa/cumbia (~15K, expansion of partial bossa)
- rock_extra: main_genre == "rock" not yet trained (~67K, for LoRA strengthening)
- blues_extra: genres tag blues
- classical: NOT extracted here — separate IMSLP scraping pipeline
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

# Local import — share chord normalization with preprocess.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tokenizer import ChordTokenizer

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
SPLITS_DIR = Path(__file__).resolve().parent.parent / "data" / "splits"

_tok = ChordTokenizer()
_SHARP_RE = re.compile(r"^([A-G])s(?!us)")


def _chordonomicon_chord(raw: str) -> str | None:
    """Same normalization as preprocess.py."""
    raw = re.sub(r"no3d$", "", raw)
    raw = _SHARP_RE.sub(r"\1#", raw)
    return _tok.normalize_chord(raw)


# ---- Genre filter rules ------------------------------------------------

def _matches(main_genre: str, genres_str: str, rules: dict) -> bool:
    """Apply main_genre + genres-substring filter rules.

    rules = {"main": ["country"], "genres_any": ["country pop"], "main_exclude": ["..."]}
    """
    mg = (main_genre or "").lower().strip()
    gs = (genres_str or "").lower()
    if mg in rules.get("main_exclude", []):
        return False
    if mg in rules.get("main", []):
        return True
    for kw in rules.get("genres_any", []):
        if kw in gs:
            return True
    return False


GENRE_FILTERS: dict[str, dict] = {
    "country": {"main": ["country"], "genres_any": ["country pop", "country rock", "modern country"]},
    "rnb_soul": {"main": ["soul"], "genres_any": ["r&b", "rnb", "soul", "motown", "neo soul"]},
    "hip_hop": {"main": ["rap"], "genres_any": ["hip hop", "hip-hop", "trap", "rap"]},
    "electronic": {"main": ["electronic"], "genres_any": ["house", "techno", "edm", "trance", "synthwave", "dance", "electro", "drum and bass", "dnb"]},
    "funk": {"main": [], "genres_any": ["funk"]},
    "folk": {"main": [], "genres_any": ["folk", "singer-songwriter", "acoustic"]},
    "gospel": {"main": [], "genres_any": ["gospel"]},
    "bossa": {"main": [], "genres_any": ["bossa", "samba", "latin", "salsa", "cumbia"]},
    "rock_extra": {"main": ["rock"], "genres_any": []},
    "blues_extra": {"main": [], "genres_any": ["blues"]},
}


# ---- Extraction --------------------------------------------------------

def parse_chordonomicon_row(row: dict) -> dict | None:
    """Parse one Chordonomicon CSV row into a unified song dict (no genre yet)."""
    chords_str = row.get("chords", "")
    if not chords_str:
        return None
    raw_tokens = [t for t in chords_str.split() if not t.startswith("<")]
    if len(raw_tokens) < 4:
        return None
    chords = [c for c in (_chordonomicon_chord(t) for t in raw_tokens) if c]
    if len(chords) < 4:
        return None
    bars = [chords[i : i + 2] for i in range(0, len(chords), 2)]
    return {
        "title": row.get("spotify_song_id", ""),
        "key": "Cmaj",
        "time_signature": "4/4",
        "bars": bars,
    }


def extract_subsets(target_genres: list[str] | None = None) -> dict[str, list[dict]]:
    """Read raw CSV, return {genre: [song_dict, ...]} for each target genre."""
    csv_path = RAW_DIR / "chordonomicon" / "chordonomicon.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found")

    if target_genres is None:
        target_genres = list(GENRE_FILTERS.keys())

    out: dict[str, list[dict]] = {g: [] for g in target_genres}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mg = row.get("main_genre", "")
            gs = row.get("genres", "")
            base = parse_chordonomicon_row(row)
            if base is None:
                continue
            for genre in target_genres:
                rules = GENRE_FILTERS[genre]
                if _matches(mg, gs, rules):
                    entry = dict(base)
                    # Map to canonical genre token name (rock_extra → rock, blues_extra → blues)
                    canonical = genre.replace("_extra", "")
                    entry["source"] = f"chordonomicon_{canonical}"
                    entry["genre"] = canonical
                    out[genre].append(entry)
                    break  # one genre per song to avoid double-counting
    return out


# ---- Output ------------------------------------------------------------

def write_jsonl(songs_by_genre: dict[str, list[dict]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for genre, songs in songs_by_genre.items():
            for s in songs:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
                n += 1
    print(f"Wrote {n} entries to {path}")


def merge_into_splits(songs_by_genre: dict[str, list[dict]], seed: int = 42) -> None:
    """Append new genre entries to existing splits (train/val/test).

    Uses the same 80/10/10 ratio as create_splits(). Per-genre splitting
    so that each new genre has presence in train/val/test independently.
    """
    import random

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    added = {"train": 0, "val": 0, "test": 0}
    for genre, songs in songs_by_genre.items():
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
            added[split] += len(chunk)

    for split, n in added.items():
        print(f"  appended {n:6d} entries to {SPLITS_DIR}/{split}.jsonl")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--genres", nargs="*", default=None, help="Subset of genres to extract")
    p.add_argument("--merge", action="store_true", help="Append to existing splits (irreversible without backup)")
    p.add_argument("--dry-run", action="store_true", help="Just print counts, no file write")
    args = p.parse_args()

    print(f"Extracting from {RAW_DIR / 'chordonomicon' / 'chordonomicon.csv'} ...")
    out = extract_subsets(args.genres)

    print("\nCounts per genre:")
    for genre, songs in out.items():
        print(f"  {genre:14s}  {len(songs):6d}")

    if args.dry_run:
        return

    out_path = PROCESSED_DIR / "songs_genres_extra.jsonl"
    write_jsonl(out, out_path)

    if args.merge:
        print("\nMerging into splits/ ...")
        merge_into_splits(out)
        print("Done. Cache files in data/cache/ are now stale — they will rebuild on next training run.")
    else:
        print(f"\nReview {out_path} then re-run with --merge to append to splits.")


if __name__ == "__main__":
    main()
