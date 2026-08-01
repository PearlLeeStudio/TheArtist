"""Build a per-genre eval set of real songs for the rank-sweep paper.

Reads from ai/data/splits/{val,test}.jsonl — the held-out 20% that none of
the 18 models (Phase 0, F1-F5, 11 LoRAs) saw during training. Using held-
out data keeps the real-song eval honest (no train-set leakage). Per-
genre balanced sample, filtered by bar count to roughly 2-4 minutes of
music at typical tempo for that genre.

Sources, with title availability:
  - billboard            — McGill Billboard (CC0), titled. Used for pop/rock.
  - jht / jazzstandards
    / wjazzd / jaah
    / choco:*            — jazz datasets, titled, one-chorus to multi-chorus.
  - chordonomicon_<g>    — per-genre Chordonomicon subsets (CC BY-NC).
                           Titles are Spotify IDs (anonymous by dataset design)
                           but the chord progressions are real songs.

Output: ai/data/eval_real_songs.jsonl
Fields per row:
  id, source, genre, title (str or None — null when anonymous),
  title_is_real (bool: true for named-song sources),
  split ("val" or "test"), key, time_signature,
  n_bars, est_duration_sec, bars (list of [slot0, slot1] pairs — keyed
  to match tokenizer.encode_sequence()'s expected field name).

Usage:
  cd ai
  uv run python training/build_eval_real_songs.py
  uv run python training/build_eval_real_songs.py --seed 7 --per-genre 12
  uv run python training/build_eval_real_songs.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent.parent
AI_DIR = REPO_DIR / "ai"
SPLITS_DIR = AI_DIR / "data" / "splits"
OUT_PATH = AI_DIR / "data" / "eval_real_songs.jsonl"

GENRES = [
    "pop", "rock", "jazz", "blues", "bossa", "classical",
    "country", "electronic", "folk", "funk", "gospel",
    "hip_hop", "rnb_soul",
]

GENRE_SOURCES: dict[str, list[str]] = {
    "pop":        ["billboard"],
    "rock":       ["billboard", "chordonomicon_rock"],
    "jazz":       ["jht", "jazzstandards", "wjazzd", "jaah",
                   "choco:real-book", "choco:jazz-corpus"],
    "blues":      ["chordonomicon_blues"],
    "bossa":      ["chordonomicon_bossa"],
    "classical":  ["chordonomicon_classical"],
    "country":    ["chordonomicon_country"],
    "electronic": ["chordonomicon_electronic"],
    "folk":       ["chordonomicon_folk"],
    "funk":       ["chordonomicon_funk"],
    "gospel":     ["chordonomicon_gospel"],
    "hip_hop":    ["chordonomicon_hip_hop"],
    "rnb_soul":   ["chordonomicon_rnb_soul"],
}

# Per-genre bar count band. Tuple (min_inclusive, max_inclusive).
LENGTH_BAND: dict[str, tuple[int, int]] = {
    "pop":        (50, 130),
    "rock":       (50, 130),
    "jazz":       (16, 96),
    "blues":      (24, 120),
    "bossa":      (24, 120),
    "classical":  (8, 50),
    "country":    (30, 120),
    "electronic": (24, 120),
    "folk":       (30, 130),
    "funk":       (20, 100),
    "gospel":     (16, 100),
    "hip_hop":    (16, 100),
    "rnb_soul":   (30, 120),
}

TEMPO_BPM: dict[str, int] = {
    "pop": 110, "rock": 120, "jazz": 140, "blues": 90, "bossa": 110,
    "classical": 70, "country": 100, "electronic": 128, "folk": 100,
    "funk": 110, "gospel": 90, "hip_hop": 90, "rnb_soul": 90,
}

NAMED_SOURCES = {"billboard", "jht", "jazzstandards", "wjazzd", "jaah",
                  "choco:real-book", "choco:jazz-corpus",
                  "chordonomicon_classical"}  # Bach chorales have piece names


def est_duration_sec(n_bars: int, bpm: int, num: int = 4) -> int:
    beats = n_bars * num
    return round(beats * 60 / bpm)


def load_splits() -> list[tuple[str, dict]]:
    """Yields (split_name, record_dict) for every val + test entry."""
    out: list[tuple[str, dict]] = []
    for split in ("val", "test"):
        p = SPLITS_DIR / f"{split}.jsonl"
        if not p.exists():
            print(f"  WARN missing {p}")
            continue
        with open(p) as f:
            for line in f:
                out.append((split, json.loads(line)))
    return out


def sample_genre(
    genre: str,
    all_entries: list[tuple[str, dict]],
    per_genre: int,
    seed: int,
) -> list[tuple[str, dict]]:
    """Return a length-filtered, randomly-sampled per-genre slice."""
    sources = set(GENRE_SOURCES[genre])
    band = LENGTH_BAND[genre]
    candidates = [
        (split, d) for split, d in all_entries
        if d.get("source") in sources
        and band[0] <= len(d.get("bars", [])) <= band[1]
    ]
    if not candidates:
        print(f"  WARN {genre:12s}  no candidates in band {band}")
        return []
    rng = random.Random(seed + (hash(genre) & 0xFFFF))
    rng.shuffle(candidates)
    return candidates[:per_genre]


def to_eval_row(split: str, d: dict, genre: str, local_idx: int) -> dict:
    src = d.get("source", "?")
    bars = d.get("bars", [])
    n_bars = len(bars)
    bpm = TEMPO_BPM[genre]
    title = d.get("title") or None
    title_is_real = src in NAMED_SOURCES and bool(title)
    return {
        "id": f"{src}__{split}__{local_idx}",
        "source": src,
        "genre": genre,
        "title": title if title_is_real else None,
        "title_is_real": title_is_real,
        "split": split,
        "key": d.get("key", "Cmaj"),
        "time_signature": d.get("time_signature", "4/4"),
        "n_bars": n_bars,
        "est_duration_sec": est_duration_sec(n_bars, bpm),
        "bars": bars,  # name matches tokenizer.encode_sequence() expected key
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per-genre", type=int, default=10)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print(f"\n=== Building eval set (per_genre={args.per_genre}, seed={args.seed}) ===")
    print(f"Source: val + test only (model did not see these during training)\n")

    all_entries = load_splits()
    rows: list[dict] = []
    per_source_n: dict[str, int] = defaultdict(int)
    duration_total = 0

    for genre in GENRES:
        sampled = sample_genre(genre, all_entries, args.per_genre, args.seed)
        for i, (split, d) in enumerate(sampled):
            r = to_eval_row(split, d, genre, i)
            rows.append(r)
            per_source_n[r["source"]] += 1
            duration_total += r["est_duration_sec"]

    print(f"{'genre':<12} {'n':>4} {'bars range':>14} {'avg dur':>9}  example_title")
    for genre in GENRES:
        gr = [r for r in rows if r["genre"] == genre]
        if not gr:
            print(f"{genre:<12} {'-':>4}")
            continue
        bars = sorted(r["n_bars"] for r in gr)
        avg_dur = sum(r["est_duration_sec"] for r in gr) / len(gr)
        ex = next((r["title"] for r in gr if r["title_is_real"] and r["title"]), None)
        ex_str = f'"{ex[:40]}"' if ex else "(anon)"
        print(f"{genre:<12} {len(gr):>4} {bars[0]:>4}-{bars[-1]:<4}({len(bars)})  {avg_dur:>6.0f}s   {ex_str}")

    print(f"\nTotal: {len(rows)} songs across {len(GENRES)} genres")
    print(f"By source: {dict(sorted(per_source_n.items(), key=lambda x: -x[1]))}")
    print(f"Named-title songs: {sum(1 for r in rows if r['title_is_real'])} / {len(rows)}")
    print(f"Total est. duration: {duration_total // 60} min")

    if args.dry_run:
        print("\n[dry-run] not writing.")
        return

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(rows)} eval rows → {OUT_PATH.relative_to(REPO_DIR)}")


if __name__ == "__main__":
    main()
