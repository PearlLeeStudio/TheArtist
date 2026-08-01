"""Aggregate multi-seed Full FT runs (seeds 42, 123, 7) into per-genre
mean ± std for paper §6.2 PEFT comparison Full FT row.

Reads existing seed=42 ckpt + seeds 123/7 added by the phase-2 chain run
step 1. For each genre, evaluates all 3 seed ckpts on the held-out test
split and computes mean/std/CI.

Output: ai/results/full_ft_multi_seed.csv + full_ft_multi_seed_summary.csv
"""
from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import ChordSequenceDataset
from lora_train import _resize_embedding_with_init, evaluate
from model import MusicTransformer
from tokenizer import ChordTokenizer

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"
CKPT_ROOT = PROJECT_DIR / "checkpoints"
SPLITS_DIR = PROJECT_DIR / "data" / "splits"

GENRES = [
    "blues", "bossa", "classical", "country", "electronic",
    "folk", "funk", "gospel", "hip_hop", "rnb_soul", "rock",
]
SEEDS = [42, 123, 7]


def _ckpt_path(genre: str, seed: int) -> Path:
    """seed=42 = existing ckpt without seed suffix; others use _seedN."""
    if seed == 42:
        return CKPT_ROOT / f"ft_f1_full_ft_{genre}" / "best.pt"
    return CKPT_ROOT / f"ft_f1_full_ft_{genre}_seed{seed}" / "best.pt"


def _evaluate_ckpt(ckpt_path: Path, genre: str, device: torch.device, tokenizer: ChordTokenizer) -> tuple[float, float, int] | None:
    if not ckpt_path.exists():
        return None
    model = MusicTransformer(
        vocab_size=359, d_model=512, n_heads=8, d_ff=2048, n_layers=8,
        max_seq_len=256, dropout=0.0, pad_id=tokenizer.pad_id,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    ds = ChordSequenceDataset(
        SPLITS_DIR / "test.jsonl",
        tokenizer=tokenizer,
        genre_filter=genre,
        sources=[f"chordonomicon_{genre}"] if genre != "jazz" else None,
        max_seq_len=256,
    )
    if len(ds) == 0:
        return None
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)
    loss, top1, top5 = evaluate(model, loader, device, tokenizer.pad_id)
    return top1, top5, len(ds)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = ChordTokenizer(include_extra_genres=True)

    rows = []
    for genre in GENRES:
        seed_results = []
        for seed in SEEDS:
            ckpt = _ckpt_path(genre, seed)
            print(f"[{genre} seed={seed}] {ckpt} {'(exists)' if ckpt.exists() else '(MISSING)'}")
            result = _evaluate_ckpt(ckpt, genre, device, tokenizer)
            if result:
                top1, top5, n = result
                rows.append({"genre": genre, "seed": seed, "full_ft_top1": round(top1, 2),
                             "full_ft_top5": round(top5, 2), "n_sequences": n})
                seed_results.append(top1)

    # Aggregate per-genre
    summary_rows = []
    for genre in GENRES:
        per_genre = [r["full_ft_top1"] for r in rows if r["genre"] == genre]
        if not per_genre:
            continue
        n = len(per_genre)
        mean = statistics.mean(per_genre)
        std = statistics.stdev(per_genre) if n > 1 else 0.0
        # Welch-style 95% CI via Student t (n=3 → t=4.303)
        t95 = 4.303 if n == 3 else 1.96
        ci_half = t95 * std / math.sqrt(max(n, 1)) if n > 1 else 0.0
        summary_rows.append({
            "genre": genre, "n_seeds": n,
            "full_ft_mean": round(mean, 2),
            "full_ft_std": round(std, 3),
            "full_ft_ci_low": round(mean - ci_half, 2),
            "full_ft_ci_high": round(mean + ci_half, 2),
            "min_top1": round(min(per_genre), 2),
            "max_top1": round(max(per_genre), 2),
        })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_long = RESULTS_DIR / "full_ft_multi_seed.csv"
    with open(out_long, "w", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Wrote {out_long}")

    out_summary = RESULTS_DIR / "full_ft_multi_seed_summary.csv"
    with open(out_summary, "w", newline="") as f:
        if summary_rows:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
    print(f"Wrote {out_summary}")


if __name__ == "__main__":
    main()
