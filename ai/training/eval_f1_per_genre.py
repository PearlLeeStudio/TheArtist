"""Measure F1 base (no LoRA) val top-1/top-5/loss on each of the 11 LoRA genre val splits.

Mirrors lora_train.py setup exactly so numbers are directly comparable:
- Load F1 base from checkpoints/ft_jazz_pop80/best.pt
- Resize token_emb + out_proj from 351 → 359, init new rows from [GENRE:none]
- For each chordonomicon_<genre> val split: include_extra_genres=True, augment_keys=False
- Use the same evaluate() loop as lora_train.py
- Write CSV to ai/results/f1_per_genre_baseline.csv

Usage:
    cd ai
    uv run python training/eval_f1_per_genre.py
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import ChordSequenceDataset
from lora_train import _resize_embedding_with_init, evaluate
from model import MusicTransformer
from tokenizer import ChordTokenizer
from _loader_helpers import strip_compile_prefix

PROJECT_DIR = Path(__file__).resolve().parent.parent
BASE_CKPT = PROJECT_DIR / "checkpoints" / "ft_jazz_pop80" / "best.pt"
OUT_CSV = PROJECT_DIR / "results" / "f1_per_genre_baseline.csv"

GENRES = [
    "blues", "bossa", "classical", "country", "electronic",
    "folk", "funk", "gospel", "hip_hop", "rnb_soul", "rock",
]


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    old_tok = ChordTokenizer(include_extra_genres=False)
    new_tok = ChordTokenizer(include_extra_genres=True)
    old_vocab = old_tok.vocab_size
    new_vocab = new_tok.vocab_size
    print(f"Vocab: {old_vocab} (base) -> {new_vocab} (+{new_vocab - old_vocab} extra genres)")

    print(f"Loading F1 base from {BASE_CKPT}")
    ckpt = torch.load(BASE_CKPT, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})

    model = MusicTransformer(
        vocab_size=old_vocab,
        d_model=cfg.get("d_model", 512),
        n_heads=cfg.get("n_heads", 8),
        d_ff=cfg.get("d_ff", 2048),
        n_layers=cfg.get("n_layers", 8),
        max_seq_len=cfg.get("max_seq_len", 256),
        dropout=0.0,
        pad_id=old_tok.pad_id,
    )
    model.load_state_dict(strip_compile_prefix(ckpt["model_state_dict"]))

    none_id = new_tok.encode_genre("none")
    assert none_id is not None
    _resize_embedding_with_init(model, old_vocab, new_vocab, none_id)
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters (post-resize): {n_params:,}")

    criterion = nn.CrossEntropyLoss(ignore_index=old_tok.pad_id, reduction="sum")

    rows: list[dict] = []
    for genre in GENRES:
        source = f"chordonomicon_{genre}"
        val_ds = ChordSequenceDataset(
            split="val", max_seq_len=cfg.get("max_seq_len", 256),
            augment_keys=False, sources=[source], tokenizer_version="v1",
            include_extra_genres=True,
        )
        if len(val_ds) == 0:
            print(f"  {genre}: 0 val sequences — SKIP")
            continue
        loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
        metrics = evaluate(model, loader, criterion, new_vocab, old_tok.pad_id, device, use_amp=True)
        ppl = math.exp(min(metrics["loss"], 10))
        print(f"  {genre:12s} | val_n={len(val_ds):5d} | "
              f"loss={metrics['loss']:.4f} | ppl={ppl:.2f} | "
              f"top1={metrics['top1_acc']:.2f}% | top5={metrics['top5_acc']:.2f}%")
        rows.append({
            "genre": genre,
            "val_sequences": len(val_ds),
            "val_loss": round(metrics["loss"], 4),
            "val_ppl": round(ppl, 3),
            "val_top1": round(metrics["top1_acc"], 2),
            "val_top5": round(metrics["top5_acc"], 2),
        })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["genre", "val_sequences", "val_loss", "val_ppl", "val_top1", "val_top5"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows -> {OUT_CSV}")


if __name__ == "__main__":
    main()
