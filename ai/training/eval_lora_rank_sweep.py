"""Evaluate every saved LoRA (and full-FT) checkpoint on its genre's val split
under the EXACT conditions used by eval_f1_per_genre.py, so the numbers are
directly comparable to ai/results/f1_per_genre_baseline.csv.

For each (genre, rank) in 11 × {4, 8, 16, 32, 64} plus full-FT anchors:
  - Build base MusicTransformer with old_vocab (351)
  - Load F1 base weights
  - Resize embedding 351 → 359, init new rows from [GENRE:none]
  - For LoRA cells: PeftModel.from_pretrained(adapter_dir), then overlay the
    saved token_emb/out_proj state from embedding_extension.pt
  - For full-FT cells: build model with new_vocab (359), load best.pt state
  - Build val DataLoader with chordonomicon_<genre>, augment_keys=False,
    include_extra_genres=True (same as baseline)
  - Run lora_train.evaluate() with pad_id=old_tok.pad_id (matches baseline call)

Outputs:
  ai/results/lora_rank_sweep.csv   long format, one row per cell
  ai/results/lora_rank_sweep.md    wide table, rows=genre, cols=rank

Usage:
  cd ai
  uv run python training/eval_lora_rank_sweep.py
  uv run python training/eval_lora_rank_sweep.py --device cpu
  uv run python training/eval_lora_rank_sweep.py --only blues,rock
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch
import torch.nn as nn
from peft import PeftModel
from torch.utils.data import DataLoader

from dataset import ChordSequenceDataset
from lora_train import _resize_embedding_with_init, evaluate
from model import MusicTransformer
from tokenizer import ChordTokenizer
from _loader_helpers import strip_compile_prefix

PROJECT_DIR = Path(__file__).resolve().parent.parent
BASE_CKPT = PROJECT_DIR / "checkpoints" / "ft_jazz_pop80" / "best.pt"
CKPT_ROOT = PROJECT_DIR / "checkpoints"
BASELINE_CSV = PROJECT_DIR / "results" / "f1_per_genre_baseline.csv"
OUT_CSV = PROJECT_DIR / "results" / "lora_rank_sweep.csv"
OUT_MD = PROJECT_DIR / "results" / "lora_rank_sweep.md"

GENRES = [
    "blues", "bossa", "classical", "country", "electronic",
    "folk", "funk", "gospel", "hip_hop", "rnb_soul", "rock",
]
RANKS = [4, 8, 16, 32, 64]
FULL_FT_GENRES = ["classical", "rock"]


def _ckpt_dir_for(genre: str, rank: int) -> Path:
    suffix = "" if rank == 8 else f"_r{rank}"
    return CKPT_ROOT / f"ft_f1_lora_{genre}{suffix}"


def _full_ft_dir_for(genre: str) -> Path:
    return CKPT_ROOT / f"ft_f1_full_ft_{genre}"


def _is_complete_lora(d: Path) -> bool:
    return (d / "adapter" / "adapter_config.json").exists() and (d / "embedding_extension.pt").exists()


def _is_complete_full_ft(d: Path) -> bool:
    return (d / "best.pt").exists()


def _load_baseline() -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if not BASELINE_CSV.exists():
        return rows
    with open(BASELINE_CSV) as f:
        for r in csv.DictReader(f):
            rows[r["genre"]] = {
                "val_loss": float(r["val_loss"]),
                "val_ppl": float(r["val_ppl"]),
                "val_top1": float(r["val_top1"]),
                "val_top5": float(r["val_top5"]),
                "val_sequences": int(r["val_sequences"]),
            }
    return rows


def _build_base_model(device: torch.device, vocab_size: int, dropout: float = 0.0) -> MusicTransformer:
    """Build base model (no LoRA) with F1 weights and resized embedding."""
    ckpt = torch.load(BASE_CKPT, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    model = MusicTransformer(
        vocab_size=vocab_size,
        d_model=cfg.get("d_model", 512),
        n_heads=cfg.get("n_heads", 8),
        d_ff=cfg.get("d_ff", 2048),
        n_layers=cfg.get("n_layers", 8),
        max_seq_len=cfg.get("max_seq_len", 256),
        dropout=dropout,
        pad_id=None,
    )
    model.load_state_dict(strip_compile_prefix(ckpt["model_state_dict"]))
    return model


def _eval_lora_cell(
    genre: str, rank: int, device: torch.device,
    old_tok: ChordTokenizer, new_tok: ChordTokenizer,
) -> dict | None:
    ckpt_dir = _ckpt_dir_for(genre, rank)
    if not _is_complete_lora(ckpt_dir):
        return None
    old_vocab, new_vocab = old_tok.vocab_size, new_tok.vocab_size

    base = _build_base_model(device, vocab_size=old_vocab)
    none_id = new_tok.encode_genre("none")
    assert none_id is not None
    _resize_embedding_with_init(base, old_vocab, new_vocab, none_id)

    model = PeftModel.from_pretrained(base, str(ckpt_dir / "adapter"))

    emb_ext = torch.load(ckpt_dir / "embedding_extension.pt", map_location="cpu", weights_only=False)
    model.base_model.model.token_emb.load_state_dict(emb_ext["token_emb_state"])
    model.base_model.model.out_proj.load_state_dict(emb_ext["out_proj_state"])

    model.to(device).eval()

    val_ds = ChordSequenceDataset(
        split="val", max_seq_len=256, augment_keys=False,
        sources=[f"chordonomicon_{genre}"], tokenizer_version="v1",
        include_extra_genres=True,
    )
    if len(val_ds) == 0:
        return None
    loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
    criterion = nn.CrossEntropyLoss(ignore_index=old_tok.pad_id, reduction="sum")
    metrics = evaluate(model, loader, criterion, new_vocab, old_tok.pad_id, device, use_amp=(device.type == "cuda"))

    del model, base
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "val_sequences": len(val_ds),
        "val_loss": round(metrics["loss"], 4),
        "val_ppl": round(math.exp(min(metrics["loss"], 10)), 3),
        "val_top1": round(metrics["top1_acc"], 2),
        "val_top5": round(metrics["top5_acc"], 2),
    }


def _eval_full_ft_cell(
    genre: str, device: torch.device,
    old_tok: ChordTokenizer, new_tok: ChordTokenizer,
) -> dict | None:
    ckpt_dir = _full_ft_dir_for(genre)
    if not _is_complete_full_ft(ckpt_dir):
        return None
    new_vocab = new_tok.vocab_size

    ckpt = torch.load(ckpt_dir / "best.pt", map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    model = MusicTransformer(
        vocab_size=new_vocab,
        d_model=cfg.get("d_model", 512),
        n_heads=cfg.get("n_heads", 8),
        d_ff=cfg.get("d_ff", 2048),
        n_layers=cfg.get("n_layers", 8),
        max_seq_len=cfg.get("max_seq_len", 256),
        dropout=0.0,
        pad_id=new_tok.pad_id,
    )
    model.load_state_dict(strip_compile_prefix(ckpt["model_state_dict"]))
    model.to(device).eval()

    val_ds = ChordSequenceDataset(
        split="val", max_seq_len=256, augment_keys=False,
        sources=[f"chordonomicon_{genre}"], tokenizer_version="v1",
        include_extra_genres=True,
    )
    if len(val_ds) == 0:
        return None
    loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
    criterion = nn.CrossEntropyLoss(ignore_index=old_tok.pad_id, reduction="sum")
    metrics = evaluate(model, loader, criterion, new_vocab, old_tok.pad_id, device, use_amp=(device.type == "cuda"))

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "val_sequences": len(val_ds),
        "val_loss": round(metrics["loss"], 4),
        "val_ppl": round(math.exp(min(metrics["loss"], 10)), 3),
        "val_top1": round(metrics["top1_acc"], 2),
        "val_top5": round(metrics["top5_acc"], 2),
    }


def _write_markdown(rows: list[dict], baseline: dict[str, dict]) -> None:
    """Wide table: rows=genre, columns=baseline + each rank + full_ft. Cell = top1 (Δ)."""
    by_genre: dict[str, dict[str, dict]] = {}
    for r in rows:
        by_genre.setdefault(r["genre"], {})[str(r["rank"])] = r
    cols = ["baseline", "r=4", "r=8", "r=16", "r=32", "r=64", "full_ft"]
    header = "| genre | " + " | ".join(cols) + " |"
    sep = "|---|" + "---|" * len(cols)
    out = ["# LoRA rank sweep — val top-1 (Δ vs F1 baseline)", "",
           "Same val split / same `evaluate()` / same `pad_id` / same vocab resize as `eval_f1_per_genre.py`.",
           "Cell format: `top1% (Δ pp)` against the F1 base (no LoRA) row for that genre.",
           "", header, sep]
    for g in GENRES:
        b = baseline.get(g)
        base_str = f"{b['val_top1']:.2f}" if b else "—"
        cells = [base_str]
        for rank_label in ["4", "8", "16", "32", "64", "full_ft"]:
            cell = by_genre.get(g, {}).get(rank_label)
            if cell is None:
                cells.append("—")
                continue
            if b is None:
                cells.append(f"{cell['val_top1']:.2f}")
            else:
                delta = cell["val_top1"] - b["val_top1"]
                sign = "+" if delta >= 0 else ""
                cells.append(f"{cell['val_top1']:.2f} ({sign}{delta:.2f})")
        out.append(f"| {g} | " + " | ".join(cells) + " |")
    OUT_MD.write_text("\n".join(out) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--only", type=str, default="", help="comma-separated genre subset (e.g. 'blues,rock')")
    args = p.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}  (will share with other jobs if any)")

    only = {g.strip() for g in args.only.split(",") if g.strip()}
    genres = [g for g in GENRES if not only or g in only]

    old_tok = ChordTokenizer(include_extra_genres=False)
    new_tok = ChordTokenizer(include_extra_genres=True)
    print(f"Vocab: {old_tok.vocab_size} (base) -> {new_tok.vocab_size} (+{new_tok.vocab_size - old_tok.vocab_size} extra genres)")

    baseline = _load_baseline()
    if not baseline:
        print(f"WARN: baseline CSV missing at {BASELINE_CSV} — Δ columns will be empty")

    rows: list[dict] = []
    total = len(genres) * len(RANKS) + sum(1 for g in genres if g in FULL_FT_GENRES)
    done = 0
    for genre in genres:
        for rank in RANKS:
            done += 1
            r = _eval_lora_cell(genre, rank, device, old_tok, new_tok)
            if r is None:
                print(f"[{done}/{total}] {genre:12s} r={rank:<3d}  SKIP (not complete)")
                continue
            b = baseline.get(genre, {})
            print(f"[{done}/{total}] {genre:12s} r={rank:<3d}  "
                  f"loss={r['val_loss']:.4f} ppl={r['val_ppl']:.2f} "
                  f"top1={r['val_top1']:.2f}% top5={r['val_top5']:.2f}%")
            rows.append({
                "genre": genre, "rank": rank,
                "val_sequences": r["val_sequences"],
                "val_loss": r["val_loss"], "val_ppl": r["val_ppl"],
                "val_top1": r["val_top1"], "val_top5": r["val_top5"],
                "dval_top1_vs_baseline": round(r["val_top1"] - b.get("val_top1", 0.0), 2) if b else "",
                "dval_top5_vs_baseline": round(r["val_top5"] - b.get("val_top5", 0.0), 2) if b else "",
                "dval_loss_vs_baseline": round(r["val_loss"] - b.get("val_loss", 0.0), 4) if b else "",
            })

        if genre in FULL_FT_GENRES:
            done += 1
            r = _eval_full_ft_cell(genre, device, old_tok, new_tok)
            if r is None:
                print(f"[{done}/{total}] {genre:12s} full_ft  SKIP (not complete)")
            else:
                b = baseline.get(genre, {})
                print(f"[{done}/{total}] {genre:12s} full_ft  "
                      f"loss={r['val_loss']:.4f} top1={r['val_top1']:.2f}% top5={r['val_top5']:.2f}%")
                rows.append({
                    "genre": genre, "rank": "full_ft",
                    "val_sequences": r["val_sequences"],
                    "val_loss": r["val_loss"], "val_ppl": r["val_ppl"],
                    "val_top1": r["val_top1"], "val_top5": r["val_top5"],
                    "dval_top1_vs_baseline": round(r["val_top1"] - b.get("val_top1", 0.0), 2) if b else "",
                    "dval_top5_vs_baseline": round(r["val_top5"] - b.get("val_top5", 0.0), 2) if b else "",
                    "dval_loss_vs_baseline": round(r["val_loss"] - b.get("val_loss", 0.0), 4) if b else "",
                })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "genre", "rank", "val_sequences",
            "val_loss", "val_ppl", "val_top1", "val_top5",
            "dval_top1_vs_baseline", "dval_top5_vs_baseline", "dval_loss_vs_baseline",
        ])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows -> {OUT_CSV}")

    _write_markdown(rows, baseline)
    print(f"Wrote markdown table -> {OUT_MD}")


if __name__ == "__main__":
    main()
