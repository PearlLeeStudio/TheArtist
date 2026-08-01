"""Evaluate Phase 0 base ablation experiments.

Inputs:
  - checkpoints/phase0_full_ft_<genre>/best.pt  (11 cells, full FT from Phase 0 base)
  - checkpoints/phase0_lora_<genre>_r<rank>/    (11 cells, LoRA from Phase 0 base)

For each cell, evaluates on the held-out test split per genre. Outputs:
  - ai/results/phase0_full_ft.csv       (Phase 0 + Full FT per genre)
  - ai/results/phase0_lora.csv          (Phase 0 + LoRA per genre)
  - ai/results/phase0_vs_f1.csv         (joined: F1 vs Phase 0 base, both methods)

Tests the F1 base justification ablation: does F1's jazz harmony in
the base actually help downstream adaptation, vs starting from Phase 0?
"""
from __future__ import annotations

import csv
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
CKPT_ROOT = PROJECT_DIR / "checkpoints"
SPLITS_DIR = PROJECT_DIR / "data" / "splits"
RESULTS_DIR = PROJECT_DIR / "results"

GENRES = [
    "blues", "bossa", "classical", "country", "electronic",
    "folk", "funk", "gospel", "hip_hop", "rnb_soul", "rock",
]
SELECTED_RANKS = {
    "blues": 32, "bossa": 16, "classical": 64, "country": 4, "electronic": 8,
    "folk": 32, "funk": 64, "gospel": 4, "hip_hop": 4, "rnb_soul": 4, "rock": 16,
}


def _build_test_loader(genre: str, tokenizer: ChordTokenizer):
    ds = ChordSequenceDataset(
        SPLITS_DIR / "test.jsonl",
        tokenizer=tokenizer,
        genre_filter=genre,
        sources=[f"chordonomicon_{genre}"] if genre != "jazz" else None,
        max_seq_len=256,
    )
    if len(ds) == 0:
        return None, 0
    return DataLoader(ds, batch_size=32, shuffle=False, num_workers=2), len(ds)


def _eval_full_ft(ckpt_path: Path, genre: str, device, tokenizer):
    if not ckpt_path.exists():
        return None
    model = MusicTransformer(
        vocab_size=359, d_model=512, n_heads=8, d_ff=2048, n_layers=8,
        max_seq_len=256, dropout=0.0, pad_id=tokenizer.pad_id,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(strip_compile_prefix(ckpt["model_state_dict"]))
    model.to(device).eval()
    loader, n = _build_test_loader(genre, tokenizer)
    if loader is None:
        return None
    loss, top1, top5 = evaluate(model, loader, device, tokenizer.pad_id)
    return {"top1": round(top1, 2), "top5": round(top5, 2), "n_sequences": n}


def _eval_lora(adapter_dir: Path, genre: str, device, tokenizer):
    if not (adapter_dir / "adapter" / "adapter_config.json").exists():
        return None
    # Build Phase 0 base + grow vocab + apply adapter + apply embedding extension
    phase0_ckpt = CKPT_ROOT / "phase0_pop_baseline" / "best.pt"
    model = MusicTransformer(
        vocab_size=351, d_model=512, n_heads=8, d_ff=2048, n_layers=8,
        max_seq_len=256, dropout=0.0, pad_id=tokenizer.pad_id,
    )
    ckpt = torch.load(phase0_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(strip_compile_prefix(ckpt["model_state_dict"]))
    _resize_embedding_with_init(model, 351, 359, tokenizer.token2id["[GENRE:none]"])
    model = PeftModel.from_pretrained(model, str(adapter_dir / "adapter"))
    # Overlay embedding extension
    ext_path = adapter_dir / "embedding_extension.pt"
    if ext_path.exists():
        ext = torch.load(ext_path, map_location="cpu", weights_only=False)
        with torch.no_grad():
            model.base_model.model.token_emb.load_state_dict(ext["token_emb_state"])
            model.base_model.model.out_proj.load_state_dict(ext["out_proj_state"])
    model.to(device).eval()
    loader, n = _build_test_loader(genre, tokenizer)
    if loader is None:
        return None
    loss, top1, top5 = evaluate(model, loader, device, tokenizer.pad_id)
    return {"top1": round(top1, 2), "top5": round(top5, 2), "n_sequences": n}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = ChordTokenizer(include_extra_genres=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    full_rows = []
    lora_rows = []
    for genre in GENRES:
        # Phase 0 + Full FT
        ckpt = CKPT_ROOT / f"phase0_full_ft_{genre}" / "best.pt"
        print(f"[phase0_full_ft] {genre}: {ckpt} {'(exists)' if ckpt.exists() else '(MISSING)'}")
        result = _eval_full_ft(ckpt, genre, device, tokenizer)
        if result:
            full_rows.append({"genre": genre, **result})

        # Phase 0 + LoRA
        rank = SELECTED_RANKS[genre]
        adapter_dir = CKPT_ROOT / f"phase0_lora_{genre}_r{rank}"
        print(f"[phase0_lora] {genre} r={rank}: {adapter_dir} {'(exists)' if (adapter_dir/'adapter'/'adapter_config.json').exists() else '(MISSING)'}")
        result = _eval_lora(adapter_dir, genre, device, tokenizer)
        if result:
            lora_rows.append({"genre": genre, "rank": rank, **result})

    out_full = RESULTS_DIR / "phase0_full_ft.csv"
    with open(out_full, "w", newline="") as f:
        if full_rows:
            writer = csv.DictWriter(f, fieldnames=list(full_rows[0].keys()))
            writer.writeheader()
            writer.writerows(full_rows)
    print(f"Wrote {out_full}")

    out_lora = RESULTS_DIR / "phase0_lora.csv"
    with open(out_lora, "w", newline="") as f:
        if lora_rows:
            writer = csv.DictWriter(f, fieldnames=list(lora_rows[0].keys()))
            writer.writeheader()
            writer.writerows(lora_rows)
    print(f"Wrote {out_lora}")


if __name__ == "__main__":
    main()
