"""F1 Stage A: evaluate trained cells on ORIGINAL (full) test set.

Controlled-data-size comparison:

> sub-sample train + val to funk size; evaluate on the **original held-out
> test set** for each genre (test set untouched).

My subsample_funk_size.py writes sub-sampled train/val data. The
funksize_subsample dataset.py path filters to source=funksize_subsample_<genre>.
For TEST evaluation we want sources=["chordonomicon_<genre>"] (original
full test). This script loads each Stage A cell, runs forward on the
original test, and writes results to controlled_funk_size.csv with
test_top1 / test_top5 / test_loss — overriding the sub-sampled val
metrics that eval_phase3_cells.py would otherwise report.

Loaded via P13c (added between P13 training and P14b eval-CSV-collator).
"""
from __future__ import annotations
import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn
from peft import PeftModel
from torch.utils.data import DataLoader

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "training"))

from dataset import ChordSequenceDataset
from model import MusicTransformer
from tokenizer import ChordTokenizer, EXTRA_GENRES
from _loader_helpers import strip_compile_prefix
from lora_train import _resize_embedding_with_init

CKPT_ROOT = PROJECT_DIR / "checkpoints"
F1_BASE = CKPT_ROOT / "ft_jazz_pop80" / "best.pt"
OUT_CSV = PROJECT_DIR / "results" / "controlled_funk_size.csv"

GENRES_FUNK_SIZED = ["blues", "bossa", "country", "electronic", "folk",
                     "funk", "gospel", "hip_hop", "rnb_soul", "rock"]
METHODS = ["lora", "bitfit", "ia3", "full"]

# Same selected-rank as the main grid — Stage A uses these per generate_phase3_configs
LORA_RANK = {
    "blues": 32, "bossa": 16, "country": 4, "electronic": 8, "folk": 32,
    "funk": 4, "gospel": 4, "hip_hop": 4, "rnb_soul": 4, "rock": 16,
}


def _build_f1(new_tok: ChordTokenizer, device):
    """Build F1 base extended to new genre vocab (351→359)."""
    old_tok = ChordTokenizer(include_extra_genres=False)
    m = MusicTransformer(
        vocab_size=old_tok.vocab_size,
        d_model=512, n_heads=8, d_ff=2048, n_layers=8,
        max_seq_len=256, dropout=0.0, pad_id=None,
    )
    ckpt = torch.load(F1_BASE, map_location="cpu", weights_only=False)
    m.load_state_dict(strip_compile_prefix(ckpt["model_state_dict"]))
    none_id = new_tok.encode_genre("none")
    _resize_embedding_with_init(m, old_tok.vocab_size, new_tok.vocab_size, none_id)
    return m.to(device).eval()


def _load_cell(method: str, genre: str, new_tok: ChordTokenizer, device):
    """Load Stage A trained cell, returning model in eval mode."""
    cell_dir = CKPT_ROOT / f"stageA_{method}_{genre}_funksize"
    if method in ("lora", "ia3"):
        if not (cell_dir / "adapter" / "adapter_config.json").exists():
            return None
        base = _build_f1(new_tok, torch.device("cpu"))
        model = PeftModel.from_pretrained(base, str(cell_dir / "adapter"))
        emb_ext = torch.load(cell_dir / "embedding_extension.pt",
                             map_location="cpu", weights_only=False)
        model.base_model.model.token_emb.load_state_dict(emb_ext["token_emb_state"])
        model.base_model.model.out_proj.load_state_dict(emb_ext["out_proj_state"])
        return model.to(device).eval()
    if method in ("bitfit", "full"):
        if not (cell_dir / "best.pt").exists():
            return None
        ckpt = torch.load(cell_dir / "best.pt", map_location="cpu", weights_only=False)
        m = MusicTransformer(
            vocab_size=new_tok.vocab_size,
            d_model=512, n_heads=8, d_ff=2048, n_layers=8,
            max_seq_len=256, dropout=0.0, pad_id=None,
        )
        m.load_state_dict(strip_compile_prefix(ckpt["model_state_dict"]))
        return m.to(device).eval()
    return None


def _eval_on_test(model, genre: str, new_tok: ChordTokenizer, device):
    """Forward-pass on ORIGINAL test set for the given genre."""
    ds = ChordSequenceDataset(
        split="test", max_seq_len=256, augment_keys=False,
        sources=[f"chordonomicon_{genre}"], tokenizer_version="v1",
        include_extra_genres=True,
    )
    if len(ds) == 0:
        return None
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
    pad_id = new_tok.pad_id

    total_tokens = 0
    correct_top1 = 0
    correct_top5 = 0
    loss_sum = 0.0
    crit = nn.CrossEntropyLoss(ignore_index=pad_id, reduction="sum")

    # Reloaded PEFT models carry task_type=FEATURE_EXTRACTION, whose forward
    # injects attention_mask that the custom MusicTransformer.forward(x) does not
    # accept. Call the inner LoRA/IA3-injected module directly (adapter still
    # applies, since it is swapped in at the layer level), bypassing that wrapper.
    fwd = model.base_model.model if isinstance(model, PeftModel) else model

    with torch.no_grad():
        for batch in loader:
            inp = batch["input_ids"].to(device)
            tgt = batch["target_ids"].to(device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = fwd(inp)
            mask = tgt != pad_id
            n = mask.sum().item()
            if n == 0:
                continue
            top1 = logits.argmax(dim=-1)
            correct_top1 += ((top1 == tgt) & mask).sum().item()
            _, top5 = logits.topk(5, dim=-1)
            correct_top5 += ((top5 == tgt.unsqueeze(-1)) & mask.unsqueeze(-1)).any(dim=-1).sum().item()
            loss_sum += crit(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1)).item()
            total_tokens += n

    if total_tokens == 0:
        return None
    return {
        "test_top1": round(100 * correct_top1 / total_tokens, 2),
        "test_top5": round(100 * correct_top5 / total_tokens, 2),
        "test_loss": round(loss_sum / total_tokens, 4),
        "n_test_tokens": total_tokens,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    new_tok = ChordTokenizer(include_extra_genres=True)

    print(f"Using device: {device}")
    rows = []
    for genre in GENRES_FUNK_SIZED:
        for method in METHODS:
            model = _load_cell(method, genre, new_tok, device)
            if model is None:
                print(f"  SKIP stageA_{method}_{genre}_funksize (checkpoint missing)")
                rows.append({
                    "cell": f"stageA_{method}_{genre}_funksize",
                    "genre": genre, "method": method,
                    "test_top1": None, "test_top5": None, "test_loss": None,
                    "n_test_tokens": 0,
                    "note": "checkpoint missing",
                })
                continue
            metrics = _eval_on_test(model, genre, new_tok, device)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if metrics is None:
                print(f"  SKIP stageA_{method}_{genre}_funksize (no test data)")
                continue
            row = {
                "cell": f"stageA_{method}_{genre}_funksize",
                "genre": genre, "method": method,
                **metrics,
            }
            rows.append(row)
            print(f"  stageA_{method}_{genre}_funksize: "
                  f"top1={metrics['test_top1']} top5={metrics['test_top5']} "
                  f"loss={metrics['test_loss']} (n={metrics['n_test_tokens']})")

    if rows:
        cols = sorted({k for r in rows for k in r.keys()})
        with OUT_CSV.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nWrote {OUT_CSV} ({len(rows)} cells)")


if __name__ == "__main__":
    main()
