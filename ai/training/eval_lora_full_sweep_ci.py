"""Re-evaluate every LoRA rank-sweep cell with token counts + Wilson 95% CIs.

Uses the same eval loop as `eval_lora_selected_split.py` (same `evaluate()`
shape, same `pad_id`, same vocab-resize path), but iterates over every
genre x rank cell instead of only the validation-selected rank.

Output:

- `ai/results/lora_rank_sweep_ci.csv` — per cell:
  genre, rank, split, sequences, tokens,
  f1_top1, lora_top1, delta_top1,
  lora_top1_ci_low, lora_top1_ci_high
- `ai/results/lora_rank_sweep_ci.md` — wide markdown table per genre.

Usage:
    cd ai
    UV_CACHE_DIR=/tmp/uv-cache uv run python training/eval_lora_full_sweep_ci.py --split val
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
from lora_train import _resize_embedding_with_init
from model import MusicTransformer
from tokenizer import ChordTokenizer
from _loader_helpers import strip_compile_prefix

PROJECT_DIR = Path(__file__).resolve().parent.parent
BASE_CKPT = PROJECT_DIR / "checkpoints" / "ft_jazz_pop80" / "best.pt"
CKPT_ROOT = PROJECT_DIR / "checkpoints"
RESULTS_DIR = PROJECT_DIR / "results"

GENRES = [
    "blues", "bossa", "classical", "country", "electronic",
    "folk", "funk", "gospel", "hip_hop", "rnb_soul", "rock",
]
RANKS = [4, 8, 16, 32, 64]


def _ckpt_dir_for(genre: str, rank: int) -> Path:
    suffix = "" if rank == 8 else f"_r{rank}"
    return CKPT_ROOT / f"ft_f1_lora_{genre}{suffix}"


def _wilson_interval(correct: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    phat = correct / total
    denom = 1 + z * z / total
    center = (phat + z * z / (2 * total)) / denom
    half = z * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total) / denom
    return max(0.0, center - half) * 100, min(1.0, center + half) * 100


def _build_base_model(vocab_size: int) -> MusicTransformer:
    ckpt = torch.load(BASE_CKPT, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    model = MusicTransformer(
        vocab_size=vocab_size,
        d_model=cfg.get("d_model", 512),
        n_heads=cfg.get("n_heads", 8),
        d_ff=cfg.get("d_ff", 2048),
        n_layers=cfg.get("n_layers", 8),
        max_seq_len=cfg.get("max_seq_len", 256),
        dropout=0.0,
        pad_id=None,
    )
    model.load_state_dict(strip_compile_prefix(ckpt["model_state_dict"]))
    return model


def _load_f1(old_tok: ChordTokenizer, new_tok: ChordTokenizer, device: torch.device) -> MusicTransformer:
    model = _build_base_model(old_tok.vocab_size)
    none_id = new_tok.encode_genre("none")
    assert none_id is not None
    _resize_embedding_with_init(model, old_tok.vocab_size, new_tok.vocab_size, none_id)
    return model.to(device).eval()


def _load_lora(
    genre: str,
    rank: int,
    old_tok: ChordTokenizer,
    new_tok: ChordTokenizer,
    device: torch.device,
) -> PeftModel:
    ckpt_dir = _ckpt_dir_for(genre, rank)
    if not (ckpt_dir / "adapter" / "adapter_config.json").exists():
        raise FileNotFoundError(f"Missing adapter for {genre} r={rank}: {ckpt_dir}")
    if not (ckpt_dir / "embedding_extension.pt").exists():
        raise FileNotFoundError(f"Missing embedding_extension.pt: {ckpt_dir}")

    base = _load_f1(old_tok, new_tok, torch.device("cpu"))
    model = PeftModel.from_pretrained(base, str(ckpt_dir / "adapter"))
    emb_ext = torch.load(ckpt_dir / "embedding_extension.pt", map_location="cpu", weights_only=False)
    model.base_model.model.token_emb.load_state_dict(emb_ext["token_emb_state"])
    model.base_model.model.out_proj.load_state_dict(emb_ext["out_proj_state"])
    return model.to(device).eval()


def _loader(split: str, genre: str, batch_size: int, num_workers: int) -> tuple[ChordSequenceDataset, DataLoader]:
    ds = ChordSequenceDataset(
        split=split,
        max_seq_len=256,
        augment_keys=False,
        sources=[f"chordonomicon_{genre}"],
        tokenizer_version="v1",
        include_extra_genres=True,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return ds, loader


def _evaluate_counts(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    vocab_size: int,
    pad_id: int,
    device: torch.device,
) -> dict:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    top1_correct = 0
    top5_correct = 0
    with torch.no_grad():
        for batch in loader:
            inp = batch["input_ids"].to(device)
            tgt = batch["target_ids"].to(device)
            if isinstance(model, PeftModel):
                logits = model.base_model.model(inp)
            else:
                logits = model(inp)
            flat_logits = logits.reshape(-1, vocab_size)
            flat_tgt = tgt.reshape(-1)
            loss = criterion(flat_logits, flat_tgt)
            mask = flat_tgt != pad_id
            valid = int(mask.sum().item())
            total_loss += float(loss.item())
            total_tokens += valid

            top1 = flat_logits.argmax(dim=-1)
            top1_correct += int(((top1 == flat_tgt) & mask).sum().item())

            top5 = flat_logits.topk(5, dim=-1).indices
            top5_hit = (top5 == flat_tgt.unsqueeze(-1)).any(dim=-1)
            top5_correct += int((top5_hit & mask).sum().item())

    loss = total_loss / max(1, total_tokens)
    top1 = top1_correct / max(1, total_tokens) * 100
    top5 = top5_correct / max(1, total_tokens) * 100
    ci_low, ci_high = _wilson_interval(top1_correct, total_tokens)
    return {
        "tokens": total_tokens,
        "loss": loss,
        "ppl": math.exp(min(loss, 10)),
        "top1": top1,
        "top5": top5,
        "top1_correct": top1_correct,
        "top5_correct": top5_correct,
        "top1_ci_low": ci_low,
        "top1_ci_high": ci_high,
    }


def _format_pp(x: float) -> str:
    return f"+{x:.2f}" if x >= 0 else f"{x:.2f}"


def _write_markdown(split: str, rows: list[dict], out_md: Path) -> None:
    by_genre: dict[str, dict[int, dict]] = {}
    for r in rows:
        by_genre.setdefault(r["genre"], {})[r["rank"]] = r

    lines = [
        f"# Full LoRA rank sweep with Wilson 95% CIs ({split})",
        "",
        f"55 cells (11 genres x 5 ranks) re-evaluated on the {split} split using the same",
        "`evaluate()` and `pad_id` path as `eval_lora_selected_split.py`. Top-1 CIs are",
        "Wilson 95% intervals over non-pad next-token targets. Cell format:",
        "`top1% [CI_low, CI_high] (delta vs F1)`.",
        "",
        "| Genre | Tokens | F1 top-1 | r=4 | r=8 | r=16 | r=32 | r=64 |",
        "|---|---:|---:|---|---|---|---|---|",
    ]
    for genre in GENRES:
        cells = by_genre.get(genre, {})
        if not cells:
            continue
        any_row = next(iter(cells.values()))
        f1_top1 = any_row["f1_top1"]
        tokens = any_row["tokens"]
        line = [
            genre,
            f"{tokens:,}",
            f"{f1_top1:.2f}",
        ]
        for r_val in RANKS:
            cell = cells.get(r_val)
            if cell is None:
                line.append("--")
                continue
            line.append(
                f"{cell['lora_top1']:.2f} "
                f"[{cell['lora_top1_ci_low']:.2f}, {cell['lora_top1_ci_high']:.2f}] "
                f"({_format_pp(cell['delta_top1'])})"
            )
        lines.append("| " + " | ".join(line) + " |")

    lines += [
        "",
        "**Reading the CIs.** A LoRA cell's gain over F1 is statistically supported when",
        "the cell's CI lies above the F1 baseline top-1. Cells whose CI overlaps the F1",
        "value should be treated as not significantly different at p<0.05.",
    ]
    out_md.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--genres", nargs="*", default=None,
        help="Optional subset of genres to evaluate (default: all 11).",
    )
    parser.add_argument(
        "--ranks", nargs="*", type=int, default=None,
        help="Optional subset of ranks to evaluate (default: 4 8 16 32 64).",
    )
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    genres = args.genres or GENRES
    ranks = args.ranks or RANKS

    old_tok = ChordTokenizer(include_extra_genres=False)
    new_tok = ChordTokenizer(include_extra_genres=True)
    criterion = nn.CrossEntropyLoss(ignore_index=old_tok.pad_id, reduction="sum")

    print("Loading F1 baseline once")
    f1 = _load_f1(old_tok, new_tok, device)

    rows: list[dict] = []
    f1_cache: dict[str, dict] = {}
    total_cells = len(genres) * len(ranks)
    cell_i = 0
    for g_i, genre in enumerate(genres, start=1):
        ds, loader = _loader(args.split, genre, args.batch_size, args.num_workers)
        print(f"[genre {g_i}/{len(genres)}] {genre} split={args.split} n={len(ds)}")

        if genre not in f1_cache:
            f1_cache[genre] = _evaluate_counts(
                f1, loader, criterion, new_tok.vocab_size, old_tok.pad_id, device,
            )
        f1_metrics = f1_cache[genre]

        for rank in ranks:
            cell_i += 1
            try:
                lora = _load_lora(genre, rank, old_tok, new_tok, device)
            except FileNotFoundError as exc:
                print(f"    [{cell_i}/{total_cells}] r={rank}: SKIP ({exc})")
                continue
            lora_metrics = _evaluate_counts(
                lora, loader, criterion, new_tok.vocab_size, old_tok.pad_id, device,
            )
            del lora
            if device.type == "cuda":
                torch.cuda.empty_cache()

            row = {
                "genre": genre,
                "rank": rank,
                "split": args.split,
                "sequences": len(ds),
                "tokens": lora_metrics["tokens"],
                "f1_loss": round(f1_metrics["loss"], 4),
                "f1_top1": round(f1_metrics["top1"], 2),
                "f1_top5": round(f1_metrics["top5"], 2),
                "lora_loss": round(lora_metrics["loss"], 4),
                "lora_top1": round(lora_metrics["top1"], 2),
                "lora_top5": round(lora_metrics["top5"], 2),
                "lora_top1_ci_low": round(lora_metrics["top1_ci_low"], 2),
                "lora_top1_ci_high": round(lora_metrics["top1_ci_high"], 2),
            }
            row["delta_top1"] = round(row["lora_top1"] - row["f1_top1"], 2)
            row["delta_top5"] = round(row["lora_top5"] - row["f1_top5"], 2)
            row["delta_loss"] = round(row["lora_loss"] - row["f1_loss"], 4)
            rows.append(row)
            print(
                f"    [{cell_i}/{total_cells}] r={rank}: F1 {row['f1_top1']:.2f} -> "
                f"LoRA {row['lora_top1']:.2f} ({_format_pp(row['delta_top1'])} pp), "
                f"CI [{row['lora_top1_ci_low']:.2f}, {row['lora_top1_ci_high']:.2f}]"
            )

    out_csv = RESULTS_DIR / "lora_rank_sweep_ci.csv"
    out_md = RESULTS_DIR / "lora_rank_sweep_ci.md"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "genre", "rank", "split", "sequences", "tokens",
        "f1_loss", "f1_top1", "f1_top5",
        "lora_loss", "lora_top1", "lora_top5",
        "delta_top1", "delta_top5", "delta_loss",
        "lora_top1_ci_low", "lora_top1_ci_high",
    ]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _write_markdown(args.split, rows, out_md)
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
