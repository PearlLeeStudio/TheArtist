"""Combine seed=42 (existing rank-sweep) + seed=123 + seed=7 (multi-seed)
into a single per-genre 3-seed table.

For each (genre, chosen rank) cell we want:
  - 3 seed top-1 values (one per seed)
  - mean ± std
  - paired delta vs F1 (mean_lora - f1)
  - simple Welch-style 95% CI on the mean

Output:
  ai/results/lora_multi_seed.csv         — per (genre, seed) row + summary
  ai/results/lora_multi_seed.md          — markdown summary for paper §7.x

Usage:
  cd ai
  python training/aggregate_multi_seed.py
"""

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path

import torch
import torch.nn as nn
from peft import PeftModel
from torch.utils.data import DataLoader

from dataset import ChordSequenceDataset
from lora_train import _resize_embedding_with_init
from model import MusicTransformer
from tokenizer import ChordTokenizer

PROJECT_DIR = Path(__file__).resolve().parent.parent
LORA_TEST_CSV = PROJECT_DIR / "results" / "lora_selected_test.csv"
RESULTS_DIR = PROJECT_DIR / "results"
BASE_CKPT = PROJECT_DIR / "checkpoints" / "ft_jazz_pop80" / "best.pt"
CKPT_ROOT = PROJECT_DIR / "checkpoints"

GENRES = [
    "blues", "bossa", "classical", "country", "electronic",
    "folk", "funk", "gospel", "hip_hop", "rnb_soul", "rock",
]
SEEDS = [42, 123, 7]


def _ckpt_dir_for(genre: str, rank: int, seed: int) -> Path:
    """seed=42 = existing checkpoint (no _seed suffix);
    seed=123/7 = new multi-seed checkpoint with _seed<N> suffix."""
    suffix_rank = "" if rank == 8 else f"_r{rank}"
    if seed == 42:
        return CKPT_ROOT / f"ft_f1_lora_{genre}{suffix_rank}"
    return CKPT_ROOT / f"ft_f1_lora_{genre}{suffix_rank}_seed{seed}"


def _selected_ranks() -> dict[str, int]:
    out = {}
    with open(LORA_TEST_CSV) as f:
        for r in csv.DictReader(f):
            out[r["genre"]] = int(r["rank"])
    return out


def _build_base_model(vocab_size: int) -> MusicTransformer:
    ckpt = torch.load(BASE_CKPT, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    m = MusicTransformer(
        vocab_size=vocab_size,
        d_model=cfg.get("d_model", 512),
        n_heads=cfg.get("n_heads", 8),
        d_ff=cfg.get("d_ff", 2048),
        n_layers=cfg.get("n_layers", 8),
        max_seq_len=cfg.get("max_seq_len", 256),
        dropout=0.0,
        pad_id=None,
    )
    m.load_state_dict(ckpt["model_state_dict"])
    return m


def _load_f1(old_tok: ChordTokenizer, new_tok: ChordTokenizer, device: torch.device) -> MusicTransformer:
    m = _build_base_model(old_tok.vocab_size)
    none_id = new_tok.encode_genre("none")
    assert none_id is not None
    _resize_embedding_with_init(m, old_tok.vocab_size, new_tok.vocab_size, none_id)
    return m.to(device).eval()


def _load_lora(genre: str, rank: int, seed: int, old_tok, new_tok, device) -> PeftModel:
    base = _load_f1(old_tok, new_tok, torch.device("cpu"))
    ckpt_dir = _ckpt_dir_for(genre, rank, seed)
    model = PeftModel.from_pretrained(base, str(ckpt_dir / "adapter"))
    emb_ext = torch.load(ckpt_dir / "embedding_extension.pt", map_location="cpu", weights_only=False)
    model.base_model.model.token_emb.load_state_dict(emb_ext["token_emb_state"])
    model.base_model.model.out_proj.load_state_dict(emb_ext["out_proj_state"])
    return model.to(device).eval()


def _loader(split: str, genre: str) -> tuple[ChordSequenceDataset, DataLoader]:
    ds = ChordSequenceDataset(
        split=split,
        max_seq_len=256,
        augment_keys=False,
        sources=[f"chordonomicon_{genre}"],
        tokenizer_version="v1",
        include_extra_genres=True,
    )
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=False)
    return ds, loader


def _eval_top1(model, loader, vocab_size, pad_id, device) -> float:
    model.eval()
    top1_correct = 0
    total_tokens = 0
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
            mask = flat_tgt != pad_id
            top1 = flat_logits.argmax(dim=-1)
            top1_correct += int(((top1 == flat_tgt) & mask).sum().item())
            total_tokens += int(mask.sum().item())
    return top1_correct / max(1, total_tokens) * 100


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    selected = _selected_ranks()
    old_tok = ChordTokenizer(include_extra_genres=False)
    new_tok = ChordTokenizer(include_extra_genres=True)

    print("Loading F1 baseline once")
    f1 = _load_f1(old_tok, new_tok, device)

    rows: list[dict] = []
    summary_rows: list[dict] = []

    for genre in GENRES:
        rank = selected[genre]
        ds, loader = _loader("test", genre)
        f1_top1 = _eval_top1(f1, loader, new_tok.vocab_size, old_tok.pad_id, device)
        print(f"[{genre}] r={rank} test n={len(ds)} F1 top-1={f1_top1:.2f}")
        seed_top1: list[float] = []
        for seed in SEEDS:
            ckpt_dir = _ckpt_dir_for(genre, rank, seed)
            if not (ckpt_dir / "adapter" / "adapter_config.json").exists():
                print(f"  seed={seed}: SKIP (no adapter at {ckpt_dir})")
                continue
            lora = _load_lora(genre, rank, seed, old_tok, new_tok, device)
            top1 = _eval_top1(lora, loader, new_tok.vocab_size, old_tok.pad_id, device)
            del lora
            if device.type == "cuda":
                torch.cuda.empty_cache()
            seed_top1.append(top1)
            rows.append({
                "genre": genre, "rank": rank, "seed": seed,
                "f1_top1": round(f1_top1, 2),
                "lora_top1": round(top1, 2),
                "delta_top1": round(top1 - f1_top1, 2),
            })
            print(f"  seed={seed}: LoRA top-1={top1:.2f} (Δ {top1 - f1_top1:+.2f})")

        if not seed_top1:
            continue
        mean = statistics.mean(seed_top1)
        sd = statistics.pstdev(seed_top1) if len(seed_top1) > 1 else 0.0
        n = len(seed_top1)
        # 95% CI on mean using t-approx for tiny n. With n=3, t_{0.975, 2} = 4.303.
        if n == 3:
            t_crit = 4.303
        elif n == 2:
            t_crit = 12.706
        else:
            t_crit = 1.96
        ci_half = t_crit * sd / math.sqrt(n) if n > 1 else 0.0
        summary_rows.append({
            "genre": genre, "rank": rank, "n_seeds": n,
            "f1_top1": round(f1_top1, 2),
            "lora_mean": round(mean, 2),
            "lora_std": round(sd, 2),
            "lora_ci_low": round(mean - ci_half, 2),
            "lora_ci_high": round(mean + ci_half, 2),
            "mean_delta": round(mean - f1_top1, 2),
            "min_delta": round(min(seed_top1) - f1_top1, 2),
            "max_delta": round(max(seed_top1) - f1_top1, 2),
        })

    # Write CSVs
    if rows:
        out = RESULTS_DIR / "lora_multi_seed.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {out}")

    if summary_rows:
        out = RESULTS_DIR / "lora_multi_seed_summary.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
        print(f"Wrote {out}")

    # Markdown
    if summary_rows:
        lines = [
            "# Multi-seed LoRA reliability (chosen rank x 3 seeds, held-out test)",
            "",
            f"For each genre's validation-selected rank, three seeds {SEEDS} were trained",
            "with otherwise-identical configs and evaluated on the held-out test split.",
            "Δ is mean LoRA test top-1 minus F1 test top-1; the [min, max] columns",
            "report the smallest and largest single-seed Δ across the three seeds.",
            "The 95% CI uses Student t with n=3 (t_{0.975, 2} = 4.303); when fewer",
            "than 3 adapters are available the CI widens accordingly.",
            "",
            "| Genre | Rank | n seeds | F1 top-1 | LoRA top-1 mean ± std | Δ (mean) | Δ [min, max] | LoRA 95% CI |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in summary_rows:
            lines.append(
                f"| {r['genre']} | {r['rank']} | {r['n_seeds']} | {r['f1_top1']:.2f} | "
                f"{r['lora_mean']:.2f} ± {r['lora_std']:.2f} | "
                f"{r['mean_delta']:+.2f} | "
                f"[{r['min_delta']:+.2f}, {r['max_delta']:+.2f}] | "
                f"[{r['lora_ci_low']:.2f}, {r['lora_ci_high']:.2f}] |"
            )
        # Aggregate over the 11 genres
        deltas = [r["mean_delta"] for r in summary_rows]
        non_chorale = [r["mean_delta"] for r in summary_rows if r["genre"] != "classical"]
        lines += [
            "",
            f"Across {len(summary_rows)} genres: mean Δ = {statistics.mean(deltas):+.2f}pp, "
            f"median Δ = {statistics.median(deltas):+.2f}pp, "
            f"trimmed (excl. Bach chorale) mean Δ = {statistics.mean(non_chorale):+.2f}pp.",
        ]
        out = RESULTS_DIR / "lora_multi_seed.md"
        out.write_text("\n".join(lines) + "\n")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
