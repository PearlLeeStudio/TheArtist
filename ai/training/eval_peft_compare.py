"""Evaluate the PEFT-comparison sweep on the held-out test split.

For each (genre, method) cell, load the trained checkpoint and report
test top-1 / top-5 + trainable-parameter count + Wilson 95% CI.

Conditions:
  - LoRA          — chosen-rank LoRA (already trained, reuse adapter)
  - IA3           — ai/checkpoints/ft_f1_ia3_<genre>/ (new)
  - BitFit        — ai/checkpoints/ft_f1_bitfit_<genre>/best.pt (new, full state)
  - PrefixTuning  — ai/checkpoints/ft_f1_prefix_<genre>/ (new, custom impl)
  - Full FT       — ai/checkpoints/ft_f1_full_ft_<genre>/best.pt
                    (existing for classical / rock; new for country)

Output:
  ai/results/peft_compare.csv
  ai/results/peft_compare.md
"""

from __future__ import annotations

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
from prefix_tuning import PrefixTuningWrapper
from tokenizer import ChordTokenizer

PROJECT_DIR = Path(__file__).resolve().parent.parent
BASE_CKPT = PROJECT_DIR / "checkpoints" / "ft_jazz_pop80" / "best.pt"
CKPT_ROOT = PROJECT_DIR / "checkpoints"
LORA_TEST_CSV = PROJECT_DIR / "results" / "lora_selected_test.csv"
RESULTS_DIR = PROJECT_DIR / "results"

GENRES_3 = ["classical", "country", "rock"]
GENRES_11 = [
    "blues", "bossa", "classical", "country", "electronic",
    "folk", "funk", "gospel", "hip_hop", "rnb_soul", "rock",
]
METHODS = ["lora", "ia3", "bitfit", "prefix_tuning", "full"]


def _ckpt_dir(genre: str, method: str, lora_rank: int | None, seed: int = 42) -> Path:
    # Naming inconsistency: seed=42 cells (A2/A3/B3_lora_seed42) have no
    # seed suffix; seed=123/7 cells (B2_*/B3_lora_multi_seed) have _seed{N}.
    # Additionally, LoRA cells have a rank suffix that differs by genre:
    #   - electronic: NEVER has _r suffix (both seed42 and multi-seed
    #     are at `ft_f1_lora_electronic[_seed{N}]`, confirmed against
    #     chain's lora_5ep_dirs and multi_seed/electronic_r8_seed*.yaml's
    #     run_name = ft_f1_lora_electronic_seed{N}, no _r8).
    #   - other genres at seed=42 with default r=8: no _r suffix
    #   - other genres at seed=42 with non-default rank: _r{rank}
    #   - other genres at seed=123/7: _r{rank} always present
    seed_suffix = "" if seed == 42 else f"_seed{seed}"
    if method == "lora":
        if genre == "electronic":
            rank_suffix = ""  # electronic naming convention: never has _r suffix
        elif seed == 42:
            rank_suffix = "" if lora_rank == 8 else f"_r{lora_rank}"
        else:
            rank_suffix = f"_r{lora_rank}"
        return CKPT_ROOT / f"ft_f1_lora_{genre}{rank_suffix}{seed_suffix}"
    if method == "ia3":
        return CKPT_ROOT / f"ft_f1_ia3_{genre}{seed_suffix}"
    if method == "bitfit":
        return CKPT_ROOT / f"ft_f1_bitfit_{genre}{seed_suffix}"
    if method == "prefix_tuning":
        return CKPT_ROOT / f"ft_f1_prefix_{genre}{seed_suffix}"
    if method == "full":
        return CKPT_ROOT / f"ft_f1_full_ft_{genre}{seed_suffix}"
    raise ValueError(method)


def _selected_rank(genre: str) -> int:
    with open(LORA_TEST_CSV) as f:
        for r in csv.DictReader(f):
            if r["genre"] == genre:
                return int(r["rank"])
    raise KeyError(genre)


def _wilson_interval(correct: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    z = 1.959963984540054
    phat = correct / total
    denom = 1 + z * z / total
    center = (phat + z * z / (2 * total)) / denom
    half = z * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total) / denom
    return max(0.0, center - half) * 100, min(1.0, center + half) * 100


def _strip_compile_prefix(sd: dict) -> dict:
    # Checkpoints saved while wrapped in torch.compile carry an _orig_mod.
    # prefix on every key. Strip it so plain MusicTransformer.load_state_dict
    # matches. Idempotent: keys without the prefix pass through unchanged.
    if not any(k.startswith("_orig_mod.") for k in sd):
        return sd
    return {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in sd.items()}


def _build_base(vocab_size: int) -> MusicTransformer:
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
    m.load_state_dict(_strip_compile_prefix(ckpt["model_state_dict"]))
    return m


def _load_f1(old_tok: ChordTokenizer, new_tok: ChordTokenizer, device: torch.device) -> MusicTransformer:
    m = _build_base(old_tok.vocab_size)
    none_id = new_tok.encode_genre("none")
    assert none_id is not None
    _resize_embedding_with_init(m, old_tok.vocab_size, new_tok.vocab_size, none_id)
    return m.to(device).eval()


def _load_for_method(
    genre: str,
    method: str,
    lora_rank: int,
    new_tok: ChordTokenizer,
    old_tok: ChordTokenizer,
    device: torch.device,
    seed: int = 42,
):
    if method in ("lora", "ia3"):
        ckpt_dir = _ckpt_dir(genre, method, lora_rank, seed=seed)
        base = _load_f1(old_tok, new_tok, torch.device("cpu"))
        model = PeftModel.from_pretrained(base, str(ckpt_dir / "adapter"))
        emb_ext = torch.load(ckpt_dir / "embedding_extension.pt", map_location="cpu", weights_only=False)
        model.base_model.model.token_emb.load_state_dict(emb_ext["token_emb_state"])
        model.base_model.model.out_proj.load_state_dict(emb_ext["out_proj_state"])
        return model.to(device).eval()
    if method in ("bitfit", "full"):
        ckpt_dir = _ckpt_dir(genre, method, lora_rank, seed=seed)
        ckpt = torch.load(ckpt_dir / "best.pt", map_location="cpu", weights_only=False)
        sd = _strip_compile_prefix(ckpt["model_state_dict"])
        # Vocab size in the saved full-state model is the *extended* vocab
        # (lora_train.py resizes before saving), so build with new_tok size.
        m = MusicTransformer(
            vocab_size=new_tok.vocab_size,
            d_model=512, n_heads=8, d_ff=2048, n_layers=8,
            max_seq_len=256, dropout=0.0, pad_id=None,
        )
        m.load_state_dict(sd)
        return m.to(device).eval()
    if method == "prefix_tuning":
        ckpt_dir = _ckpt_dir(genre, method, lora_rank, seed=seed)
        # 1) build & resize base in the same way as _load_f1
        base = _load_f1(old_tok, new_tok, torch.device("cpu"))
        # 2) wrap with PrefixTuningWrapper at the saved prefix_len
        pref_ckpt = torch.load(ckpt_dir / "prefix_state.pt", map_location="cpu", weights_only=False)
        prefix_len = pref_ckpt.get("prefix_len", 20)
        wrapper = PrefixTuningWrapper(base, prefix_len=prefix_len)
        wrapper.load_prefix_state_dict(pref_ckpt["prefix_state"])
        # 3) overwrite extended embedding rows with what was saved
        emb_ext = torch.load(ckpt_dir / "embedding_extension.pt", map_location="cpu", weights_only=False)
        wrapper.model.token_emb.load_state_dict(emb_ext["token_emb_state"])
        wrapper.model.out_proj.load_state_dict(emb_ext["out_proj_state"])
        return wrapper.to(device).eval()
    raise ValueError(method)


def _trainable_count(model: torch.nn.Module, method: str) -> tuple[int, int]:
    # For peft-wrapped models, requires_grad is the right filter even at
    # eval time. For freshly-loaded full-state models we synthetically
    # mark trainable per method definition.
    if method == "lora":
        # Mirror the lora_train.py trainable set: lora_* params + token_emb + out_proj.
        trainable = sum(
            p.numel()
            for n, p in model.named_parameters()
            if ("lora_" in n) or ("token_emb" in n) or ("out_proj" in n)
        )
    elif method == "ia3":
        trainable = sum(
            p.numel()
            for n, p in model.named_parameters()
            if ("ia3_" in n) or ("token_emb" in n) or ("out_proj" in n)
        )
    elif method == "bitfit":
        trainable = sum(
            p.numel()
            for n, p in model.named_parameters()
            if n.endswith(".bias") or ("token_emb" in n) or ("out_proj" in n)
        )
    elif method == "prefix_tuning":
        trainable = sum(
            p.numel()
            for n, p in model.named_parameters()
            if ("prefix_encoder" in n) or ("token_emb" in n) or ("out_proj" in n)
        )
    else:  # full
        trainable = sum(p.numel() for p in model.parameters())
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def _loader(genre: str) -> tuple[ChordSequenceDataset, DataLoader]:
    ds = ChordSequenceDataset(
        split="test",
        max_seq_len=256,
        augment_keys=False,
        sources=[f"chordonomicon_{genre}"],
        tokenizer_version="v1",
        include_extra_genres=True,
    )
    return ds, DataLoader(ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=False)


def _eval_counts(model, loader, vocab_size, pad_id, device) -> dict:
    model.eval()
    top1_correct = 0
    top5_correct = 0
    total_tokens = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, reduction="sum")
    with torch.no_grad():
        for batch in loader:
            inp = batch["input_ids"].to(device)
            tgt = batch["target_ids"].to(device)
            if isinstance(model, PeftModel):
                logits = model.base_model.model(inp)
            else:
                logits = model(inp)
            fl = logits.reshape(-1, vocab_size)
            ft = tgt.reshape(-1)
            mask = ft != pad_id
            total_loss += float(criterion(fl, ft).item())
            total_tokens += int(mask.sum().item())
            top1 = fl.argmax(dim=-1)
            top1_correct += int(((top1 == ft) & mask).sum().item())
            top5 = fl.topk(5, dim=-1).indices
            top5_hit = (top5 == ft.unsqueeze(-1)).any(dim=-1)
            top5_correct += int((top5_hit & mask).sum().item())
    ci = _wilson_interval(top1_correct, total_tokens)
    return {
        "tokens": total_tokens,
        "loss": total_loss / max(1, total_tokens),
        "top1": top1_correct / max(1, total_tokens) * 100,
        "top5": top5_correct / max(1, total_tokens) * 100,
        "top1_ci_low": ci[0],
        "top1_ci_high": ci[1],
    }


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--all-genres", action="store_true",
        help="Evaluate all 11 LoRA target genres (default: 3 representative).",
    )
    args = ap.parse_args()
    genres = GENRES_11 if args.all_genres else GENRES_3
    out_suffix = "_all" if args.all_genres else ""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; genres: {genres}")

    old_tok = ChordTokenizer(include_extra_genres=False)
    new_tok = ChordTokenizer(include_extra_genres=True)
    f1 = _load_f1(old_tok, new_tok, device)

    # Multi-seed iteration: when --all-genres, eval across {42, 123, 7} so the
    # output CSV has 5 methods × 11 genres × 3 seeds = 165 rows + 11 F1
    # baseline rows (F1 baseline doesn't depend on seed). For default
    # (3-genre) mode, keep single-seed eval for backward compatibility.
    seeds = [42, 123, 7] if args.all_genres else [42]

    rows: list[dict] = []
    for genre in genres:
        lora_rank = _selected_rank(genre)
        ds, loader = _loader(genre)
        print(f"[{genre}] test n={len(ds)} lora_rank={lora_rank}")
        f1_metrics = _eval_counts(f1, loader, new_tok.vocab_size, old_tok.pad_id, device)
        # F1 baseline is seed-independent — emit once per genre with seed=None
        rows.append({
            "genre": genre, "method": "F1 (no adapt)", "seed": "",
            "rank_or_param": "",
            "trainable_params": 0, "total_params": sum(p.numel() for p in f1.parameters()),
            "test_top1": round(f1_metrics["top1"], 2),
            "test_top5": round(f1_metrics["top5"], 2),
            "test_loss": round(f1_metrics["loss"], 4),
            "delta_top1": 0.0,
            "ci_low": round(f1_metrics["top1_ci_low"], 2),
            "ci_high": round(f1_metrics["top1_ci_high"], 2),
        })
        print(f"  F1 only: top-1 {f1_metrics['top1']:.2f}")

        for method in METHODS:
            for seed in seeds:
                ckpt_path = _ckpt_dir(genre, method, lora_rank, seed=seed)
                if method in {"full", "bitfit"}:
                    ok = (ckpt_path / "best.pt").exists()
                elif method == "prefix_tuning":
                    ok = (ckpt_path / "prefix_state.pt").exists() \
                        and (ckpt_path / "embedding_extension.pt").exists()
                else:
                    ok = (ckpt_path / "adapter" / "adapter_config.json").exists()
                if not ok:
                    print(f"  {method} seed={seed}: SKIP (no checkpoint at {ckpt_path})")
                    continue
                try:
                    model = _load_for_method(genre, method, lora_rank, new_tok, old_tok, device, seed=seed)
                except Exception as exc:
                    print(f"  {method} seed={seed}: SKIP (load failed: {exc})")
                    continue
                trainable, total = _trainable_count(model, method)
                m = _eval_counts(model, loader, new_tok.vocab_size, old_tok.pad_id, device)
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                delta = m["top1"] - f1_metrics["top1"]
                rank_or_param = (f"r={lora_rank}" if method == "lora"
                                 else f"k/v scale" if method == "ia3"
                                 else "bias only" if method == "bitfit"
                                 else "prefix_len=20" if method == "prefix_tuning"
                                 else "all params")
                rows.append({
                    "genre": genre, "method": method, "seed": seed,
                    "rank_or_param": rank_or_param,
                    "trainable_params": trainable, "total_params": total,
                    "test_top1": round(m["top1"], 2),
                    "test_top5": round(m["top5"], 2),
                    "test_loss": round(m["loss"], 4),
                    "delta_top1": round(delta, 2),
                    "ci_low": round(m["top1_ci_low"], 2),
                    "ci_high": round(m["top1_ci_high"], 2),
                })
                print(
                    f"  {method} seed={seed}: top-1 {m['top1']:.2f} (Δ {delta:+.2f}, CI "
                    f"[{m['top1_ci_low']:.2f}, {m['top1_ci_high']:.2f}]), "
                    f"trainable {trainable:,} ({100*trainable/total:.2f}%)"
                )

    out_csv = RESULTS_DIR / f"peft_compare{out_suffix}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_csv}")

    # Markdown wide table per genre
    lines = [
        "# PEFT comparison (held-out test split)",
        "",
        "For each of three representative genres (Bach chorale OOD, country mid-gain,",
        "rock small-gain), five PEFT methods are compared against the F1 base on the",
        "held-out test split: LoRA, IA3, BitFit, custom prefix-tuning (Li & Liang 2021,",
        "20 virtual tokens), and full fine-tune. Trainable parameters include each",
        "method's freshly-added components plus the new genre embedding rows",
        "(token_emb + out_proj), which all methods inherit so the new-genre marker is",
        "learnable.",
        "",
        "| Genre | Method | Setting | Trainable params | % of total | Test top-1 | Δ vs F1 | 95% CI |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        if r["method"] == "F1 (no adapt)":
            tp = "—"
            pct = "—"
            delta = "—"
        else:
            tp = f"{r['trainable_params']:,}"
            pct = f"{100*r['trainable_params']/r['total_params']:.2f}%"
            delta = f"{r['delta_top1']:+.2f}"
        lines.append(
            f"| {r['genre']} | {r['method']} | {r['rank_or_param']} | "
            f"{tp} | {pct} | {r['test_top1']:.2f} | {delta} | "
            f"[{r['ci_low']:.2f}, {r['ci_high']:.2f}] |"
        )
    out_md = RESULTS_DIR / f"peft_compare{out_suffix}.md"
    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
