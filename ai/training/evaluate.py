"""Quantitative evaluation for Music Transformer checkpoints.

Usage:
    cd ai

    # Evaluate Phase A baseline
    uv run python training/evaluate.py --checkpoint checkpoints/phase_a_baseline/best.pt

    # Evaluate Phase B jazz (jazz-only test songs)
    uv run python training/evaluate.py --checkpoint checkpoints/phase_b_jazz/best.pt --sources jht wjazzd

    # Compare Phase A vs Phase B on jazz test data
    uv run python training/evaluate.py \
        --checkpoint checkpoints/phase_a_baseline/best.pt checkpoints/phase_b_jazz/best.pt \
        --sources jht wjazzd
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import ChordSequenceDataset
from model import MusicTransformer
from tokenizer import ChordTokenizer

PROJECT_DIR = Path(__file__).resolve().parent.parent


def compute_test_metrics(
    model: nn.Module,
    tokenizer: ChordTokenizer,
    test_loader: DataLoader,
    device: torch.device,
) -> dict:
    """Compute loss, perplexity, and token-level accuracy on test set."""
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id, reduction="sum")

    total_loss = 0.0
    total_tokens = 0
    correct_top1 = 0
    correct_top5 = 0
    correct_top10 = 0
    chord_correct_top1 = 0
    chord_correct_top5 = 0
    chord_total = 0

    # Chord token ID range
    special_max = tokenizer.BAR  # 0=PAD, 1=BOS, 2=EOS, 3=BAR
    meta_ids = set()
    for tok, tid in tokenizer.token2id.items():
        if tok.startswith("[KEY:") or tok.startswith("[TIME:") or tok.startswith("[GENRE:"):
            meta_ids.add(tid)

    with torch.no_grad():
        for batch in test_loader:
            inp = batch["input_ids"].to(device)
            tgt = batch["target_ids"].to(device)

            logits = model(inp)  # (B, L, V)
            B, L, V = logits.shape

            # Loss & perplexity
            loss = criterion(logits.reshape(-1, V), tgt.reshape(-1))
            non_pad = (tgt != tokenizer.pad_id).sum().item()
            total_loss += loss.item()
            total_tokens += non_pad

            # Top-k accuracy (all tokens)
            mask = tgt != tokenizer.pad_id  # (B, L)
            preds_top1 = logits.argmax(dim=-1)  # (B, L)
            _, preds_top5 = logits.topk(5, dim=-1)   # (B, L, 5)
            _, preds_top10 = logits.topk(10, dim=-1)  # (B, L, 10)

            correct_top1 += ((preds_top1 == tgt) & mask).sum().item()
            correct_top5 += ((preds_top5 == tgt.unsqueeze(-1)).any(dim=-1) & mask).sum().item()
            correct_top10 += ((preds_top10 == tgt.unsqueeze(-1)).any(dim=-1) & mask).sum().item()

            # Chord-only accuracy (exclude special/meta tokens)
            for b in range(B):
                for l in range(L):
                    if not mask[b, l]:
                        continue
                    target_id = tgt[b, l].item()
                    if target_id <= special_max or target_id in meta_ids:
                        continue
                    chord_total += 1
                    if preds_top1[b, l].item() == target_id:
                        chord_correct_top1 += 1
                    if target_id in preds_top5[b, l].tolist():
                        chord_correct_top5 += 1

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20))

    return {
        "test_loss": round(avg_loss, 4),
        "test_ppl": round(ppl, 2),
        "top1_acc": round(correct_top1 / max(total_tokens, 1) * 100, 2),
        "top5_acc": round(correct_top5 / max(total_tokens, 1) * 100, 2),
        "top10_acc": round(correct_top10 / max(total_tokens, 1) * 100, 2),
        "chord_top1_acc": round(chord_correct_top1 / max(chord_total, 1) * 100, 2),
        "chord_top5_acc": round(chord_correct_top5 / max(chord_total, 1) * 100, 2),
        "total_tokens": total_tokens,
        "chord_tokens": chord_total,
    }


def generate_continuation_eval(
    model: nn.Module,
    tokenizer: ChordTokenizer,
    test_songs: list[dict],
    device: torch.device,
    n_bars_context: int = 4,
    n_bars_predict: int = 4,
    max_songs: int = 100,
) -> dict:
    """Mask last N bars of test songs, generate, compare with ground truth.

    Measures how well the model predicts actual chord progressions
    given preceding context.
    """
    model.eval()
    total = 0
    exact_match = 0
    root_match = 0
    quality_match = 0
    bar_exact = 0
    bar_total = 0

    for song in test_songs[:max_songs]:
        ids = tokenizer.encode_sequence(song)
        # Find bar positions
        bar_positions = [i for i, t in enumerate(ids) if t == tokenizer.bar_id]
        if len(bar_positions) < n_bars_context + n_bars_predict:
            continue

        # Split: context = first bars, target = last n_bars_predict bars
        split_idx = bar_positions[-n_bars_predict]
        context_ids = ids[:split_idx]
        target_ids = ids[split_idx:]

        # Remove EOS from target for comparison
        target_tokens = [tokenizer.id2token.get(t, "") for t in target_ids if t != tokenizer.eos_id]

        # Generate
        prompt = torch.tensor([context_ids], device=device)
        max_gen = len(target_ids) + 8  # small buffer
        output = model.generate(
            prompt,
            max_new_tokens=max_gen,
            temperature=0.0001,  # near-greedy
            top_k=1,
            eos_id=tokenizer.eos_id,
        )
        gen_ids = output[0, len(context_ids):].tolist()
        # Remove EOS/PAD from generated
        gen_tokens = []
        for t in gen_ids:
            if t in (tokenizer.eos_id, tokenizer.pad_id):
                break
            gen_tokens.append(tokenizer.id2token.get(t, ""))

        # Compare chord by chord
        target_chords = [t for t in target_tokens if not t.startswith("[") and t not in ("", "[UNK]")]
        gen_chords = [t for t in gen_tokens if not t.startswith("[") and t not in ("", "[UNK]")]

        for tc, gc in zip(target_chords, gen_chords):
            total += 1
            if tc == gc:
                exact_match += 1
            # Root match (first 1-2 chars)
            tr = tc[0] + (tc[1] if len(tc) > 1 and tc[1] in ("b", "#") else "")
            gr = gc[0] + (gc[1] if len(gc) > 1 and gc[1] in ("b", "#") else "")
            if tr == gr:
                root_match += 1
            # Quality match
            tq = tc[len(tr):]
            gq = gc[len(gr):]
            if tq == gq:
                quality_match += 1

        # Bar-level exact match
        target_bars = _split_by_bar(target_tokens)
        gen_bars = _split_by_bar(gen_tokens)
        for tb, gb in zip(target_bars, gen_bars):
            bar_total += 1
            if tb == gb:
                bar_exact += 1

    return {
        "continuation_chord_acc": round(exact_match / max(total, 1) * 100, 2),
        "continuation_root_acc": round(root_match / max(total, 1) * 100, 2),
        "continuation_quality_acc": round(quality_match / max(total, 1) * 100, 2),
        "continuation_bar_acc": round(bar_exact / max(bar_total, 1) * 100, 2),
        "continuation_chords_evaluated": total,
        "continuation_bars_evaluated": bar_total,
    }


def _split_by_bar(tokens: list[str]) -> list[list[str]]:
    """Split token list into bars (groups between [BAR] tokens)."""
    bars: list[list[str]] = []
    current: list[str] = []
    for t in tokens:
        if t == "[BAR]":
            if current:
                bars.append(current)
            current = []
        elif not t.startswith("["):
            current.append(t)
    if current:
        bars.append(current)
    return bars


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[MusicTransformer, dict]:
    """Load model from checkpoint."""
    tokenizer = ChordTokenizer()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    model = MusicTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=config.get("d_model", 512),
        n_heads=config.get("n_heads", 8),
        d_ff=config.get("d_ff", 2048),
        n_layers=config.get("n_layers", 8),
        max_seq_len=config.get("max_seq_len", 512),
        dropout=0.0,  # no dropout at eval
        pad_id=tokenizer.pad_id,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    info = {
        "epoch": ckpt.get("epoch", "?"),
        "val_loss": ckpt.get("val_loss", "?"),
    }
    return model, info


def load_test_songs(sources: list[str] | None = None) -> list[dict]:
    """Load raw test songs (for continuation eval)."""
    test_file = PROJECT_DIR / "data" / "splits" / "test.jsonl"
    songs = []
    with open(test_file) as f:
        for line in f:
            song = json.loads(line)
            if sources and song.get("source") not in sources:
                continue
            songs.append(song)
    return songs


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Music Transformer")
    parser.add_argument(
        "--checkpoint", type=str, nargs="+", required=True,
        help="Path(s) to checkpoint file(s)",
    )
    parser.add_argument(
        "--sources", type=str, nargs="*", default=None,
        help="Filter test data by source (e.g. jht wjazzd)",
    )
    parser.add_argument(
        "--context-bars", type=int, default=4,
        help="Number of context bars for continuation eval",
    )
    parser.add_argument(
        "--predict-bars", type=int, default=4,
        help="Number of bars to predict in continuation eval",
    )
    parser.add_argument(
        "--max-songs", type=int, default=100,
        help="Max songs for continuation eval (slow due to autoregressive generation)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size for test loss computation",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = ChordTokenizer()
    sources = args.sources

    # Load test data
    test_ds = ChordSequenceDataset(
        split="test", max_seq_len=512, augment_keys=False, sources=sources,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True,
    )
    test_songs = load_test_songs(sources)

    src_label = ", ".join(sources) if sources else "all"
    print(f"Test data: {len(test_ds)} sequences, {len(test_songs)} songs (sources: {src_label})")
    print(f"Continuation eval: {args.context_bars} context bars → {args.predict_bars} predicted bars")
    print(f"{'=' * 70}\n")

    all_results = []

    for ckpt_path_str in args.checkpoint:
        ckpt_path = Path(ckpt_path_str)
        if not ckpt_path.is_absolute():
            ckpt_path = PROJECT_DIR / ckpt_path
        print(f"Checkpoint: {ckpt_path.name} ({ckpt_path.parent.name}/)")

        model, info = load_model(ckpt_path, device)
        print(f"  Trained epoch: {info['epoch']}, val_loss: {info['val_loss']}")

        # 1. Test loss & accuracy
        print("  Computing test metrics...", end=" ", flush=True)
        metrics = compute_test_metrics(model, tokenizer, test_loader, device)
        print("done")

        # 2. Continuation eval
        print(f"  Running continuation eval ({min(args.max_songs, len(test_songs))} songs)...", end=" ", flush=True)
        cont_metrics = generate_continuation_eval(
            model, tokenizer, test_songs, device,
            n_bars_context=args.context_bars,
            n_bars_predict=args.predict_bars,
            max_songs=args.max_songs,
        )
        print("done")

        result = {**metrics, **cont_metrics, "checkpoint": ckpt_path.name}
        all_results.append(result)

        print()
        print(f"  ┌─ Test Loss / Perplexity ─────────────────────────┐")
        print(f"  │  Loss: {metrics['test_loss']:<10} Perplexity: {metrics['test_ppl']:<10} │")
        print(f"  └─────────────────────────────────────────────────┘")
        print(f"  ┌─ Next-Token Accuracy (all tokens) ──────────────┐")
        print(f"  │  Top-1: {metrics['top1_acc']:>6}%   Top-5: {metrics['top5_acc']:>6}%   Top-10: {metrics['top10_acc']:>6}% │")
        print(f"  │  ({metrics['total_tokens']} tokens evaluated)                    │")
        print(f"  └─────────────────────────────────────────────────┘")
        print(f"  ┌─ Chord-Only Accuracy ───────────────────────────┐")
        print(f"  │  Top-1: {metrics['chord_top1_acc']:>6}%   Top-5: {metrics['chord_top5_acc']:>6}%               │")
        print(f"  │  ({metrics['chord_tokens']} chord tokens evaluated)              │")
        print(f"  └─────────────────────────────────────────────────┘")
        print(f"  ┌─ Continuation Generation ───────────────────────┐")
        print(f"  │  Chord exact: {cont_metrics['continuation_chord_acc']:>6}%  Root: {cont_metrics['continuation_root_acc']:>6}%       │")
        print(f"  │  Quality:     {cont_metrics['continuation_quality_acc']:>6}%  Bar:  {cont_metrics['continuation_bar_acc']:>6}%       │")
        print(f"  │  ({cont_metrics['continuation_chords_evaluated']} chords, {cont_metrics['continuation_bars_evaluated']} bars evaluated)          │")
        print(f"  └─────────────────────────────────────────────────┘")
        print()

        del model
        torch.cuda.empty_cache()

    # Comparison table (if multiple checkpoints)
    if len(all_results) > 1:
        print(f"{'=' * 70}")
        print("COMPARISON")
        print(f"{'=' * 70}")
        header = f"{'Metric':<28}"
        for r in all_results:
            header += f" {r['checkpoint']:>18}"
        print(header)
        print("-" * len(header))
        for key in ["test_loss", "test_ppl", "top1_acc", "top5_acc",
                     "chord_top1_acc", "chord_top5_acc",
                     "continuation_chord_acc", "continuation_root_acc",
                     "continuation_bar_acc"]:
            row = f"{key:<28}"
            for r in all_results:
                val = r[key]
                suffix = "%" if "acc" in key else ""
                row += f" {val:>17}{suffix}"
            print(row)
        print()


if __name__ == "__main__":
    main()
