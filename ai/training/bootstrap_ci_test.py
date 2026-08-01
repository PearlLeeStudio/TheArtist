"""D6: Bootstrap 95% CI on the test set for every (method, genre, seed) cell.

For each of the 165 cells (5 methods × 11 genres × 3 seeds), load the
checkpoint, run a single test-set forward pass to collect per-token
correctness (top-1 hit/miss), then bootstrap-resample the per-token
correctness 1000 times to get a 95% percentile CI on top-1 accuracy.

Reuses helpers from eval_peft_compare.py for model loading + data
loaders. Forward pass is batched on GPU, but each cell is processed
sequentially (single-GPU consumer).

Output:
    results/bootstrap_ci_test.csv  — one row per cell with point top-1,
                                     CI lower / upper, n_test_samples
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel

import eval_peft_compare as epc
from tokenizer import ChordTokenizer

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_CSV = PROJECT_DIR / "results" / "bootstrap_ci_test.csv"
N_BOOTSTRAP = 1000
SEEDS = [42, 123, 7]


def _ckpt_dir_seed(genre: str, method: str, lora_rank: int | None, seed: int) -> Path:
    base = epc._ckpt_dir(genre, method, lora_rank)
    if seed == 42:
        return base
    return base.parent / f"{base.name}_seed{seed}"


def per_token_correct(model, loader, vocab_size, pad_id, device) -> np.ndarray:
    """Return a 1D array of 0/1 indicating per-token top-1 correctness."""
    model.eval()
    flags: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            inp = batch["input_ids"].to(device)
            tgt = batch["target_ids"].to(device)
            logits = model.base_model.model(inp) if isinstance(model, PeftModel) else model(inp)
            fl = logits.reshape(-1, vocab_size)
            ft = tgt.reshape(-1)
            mask = ft != pad_id
            top1 = fl.argmax(dim=-1)
            hit = ((top1 == ft) & mask).cpu().numpy().astype(np.int8)
            keep = mask.cpu().numpy().astype(bool)
            flags.append(hit[keep])
    return np.concatenate(flags) if flags else np.zeros(0, dtype=np.int8)


def bootstrap_ci(correct: np.ndarray, n_boot: int = N_BOOTSTRAP, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    if len(correct) == 0:
        return (0.0, 0.0, 0.0)
    boot_means = np.empty(n_boot)
    n = len(correct)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = correct[idx].mean()
    point = float(correct.mean()) * 100
    lo = float(np.percentile(boot_means, 2.5)) * 100
    hi = float(np.percentile(boot_means, 97.5)) * 100
    return (point, lo, hi)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; n_bootstrap={N_BOOTSTRAP}")

    old_tok = ChordTokenizer(include_extra_genres=False)
    new_tok = ChordTokenizer(include_extra_genres=True)

    rows: list[dict] = []
    for genre in epc.GENRES_11:
        try:
            lora_rank = epc._selected_rank(genre)
        except KeyError:
            print(f"[{genre}] no selected LoRA rank, skipping LoRA cells")
            lora_rank = None
        ds, loader = epc._loader(genre)
        print(f"[{genre}] test tokens loader ready (n_seq={len(ds)})")

        for method in epc.METHODS:
            for seed in SEEDS:
                ckpt = _ckpt_dir_seed(genre, method, lora_rank, seed)
                if method in {"full", "bitfit"}:
                    if not (ckpt / "best.pt").exists():
                        continue
                elif method == "prefix_tuning":
                    if not ((ckpt / "prefix_state.pt").exists()
                            and (ckpt / "embedding_extension.pt").exists()):
                        continue
                else:  # lora, ia3
                    if not (ckpt / "adapter" / "adapter_config.json").exists():
                        continue
                try:
                    model = epc._load_for_method(genre, method, lora_rank, new_tok, old_tok, device)
                except Exception as exc:
                    print(f"  {method}/{genre}/seed{seed}: SKIP load ({exc})")
                    continue
                correct = per_token_correct(model, loader, new_tok.vocab_size, old_tok.pad_id, device)
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                point, lo, hi = bootstrap_ci(correct, seed=seed)
                rows.append({
                    "method": method, "genre": genre, "seed": seed,
                    "n_test_tokens": int(len(correct)),
                    "test_top1": round(point, 3),
                    "ci_low": round(lo, 3), "ci_high": round(hi, 3),
                    "ci_width": round(hi - lo, 3),
                })
                print(f"  {method}/{genre}/seed{seed}: top1={point:.2f} CI=[{lo:.2f}, {hi:.2f}]")

    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "genre", "seed", "n_test_tokens",
                                          "test_top1", "ci_low", "ci_high", "ci_width"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
