"""Smoke test for torch.compile compatibility with the custom
MusicTransformer + PeftModel wrap on RTX 4070 Laptop.

Builds F1 base + a tiny LoRA adapter, wraps with torch.compile, runs
a handful of forward + backward passes, and reports timing.

Verdict logic:
  - PASS:  compile succeeds + 5+ forward/backward steps complete + per-step
           time is at or below the eager baseline (no per-shape recompile blow-up).
  - FAIL:  any exception during compile or first forward, OR per-step time
           is >2x baseline (recompilation thrashing).

Usage:
    cd ai
    UV_CACHE_DIR=/tmp/uv-cache uv run python training/smoke_test_torch_compile.py

Exits 0 on PASS, 1 on FAIL. Intended to gate the Phase 2 launch decision
about whether to enable torch.compile globally in lora_train.py.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

# cudnn.benchmark already in lora_train.py; mirror here for parity
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "training"))

from model import MusicTransformer  # noqa: E402
from tokenizer import ChordTokenizer  # noqa: E402
from lora_train import _resize_embedding_with_init  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
F1_BASE = PROJECT_DIR / "checkpoints" / "ft_jazz_pop80" / "best.pt"

N_WARMUP = 3
N_TIMED = 5
BATCH = 32
SEQ_LEN = 256
RANK = 4


def _build_model():
    tokenizer = ChordTokenizer(include_extra_genres=True)
    model = MusicTransformer(
        vocab_size=351, d_model=512, n_heads=8, d_ff=2048, n_layers=8,
        max_seq_len=256, dropout=0.1, pad_id=tokenizer.pad_id,
    )
    ckpt = torch.load(F1_BASE, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    none_id = tokenizer.token2id["[GENRE:none]"]
    _resize_embedding_with_init(model, 351, 359, none_id)
    lora_cfg = LoraConfig(
        r=RANK, lora_alpha=RANK * 2, lora_dropout=0.05,
        target_modules=["w_q", "w_k", "w_v"], bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.to(DEVICE).train()
    return model


def _run_steps(model, label: str, n_steps: int):
    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    crit = nn.CrossEntropyLoss(ignore_index=0)
    torch.cuda.synchronize()
    t_start = time.time()
    for i in range(n_steps):
        x = torch.randint(0, 359, (BATCH, SEQ_LEN), device=DEVICE)
        y = torch.randint(0, 359, (BATCH, SEQ_LEN), device=DEVICE)
        optim.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=True):
            logits = model(x)
            loss = crit(logits.view(-1, logits.shape[-1]), y.view(-1))
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
    torch.cuda.synchronize()
    elapsed = time.time() - t_start
    per_step = elapsed / n_steps
    print(f"  {label}: {n_steps} steps in {elapsed:.2f}s ({per_step*1000:.1f} ms/step)")
    return per_step


def main():
    if not F1_BASE.exists():
        print(f"F1 base not found at {F1_BASE}; aborting smoke test")
        sys.exit(1)
    print(f"Device: {DEVICE} | warmup={N_WARMUP} timed={N_TIMED} batch={BATCH} seq={SEQ_LEN} rank={RANK}")

    print("\n[Phase 1/3] Eager baseline (no torch.compile)")
    eager_model = _build_model()
    _run_steps(eager_model, "  warmup", N_WARMUP)
    eager_per_step = _run_steps(eager_model, "  timed ", N_TIMED)
    del eager_model
    torch.cuda.empty_cache()

    print("\n[Phase 2/3] torch.compile (mode='default')")
    try:
        compiled_model = _build_model()
        compiled_model = torch.compile(compiled_model, mode="default")
    except Exception as e:
        print(f"  FAIL: torch.compile raised at compile-call: {e}")
        sys.exit(1)

    try:
        # First forward triggers actual JIT compile — can be slow (1-5 min)
        print("  triggering JIT compile (first forward, may take a minute)...")
        _run_steps(compiled_model, "  warmup", N_WARMUP)
        compiled_per_step = _run_steps(compiled_model, "  timed ", N_TIMED)
    except Exception as e:
        print(f"  FAIL: torch.compile raised during forward/backward: {e}")
        sys.exit(1)

    print("\n[Phase 3/3] Verdict")
    speedup = (eager_per_step - compiled_per_step) / eager_per_step * 100
    print(f"  eager:    {eager_per_step*1000:.1f} ms/step")
    print(f"  compiled: {compiled_per_step*1000:.1f} ms/step  ({speedup:+.1f}% vs eager)")

    if compiled_per_step > eager_per_step * 1.5:
        print("  FAIL: compiled is >1.5x slower than eager (possible recompile thrashing)")
        sys.exit(1)
    if speedup < 0:
        print("  PASS-WEAK: compile works but no speedup (or slightly slower); decide whether to skip")
        sys.exit(0)
    print(f"  PASS: torch.compile gives ~{speedup:.1f}% speedup. Safe to enable globally.")
    sys.exit(0)


if __name__ == "__main__":
    main()
