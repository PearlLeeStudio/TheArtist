"""LoRA fine-tune script for genre adapters on top of F1 base.

Strategy (Pearl 2026-05-09):
- F1 (`ft_jazz_pop80/best.pt`) frozen as base.
- Per-genre LoRA adapter on attention QKV (target_modules=["w_q","w_k","w_v"]).
- Tokenizer extended with EXTRA_GENRES (8 new tokens at end of vocab).
- Embedding layer resized: new genre rows initialized from [GENRE:none].
- Trainable: LoRA adapter weights + new genre embedding rows. Everything else frozen.

Output:
- adapter weights → checkpoints/<run_name>/adapter/
- extended tokenizer → checkpoints/<run_name>/tokenizer.json
- training log + eval CSV alongside

Usage:
    cd ai
    uv run python training/lora_train.py --config training/configs/ft_f1_lora_country.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from peft import LoraConfig, IA3Config, get_peft_model

# 2026-05-20 Pearl: GPU efficiency on RTX 4070 Laptop (8GB VRAM, 256 GB/s mem bw).
# cudnn.benchmark caches the fastest kernel for the fixed input shape
# (max_seq_len=256 padded batches), giving 5-15% speedup with no correctness
# risk. Input shapes are stable across batches so the benchmark cache hits.
torch.backends.cudnn.benchmark = True
# Allow TF32 matmul precision (default 'highest' in PyTorch >=2.0). For
# non-AMP matmuls this trades a small precision loss for throughput; AMP
# path is the dominant compute so the impact is mostly on the eval matmuls.
torch.set_float32_matmul_precision("high")
# Belt-and-suspenders TF32 flags (default True on Ampere/Ada but explicit
# in case PyTorch defaults shift):
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# Dynamo safety net + max compile retention. Default cache_size_limit=8
# is too small for train+val+AMP variants — was hitting the cap after
# epoch 0 and silently falling back to eager (only epoch 0 saw +45%
# speedup; epochs 1-7 ran eager). Raising to 64 keeps train forward,
# train backward, val forward, mixed-precision variants, and last-batch
# shape variation all cached simultaneously. accumulated_cache_size_limit
# prevents early-epoch graphs from being evicted as new ones compile.
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.accumulated_cache_size_limit = 256

from dataset import ChordSequenceDataset
from model import MusicTransformer
from prefix_tuning import PrefixTuningWrapper
from tokenizer import ChordTokenizer, EXTRA_GENRES

PROJECT_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_DIR / "logs"


# ---- Logging -----------------------------------------------------------

def _setup_logging(run_name: str) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{run_name}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    logger = logging.getLogger("lora_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.info(f"Logging to {log_path}")
    return logger


# ---- Vocab extension ---------------------------------------------------

def _resize_embedding_with_init(
    model: MusicTransformer,
    old_vocab_size: int,
    new_vocab_size: int,
    none_token_id: int,
) -> None:
    """Grow token_emb and out_proj to new_vocab_size; init new rows from [GENRE:none]."""
    if new_vocab_size <= old_vocab_size:
        return

    d_model = model.token_emb.embedding_dim
    pad_id = model.token_emb.padding_idx

    # token_emb: copy old, init new from [GENRE:none] row
    new_emb = nn.Embedding(new_vocab_size, d_model, padding_idx=pad_id)
    with torch.no_grad():
        new_emb.weight[:old_vocab_size] = model.token_emb.weight
        none_row = model.token_emb.weight[none_token_id]
        for i in range(old_vocab_size, new_vocab_size):
            new_emb.weight[i] = none_row
    model.token_emb = new_emb

    # out_proj: copy old, init new from [GENRE:none] row
    new_out = nn.Linear(d_model, new_vocab_size, bias=False)
    with torch.no_grad():
        new_out.weight[:old_vocab_size] = model.out_proj.weight
        none_row = model.out_proj.weight[none_token_id]
        for i in range(old_vocab_size, new_vocab_size):
            new_out.weight[i] = none_row
    model.out_proj = new_out


# ---- Eval --------------------------------------------------------------

def evaluate(model, loader, criterion, vocab_size, pad_id, device, use_amp=True) -> dict:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    top1_correct = 0
    top5_correct = 0
    with torch.no_grad():
        for batch in loader:
            inp = batch["input_ids"].to(device)
            tgt = batch["target_ids"].to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(inp)
            flat_logits = logits.reshape(-1, vocab_size)
            flat_tgt = tgt.reshape(-1)
            loss = criterion(flat_logits, flat_tgt)
            mask = flat_tgt != pad_id
            valid = mask.sum().item()
            total_loss += loss.item()
            total_tokens += valid
            top1 = flat_logits.argmax(dim=-1)
            top1_correct += ((top1 == flat_tgt) & mask).sum().item()
            top5 = flat_logits.topk(5, dim=-1).indices
            top5_hit = (top5 == flat_tgt.unsqueeze(-1)).any(dim=-1)
            top5_correct += (top5_hit & mask).sum().item()
    avg_loss = total_loss / max(1, total_tokens)
    return {
        "loss": avg_loss,
        "perplexity": math.exp(min(avg_loss, 10)),
        "top1_acc": top1_correct / max(1, total_tokens) * 100,
        "top5_acc": top5_correct / max(1, total_tokens) * 100,
    }


# ---- Training ----------------------------------------------------------

def train(config: dict) -> None:
    run_name = config["run_name"]
    target_genre = config["target_genre"]  # e.g. "country" — token name
    base_ckpt_path = Path(config["base_checkpoint"])  # e.g. checkpoints/ft_jazz_pop80/best.pt
    sources = config["sources"]  # e.g. ["chordonomicon_country"]

    log = _setup_logging(run_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    if device.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name()}")

    # Seed (for Experiment 3 multi-seed comparison). Default 42 matches
    # the LoRA runs reported in the original 11-task study.
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    log.info(f"Seed: {seed}")

    ckpt_dir = PROJECT_DIR / "checkpoints" / run_name
    adapter_dir = ckpt_dir / "adapter"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- Tokenizers: old (F1 base) and new (extended) ----
    old_tokenizer = ChordTokenizer(include_extra_genres=False)
    new_tokenizer = ChordTokenizer(include_extra_genres=True)
    old_vocab = old_tokenizer.vocab_size
    new_vocab = new_tokenizer.vocab_size
    log.info(f"Vocab: {old_vocab} (base) → {new_vocab} (+{new_vocab - old_vocab} extra genres)")
    new_tokenizer.save(ckpt_dir / "tokenizer.json")

    # Sanity check: target_genre is among extra genres
    if target_genre not in EXTRA_GENRES and target_genre not in (
        # also allow strengthening of existing-vocab genres
        "rock", "blues", "bossa"
    ):
        log.warning(f"target_genre='{target_genre}' is not in EXTRA_GENRES nor strengthening vocab")

    # --- Data ----
    max_seq_len = config.get("max_seq_len", 256)
    batch_size = config.get("batch_size", 32)
    num_workers = config.get("num_workers", 4)

    train_ds = ChordSequenceDataset(
        split="train", max_seq_len=max_seq_len, augment_keys=True,
        sources=sources, tokenizer_version="v1",
        include_extra_genres=True,
    )
    val_ds = ChordSequenceDataset(
        split="val", max_seq_len=max_seq_len, augment_keys=False,
        sources=sources, tokenizer_version="v1",
        include_extra_genres=True,
    )

    n_train = len(train_ds)
    log.info(f"Train: {n_train} sequences ({n_train // 12} songs x 12 keys)")
    log.info(f"Val:   {len(val_ds)} sequences")

    persistent = num_workers > 0
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True, persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, persistent_workers=persistent,
    )

    # --- Model: build with OLD vocab to match F1 base, then extend ----
    model = MusicTransformer(
        vocab_size=old_vocab,
        d_model=config.get("d_model", 512),
        n_heads=config.get("n_heads", 8),
        d_ff=config.get("d_ff", 2048),
        n_layers=config.get("n_layers", 8),
        max_seq_len=max_seq_len,
        dropout=config.get("dropout", 0.1),
        pad_id=old_tokenizer.pad_id,
    )

    # Load F1 base
    if not base_ckpt_path.is_absolute():
        base_ckpt_path = PROJECT_DIR / base_ckpt_path
    log.info(f"Loading F1 base from {base_ckpt_path}")
    ckpt = torch.load(base_ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    # Extend to new vocab — copy old rows, init new from [GENRE:none]
    none_id = new_tokenizer.encode_genre("none")
    _resize_embedding_with_init(model, old_vocab, new_vocab, none_id)
    model.to(device)
    log.info(f"Model parameters (post-resize): {sum(p.numel() for p in model.parameters()):,}")

    # --- PEFT wrap selector -------------------------------------------------
    # Backward-compat: `disable_lora: true` from older configs maps to the new
    # peft_method="full". `peft_method` defaults to "lora" so all existing
    # configs without the field continue to use LoRA exactly as before.
    disable_lora = config.get("disable_lora", False)
    peft_method = config.get("peft_method", "full" if disable_lora else "lora").lower()

    if peft_method == "full":
        # Full fine-tune mode: skip the peft wrap and train every parameter.
        log.info("Full fine-tune mode (peft_method=full): no PEFT wrap, all params trainable")
        for p in model.parameters():
            p.requires_grad = True
    elif peft_method == "lora":
        # LoRA wrap with peft, target attention QKV
        lora_r = config.get("lora_r", 8)
        lora_alpha = config.get("lora_alpha", 16)
        lora_dropout = config.get("lora_dropout", 0.05)
        target_modules = config.get("lora_target_modules", ["w_q", "w_k", "w_v"])

        lora_cfg = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)

        # Make new embedding rows trainable (peft freezes everything by default)
        # token_emb / out_proj are NOT in target_modules, so they're frozen.
        # We need new genre rows trainable. Approach: keep entire token_emb +
        # out_proj trainable (small overhead; vocab=~370 tokens × 512 dim).
        for name, param in model.named_parameters():
            if "token_emb" in name or "out_proj" in name:
                param.requires_grad = True
    elif peft_method == "ia3":
        # IA3: per-feature scaling on attention K/V and the FFN expansion
        # output. PEFT lib supports custom target_modules + feedforward_modules.
        ia3_targets = config.get("ia3_target_modules", ["w_k", "w_v"])
        ia3_feedforward = config.get("ia3_feedforward_modules", [])
        ia3_cfg = IA3Config(
            target_modules=ia3_targets,
            feedforward_modules=ia3_feedforward,
        )
        model = get_peft_model(model, ia3_cfg)
        for name, param in model.named_parameters():
            if "token_emb" in name or "out_proj" in name:
                param.requires_grad = True
    elif peft_method == "bitfit":
        # BitFit: train every .bias parameter + the new genre embedding rows.
        for p in model.parameters():
            p.requires_grad = False
        for name, param in model.named_parameters():
            if name.endswith(".bias"):
                param.requires_grad = True
            if "token_emb" in name or "out_proj" in name:
                param.requires_grad = True
    elif peft_method == "prefix_tuning":
        # Prefix-tuning: wrap base in PrefixTuningWrapper. Base params are
        # frozen by the wrapper constructor; only the prefix table trains.
        # We additionally re-enable token_emb / out_proj so the freshly-
        # added genre embedding rows can move — mirrors LoRA/IA3 above.
        prefix_len = config.get("prefix_len", 20)
        prefix_init_std = config.get("prefix_init_std", 0.02)
        model = PrefixTuningWrapper(
            model,
            prefix_len=prefix_len,
            init_std=prefix_init_std,
        )
        for name, param in model.named_parameters():
            if "token_emb" in name or "out_proj" in name:
                param.requires_grad = True
    elif peft_method == "embedding_only":
        # Control-token baseline: freeze the entire transformer; only the new
        # genre token embedding rows (and their out_proj counterparts) train.
        # Tests "why not just genre tokens?" — adapter-less symbolic conditioning.
        for p in model.parameters():
            p.requires_grad = False
        for name, param in model.named_parameters():
            if "token_emb" in name or "out_proj" in name:
                param.requires_grad = True
    else:
        raise ValueError(
            f"Unknown peft_method={peft_method!r}; expected one of "
            "'lora', 'full', 'ia3', 'bitfit', 'prefix_tuning', 'embedding_only'."
        )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info(f"PEFT method: {peft_method}")
    log.info(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    if peft_method == "lora":
        log.info(f"LoRA r={config.get('lora_r', 8)} alpha={config.get('lora_alpha', 16)} target={config.get('lora_target_modules', ['w_q', 'w_k', 'w_v'])}")
    elif peft_method == "ia3":
        log.info(f"IA3 target={config.get('ia3_target_modules', ['w_k', 'w_v'])} feedforward={config.get('ia3_feedforward_modules', [])}")
    elif peft_method == "prefix_tuning":
        log.info(f"PrefixTuning prefix_len={config.get('prefix_len', 20)} init_std={config.get('prefix_init_std', 0.02)}")

    # torch.compile wrap for GPU efficiency on 4070 Laptop. PEFT's LoRA
    # path tripped inductor at first forward with 'float has no attribute
    # meta' — module-level torch._dynamo.config.suppress_errors=True now
    # lets dynamo fall back to eager per sub-graph silently instead of
    # crashing. Compatible sub-graphs still get compiled.
    if config.get("torch_compile", True):
        try:
            model = torch.compile(model, mode="default")
            log.info("torch.compile enabled (mode=default, suppress_errors=True for graceful fallback)")
        except Exception as exc:
            log.warning(f"torch.compile failed ({exc!r}); continuing in eager mode")

    # --- Optimizer & loss ----
    lr = config.get("lr", 3e-4)
    total_epochs = config.get("epochs", 5)
    warmup_epochs = config.get("warmup_epochs", 1)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=config.get("weight_decay", 0.01),
        betas=(0.9, 0.98), eps=1e-9,
    )

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    criterion = nn.CrossEntropyLoss(ignore_index=new_tokenizer.pad_id)
    eval_criterion = nn.CrossEntropyLoss(ignore_index=new_tokenizer.pad_id, reduction="sum")
    use_amp = config.get("use_amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    accum = config.get("gradient_accumulation_steps", 1)

    writer = SummaryWriter(log_dir=str(PROJECT_DIR / "runs" / run_name))
    with open(ckpt_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    log.info(f"Training: {total_epochs} epochs, lr={lr}, batch={batch_size}, accum={accum}")

    # Memory probe at startup: catches misconfigured batch_size before wasting epoch time.
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        free_b, total_b = torch.cuda.mem_get_info()
        log.info(f"  GPU mem at start: {free_b/1e9:.2f}GB free / {total_b/1e9:.2f}GB total")

    best_val = float("inf")
    for epoch in range(total_epochs):
        model.train()
        train_loss_sum = 0.0
        optimizer.zero_grad()
        t0 = time.time()
        for step, batch in enumerate(train_loader):
            inp = batch["input_ids"].to(device)
            tgt = batch["target_ids"].to(device)
            try:
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(inp)
                    loss = criterion(logits.reshape(-1, new_vocab), tgt.reshape(-1)) / accum
                scaler.scale(loss).backward()
            except torch.cuda.OutOfMemoryError:
                # Re-raise so main()'s OOM retry loop halves batch_size and restarts
                # training from scratch with a smaller batch. No silent skip — paper
                # cell completes (just slower).
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    peak_b = torch.cuda.max_memory_allocated()
                    log.error(f"  OOM at epoch {epoch} step {step} batch={batch_size}: peak {peak_b/1e9:.2f}GB. main() will retry with batch_size {batch_size//2}.")
                raise
            if (step + 1) % accum == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    config.get("max_grad_norm", 1.0),
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            train_loss_sum += loss.item() * accum

        avg_train = train_loss_sum / max(1, len(train_loader))
        # REVERT 2026-05-27 19:00: `with torch._dynamo.disable():` is not
        # a context manager in this torch version — crashed bossa_r16_seed123.
        # Reverting to plain eval; live with epochs 1-7 falling back to eager
        # (verified safe: results correct, only ~5% slower per cell).
        val_metrics = evaluate(model, val_loader, eval_criterion, new_vocab, new_tokenizer.pad_id, device, use_amp)
        scheduler.step()

        writer.add_scalar("loss/train", avg_train, epoch)
        writer.add_scalar("loss/val", val_metrics["loss"], epoch)
        writer.add_scalar("ppl/val", val_metrics["perplexity"], epoch)
        writer.add_scalar("top1/val", val_metrics["top1_acc"], epoch)
        writer.add_scalar("top5/val", val_metrics["top5_acc"], epoch)

        # Defragment + peak-mem log between epochs. Helps avoid epoch-N OOMs
        # from caching allocator fragmentation accumulated across epochs.
        if device.type == "cuda":
            peak_b = torch.cuda.max_memory_allocated()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            mem_note = f" peak={peak_b/1e9:.2f}GB"
        else:
            mem_note = ""
        log.info(
            f"Epoch {epoch:3d}/{total_epochs} | "
            f"train={avg_train:.4f} val={val_metrics['loss']:.4f} "
            f"top1={val_metrics['top1_acc']:.1f}% top5={val_metrics['top5_acc']:.1f}% | "
            f"{time.time()-t0:.0f}s{mem_note}"
        )

        # NaN guard: if training diverged and val_loss is NaN/Inf, the
        # comparison `nan < best_val` is False so we wouldn't save here —
        # but be explicit so downstream scripts never see a NaN best.pt
        # if training recovers later via a fluky finite loss.
        if math.isfinite(val_metrics["loss"]) and val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            if peft_method in ("full", "bitfit"):
                # Full fine-tune or BitFit: save entire model state as best.pt.
                # Unwrap torch.compile so saved keys don't carry _orig_mod prefix.
                base = getattr(model, "_orig_mod", model)
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": base.state_dict(),
                    "val_loss": best_val,
                    "config": config,
                }, ckpt_dir / "best.pt")
                log.info(f"  -> New best val={best_val:.4f}, full checkpoint saved")
            elif peft_method == "prefix_tuning":
                # Prefix-tuning: save just the prefix table + the extended
                # embedding rows. Loading at eval time rebuilds the base
                # from the F1 checkpoint, resizes the embedding, wraps
                # with PrefixTuningWrapper, then loads these two files.
                torch.save({
                    "epoch": epoch,
                    "prefix_state": model.prefix_state_dict(),
                    "prefix_len": model.prefix_len,
                    "val_loss": best_val,
                    "config": config,
                }, ckpt_dir / "prefix_state.pt")
                torch.save({
                    "epoch": epoch,
                    "token_emb_state": model.model.token_emb.state_dict(),
                    "out_proj_state": model.model.out_proj.state_dict(),
                    "val_loss": best_val,
                    "config": config,
                }, ckpt_dir / "embedding_extension.pt")
                log.info(f"  -> New best val={best_val:.4f}, prefix+embedding saved")
            elif peft_method == "embedding_only":
                # Control-token baseline: no adapter wrap, no save_pretrained.
                # Just the extended token_emb + out_proj rows that trained.
                # _orig_mod unwraps torch.compile if applied; otherwise
                # accesses the raw MusicTransformer directly.
                base = getattr(model, "_orig_mod", model)
                torch.save({
                    "epoch": epoch,
                    "token_emb_state": base.token_emb.state_dict(),
                    "out_proj_state": base.out_proj.state_dict(),
                    "val_loss": best_val,
                    "config": config,
                }, ckpt_dir / "embedding_extension.pt")
                log.info(f"  -> New best val={best_val:.4f}, embedding-only saved")
            else:
                adapter_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(str(adapter_dir))
                # Patch adapter_config.json so HF model card parsing does not
                # emit the "must be a string" warnings on null fields. peft
                # leaves base_model_name_or_path and task_type as None for
                # non-HF-transformer base models (Music Transformer is custom);
                # we set sensible string values explicitly.
                _adapter_cfg_path = adapter_dir / "adapter_config.json"
                if _adapter_cfg_path.exists():
                    import json as _json
                    _cfg = _json.loads(_adapter_cfg_path.read_text())
                    _cfg.setdefault("base_model_name_or_path",
                                    "PearlLeeStudio/TheArtist-MusicTransformer-ft-pop80")
                    if _cfg.get("base_model_name_or_path") is None:
                        _cfg["base_model_name_or_path"] = "PearlLeeStudio/TheArtist-MusicTransformer-ft-pop80"
                    if _cfg.get("task_type") is None:
                        _cfg["task_type"] = "FEATURE_EXTRACTION"
                    _adapter_cfg_path.write_text(_json.dumps(_cfg, indent=2))
                torch.save({
                    "epoch": epoch,
                    "token_emb_state": model.base_model.model.token_emb.state_dict(),
                    "out_proj_state": model.base_model.model.out_proj.state_dict(),
                    "val_loss": best_val,
                    "config": config,
                }, ckpt_dir / "embedding_extension.pt")
                log.info(f"  -> New best val={best_val:.4f}, adapter+embedding saved")

    writer.close()
    log.info(f"Done. Best val_loss={best_val:.4f}")
    if peft_method in ("full", "bitfit"):
        log.info(f"Full checkpoint: {ckpt_dir / 'best.pt'}")
    elif peft_method == "prefix_tuning":
        log.info(f"Prefix:     {ckpt_dir / 'prefix_state.pt'}")
        log.info(f"Embedding:  {ckpt_dir / 'embedding_extension.pt'}")
    elif peft_method == "embedding_only":
        log.info(f"Embedding:  {ckpt_dir / 'embedding_extension.pt'}")
    else:
        log.info(f"Adapter:    {adapter_dir}")
        log.info(f"Embedding:  {ckpt_dir / 'embedding_extension.pt'}")
    log.info(f"Tokenizer:  {ckpt_dir / 'tokenizer.json'}")


def main() -> None:
    p = argparse.ArgumentParser(description="Train per-genre LoRA adapter on F1 base")
    p.add_argument("--config", type=str, required=True, help="YAML config path")
    args = p.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # OOM auto-retry: halve batch_size up to 3 times before giving up.
    # Keeps effective batch the same by raising gradient_accumulation_steps in lockstep,
    # so convergence behavior matches the original config.
    max_oom_retries = 3
    for attempt in range(max_oom_retries + 1):
        try:
            train(config)
            return
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            old_bs = config.get("batch_size", 32)
            old_accum = config.get("gradient_accumulation_steps", 1)
            if old_bs <= 4 or attempt == max_oom_retries:
                log.error(f"OOM persists at batch_size={old_bs} after {attempt+1} attempts — giving up on this cell")
                raise
            new_bs = max(4, old_bs // 2)
            new_accum = old_accum * (old_bs // new_bs)
            log.warning(f"OOM caught (attempt {attempt+1}/{max_oom_retries+1}). "
                        f"Retrying with batch_size {old_bs}→{new_bs}, accum {old_accum}→{new_accum} "
                        f"(same effective batch)")
            config["batch_size"] = new_bs
            config["gradient_accumulation_steps"] = new_accum


if __name__ == "__main__":
    main()
