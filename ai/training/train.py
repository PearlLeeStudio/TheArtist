"""Training script for Music Transformer — Paper experiment version.

Adds split evaluation (pop/jazz test sets), per-epoch CSV logging,
and TensorBoard per-genre metrics for the genre ratio experiment.

Usage:
    cd ai
    uv run python training/train.py --config training/configs/mix_pop80_jazz20.yaml
    uv run tensorboard --logdir runs/
"""

from __future__ import annotations

import argparse
import csv
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

from dataset import ChordSequenceDataset
from model import MusicTransformer
from tokenizer import ChordTokenizer


def _get_tokenizer_class(version: str) -> type:
    """Select ChordTokenizer (v1) or ChordTokenizerV2 based on config."""
    if version == "v2":
        from tokenizer_v2 import ChordTokenizerV2
        return ChordTokenizerV2
    return ChordTokenizer

PROJECT_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_DIR / "logs"


def _setup_logging(run_name: str) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{run_name}_{time.strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger("train")
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


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    tokenizer: ChordTokenizer,
    device: torch.device,
    use_amp: bool = True,
) -> dict:
    """Evaluate on a single dataset. Returns loss, perplexity, top-1, top-5."""
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

            flat_logits = logits.reshape(-1, tokenizer.vocab_size)
            flat_tgt = tgt.reshape(-1)

            loss = criterion(flat_logits, flat_tgt)  # reduction="sum"
            mask = flat_tgt != tokenizer.pad_id
            valid = mask.sum().item()

            total_loss += loss.item()  # already summed by criterion
            total_tokens += valid

            # top-1
            top1 = flat_logits.argmax(dim=-1)
            top1_correct += ((top1 == flat_tgt) & mask).sum().item()

            # top-5
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


# ------------------------------------------------------------------
# CSV logging
# ------------------------------------------------------------------

CSV_FIELDS = [
    "epoch", "lr", "train_loss",
    "val_loss", "val_ppl", "val_top1", "val_top5",
    "pop_loss", "pop_ppl", "pop_top1", "pop_top5",
    "jazz_loss", "jazz_ppl", "jazz_top1", "jazz_top5",
    # Filled only when select_best_on=jazz_val; empty for legacy runs.
    "jazz_val_loss", "jazz_val_ppl", "jazz_val_top1", "jazz_val_top5",
]


def _write_csv_header(csv_path: Path) -> None:
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()


def _append_csv_row(csv_path: Path, row: dict) -> None:
    with open(csv_path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def train(config: dict) -> None:
    run_name = config.get("run_name", "default")
    log = _setup_logging(run_name)

    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    if device.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name()}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        log.info(f"VRAM: {vram:.1f} GB")

    # ---- Paths ----
    ckpt_dir = PROJECT_DIR / "checkpoints" / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = PROJECT_DIR / "runs" / run_name

    # ---- Tokenizer ----
    tokenizer_version = config.get("tokenizer_version", "v1")
    tokenizer = _get_tokenizer_class(tokenizer_version)()
    tokenizer.save(ckpt_dir / "tokenizer.json")
    log.info(f"Tokenizer: {tokenizer_version} (vocab={tokenizer.vocab_size})")

    # ---- Data ----
    max_seq_len = config.get("max_seq_len", 512)
    batch_size = config.get("batch_size", 16)
    num_workers = config.get("num_workers", 4)

    sources = config.get("sources")
    pop_mix_count = config.get("pop_mix_count")

    log.info(f"Data config: sources={sources}, pop_mix_count={pop_mix_count}")

    train_ds = ChordSequenceDataset(
        split="train", max_seq_len=max_seq_len, augment_keys=True,
        sources=sources, pop_mix_count=pop_mix_count,
        tokenizer_version=tokenizer_version,
    )
    val_ds = ChordSequenceDataset(
        split="val", max_seq_len=max_seq_len, augment_keys=False,
        sources=sources, pop_mix_count=pop_mix_count,
        tokenizer_version=tokenizer_version,
    )

    n_train = len(train_ds)
    if train_ds.augment_keys:
        log.info(f"Train: {n_train} sequences ({n_train // 12} songs x 12 keys)")
    else:
        log.info(f"Train: {n_train} sequences")
    log.info(f"Val:   {len(val_ds)} sequences")

    # Genre-specific test loaders (always full test set, no subsampling)
    pop_test_ds = ChordSequenceDataset(
        split="test", max_seq_len=max_seq_len, augment_keys=False,
        sources=["chordonomicon", "billboard"],
        tokenizer_version=tokenizer_version,
    )
    jazz_sources = ["jht", "wjazzd", "jaah", "jazzstandards", "choco:real-book", "choco:jazz-corpus", "bunks"]
    jazz_test_ds = ChordSequenceDataset(
        split="test", max_seq_len=max_seq_len, augment_keys=False,
        sources=jazz_sources,
        tokenizer_version=tokenizer_version,
    )
    log.info(f"Pop test:  {len(pop_test_ds)} sequences")
    log.info(f"Jazz test: {len(jazz_test_ds)} sequences")

    # Best-checkpoint selection metric. Default "val" = mixed-val loss (historic
    # behavior). "jazz_val" selects on a jazz-only VAL subset instead — needed for
    # fine-tunes whose mixed val is pop-dominated, where min mixed-val loss can
    # retain the un-adapted init as "best" (the released-F1 artifact, 2026-06-11).
    # Selection uses the val split only; the jazz TEST loader stays report-only.
    select_best_on = config.get("select_best_on", "val")
    jazz_val_ds = None
    if select_best_on == "jazz_val":
        jazz_val_ds = ChordSequenceDataset(
            split="val", max_seq_len=max_seq_len, augment_keys=False,
            sources=jazz_sources,
            tokenizer_version=tokenizer_version,
        )
        log.info(f"Jazz val (best-selection metric): {len(jazz_val_ds)} sequences")
    elif select_best_on != "val":
        raise ValueError(f"select_best_on must be 'val' or 'jazz_val', got {select_best_on!r}")

    persistent = config.get("persistent_workers", False) and num_workers > 0
    prefetch = config.get("prefetch_factor", 2) if num_workers > 0 else None

    def make_loader(ds, shuffle=False, drop_last=False):
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=True, drop_last=drop_last,
            persistent_workers=persistent, prefetch_factor=prefetch,
        )

    train_loader = make_loader(train_ds, shuffle=True, drop_last=True)
    val_loader = make_loader(val_ds)
    pop_test_loader = make_loader(pop_test_ds)
    jazz_test_loader = make_loader(jazz_test_ds)
    jazz_val_loader = make_loader(jazz_val_ds) if jazz_val_ds is not None else None

    # ---- Model ----
    model = MusicTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=config.get("d_model", 512),
        n_heads=config.get("n_heads", 8),
        d_ff=config.get("d_ff", 2048),
        n_layers=config.get("n_layers", 8),
        max_seq_len=max_seq_len,
        dropout=config.get("dropout", 0.1),
        pad_id=tokenizer.pad_id,
    ).to(device)

    log.info(f"Parameters: {model.count_parameters():,}")
    log.info(f"Vocab size: {tokenizer.vocab_size}")

    # ---- Resume ----
    start_epoch = 0
    best_val_loss = float("inf")
    best_epoch = -1
    best_is_init = False
    resume_path = config.get("resume_from")
    # For fine-tune (different val distribution), reset best so new best.pt saves.
    # Default True: resume means fine-tune. Set to False to continue same training.
    reset_best = config.get("reset_best_on_resume", True)
    init_fingerprint = None
    if resume_path:
        rp = Path(resume_path)
        if not rp.is_absolute():
            rp = PROJECT_DIR / rp
        if rp.exists():
            ckpt = torch.load(rp, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            # Fingerprint of the resumed init weights — used to detect the
            # released-F1 failure mode (2026-06-11): warmup lr≈0 + pop-dominated
            # val meant the un-finetuned init was saved as best.pt forever.
            init_fingerprint = _weights_fingerprint(model)
            if reset_best:
                best_val_loss = float("inf")
                log.info(f"Resumed from {rp} (epoch {start_epoch}, best_val reset for fine-tune)")
            else:
                best_val_loss = ckpt.get("val_loss", float("inf"))
                log.info(f"Resumed from {rp} (epoch {start_epoch}, val_loss={best_val_loss:.4f})")
            # Same-run resume (host reboot / crash recovery): restore best from
            # the run's own best.pt so a worse later epoch can never overwrite
            # it. Overrides reset_best — resuming a run is not a new fine-tune.
            own_best = ckpt_dir / "best.pt"
            if rp.resolve().parent == ckpt_dir.resolve() and own_best.exists():
                prev_best = torch.load(own_best, map_location="cpu", weights_only=False)
                best_val_loss = prev_best.get("val_loss", best_val_loss)
                best_epoch = prev_best.get("epoch", best_epoch)
                log.info(
                    f"Same-run resume: best restored from best.pt "
                    f"(epoch {best_epoch}, selection_loss={best_val_loss:.4f})"
                )

    # ---- Optimizer & Scheduler ----
    lr = config.get("lr", 1e-4)
    total_epochs = config.get("epochs", 100)
    warmup_epochs = config.get("warmup_epochs", 5)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr,
        weight_decay=config.get("weight_decay", 0.01),
        betas=(0.9, 0.98), eps=1e-9,
    )

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if resume_path and rp.exists():
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        else:
            for _ in range(start_epoch):
                scheduler.step()

    # ---- Loss & AMP ----
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    eval_criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id, reduction="sum")
    use_amp = config.get("use_amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    accum_steps = config.get("gradient_accumulation_steps", 1)

    # ---- TensorBoard & CSV ----
    writer = SummaryWriter(log_dir=str(log_dir))
    with open(ckpt_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    csv_path = ckpt_dir / "eval_results.csv"
    if not csv_path.exists() or start_epoch == 0:
        _write_csv_header(csv_path)

    # ---- Training loop ----
    ckpt_every = config.get("checkpoint_every", 10)
    patience = config.get("patience", 20)
    patience_ctr = 0
    log_every = config.get("log_every_steps", 1000)

    log.info("=" * 60)
    log.info(f"Training: epochs {start_epoch}→{total_epochs}, batch={batch_size}")
    log.info(f"AMP={use_amp}, accum={accum_steps}, effective_batch={batch_size * accum_steps}")
    log.info("=" * 60)

    for epoch in range(start_epoch, total_epochs):
        # ---- Train ----
        model.train()
        train_loss_sum = 0.0
        optimizer.zero_grad()
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            inp = batch["input_ids"].to(device)
            tgt = batch["target_ids"].to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(inp)
                loss = criterion(
                    logits.reshape(-1, tokenizer.vocab_size),
                    tgt.reshape(-1),
                )
                loss = loss / accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    model.parameters(), config.get("max_grad_norm", 1.0)
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss_sum += loss.item() * accum_steps

            if log_every and (step + 1) % log_every == 0:
                avg_so_far = train_loss_sum / (step + 1)
                elapsed = time.time() - t0
                steps_per_sec = (step + 1) / elapsed
                eta_epoch = (len(train_loader) - step - 1) / steps_per_sec
                log.info(
                    f"  step {step+1}/{len(train_loader)} | "
                    f"loss={avg_so_far:.4f} | "
                    f"{steps_per_sec:.1f} steps/s | "
                    f"ETA epoch: {eta_epoch/60:.0f}min"
                )

        avg_train = train_loss_sum / max(1, len(train_loader))
        dt = time.time() - t0

        # ---- Evaluate on val, pop test, jazz test ----
        val_metrics = evaluate(model, val_loader, eval_criterion, tokenizer, device, use_amp)
        pop_metrics = evaluate(model, pop_test_loader, eval_criterion, tokenizer, device, use_amp)
        jazz_metrics = evaluate(model, jazz_test_loader, eval_criterion, tokenizer, device, use_amp)

        avg_val = val_metrics["loss"]

        jazz_val_metrics = None
        if jazz_val_loader is not None:
            jazz_val_metrics = evaluate(model, jazz_val_loader, eval_criterion, tokenizer, device, use_amp)
            selection_loss = jazz_val_metrics["loss"]
        else:
            selection_loss = avg_val

        # ---- Scheduler ----
        scheduler.step()
        cur_lr = optimizer.param_groups[0]["lr"]

        # ---- TensorBoard ----
        writer.add_scalar("loss/train", avg_train, epoch)
        writer.add_scalar("loss/val", avg_val, epoch)
        writer.add_scalar("lr", cur_lr, epoch)
        writer.add_scalar("ppl/train", math.exp(min(avg_train, 10)), epoch)
        writer.add_scalar("ppl/val", val_metrics["perplexity"], epoch)

        for prefix, metrics in [("pop", pop_metrics), ("jazz", jazz_metrics)]:
            writer.add_scalar(f"{prefix}/loss", metrics["loss"], epoch)
            writer.add_scalar(f"{prefix}/ppl", metrics["perplexity"], epoch)
            writer.add_scalar(f"{prefix}/top1_acc", metrics["top1_acc"], epoch)
            writer.add_scalar(f"{prefix}/top5_acc", metrics["top5_acc"], epoch)

        # ---- CSV ----
        csv_row = {
            "epoch": epoch,
            "lr": f"{cur_lr:.2e}",
            "train_loss": f"{avg_train:.4f}",
        }
        csv_metric_blocks = [("val", val_metrics), ("pop", pop_metrics), ("jazz", jazz_metrics)]
        if jazz_val_metrics is not None:
            csv_metric_blocks.append(("jazz_val", jazz_val_metrics))
            writer.add_scalar("jazz_val/loss", jazz_val_metrics["loss"], epoch)
            writer.add_scalar("jazz_val/top1_acc", jazz_val_metrics["top1_acc"], epoch)
        for prefix, metrics in csv_metric_blocks:
            csv_row[f"{prefix}_loss"] = f"{metrics['loss']:.4f}"
            csv_row[f"{prefix}_ppl"] = f"{metrics['perplexity']:.2f}"
            csv_row[f"{prefix}_top1"] = f"{metrics['top1_acc']:.2f}"
            csv_row[f"{prefix}_top5"] = f"{metrics['top5_acc']:.2f}"
        _append_csv_row(csv_path, csv_row)

        # ---- Console log ----
        log.info(
            f"Epoch {epoch:3d}/{total_epochs} | "
            f"train={avg_train:.4f} val={avg_val:.4f} | "
            f"pop_top1={pop_metrics['top1_acc']:.1f}% jazz_top1={jazz_metrics['top1_acc']:.1f}% | "
            f"lr={cur_lr:.2e} | {dt:.0f}s"
        )

        # ---- Checkpoint ----
        # best.pt metadata stores the SELECTION metric (mixed val by default,
        # jazz-val when select_best_on=jazz_val) — see the data-section comment.
        is_best = selection_loss < best_val_loss
        if is_best:
            best_val_loss = selection_loss
            best_epoch = epoch
            patience_ctr = 0
            _save_ckpt(ckpt_dir / "best.pt", epoch, model, optimizer, scheduler, selection_loss, config)
            log.info(f"  -> New best {select_best_on}_loss={best_val_loss:.4f}")
            best_is_init = init_fingerprint is not None and _weights_fingerprint(model) == init_fingerprint
            if best_is_init:
                log.warning(
                    "  !! best.pt weights are IDENTICAL to the resumed init — "
                    "no fine-tuning is captured (warmup lr may be ~0 this epoch, "
                    "or val mix favors the init). If this stays best to the end, "
                    "the released checkpoint will be the un-adapted base "
                    "(this exact failure shipped as F1, found 2026-06-11)."
                )
        else:
            patience_ctr += 1

        if (epoch + 1) % ckpt_every == 0:
            _save_ckpt(
                ckpt_dir / f"epoch_{epoch:04d}.pt",
                epoch, model, optimizer, scheduler, avg_val, config,
            )

        # ---- Early stopping ----
        if patience_ctr >= patience:
            log.info(f"Early stopping at epoch {epoch} (patience={patience})")
            break

    writer.close()
    if best_is_init:
        log.warning(
            "!!!! FINAL best.pt == RESUMED INIT WEIGHTS — this run captured no "
            "fine-tuning in its released artifact. Do NOT ship this best.pt as an "
            "adapted model. Inspect epoch_*.pt checkpoints or change "
            "select_best_on (see ft_jazz_pop80_v2.yaml)."
        )
    log.info(f"Done. Best val_loss={best_val_loss:.4f}")
    log.info(f"Checkpoints: {ckpt_dir}")
    log.info(f"CSV results: {csv_path}")
    log.info(f"TensorBoard: tensorboard --logdir {log_dir}")


def _weights_fingerprint(model: nn.Module) -> str:
    """Weight fingerprint (sha256 over the FULL state dict).

    Used to detect a best.pt that is identical to the resumed init — the
    failure mode that shipped phase0 weights as F1 (2026-06-11). Hashes every
    tensor (partial hashing can miss runs where the sampled tensors are
    frozen); ~1 s for 25M params, called only at resume and best-saves.
    """
    import hashlib
    h = hashlib.sha256()
    sd = model.state_dict()
    for key in sorted(sd):
        h.update(key.encode())
        h.update(sd[key].detach().cpu().numpy().tobytes())
    return h.hexdigest()


def _save_ckpt(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    val_loss: float,
    config: dict,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_loss,
            "config": config,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Music Transformer")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.resume:
        config["resume_from"] = args.resume

    # Optional run seed (2026-06-12, v2 multi-seed restoration runs). Covers
    # torch RNG (dropout, DataLoader shuffling via default generator) and
    # python random. Absent -> previous unseeded behavior unchanged.
    if "seed" in config:
        import random as _random
        torch.manual_seed(int(config["seed"]))
        _random.seed(int(config["seed"]))

    train(config)


if __name__ == "__main__":
    main()
