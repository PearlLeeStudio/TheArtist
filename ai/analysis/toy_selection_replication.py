"""Toy second-domain replication of the checkpoint-selection failure (ICASSP paper §hardening).

Reproduces, on fully synthetic data outside the music pipeline, the two-factor
mechanism of the F1 incident:
  Factor 1: best-checkpoint selection by min loss on a rehearsal-dominated mixed
            validation set, while fine-tuning toward a target distribution.
  Factor 2: resume + warmup-from-zero + per-epoch scheduler stepping
            => the first fine-tune epoch runs at lr=0 (zero-update epoch), so a
            bit-identical copy of the init sits in the candidate pool.

Setup mirrors the incident's proportions: pretrain a tiny causal transformer LM
on source distribution A (order-1 Markov over a 32-token vocab), then resume at
epoch 3 and fine-tune on a rehearsal mix (~87% A / 13% B). Mixed-validation
loss is evaluated analytically at source shares {0.50, 0.87, 0.98} (0.98
mirrors the incident's 179/10,179 jazz share), selection is recomputed per
composition and under the target criterion, and every epoch checkpoint is
SHA-256 hashed against the init.

Output: ai/results/toy_selection_replication_20260729.csv (+ stdout summary).
Run:    cd ai && uv run python analysis/toy_selection_replication.py
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import torch
import torch.nn as nn

VOCAB = 32
SEQ_LEN = 64
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
SRC_TRAIN, TGT_TRAIN = 8000, 1200          # ~87% / 13% rehearsal mix
SRC_VAL, TGT_VAL, TGT_TEST = 800, 400, 400
VAL_SOURCE_SHARES = [0.50, 0.87, 0.98]     # mixed-val composition sweep; 0.98 mirrors the incident (179/10,179 jazz)
PRETRAIN_EPOCHS = 30                       # pretrain to source convergence (the incident's precondition)
FT_EPOCHS = list(range(3, 10))             # resume at epoch 3, run through 9
WARMUP_EPOCHS = 2                          # from zero, stepped per epoch
BASE_LR = 1e-3
BATCH = 128
SEEDS = [42, 123, 7]
OUT = Path(__file__).resolve().parents[1] / "results" / "toy_selection_replication_20260729.csv"


def markov_matrix(gen: torch.Generator, concentration: float = 0.3) -> torch.Tensor:
    """Random sparse-ish row-stochastic transition matrix."""
    logits = torch.randn(VOCAB, VOCAB, generator=gen) / concentration
    return torch.softmax(logits, dim=-1)


def sample_seqs(mat: torch.Tensor, n: int, gen: torch.Generator) -> torch.Tensor:
    seqs = torch.empty(n, SEQ_LEN, dtype=torch.long)
    state = torch.randint(0, VOCAB, (n,), generator=gen)
    seqs[:, 0] = state
    for t in range(1, SEQ_LEN):
        probs = mat[state]
        state = torch.multinomial(probs, 1, generator=gen).squeeze(1)
        seqs[:, t] = state
    return seqs


class TinyLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(VOCAB, D_MODEL)
        self.pos = nn.Embedding(SEQ_LEN, D_MODEL)
        layer = nn.TransformerEncoderLayer(
            D_MODEL, N_HEADS, dim_feedforward=4 * D_MODEL,
            batch_first=True, dropout=0.0, norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, N_LAYERS)
        self.head = nn.Linear(D_MODEL, VOCAB)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.shape[1]
        mask = torch.triu(torch.full((t, t), float("-inf")), diagonal=1)
        h = self.emb(x) + self.pos(torch.arange(t))
        return self.head(self.enc(h, mask=mask))


def state_hash(model: nn.Module) -> str:
    h = hashlib.sha256()
    for _, v in sorted(model.state_dict().items()):
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def eval_loss(model: nn.Module, data: torch.Tensor) -> float:
    model.eval()
    crit = nn.CrossEntropyLoss(reduction="sum")
    total, count = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(data), BATCH):
            x = data[i:i + BATCH]
            logits = model(x[:, :-1])
            total += crit(logits.reshape(-1, VOCAB), x[:, 1:].reshape(-1)).item()
            count += x[:, 1:].numel()
    return total / count


def train_epoch(model: nn.Module, data: torch.Tensor, opt: torch.optim.Optimizer,
                lr: float, gen: torch.Generator) -> None:
    model.train()
    for g in opt.param_groups:
        g["lr"] = lr
    crit = nn.CrossEntropyLoss()
    perm = torch.randperm(len(data), generator=gen)
    for i in range(0, len(data), BATCH):
        x = data[perm[i:i + BATCH]]
        loss = crit(model(x[:, :-1]).reshape(-1, VOCAB), x[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()  # lr=0 => zero parameter update (weights bit-identical); Adam moment
        # buffers may still accumulate during the zero-lr epoch, as in real pipelines


def run_seed(seed: int) -> list[dict]:
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    # Two fixed, genuinely different Markov sources (shared across seeds via gen2)
    gen2 = torch.Generator().manual_seed(1000)
    mat_src, mat_tgt = markov_matrix(gen2), markov_matrix(gen2)

    src_train = sample_seqs(mat_src, SRC_TRAIN, gen)
    tgt_train = sample_seqs(mat_tgt, TGT_TRAIN, gen)
    mix_train = torch.cat([src_train, tgt_train])
    src_val = sample_seqs(mat_src, SRC_VAL, gen)
    tgt_val = sample_seqs(mat_tgt, TGT_VAL, gen)
    tgt_test = sample_seqs(mat_tgt, TGT_TEST, gen)

    model = TinyLM()
    opt = torch.optim.Adam(model.parameters(), lr=BASE_LR)

    # Phase 0: pretrain to convergence on pure source
    for _ in range(PRETRAIN_EPOCHS):
        train_epoch(model, src_train, opt, BASE_LR, gen)
    init_hash = state_hash(model)

    # Fine-tune: resume at epoch 3; warmup from zero, stepped per epoch.
    # Mixed val loss at source share s is exactly s*L_src + (1-s)*L_tgt
    # (equal-length sequences), so the composition sweep is computed analytically.
    rows = []
    for ep in FT_EPOCHS:
        lr = BASE_LR * min(1.0, (ep - FT_EPOCHS[0]) / WARMUP_EPOCHS)
        train_epoch(model, mix_train, opt, lr, gen)
        l_src = eval_loss(model, src_val)
        l_tgt = eval_loss(model, tgt_val)
        row = {
            "seed": seed, "epoch": ep, "lr": lr,
            "src_val_loss": round(l_src, 5),
            "target_val_loss": round(l_tgt, 5),
            "target_test_loss": round(eval_loss(model, tgt_test), 5),
            "hash16": state_hash(model),
            "is_init": state_hash(model) == init_hash,
        }
        for share in VAL_SOURCE_SHARES:
            row[f"mixed_val_s{int(share*100)}"] = round(share * l_src + (1 - share) * l_tgt, 5)
        rows.append(row)
    return rows


def main() -> None:
    all_rows = []
    for seed in SEEDS:
        rows = run_seed(seed)
        all_rows.extend(rows)
        by_tgt = min(rows, key=lambda r: r["target_val_loss"])
        line = [f"seed {seed}: target-val pick = ep{by_tgt['epoch']}"
                f" (init={by_tgt['is_init']}, tgt test {by_tgt['target_test_loss']:.4f})"]
        for share in VAL_SOURCE_SHARES:
            key = f"mixed_val_s{int(share*100)}"
            pick = min(rows, key=lambda r: r[key])
            mono = all(rows[i][key] <= rows[i + 1][key] for i in range(len(rows) - 1))
            line.append(f"  s={share:.2f}: pick ep{pick['epoch']}"
                        f" (init={pick['is_init']}, tgt test {pick['target_test_loss']:.4f},"
                        f" monotone={mono})")
        line.append(f"  epoch-3 zero-update (== init): {rows[0]['is_init']};"
                    f" src forgetting {rows[0]['src_val_loss']:.4f} -> {rows[-1]['src_val_loss']:.4f};"
                    f" target {rows[0]['target_val_loss']:.4f} -> {rows[-1]['target_val_loss']:.4f}")
        print(chr(10).join(line))
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
