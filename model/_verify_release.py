"""Pre-release integrity gate for checkpoint uploads.

Run BEFORE model/_upload.py or any re-upload. Catches the two failure modes
found 2026-06-11:

  1. Duplicate weights across differently-named releases — the released F1
     (ft_jazz_pop80/best.pt) was byte-identical to phase0_pop_baseline/best.pt
     because best-checkpoint selection retained the un-finetuned init.
     → pairwise sha256 over full state dicts; any collision FAILS the gate.

  2. Rank-label drift — the model card / paper rank table disagreeing with the
     adapter actually staged for upload (val_top1- vs val_loss-selected ranks
     differed for 6 genres).
     → each ft_f1_lora_<genre>/adapter/adapter_config.json "r" is compared
       against results/lora_published_ranks_*.csv when present.

Usage (from repo root):
    uv run --directory ai python ../model/_verify_release.py
    uv run --directory ai python ../model/_verify_release.py --strict   # exit 1 on warnings too
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import torch

AI_DIR = Path(__file__).resolve().parent.parent / "ai"
CKPT_DIR = AI_DIR / "checkpoints"

FULL_MODELS = [
    "phase0_pop_baseline", "ft_jazz_pop80", "ft_jazz_pop80_v2",
    "ft_jazz_pop67", "ft_jazz_pop50", "ft_jazz_pop29", "ft_jazz_only",
]

# Disclosed weight identities (documented errata) — downgraded to warnings so the
# gate still blocks any NEW collision while letting releases proceed alongside a
# known, already-published one. Unordered pairs.
KNOWN_COLLISIONS = {
    # 2026-06-11 erratum (on both model cards): released F1 retained the Phase-0
    # init via min-mixed-val checkpoint selection. Corrected retrain released
    # separately as ft_jazz_pop80_v2 — that one must stay hash-distinct.
    frozenset({"phase0_pop_baseline", "ft_jazz_pop80"}),
}
LORA_GENRES = [
    "blues", "bossa", "classical", "country", "electronic",
    "folk", "funk", "gospel", "hip_hop", "rnb_soul", "rock",
]


def _full_hash(pt_path: Path) -> str:
    ck = torch.load(pt_path, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict") or ck.get("model") or ck
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true", help="exit 1 on warnings too")
    args = ap.parse_args()

    failures: list[str] = []
    warnings: list[str] = []

    # --- Gate 1: pairwise weight identity across full models ---
    hashes: dict[str, str] = {}
    for name in FULL_MODELS:
        pt = CKPT_DIR / name / "best.pt"
        if not pt.exists():
            warnings.append(f"missing checkpoint (skipped): {pt}")
            continue
        hashes[name] = _full_hash(pt)
        print(f"  {name:24s} {hashes[name][:16]}")
    seen: dict[str, str] = {}
    for name, h in hashes.items():
        if h in seen:
            if frozenset({name, seen[h]}) in KNOWN_COLLISIONS:
                warnings.append(
                    f"known disclosed collision: {name} == {seen[h]} "
                    f"(sha256 {h[:16]}…) — erratum already on both cards"
                )
            else:
                failures.append(
                    f"WEIGHT COLLISION: {name} == {seen[h]} (sha256 {h[:16]}…) — "
                    f"one of these releases is not the model its name claims."
                )
        seen[h] = name

    # --- Gate 2: adapter rank labels vs staged adapters ---
    rank_csvs = sorted((AI_DIR / "results").glob("lora_published_ranks_*.csv"))
    expected: dict[str, int] = {}
    if rank_csvs:
        with open(rank_csvs[-1]) as f:
            expected = {r["genre"]: int(r["rank"]) for r in csv.DictReader(f)}
    for genre in LORA_GENRES:
        cfg = CKPT_DIR / f"ft_f1_lora_{genre}" / "adapter" / "adapter_config.json"
        if not cfg.exists():
            warnings.append(f"missing adapter (skipped): {cfg}")
            continue
        actual = json.loads(cfg.read_text())["r"]
        if expected and genre in expected and expected[genre] != actual:
            failures.append(
                f"RANK DRIFT: {genre} adapter r={actual} but published-rank table says {expected[genre]}"
            )

    print()
    for w in warnings:
        print(f"  WARN  {w}")
    for fmsg in failures:
        print(f"  FAIL  {fmsg}")
    if failures or (args.strict and warnings):
        print("\nRELEASE GATE: FAILED — do not upload.")
        sys.exit(1)
    print("\nRELEASE GATE: PASSED.")


if __name__ == "__main__":
    main()
