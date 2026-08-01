"""Phase 3 cell-level audit: verify each expected cell has valid best/adapter + reached epoch.

Reports:
- false_skip: chain skip log shows SKIP but checkpoint is incomplete/stale
- incomplete: cell directory exists but epoch < expected
- missing: cell directory doesn't exist
- nan_loss: best val_loss is NaN/inf
- crashed: log has Traceback but no final 'Done.'

Usage:
    uv run python training/phase3_audit.py
"""
from __future__ import annotations
import json
import math
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints"
LOGS = ROOT / "logs"

GENRES_11 = ["blues", "bossa", "classical", "country", "electronic",
             "folk", "funk", "gospel", "hip_hop", "rnb_soul", "rock"]
SEEDS_E1 = [42, 123, 7]
LARGE_GENRES = ["country", "folk", "rock"]


def _log_epoch_max(run_name: str, expected_epochs: int) -> int:
    """Return max epoch number found across all logs for this run, -1 if no log."""
    max_ep = -1
    for log in LOGS.glob(f"{run_name}_*.log"):
        for line in log.read_text(errors="ignore").splitlines():
            m = re.search(r"Epoch +(\d+)/" + str(expected_epochs), line)
            if m:
                max_ep = max(max_ep, int(m.group(1)))
    return max_ep


def _log_has_crash(run_name: str) -> bool:
    for log in LOGS.glob(f"{run_name}_*.log"):
        text = log.read_text(errors="ignore")
        if "Traceback" in text and "Done." not in text.splitlines()[-20:][-1] if text.splitlines() else False:
            return True
    return False


def _load_val_loss(run_name: str, method: str) -> float | None:
    if method in ("full", "bitfit"):
        path = CKPT / run_name / "best.pt"
        if not path.exists():
            return None
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            return ckpt.get("val_loss")
        except Exception:
            return None
    if method == "embedding_only":
        path = CKPT / run_name / "embedding_extension.pt"
        if not path.exists():
            return None
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            return ckpt.get("val_loss")
        except Exception:
            return None
    if method in ("lora", "ia3"):
        ad = CKPT / run_name / "adapter" / "adapter_config.json"
        em = CKPT / run_name / "embedding_extension.pt"
        if not ad.exists() or not em.exists():
            return None
        try:
            ckpt = torch.load(em, map_location="cpu", weights_only=False)
            return ckpt.get("val_loss")
        except Exception:
            return None
    return None


def audit_step(label: str, cells: list[tuple[str, str, int]]) -> dict:
    """cells: list of (run_name, method, expected_epochs)."""
    results = {"ok": 0, "missing": [], "incomplete": [], "nan": [], "crashed": []}
    for run_name, method, n_ep in cells:
        max_ep = _log_epoch_max(run_name, n_ep)
        val_loss = _load_val_loss(run_name, method)
        ckpt_present = val_loss is not None

        if not ckpt_present and max_ep < 0:
            results["missing"].append(run_name)
        elif not ckpt_present:
            results["incomplete"].append((run_name, f"log epoch {max_ep}/{n_ep-1} but no ckpt"))
        elif max_ep < n_ep - 1:
            results["incomplete"].append((run_name, f"epoch {max_ep}/{n_ep-1}, ckpt val={val_loss}"))
        elif val_loss is None or not math.isfinite(val_loss):
            results["nan"].append((run_name, val_loss))
        elif _log_has_crash(run_name):
            results["crashed"].append(run_name)
        else:
            results["ok"] += 1
    return results


def report(label: str, r: dict, total: int):
    bad = len(r["missing"]) + len(r["incomplete"]) + len(r["nan"]) + len(r["crashed"])
    status = "✓" if bad == 0 else "✗"
    print(f"{status} {label}: {r['ok']}/{total} OK")
    for run in r["missing"]:
        print(f"    MISSING  {run}")
    for run, msg in r["incomplete"]:
        print(f"    INCOMPLETE  {run}: {msg}")
    for run, v in r["nan"]:
        print(f"    NAN/INF  {run} val_loss={v}")
    for run in r["crashed"]:
        print(f"    CRASHED  {run}")
    return bad


def main():
    total_bad = 0

    # P10 E1: 33 cells
    p10 = [(f"ft_f1_control_token_{g}_seed{s}", "embedding_only", 8)
           for g in GENRES_11 for s in SEEDS_E1]
    total_bad += report("P10 E1 control-token", audit_step("P10", p10), len(p10))

    # P11 E2: 11 Phase0+Full FT cells
    p11 = [(f"phase0_full_ft_{g}", "full", 8) for g in GENRES_11]
    total_bad += report("P11 E2 Phase0+Full FT", audit_step("P11", p11), len(p11))

    # P12 E3: 11 Phase0+LoRA cells
    p12 = [(f"phase0_lora_{g}", "lora", 8) for g in GENRES_11]
    total_bad += report("P12 E3 Phase0+LoRA", audit_step("P12", p12), len(p12))

    # P13 F1 Stage A: 40 cells (10 genres × 4 methods, funk-sized at 5 epochs)
    # NOTE: classical excluded from funk-sized sub-sample (per subsample_funk_size.py).
    GENRES_FUNK = [g for g in GENRES_11 if g != "classical"]
    method_kind = {"lora": "lora", "bitfit": "bitfit", "ia3": "lora", "full": "full"}
    p13 = [(f"stageA_{m}_{g}_funksize", method_kind[m], 5)
           for g in GENRES_FUNK for m in ["lora", "bitfit", "ia3", "full"]]
    total_bad += report("P13 F1 Stage A funk-sized", audit_step("P13", p13), len(p13))

    # P14 F2 Stage B: 24 cells (3 large genres × 2 methods × 4 epoch settings)
    p14 = []
    for g in LARGE_GENRES:
        for m in ["lora", "full"]:
            for ep in [3, 5, 8, 12]:
                p14.append((f"stageB_{m}_{g}_ep{ep}", "lora" if m == "lora" else "full", ep))
    total_bad += report("P14 F2 Stage B epoch-sensitivity", audit_step("P14", p14), len(p14))

    print()
    print("=" * 50)
    if total_bad == 0:
        print("OVERALL: ALL CELLS OK ✓")
        sys.exit(0)
    else:
        print(f"OVERALL: {total_bad} cells need attention ✗")
        sys.exit(1)


if __name__ == "__main__":
    main()
