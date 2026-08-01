"""Phase 3 P14b: Evaluate all newly-trained phase 3 cells.

Loops through E1 (control-token), E2 (Phase 0 + Full FT), E3 (Phase 0 +
LoRA), F1 (Stage A controlled funk-size), F2 (Stage B epoch sensitivity)
checkpoints and writes per-cell val/test top-1, top-5, val_loss CSVs.

Outputs:
    results/control_token_baseline.csv  (E1, 33 cells × 3 cols)
    results/phase0_full_ft.csv          (E2, 11 cells)
    results/phase0_lora.csv             (E3, 11 cells)
    results/controlled_funk_size.csv    (F1, 40 cells)
    results/epoch_sensitivity.csv       (F2, 24 cells)

Reads val_loss from each cell's training log (last "Done. Best val_loss"
line) rather than re-running forward inference, since each cell's
training already produced this metric on its val split.
"""
from __future__ import annotations
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"
RESULTS.mkdir(parents=True, exist_ok=True)

GENRES = ["blues", "bossa", "classical", "country", "electronic", "folk",
          "funk", "gospel", "hip_hop", "rnb_soul", "rock"]
GENRES_FUNK_SIZED = [g for g in GENRES if g != "classical"]
GENRES_DATA_RICH = ["rock", "country", "folk"]


def _last_metrics(log_pattern: str) -> dict:
    """Find latest log matching pattern, extract last 'Done. Best val_loss' + last 'Epoch 7/8' top1/top5."""
    matches = sorted(LOGS.glob(log_pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        return {}
    log = matches[0]
    txt = log.read_text()
    out = {"log_file": log.name}

    # Best val_loss from "Done." line
    m = re.search(r"Done\. Best val_loss=([0-9.]+)", txt)
    if m:
        out["val_loss"] = float(m.group(1))

    # Top-1 / top-5 from the LAST "Epoch X/Y" line. Y varies by config:
    # F1 Stage A = 5 epochs, F2 Stage B = {3, 5, 8, 12}, E1/E2/E3 = 8.
    epoch_lines = re.findall(
        r"Epoch +(\d+)/(\d+) \| train=([0-9.]+) val=([0-9.]+) top1=([0-9.]+)% top5=([0-9.]+)%", txt)
    if epoch_lines:
        ep, total_ep, tr, va, t1, t5 = epoch_lines[-1]
        out["epoch_reached"] = int(ep)
        out["epochs_configured"] = int(total_ep)
        out["train_loss"] = float(tr)
        out["val_loss_at_ep"] = float(va)
        out["top1"] = float(t1)
        out["top5"] = float(t5)

    return out


def _write_csv(rows: list[dict], out: Path):
    if not rows:
        out.write_text("(no cells found)\n")
        return
    cols = sorted({k for r in rows for k in r.keys()})
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  Wrote {out} ({len(rows)} rows)")


def main():
    # ---- E1 control-token ----
    rows = []
    for genre in GENRES:
        for seed in (42, 123, 7):
            run_name = f"ft_f1_control_token_{genre}_seed{seed}"
            m = _last_metrics(f"{run_name}_2026*.log")
            m.update({"cell": run_name, "method": "control_token",
                      "genre": genre, "seed": seed})
            rows.append(m)
    _write_csv(rows, RESULTS / "control_token_baseline.csv")

    # ---- E2 Phase 0 + Full FT ----
    rows = []
    for genre in GENRES:
        run_name = f"phase0_full_ft_{genre}"
        m = _last_metrics(f"{run_name}_2026*.log")
        m.update({"cell": run_name, "method": "phase0_full_ft",
                  "genre": genre, "seed": 42})
        rows.append(m)
    _write_csv(rows, RESULTS / "phase0_full_ft.csv")

    # ---- E3 Phase 0 + LoRA ----
    rows = []
    for genre in GENRES:
        run_name = f"phase0_lora_{genre}"
        m = _last_metrics(f"{run_name}_2026*.log")
        m.update({"cell": run_name, "method": "phase0_lora",
                  "genre": genre, "seed": 42})
        rows.append(m)
    _write_csv(rows, RESULTS / "phase0_lora.csv")

    # ---- F1 Stage A controlled funk-size ----
    # SKIPPED here: Stage A cells need test eval on the ORIGINAL full test
    # set (not sub-sampled). That's done by
    # eval_stage_a_on_original_test.py (P14c in phase3 chain). Writing
    # sub-sampled val metrics here would be misleading.
    pass

    # ---- F2 Stage B epoch sensitivity ----
    rows = []
    for genre in GENRES_DATA_RICH:
        for method in ("lora", "full"):
            for n_ep in (3, 5, 8, 12):
                run_name = f"stageB_{method}_{genre}_ep{n_ep}"
                m = _last_metrics(f"{run_name}_2026*.log")
                m.update({"cell": run_name, "method": f"stageB_{method}",
                          "genre": genre, "seed": 42, "epochs": n_ep})
                rows.append(m)
    _write_csv(rows, RESULTS / "epoch_sensitivity.csv")


if __name__ == "__main__":
    main()
