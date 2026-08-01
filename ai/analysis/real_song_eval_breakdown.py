"""Real-song eval breakdown — turn ai/results/real_song_eval.csv into the
paper-ready summary tables that PAPER.md §6.2 currently defers.

Inputs:
  ai/results/real_song_eval.csv   (2,210 rows = 18 models × 13 genres × 10 songs)

Outputs:
  ai/results/real_song_breakdown.csv     — per (genre, model) mean/std top-1, n
  ai/results/real_song_target_lora.csv   — per genre: F1 vs the target LoRA, win rate
  ai/results/real_song_breakdown.md      — markdown summary for the paper
"""

from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
EVAL_CSV = PROJECT_DIR / "results" / "real_song_eval.csv"
OUT_BREAKDOWN_CSV = PROJECT_DIR / "results" / "real_song_breakdown.csv"
OUT_TARGET_CSV = PROJECT_DIR / "results" / "real_song_target_lora.csv"
OUT_MD = PROJECT_DIR / "results" / "real_song_breakdown.md"

# Genres the paper covers in the LoRA expansion. Pop / jazz use the F1 base
# directly, so the "target LoRA" for those is F1 itself.
ALL_GENRES = [
    "blues", "bossa", "classical", "country", "electronic",
    "folk", "funk", "gospel", "hip_hop", "jazz", "pop", "rnb_soul", "rock",
]
LORA_GENRES = [g for g in ALL_GENRES if g not in {"jazz", "pop"}]
F_SERIES = ["Phase0", "F1", "F2", "F3", "F4", "F5"]


def _model_for_genre(genre: str) -> str:
    if genre == "pop":
        return "F1"
    if genre == "jazz":
        return "F4"
    return f"LoRA-{genre}"


def _split_of(song_id: str) -> str:
    if "__test__" in song_id:
        return "test"
    if "__val__" in song_id:
        return "val"
    return "other"


def main() -> None:
    rows: list[dict] = []
    with open(EVAL_CSV) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    print(f"loaded {len(rows)} rows from {EVAL_CSV.name}")

    # --- Aggregate (genre, model) --------------------------------------------
    by_gm: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        g = r["genre"]
        m = r["model"]
        if g not in ALL_GENRES:
            continue
        try:
            top1 = float(r["val_top1"])
        except ValueError:
            continue
        by_gm[(g, m)].append(top1)

    breakdown_rows: list[dict] = []
    for (g, m), vals in sorted(by_gm.items()):
        if not vals:
            continue
        mean = statistics.mean(vals)
        sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        breakdown_rows.append({
            "genre": g,
            "model": m,
            "n": len(vals),
            "top1_mean": round(mean, 2),
            "top1_std": round(sd, 2),
            "top1_min": round(min(vals), 2),
            "top1_max": round(max(vals), 2),
        })
    with open(OUT_BREAKDOWN_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(breakdown_rows[0].keys()))
        w.writeheader()
        w.writerows(breakdown_rows)
    print(f"wrote {OUT_BREAKDOWN_CSV.name} ({len(breakdown_rows)} rows)")

    # --- Per-genre target-LoRA vs F1 paired comparison -----------------------
    songs_per_genre: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        g = r["genre"]
        if g in ALL_GENRES:
            songs_per_genre[g].append(r)

    target_rows: list[dict] = []
    for g in ALL_GENRES:
        target_model = _model_for_genre(g)
        # Per song id, gather F1 top1 and target_model top1 if present.
        by_song: dict[str, dict[str, float]] = defaultdict(dict)
        by_song_split: dict[str, str] = {}
        for r in songs_per_genre[g]:
            sid = r["song_id"]
            try:
                v = float(r["val_top1"])
            except ValueError:
                continue
            by_song[sid][r["model"]] = v
            by_song_split[sid] = _split_of(sid)

        f1_vals, tgt_vals, deltas = [], [], []
        wins = ties = losses = 0
        val_n = test_n = 0
        for sid, mvals in by_song.items():
            f1 = mvals.get("F1")
            tgt = mvals.get(target_model)
            if f1 is None or tgt is None:
                continue
            f1_vals.append(f1)
            tgt_vals.append(tgt)
            d = tgt - f1
            deltas.append(d)
            if d > 0.0:
                wins += 1
            elif d < 0.0:
                losses += 1
            else:
                ties += 1
            if by_song_split[sid] == "test":
                test_n += 1
            elif by_song_split[sid] == "val":
                val_n += 1

        if not f1_vals:
            continue
        mean_f1 = statistics.mean(f1_vals)
        mean_tgt = statistics.mean(tgt_vals)
        mean_delta = statistics.mean(deltas)
        sd_delta = statistics.pstdev(deltas) if len(deltas) > 1 else 0.0
        n = len(deltas)
        # Paired t-statistic for descriptive use only (small n).
        if n > 1 and sd_delta > 0:
            t_stat = mean_delta / (sd_delta / math.sqrt(n))
        else:
            t_stat = float("nan")

        target_rows.append({
            "genre": g,
            "target_model": target_model,
            "n_songs": n,
            "n_val": val_n,
            "n_test": test_n,
            "f1_mean_top1": round(mean_f1, 2),
            "target_mean_top1": round(mean_tgt, 2),
            "mean_delta": round(mean_delta, 2),
            "delta_std": round(sd_delta, 2),
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "win_rate_pct": round(wins / n * 100, 1) if n else 0.0,
            "paired_t_descriptive": round(t_stat, 2) if not math.isnan(t_stat) else "",
        })

    with open(OUT_TARGET_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(target_rows[0].keys()))
        w.writeheader()
        w.writerows(target_rows)
    print(f"wrote {OUT_TARGET_CSV.name} ({len(target_rows)} rows)")

    # --- Markdown summary for the paper --------------------------------------
    lines = [
        "# Real-song eval breakdown",
        "",
        "Source: `ai/results/real_song_eval.csv` — 18 models × 13 genres × 10",
        "songs = 2,210 (model, song) cells. Eval split per song is drawn only",
        "from the underlying corpora's val/test partitions (no train leakage),",
        "and chord-rich titles dominate the slice (selection bias).",
        "",
        "## A. Target-LoRA vs F1, paired per song",
        "",
        "For each genre `g`, the *target model* is `F1` for pop, `F4` for jazz,",
        "and `LoRA-g` otherwise. Numbers are paired per song: F1 and the",
        "target model evaluate the same 10-song slice.",
        "",
        "| Genre | Target | n | val/test | F1 top-1 | Target top-1 | Δ (target − F1) | wins / ties / losses | Win rate |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for r in target_rows:
        lines.append(
            f"| {r['genre']} | {r['target_model']} | {r['n_songs']} | "
            f"{r['n_val']}/{r['n_test']} | "
            f"{r['f1_mean_top1']:.2f} | {r['target_mean_top1']:.2f} | "
            f"{r['mean_delta']:+.2f} ± {r['delta_std']:.2f} | "
            f"{r['wins']} / {r['ties']} / {r['losses']} | "
            f"{r['win_rate_pct']:.1f}% |"
        )

    # 11 LoRA target genres aggregate
    lora_rows = [r for r in target_rows if r["genre"] in LORA_GENRES]
    if lora_rows:
        mean_delta = statistics.mean(r["mean_delta"] for r in lora_rows)
        n_genres_pos = sum(1 for r in lora_rows if r["mean_delta"] > 0)
        majority_win = sum(1 for r in lora_rows if r["win_rate_pct"] > 50)
        lines += [
            "",
            f"Across the 11 target-LoRA genres, mean target − F1 = "
            f"{mean_delta:+.2f} pp, target beats F1 on a per-song mean for "
            f"{n_genres_pos}/11 genres, and wins on more than half the songs "
            f"in {majority_win}/11 genres.",
        ]

    # --- B. Per (genre, model) breakdown table --------------------------------
    lines += [
        "",
        "## B. Per-genre breakdown across all 18 models",
        "",
        "Mean top-1 (± std over 10 songs) for every (genre, model) cell.",
        "F-series models in the first six columns, LoRA adapters in the next",
        "11. Bold-style emphasis is intentionally omitted to keep the table",
        "scannable across both LBD and main-paper rendering.",
        "",
    ]

    # Build a wide table: rows = genres, columns = models.
    model_order = F_SERIES + [f"LoRA-{g}" for g in LORA_GENRES]
    header = ["Genre"] + model_order
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] + ["---:"] * len(model_order)) + "|")
    bd: dict[tuple[str, str], dict] = {(r["genre"], r["model"]): r for r in breakdown_rows}
    for g in ALL_GENRES:
        row = [g]
        for m in model_order:
            cell = bd.get((g, m))
            if not cell:
                row.append("--")
            else:
                row.append(f"{cell['top1_mean']:.1f}")
        lines.append("| " + " | ".join(row) + " |")

    # --- C. Caveats ----------------------------------------------------------
    lines += [
        "",
        "## C. Caveats",
        "",
        "- The 130 songs are user-transcribed chord progressions drawn from val",
        "  and test partitions of the underlying corpora. Songs without rich",
        "  chord content are systematically under-represented for the same",
        "  reason discussed in §4.2 (Chordonomicon selection bias).",
        "- Per-song top-1 is a noisy quantity at 10 songs per genre. The",
        "  paired Δ in Table A removes between-song variance, but the absolute",
        "  numbers should be read as model-card sanity rather than primary",
        "  evidence; the per-genre validation split (`§7.1`) and held-out test",
        "  split (`§7.2`) remain the primary tables.",
        "- Pop and jazz use the F1 / F4 base directly, so their `Target` rows",
        "  in Table A are by construction Δ = 0; they are kept for table",
        "  symmetry rather than as separate evidence.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_MD.name} ({sum(1 for _ in lines)} lines)")


if __name__ == "__main__":
    main()
