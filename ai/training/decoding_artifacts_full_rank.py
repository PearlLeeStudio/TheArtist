"""§3.4: Decoding-artifact sweep across all 5 LoRA ranks per genre.

Extends decoding_artifacts.py — that one only samples (F1, validation-selected
LoRA) for each genre. This one sweeps every (genre, rank) cell so the paper
§7.x decoding-artifact table can report whether rank affects artifact rates.

Total: 11 genres × 5 ranks × n_samples generations + 11 F1 baselines. With
defaults (n_samples=30, max_new_tokens=64) this is ~1,800 sampling calls; on
RTX 4070 ~6-8h.

Output:
  ai/results/decoding_artifacts_full_rank.csv
  ai/results/decoding_artifacts_full_rank.md
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from decoding_artifacts import (
    GENRES,
    _analyze_continuation,
    _build_prompt,
    _ckpt_dir_for,
    _generate_batch,
    _load_f1,
    _load_lora,
    _summarize,
)
from tokenizer import ChordTokenizer

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"
RANKS = [4, 8, 16, 32, 64]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    old_tok = ChordTokenizer(include_extra_genres=False)
    new_tok = ChordTokenizer(include_extra_genres=True)

    print("Loading F1 baseline once")
    f1 = _load_f1(old_tok, new_tok, device)

    records: list[dict] = []
    total_cells = len(GENRES) * len(RANKS)
    cell = 0
    for genre in GENRES:
        prompt = _build_prompt(new_tok, genre)
        prompt_len = len(prompt)

        # F1 baseline once per genre (rank-independent)
        f1_seqs = _generate_batch(
            f1, prompt, args.n_samples, args.max_new_tokens,
            new_tok.eos_id, args.temperature, args.top_p, device,
        )
        f1_records = [
            _analyze_continuation(new_tok.decode(ids[prompt_len:])) for ids in f1_seqs
        ]
        f1_summary = _summarize("F1", f1_records)

        for rank in RANKS:
            cell += 1
            ckpt_dir = _ckpt_dir_for(genre, rank)
            if not (ckpt_dir / "adapter" / "adapter_config.json").exists():
                print(f"[{cell}/{total_cells}] {genre} r={rank} SKIP — no adapter at {ckpt_dir}")
                continue
            print(f"[{cell}/{total_cells}] {genre} r={rank} sampling…")

            # Re-emit the F1 row alongside each rank row so the CSV is rectangular.
            records.append({"genre": genre, "rank": rank, **f1_summary})

            lora = _load_lora(genre, rank, old_tok, new_tok, device)
            lora_seqs = _generate_batch(
                lora, prompt, args.n_samples, args.max_new_tokens,
                new_tok.eos_id, args.temperature, args.top_p, device,
            )
            lora_records = [
                _analyze_continuation(new_tok.decode(ids[prompt_len:])) for ids in lora_seqs
            ]
            records.append({"genre": genre, "rank": rank, **_summarize("LoRA", lora_records)})
            del lora
            if device.type == "cuda":
                torch.cuda.empty_cache()

    fieldnames = [
        "genre", "rank", "model", "n_samples",
        "special_leak_rate", "extra_eos_rate",
        "repeat_collapse_rate", "low_diversity_rate", "mean_unique_chords",
    ]
    out_csv = RESULTS_DIR / "decoding_artifacts_full_rank.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(records)
    print(f"Wrote {out_csv}")

    # Markdown summary grouped by (genre, rank) pair, F1 row right above each LoRA row.
    by_cell: dict[tuple[str, int], dict[str, dict]] = {}
    for r in records:
        by_cell.setdefault((r["genre"], r["rank"]), {})[r["model"]] = r

    lines = [
        "# Decoding artifact rates × all 5 LoRA ranks per genre",
        "",
        f"Sampled {args.n_samples} continuations per (genre, rank, model) from a",
        "neutral header prompt (`[BOS] [KEY:Cmaj] [TIME:4/4] [GENRE:g] [BAR]`),",
        f"max_new_tokens={args.max_new_tokens}, temperature={args.temperature},",
        f"top_p={args.top_p}. See `decoding_artifacts.md` for metric definitions.",
        "",
        "| Genre | Rank | Model | Special leak | Extra EOS | Repeat collapse | Low diversity | Mean unique chords |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for g in GENRES:
        for rank in RANKS:
            cell_data = by_cell.get((g, rank))
            if not cell_data:
                continue
            for m in ("F1", "LoRA"):
                r = cell_data.get(m)
                if not r:
                    continue
                lines.append(
                    f"| {g} | {rank} | {m} | {r['special_leak_rate']*100:.1f}% | "
                    f"{r['extra_eos_rate']*100:.1f}% | {r['repeat_collapse_rate']*100:.1f}% | "
                    f"{r['low_diversity_rate']*100:.1f}% | {r['mean_unique_chords']:.1f} |"
                )

    out_md = RESULTS_DIR / "decoding_artifacts_full_rank.md"
    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
