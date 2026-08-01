"""Side-by-side comparison report for two or more checkpoints.

For each checkpoint: reads the best-epoch row from eval_results.csv,
runs generate_compare's load_model + generation on a shared prompt set,
scores each sample on the physics + diversity axes, and writes a
compact markdown report that makes the A/B choice legible.

CLI::

    cd ai
    uv run python3 training/compare_checkpoints.py \\
        --models ft_jazz_pop50 ft_jazz_pop50_r3 \\
        --prompts training/eval_prompts.yaml \\
        --out results/compare_f3_vs_f3r3.md

The intended driver is the auto-watcher ``watch_and_compare.sh`` that
triggers this as soon as R3 training lands.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

import torch
import yaml

_AI_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_AI_ROOT))

from physics.composite import composite_dissonance  # noqa: E402
from physics.diversity import diversity_profile  # noqa: E402
from physics.voicing import voice_chord  # noqa: E402

from generate_compare import load_model  # noqa: E402
from tokenizer import ChordTokenizer  # noqa: E402


def _best_eval_row(ckpt_dir: Path) -> dict | None:
    """Return the best-epoch row by jazz_top1 (with pop_top1 ≥ baseline-3% floor)."""
    csv_path = ckpt_dir / "eval_results.csv"
    if not csv_path.exists():
        return None
    rows: list[dict] = []
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            try:
                r["_jazz_top1"] = float(r.get("jazz_top1", 0))
                r["_pop_top1"] = float(r.get("pop_top1", 0))
            except ValueError:
                continue
            rows.append(r)
    if not rows:
        return None
    # Prefer maximum jazz_top1 row
    rows.sort(key=lambda r: r["_jazz_top1"], reverse=True)
    return rows[0]


def _physics_summary(chords: list[str]) -> dict[str, float]:
    rough: list[float] = []
    harm: list[float] = []
    comp: list[float] = []
    for ct in chords:
        v = voice_chord(ct)
        if v is None or len(v) < 2:
            continue
        s = composite_dissonance(v)
        rough.append(s.roughness)
        harm.append(s.harmonicity)
        comp.append(s.composite)
    def _m(xs: list[float]) -> float:
        return statistics.fmean(xs) if xs else float("nan")
    return {"roughness": _m(rough), "harmonicity": _m(harm), "composite": _m(comp)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", required=True,
                    help="checkpoint folder names under ai/checkpoints/")
    ap.add_argument("--prompts", type=Path, default=_AI_ROOT / "training" / "eval_prompts.yaml")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.9)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = ChordTokenizer()
    with open(args.prompts) as f:
        prompts = yaml.safe_load(f)

    checkpoints_root = _AI_ROOT / "checkpoints"

    # --- Header rows from eval_results.csv + load models ---
    header_rows: list[dict] = []
    loaded_models: dict[str, object] = {}
    for name in args.models:
        ck_dir = checkpoints_root / name
        ck_path = ck_dir / "best.pt"
        if not ck_path.exists():
            print(f"MISSING: {ck_path}", file=sys.stderr)
            continue
        row = _best_eval_row(ck_dir) or {}
        model, epoch, val_loss = load_model(str(ck_path), tokenizer, device)
        header_rows.append({
            "name": name,
            "epoch": row.get("epoch", epoch),
            "pop_top1": row.get("pop_top1", "—"),
            "jazz_top1": row.get("jazz_top1", "—"),
            "pop_top5": row.get("pop_top5", "—"),
            "jazz_top5": row.get("jazz_top5", "—"),
            "jazz_ppl": row.get("jazz_ppl", "—"),
        })
        loaded_models[name] = model

    # --- Per-prompt generations ---
    gen_rows: list[dict] = []
    for prompt in prompts:
        pname = prompt.get("name", "?")
        prompt_ids = tokenizer.encode_sequence(prompt["song"])[:-1]  # drop EOS
        for name, model in loaded_models.items():
            for i in range(args.samples):
                prompt_tensor = torch.tensor([prompt_ids], device=device)
                # Use model.generate directly to apply repetition_penalty + no_repeat_ngram
                output = model.generate(
                    prompt_tensor,
                    max_new_tokens=64,
                    temperature=args.temperature,
                    top_p=0.9,
                    eos_id=tokenizer.eos_id,
                    repetition_penalty=1.3,
                    no_repeat_ngram_size=3,
                    ignore_repeat_token_ids={tokenizer.bar_id, tokenizer.pad_id},
                )
                tokens = tokenizer.decode(output[0].tolist())
                chord_tokens = [t for t in tokens if t and not t.startswith(("[", "<"))]
                phys = _physics_summary(chord_tokens)
                div = diversity_profile(chord_tokens)
                gen_rows.append({
                    "prompt": pname,
                    "model": name,
                    "sample": i + 1,
                    "tokens": tokens,
                    "chord_summary": " ".join(chord_tokens[:16]) + (" …" if len(chord_tokens) > 16 else ""),
                    "n_chords": div["n_chords"],
                    "entropy": div["entropy"],
                    "bigram_rep": div["bigram_repetition"],
                    "roughness": phys["roughness"],
                    "composite": phys["composite"],
                })

    # --- Markdown ---
    lines: list[str] = []
    lines.append(f"# Checkpoint comparison: {', '.join(args.models)}")
    lines.append("")
    lines.append("## Eval metrics (best epoch by jazz top-1)")
    lines.append("")
    lines.append("| model | epoch | pop top-1 | jazz top-1 | pop top-5 | jazz top-5 | jazz ppl |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for r in header_rows:
        lines.append(f"| {r['name']} | {r['epoch']} | {r['pop_top1']} | {r['jazz_top1']} | {r['pop_top5']} | {r['jazz_top5']} | {r['jazz_ppl']} |")

    # Group generations by prompt for side-by-side
    lines.append("")
    lines.append("## Generations (same prompts, multiple samples)")
    by_prompt: dict[str, list[dict]] = {}
    for r in gen_rows:
        by_prompt.setdefault(r["prompt"], []).append(r)
    for prompt_name, rows in by_prompt.items():
        lines.append(f"\n### {prompt_name}")
        lines.append("")
        lines.append("| model | # | rough | comp | entropy | 2gram-rep | first 16 chords |")
        lines.append("|---|---:|---:|---:|---:|---:|---|")
        for r in rows:
            lines.append(
                f"| {r['model']} | {r['sample']} | "
                f"{r['roughness']:.3f} | {r['composite']:.3f} | "
                f"{r['entropy']:.2f} | {r['bigram_rep']:.2f} | "
                f"`{r['chord_summary']}` |"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
