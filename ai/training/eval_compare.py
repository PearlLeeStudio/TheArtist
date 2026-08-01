"""Multi-axis evaluation over generation_comparison.csv.

Reads the existing per-checkpoint generation dump, extracts chord tokens
from each sample, and scores every sample on three axes:

    - physics:   composite (Sethares roughness + Stolzenburg harmonicity)
    - diversity: chord entropy, unique ratio, n-gram repetition
    - theory:    (optional) theory-coherence via `rag.coherence`
                 — calls a local LLM per transition; slow; cached.

Emits four artefacts under ``ai/results/``::

    eval_long.csv      per-sample + (optional) per-transition rows
    eval_summary.csv   per-(prompt, experiment) aggregated axes
    eval_pairwise.csv  per-prompt pairwise deltas across experiments
    eval_report.md     human-readable summary

Physics + diversity pass is fast (seconds). Enable the theory axis only
when you have the time — LLM calls are cached by (chord_a, chord_b) in
``ai/data/cache/coherence_cache.jsonl`` so reruns are cheap.

CLI::

    cd ai
    uv run python -m training.eval_compare                       # physics + diversity
    uv run python -m training.eval_compare --with-theory         # +coherence (slow)
    uv run python -m training.eval_compare --with-theory --limit-groups 5   # smoke
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import logging
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

_AI_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_AI_ROOT))

from physics.composite import composite_dissonance  # noqa: E402
from physics.diversity import diversity_profile  # noqa: E402
from physics.voicing import voice_chord  # noqa: E402

log = logging.getLogger(__name__)

_DEFAULT_INPUT = _AI_ROOT / "results" / "generation_comparison.csv"
_DEFAULT_OUTDIR = _AI_ROOT / "results"
_COHERENCE_CACHE = _AI_ROOT / "data" / "cache" / "coherence_cache.jsonl"


def _extract_chords(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t and not t.startswith(("[", "<"))]


def _physics_mean(chords: list[str]) -> dict[str, float]:
    """Mean composite / roughness / harmonicity across chords in a sequence."""
    rs: list[float] = []
    hs: list[float] = []
    cs: list[float] = []
    for ct in chords:
        v = voice_chord(ct)
        if v is None or len(v) < 2:
            continue
        s = composite_dissonance(v)
        rs.append(s.roughness)
        hs.append(s.harmonicity)
        cs.append(s.composite)
    def _m(xs: list[float]) -> float:
        return statistics.fmean(xs) if xs else float("nan")
    return {
        "roughness_mean":   _m(rs),
        "harmonicity_mean": _m(hs),
        "composite_mean":   _m(cs),
        "n_scored_chords":  len(cs),
    }


def _load_coherence_cache(path: Path) -> dict[tuple[str, str], dict]:
    if not path.exists():
        return {}
    out: dict[tuple[str, str], dict] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        out[(rec["chord_a"], rec["chord_b"])] = rec
    return out


def _append_coherence_cache(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _coherence_for_sequence(
    chords: list[str], retriever, llm_client, llm_model: str,
    cache: dict[tuple[str, str], dict], cache_path: Path,
) -> dict[str, float | list[dict]]:
    """Score consecutive (a, b) pairs via rag.coherence, with on-disk cache."""
    from rag.coherence import score_transition  # deferred import

    per_trans: list[dict] = []
    sims: list[float] = []
    yeses = 0
    for a, b in zip(chords, chords[1:]):
        key = (a, b)
        if key in cache:
            r = cache[key]
        else:
            ts = score_transition(a, b, retriever, llm_client, llm_model=llm_model)
            r = asdict(ts)
            cache[key] = r
            _append_coherence_cache(cache_path, r)
        per_trans.append(r)
        sims.append(float(r["top1_similarity"]))
        yeses += 1 if r["llm_justified"] else 0
    n = len(per_trans)
    return {
        "theory_mean_similarity": (sum(sims) / n) if n else float("nan"),
        "theory_justify_rate":    (yeses / n) if n else float("nan"),
        "theory_combined":        ((sum(sims) / n) * 0.5 + (yeses / n) * 0.5) if n else float("nan"),
        "n_transitions":          n,
        "per_transition":         per_trans,
    }


def _load_generations(path: Path, limit_groups: int | None = None) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        rows.extend(csv.DictReader(f))
    if limit_groups is None:
        return rows
    seen_groups: set[tuple[str, str, str]] = set()
    kept: list[dict] = []
    for r in rows:
        g = (r["prompt"], r["experiment"], r["epoch"])
        if g not in seen_groups:
            if len(seen_groups) >= limit_groups:
                continue
            seen_groups.add(g)
        if g in seen_groups:
            kept.append(r)
    return kept


def _write_long_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _write_summary_csv(path: Path, rows: list[dict]) -> None:
    _write_long_csv(path, rows)


def _compute_pairwise(summary: list[dict]) -> list[dict]:
    by_prompt: dict[str, list[dict]] = defaultdict(list)
    for r in summary:
        by_prompt[r["prompt"]].append(r)
    out: list[dict] = []
    keys = ["composite_mean", "roughness_mean", "entropy",
            "theory_mean_similarity", "theory_justify_rate", "theory_combined"]
    for prompt, rows in by_prompt.items():
        # aggregate per experiment (mean across samples if multiple)
        by_exp: dict[str, dict[str, float]] = {}
        for r in rows:
            e = r["experiment"]
            by_exp.setdefault(e, {k: [] for k in keys})
            for k in keys:
                v = r.get(k)
                if v not in (None, "", float("nan")) and isinstance(v, (int, float)) and v == v:
                    by_exp[e][k].append(v)
        exp_means = {
            e: {k: (statistics.fmean(vs) if vs else float("nan")) for k, vs in lists.items()}
            for e, lists in by_exp.items()
        }
        exps = sorted(exp_means)
        for i, a in enumerate(exps):
            for b in exps[i + 1:]:
                row = {"prompt": prompt, "experiment_a": a, "experiment_b": b}
                for k in keys:
                    va, vb = exp_means[a].get(k, float("nan")), exp_means[b].get(k, float("nan"))
                    if va == va and vb == vb:
                        row[f"delta_{k}"] = vb - va
                    else:
                        row[f"delta_{k}"] = float("nan")
                out.append(row)
    return out


def _write_report(path: Path, summary: list[dict], pairwise: list[dict], with_theory: bool) -> None:
    lines: list[str] = []
    lines.append("# Evaluation Report")
    lines.append("")
    lines.append("Multi-axis evaluation over `generation_comparison.csv` samples.")
    lines.append("")
    lines.append("## Per-experiment medians across prompts")
    lines.append("")
    by_exp: dict[str, list[dict]] = defaultdict(list)
    for r in summary:
        by_exp[r["experiment"]].append(r)
    metric_keys = ["composite_mean", "roughness_mean", "harmonicity_mean",
                   "entropy", "bigram_repetition"]
    if with_theory:
        metric_keys += ["theory_mean_similarity", "theory_justify_rate", "theory_combined"]
    header = "| experiment | " + " | ".join(metric_keys) + " |"
    sep = "|" + "|".join(["---"] * (len(metric_keys) + 1)) + "|"
    lines.append(header)
    lines.append(sep)
    for exp in sorted(by_exp):
        row = [exp]
        for k in metric_keys:
            vals = [r.get(k) for r in by_exp[exp]
                    if r.get(k) not in (None, "") and isinstance(r.get(k), (int, float))
                    and r.get(k) == r.get(k)]
            row.append(f"{statistics.median(vals):.3f}" if vals else "—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Notable pairwise deltas (theory_combined if present, else composite_mean)")
    lines.append("")
    prim = "delta_theory_combined" if with_theory else "delta_composite_mean"
    ranked = [p for p in pairwise if isinstance(p.get(prim), (int, float)) and p[prim] == p[prim]]
    ranked.sort(key=lambda p: abs(p[prim]), reverse=True)
    for p in ranked[:10]:
        lines.append(
            f"- **{p['prompt']}**  {p['experiment_a']} vs {p['experiment_b']}  "
            f"{prim} = {p[prim]:+.3f}"
        )
    lines.append("")
    lines.append("See `eval_long.csv`, `eval_summary.csv`, `eval_pairwise.csv` for full detail.")
    path.write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=_DEFAULT_INPUT)
    ap.add_argument("--outdir", type=Path, default=_DEFAULT_OUTDIR)
    ap.add_argument("--with-theory", action="store_true",
                    help="also score the theory axis via rag.coherence (slow; cached)")
    ap.add_argument("--limit-groups", type=int, default=None,
                    help="keep first N (prompt, experiment, epoch) groups for smoke test")
    ap.add_argument("--collection", default="berklee")
    ap.add_argument("--llm-base-url", default="http://localhost:11434/v1")
    ap.add_argument("--llm-model", default="gemma3:12b")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.input.exists():
        raise FileNotFoundError(args.input)

    rows = _load_generations(args.input, args.limit_groups)
    log.info("loaded %d sample rows from %s", len(rows), args.input)

    retriever = None
    llm_client = None
    cache: dict[tuple[str, str], dict] = {}
    if args.with_theory:
        from openai import OpenAI
        from rag.retrieve import Retriever
        retriever = Retriever(collection=args.collection)
        llm_client = OpenAI(base_url=args.llm_base_url, api_key="ollama")
        cache = _load_coherence_cache(_COHERENCE_CACHE)
        log.info("theory axis enabled; cache size %d", len(cache))

    long_rows: list[dict] = []
    summary_rows: list[dict] = []
    t_start = time.perf_counter()
    for idx, r in enumerate(rows, start=1):
        tokens = ast.literal_eval(r["generated_text"])
        chords = _extract_chords(tokens)
        phys = _physics_mean(chords)
        div = diversity_profile(chords)
        base = {
            "prompt": r["prompt"],
            "experiment": r["experiment"],
            "epoch": r["epoch"],
            "sample": int(r["sample"]),
            **phys,
            "entropy": div["entropy"],
            "unique_ratio": div["unique_ratio"],
            "bigram_repetition": div["bigram_repetition"],
            "trigram_repetition": div["trigram_repetition"],
            "n_chords": div["n_chords"],
        }
        if args.with_theory and retriever is not None and llm_client is not None:
            theo = _coherence_for_sequence(
                chords, retriever, llm_client, args.llm_model, cache, _COHERENCE_CACHE,
            )
            base["theory_mean_similarity"] = theo["theory_mean_similarity"]
            base["theory_justify_rate"]    = theo["theory_justify_rate"]
            base["theory_combined"]        = theo["theory_combined"]
            base["n_transitions"]          = theo["n_transitions"]
            for t_idx, pt in enumerate(theo["per_transition"]):  # type: ignore[union-attr]
                long_rows.append({
                    **{k: base[k] for k in ("prompt", "experiment", "epoch", "sample")},
                    "transition_idx": t_idx,
                    **pt,
                })
        summary_rows.append(base)
        if idx % 5 == 0 or idx == len(rows):
            log.info("  %d/%d samples scored (%.1fs)", idx, len(rows), time.perf_counter() - t_start)

    args.outdir.mkdir(parents=True, exist_ok=True)
    _write_summary_csv(args.outdir / "eval_summary.csv", summary_rows)
    if long_rows:
        _write_long_csv(args.outdir / "eval_long.csv", long_rows)
    pairwise = _compute_pairwise(summary_rows)
    _write_long_csv(args.outdir / "eval_pairwise.csv", pairwise)
    _write_report(args.outdir / "eval_report.md", summary_rows, pairwise, args.with_theory)

    log.info("wrote %d summary rows, %d long rows, %d pairwise rows to %s",
             len(summary_rows), len(long_rows), len(pairwise), args.outdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
