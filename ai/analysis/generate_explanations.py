"""Generate R2 theory explanations for the hallucination audit (§4B/§9).

Produces the input that `analysis/hallucination_check.py` verifies: real
production-path explanations (rag/explain.py → Retriever over berklee+OMT →
ollama LLM judge) over held-out chord transitions from the 130-song eval set,
restricted to RAG-enabled genres (jazz family + classical — elsewhere R2 is
gated off in serving).

Sampling: per song, consecutive chord-pair transitions from the flattened
bars; seeded subsample to --n-transitions total, balanced jazz vs classical.

Output: results/outputs/r2_explanations_{stamp}.jsonl — one Explanation per
line (production schema: chord_a/chord_b/concept/explanation/passage_used/
top1_chapter/top1_section/top1_page_start/top1_page_end/top1_similarity,
plus provenance meta fields song_id/genre).

Usage (from ai/, with `ollama serve` running and gemma3:4b pulled):
    uv run python -m analysis.generate_explanations --n-transitions 100
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import random
import sys
import time
from pathlib import Path

AI_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AI_ROOT))

from openai import OpenAI  # noqa: E402

from rag.explain import explain_transition  # noqa: E402
from rag.retrieve import Retriever  # noqa: E402

EVAL_JSONL = AI_ROOT / "data" / "eval_real_songs.jsonl"
OUT_DIR = AI_ROOT / "results" / "outputs"
RAG_GENRES = {"jazz": 60, "classical": 40}  # transition budget per genre (scaled to --n-transitions)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-transitions", type=int, default=100)
    ap.add_argument("--model", default="gemma3:4b")
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stamp", default="20260612")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    pool: dict[str, list[dict]] = {g: [] for g in RAG_GENRES}
    with open(EVAL_JSONL) as f:
        for line in f:
            song = json.loads(line)
            if song["genre"] not in RAG_GENRES:
                continue
            flat = [c for bar in song["bars"] for c in bar if c]
            for a, b in zip(flat, flat[1:]):
                if a != b:
                    pool[song["genre"]].append(
                        {"chord_a": a, "chord_b": b,
                         "song_id": song["id"], "genre": song["genre"]})

    scale = args.n_transitions / sum(RAG_GENRES.values())
    picks: list[dict] = []
    for g, budget in RAG_GENRES.items():
        n = min(len(pool[g]), max(1, round(budget * scale)))
        picks += rng.sample(pool[g], n)
        print(f"{g}: {len(pool[g])} transitions in pool, sampled {n}")

    retriever = Retriever(collections=["berklee", "openmusictheory"])
    client = OpenAI(base_url=args.base_url, api_key="ollama")

    out_path = OUT_DIR / f"r2_explanations_{args.stamp}.jsonl"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    with open(out_path, "w") as out:
        for i, p in enumerate(picks, 1):
            t0 = time.perf_counter()
            e = explain_transition(p["chord_a"], p["chord_b"], retriever,
                                   client, args.model)
            row = dataclasses.asdict(e)
            row["song_id"], row["genre"] = p["song_id"], p["genre"]
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(f"  [{i}/{len(picks)}] {p['chord_a']} -> {p['chord_b']} "
                  f"({e.concept}) {time.perf_counter() - t0:.1f}s", flush=True)
    dt = time.perf_counter() - t_start
    print(f"\nWrote {len(picks)} explanations -> {out_path} ({dt/60:.1f} min total)")


if __name__ == "__main__":
    main()
