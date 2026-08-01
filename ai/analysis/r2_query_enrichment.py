"""R2 retrieval-query enrichment experiment (follow-up to noise audit §1).

The production template ("jazz harmony progression: {a} resolving to {b},
voice leading and functional analysis") is dominated by its fixed wording —
every chord pair retrieves the same Berklee chunk. This experiment tests a
chord-functional enrichment: the query embeds the root-motion interval and the
two chord qualities as words, giving the embedder discriminative content.

Two measurements, old vs enriched template:
  A. Retrieval diversity over the 100 audited explanation transitions:
     distinct top-1 chunks, share of the modal chunk, OMT-collection
     surfacing rate (classical transitions should reach OMT).
  B. Selection impact on a seeded sample of RAG-active humaneval contexts:
     re-score each pool candidate's R2 (mean retrieval distance over its
     transitions) under both templates and count how often the C2 pick
     (composite r1+r2) changes.

NO production change: rag/explain.py & rag/coherence.py keep the old template
until this experiment justifies adoption (then exp2 must be re-run).

Output: results/r2_query_enrichment_20260612.csv + .md
Usage (from ai/): uv run python -m analysis.r2_query_enrichment
"""
from __future__ import annotations

import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

AI_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AI_ROOT))
sys.path.insert(0, str(AI_ROOT / "training"))

from rag.retrieve import Retriever  # noqa: E402
from tokenizer import ChordTokenizer  # noqa: E402
from humaneval._util import split_norm_token  # noqa: E402

RESULTS = AI_ROOT / "results"
STAMP = "20260612"
PC = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5,
      "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10,
      "Bb": 10, "B": 11}
INTERVAL = {0: "static root", 1: "half-step up", 2: "whole-step up",
            3: "minor-third up", 4: "major-third up", 5: "fourth up (fifth down)",
            6: "tritone", 7: "fifth up (fourth down)", 8: "minor-sixth up",
            9: "major-sixth up", 10: "minor-seventh up", 11: "half-step down"}
N_CONTEXTS = 50


def old_query(a: str, b: str) -> str:
    return (f"jazz harmony progression: {a} resolving to {b}, "
            f"voice leading and functional analysis")


def enriched_query(a: str, b: str) -> str:
    na, nb = ChordTokenizer.normalize_chord(a), ChordTokenizer.normalize_chord(b)
    if not (na and nb):
        return old_query(a, b)
    ra, qa = split_norm_token(na)
    rb, qb = split_norm_token(nb)
    iv = INTERVAL.get((PC.get(rb, 0) - PC.get(ra, 0)) % 12, "")
    return (f"{a} to {b} chord progression: root motion {iv}, "
            f"{qa or 'major'} chord moving to {qb or 'major'} chord, "
            f"harmonic function and resolution")


def transitions(bars) -> list[tuple[str, str]]:
    flat = [c for bar in bars or [] for c in bar or [] if c]
    return [(x, y) for x, y in zip(flat, flat[1:]) if x != y]


def main() -> None:
    retriever = Retriever(collections=["berklee", "openmusictheory"])
    cache: dict[str, tuple[str, float]] = {}

    def top1(query: str) -> tuple[str, float]:
        if query not in cache:
            hits = retriever.query(query, k=1)
            cache[query] = ((f"{hits[0].chapter}|{hits[0].section}|p{hits[0].page_start}",
                             float(hits[0].distance)) if hits else ("none", 1.0))
        return cache[query]

    # ---- A. diversity over the audited 100 transitions ----
    pairs = []
    with open(RESULTS / "outputs" / "r2_explanations_20260612.jsonl") as f:
        for line in f:
            r = json.loads(line)
            pairs.append((r["chord_a"], r["chord_b"], r["genre"]))

    div_rows = []
    for tmpl_name, fn in [("old", old_query), ("enriched", enriched_query)]:
        chunks = Counter()
        omt = 0
        for a, b, genre in pairs:
            chunk, _ = top1(fn(a, b))
            chunks[chunk] += 1
            if "openmusictheory" in chunk.lower() or chunk.startswith("|"):
                omt += 1
        # OMT chunks have empty berklee-style chapter; detect via known marker:
        div_rows.append({
            "template": tmpl_name, "n": len(pairs),
            "distinct_top1": len(chunks),
            "modal_share_pct": round(100 * chunks.most_common(1)[0][1] / len(pairs), 1),
            "modal_chunk": chunks.most_common(1)[0][0][:60],
        })
        print(f"A[{tmpl_name}] distinct={len(chunks)} modal={div_rows[-1]['modal_share_pct']}% "
              f"({div_rows[-1]['modal_chunk']})")

    # ---- B. C2-pick impact on RAG-active contexts ----
    rng = random.Random(42)
    ctxs = []
    with open(AI_ROOT / "humaneval" / "candidates_full_20260611.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if not r.get("excluded") and r.get("rag_active"):
                ctxs.append(r)
    sample = rng.sample(ctxs, min(N_CONTEXTS, len(ctxs)))

    changed = same = 0
    for r in sample:
        scores = {}
        for tmpl_name, fn in [("old", old_query), ("enriched", enriched_query)]:
            r2s = []
            for cand in r["pool"]:
                tr = transitions(cand["bars"])
                if tr:
                    r2s.append(sum(top1(fn(a, b))[1] for a, b in tr) / len(tr))
                else:
                    r2s.append(1.0)
            # composite mirrors serving: alpha*r1 + beta*r2 (alpha=beta=1)
            comp = [c["r1"] + r2s[i] for i, c in enumerate(r["pool"])]
            scores[tmpl_name] = min(range(len(comp)), key=lambda i: comp[i])
        if scores["old"] != scores["enriched"]:
            changed += 1
        else:
            same += 1
    pick_change_pct = round(100 * changed / (changed + same), 1)
    print(f"B: C2 pick changes in {changed}/{changed + same} RAG-active contexts "
          f"({pick_change_pct}%)")

    out = RESULTS / f"r2_query_enrichment_{STAMP}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["measure", "template", "value", "detail"])
        for d in div_rows:
            w.writerow(["top1_diversity", d["template"], d["distinct_top1"],
                        f"modal {d['modal_share_pct']}% = {d['modal_chunk']}"])
        w.writerow(["c2_pick_change_pct", "enriched_vs_old", pick_change_pct,
                    f"n={changed + same} RAG-active contexts, seed 42"])

    old_d, new_d = div_rows[0], div_rows[1]
    (RESULTS / f"r2_query_enrichment_{STAMP}.md").write_text(
        f"# R2 query enrichment experiment ({STAMP})\n\n"
        f"Old template: {old_d['distinct_top1']} distinct top-1 chunks over "
        f"{old_d['n']} transitions (modal chunk {old_d['modal_share_pct']}%).\n"
        f"Enriched template: {new_d['distinct_top1']} distinct "
        f"(modal {new_d['modal_share_pct']}%).\n\n"
        f"C2-pick impact: the enriched template changes the composite pick in "
        f"**{pick_change_pct}%** of {changed + same} sampled RAG-active contexts.\n\n"
        "Production unchanged pending this review; adopting the enriched "
        "template requires re-running exp2 as a new condition "
        "(`analysis/exp2_condition_metrics.py`). Script: "
        "`analysis/r2_query_enrichment.py`.\n")
    print(f"Wrote {out.name} (+ .md)")


if __name__ == "__main__":
    main()
