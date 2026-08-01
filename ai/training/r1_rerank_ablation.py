"""R1 ablation: Sethares physics rerank vs greedy baselines (paper §3.2 / §7.x).

For each genre's validation-selected LoRA, we measure next-chord top-1 accuracy
under three decoding conditions:

  - F1 greedy            : argmax from F1 base only (no LoRA, no rerank)
  - LoRA greedy          : argmax from F1 + selected-rank LoRA adapter
  - LoRA + R1 rerank     : sample K continuations from F1+LoRA, rerank by
                           mean Sethares roughness ascending, take #1, score
                           its first chord token against the gold next chord

Per held-out test prompt:
  1. Pick a random 8-chord prefix (drawing a sequence from the genre's test
     loader, slicing chord tokens only, taking a random window of size 8).
  2. Gold = the 9th chord token.
  3. Each of the three conditions produces a predicted "next chord" token.
  4. Record per-prompt hits/misses; aggregate per (genre, condition).

R1 cells whose LoRA adapter is missing are SKIPped, not crashed.

Output:
  ai/results/r1_rerank_ablation.csv   — one row per (genre, condition)
  ai/results/r1_rerank_ablation.md    — markdown table + aggregate footer

Usage (from ai/):
  UV_CACHE_DIR=/tmp/uv-cache uv run python training/r1_rerank_ablation.py
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import torch
from peft import PeftModel  # type: ignore[import-untyped]

# Make `from physics.rerank ...` importable when run as a script from ai/.
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from physics.rerank import mean_roughness  # noqa: E402

from dataset import ChordSequenceDataset  # noqa: E402
from decoding_artifacts import (  # noqa: E402  - reuse loader + sampler
    _ckpt_dir_for,
    _generate_batch,
    _load_f1,
    _load_lora,
    _selected_ranks,
)
from tokenizer import ChordTokenizer  # noqa: E402

RESULTS_DIR = PROJECT_DIR / "results"

GENRES = [
    "blues", "bossa", "classical", "country", "electronic",
    "folk", "funk", "gospel", "hip_hop", "rnb_soul", "rock",
]

PREFIX_LEN = 8  # how many chord tokens of prefix to feed as prompt context


def _build_prompt_from_prefix(
    tok: ChordTokenizer, genre: str, chord_prefix_ids: list[int]
) -> list[int]:
    """Header + N chord tokens. Mirrors decoding_artifacts._build_prompt,
    then appends a [BAR] and the chord prefix so the next predicted token
    is a chord-shaped continuation."""
    out = [tok.bos_id]
    for special in (
        tok.encode_key("Cmaj"),
        tok.encode_time_sig("4/4"),
        tok.encode_genre(genre),
    ):
        if special is not None:
            out.append(special)
    out.append(tok.bar_id)
    out.extend(chord_prefix_ids)
    return out


def _chord_token_id(tid: int, tok: ChordTokenizer) -> bool:
    """True iff `tid` decodes to a chord-shaped token (not a special)."""
    t = tok.id2token.get(int(tid), "")
    return bool(t) and not t.startswith(("[", "<"))


def _extract_chord_id_stream(seq_ids: list[int], tok: ChordTokenizer) -> list[int]:
    """Drop specials/[BAR]/[BOS]/[EOS]/key/time/genre — keep only chord ids."""
    return [int(i) for i in seq_ids if _chord_token_id(i, tok)]


def _first_chord_token_id(generated_ids: list[int], tok: ChordTokenizer) -> int | None:
    """First chord-shaped id in a continuation (skip [BAR] / specials)."""
    for tid in generated_ids:
        if _chord_token_id(tid, tok):
            return int(tid)
    return None


@torch.no_grad()
def _greedy_next(
    model: torch.nn.Module,
    prompt_ids: list[int],
    max_new_tokens: int,
    eos_id: int,
    device: torch.device,
    tok: ChordTokenizer,
) -> int | None:
    """Greedy decode until we hit a chord-shaped token or run out of budget.
    Returns the predicted chord-token id, or None if we never produced one."""
    ids = torch.tensor([prompt_ids], device=device)
    for _ in range(max_new_tokens):
        if isinstance(model, PeftModel):
            logits = model.base_model.model(ids)
        else:
            logits = model(ids)
        next_id = int(logits[:, -1, :].argmax(dim=-1).item())
        if next_id == eos_id:
            return None
        if _chord_token_id(next_id, tok):
            return next_id
        ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=-1)
    return None


def _build_test_prompts(
    genre: str,
    tok: ChordTokenizer,
    n_prompts: int,
    rng: random.Random,
) -> list[tuple[list[int], int]]:
    """Return up to `n_prompts` (prefix_chord_ids, gold_next_chord_id) pairs.
    Drawn from this genre's held-out test split via ChordSequenceDataset."""
    ds = ChordSequenceDataset(
        split="test",
        max_seq_len=256,
        augment_keys=False,
        sources=[f"chordonomicon_{genre}"],
        tokenizer_version="v1",
        include_extra_genres=True,
    )
    n_seq = ds.input_ids.shape[0]
    if n_seq == 0:
        return []

    prompts: list[tuple[list[int], int]] = []
    # Walk a random permutation of sequences until we have enough usable ones.
    order = list(range(n_seq))
    rng.shuffle(order)
    for idx in order:
        if len(prompts) >= n_prompts:
            break
        seq = ds.input_ids[idx].tolist()
        chord_ids = _extract_chord_id_stream(seq, tok)
        if len(chord_ids) < PREFIX_LEN + 1:
            continue
        start_max = len(chord_ids) - (PREFIX_LEN + 1)
        start = rng.randint(0, start_max)
        prefix = chord_ids[start : start + PREFIX_LEN]
        gold = chord_ids[start + PREFIX_LEN]
        prompts.append((prefix, gold))
    return prompts


@torch.no_grad()
def _sequence_logprob(
    model: PeftModel,
    full_ids: list[int],
    prompt_len: int,
    device: torch.device,
) -> float:
    """Teacher-forced log-probability of the continuation under the model."""
    ids = torch.tensor([full_ids], device=device)
    with torch.no_grad():
        # PEFT wrapper injects kwargs MusicTransformer doesn't accept — call
        # through base_model.model like decoding_artifacts._generate_batch.
        if hasattr(model, "base_model"):
            logits = model.base_model.model(ids[:, :-1])
        else:
            logits = model(ids[:, :-1])
        logprobs = torch.log_softmax(logits, dim=-1)
    total = 0.0
    for pos in range(prompt_len - 1, ids.shape[1] - 1):
        total += float(logprobs[0, pos, ids[0, pos + 1]])
    return total


def _sampled_predictions(
    lora_model: PeftModel,
    prompt_ids: list[int],
    n_samples: int,
    max_new_tokens: int,
    eos_id: int,
    temperature: float,
    top_p: float,
    device: torch.device,
    tok: ChordTokenizer,
) -> dict[str, int | None]:
    """Sample K continuations ONCE, then derive three selection rules from the
    same pool so the sampling-vs-rerank effects are separable (2026-06-12
    confound fix — greedy→sampling and R1 used to change together):

      r1      — rerank by mean Sethares roughness ascending (the R1 condition)
      random  — take sample #0 (a fair stochastic draw from the same sampler)
      maxlik  — take the highest teacher-forced sequence log-prob (the
                sampling-only likelihood control)
    """
    seqs = _generate_batch(
        lora_model, prompt_ids, n_samples, max_new_tokens,
        eos_id, temperature, top_p, device,
    )
    prompt_len = len(prompt_ids)

    candidates: list[list[int]] = []
    candidate_tok_lists: list[list[str]] = []
    full_seqs: list[list[int]] = []
    for ids in seqs:
        cont_ids = ids[prompt_len:]
        candidates.append(cont_ids)
        candidate_tok_lists.append(tok.decode(cont_ids))
        full_seqs.append(ids)

    if not candidates:
        return {"r1": None, "random": None, "maxlik": None}

    # R1: mean roughness over the continuation (rerank module skips specials /
    # unparseable chords). NaN candidates drop to the back so they only win
    # when every candidate is unscorable.
    scored = []
    for i, toks in enumerate(candidate_tok_lists):
        score = mean_roughness(toks)
        sort_key = float("inf") if score != score else score
        scored.append((sort_key, i))
    scored.sort(key=lambda x: x[0])
    r1_idx = scored[0][1]

    # Likelihood control: argmax teacher-forced continuation log-prob.
    lps = [_sequence_logprob(lora_model, fs, prompt_len, device) for fs in full_seqs]
    maxlik_idx = max(range(len(lps)), key=lambda i: lps[i])

    return {
        "r1": _first_chord_token_id(candidates[r1_idx], tok),
        "random": _first_chord_token_id(candidates[0], tok),
        "maxlik": _first_chord_token_id(candidates[maxlik_idx], tok),
    }


def _hit_rate(hits: int, total: int) -> float:
    return round(hits / total * 100, 2) if total > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=8,
                        help="K candidates to sample per prompt for R1 rerank")
    parser.add_argument("--max-new-tokens", type=int, default=16,
                        help="Continuation length budget per sample / greedy decode")
    parser.add_argument("--n-prompts-per-genre", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")
    print(
        f"Config: K={args.n_samples} max_new={args.max_new_tokens} "
        f"n_prompts={args.n_prompts_per_genre} "
        f"temp={args.temperature} top_p={args.top_p} seed={args.seed}"
    )

    old_tok = ChordTokenizer(include_extra_genres=False)
    new_tok = ChordTokenizer(include_extra_genres=True)
    selected = _selected_ranks()

    print("Loading F1 baseline once")
    f1 = _load_f1(old_tok, new_tok, device)

    rows: list[dict] = []
    rng = random.Random(args.seed)

    for i, genre in enumerate(GENRES, start=1):
        rank = selected.get(genre)
        if rank is None:
            print(f"[{i}/{len(GENRES)}] {genre}: SKIP — no selected rank")
            continue
        ckpt_dir = _ckpt_dir_for(genre, rank)
        if not (ckpt_dir / "adapter" / "adapter_config.json").exists():
            print(f"[{i}/{len(GENRES)}] {genre} r={rank}: SKIP — adapter missing at {ckpt_dir}")
            continue

        prompts = _build_test_prompts(genre, new_tok, args.n_prompts_per_genre, rng)
        if not prompts:
            print(f"[{i}/{len(GENRES)}] {genre}: SKIP — no usable test prompts")
            continue
        n_prompts = len(prompts)
        print(f"[{i}/{len(GENRES)}] {genre} r={rank}: {n_prompts} prompts")

        lora = _load_lora(genre, rank, old_tok, new_tok, device)

        f1_hits = 0
        lora_greedy_hits = 0
        r1_hits = 0
        sample_random_hits = 0
        sample_maxlik_hits = 0
        for j, (prefix, gold) in enumerate(prompts, start=1):
            prompt_ids = _build_prompt_from_prefix(new_tok, genre, prefix)

            f1_pred = _greedy_next(
                f1, prompt_ids, args.max_new_tokens,
                new_tok.eos_id, device, new_tok,
            )
            if f1_pred == gold:
                f1_hits += 1

            lora_pred = _greedy_next(
                lora, prompt_ids, args.max_new_tokens,
                new_tok.eos_id, device, new_tok,
            )
            if lora_pred == gold:
                lora_greedy_hits += 1

            preds = _sampled_predictions(
                lora, prompt_ids, args.n_samples, args.max_new_tokens,
                new_tok.eos_id, args.temperature, args.top_p, device, new_tok,
            )
            if preds["r1"] == gold:
                r1_hits += 1
            if preds["random"] == gold:
                sample_random_hits += 1
            if preds["maxlik"] == gold:
                sample_maxlik_hits += 1

            if j % 5 == 0 or j == n_prompts:
                print(
                    f"  [{j}/{n_prompts}] F1={f1_hits} LoRA-greedy={lora_greedy_hits} "
                    f"LoRA+R1={r1_hits} smp-rand={sample_random_hits} "
                    f"smp-maxlik={sample_maxlik_hits}"
                )

        f1_acc = _hit_rate(f1_hits, n_prompts)
        lora_greedy_acc = _hit_rate(lora_greedy_hits, n_prompts)
        r1_acc = _hit_rate(r1_hits, n_prompts)
        rand_acc = _hit_rate(sample_random_hits, n_prompts)
        maxlik_acc = _hit_rate(sample_maxlik_hits, n_prompts)
        delta_r1_vs_lora = round(r1_acc - lora_greedy_acc, 2)
        delta_r1_vs_f1 = round(r1_acc - f1_acc, 2)
        delta_lora_vs_f1 = round(lora_greedy_acc - f1_acc, 2)

        for cond, hits, acc in (
            ("f1_greedy", f1_hits, f1_acc),
            ("lora_greedy", lora_greedy_hits, lora_greedy_acc),
            ("lora_plus_r1", r1_hits, r1_acc),
            ("lora_sample_random", sample_random_hits, rand_acc),
            ("lora_sample_maxlik", sample_maxlik_hits, maxlik_acc),
        ):
            rows.append({
                "genre": genre,
                "rank": rank,
                "n_prompts": n_prompts,
                "condition": cond,
                "hits": hits,
                "top1_acc": acc,
                "delta_vs_f1_greedy": round(acc - f1_acc, 2),
                "delta_vs_lora_greedy": round(acc - lora_greedy_acc, 2),
            })

        print(
            f"  → F1={f1_acc:.2f}%  LoRA-greedy={lora_greedy_acc:.2f}%  "
            f"LoRA+R1={r1_acc:.2f}%  Δ(R1-LoRA)={delta_r1_vs_lora:+.2f}pp"
        )

        del lora
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not rows:
        print("No rows produced — every genre was skipped. Exiting.")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "genre", "rank", "n_prompts", "condition", "hits", "top1_acc",
        "delta_vs_f1_greedy", "delta_vs_lora_greedy",
    ]
    out_csv = RESULTS_DIR / "r1_rerank_ablation.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_csv}")

    # ---- Markdown ----
    by_g: dict[str, dict[str, dict]] = {}
    for r in rows:
        by_g.setdefault(r["genre"], {})[r["condition"]] = r

    lines = [
        "# R1 Sethares physics rerank ablation (held-out test, next-chord top-1)",
        "",
        f"For each genre's validation-selected LoRA rank, {args.n_prompts_per_genre}",
        "test prompts were drawn from the held-out Chordonomicon test split.",
        f"Each prompt = neutral header (`[BOS] [KEY:Cmaj] [TIME:4/4] [GENRE:g] [BAR]`)",
        f"followed by a random {PREFIX_LEN}-chord prefix sliced from a test sequence;",
        "the gold next chord is the chord token immediately after the prefix.",
        "",
        "Five conditions are evaluated on the same prompts (the three sampled",
        "conditions are selection rules over the SAME K-sample pool, so the",
        "sampling effect and the rerank effect are separable):",
        "",
        "- **f1_greedy**: argmax from F1 base only (no LoRA, no rerank)",
        "- **lora_greedy**: argmax from F1 + selected-rank LoRA adapter",
        f"- **lora_plus_r1**: sample K={args.n_samples} continuations from F1+LoRA",
        f"  (temperature={args.temperature}, top_p={args.top_p}, max_new_tokens={args.max_new_tokens}),",
        "  rerank by mean Sethares roughness ascending, take #1, score its first",
        "  chord-shaped token vs the gold next chord.",
        "- **lora_sample_random**: same pool, take sample #0 (sampling-only control)",
        "- **lora_sample_maxlik**: same pool, take the highest teacher-forced",
        "  sequence log-prob (sampling + likelihood-selection control)",
        "",
        f"Random seed = {args.seed}. Cells whose LoRA adapter does not exist were skipped.",
        "Higher top-1 is better; Δ columns are in percentage points.",
        "",
        "| Genre | Rank | n prompts | F1 greedy | LoRA greedy | LoRA + R1 | smp random | smp maxlik | Δ R1 vs LoRA | Δ R1 vs F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for g in GENRES:
        cells = by_g.get(g)
        if not cells:
            continue
        f1c = cells.get("f1_greedy")
        lgc = cells.get("lora_greedy")
        r1c = cells.get("lora_plus_r1")
        rnd = cells.get("lora_sample_random")
        mlk = cells.get("lora_sample_maxlik")
        if not (f1c and lgc and r1c and rnd and mlk):
            continue
        lines.append(
            f"| {g} | {f1c['rank']} | {f1c['n_prompts']} | "
            f"{f1c['top1_acc']:.2f}% | {lgc['top1_acc']:.2f}% | "
            f"{r1c['top1_acc']:.2f}% | {rnd['top1_acc']:.2f}% | "
            f"{mlk['top1_acc']:.2f}% | {r1c['delta_vs_lora_greedy']:+.2f}pp | "
            f"{r1c['delta_vs_f1_greedy']:+.2f}pp |"
        )

    # Aggregate over evaluated genres
    f1_rows = [r for r in rows if r["condition"] == "f1_greedy"]
    lora_rows = [r for r in rows if r["condition"] == "lora_greedy"]
    r1_rows = [r for r in rows if r["condition"] == "lora_plus_r1"]
    if f1_rows and lora_rows and r1_rows:
        def _mean(rs, key):
            return round(sum(r[key] for r in rs) / len(rs), 2)
        n_genres = len(f1_rows)
        mean_f1 = _mean(f1_rows, "top1_acc")
        mean_lora = _mean(lora_rows, "top1_acc")
        mean_r1 = _mean(r1_rows, "top1_acc")
        mean_delta_r1_lora = _mean(r1_rows, "delta_vs_lora_greedy")
        mean_delta_r1_f1 = _mean(r1_rows, "delta_vs_f1_greedy")
        n_r1_wins = sum(
            1 for r in r1_rows if r["delta_vs_lora_greedy"] > 0
        )
        n_r1_ties = sum(
            1 for r in r1_rows if r["delta_vs_lora_greedy"] == 0
        )
        n_r1_losses = n_genres - n_r1_wins - n_r1_ties
        lines += [
            "",
            "## Aggregate",
            "",
            f"Across {n_genres} evaluated genres:",
            "",
            f"- Mean F1-greedy top-1: {mean_f1:.2f}%",
            f"- Mean LoRA-greedy top-1: {mean_lora:.2f}%",
            f"- Mean LoRA+R1 top-1: {mean_r1:.2f}%",
            f"- Mean Δ (R1 − LoRA-greedy): {mean_delta_r1_lora:+.2f}pp",
            f"- Mean Δ (R1 − F1-greedy): {mean_delta_r1_f1:+.2f}pp",
            f"- R1 wins vs LoRA-greedy: {n_r1_wins} / {n_genres} "
            f"(ties {n_r1_ties}, losses {n_r1_losses})",
            "",
            f"Config: K={args.n_samples}, n_prompts_per_genre={args.n_prompts_per_genre}, "
            f"temperature={args.temperature}, top_p={args.top_p}, "
            f"max_new_tokens={args.max_new_tokens}, prefix_len={PREFIX_LEN}, "
            f"seed={args.seed}.",
        ]

    out_md = RESULTS_DIR / "r1_rerank_ablation.md"
    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
