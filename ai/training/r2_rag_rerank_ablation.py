"""§3.2 ablation — does R2 theory-RAG rerank improve next-chord prediction?

Pipeline (per genre, per held-out test prompt):

  1. Build an 8-chord prefix from a test sequence; gold = 9th chord token.
  2. From F1 + per-genre LoRA, sample K candidate continuations.
  3. For each candidate's first chord token, build a transition query
     ``prev_chord -> cand_chord`` and run the existing Berklee + OMT RAG
     coherence judge (``rag.coherence.score_transition``) to score 0..1.
  4. Pick the highest-coherence candidate (ties → first by sample index);
     record top-1 accuracy of its first chord vs the gold 9th chord.
  5. Also record greedy F1+LoRA and greedy F1 baselines on the same prompt.

Genres outside ``RAG_ENABLED_GENRES`` (i.e. not in the jazz family and not
classical) record `--` for r2_acc, because the corpora (Berklee Book of
Jazz Harmony + Open Music Theory v2) do not cover them. The deployed
backend falls back to greedy LoRA for those genres anyway — see
``backend/app/services/inference.py::_should_use_rag``.

Output:
  ai/results/r2_rag_rerank_ablation.csv
  ai/results/r2_rag_rerank_ablation.md

Usage (executed after the multi-seed sweep):
  cd ai
  UV_CACHE_DIR=/tmp/uv-cache uv run python training/r2_rag_rerank_ablation.py

Note: this script DOES touch GPU (sampling from F1 + LoRA on `cuda`) AND
ollama (the RAG coherence judge talks to ``http://localhost:11434/v1``
running ``qwen3.6:27b-q4_K_M`` per the 2026-05-11 commit). Both must be
healthy before invocation; otherwise rows degrade to per-cell skip.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
import traceback
from pathlib import Path

import torch
from peft import PeftModel  # type: ignore[import-untyped]
from torch.utils.data import DataLoader

# Reuse the canonical loaders from the rest of the §3 pipeline.
from dataset import ChordSequenceDataset
from decoding_artifacts import (
    _generate_batch,
    _load_f1,
    _load_lora,
    _selected_ranks,
)
from tokenizer import ChordTokenizer

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"

# Mirror of the 11-LoRA genre list used in the rest of the §3 sweeps.
GENRES = [
    "blues", "bossa", "classical", "country", "electronic",
    "folk", "funk", "gospel", "hip_hop", "rnb_soul", "rock",
]

# Mirror of backend/app/services/inference.py::JAZZ_FAMILY_GENRES.
# "bossa nova" is omitted here because the LoRA training key is "bossa";
# the backend keeps both spellings to be tolerant of upstream callers.
JAZZ_FAMILY_GENRES: set[str] = {"jazz", "blues", "bossa", "rnb_soul"}
RAG_ENABLED_GENRES: set[str] = JAZZ_FAMILY_GENRES | {"classical"}

# Prompt-prefix protocol: 8 chord tokens of context, gold = 9th chord token.
PREFIX_CHORDS = 8


def _is_chord_token(tok: str) -> bool:
    """Reject [PAD]/[BOS]/[EOS]/[BAR]/[KEY:*]/[TIME:*]/[GENRE:*]; keep chords."""
    return bool(tok) and not tok.startswith(("[", "<"))


def _build_prefix_and_gold(
    ids: list[int],
    new_tok: ChordTokenizer,
) -> tuple[list[int], str, list[str]] | None:
    """Slice a tokenized sequence into (prefix_ids, gold_chord, prefix_chords).

    Walks the token stream, keeps every token up to AND INCLUDING the 8th
    chord token; gold is the 9th chord token. Special / structural tokens
    (BOS, KEY, TIME, GENRE, BAR) ride along inside the prefix so the model
    sees its usual conditioning header. Returns None if the sequence is
    too short or contains a PAD before the gold chord.
    """
    prefix: list[int] = []
    chords_seen = 0
    chord_str_history: list[str] = []
    pad_id = new_tok.pad_id
    for i, tid in enumerate(ids):
        nid = int(tid)
        if nid == pad_id:
            return None
        tok = new_tok.id2token.get(nid, "")
        prefix.append(nid)
        if _is_chord_token(tok):
            chords_seen += 1
            chord_str_history.append(tok)
            if chords_seen == PREFIX_CHORDS:
                # Gold = next chord token strictly after this index.
                for j in range(i + 1, len(ids)):
                    njd = int(ids[j])
                    if njd == pad_id:
                        return None
                    ntok = new_tok.id2token.get(njd, "")
                    if _is_chord_token(ntok):
                        return prefix, ntok, chord_str_history
                return None
    return None


def _collect_prompts(
    new_tok: ChordTokenizer,
    genre: str,
    n_prompts: int,
) -> list[dict]:
    """Pull up to `n_prompts` (prefix, gold, history) cells from the per-genre
    held-out test split. Same source filter as aggregate_multi_seed.py."""
    ds = ChordSequenceDataset(
        split="test",
        max_seq_len=256,
        augment_keys=False,
        sources=[f"chordonomicon_{genre}"],
        tokenizer_version="v1",
        include_extra_genres=True,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    out: list[dict] = []
    for batch in loader:
        # input_ids row is [seq_len-1]; trim trailing PADs implicitly via _build_prefix_and_gold.
        ids = batch["input_ids"][0].tolist()
        built = _build_prefix_and_gold(ids, new_tok)
        if built is None:
            continue
        prefix_ids, gold_tok, prefix_chords = built
        out.append({
            "prefix_ids": prefix_ids,
            "gold": gold_tok,
            "prefix_chords": prefix_chords,
        })
        if len(out) >= n_prompts:
            break
    return out


@torch.no_grad()
def _greedy_next_chord(
    model: torch.nn.Module,
    prefix_ids: list[int],
    new_tok: ChordTokenizer,
    max_steps: int,
    device: torch.device,
) -> str | None:
    """Greedy-decode until the first chord token after the prefix. Returns the
    chord token string or None if no chord appeared within max_steps."""
    ids = torch.tensor([prefix_ids], device=device)
    for _ in range(max_steps):
        if isinstance(model, PeftModel):
            logits = model.base_model.model(ids)
        else:
            logits = model(ids)
        next_id = int(logits[:, -1, :].argmax(dim=-1).item())
        if next_id == new_tok.eos_id:
            return None
        ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=-1)
        tok = new_tok.id2token.get(next_id, "")
        if _is_chord_token(tok):
            return tok
    return None


def _first_chord_after_prefix(
    sample_ids: list[int],
    prefix_len: int,
    new_tok: ChordTokenizer,
) -> str | None:
    """Scan a sampled sequence past the prompt-prefix for the first chord token."""
    for tid in sample_ids[prefix_len:]:
        tok = new_tok.id2token.get(int(tid), "")
        if tok in ("", "[EOS]"):
            return None
        if _is_chord_token(tok):
            return tok
    return None


def _safe_score_transition(retriever, llm_client, llm_model: str, prev_chord: str, cand_chord: str):
    """Wrap rag.coherence.score_transition so a single LLM hiccup doesn't tank
    the whole genre. On error returns (similarity=NaN, justified=False)."""
    try:
        from rag.coherence import score_transition  # imported lazily to avoid hard dep at module load
        return score_transition(prev_chord, cand_chord, retriever, llm_client, llm_model=llm_model)
    except Exception as e:  # noqa: BLE001
        print(f"    [coherence error] {type(e).__name__}: {e}", flush=True)
        return None


def _coherence_score(ts_result) -> float:
    """Combine top-1 cosine similarity + LLM justify bit into one 0..1 score,
    matching the aggregator in rag.coherence.summarise (mean of the two)."""
    if ts_result is None:
        return float("-inf")
    sim = float(getattr(ts_result, "top1_similarity", 0.0))
    just = 1.0 if bool(getattr(ts_result, "llm_justified", False)) else 0.0
    return 0.5 * sim + 0.5 * just


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=8,
                        help="K candidate continuations sampled per prompt")
    parser.add_argument("--max-new-tokens", type=int, default=16,
                        help="Max tokens generated per candidate (just need the first chord)")
    parser.add_argument("--n-prompts-per-genre", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--rag-llm", default="gemma3:4b",
                        help="ollama-served judge model. Default gemma3:4b fits"
                             " entirely in the 8GB laptop GPU alongside F1+LoRA"
                             " sampling (~4.3GB combined) and runs at ~33 tok/s."
                             " qwen3.6:27b-q4_K_M is ~30x slower on this hardware"
                             " because it spills to CPU (17.4GB > 8GB).")
    parser.add_argument("--rag-base-url", default="http://localhost:11434/v1")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    old_tok = ChordTokenizer(include_extra_genres=False)
    new_tok = ChordTokenizer(include_extra_genres=True)
    selected = _selected_ranks()

    print("Loading F1 baseline once")
    f1 = _load_f1(old_tok, new_tok, device)

    # Lazy import of the RAG stack so that genres with rag disabled can still
    # produce a row even if chromadb / ollama happen to be cold. If neither
    # retriever nor client constructs cleanly we mark the entire RAG path
    # as unavailable rather than crashing the script.
    retriever = None
    llm_client = None
    rag_init_error: str | None = None
    try:
        from rag.retrieve import Retriever  # noqa: WPS433
        from rag.coherence import make_llm_client  # noqa: WPS433
        retriever = Retriever(collections=["berklee", "openmusictheory"])
        llm_client = make_llm_client(base_url=args.rag_base_url)
        print(f"RAG ready: retriever=[berklee, openmusictheory], llm={args.rag_llm}")
    except Exception as e:  # noqa: BLE001
        rag_init_error = f"{type(e).__name__}: {e}"
        traceback.print_exc()
        print(f"WARN: RAG init failed — r2_acc cells will be `--`. Reason: {rag_init_error}")

    records: list[dict] = []
    for genre in GENRES:
        rank = selected[genre]
        rag_supported = genre in RAG_ENABLED_GENRES
        rag_active = rag_supported and retriever is not None and llm_client is not None
        rag_note = (
            "ok" if rag_active
            else ("unsupported" if not rag_supported else f"init-fail: {rag_init_error}")
        )
        print(f"\n=== {genre} r={rank} | rag={rag_note} ===")

        prompts = _collect_prompts(new_tok, genre, args.n_prompts_per_genre)
        if not prompts:
            print(f"  no usable test prompts (need ≥{PREFIX_CHORDS + 1} chords) — skip")
            records.append({
                "genre": genre, "rank": rank, "n_prompts": 0,
                "rag_supported": rag_supported, "rag_note": "no-prompts",
                "f1_greedy_acc": "", "lora_greedy_acc": "", "r2_acc": "",
                "delta_r2_vs_lora_greedy": "",
                "n_samples_per_prompt": args.n_samples,
            })
            continue
        print(f"  built {len(prompts)} test prompts (gold = 9th chord)")

        lora = _load_lora(genre, rank, old_tok, new_tok, device)

        f1_correct = 0
        lora_correct = 0
        r2_correct = 0
        r2_eligible = 0  # prompts where the rerank actually produced a chord candidate
        t_genre0 = time.perf_counter()

        for pi, prompt in enumerate(prompts, start=1):
            prefix_ids = prompt["prefix_ids"]
            prefix_len = len(prefix_ids)
            gold = prompt["gold"]
            prev_chord = prompt["prefix_chords"][-1]

            # 1. Greedy F1 baseline.
            f1_pred = _greedy_next_chord(f1, prefix_ids, new_tok, args.max_new_tokens, device)
            if f1_pred == gold:
                f1_correct += 1

            # 2. Greedy LoRA baseline.
            lora_pred = _greedy_next_chord(lora, prefix_ids, new_tok, args.max_new_tokens, device)
            if lora_pred == gold:
                lora_correct += 1

            # 3. K candidates from LoRA + RAG rerank (only for rag-supported genres).
            if not rag_active:
                if pi % 5 == 0:
                    print(f"  [{pi}/{len(prompts)}] gold={gold} | f1={f1_pred} | lora={lora_pred}")
                continue

            sampled = _generate_batch(
                lora,
                prefix_ids,
                args.n_samples,
                args.max_new_tokens,
                new_tok.eos_id,
                args.temperature,
                args.top_p,
                device,
            )
            best_chord: str | None = None
            best_score = float("-inf")
            for sample_ids in sampled:
                cand_chord = _first_chord_after_prefix(sample_ids, prefix_len, new_tok)
                if cand_chord is None:
                    continue
                ts = _safe_score_transition(
                    retriever, llm_client, args.rag_llm, prev_chord, cand_chord,
                )
                score = _coherence_score(ts)
                if score > best_score:
                    best_score = score
                    best_chord = cand_chord
            if best_chord is not None:
                r2_eligible += 1
                if best_chord == gold:
                    r2_correct += 1

            if pi % 5 == 0:
                print(f"  [{pi}/{len(prompts)}] gold={gold} | f1={f1_pred} | "
                      f"lora={lora_pred} | r2={best_chord} (score={best_score:.3f})")

        del lora
        if device.type == "cuda":
            torch.cuda.empty_cache()

        n = len(prompts)
        f1_acc = f1_correct / n * 100
        lora_acc = lora_correct / n * 100
        if rag_active and r2_eligible > 0:
            r2_acc = r2_correct / r2_eligible * 100
            delta = round(r2_acc - lora_acc, 2)
            r2_acc_out: float | str = round(r2_acc, 2)
            delta_out: float | str = delta
        else:
            r2_acc_out = ""
            delta_out = ""

        dt = time.perf_counter() - t_genre0
        print(f"  done in {dt:.0f}s | f1={f1_acc:.2f} | lora={lora_acc:.2f} | "
              f"r2={r2_acc_out} | delta={delta_out}")

        records.append({
            "genre": genre,
            "rank": rank,
            "n_prompts": n,
            "n_samples_per_prompt": args.n_samples,
            "rag_supported": rag_supported,
            "rag_note": rag_note,
            "f1_greedy_acc": round(f1_acc, 2),
            "lora_greedy_acc": round(lora_acc, 2),
            "r2_acc": r2_acc_out,
            "delta_r2_vs_lora_greedy": delta_out,
            "r2_eligible_prompts": r2_eligible if rag_active else "",
        })

    # Write CSV.
    fieldnames = [
        "genre", "rank", "n_prompts", "n_samples_per_prompt",
        "rag_supported", "rag_note",
        "f1_greedy_acc", "lora_greedy_acc", "r2_acc", "delta_r2_vs_lora_greedy",
        "r2_eligible_prompts",
    ]
    out_csv = RESULTS_DIR / "r2_rag_rerank_ablation.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(records)
    print(f"\nWrote {out_csv}")

    # Markdown summary.
    lines = [
        "# §3.2 R2 theory-RAG rerank ablation (held-out test, next-chord top-1)",
        "",
        f"For each genre's validation-selected LoRA rank, we sampled "
        f"K={args.n_samples} continuations from F1 + LoRA per held-out prompt "
        f"(temperature={args.temperature}, top_p={args.top_p}, "
        f"max_new_tokens={args.max_new_tokens}), scored each candidate's first "
        "chord against the previous chord with the merged Berklee + Open Music "
        f"Theory v2 retriever and the ollama-served LLM judge "
        f"(`{args.rag_llm}`), and picked the highest-coherence candidate. "
        f"`r2_acc` is its top-1 vs the gold 9th chord on up to "
        f"{args.n_prompts_per_genre} prompts per genre. `lora_greedy_acc` and "
        "`f1_greedy_acc` are greedy-argmax baselines on the same prompts.",
        "",
        "Coverage policy: the Berklee corpus covers the jazz family; "
        "Open Music Theory v2 covers classical / general harmony. Other "
        "genres receive no targeted theory rerank and fall back to greedy "
        "LoRA in the deployed system (see "
        "`backend/app/services/inference.py::_should_use_rag`). They are "
        "marked `--` in the `r2_acc` column with the footnote below.",
        "",
        "| Genre | Rank | n prompts | F1 greedy | LoRA greedy | R2 RAG-rerank | Δ R2 vs LoRA |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in records:
        if r["r2_acc"] == "":
            r2_cell = "--"
            delta_cell = "--"
        else:
            r2_cell = f"{r['r2_acc']:.2f}"
            delta_cell = f"{r['delta_r2_vs_lora_greedy']:+.2f}"
        lines.append(
            f"| {r['genre']} | {r['rank']} | {r['n_prompts']} | "
            f"{r['f1_greedy_acc']:.2f} | {r['lora_greedy_acc']:.2f} | "
            f"{r2_cell} | {delta_cell} |"
        )

    # Aggregate Δ across RAG-supported genres only.
    supported = [
        r for r in records
        if r["rag_supported"] and isinstance(r["delta_r2_vs_lora_greedy"], (int, float))
    ]
    if supported:
        deltas = [float(r["delta_r2_vs_lora_greedy"]) for r in supported]
        lines += [
            "",
            f"Across {len(supported)} RAG-supported genres: "
            f"mean Δ = {statistics.mean(deltas):+.2f}pp, "
            f"median Δ = {statistics.median(deltas):+.2f}pp, "
            f"min Δ = {min(deltas):+.2f}pp, "
            f"max Δ = {max(deltas):+.2f}pp.",
        ]
    else:
        lines += [
            "",
            "No RAG-supported genres produced a Δ in this run "
            f"(rag_init_error={rag_init_error or 'none'}). All r2_acc cells are `--`.",
        ]

    lines += [
        "",
        "Footnote: `--` = RAG unsupported (no theory corpus for genre). "
        "The Berklee Book of Jazz Harmony covers jazz family "
        "(jazz/blues/bossa/rnb_soul); Open Music Theory v2 covers classical and "
        "general Western harmony. Pop / rock / country / hip-hop / electronic / "
        "funk / folk / gospel are not represented in either book, so the LLM "
        "judge has no relevant passages to retrieve. The deployed backend "
        "skips RAG for those genres and uses greedy LoRA — see the gate at "
        "`backend/app/services/inference.py::_should_use_rag`.",
    ]

    out_md = RESULTS_DIR / "r2_rag_rerank_ablation.md"
    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    sys.exit(main() or 0)
