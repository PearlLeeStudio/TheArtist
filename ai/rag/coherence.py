"""Theory-coherence scoring for chord transitions.

For each chord transition in a progression, the scorer:
    1. Embeds a transition query against the Berklee chromadb collection.
    2. Retrieves the top-1 passage (source, chapter, section, similarity).
    3. Asks a local LLM (default: gemma3:4b via ollama) whether that
       passage actually justifies the transition, in strict JSON.
    4. Emits per-transition records; an aggregator then produces the
       mean cosine, LLM yes-rate, and combined coherence score.

This is the "theory axis" numeric output. It complements
``ai/physics/composite.py`` (physics axis) and
``ai/physics/diversity.py`` (diversity axis).

Judge-model selection rationale: gemma3:4b is the only judge that
fits alongside F1+LoRA inference on an 8GB GPU and delivers <10s/call
latency — required for live demo where 5-10 RAG calls fire per
generated progression on jazz-family and classical genres. Larger
models (gemma3:12b at 8.1GB, qwen3.6:27b at 17.4GB) spill to CPU and
push per-call latency past 30s / 4min respectively, breaking the
demo UX.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from typing import Iterable, Iterator

from openai import OpenAI

from .retrieve import Retriever

log = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://localhost:11434/v1"
_DEFAULT_MODEL = "gemma3:4b"

_SYSTEM_PROMPT = """You are a music-theory evaluator. You read up to three passages from a jazz harmony textbook and judge whether ANY of them supports a specific chord transition.

Return strict JSON with exactly these keys:
    {"justified": <true|false>, "decisive_passage": <1|2|3|null>, "reason": "<one sentence, plain prose>"}

Rules:
- "justified" = true if AT LEAST ONE passage explicitly discusses this transition OR the harmonic category it belongs to. Recognised categories include cadences (ii-V-I, V-I authentic, IV-I plagal, -V half, deceptive), dominants (primary V, secondary V/x, extended/back-cycling, altered, tritone sub), modal interchange and parallel-mode borrows (Picardy third, iv-in-major, bIII/bVI/bVII), chromatic mediants (I-bIII, I-bVI, I-III, I-VI), the Neapolitan (bII), augmented sixths (It/Fr/Ger), passing/approach chords (passing diminished, chromatic approach), turnarounds, pedal point, modulation (pivot chord, common-tone, direct), and the seven church modes (Ionian, Dorian, Phrygian, Lydian, Mixolydian, Aeolian, Locrian). If true, report the 1-indexed `decisive_passage`.
- "justified" = false when NONE of the passages covers this transition even by implication. `decisive_passage` must be null.
- A passing mention of a related concept is NOT justification. The passage must discuss the transition's *function* or *category*.
- Use the full harmonic context. Dm7 → G7 is a ii-V in C major (and related keys). G7 → Cmaj7 is a V-I resolution (authentic cadence). Am7 → D7 is a V7/V or ii-V in G. Recognise these canonical patterns even when the passage uses Roman numerals or generic names rather than the specific chord symbols.
- Do not include any text outside the JSON object."""

_USER_TEMPLATE = (
    "Passage 1:\n{p1}\n\n"
    "Passage 2:\n{p2}\n\n"
    "Passage 3:\n{p3}\n\n"
    "Chord transition: {a} → {b}\n\n"
    "Does any passage justify this transition?"
)


@dataclass
class TransitionScore:
    chord_a: str
    chord_b: str
    top1_passage: str         # highest-similarity passage (for reporting)
    top1_chapter: str
    top1_section: str
    top1_page_start: int
    top1_page_end: int
    top1_similarity: float
    llm_justified: bool
    llm_decisive_passage: int  # 1, 2, or 3 (which retrieved passage convinced the LLM); 0 if none
    llm_reason: str


def _is_chord_token(tok: str) -> bool:
    return bool(tok) and not tok.startswith(("[", "<"))


def extract_chords(tokens: Iterable[str]) -> list[str]:
    """Filter special tokens ([BOS], [BAR], etc.) out of a token stream."""
    return [t for t in tokens if _is_chord_token(t)]


def _call_llm(
    client: OpenAI, model: str, passages: list[str], a: str, b: str, timeout: float
) -> dict:
    # Pad to three passages (pass empty string if retrieval returned fewer)
    p = (passages + ["", "", ""])[:3]
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _USER_TEMPLATE.format(p1=p[0], p2=p[1], p3=p[2], a=a, b=b)},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
        timeout=timeout,
    )
    raw = (resp.choices[0].message.content or "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        i, j = raw.find("{"), raw.rfind("}")
        if i >= 0 and j > i:
            try:
                return json.loads(raw[i : j + 1])
            except json.JSONDecodeError:
                pass
        return {"justified": False, "decisive_passage": None, "reason": f"parse-fail: {raw[:80]}"}


def score_transition(
    chord_a: str,
    chord_b: str,
    retriever: Retriever,
    llm_client: OpenAI,
    llm_model: str = _DEFAULT_MODEL,
    timeout: float = 60.0,
    k: int = 3,
) -> TransitionScore:
    """Score a single A→B transition against the Berklee RAG + LLM judge.

    Retrieves top-k passages and asks the LLM whether any of them justifies
    the transition. The query is augmented with theory-oriented qualifiers
    ("harmonic progression", "voice leading", "functional analysis") to bias
    retrieval toward explanatory passages rather than stray chord-symbol
    occurrences in examples.
    """
    query_text = (
        f"jazz harmony progression: {chord_a} resolving to {chord_b}, "
        f"voice leading and functional analysis"
    )
    hits = retriever.query(query_text, k=k)
    if not hits:
        return TransitionScore(
            chord_a=chord_a, chord_b=chord_b,
            top1_passage="", top1_chapter="", top1_section="",
            top1_page_start=0, top1_page_end=0, top1_similarity=0.0,
            llm_justified=False, llm_decisive_passage=0,
            llm_reason="no passage retrieved",
        )
    top1 = hits[0]
    similarity = max(0.0, 1.0 - float(top1.distance))
    passages = [h.text for h in hits]
    verdict = _call_llm(llm_client, llm_model, passages, chord_a, chord_b, timeout)
    decisive = verdict.get("decisive_passage")
    decisive_idx = int(decisive) if isinstance(decisive, int) or (isinstance(decisive, str) and decisive.isdigit()) else 0
    return TransitionScore(
        chord_a=chord_a,
        chord_b=chord_b,
        top1_passage=top1.text,
        top1_chapter=top1.chapter,
        top1_section=top1.section,
        top1_page_start=top1.page_start,
        top1_page_end=top1.page_end,
        top1_similarity=similarity,
        llm_justified=bool(verdict.get("justified", False)),
        llm_decisive_passage=decisive_idx,
        llm_reason=str(verdict.get("reason", ""))[:240],
    )


def score_sequence(
    tokens: Iterable[str],
    retriever: Retriever,
    llm_client: OpenAI,
    llm_model: str = _DEFAULT_MODEL,
    timeout: float = 60.0,
    *,
    progress: bool = False,
) -> Iterator[TransitionScore]:
    """Score all A→B pairs across consecutive chord tokens in a sequence.

    Yields per transition so long sequences can stream into a CSV writer
    without materialising the full list.
    """
    chords = extract_chords(tokens)
    pairs = list(zip(chords, chords[1:]))
    for i, (a, b) in enumerate(pairs, start=1):
        t0 = time.perf_counter()
        score = score_transition(a, b, retriever, llm_client, llm_model, timeout)
        if progress:
            dt = time.perf_counter() - t0
            log.info(
                "transition %d/%d  %s → %s  sim=%.3f  justified=%s  %.1fs",
                i, len(pairs), a, b, score.top1_similarity, score.llm_justified, dt,
            )
        yield score


def summarise(scores: list[TransitionScore]) -> dict[str, float]:
    """Aggregate transition-level scores into progression-level metrics.

    Returns a dict with mean_similarity, justify_rate, combined (mean of
    the two, clamped to [0, 1]), and n_transitions.
    """
    n = len(scores)
    if n == 0:
        return {
            "mean_similarity": float("nan"),
            "justify_rate": float("nan"),
            "combined": float("nan"),
            "n_transitions": 0,
        }
    mean_sim = sum(s.top1_similarity for s in scores) / n
    justify = sum(1 for s in scores if s.llm_justified) / n
    return {
        "mean_similarity": mean_sim,
        "justify_rate": justify,
        "combined": 0.5 * mean_sim + 0.5 * justify,
        "n_transitions": n,
    }


def make_llm_client(base_url: str = _DEFAULT_BASE_URL) -> OpenAI:
    """Convenience constructor matching the tagger / retriever conventions."""
    return OpenAI(base_url=base_url, api_key="ollama")


# CLI for quick ad-hoc scoring of a progression (paths below assume `cd ai`).
# Example:
#     uv run python -m rag.coherence --chords "Dm7 G7 Cmaj7 Am7 Dm7 G7"
if __name__ == "__main__":
    import argparse
    import csv
    import sys

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chords", required=True, help="space-separated chord progression")
    ap.add_argument("--collection", default="berklee")
    ap.add_argument("--base-url", default=_DEFAULT_BASE_URL)
    ap.add_argument("--model", default=_DEFAULT_MODEL)
    ap.add_argument("--out", default=None, help="optional CSV output path")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    retriever = Retriever(collection=args.collection)
    client = make_llm_client(args.base_url)

    rows = list(
        score_sequence(
            args.chords.split(),
            retriever,
            client,
            llm_model=args.model,
            progress=True,
        )
    )
    summary = summarise(rows)
    print("\n--- summary ---")
    print(json.dumps(summary, indent=2))

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
            w.writeheader()
            for r in rows:
                w.writerow(asdict(r))
        print(f"wrote {len(rows)} transitions → {args.out}")
    sys.exit(0)
