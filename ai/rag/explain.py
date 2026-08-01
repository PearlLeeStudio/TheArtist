"""Explain a chord progression using the Berklee RAG — human-readable output.

Sibling of `rag.coherence` (metric / yes-no) and `rag.retrieve` (raw
top-k). For each transition, retrieves top-k passages and asks the LLM
for a short, citable explanation ("this is a ii-V-I cadence …"). The
output is prose + a chapter/page pointer, suitable for UI tooltips or
paper qualitative examples.

CLI::

    cd ai && uv run python3 -m rag.explain --chords "Dm7 G7 Cmaj7 Am7 D7 G7"
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import textwrap
import time
from dataclasses import dataclass
from typing import Iterable, Iterator

from openai import OpenAI

from .retrieve import Retriever

log = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://localhost:11434/v1"
_DEFAULT_MODEL = "gemma3:4b"

_SYSTEM_PROMPT = """You are a jazz-harmony tutor. You will read up to three passages from a jazz harmony textbook and explain a specific chord transition in one short paragraph (2-4 sentences).

Return strict JSON:
    {"explanation": "<2-4 sentences citing the relevant concept(s) from the passages>",
     "concept": "<one short tag drawn from: ii-V, V-I, plagal cadence, half cadence, deceptive cadence, secondary dominant, tritone sub, modal interchange, Picardy third, chromatic mediant, Neapolitan, augmented sixth, passing diminished, approach chord, turnaround, pedal point, pivot modulation, common-tone modulation, mode (Ionian/Dorian/Phrygian/Lydian/Mixolydian/Aeolian/Locrian), …>",
     "passage_used": <1|2|3|null>}

Rules:
- Stay grounded in the passages — if a passage describes a relevant pattern (ii-V, secondary dominant, modal interchange, plagal/half/deceptive cadence, Picardy third, chromatic mediant, Neapolitan, augmented sixth, modal scale colour, etc.), cite that concept by name.
- Name the category even when the passages use Roman numerals and you must map the specific chord symbols. Dm7 → G7 is a ii-V in C. G7 → Cmaj7 is a V-I authentic cadence. F → C is a IV-I plagal. Am → A is a Picardy third in A minor. C → Eb in C major is a chromatic mediant / bIII modal interchange. Am7 → D7 is a V7/V (secondary dominant to V).
- If NONE of the passages covers this transition, set "concept" to "uncovered", "passage_used" to null, and say so briefly in the explanation.
- Never invent theory outside what the passages support. Do not include anything outside the JSON object."""

_USER_TEMPLATE = (
    "Passage 1:\n{p1}\n\n"
    "Passage 2:\n{p2}\n\n"
    "Passage 3:\n{p3}\n\n"
    "Explain the chord transition: {a} → {b}"
)


@dataclass
class Explanation:
    chord_a: str
    chord_b: str
    concept: str
    explanation: str
    passage_used: int  # 1/2/3, 0 = uncovered
    top1_chapter: str
    top1_section: str
    top1_page_start: int
    top1_page_end: int
    top1_similarity: float


def _is_chord(tok: str) -> bool:
    return bool(tok) and not tok.startswith(("[", "<"))


def _call_llm(client: OpenAI, model: str, passages: list[str], a: str, b: str, timeout: float) -> dict:
    p = (passages + ["", "", ""])[:3]
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _USER_TEMPLATE.format(p1=p[0], p2=p[1], p3=p[2], a=a, b=b)},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
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
        return {"concept": "parse-fail", "explanation": raw[:200], "passage_used": None}


def explain_transition(
    chord_a: str,
    chord_b: str,
    retriever: Retriever,
    llm_client: OpenAI,
    llm_model: str = _DEFAULT_MODEL,
    timeout: float = 60.0,
    k: int = 3,
) -> Explanation:
    query = (
        f"jazz harmony progression: {chord_a} resolving to {chord_b}, "
        f"voice leading and functional analysis"
    )
    hits = retriever.query(query, k=k)
    if not hits:
        return Explanation(
            chord_a, chord_b, "no-retrieval",
            "No theory passage retrieved for this transition.",
            0, "", "", 0, 0, 0.0,
        )
    top1 = hits[0]
    similarity = max(0.0, 1.0 - float(top1.distance))
    verdict = _call_llm(llm_client, llm_model, [h.text for h in hits], chord_a, chord_b, timeout)
    pu = verdict.get("passage_used")
    pu_idx = int(pu) if isinstance(pu, int) or (isinstance(pu, str) and pu.isdigit()) else 0
    return Explanation(
        chord_a=chord_a, chord_b=chord_b,
        concept=str(verdict.get("concept", "unknown"))[:40],
        explanation=str(verdict.get("explanation", ""))[:400],
        passage_used=pu_idx,
        top1_chapter=top1.chapter,
        top1_section=top1.section,
        top1_page_start=top1.page_start,
        top1_page_end=top1.page_end,
        top1_similarity=similarity,
    )


def explain_progression(
    tokens: Iterable[str],
    retriever: Retriever,
    llm_client: OpenAI,
    llm_model: str = _DEFAULT_MODEL,
    *,
    progress: bool = False,
) -> Iterator[Explanation]:
    chords = [t for t in tokens if _is_chord(t)]
    pairs = list(zip(chords, chords[1:]))
    for i, (a, b) in enumerate(pairs, start=1):
        t0 = time.perf_counter()
        e = explain_transition(a, b, retriever, llm_client, llm_model)
        if progress:
            dt = time.perf_counter() - t0
            log.info("  [%d/%d] %s → %s  (%s, p%d-%d)  %.1fs",
                     i, len(pairs), a, b, e.concept, e.top1_page_start, e.top1_page_end, dt)
        yield e


def _pretty_print(e: Explanation) -> str:
    head = f"{e.chord_a} → {e.chord_b}   [{e.concept}]"
    pages = f"p{e.top1_page_start}" + (f"-{e.top1_page_end}" if e.top1_page_end != e.top1_page_start else "")
    ref = f"{e.top1_chapter or '(no chapter)'} · {e.top1_section or '(no section)'} · {pages}"
    body = textwrap.fill(e.explanation, width=100, initial_indent="    ", subsequent_indent="    ")
    return f"\n{head}\n    ref: {ref}\n{body}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chords", required=True, help="space-separated chord progression")
    ap.add_argument("--collection", default="berklee")
    ap.add_argument("--base-url", default=_DEFAULT_BASE_URL)
    ap.add_argument("--model", default=_DEFAULT_MODEL)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    retriever = Retriever(collection=args.collection)
    client = OpenAI(base_url=args.base_url, api_key="ollama")

    explanations = list(explain_progression(
        args.chords.split(), retriever, client, llm_model=args.model, progress=True,
    ))
    print(f"\n=== {args.chords}  ({len(explanations)} transitions) ===")
    for e in explanations:
        print(_pretty_print(e))
    return 0


if __name__ == "__main__":
    sys.exit(main())
