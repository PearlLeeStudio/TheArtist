"""Source-isolated, offline theory retrieval: OMT on, Berklee off by default.

TF-IDF retrieval needs no embedding model, network call or LLM. Distances measure
textual relevance, not musical correctness. Notes are labeled source excerpts,
not generated explanations. Corpus preparation is a separate explicit step.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping

from app.models.schemas import TheoryCitation, TheoryExplanation
from app.services.env_flags import env_bool
from app.services.voice_leading import _parse as parse_chord

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3]
OMT_PATH = ROOT / "ai/data/openmusictheory/reviewed.json"
BERKLEE_PATH = ROOT / "ai/data/cache/berklee.chunks.jsonl"
MANIFEST_PATH = ROOT / "ai/rag/omt_manifest.json"
MAX_CORPUS_BYTES = 8 * 1024 * 1024
MAX_PASSAGES = 4096
STOP_WORDS = frozenset("a an and are as at be been but by for from has have in into is it its of on or that the their these this those to was we when which with would".split())


@dataclass(frozen=True)
class Policy:
    omt: bool
    berklee: bool
    omt_path: str
    berklee_path: str

    @property
    def sources(self) -> tuple[str, ...]:
        return tuple(name for name, enabled in (("omt", self.omt), ("berklee", self.berklee)) if enabled)


def policy(env: Mapping[str, str] | None = None) -> Policy:
    env = os.environ if env is None else env
    disabled = env_bool("THEARTIST_DISABLE_RAG", False, env)
    omt = env_bool("THEARTIST_RAG_OMT_ENABLED", True, env)
    berklee = env_bool("THEARTIST_RAG_BERKLEE_ENABLED", False, env)
    if berklee and not disabled and env_bool("THEARTIST_PUBLIC_HOST", False, env):
        raise ValueError("Berklee RAG is restricted to explicit private research; disable it on a public host")
    return Policy(
        omt and not disabled, berklee and not disabled,
        env.get("THEARTIST_RAG_OMT_PATH") or str(OMT_PATH),
        env.get("THEARTIST_RAG_BERKLEE_PATH") or str(BERKLEE_PATH),
    )


def tokens(text: str) -> Counter[str]:
    # Keep Roman numerals; normalize ordinary plurals for lightweight prose lookup.
    words = re.findall(r"[a-z]+", text.lower())
    return Counter(word[:-1] if len(word) > 4 and word.endswith("s") else word
                   for word in words if word not in STOP_WORDS)


@dataclass(frozen=True)
class Passage:
    text: str
    section: str
    page_start: int
    page_end: int
    citation: TheoryCitation


@dataclass(frozen=True)
class Hit:
    passage: Passage
    distance: float


def transition_query(a: str, b: str, genre: str | None = None) -> str:
    """Retrieve related concepts; two chords alone do not establish key or cadence."""
    first, second = parse_chord(a), parse_chord(b)
    terms = ["harmonic progression"]
    if first and second:
        root_a, quality_a, bass_a = first
        root_b, quality_b, bass_b = second
        movement = (root_b - root_a) % 12
        minor_a = quality_a.startswith(("m", "min")) and not quality_a.startswith("maj")
        minor_b = quality_b.startswith(("m", "min")) and not quality_b.startswith("maj")
        dominant_a = quality_a.startswith(("7", "9", "11", "13"))
        dominant_b = quality_b.startswith(("7", "9", "11", "13"))
        if movement == 0:
            if minor_a != minor_b:
                terms.append("modal mixture parallel major minor")
            elif (bass_a if bass_a is not None else root_a) != (bass_b if bass_b is not None else root_b):
                terms.append("change bass prolongation inversion")
            else:
                terms.append("change figure prolongation repeated bass")
        elif movement == 5 and dominant_a:
            terms.append("dominant tonic resolution authentic cadence applied tonicization")
        elif movement == 5 and minor_a and dominant_b:
            terms.append("subdominant predominant dominant harmonic function")
        elif movement == 7 and not minor_a:
            terms.append("plagal subdominant tonic prolongation")
        elif "dim" in quality_a or quality_a == "m7b5":
            terms.append("diminished seventh leading tone dissonance applied")
        elif movement in {1, 11}:
            terms.append("chromatic stepwise passing bass voice leading")
        else:
            terms.append("harmonic function chord tendency")
        if bass_a is not None or bass_b is not None:
            terms.append("inversion bass voice leading")
    if genre in {"pop", "rock", "folk", "country", "electronic", "hip_hop"}:
        terms.append("pop rock schemata")
    return " ".join(terms)


class TheoryRetriever:
    def __init__(self, passages: list[Passage]):
        self.passages = tuple(passages)
        counts = [tokens(p.text + " " + p.section + " " + p.citation.title) for p in passages]
        document_frequency = Counter(word for count in counts for word in count)
        self.idf = {word: math.log((1 + len(counts)) / (1 + freq)) + 1 for word, freq in document_frequency.items()}
        self.vectors = tuple(self._vector(count) for count in counts)

    def _vector(self, counts: Counter[str]) -> dict[str, float]:
        weights = {word: (1 + math.log(count)) * self.idf[word] for word, count in counts.items() if word in self.idf}
        norm = math.sqrt(sum(weight * weight for weight in weights.values()))
        return {word: weight / norm for word, weight in weights.items()} if norm else {}

    def query(self, text: str, k: int = 1) -> list[Hit]:
        if not 1 <= k <= 10:
            raise ValueError("Theory retrieval k must be between 1 and 10")
        query = self._vector(tokens(text))
        scored = [(sum(weight * vector.get(word, 0) for word, weight in query.items()), i)
                  for i, vector in enumerate(self.vectors)]
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [Hit(self.passages[i], 1 - min(1.0, score)) for score, i in scored[:k] if score > 0]

    @lru_cache(maxsize=2048)
    def transition(self, a: str, b: str, genre: str | None = None) -> Hit | None:
        if parse_chord(a) is None or parse_chord(b) is None:
            return None
        hits = self.query(transition_query(a, b, genre))
        return hits[0] if hits else None

    def explain(self, a: str, b: str, genre: str | None = None) -> TheoryExplanation | None:
        hit = self.transition(a, b, genre)
        if hit is None:
            return None
        passage = hit.passage
        excerpt = passage.text
        if len(excerpt) > 400:
            excerpt = excerpt[:397].rsplit(" ", 1)[0] + "…"
        citation = passage.citation.model_copy(deep=True, update={"changes": passage.citation.changes + " Display excerpt may be shortened."})
        return TheoryExplanation(
            chord_a=a, chord_b=b, concept=passage.section or passage.citation.title,
            explanation=excerpt, chapter=passage.citation.title, section=passage.section,
            page_start=passage.page_start, page_end=passage.page_end,
            kind="source_excerpt", citation=citation,
        )


def _read(path: str) -> str:
    with Path(path).open("rb") as file:
        data = file.read(MAX_CORPUS_BYTES + 1)
    if len(data) > MAX_CORPUS_BYTES:
        raise ValueError("Theory corpus exceeds the 8 MiB limit")
    return data.decode("utf8")


def load_omt(path: str) -> list[Passage]:
    from hashlib import sha256
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf8"))
    allowed = {(page["url"], checksum): page for page in manifest["pages"] for checksum in page["paragraphs"]}
    corpus = json.loads(_read(path))
    if not isinstance(corpus, dict) or corpus.get("version") != 1 or corpus.get("source") != "omt" or not isinstance(corpus.get("passages"), list):
        raise ValueError("Not a reviewed OMT corpus")
    passages = []
    seen = set()
    for item in corpus["passages"]:
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            raise ValueError("OMT passage must contain prose text")
        checksum = sha256(item["text"].encode("utf8")).hexdigest()
        key = (item["url"], checksum)
        page = allowed.get(key)
        if page is None or key in seen or item.get("source") != "omt" or item.get("id") != checksum:
            raise ValueError("OMT corpus contains unreviewed, duplicate or mislabeled material")
        seen.add(key)
        # Attribution comes from the reviewed manifest, never untrusted corpus labels.
        citation = TheoryCitation(source="omt", title=page["title"], url=page["url"],
            authors=manifest["authors"], editors=manifest["editors"], license="CC BY-SA 4.0", license_url=manifest["license_url"],
            copyright=manifest["copyright"], changes="Selected prose; HTML formatting normalized; media excluded.", passage_id=checksum)
        passages.append(Passage(item["text"], page["sections"][checksum], 0, 0, citation))
    if seen != set(allowed):
        raise ValueError("OMT corpus is incomplete; rerun setup-omt")
    return passages


def load_berklee(path: str) -> list[Passage]:
    """Legacy private chunks only; never called unless Berklee is explicitly enabled."""
    passages = []
    for line in _read(path).splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict) or item.get("source") != "berklee" or not isinstance(item.get("text"), str):
            raise ValueError("Berklee corpus has an invalid source or text")
        citation = TheoryCitation(source="berklee", title="The Berklee Book of Jazz Harmony",
            authors=["Joe Mulholland", "Tom Hojnacki"], license="All rights reserved — private research only",
            passage_id=item["chunk_id"], changes="Selected private research excerpt.")
        passages.append(Passage(item["text"], item.get("section", ""), item.get("page_start", 0), item.get("page_end", 0), citation))
        if len(passages) > MAX_PASSAGES:
            raise ValueError("Too many Berklee passages")
    if not passages:
        raise ValueError("Berklee corpus is empty")
    return passages


def _signature(path: str) -> tuple[int, int] | None:
    try:
        stat = Path(path).stat()
        return stat.st_mtime_ns, stat.st_size
    except OSError:
        return None


@lru_cache(maxsize=8)
def _runtime(settings: Policy, signatures: tuple) -> tuple[TheoryRetriever | None, tuple[tuple[str, str], ...]]:
    passages, statuses = [], []
    for name, enabled, path, loader in (
        ("omt", settings.omt, settings.omt_path, load_omt),
        ("berklee", settings.berklee, settings.berklee_path, load_berklee),
    ):
        if not enabled:
            statuses.append((name, "disabled"))
            continue
        try:
            passages.extend(loader(path))
            statuses.append((name, "ready"))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            state = "missing" if isinstance(exc, FileNotFoundError) else "invalid"
            statuses.append((name, state))
            log.warning("%s theory corpus %s; retrieval for this source is unavailable (%s)", name, state, type(exc).__name__)
    return (TheoryRetriever(passages) if passages else None), tuple(statuses)


def runtime(env: Mapping[str, str] | None = None):
    settings = policy(env)
    # Disabled corpora are not even stat'ed. A change in source policy or corpus
    # invalidates the whole index and its bounded per-transition cache.
    signatures = tuple(_signature(path) for enabled, path in ((settings.omt, settings.omt_path), (settings.berklee, settings.berklee_path)) if enabled)
    if settings.omt:
        signatures += (_signature(str(MANIFEST_PATH)),)
    return _runtime(settings, signatures)


def status() -> dict:
    try:
        retriever, states = runtime()
        return {"enabled": bool(policy().sources), "ready": retriever is not None,
                "sources": dict(states), "method": "local_tfidf", "notes": "source_excerpt"}
    except ValueError:
        return {"enabled": False, "ready": False, "sources": {}, "error": "invalid_source_configuration"}
