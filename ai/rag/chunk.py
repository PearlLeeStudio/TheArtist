"""Paragraph-level chunker for the RAG corpus.

Reads page records (JSONL from `extract.py`), groups paragraphs into
chunks targeting `target_words` (default 250), splits any single
paragraph longer than `max_words` at sentence boundaries, and tags each
chunk with the most recently seen chapter and section heading. Output is
JSONL ready for embedding + LLM topic-tagging downstream.

CLI::

    cd ai && uv run python -m rag.chunk data/cache/<stem>.pages.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

_CHAPTER_RE = re.compile(r"^\s*CHAPTER\s+\d+\b.*", re.IGNORECASE)
# Section heading heuristic: short all-caps line, no terminal punctuation.
_SECTION_RE = re.compile(r"^[A-Z][A-Z0-9\s\-:&'/]{4,80}$")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class ChunkRecord:
    source: str
    chunk_id: int
    page_start: int
    page_end: int
    chapter: str
    section: str
    text: str
    word_count: int


def _word_count(s: str) -> int:
    return len(s.split())


def _is_chapter(line: str) -> bool:
    return bool(_CHAPTER_RE.match(line))


def _is_section(line: str) -> bool:
    s = line.strip()
    if len(s) < 5 or len(s) > 80 or s.endswith((".", ",", ";", ":")):
        return False
    return bool(_SECTION_RE.match(s))


def _walk_paragraphs(pages: list[dict]) -> list[dict]:
    """Yield ordered paragraph records carrying current chapter/section context."""
    out = []
    chapter = ""
    section = ""
    for p in pages:
        # Update headings from any matching lines on this page (in order).
        for line in p["text"].split("\n"):
            stripped = line.strip()
            if _is_chapter(stripped):
                chapter = stripped
            elif _is_section(stripped):
                section = stripped
        # Split body into paragraphs on blank lines.
        for para in (q.strip() for q in p["text"].split("\n\n")):
            if not para:
                continue
            # Skip paragraphs that are nothing but a heading.
            head_only = all(
                _is_chapter(l.strip()) or _is_section(l.strip())
                for l in para.split("\n")
                if l.strip()
            )
            if head_only:
                continue
            out.append({
                "para": para,
                "page": p["page"],
                "chapter": chapter,
                "section": section,
            })
    return out


def _split_long(para: str, max_words: int) -> list[str]:
    sentences = _SENTENCE_RE.split(para)
    chunks: list[str] = []
    cur: list[str] = []
    cur_w = 0
    for sent in sentences:
        w = _word_count(sent)
        if cur and cur_w + w > max_words:
            chunks.append(" ".join(cur))
            cur, cur_w = [sent], w
        else:
            cur.append(sent)
            cur_w += w
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def chunk_pages(
    pages: list[dict],
    target_words: int = 250,
    min_words: int = 80,
    max_words: int = 450,
) -> list[ChunkRecord]:
    """Group page paragraphs into chunks of ~target_words; preserve section context."""
    if not pages:
        return []
    source = pages[0]["source"]
    items = _walk_paragraphs(pages)

    chunks: list[ChunkRecord] = []
    buf_paras: list[str] = []
    buf_words = 0
    buf_pages: set[int] = set()
    buf_chapter = ""
    buf_section = ""
    chunk_id = 0

    def flush() -> None:
        nonlocal chunk_id, buf_paras, buf_words, buf_pages
        if not buf_paras:
            return
        pgs = sorted(buf_pages)
        chunks.append(ChunkRecord(
            source=source,
            chunk_id=chunk_id,
            page_start=pgs[0],
            page_end=pgs[-1],
            chapter=buf_chapter,
            section=buf_section,
            text="\n\n".join(buf_paras),
            word_count=buf_words,
        ))
        chunk_id += 1
        buf_paras = []
        buf_words = 0
        buf_pages = set()

    for it in items:
        para, w = it["para"], _word_count(it["para"])

        if buf_paras and (it["chapter"] != buf_chapter or it["section"] != buf_section):
            flush()

        if w > max_words:
            flush()
            for piece in _split_long(para, max_words):
                pw = _word_count(piece)
                chunks.append(ChunkRecord(
                    source=source,
                    chunk_id=chunk_id,
                    page_start=it["page"],
                    page_end=it["page"],
                    chapter=it["chapter"],
                    section=it["section"],
                    text=piece,
                    word_count=pw,
                ))
                chunk_id += 1
            buf_chapter, buf_section = it["chapter"], it["section"]
            continue

        if buf_words + w > max_words and buf_words >= min_words:
            flush()

        buf_paras.append(para)
        buf_words += w
        buf_pages.add(it["page"])
        buf_chapter, buf_section = it["chapter"], it["section"]

        if buf_words >= target_words:
            flush()

    flush()
    return chunks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pages_jsonl", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--target-words", type=int, default=250)
    ap.add_argument("--min-words", type=int, default=80)
    ap.add_argument("--max-words", type=int, default=450)
    args = ap.parse_args()

    pages = [json.loads(l) for l in args.pages_jsonl.read_text().splitlines() if l.strip()]
    chunks = chunk_pages(pages, args.target_words, args.min_words, args.max_words)

    out = args.out or args.pages_jsonl.with_suffix(".chunks.jsonl").with_name(
        args.pages_jsonl.stem.replace(".pages", "") + ".chunks.jsonl"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

    total_words = sum(c.word_count for c in chunks)
    avg = total_words // max(1, len(chunks))
    print(f"{args.pages_jsonl.name}: {len(chunks)} chunks, {total_words} words, avg {avg} → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
