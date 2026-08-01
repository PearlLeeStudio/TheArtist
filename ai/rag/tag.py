"""LLM topic-tagging via a local OpenAI-compatible HTTP server.

Sends each RAG chunk to a locally-served Gemma 3 (or compatible) model and
asks for a JSON object with primary topics, secondary topics, chord
qualities, and functional categories. Writes per-chunk results to a JSONL
cache so reruns are resumable; after the cache is complete, re-upserts the
chromadb collection's metadata with the new tags.

Default backend: **ollama** at http://localhost:11434/v1 (WSL-native).
Also works with LM Studio on the Windows side at http://localhost:1234/v1
under WSL2 mirrored networking — pass `--base-url` to switch.

Prerequisites (ollama path):
    curl -fsSL https://ollama.com/install.sh | sh
    ollama pull gemma3:12b   # ~7 GB, auto Q4_K_M

CLI::

    cd ai && uv run python -m rag.tag data/cache/<stem>.chunks.jsonl
    cd ai && uv run python -m rag.tag data/cache/<stem>.chunks.jsonl \\
        --base-url http://localhost:1234/v1 --model gemma-3-12b-it   # LM Studio
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import chromadb
from openai import OpenAI

from .embed import _DEFAULT_DB_DIR, _collection_name

_DEFAULT_BASE_URL = "http://localhost:11434/v1"  # ollama
_DEFAULT_MODEL = "qwen3.6:27b-q4_K_M"
_DEFAULT_TIMEOUT = 60.0

SYSTEM_PROMPT = """You are a music-harmony analyst. Read a passage from a music theory textbook (jazz, classical, or popular harmony) and return ONLY a JSON object with these fields:

- primary_topics:        array of 1-3 main harmonic concepts (e.g., "tritone substitution", "modal interchange", "secondary dominants", "voice leading", "rhythm changes", "picardy third", "chromatic mediant", "neapolitan chord", "augmented sixth chord", "plagal cadence", "half cadence", "deceptive cadence", "pedal point", "pivot chord modulation", "common-tone modulation").
- secondary_topics:      array of 0-3 supporting concepts mentioned in passing.
- chord_qualities:       array of chord qualities discussed by name (e.g., "maj7", "m7", "7", "m7b5", "dim7", "alt", "sus4", "13").
- functional_categories: array drawn ONLY from {"tonic", "subdominant", "dominant", "subdominant minor", "modal", "passing", "approach", "turnaround", "cadence"}.
- modes:                 array drawn ONLY from {"ionian", "dorian", "phrygian", "lydian", "mixolydian", "aeolian", "locrian"} — only modes explicitly named or characterised in the passage. Empty array if none.

Return strict JSON. No commentary. Use lowercase tag strings."""

USER_TEMPLATE = "Passage:\n\n{text}"


def _tag_one(client: OpenAI, model: str, text: str, timeout: float) -> dict:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(text=text)},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
        timeout=timeout,
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: salvage a {...} substring.
        i, j = raw.find("{"), raw.rfind("}")
        if i >= 0 and j > i:
            return json.loads(raw[i : j + 1])
        return {}


def _load_cache(cache_path: Path) -> dict[int, dict]:
    if not cache_path.exists():
        return {}
    out = {}
    for line in cache_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        out[rec["chunk_id"]] = rec
    return out


def _flatten_metadata(tags: dict, base_meta: dict) -> dict:
    """chromadb metadata must be primitives; collapse list fields to comma-separated strings."""
    out = dict(base_meta)
    for key in ("primary_topics", "secondary_topics", "chord_qualities", "functional_categories", "modes"):
        vals = tags.get(key) or []
        if isinstance(vals, str):
            vals = [vals]
        out[f"tag_{key}"] = ", ".join(str(v).lower() for v in vals)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("chunks_jsonl", type=Path)
    ap.add_argument("--db-dir", type=Path, default=_DEFAULT_DB_DIR)
    ap.add_argument("--base-url", default=_DEFAULT_BASE_URL)
    ap.add_argument("--model", default=_DEFAULT_MODEL)
    ap.add_argument("--timeout", type=float, default=_DEFAULT_TIMEOUT)
    ap.add_argument("--cache", type=Path, default=None,
                    help="JSONL cache path (defaults next to chunks file)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Tag only the first N chunks (smoke test)")
    args = ap.parse_args()

    if not args.chunks_jsonl.exists():
        raise FileNotFoundError(args.chunks_jsonl)

    chunks = [
        json.loads(line)
        for line in args.chunks_jsonl.read_text().splitlines()
        if line.strip()
    ]
    if args.limit:
        chunks = chunks[: args.limit]

    cache_path = args.cache or args.chunks_jsonl.with_suffix(".tags.jsonl").with_name(
        args.chunks_jsonl.stem.replace(".chunks", "") + ".tags.jsonl"
    )
    cache = _load_cache(cache_path)
    print(f"loaded {len(cache)} cached tags from {cache_path}")

    client = OpenAI(base_url=args.base_url, api_key="ollama")

    todo = [c for c in chunks if c["chunk_id"] not in cache]
    print(f"tagging {len(todo)} new chunks via {args.base_url} ({args.model})")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    with cache_path.open("a") as cache_fp:
        for i, c in enumerate(todo, start=1):
            t0 = time.perf_counter()
            try:
                tags = _tag_one(client, args.model, c["text"], args.timeout)
            except Exception as e:
                print(f"  [{i}/{len(todo)}] chunk {c['chunk_id']}: ERROR {type(e).__name__}: {e}")
                continue
            rec = {"chunk_id": c["chunk_id"], "tags": tags}
            cache[c["chunk_id"]] = rec
            cache_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
            cache_fp.flush()
            dt = time.perf_counter() - t0
            primary = ", ".join(tags.get("primary_topics", [])[:2])
            print(f"  [{i}/{len(todo)}] chunk {c['chunk_id']:3d}  {dt:5.1f}s  primary=[{primary}]")
    print(f"\ntagging done in {time.perf_counter() - t_start:.0f}s; cache size {len(cache)}")

    # Re-upsert chromadb metadata with new tags.
    coll_name = _collection_name(args.chunks_jsonl)
    db = chromadb.PersistentClient(path=str(args.db_dir))
    try:
        coll = db.get_collection(coll_name)
    except Exception:
        print(f"collection '{coll_name}' not found; run rag.embed first")
        return 1

    ids = [f"{c['source']}::{c['chunk_id']}" for c in chunks if c["chunk_id"] in cache]
    metas = [
        _flatten_metadata(
            cache[c["chunk_id"]]["tags"],
            {
                "source": c["source"],
                "chunk_id": c["chunk_id"],
                "page_start": c["page_start"],
                "page_end": c["page_end"],
                "chapter": c["chapter"],
                "section": c["section"],
                "word_count": c["word_count"],
            },
        )
        for c in chunks
        if c["chunk_id"] in cache
    ]
    coll.update(ids=ids, metadatas=metas)
    print(f"updated {len(ids)} chromadb metadata entries in '{coll_name}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
