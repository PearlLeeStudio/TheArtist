"""Embed RAG chunks with Snowflake/snowflake-arctic-embed-l-v2.0 → persist to chromadb.

Arctic-embed-l-v2.0 is a 568M-parameter BERT-style embedder, 1024 dim,
top of MTEB English (~71). Stable with current transformers; loads
cleanly without trust_remote_code custom code paths. Runs in fp16 on a
consumer GPU at ~1.5 GB VRAM.

For queries use the `query: ` prefix (Snowflake convention); passages
have no prefix.

CLI::

    cd ai && uv run python -m rag.embed data/cache/<stem>.chunks.jsonl
    # → vector store at data/cache/chroma_db/, collection name = stem
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import chromadb
import torch
from sentence_transformers import SentenceTransformer

_AI_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DB_DIR = _AI_ROOT / "data" / "cache" / "chroma_db"
_DEFAULT_MODEL = "Snowflake/snowflake-arctic-embed-l-v2.0"
QUERY_PREFIX = "query: "  # exported for retriever use


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _read_chunks(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _collection_name(chunks_path: Path) -> str:
    """Derive a chromadb collection name from the chunk file stem."""
    stem = chunks_path.stem.removesuffix(".chunks")
    return "".join(c if c.isalnum() else "_" for c in stem.lower())[:60].strip("_")


def embed_and_store(
    chunks_path: Path,
    db_dir: Path = _DEFAULT_DB_DIR,
    model_name: str = _DEFAULT_MODEL,
    batch_size: int = 16,
) -> str:
    """Embed `chunks_path` JSONL and upsert into chromadb.

    Returns the collection name.
    """
    chunks = _read_chunks(chunks_path)
    if not chunks:
        raise ValueError(f"no chunks in {chunks_path}")

    db_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(db_dir))
    coll_name = _collection_name(chunks_path)

    device = _device()
    print(f"loading {model_name} on {device} …")
    t0 = time.perf_counter()
    model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
    dim = model.get_sentence_embedding_dimension()
    print(f"  loaded in {time.perf_counter() - t0:.1f}s, dim={dim}")

    coll = client.get_or_create_collection(
        name=coll_name,
        metadata={"hnsw:space": "cosine", "embed_model": model_name, "dim": dim},
    )

    texts = [c["text"] for c in chunks]
    ids = [f"{c['source']}::{c['chunk_id']}" for c in chunks]
    metas = [
        {
            "source": c["source"],
            "chunk_id": c["chunk_id"],
            "page_start": c["page_start"],
            "page_end": c["page_end"],
            "chapter": c["chapter"],
            "section": c["section"],
            "word_count": c["word_count"],
        }
        for c in chunks
    ]

    print(f"embedding {len(texts)} chunks (batch={batch_size}, dim={dim}) …")
    t0 = time.perf_counter()
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    elapsed = time.perf_counter() - t0
    print(f"  embedded in {elapsed:.1f}s ({len(texts) / elapsed:.1f} chunks/s)")

    coll.upsert(ids=ids, embeddings=emb.tolist(), documents=texts, metadatas=metas)
    print(f"upserted {len(ids)} chunks → collection {coll_name!r} at {db_dir}")
    return coll_name


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("chunks_jsonl", type=Path)
    ap.add_argument("--db-dir", type=Path, default=_DEFAULT_DB_DIR)
    ap.add_argument("--model", default=_DEFAULT_MODEL)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    if not args.chunks_jsonl.exists():
        raise FileNotFoundError(args.chunks_jsonl)

    embed_and_store(
        args.chunks_jsonl,
        db_dir=args.db_dir,
        model_name=args.model,
        batch_size=args.batch_size,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
