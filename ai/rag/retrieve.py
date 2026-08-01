"""Top-k retrieval from one or more chromadb RAG collections.

Embeds the query with the same Snowflake arctic-embed-l-v2 model used at
indexing time (with the `query: ` prefix per Snowflake convention) and
returns the highest-similarity chunks with their chapter/section
context. Multi-collection mode queries each collection with the same
embedding, then merges the results by distance to take the global top k.

Programmatic use::

    from rag.retrieve import Retriever
    r = Retriever(collections=["berklee", "openmusictheory"])
    hits = r.query("Picardy third", k=5)

CLI::

    cd ai && uv run python -m rag.retrieve "modal interchange"
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import chromadb
import torch
from sentence_transformers import SentenceTransformer

from .embed import _DEFAULT_DB_DIR, _DEFAULT_MODEL, QUERY_PREFIX

_DEFAULT_COLLECTION = "berklee"


@dataclass
class Hit:
    chunk_id: str
    distance: float
    chapter: str
    section: str
    page_start: int
    page_end: int
    text: str
    source: str = ""  # collection / book name (e.g., "berklee", "openmusictheory")


class Retriever:
    """Query-time wrapper around chromadb + Snowflake arctic embedder.

    Accepts either a single collection name (legacy) or a list of names.
    Multi-collection retrieval queries each, merges by distance, returns top k.
    """

    def __init__(
        self,
        collection: str | None = None,
        collections: list[str] | tuple[str, ...] | None = None,
        db_dir: Path = _DEFAULT_DB_DIR,
        model_name: str = _DEFAULT_MODEL,
        device: str | None = None,
    ) -> None:
        if collections is None:
            collections = [collection or _DEFAULT_COLLECTION]
        self.client = chromadb.PersistentClient(path=str(db_dir))
        self.colls = [(name, self.client.get_collection(name)) for name in collections]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, trust_remote_code=True, device=self.device)

    def query(self, text: str, k: int = 5) -> list[Hit]:
        emb = self.model.encode(
            [QUERY_PREFIX + text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0].tolist()
        merged: list[Hit] = []
        for name, coll in self.colls:
            res = coll.query(query_embeddings=[emb], n_results=k)
            for cid, dist, meta, doc in zip(
                res["ids"][0], res["distances"][0], res["metadatas"][0], res["documents"][0]
            ):
                merged.append(Hit(
                    chunk_id=cid,
                    distance=dist,
                    chapter=meta.get("chapter", ""),
                    section=meta.get("section", ""),
                    page_start=meta.get("page_start", 0),
                    page_end=meta.get("page_end", 0),
                    text=doc,
                    source=name,
                ))
        merged.sort(key=lambda h: h.distance)
        return merged[:k]


def _format(hit: Hit, snippet_chars: int = 320) -> str:
    head = (hit.chapter or "(no chapter)") + " · " + (hit.section or "(no section)")
    pages = f"p{hit.page_start}" + (f"-{hit.page_end}" if hit.page_end != hit.page_start else "")
    src = f"[{hit.source}] " if hit.source else ""
    snippet = textwrap.shorten(hit.text.replace("\n", " "), width=snippet_chars, placeholder="…")
    return f"  [{hit.distance:.4f}] {src}{pages} | {head}\n    {snippet}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("query", nargs="+")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--collection", action="append", default=None,
                    help="Collection to query. Pass multiple times to merge across books.")
    args = ap.parse_args()

    cols = args.collection or [_DEFAULT_COLLECTION]
    r = Retriever(collections=cols)
    text = " ".join(args.query)
    hits = r.query(text, k=args.k)
    print(f'\nQuery: "{text}"   (top {len(hits)} hits across {cols})\n')
    for h in hits:
        print(_format(h))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
