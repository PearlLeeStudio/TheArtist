"""PDF → page-level text extraction.

Wraps pypdf with a small dataclass record so downstream chunkers can
preserve page numbers and source filenames for citation. Strips form-feed
characters and collapses runs of internal whitespace; preserves paragraph
breaks (double newlines) so the chunker can split on them.

CLI:
    cd ai && uv run python -m rag.extract <pdf_path> [--out <jsonl>]

Default output is `ai/data/cache/<stem>.pages.jsonl`.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from pypdf import PdfReader

_AI_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CACHE = _AI_ROOT / "data" / "cache"

_WS_RE = re.compile(r"[ \t\f\v]+")
_NL3PLUS_RE = re.compile(r"\n{3,}")


@dataclass
class PageRecord:
    source: str          # filename (no path)
    page: int            # 1-indexed
    text: str            # cleaned text


def _clean(text: str) -> str:
    text = text.replace("\f", "\n\n")
    text = _WS_RE.sub(" ", text)
    text = _NL3PLUS_RE.sub("\n\n", text)
    return text.strip()


def extract_pdf(path: Path) -> list[PageRecord]:
    """Extract one PageRecord per page; skip pages whose extraction is empty."""
    reader = PdfReader(str(path))
    out: list[PageRecord] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception:
            raw = ""
        cleaned = _clean(raw)
        if not cleaned:
            continue
        out.append(PageRecord(source=path.name, page=i, text=cleaned))
    return out


def write_jsonl(records: list[PageRecord], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pdf", type=Path)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    if not args.pdf.exists():
        raise FileNotFoundError(args.pdf)

    out = args.out or (_DEFAULT_CACHE / f"{args.pdf.stem}.pages.jsonl")
    records = extract_pdf(args.pdf)
    write_jsonl(records, out)
    print(f"{args.pdf.name}: extracted {len(records)} non-empty pages → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
