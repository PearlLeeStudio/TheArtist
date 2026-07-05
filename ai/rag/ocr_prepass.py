"""Detect image-only PDFs and OCR them in-place with ocrmypdf.

Scans ``ai/data/physics/`` (configurable) with pypdf. For each PDF, if
no page yields extractable text, the file is flagged for OCR. Runs
``ocrmypdf --skip-text`` against flagged files (skip-text preserves any
existing text layer, OCRs only image-only pages).

Prereq (not installed automatically):
    sudo apt install ocrmypdf tesseract-ocr
    # or on macOS: brew install ocrmypdf

CLI::

    cd ai && uv run python3 -m rag.ocr_prepass               # scan + OCR all image-only
    cd ai && uv run python3 -m rag.ocr_prepass --dry-run     # scan only, no OCR
    cd ai && uv run python3 -m rag.ocr_prepass --dir data/berklee   # different source dir
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from pypdf import PdfReader

_AI_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DIR = _AI_ROOT / "data" / "physics"


def has_text_layer(pdf: Path) -> bool:
    """True iff any page of ``pdf`` yields non-empty text via pypdf."""
    try:
        reader = PdfReader(str(pdf))
    except Exception:
        return False
    for page in reader.pages:
        try:
            if (page.extract_text() or "").strip():
                return True
        except Exception:
            continue
    return False


def flag_image_only_pdfs(directory: Path) -> list[Path]:
    """Return paths to PDFs under ``directory`` that have no text layer."""
    pdfs = sorted(directory.glob("*.pdf"))
    return [p for p in pdfs if not has_text_layer(p)]


def run_ocr(pdf: Path, languages: str = "eng") -> bool:
    """Run ocrmypdf in-place on ``pdf``; return True on success."""
    cmd = ["ocrmypdf", "--skip-text", "--language", languages, str(pdf), str(pdf)]
    try:
        subprocess.run(cmd, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  OCR failed on {pdf.name}: {e}", file=sys.stderr)
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=_DEFAULT_DIR,
                    help="directory containing PDFs (default: ai/data/physics)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report image-only PDFs but do not run ocrmypdf")
    ap.add_argument("--languages", default="eng",
                    help="ocrmypdf --language (e.g. 'eng' or 'eng+fra')")
    args = ap.parse_args()

    if not args.dir.exists():
        print(f"directory not found: {args.dir}", file=sys.stderr)
        return 2

    flagged = flag_image_only_pdfs(args.dir)
    if not flagged:
        print(f"no image-only PDFs under {args.dir}")
        return 0

    print(f"{len(flagged)} image-only PDF(s) under {args.dir}:")
    for p in flagged:
        print(f"  {p.name}")

    if args.dry_run:
        return 0

    if shutil.which("ocrmypdf") is None:
        print("\nocrmypdf not installed; install with:", file=sys.stderr)
        print("    sudo apt install ocrmypdf tesseract-ocr", file=sys.stderr)
        print("    # or: brew install ocrmypdf", file=sys.stderr)
        return 3

    ok = 0
    for p in flagged:
        print(f"\nOCR: {p.name}")
        if run_ocr(p, languages=args.languages):
            ok += 1
    print(f"\n{ok}/{len(flagged)} OCR'd successfully.")
    return 0 if ok == len(flagged) else 4


if __name__ == "__main__":
    sys.exit(main())
