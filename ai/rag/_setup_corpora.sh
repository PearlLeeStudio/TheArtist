#!/bin/bash
# Fetch the source PDFs and run the RAG indexing pipeline end-to-end.
#
# RAG corpora live at:
#   ai/data/berklee/         The BERKLEE book of JAZZ HARMONY.pdf  (manual: copyright)
#   ai/data/openmusictheory/ OpenMusicTheory.pdf                    (CC BY-SA 4.0)
#
# Both PDF directories are gitignored. Re-running this on a fresh checkout
# downloads the open one and runs extract → chunk → embed → tag for both,
# leaving chromadb collections berklee + openmusictheory ready for the
# Retriever in backend/app/services/inference.py.
#
# Berklee book is copyrighted; you must place your own legitimate copy at
# ai/data/berklee/The BERKLEE book of JAZZ HARMONY.pdf before running.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AI_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$AI_DIR"

OMT_DIR="data/openmusictheory"
OMT_PDF="$OMT_DIR/OpenMusicTheory.pdf"
OMT_URL="https://viva.pressbooks.pub/openmusictheory/open/download?type=pdf"

if [ ! -f "$OMT_PDF" ]; then
  echo "Downloading Open Music Theory v2 (CC BY-SA 4.0) ..."
  mkdir -p "$OMT_DIR"
  curl -sL -o "$OMT_PDF" "$OMT_URL"
fi
echo "  $(ls -lh "$OMT_PDF" | awk '{print $5, $9}')"

BERKLEE_PDF="data/berklee/The BERKLEE book of JAZZ HARMONY.pdf"
if [ ! -f "$BERKLEE_PDF" ]; then
  echo
  echo "ERROR: $BERKLEE_PDF is missing. Place a legitimate copy there before continuing."
  echo "       (Copyright; we cannot redistribute.)"
  exit 1
fi

echo
echo "=== extract ==="
uv run python -m rag.extract "$BERKLEE_PDF"             --out data/cache/berklee.pages.jsonl
uv run python -m rag.extract "$OMT_PDF"                 --out data/cache/openmusictheory.pages.jsonl

echo
echo "=== chunk ==="
uv run python -m rag.chunk data/cache/berklee.pages.jsonl
uv run python -m rag.chunk data/cache/openmusictheory.pages.jsonl

echo
echo "=== embed (Snowflake arctic-embed-l-v2) ==="
uv run python -m rag.embed data/cache/berklee.chunks.jsonl
uv run python -m rag.embed data/cache/openmusictheory.chunks.jsonl

echo
echo "=== tag (qwen3.6:27b-q4_K_M via ollama) ==="
echo "  Berklee (~268 chunks, ~25 min) ..."
uv run python -m rag.tag data/cache/berklee.chunks.jsonl
echo "  OMT (~1198 chunks, ~100 min) ..."
uv run python -m rag.tag data/cache/openmusictheory.chunks.jsonl

echo
echo "Done. Verify:"
echo "  uv run python -m rag.retrieve --collection berklee --collection openmusictheory \"Picardy third\""
