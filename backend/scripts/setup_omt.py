"""Download only the hash-reviewed, licensed OMT prose used by the product."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ai"))
from rag.omt import setup

if __name__ == "__main__":
    setup()
