#!/usr/bin/env -S uv run --with huggingface_hub python
"""Inject a paper + code links line right after the H1 title in each model card.

Re-adds the *useful* part of nielsr's PR #1 (links to the HF Papers page and the
code repo) onto the full original cards — WITHOUT the content deletions that PR
also bundled in. Idempotent: skipped if the HF Papers link is already present.
Touches only the line right after the first '# ' heading; Demo section, eval
tables, licenses, and citation are left untouched.

Usage:
    uv run python model/_patch_links.py            # all 17
    uv run python model/_patch_links.py --dry-run
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from _patch_readmes import LORAS, RUNS  # folder list — single source of truth

MODEL_DIR = Path(__file__).resolve().parent
PAPER_URL = "https://huggingface.co/papers/2606.07334"
CODE_URL = "https://github.com/PearlLeeStudio/TheArtist"
LINKS_LINE = (
    "Presented in the paper [How Far Can Chord-Symbol Time-Series Adaptation "
    f"Carry Genre Identity?]({PAPER_URL}) · Code: "
    f"[github.com/PearlLeeStudio/TheArtist]({CODE_URL})"
)


def patch(text: str) -> tuple[str, bool]:
    if PAPER_URL in text:
        return text, False
    m = re.search(r"^# .+$", text, flags=re.MULTILINE)
    if not m:
        return text, False
    at = m.end()
    return text[:at] + "\n\n" + LINKS_LINE + text[at:], True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    n = 0
    for folder, _ in RUNS + LORAS:
        p = MODEL_DIR / folder / "README.md"
        if not p.exists():
            print(f"  skip {folder}: no README.md")
            continue
        text = p.read_text()
        new, changed = patch(text)
        if changed:
            n += 1
            print(f"  + links  {folder}")
            if not args.dry_run:
                p.write_text(new)
        else:
            print(f"  ok       {folder} (already has paper link)")
    print(f"\n{n} cards got the links line" + (" (dry-run)" if args.dry_run else "") + ".")


if __name__ == "__main__":
    main()
