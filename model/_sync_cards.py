#!/usr/bin/env -S uv run --with huggingface_hub python
"""Sync local model cards with the latest Hub state, then normalize frontmatter.

This is the "read the latest model card and reflect onto it" step. Run it
*before* any card-patch / regenerate / re-upload flow so those scripts start
from whatever is currently live on the Hub (e.g. a merged community PR that
added a `pipeline_tag` or links), instead of clobbering it with a stale local
copy.

Two independent actions, both idempotent:

  --pull   Download each repo's current README.md from the Hub and overwrite
           the local model/<folder>/README.md. This is the literal "read the
           latest card" — it pulls in changes merged on the Hub (e.g. nielsr's
           pipeline-tag PR) so the local copy is no longer stale.

  (always) Ensure `pipeline_tag: text-generation` is present in each local
           card's YAML frontmatter. Inserted right after `library_name:` if
           missing; left untouched if already there. This guarantees the tag
           survives even for cards whose Hub copy never received the PR.

Typical batch-update flow (e.g. adding a demo-video link to all 17 cards):

    uv run python model/_sync_cards.py --pull        # 1. read latest from Hub + ensure tag
    uv run python model/_patch_usage_blocks.py        # 2. apply the batch edit locally
    uv run python model/_patch_readmes.py             # 3. push READMEs back to the Hub

The full-regen generators (`_generate_lora_cards.py`, `_release_best_ranks.py`)
now bake `pipeline_tag: text-generation` into their templates, so a from-scratch
LoRA regeneration also keeps the tag. The F-series updater
(`_release_f_series_cards.py`) already reads the local card and only replaces
its marked eval block, so it preserves frontmatter + body untouched.

Usage:
    uv run python model/_sync_cards.py                 # ensure tag in all 17 (offline)
    uv run python model/_sync_cards.py --pull          # pull latest from Hub first, then ensure
    uv run python model/_sync_cards.py --only-base     # 6 F-series repos only
    uv run python model/_sync_cards.py --only-loras    # 11 LoRA repos only
    uv run python model/_sync_cards.py --repos ft_jazz_pop80
    uv run python model/_sync_cards.py --pull --dry-run
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

# Single source of truth for the folder -> Hub-repo mapping (avoids drift).
from _patch_readmes import LORAS, ORG, RUNS

MODEL_DIR = Path(__file__).resolve().parent
PIPELINE_TAG = "text-generation"


def ensure_pipeline_tag(text: str, tag: str = PIPELINE_TAG) -> tuple[str, bool]:
    """Insert `pipeline_tag: <tag>` into the leading YAML frontmatter if absent.

    Idempotent: returns (text, False) when the key already exists or there is
    no frontmatter to edit. Only the frontmatter block is touched; body is
    never modified.
    """
    m = re.match(r"^---\n(.*?)\n---\n", text, flags=re.DOTALL)
    if not m:
        return text, False
    fm = m.group(1)
    if re.search(r"^pipeline_tag:", fm, flags=re.MULTILINE):
        return text, False

    lines = fm.split("\n")
    insert_at = None
    for i, ln in enumerate(lines):  # prefer right after library_name:
        if ln.startswith("library_name:"):
            insert_at = i + 1
            break
    if insert_at is None:           # fall back to after license:
        for i, ln in enumerate(lines):
            if ln.startswith("license:"):
                insert_at = i + 1
                break
    if insert_at is None:           # last resort: top of frontmatter
        insert_at = 0

    lines.insert(insert_at, f"pipeline_tag: {tag}")
    new_fm = "\n".join(lines)
    return "---\n" + new_fm + "\n---\n" + text[m.end():], True


def pull_from_hub(repo_id: str, dest: Path, dry_run: bool) -> str:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

    try:
        cached = hf_hub_download(
            repo_id=repo_id, filename="README.md", repo_type="model",
            force_download=True,
        )
    except (EntryNotFoundError, RepositoryNotFoundError) as e:
        return f"  pull  {repo_id}: skip ({type(e).__name__}) — keeping local"
    if dry_run:
        return f"  pull  {repo_id}: would overwrite {dest.name} from Hub"
    shutil.copyfile(cached, dest)
    return f"  pull  {repo_id}: README.md ← Hub"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    scope = ap.add_mutually_exclusive_group()
    scope.add_argument("--only-base", action="store_true", help="6 F-series repos only")
    scope.add_argument("--only-loras", action="store_true", help="11 LoRA repos only")
    ap.add_argument("--repos", nargs="+", help="explicit folder names (overrides scope)")
    ap.add_argument("--pull", action="store_true",
                    help="download each repo's current README from the Hub first")
    ap.add_argument("--dry-run", action="store_true", help="report only, no writes")
    args = ap.parse_args()

    if args.repos:
        all_pairs = dict(RUNS + LORAS)
        missing = [f for f in args.repos if f not in all_pairs]
        if missing:
            ap.error(f"unknown folder(s): {', '.join(missing)}")
        jobs = [(f, all_pairs[f]) for f in args.repos]
    else:
        jobs = []
        if not args.only_loras:
            jobs += RUNS
        if not args.only_base:
            jobs += LORAS

    print(f"Syncing {len(jobs)} cards"
          + (" with Hub pull" if args.pull else " (frontmatter only)")
          + (" [dry-run]" if args.dry_run else "") + " ...")

    pulled = tagged = 0
    for folder, repo_name in jobs:
        repo_id = f"{ORG}/{repo_name}"
        readme = MODEL_DIR / folder / "README.md"

        if args.pull:
            msg = pull_from_hub(repo_id, readme, args.dry_run)
            print(msg)
            if "← Hub" in msg or "would overwrite" in msg:
                pulled += 1

        if not readme.exists():
            print(f"  tag   {folder}: skip — no README.md")
            continue
        original = readme.read_text()
        updated, changed = ensure_pipeline_tag(original)
        if changed:
            tagged += 1
            print(f"  tag   {folder}: + pipeline_tag: {PIPELINE_TAG}")
            if not args.dry_run:
                readme.write_text(updated)
        else:
            print(f"  tag   {folder}: ok (already present)")

    print(f"\nDone. pulled {pulled}, tag-inserted {tagged}"
          + (" (dry-run — nothing written)" if args.dry_run else "")
          + ".\nNext: edit cards as needed, then `python model/_patch_readmes.py` to push.")


if __name__ == "__main__":
    main()
