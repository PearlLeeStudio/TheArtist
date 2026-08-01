#!/usr/bin/env -S uv run --with huggingface_hub python
"""List (and optionally merge) open community Pull Requests across all 17
TheArtist model repos on the Hub.

nielsr-style "improve the model card" PRs (pipeline_tag, links) often land on
every repo in a collection at once. This enumerates them so you can see what
is open before accepting, then merges them in bulk.

Read-only by default. `--merge` performs the merges (needs a write token:
`hf auth login` or HF_TOKEN). After merging, run
`python model/_sync_cards.py --pull` to bring the merged Hub state down to your
local cards.

Usage:
    uv run python model/_merge_prs.py                 # list all open PRs (read-only)
    uv run python model/_merge_prs.py --author nielsr  # only PRs by this author
    uv run python model/_merge_prs.py --merge          # merge every open PR in scope
    uv run python model/_merge_prs.py --merge --author nielsr --dry-run
    uv run python model/_merge_prs.py --only-base --merge
"""
from __future__ import annotations

import argparse

from huggingface_hub import HfApi

# Single source of truth for the folder -> Hub-repo mapping (avoids drift).
from _patch_readmes import LORAS, ORG, RUNS


def collect_open_prs(api: HfApi, repo_id: str, author: str | None) -> list:
    out = []
    for d in api.get_repo_discussions(repo_id=repo_id, repo_type="model"):
        if not d.is_pull_request or d.status != "open":
            continue
        if author and author.lower() not in (d.author or "").lower():
            continue
        out.append(d)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    scope = ap.add_mutually_exclusive_group()
    scope.add_argument("--only-base", action="store_true", help="6 F-series repos only")
    scope.add_argument("--only-loras", action="store_true", help="11 LoRA repos only")
    ap.add_argument("--repos", nargs="+", help="explicit folder names (overrides scope)")
    ap.add_argument("--author", default=None, help="restrict to PRs by this author (substring)")
    ap.add_argument("--merge", action="store_true", help="merge the listed PRs (write op)")
    ap.add_argument("--comment", default="Thanks! Merging — bulk accept across the collection.")
    ap.add_argument("--dry-run", action="store_true", help="with --merge: report only, no merge")
    args = ap.parse_args()

    if args.repos:
        all_pairs = dict(RUNS + LORAS)
        bad = [f for f in args.repos if f not in all_pairs]
        if bad:
            ap.error(f"unknown folder(s): {', '.join(bad)}")
        jobs = [(f, all_pairs[f]) for f in args.repos]
    else:
        jobs = []
        if not args.only_loras:
            jobs += RUNS
        if not args.only_base:
            jobs += LORAS

    api = HfApi()
    who = "anonymous"
    try:
        who = api.whoami().get("name", "anonymous")
    except Exception:
        pass
    print(f"Auth: {who}   |   scanning {len(jobs)} repos under {ORG}/"
          + (f"   |   author~='{args.author}'" if args.author else "") + "\n")

    total_open = 0
    to_merge: list[tuple[str, int, str]] = []
    for folder, repo_name in jobs:
        repo_id = f"{ORG}/{repo_name}"
        try:
            prs = collect_open_prs(api, repo_id, args.author)
        except Exception as e:
            print(f"  {repo_name:<48} ERROR {type(e).__name__}: {e}")
            continue
        if not prs:
            print(f"  {repo_name:<48} —")
            continue
        for d in prs:
            total_open += 1
            print(f"  {repo_name:<48} #{d.num} \"{d.title}\"  by {d.author} [{d.status}]")
            to_merge.append((repo_id, d.num, d.title))

    print(f"\n{total_open} open PR(s) in scope.")

    if not args.merge:
        if total_open:
            print("Re-run with --merge to accept them all "
                  "(then `python model/_sync_cards.py --pull`).")
        return

    if who == "anonymous":
        raise SystemExit("\n[fail] --merge needs a write token. Run `hf auth login` "
                         "or set HF_TOKEN, then retry.")

    print(f"\n=== {'DRY-RUN: would merge' if args.dry_run else 'Merging'} "
          f"{len(to_merge)} PR(s) ===")
    merged = 0
    for repo_id, num, title in to_merge:
        if args.dry_run:
            print(f"  would merge {repo_id} #{num}")
            continue
        try:
            api.merge_pull_request(repo_id=repo_id, discussion_num=num,
                                   repo_type="model", comment=args.comment)
            merged += 1
            print(f"  merged  {repo_id} #{num}")
        except Exception as e:
            print(f"  FAILED  {repo_id} #{num}: {type(e).__name__}: {e}")

    if not args.dry_run:
        print(f"\nMerged {merged}/{len(to_merge)}. "
              "Now run: python model/_sync_cards.py --pull")


if __name__ == "__main__":
    main()
