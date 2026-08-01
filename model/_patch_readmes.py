#!/usr/bin/env -S uv run --with huggingface_hub python
"""Patch README.md only on HuggingFace repos (no weight re-upload).

Use this after editing a model card in `model/<run>/README.md` to push
the new card to the Hub without touching weights, configs, or tokenizers.

Usage:
    python model/_patch_readmes.py                 # patch all 17 repos
    python model/_patch_readmes.py --only-loras    # 11 LoRA repos only
    python model/_patch_readmes.py --only-base     # 6 F-series repos only
    python model/_patch_readmes.py --repos ft_f1_lora_blues ft_jazz_pop80
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import HfApi

ORG = "PearlLeeStudio"
MODEL_DIR = Path(__file__).resolve().parent

RUNS = [
    ("phase0_pop_baseline", "TheArtist-MusicTransformer-pop-baseline"),
    ("ft_jazz_pop80",       "TheArtist-MusicTransformer-ft-pop80"),
    ("ft_jazz_pop80_v2",    "TheArtist-MusicTransformer-ft-pop80-v2"),
    ("ft_jazz_pop67",       "TheArtist-MusicTransformer-ft-pop67"),
    ("ft_jazz_pop50",       "TheArtist-MusicTransformer-ft-pop50"),
    ("ft_jazz_pop29",       "TheArtist-MusicTransformer-ft-pop29"),
    ("ft_jazz_only",        "TheArtist-MusicTransformer-ft-jazz-only"),
]

LORA_GENRES = [
    "country", "funk", "gospel", "rnb_soul", "hip_hop",
    "electronic", "folk", "classical", "rock", "blues", "bossa",
]
LORAS = [
    (f"ft_f1_lora_{g}", f"TheArtist-MusicTransformer-lora-{g.replace('_', '-')}")
    for g in LORA_GENRES
]


def _patch_one(api: HfApi, folder: str, repo_name: str, message: str) -> str:
    repo_id = f"{ORG}/{repo_name}"
    readme = MODEL_DIR / folder / "README.md"
    if not readme.exists():
        return f"  skip {folder}: no README.md at {readme}"
    api.upload_file(
        path_or_fileobj=str(readme),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message=message,
    )
    return f"  patched https://huggingface.co/{repo_id}"


def main() -> None:
    ap = argparse.ArgumentParser()
    scope = ap.add_mutually_exclusive_group()
    scope.add_argument("--only-base", action="store_true")
    scope.add_argument("--only-loras", action="store_true")
    ap.add_argument("--repos", nargs="+", help="explicit folder names (overrides scope flags)")
    ap.add_argument("--message", default="Update model card: per-genre LoRA eval table",
                    help="commit message used on the Hub")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    jobs: list[tuple[str, str]]
    if args.repos:
        all_pairs = dict(RUNS + LORAS)
        jobs = [(f, all_pairs[f]) for f in args.repos]
    else:
        jobs = []
        if not args.only_loras:
            jobs += RUNS
        if not args.only_base:
            jobs += LORAS

    api = HfApi()
    print(f"Patching {len(jobs)} repos (README.md only) under {ORG}/ ...")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_patch_one, api, f, r, args.message) for f, r in jobs]
        for fut in as_completed(futures):
            print(fut.result())


if __name__ == "__main__":
    main()
