#!/usr/bin/env -S uv run --with huggingface_hub python
"""Upload TheArtist Music Transformer checkpoints (F-series) and per-genre
LoRA adapters to HuggingFace Hub.

Prerequisites:
  1. uv tool install huggingface_hub      (installs the `hf` CLI)
  2. hf auth login                         (write-scope token from
     https://huggingface.co/settings/tokens)
  3. python model/_generate_lora_cards.py  (writes per-LoRA model cards)
  4. bash model/_stage_weights.sh          (copies weights + code into each folder)

The PearlLeeStudio organization must exist on HuggingFace.

Usage:
    python model/_upload.py                # upload everything
    python model/_upload.py --only-loras   # skip the F-series base checkpoints
    python model/_upload.py --only-base    # skip the LoRA adapters
    python model/_upload.py --repos ft_jazz_pop80_v2   # selected folders only
"""
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from huggingface_hub import HfApi, create_repo

ORG = "PearlLeeStudio"

RUNS = [
    ("phase0_pop_baseline", "TheArtist-MusicTransformer-pop-baseline"),
    ("ft_jazz_pop80",       "TheArtist-MusicTransformer-ft-pop80"),
    # F1 selection-corrected retrain (select_best_on: jazz_val, 2026-06-12).
    # Separate repo on purpose: the 11 LoRA adapters pair with the v1 base above.
    ("ft_jazz_pop80_v2",    "TheArtist-MusicTransformer-ft-pop80-v2"),
    ("ft_jazz_pop67",       "TheArtist-MusicTransformer-ft-pop67"),
    ("ft_jazz_pop50",       "TheArtist-MusicTransformer-ft-pop50"),
    ("ft_jazz_pop29",       "TheArtist-MusicTransformer-ft-pop29"),
    ("ft_jazz_only",        "TheArtist-MusicTransformer-ft-jazz-only"),
]

# 11 per-genre LoRA adapters on top of the F1 base. Genre tokens that contain
# underscores (rnb_soul, hip_hop) become hyphens in the HF repo name to keep
# the URL slug clean.
LORA_GENRES = [
    "country", "funk", "gospel", "rnb_soul", "hip_hop",
    "electronic", "folk", "classical", "rock", "blues", "bossa",
]
LORAS = [
    (f"ft_f1_lora_{g}", f"TheArtist-MusicTransformer-lora-{g.replace('_', '-')}")
    for g in LORA_GENRES
]


def _upload_one(api: HfApi, folder: str, repo_name: str, commit_message: str) -> None:
    repo_id = f"{ORG}/{repo_name}"
    print(f"Uploading {folder} -> {repo_id} ...")
    create_repo(repo_id, repo_type="model", exist_ok=True, private=False)
    api.upload_folder(
        folder_path=f"model/{folder}",
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
        ignore_patterns=["__pycache__/*", "*.pyc", "*.swp", ".DS_Store"],
    )
    print(f"  done: https://huggingface.co/{repo_id}")


def main() -> None:
    ap = argparse.ArgumentParser()
    scope = ap.add_mutually_exclusive_group()
    scope.add_argument("--only-base", action="store_true", help="skip LoRA adapters")
    scope.add_argument("--only-loras", action="store_true", help="skip F-series base checkpoints")
    scope.add_argument("--repos", nargs="+", metavar="FOLDER",
                       help="upload only these model/ folder names (e.g. ft_jazz_pop80_v2)")
    ap.add_argument("--workers", type=int, default=4, help="parallel upload workers (HF I/O is network-bound)")
    ap.add_argument("--skip-gate", action="store_true",
                    help="bypass the _verify_release.py integrity gate (conscious override only)")
    args = ap.parse_args()

    # Mandatory pre-release gate (2026-06-11): pairwise weight-hash + adapter
    # rank-label checks. Blocks the failure mode that shipped Phase-0 weights
    # as F1. Override requires an explicit --skip-gate.
    if not args.skip_gate:
        import subprocess
        import sys as _sys
        gate = subprocess.run(
            [_sys.executable, str(Path(__file__).parent / "_verify_release.py")],
        )
        if gate.returncode != 0:
            print("\n_upload.py: release gate FAILED — aborting. "
                  "Fix the failure or rerun with --skip-gate to consciously override.")
            raise SystemExit(1)

    api = HfApi()
    jobs: list[tuple[str, str, str]] = []
    if not args.only_loras:
        jobs += [(f, r, "Initial release: TheArtist chord-generation paper companion") for f, r in RUNS]
    if not args.only_base:
        jobs += [(f, r, "Initial release: TheArtist genre LoRA adapter (F1 base)") for f, r in LORAS]
    if args.repos:
        known = {f for f, _, _ in jobs}
        unknown = set(args.repos) - known
        if unknown:
            raise SystemExit(f"_upload.py: unknown --repos folder(s): {sorted(unknown)}")
        jobs = [j for j in jobs if j[0] in args.repos]

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_upload_one, api, *job) for job in jobs]
        for f in as_completed(futures):
            f.result()
    print(f"\nUploaded {len(jobs)} repos under {ORG}/.")


if __name__ == "__main__":
    main()
