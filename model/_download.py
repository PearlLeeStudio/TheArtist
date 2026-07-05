#!/usr/bin/env -S uv run --with huggingface_hub python
"""Download TheArtist Music Transformer checkpoints (F-series) and per-genre
LoRA adapters FROM HuggingFace Hub into ai/checkpoints/. The backend (app/services/inference.py) serves weights from
ai/checkpoints/<run>/, which are gitignored and live only on the Hub.

All PearlLeeStudio/* repos are public (CC BY-NC 4.0) — no token required.

Local layout produced (matches inference.CHECKPOINTS + _load_lora):
  ai/checkpoints/<run>/best.pt                                  (F-series)
  ai/checkpoints/ft_f1_lora_<genre>/adapter/adapter_config.json (LoRA)
  ai/checkpoints/ft_f1_lora_<genre>/adapter/adapter_model.safetensors
  ai/checkpoints/ft_f1_lora_<genre>/embedding_extension.pt

Usage:
    python model/_download.py                # everything (7 F-series + 11 LoRA)
    python model/_download.py --only-base    # F-series only
    python model/_download.py --only-loras   # LoRA adapters only
    python model/_download.py --smoke        # just F1 (ft_jazz_pop80)
"""
import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import hf_hub_download

ORG = "PearlLeeStudio"
REPO_ROOT = Path(__file__).resolve().parent.parent
CKPT = REPO_ROOT / "ai" / "checkpoints"

# (local run dir, hf repo name)
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

# F-series: only best.pt is strictly required (config is embedded in the .pt);
# config.json/eval_results.csv are fetched for a faithful local checkpoint dir.
F_REQUIRED = ["best.pt", "config.json"]
F_OPTIONAL = ["eval_results.csv"]


def _dl(repo: str, filename: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    p = hf_hub_download(repo_id=f"{ORG}/{repo}", filename=filename, local_dir=str(dest_dir))
    return Path(p)


def _clean_cache(run_dir: Path) -> None:
    # hf_hub_download(local_dir=...) drops a .cache/ tracker dir; remove it.
    cache = run_dir / ".cache"
    if cache.exists():
        shutil.rmtree(cache, ignore_errors=True)


def fetch_fseries(run: str, repo: str) -> str:
    dest = CKPT / run
    for fn in F_REQUIRED:
        _dl(repo, fn, dest)
    for fn in F_OPTIONAL:
        try:
            _dl(repo, fn, dest)
        except Exception:
            pass
    _clean_cache(dest)
    size = (dest / "best.pt").stat().st_size / 1e6
    return f"F  {run:<22} best.pt {size:7.1f} MB"


def fetch_lora(genre: str) -> str:
    run = f"ft_f1_lora_{genre}"
    repo = f"TheArtist-MusicTransformer-lora-{genre.replace('_', '-')}"
    dest = CKPT / run
    adapter = dest / "adapter"
    _dl(repo, "adapter_config.json", adapter)
    _dl(repo, "adapter_model.safetensors", adapter)
    _dl(repo, "embedding_extension.pt", dest)
    _clean_cache(adapter)
    _clean_cache(dest)
    size = (adapter / "adapter_model.safetensors").stat().st_size / 1e3
    return f"L  {run:<22} adapter {size:7.1f} KB"


def main() -> None:
    ap = argparse.ArgumentParser()
    scope = ap.add_mutually_exclusive_group()
    scope.add_argument("--only-base", action="store_true", help="F-series only")
    scope.add_argument("--only-loras", action="store_true", help="LoRA adapters only")
    scope.add_argument("--smoke", action="store_true", help="just F1 (ft_jazz_pop80)")
    ap.add_argument("--workers", type=int, default=6, help="parallel download workers")
    args = ap.parse_args()

    jobs = []
    if args.smoke:
        jobs.append(lambda: fetch_fseries("ft_jazz_pop80", "TheArtist-MusicTransformer-ft-pop80"))
    else:
        if not args.only_loras:
            jobs += [(lambda r=r, h=h: fetch_fseries(r, h)) for r, h in RUNS]
        if not args.only_base:
            jobs += [(lambda g=g: fetch_lora(g)) for g in LORA_GENRES]

    print(f"Downloading {len(jobs)} item(s) into {CKPT} ...")
    ok, fail = 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(j) for j in jobs]
        for f in as_completed(futures):
            try:
                print("  " + f.result())
                ok += 1
            except Exception as e:
                print(f"  FAILED: {e}")
                fail += 1
    print(f"\nDone: {ok} ok, {fail} failed.")


if __name__ == "__main__":
    main()
