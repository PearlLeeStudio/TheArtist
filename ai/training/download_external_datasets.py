"""Download external chord-progression datasets for genre LoRA training.

Targets (license-clean, NC OK per Pearl 2026-05-09 posture):
- Pop909        — pop chord+melody (CC BY-NC-SA)        ~909 songs    ~30MB
- OpenEWLD      — leadsheet, public-domain subset        ~500 songs    ~10MB
- Wikifonia     — leadsheet (via ChoCo, CC BY 4.0)       ~6K songs     ~50MB
- Free MIDI Chords — hip-hop / general (MIT)             varies        ~10MB
- Lakh MIDI Clean — multi-genre MIDI (CC0)              ~17K MIDI     ~3GB ⚠ large

각 dataset은 `data/raw/<name>/` 아래 다운로드. 이미 존재하면 skip.

Usage:
    cd ai
    uv run python training/download_external_datasets.py --datasets pop909 openewld
    uv run python training/download_external_datasets.py --all          # Lakh 포함, 디스크 ~3GB
    uv run python training/download_external_datasets.py --list         # 후보 목록만
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


# ---- Dataset registry ---------------------------------------------------

DATASETS: dict[str, dict] = {
    "pop909": {
        "url": "https://github.com/music-x-lab/POP909-Dataset.git",
        "method": "git_clone",
        "subdir": "pop909",
        "license": "CC BY-NC-SA 4.0 (NC OK per project posture)",
        "size_mb": 30,
        "songs_estimate": 909,
        "note": "Pop chord+melody+accompaniment in MIDI. Bass extraction needed.",
    },
    "openewld": {
        "url": "https://github.com/00sapo/OpenEWLD.git",
        "method": "git_clone",
        "subdir": "openewld",
        "license": "Public Domain (Wikifonia leadsheet PD subset)",
        "size_mb": 10,
        "songs_estimate": 500,
        "note": "Leadsheet MusicXML. chord 추출 필요 (music21 or 직접).",
    },
    "wikifonia": {
        "url": "https://github.com/jonnymakesmusic/wikifonia.git",
        "method": "git_clone",
        "subdir": "wikifonia",
        "license": "CC BY 4.0 (via ChoCo wrapper)",
        "size_mb": 50,
        "songs_estimate": 6000,
        "note": "Leadsheet MusicXML. License는 ChoCo wrapper에서 정확히 재확인 권장.",
    },
    "free_midi_chords": {
        "url": "https://github.com/cifkao/groove2groove.git",  # placeholder — Free MIDI Chords가 어디 있는지 재확인 필요
        "method": "git_clone",
        "subdir": "free_midi_chords",
        "license": "MIT",
        "size_mb": 10,
        "songs_estimate": "TBD",
        "note": "TODO: 정확한 출처 확인. 본 URL은 placeholder.",
    },
    "lakh_clean": {
        "url": "http://hog.ee.columbia.edu/craffel/lmd/lmd_matched.tar.gz",
        "method": "wget_extract",
        "subdir": "lakh_clean",
        "license": "CC0 (Public Domain)",
        "size_mb": 3000,
        "songs_estimate": 17000,
        "note": "⚠ Large (~3GB). chord 추출 + bass/melody 추출 위한 파이프라인 별도 필요.",
    },
}


# ---- Download helpers ---------------------------------------------------

def _has_git() -> bool:
    return shutil.which("git") is not None


def _has_wget() -> bool:
    return shutil.which("wget") is not None or shutil.which("curl") is not None


def download_git_clone(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  [skip] {dest} already exists")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  git clone {url} → {dest}")
    subprocess.check_call(["git", "clone", "--depth", "1", url, str(dest)])


def download_wget_extract(url: str, dest: Path) -> None:
    if dest.exists() and any(dest.iterdir()):
        print(f"  [skip] {dest} already populated")
        return
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / Path(url).name
    print(f"  download {url} → {archive}")
    if shutil.which("wget"):
        subprocess.check_call(["wget", "-c", url, "-O", str(archive)])
    else:
        subprocess.check_call(["curl", "-L", url, "-o", str(archive)])
    print(f"  extract {archive}")
    subprocess.check_call(["tar", "xzf", str(archive), "-C", str(dest)])


def download_dataset(name: str, info: dict) -> None:
    dest = RAW_DIR / info["subdir"]
    print(f"\n=== {name} ({info['license']}, ~{info['size_mb']}MB) ===")
    print(f"  note: {info['note']}")
    if info["method"] == "git_clone":
        if not _has_git():
            print("  [error] git not found")
            return
        download_git_clone(info["url"], dest)
    elif info["method"] == "wget_extract":
        if not _has_wget():
            print("  [error] neither wget nor curl available")
            return
        download_wget_extract(info["url"], dest)
    else:
        print(f"  [error] unknown method: {info['method']}")


# ---- Main ---------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="*", default=None,
                   help=f"Subset to download. Options: {list(DATASETS.keys())}")
    p.add_argument("--all", action="store_true", help="Download all (incl. Lakh ~3GB)")
    p.add_argument("--list", action="store_true", help="List dataset registry only")
    args = p.parse_args()

    if args.list:
        for name, info in DATASETS.items():
            print(f"  {name:20s} {info['license']:50s} ~{info['size_mb']}MB  ~{info['songs_estimate']} songs")
            print(f"  {'':20s} {info['note']}")
        return

    if args.all:
        targets = list(DATASETS.keys())
    elif args.datasets:
        targets = args.datasets
    else:
        # Default: small license-clean datasets, skip Lakh
        targets = ["pop909", "openewld", "wikifonia"]
        print(f"No --datasets specified, defaulting to {targets} (skip Lakh due to size).")

    for name in targets:
        if name not in DATASETS:
            print(f"[skip] unknown dataset: {name}")
            continue
        try:
            download_dataset(name, DATASETS[name])
        except subprocess.CalledProcessError as e:
            print(f"[error] {name} failed: {e}")
        except Exception as e:
            print(f"[error] {name}: {e}")

    print(f"\nDone. Datasets in {RAW_DIR}")


if __name__ == "__main__":
    main()
