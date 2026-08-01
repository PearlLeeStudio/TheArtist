"""P16: Chordonomicon provenance note generator.

Records exact source/version/license/checksum for the Chordonomicon CSV
used in this study. Required before arxiv2 release + any model card upload
(Chordonomicon license/version pinning).

Outputs:
    results/dataset_provenance.md
"""
from __future__ import annotations
import hashlib
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
CHORD_CSV = ROOT / "data" / "raw" / "chordonomicon" / "chordonomicon.csv"
OUT = ROOT / "results" / "dataset_provenance.md"


def _file_meta(path: Path):
    if not path.exists():
        return None
    size = path.stat().st_size
    # Stream SHA-256
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return {
        "size_bytes": size,
        "size_mb": round(size / (1024 ** 2), 1),
        "sha256": h.hexdigest(),
        "mtime": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
    }


def main():
    meta = _file_meta(CHORD_CSV)
    lines = [
        "# Dataset provenance — Chordonomicon",
        "",
        "Required before arxiv2 release and model-card upload.",
        "",
        "## File metadata",
        "",
    ]
    if meta is None:
        lines.append(f"**MISSING**: `{CHORD_CSV}` does not exist in this checkout.")
    else:
        lines.append(f"- Path: `ai/data/raw/chordonomicon/chordonomicon.csv`")
        lines.append(f"- Size: {meta['size_mb']} MB ({meta['size_bytes']:,} bytes)")
        lines.append(f"- SHA-256: `{meta['sha256']}`")
        lines.append(f"- Local mtime: {meta['mtime']}")
        lines.append("")
    lines.extend([
        "## Source",
        "",
        "- Origin: Chordonomicon dataset by Kantarelis et al. (TO FILL: exact source URL used to download)",
        "- Version / download date: TO FILL",
        "- License label at download time: TO FILL (public listings have shown CC BY-NC 4.0 and other labels — capture the exact label visible at download)",
        "- Citation: Kantarelis et al., *Chordonomicon: A Dataset of 666,000 Chord Progressions*, NeurIPS 2024 Datasets & Benchmarks track (verify exact venue).",
        "",
        "## Release scope",
        "",
        "Guidance:",
        "",
        "> We use Chordonomicon-derived genre-labeled chord-transcription slices under the license terms of the source artifact available at download time; raw dataset redistribution is not included in this release.",
        "",
        "Released artifacts:",
        "- HuggingFace model weights (F-series checkpoints + per-genre LoRA adapters)",
        "- Paper code/configs at the public open repo",
        "- NOT released: raw Chordonomicon CSV (license redistribution not assumed)",
        "",
        "## Reproducibility",
        "",
        "Any reproduction of these results must:",
        "1. Acquire `chordonomicon.csv` directly from the upstream source.",
        "2. Verify the SHA-256 matches this provenance note.",
        "3. Run `ai/training/extract_genre_subsets.py` to derive per-genre slices.",
        "4. Run `ai/training/dataset.py --create-splits` to materialize seed-fixed splits.",
    ])
    OUT.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
