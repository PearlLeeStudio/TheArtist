#!/usr/bin/env -S uv run --with huggingface_hub python
"""Patch the `## Usage` block in every F-series model card so external HF
users can load the checkpoint end-to-end.

The previous Usage example used `hf_hub_download(filename="best.pt")`
and then `from model import MusicTransformer`, which leaves external
users without `model.py` / `tokenizer.py` on their PYTHONPATH. This
script replaces that block with a `snapshot_download` + `sys.path`
pattern that bundles the project code from the repo itself.

Also refreshes the Citation section to include the upcoming v2 paper
(chord-symbol time-series adaptation) alongside the existing v1.

Idempotent: re-running just overwrites with the same content. Does not
touch any other section.

Usage:
    uv run python model/_patch_usage_blocks.py            # patch all 6
    uv run python model/_patch_usage_blocks.py --dry-run  # show plan
    uv run python model/_patch_usage_blocks.py --only F1
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent

F_SERIES: list[tuple[str, str, str, str]] = [
    # (model_id, folder, hf_repo, sample_genre)
    ("Phase0", "phase0_pop_baseline", "PearlLeeStudio/TheArtist-MusicTransformer-pop-baseline", "pop"),
    ("F1",     "ft_jazz_pop80",       "PearlLeeStudio/TheArtist-MusicTransformer-ft-pop80", "pop"),
    ("F1v2",   "ft_jazz_pop80_v2",    "PearlLeeStudio/TheArtist-MusicTransformer-ft-pop80-v2", "jazz"),
    ("F2",     "ft_jazz_pop67",       "PearlLeeStudio/TheArtist-MusicTransformer-ft-pop67", "pop"),
    ("F3",     "ft_jazz_pop50",       "PearlLeeStudio/TheArtist-MusicTransformer-ft-pop50", "jazz"),
    ("F4",     "ft_jazz_pop29",       "PearlLeeStudio/TheArtist-MusicTransformer-ft-pop29", "jazz"),
    ("F5",     "ft_jazz_only",        "PearlLeeStudio/TheArtist-MusicTransformer-ft-jazz-only", "jazz"),
]


def render_usage(repo_id: str, sample_genre: str) -> str:
    return f"""## Usage

The repo bundles the project's `model.py` and `tokenizer.py` at the repo
root, so external users can load the checkpoint end-to-end without
cloning anything from GitHub. `snapshot_download` materializes the full
repo on disk; `sys.path` makes the bundled `model.py` / `tokenizer.py`
importable.

Required dependencies: `torch`, `huggingface_hub`.

```python
import sys
import torch
from huggingface_hub import snapshot_download

# Download the full repo (model.py, tokenizer.py, best.pt, config.json).
ckpt_dir = snapshot_download(repo_id="{repo_id}")
sys.path.insert(0, ckpt_dir)  # so the next two imports resolve

from model import MusicTransformer
from tokenizer import ChordTokenizer

tokenizer = ChordTokenizer()
ckpt = torch.load(f"{{ckpt_dir}}/best.pt", map_location="cpu", weights_only=False)
model = MusicTransformer(
    vocab_size=tokenizer.vocab_size,
    d_model=512, n_heads=8, d_ff=2048, n_layers=8,
    max_seq_len=256, dropout=0.0, pad_id=tokenizer.pad_id,
)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Prompt = ii-V-I in C major; ask for a {sample_genre}-flavoured continuation.
song = {{
    "key": "Cmaj", "time_signature": "4/4", "genre": "{sample_genre}",
    "bars": [["Dm7", "G7"], ["Cmaj7"]],
}}
prompt_ids = tokenizer.encode_sequence(song)[:-1]
ids = torch.tensor([prompt_ids])
with torch.no_grad():
    for _ in range(32):
        logits = model(ids)
        next_id = torch.multinomial(
            torch.softmax(logits[:, -1, :] / 0.8, dim=-1), 1,
        )
        ids = torch.cat([ids, next_id], dim=-1)
        if next_id.item() == tokenizer.eos_id:
            break
print(tokenizer.decode(ids[0].tolist()))
```

For per-genre adaptation beyond pop and jazz, see the 11 LoRA adapter
repos at [PearlLeeStudio](https://huggingface.co/PearlLeeStudio) — they
chain on top of this base.

"""


CITATION_BLOCK = """## Citation

Cite the original mix-ratio paper. The companion per-genre LoRA paper
(chord-symbol time-series adaptation) is in preparation; its arXiv ID
will be added here once posted.

```bibtex
@misc{lee2026chordmix,
  title         = {Empirical Study of Pop and Jazz Mix Ratios for Genre-Adaptive Chord Generation},
  author        = {Lee, Jinju},
  year          = {2026},
  eprint        = {2605.04998},
  archivePrefix = {arXiv}
}

@misc{lee2026chordtimeseries,
  title         = {How Far Can Chord-Symbol Time-Series Adaptation Carry Genre Identity?},
  author        = {Lee, Jinju},
  year          = {2026},
  note          = {arXiv preprint, ID TBD},
}
```
"""


def patch_usage(text: str, new_usage: str) -> str:
    # Replace everything from `^## Usage` up to (but not including) the next `^## ` heading.
    pat = re.compile(r"^## Usage\n.*?(?=^## )", flags=re.MULTILINE | re.DOTALL)
    if not pat.search(text):
        return text
    return pat.sub(new_usage, text)


def patch_citation(text: str) -> str:
    # Replace from `^## Citation` to end of file (Citation is typically the last section).
    pat = re.compile(r"^## Citation\n.*\Z", flags=re.MULTILINE | re.DOTALL)
    if not pat.search(text):
        # Citation not at end? Try non-greedy up to a sentinel; otherwise append.
        return text.rstrip() + "\n\n" + CITATION_BLOCK
    return pat.sub(CITATION_BLOCK, text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", type=str, default="",
                    help="comma-separated F-series IDs (Phase0,F1,F2,F3,F4,F5)")
    args = ap.parse_args()
    only = {m.strip() for m in args.only.split(",") if m.strip()}

    patched_count = 0
    for model_id, folder, repo_id, sample_genre in F_SERIES:
        if only and model_id not in only:
            continue
        readme_path = MODEL_DIR / folder / "README.md"
        if not readme_path.exists():
            print(f"  {model_id:<8} SKIP — {readme_path} missing")
            continue
        original = readme_path.read_text()
        new_usage = render_usage(repo_id, sample_genre)
        updated = patch_usage(original, new_usage)
        updated = patch_citation(updated)
        if updated == original:
            print(f"  {model_id:<8} no change (already up to date)")
            continue
        delta_chars = len(updated) - len(original)
        print(f"  {model_id:<8} patched  Δ {delta_chars:+,} chars  ({folder})")
        if not args.dry_run:
            readme_path.write_text(updated)
            patched_count += 1

    print(f"\n{patched_count}/{len(F_SERIES)} F-series cards patched"
          + (" (dry-run)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
