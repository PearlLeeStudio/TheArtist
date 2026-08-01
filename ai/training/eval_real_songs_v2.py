"""D7: Teacher-forced real-song eval for the v2 chain's 5 methods × 11 genres × 3 seeds.

The existing eval_real_songs.py covers only F-series + LoRA cells. v2
added BitFit, IA3, Full FT, Prefix at multi-seed scale — this script
extends the same eval set (eval_real_songs.jsonl, ~130 songs from
Billboard pop standards + Choco jazz standards + JHT + Chordonomicon
per-genre held-out) to those cells.

Reuses _load_for_method from eval_peft_compare (with a seed suffix
helper) and _eval_song from eval_real_songs (teacher-forced
next-token prediction per song).

Output:
    results/real_song_eval_v2.csv — one row per (method, genre, seed, song)
"""
from __future__ import annotations

import csv
from pathlib import Path

import torch

import eval_peft_compare as epc
import eval_real_songs as ers
from tokenizer import ChordTokenizer

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_CSV = PROJECT_DIR / "results" / "real_song_eval_v2.csv"
SEEDS = [42, 123, 7]
METHODS_V2 = ["bitfit", "ia3", "full", "prefix_tuning", "lora"]


def _ckpt_dir_seed(genre: str, method: str, lora_rank: int | None, seed: int) -> Path:
    base = epc._ckpt_dir(genre, method, lora_rank)
    if seed == 42:
        return base
    return base.parent / f"{base.name}_seed{seed}"


def _cell_artifact_present(ckpt_dir: Path, method: str) -> bool:
    if method in {"full", "bitfit"}:
        return (ckpt_dir / "best.pt").exists()
    if method == "prefix_tuning":
        return (ckpt_dir / "prefix_state.pt").exists() and (ckpt_dir / "embedding_extension.pt").exists()
    # lora, ia3
    return (ckpt_dir / "adapter" / "adapter_config.json").exists()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    songs = ers.load_eval_songs()
    print(f"Eval set: {len(songs)} songs")

    new_tok = ChordTokenizer(include_extra_genres=True)
    old_tok = ChordTokenizer(include_extra_genres=False)

    rows: list[dict] = []
    for genre in epc.GENRES_11:
        try:
            lora_rank = epc._selected_rank(genre)
        except KeyError:
            lora_rank = 8
        for method in METHODS_V2:
            for seed in SEEDS:
                ckpt = _ckpt_dir_seed(genre, method, lora_rank, seed)
                if not _cell_artifact_present(ckpt, method):
                    continue
                try:
                    # eval_peft_compare._load_for_method doesn't know about seed paths,
                    # so we monkey-patch its _ckpt_dir for this single call. Done by
                    # injecting the seed-aware path via a temporary attribute.
                    orig = epc._ckpt_dir
                    epc._ckpt_dir = lambda g, m, r, _ck=ckpt, **kw: _ck  # noqa: E731
                    model = epc._load_for_method(genre, method, lora_rank, new_tok, old_tok, device)
                    epc._ckpt_dir = orig
                except Exception as exc:
                    print(f"  {method}/{genre}/seed{seed}: SKIP load ({exc})")
                    continue

                for song in songs:
                    m = ers._eval_song(model, song, new_tok, device)
                    if m["n_tokens"] == 0:
                        continue
                    rows.append({
                        "method": method, "genre_cell": genre, "seed": seed,
                        "song_id": song.get("song_id", ""),
                        "song_genre": song.get("genre", ""),
                        "title": song.get("title", ""),
                        "n_bars": song.get("n_bars", 0),
                        "n_tokens": m["n_tokens"],
                        "val_loss": round(m["val_loss"], 4),
                        "val_top1": round(m["val_top1"], 2),
                        "val_top5": round(m["val_top5"], 2),
                    })
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                print(f"  {method}/{genre}/seed{seed}: evaluated on {len(songs)} songs")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "method", "genre_cell", "seed",
            "song_id", "song_genre", "title", "n_bars",
            "n_tokens", "val_loss", "val_top1", "val_top5",
        ])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
