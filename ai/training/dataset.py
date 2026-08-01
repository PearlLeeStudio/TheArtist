"""PyTorch Dataset for chord sequences with 12-key augmentation.

Supports two modes:
  1. Source filter: sources=["chordonomicon", "billboard"] — flat list
  2. Fine-tune mix: pop_mix_count=5000 — jazz full + pop N songs subsampled

Usage:
    uv run python training/dataset.py --create-splits
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from tokenizer import ChordTokenizer, ROOTS

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_DIR = DATA_DIR / "cache"


def _get_tokenizer(version: str, include_extra_genres: bool = False) -> ChordTokenizer:
    """Return a tokenizer instance matching the requested version."""
    if version == "v2":
        from tokenizer_v2 import ChordTokenizerV2
        return ChordTokenizerV2()
    return ChordTokenizer(include_extra_genres=include_extra_genres)

POP_SOURCES = {"chordonomicon", "billboard"}
JAZZ_SOURCES = {
    "jht", "wjazzd", "jaah", "jazzstandards",
    "choco:real-book", "choco:jazz-corpus", "bunks",
}


def _build_transpose_table(tokenizer: ChordTokenizer) -> np.ndarray:
    """Build a [12, vocab_size] lookup: table[semitones][old_id] = new_id."""
    V = tokenizer.vocab_size
    table = np.zeros((12, V), dtype=np.int16)
    for s in range(12):
        for tid in range(V):
            tok = tokenizer.id2token.get(tid)
            if tok is None:
                table[s, tid] = tid
            elif tok.startswith("[KEY:"):
                new = tokenizer.transpose_key_token(tok, s)
                table[s, tid] = tokenizer.token2id.get(new, tid)
            elif tok.startswith("[") or tid <= tokenizer.BAR:
                table[s, tid] = tid
            else:
                new = tokenizer.transpose_chord_token(tok, s)
                table[s, tid] = tokenizer.token2id[new] if new and new in tokenizer.token2id else tid
    return table


class ChordSequenceDataset(Dataset):
    """In-memory chord sequences with fast 12-key augmentation.

    Args:
        sources: flat list of source names for filtering (e.g. ["chordonomicon", "billboard"])
        pop_mix_count: number of pop songs to mix with ALL jazz songs (for fine-tune experiments).
                       None = don't use mix mode. 0 = jazz only.
        subsample_seed: random seed for reproducible pop subsampling.
    """

    def __init__(
        self,
        split: str = "train",
        max_seq_len: int = 512,
        augment_keys: bool = True,
        data_dir: Path | None = None,
        sources: list[str] | None = None,
        pop_mix_count: int | None = None,
        subsample_seed: int = 42,
        tokenizer_version: str = "v1",
        include_extra_genres: bool = False,
    ) -> None:
        self.split = split
        self.max_seq_len = max_seq_len
        self.seq_len = max_seq_len - 1
        self.augment_keys = augment_keys and (split == "train")
        self.data_dir = data_dir or DATA_DIR
        self.tokenizer_version = tokenizer_version
        self.include_extra_genres = include_extra_genres
        self.tokenizer = _get_tokenizer(tokenizer_version, include_extra_genres)
        self.sources = sources
        self.pop_mix_count = pop_mix_count
        self.subsample_seed = subsample_seed

        self.transpose_table = _build_transpose_table(self.tokenizer)
        self.input_ids, self.target_ids = self._load()
        self.n_base = self.input_ids.shape[0]

    def _cache_key(self, split_file: Path) -> str:
        h = hashlib.md5()
        h.update(str(split_file.stat().st_mtime_ns).encode())
        h.update(str(split_file.stat().st_size).encode())
        h.update(f"tok:{self.tokenizer_version}".encode())
        if self.include_extra_genres:
            h.update(b"tok:extra_genres")
        if self.sources:
            h.update(json.dumps(sorted(self.sources)).encode())
            # If using funksize_subsample sources, include the supplementary
            # file's mtime/size so cache invalidates when sub-sample changes.
            if any(s.startswith("funksize_subsample_") for s in self.sources):
                extra = split_file.parent / f"funksize_subsample.{self.split}.jsonl"
                if extra.exists():
                    h.update(str(extra.stat().st_mtime_ns).encode())
                    h.update(str(extra.stat().st_size).encode())
        if self.pop_mix_count is not None:
            h.update(f"popmix:{self.pop_mix_count}:seed:{self.subsample_seed}".encode())
        return h.hexdigest()[:12]

    def _load(self) -> tuple[np.ndarray, np.ndarray]:
        split_file = self.data_dir / "splits" / f"{self.split}.jsonl"
        if not split_file.exists():
            raise FileNotFoundError(
                f"{split_file} not found.\n"
                "Run:  cd ai && uv run python training/preprocess.py && "
                "uv run python training/dataset.py --create-splits"
            )

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_key = self._cache_key(split_file)

        if self.pop_mix_count is not None:
            src_tag = f"popmix{self.pop_mix_count}_s{self.subsample_seed}"
        elif self.sources:
            src_tag = "_".join(sorted(self.sources))
        else:
            src_tag = "all"

        inp_path = CACHE_DIR / f"{self.split}_{src_tag}_base_{cache_key}_inp.npy"
        tgt_path = CACHE_DIR / f"{self.split}_{src_tag}_base_{cache_key}_tgt.npy"

        if inp_path.exists() and tgt_path.exists():
            t0 = time.time()
            inp = np.load(str(inp_path))
            tgt = np.load(str(tgt_path))
            print(f"  Cache hit: {inp.shape[0]} base sequences ({time.time()-t0:.1f}s)", flush=True)
            return inp, tgt

        t0 = time.time()
        songs = self._load_and_filter_songs(split_file)

        raw_seqs: list[list[int]] = []
        for song in songs:
            ids = self.tokenizer.encode_sequence(song)
            if len(ids) >= 6:
                raw_seqs.append(ids)
        print(f"  Tokenized: {len(raw_seqs)} songs ({time.time()-t0:.1f}s)", flush=True)

        t0 = time.time()
        sl = self.seq_len
        inp_buf = np.zeros((len(raw_seqs), sl), dtype=np.int16)
        tgt_buf = np.zeros((len(raw_seqs), sl), dtype=np.int16)

        for i, ids in enumerate(raw_seqs):
            if len(ids) > self.max_seq_len:
                ids = ids[: self.max_seq_len - 1] + [self.tokenizer.eos_id]
            inp = ids[:-1]
            tgt = ids[1:]
            length = len(inp)
            inp_buf[i, :length] = inp
            tgt_buf[i, :length] = tgt

        np.save(str(inp_path), inp_buf)
        np.save(str(tgt_path), tgt_buf)
        size_mb = (inp_path.stat().st_size + tgt_path.stat().st_size) / 1024 / 1024
        print(f"  Cached: {len(raw_seqs)} sequences, {size_mb:.0f} MB ({time.time()-t0:.1f}s)", flush=True)
        return inp_buf, tgt_buf

    def _load_and_filter_songs(self, split_file: Path) -> list[dict]:
        """Load songs with source filter or pop-mix subsampling."""
        all_songs: list[dict] = []
        with open(split_file) as f:
            for line in f:
                all_songs.append(json.loads(line))

        # Phase 3 F1 Stage A: funksize_subsample_<genre> sources live in
        # an additional split file written by subsample_funk_size.py.
        # Test split always uses the default (untouched) split — F1 cells
        # train on sub-sampled data but eval on full original test set.
        if (self.sources
            and any(s.startswith("funksize_subsample_") for s in self.sources)
            and self.split != "test"):
            extra = split_file.parent / f"funksize_subsample.{self.split}.jsonl"
            if extra.exists():
                with open(extra) as f:
                    for line in f:
                        all_songs.append(json.loads(line))

        # Flat source filter (for Phase 0 or test loaders)
        if self.sources:
            return [s for s in all_songs if s.get("source") in self.sources]

        # Pop-mix mode: all jazz + N pop songs (for fine-tune experiments)
        if self.pop_mix_count is not None:
            jazz_songs = [s for s in all_songs if s.get("source") in JAZZ_SOURCES]
            pop_songs = [s for s in all_songs if s.get("source") in POP_SOURCES]

            if self.pop_mix_count > 0 and pop_songs:
                rng = random.Random(self.subsample_seed)
                n = min(self.pop_mix_count, len(pop_songs))
                pop_sample = rng.sample(pop_songs, n)
            else:
                pop_sample = []

            result = jazz_songs + pop_sample
            rng = random.Random(self.subsample_seed)
            rng.shuffle(result)

            print(f"  Pop-mix: jazz={len(jazz_songs)}, pop={len(pop_sample)}, "
                  f"total={len(result)}", flush=True)
            return result

        # No filter — all songs
        return all_songs

    def __len__(self) -> int:
        return self.n_base * 12 if self.augment_keys else self.n_base

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.augment_keys:
            seq_idx, semitones = divmod(idx, 12)
        else:
            seq_idx, semitones = idx, 0

        inp = self.input_ids[seq_idx]
        tgt = self.target_ids[seq_idx]

        if semitones > 0:
            inp = self.transpose_table[semitones][inp]
            tgt = self.transpose_table[semitones][tgt]

        return {
            "input_ids": torch.from_numpy(inp.astype(np.int64)),
            "target_ids": torch.from_numpy(tgt.astype(np.int64)),
        }


# ------------------------------------------------------------------
# Split creation
# ------------------------------------------------------------------

def create_splits(
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> None:
    """Create train/val/test JSONL splits from processed/songs.jsonl."""
    processed = DATA_DIR / "processed" / "songs.jsonl"
    if not processed.exists():
        raise FileNotFoundError(
            f"{processed} not found. Run:  uv run python training/preprocess.py"
        )

    songs: list[dict] = []
    with open(processed) as f:
        for line in f:
            songs.append(json.loads(line))

    random.seed(seed)
    random.shuffle(songs)

    n = len(songs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": songs[:n_train],
        "val": songs[n_train : n_train + n_val],
        "test": songs[n_train + n_val :],
    }

    splits_dir = DATA_DIR / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    for name, split_songs in splits.items():
        path = splits_dir / f"{name}.jsonl"
        with open(path, "w") as f:
            for song in split_songs:
                f.write(json.dumps(song, ensure_ascii=False) + "\n")

        by_source: dict[str, int] = {}
        for s in split_songs:
            src = s.get("source", "unknown")
            by_source[src] = by_source.get(src, 0) + 1
        breakdown = ", ".join(f"{k}={v}" for k, v in sorted(by_source.items()))
        print(f"  {name}: {len(split_songs)} songs → {path}")
        print(f"    [{breakdown}]")

    n_train_songs = len(splits["train"])
    print(f"\nWith 12-key augmentation:")
    print(f"  train: {n_train_songs} x 12 = {n_train_songs * 12} sequences")
    print(f"  val:   {len(splits['val'])} sequences (no augmentation)")
    print(f"  test:  {len(splits['test'])} sequences (no augmentation)")


if __name__ == "__main__":
    import sys

    if "--create-splits" in sys.argv:
        create_splits()
    else:
        print("Usage: uv run python training/dataset.py --create-splits")
