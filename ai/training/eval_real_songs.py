"""Teacher-forced eval of all 17 released models on the curated 130-song
real-song eval set (`ai/data/eval_real_songs.jsonl`).

The 18 models are:
  6 F-series base checkpoints
    - phase0_pop_baseline      (Phase 0 — pop pretrain)
    - ft_jazz_pop80   (F1, pop-leaning, current backend default)
    - ft_jazz_pop67   (F2)
    - ft_jazz_pop50   (F3, paper sweet-spot)
    - ft_jazz_pop29   (F4, jazz-leaning)
    - ft_jazz_only    (F5, jazz only — catastrophic forgetting endpoint)
  11 LoRA adapters at the best rank picked from
  ai/results/lora_rank_sweep.csv (min val_loss, val_top1 tiebreak)
    - country, funk, gospel, rnb_soul, hip_hop, electronic, folk,
      classical, rock, blues, bossa

For each (model, song) pair the script teacher-forces the song's full
token sequence (BOS + key + time_sig + genre + bars + EOS, truncated to
`max_seq_len=256`) and reports next-token CE loss, top-1, top-5. The
output CSV `ai/results/real_song_eval.csv` has one row per (model, song)
pair so downstream tools can aggregate freely.

The F-series cells use the *old* vocabulary (351 — `[GENRE:X]` tokens only
for jazz/pop/rock/blues/bossa); songs from the 8 genres beyond that
vocabulary get no `[GENRE:X]` token at all (omitted, not `[GENRE:none]`). That is the realistic
"how does F1 base alone handle country?" condition, and Δ vs the
matching LoRA is the headline number for the per-genre paper claim.

Usage:
  cd ai
  uv run python training/eval_real_songs.py
  uv run python training/eval_real_songs.py --device cpu
  uv run python training/eval_real_songs.py --only-models F1,LoRA-country
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel

from lora_train import _resize_embedding_with_init
from model import MusicTransformer
from tokenizer import ChordTokenizer
from _loader_helpers import strip_compile_prefix

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
CKPT_DIR = PROJECT_DIR / "checkpoints"
RESULTS_DIR = PROJECT_DIR / "results"
EVAL_JSONL = DATA_DIR / "eval_real_songs.jsonl"
SWEEP_CSV = RESULTS_DIR / "lora_rank_sweep.csv"
OUT_CSV = RESULTS_DIR / "real_song_eval.csv"
F1_BASE_CKPT = CKPT_DIR / "ft_jazz_pop80" / "best.pt"

F_SERIES_MODELS: list[tuple[str, str, str]] = [
    # (model_id, checkpoint_dir, display_name)
    ("Phase0", "phase0_pop_baseline", "Phase 0 (pop baseline)"),
    ("F1",     "ft_jazz_pop80",       "F1 (pop10K mix)"),
    ("F1v2",   "ft_jazz_pop80_v2",    "F1 v2 (pop10K mix, jazz_val-selected)"),
    ("F2",     "ft_jazz_pop67",       "F2 (pop5K mix)"),
    ("F3",     "ft_jazz_pop50",       "F3 (pop2.5K mix)"),
    ("F4",     "ft_jazz_pop29",       "F4 (pop1K mix, jazz-leaning)"),
    ("F5",     "ft_jazz_only",        "F5 (jazz only)"),
]
LORA_GENRES = [
    "country", "funk", "gospel", "rnb_soul", "hip_hop",
    "electronic", "folk", "classical", "rock", "blues", "bossa",
]
MAX_SEQ_LEN = 256


def load_eval_songs() -> list[dict]:
    if not EVAL_JSONL.exists():
        raise SystemExit(f"[fail] eval set missing at {EVAL_JSONL} — run build_eval_real_songs.py first")
    songs: list[dict] = []
    with open(EVAL_JSONL) as f:
        for line in f:
            import json
            songs.append(json.loads(line))
    return songs


def load_sweep_best_ranks() -> dict[str, int]:
    """Returns {genre: best_rank}. Falls back to rank=8 for genres missing
    from the sweep CSV (rank-sweep eval hasn't run yet)."""
    best: dict[str, int] = {}
    if SWEEP_CSV.exists():
        rows_by_genre: dict[str, list[dict]] = defaultdict(list)
        with open(SWEEP_CSV) as f:
            for r in csv.DictReader(f):
                if r["rank"] == "full_ft":
                    continue
                rows_by_genre[r["genre"]].append({
                    "rank": int(r["rank"]),
                    "val_loss": float(r["val_loss"]),
                    "val_top1": float(r["val_top1"]),
                })
        for g, rows in rows_by_genre.items():
            chosen = min(rows, key=lambda r: (r["val_loss"], -r["val_top1"]))
            best[g] = chosen["rank"]
    for g in LORA_GENRES:
        best.setdefault(g, 8)
    return best


def _ckpt_dir_for_lora(genre: str, rank: int) -> Path:
    suffix = "" if rank == 8 else f"_r{rank}"
    return CKPT_DIR / f"ft_f1_lora_{genre}{suffix}"


def _build_base_model(device: torch.device, vocab_size: int) -> MusicTransformer:
    """Build base model with F1 weights — used as the LoRA host."""
    ckpt = torch.load(F1_BASE_CKPT, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    model = MusicTransformer(
        vocab_size=vocab_size,
        d_model=cfg.get("d_model", 512),
        n_heads=cfg.get("n_heads", 8),
        d_ff=cfg.get("d_ff", 2048),
        n_layers=cfg.get("n_layers", 8),
        max_seq_len=cfg.get("max_seq_len", 256),
        dropout=0.0,
        pad_id=None,
    )
    model.load_state_dict(strip_compile_prefix(ckpt["model_state_dict"]))
    return model


def _load_f_series(ckpt_subdir: str, device: torch.device, vocab_size: int) -> MusicTransformer | None:
    p = CKPT_DIR / ckpt_subdir / "best.pt"
    if not p.exists():
        return None
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    model = MusicTransformer(
        vocab_size=vocab_size,
        d_model=cfg.get("d_model", 512),
        n_heads=cfg.get("n_heads", 8),
        d_ff=cfg.get("d_ff", 2048),
        n_layers=cfg.get("n_layers", 8),
        max_seq_len=cfg.get("max_seq_len", 256),
        dropout=0.0,
        pad_id=None,
    )
    model.load_state_dict(strip_compile_prefix(ckpt["model_state_dict"]))
    return model.to(device).eval()


def _load_lora(genre: str, rank: int, device: torch.device,
                old_vocab: int, new_vocab: int, none_id: int):
    ckpt_dir = _ckpt_dir_for_lora(genre, rank)
    if not (ckpt_dir / "adapter" / "adapter_config.json").exists():
        return None
    if not (ckpt_dir / "embedding_extension.pt").exists():
        return None
    base = _build_base_model(device, vocab_size=old_vocab)
    _resize_embedding_with_init(base, old_vocab, new_vocab, none_id)
    model = PeftModel.from_pretrained(base, str(ckpt_dir / "adapter"))
    ext = torch.load(ckpt_dir / "embedding_extension.pt", map_location="cpu", weights_only=False)
    model.base_model.model.token_emb.load_state_dict(ext["token_emb_state"])
    model.base_model.model.out_proj.load_state_dict(ext["out_proj_state"])
    return model.to(device).eval()


def _eval_song(model, song: dict, tok: ChordTokenizer, device: torch.device) -> dict:
    """Teacher-forced eval on one song. Returns {n_tokens, val_loss, val_top1, val_top5}."""
    ids = tok.encode_sequence(song)
    if len(ids) > MAX_SEQ_LEN:
        ids = ids[:MAX_SEQ_LEN]
    if len(ids) < 4:  # need at least BOS + 2 ctx tokens + a target
        return {"n_tokens": 0, "val_loss": float("nan"),
                "val_top1": float("nan"), "val_top5": float("nan")}
    inp = torch.tensor(ids[:-1], device=device).unsqueeze(0)
    tgt = torch.tensor(ids[1:], device=device).unsqueeze(0)
    # Unwrap PeftModel so we don't pass attention_mask to MusicTransformer.
    inner = getattr(model, "base_model", None)
    inner = getattr(inner, "model", inner) if inner is not None else model
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = inner(inp)
    vocab = logits.shape[-1]
    flat_logits = logits.reshape(-1, vocab)
    flat_tgt = tgt.reshape(-1)
    mask = flat_tgt != tok.pad_id
    n = mask.sum().item()
    if n == 0:
        return {"n_tokens": 0, "val_loss": float("nan"),
                "val_top1": float("nan"), "val_top5": float("nan")}
    loss = F.cross_entropy(flat_logits, flat_tgt, ignore_index=tok.pad_id, reduction="sum").item() / n
    top1 = ((flat_logits.argmax(-1) == flat_tgt) & mask).sum().item() / n * 100
    top5_hits = (flat_logits.topk(5, dim=-1).indices == flat_tgt.unsqueeze(-1)).any(-1)
    top5 = (top5_hits & mask).sum().item() / n * 100
    return {"n_tokens": int(n), "val_loss": round(loss, 4),
            "val_top1": round(top1, 2), "val_top5": round(top5, 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--only-models", type=str, default="",
                    help="comma-separated model IDs (Phase0, F1..F5, LoRA-country, ...)")
    ap.add_argument("--out", type=Path, default=OUT_CSV,
                    help="output CSV path. The file is OVERWRITTEN with only this "
                         "run's rows — subset runs (--only-models) must point at a "
                         "side file, not the canonical real_song_eval.csv")
    args = ap.parse_args()

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))
    print(f"Device: {device}")

    songs = load_eval_songs()
    print(f"Loaded {len(songs)} eval songs from {EVAL_JSONL.relative_to(PROJECT_DIR)}")

    best_ranks = load_sweep_best_ranks()
    print(f"LoRA best ranks: {best_ranks}")

    old_tok = ChordTokenizer(include_extra_genres=False)   # F-series vocab
    new_tok = ChordTokenizer(include_extra_genres=True)    # LoRA vocab
    old_vocab, new_vocab = old_tok.vocab_size, new_tok.vocab_size
    none_id = new_tok.encode_genre("none")
    assert none_id is not None
    print(f"Vocab: {old_vocab} (F-series) / {new_vocab} (+8 extra genres for LoRA)")

    only = {m.strip() for m in args.only_models.split(",") if m.strip()}
    rows: list[dict] = []

    # F-series
    for model_id, ckpt_sub, display in F_SERIES_MODELS:
        if only and model_id not in only:
            continue
        print(f"\n[{model_id}] loading {ckpt_sub} ...", flush=True)
        model = _load_f_series(ckpt_sub, device, old_vocab)
        if model is None:
            print(f"  SKIP — checkpoint missing")
            continue
        for song in songs:
            m = _eval_song(model, song, old_tok, device)
            rows.append({
                "model": model_id, "model_kind": "f_series",
                "model_display": display, "rank": "",
                "song_id": song["id"], "genre": song["genre"],
                "title": song.get("title") or "",
                "n_bars": song["n_bars"],
                "n_tokens": m["n_tokens"], "val_loss": m["val_loss"],
                "val_top1": m["val_top1"], "val_top5": m["val_top5"],
            })
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        # Per-genre quick summary
        by_g: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            if r["model"] == model_id and isinstance(r["val_top1"], (int, float)) and not math.isnan(r["val_top1"]):
                by_g[r["genre"]].append(r["val_top1"])
        summary = " ".join(f"{g}={sum(v)/len(v):.1f}" for g, v in sorted(by_g.items()))
        print(f"  {model_id} done. per-genre top1: {summary}")

    # LoRA — pick best rank per genre, eval on all 130 songs
    for genre in LORA_GENRES:
        model_id = f"LoRA-{genre}"
        if only and model_id not in only:
            continue
        rank = best_ranks[genre]
        print(f"\n[{model_id}] loading r={rank} ...", flush=True)
        model = _load_lora(genre, rank, device, old_vocab, new_vocab, none_id)
        if model is None:
            print(f"  SKIP — adapter missing for genre={genre} rank={rank}")
            continue
        for song in songs:
            m = _eval_song(model, song, new_tok, device)
            rows.append({
                "model": model_id, "model_kind": "lora",
                "model_display": f"LoRA {genre} (r={rank})", "rank": rank,
                "song_id": song["id"], "genre": song["genre"],
                "title": song.get("title") or "",
                "n_bars": song["n_bars"],
                "n_tokens": m["n_tokens"], "val_loss": m["val_loss"],
                "val_top1": m["val_top1"], "val_top5": m["val_top5"],
            })
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        # Quick per-genre top1 for this LoRA — emphasis on its target genre
        target_top1 = [r["val_top1"] for r in rows
                       if r["model"] == model_id and r["genre"] == genre
                       and isinstance(r["val_top1"], (int, float)) and not math.isnan(r["val_top1"])]
        if target_top1:
            print(f"  {model_id} done. on its own genre ({genre}): mean top1 = {sum(target_top1) / len(target_top1):.1f}%")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "model", "model_kind", "model_display", "rank",
            "song_id", "genre", "title", "n_bars",
            "n_tokens", "val_loss", "val_top1", "val_top5",
        ])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} (model × song) rows → {args.out}")


if __name__ == "__main__":
    main()
