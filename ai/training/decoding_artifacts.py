"""Quantify decoding-time artifacts per (genre, selected-rank LoRA).

For each genre and its validation-selected rank, sample N continuations from a
neutral header prompt and count:

  - special_leak_rate     — fraction of samples whose continuation contains
                            any of [PAD]/[BOS]/[KEY:*]/[TIME:*]/[GENRE:*]
                            tokens after the initial header
  - extra_eos_rate        — fraction of samples whose continuation contains
                            an [EOS] token before reaching max_new_tokens
  - repeat_collapse_rate  — fraction of samples whose tail (last 12 chord
                            tokens) is a single repeated chord, indicating a
                            collapsed loop
  - low_diversity_rate    — fraction of samples whose unique-chord count over
                            the continuation is <= 3 (loop-y or stuck output)

The same artifact pass also runs against the F1 base for comparison.

Output:
  ai/results/decoding_artifacts.csv
  ai/results/decoding_artifacts.md
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from collections import Counter
from pathlib import Path

import torch
from peft import PeftModel  # type: ignore[import-untyped]

from lora_train import _resize_embedding_with_init
from model import MusicTransformer
from tokenizer import ChordTokenizer

PROJECT_DIR = Path(__file__).resolve().parent.parent
F1_PATH = PROJECT_DIR / "checkpoints" / "ft_jazz_pop80" / "best.pt"
SELECTED_CSV = PROJECT_DIR / "results" / "lora_selected_test.csv"
RESULTS_DIR = PROJECT_DIR / "results"

GENRES = [
    "blues", "bossa", "classical", "country", "electronic",
    "folk", "funk", "gospel", "hip_hop", "rnb_soul", "rock",
]


def _ckpt_dir_for(genre: str, rank: int) -> Path:
    suffix = "" if rank == 8 else f"_r{rank}"
    return PROJECT_DIR / "checkpoints" / f"ft_f1_lora_{genre}{suffix}"


def _selected_ranks() -> dict[str, int]:
    # THEARTIST_SELECTED_RANKS_CSV overrides the rank table — used on machines
    # that only hold the HF-published adapters (val_loss-selected), whose ranks
    # differ from lora_selected_test.csv (val_top1-selected) for some genres.
    csv_path = Path(os.getenv("THEARTIST_SELECTED_RANKS_CSV", str(SELECTED_CSV)))
    out = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            out[r["genre"]] = int(r["rank"])
    return out


def _load_f1(old_tok: ChordTokenizer, new_tok: ChordTokenizer, device: torch.device) -> MusicTransformer:
    ckpt = torch.load(F1_PATH, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    model = MusicTransformer(
        vocab_size=old_tok.vocab_size,
        d_model=cfg.get("d_model", 512),
        n_heads=cfg.get("n_heads", 8),
        d_ff=cfg.get("d_ff", 2048),
        n_layers=cfg.get("n_layers", 8),
        max_seq_len=cfg.get("max_seq_len", 256),
        dropout=0.0,
        pad_id=None,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    none_id = new_tok.encode_genre("none")
    assert none_id is not None
    _resize_embedding_with_init(model, old_tok.vocab_size, new_tok.vocab_size, none_id)
    return model.to(device).eval()


def _load_lora(
    genre: str,
    rank: int,
    old_tok: ChordTokenizer,
    new_tok: ChordTokenizer,
    device: torch.device,
) -> PeftModel:
    base = _load_f1(old_tok, new_tok, torch.device("cpu"))
    ckpt_dir = _ckpt_dir_for(genre, rank)
    model = PeftModel.from_pretrained(base, str(ckpt_dir / "adapter"))
    emb_ext = torch.load(ckpt_dir / "embedding_extension.pt", map_location="cpu", weights_only=False)
    model.base_model.model.token_emb.load_state_dict(emb_ext["token_emb_state"])
    model.base_model.model.out_proj.load_state_dict(emb_ext["out_proj_state"])
    return model.to(device).eval()


@torch.no_grad()
def _generate_batch(
    model: torch.nn.Module,
    prompt_ids: list[int],
    n_samples: int,
    max_new_tokens: int,
    eos_id: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> list[list[int]]:
    """Sample N continuations from the same prompt. Returns list of generated ids
    *including* the prompt; analysis splits prompt vs continuation downstream."""
    out: list[list[int]] = []
    for _ in range(n_samples):
        ids = torch.tensor([prompt_ids], device=device)
        for _ in range(max_new_tokens):
            if isinstance(model, PeftModel):
                logits = model.base_model.model(ids)
            else:
                logits = model(ids)
            next_logits = logits[:, -1, :] / max(temperature, 1e-3)
            sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            mask = cumulative - sorted_probs >= top_p
            sorted_logits[mask] = -float("inf")
            probs = torch.softmax(sorted_logits, dim=-1)
            sampled = torch.multinomial(probs, 1)
            next_id = sorted_idx.gather(-1, sampled)
            ids = torch.cat([ids, next_id], dim=-1)
            if int(next_id.item()) == eos_id:
                break
        out.append(ids[0].tolist())
    return out


SPECIAL_RE = re.compile(r"^\[(PAD|BOS|KEY|TIME|GENRE):.*\]$|^\[(PAD|BOS|KEY|TIME|GENRE)\]$")
BAR_RE = re.compile(r"^\[BAR\]$")
EOS_RE = re.compile(r"^\[EOS\]$")


def _classify_token(tok: str) -> str:
    if EOS_RE.match(tok):
        return "eos"
    if SPECIAL_RE.match(tok):
        return "special"
    if BAR_RE.match(tok):
        return "bar"
    return "chord"


def _analyze_continuation(tokens: list[str]) -> dict:
    cats = [_classify_token(t) for t in tokens]
    n = len(tokens)
    chords = [t for t, c in zip(tokens, cats) if c == "chord"]
    return {
        "n_tokens": n,
        "n_chords": len(chords),
        "n_unique_chords": len(set(chords)),
        "has_special_leak": any(c == "special" for c in cats),
        "has_eos": any(c == "eos" for c in cats),
        "tail_repeat": (len(chords) >= 12 and len(set(chords[-12:])) == 1),
    }


def _build_prompt(tok: ChordTokenizer, genre: str) -> list[int]:
    """Neutral header prompt: [BOS] [KEY:Cmaj] [TIME:4/4] [GENRE:g] [BAR]."""
    out = [tok.bos_id]
    for special_tok in (
        tok.encode_key("Cmaj"),
        tok.encode_time_sig("4/4"),
        tok.encode_genre(genre),
    ):
        if special_tok is not None:
            out.append(special_tok)
    out.append(tok.bar_id)
    return out


def _summarize(model_label: str, all_records: list[dict]) -> dict:
    n = len(all_records)
    if n == 0:
        return {
            "model": model_label, "n_samples": 0,
            "special_leak_rate": 0.0, "extra_eos_rate": 0.0,
            "repeat_collapse_rate": 0.0, "low_diversity_rate": 0.0,
            "mean_unique_chords": 0.0,
        }
    special = sum(1 for r in all_records if r["has_special_leak"])
    eos = sum(1 for r in all_records if r["has_eos"])
    repeat = sum(1 for r in all_records if r["tail_repeat"])
    low_div = sum(1 for r in all_records if r["n_unique_chords"] <= 3)
    return {
        "model": model_label,
        "n_samples": n,
        "special_leak_rate": round(special / n, 3),
        "extra_eos_rate": round(eos / n, 3),
        "repeat_collapse_rate": round(repeat / n, 3),
        "low_diversity_rate": round(low_div / n, 3),
        "mean_unique_chords": round(sum(r["n_unique_chords"] for r in all_records) / n, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    old_tok = ChordTokenizer(include_extra_genres=False)
    new_tok = ChordTokenizer(include_extra_genres=True)
    selected = _selected_ranks()

    # Reload F1 baseline once.
    print("Loading F1 baseline once")
    f1 = _load_f1(old_tok, new_tok, device)

    records: list[dict] = []
    for i, genre in enumerate(GENRES, start=1):
        rank = selected[genre]
        prompt = _build_prompt(new_tok, genre)
        prompt_len = len(prompt)
        print(f"[{i}/{len(GENRES)}] {genre} r={rank} prompt_len={prompt_len}")

        # F1 baseline for this genre prompt
        f1_seqs = _generate_batch(
            f1, prompt, args.n_samples, args.max_new_tokens,
            new_tok.eos_id, args.temperature, args.top_p, device,
        )
        f1_records = []
        for ids in f1_seqs:
            cont_tokens = new_tok.decode(ids[prompt_len:])
            f1_records.append(_analyze_continuation(cont_tokens))
        records.append({"genre": genre, "rank": rank, **_summarize("F1", f1_records)})

        # LoRA for this genre
        lora = _load_lora(genre, rank, old_tok, new_tok, device)
        lora_seqs = _generate_batch(
            lora, prompt, args.n_samples, args.max_new_tokens,
            new_tok.eos_id, args.temperature, args.top_p, device,
        )
        lora_records = []
        for ids in lora_seqs:
            cont_tokens = new_tok.decode(ids[prompt_len:])
            lora_records.append(_analyze_continuation(cont_tokens))
        records.append({"genre": genre, "rank": rank, **_summarize("LoRA", lora_records)})
        del lora
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Write CSV
    fieldnames = [
        "genre", "rank", "model", "n_samples",
        "special_leak_rate", "extra_eos_rate",
        "repeat_collapse_rate", "low_diversity_rate", "mean_unique_chords",
    ]
    out_csv = RESULTS_DIR / "decoding_artifacts.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(records)
    print(f"Wrote {out_csv}")

    # Markdown summary
    by_g: dict[str, dict[str, dict]] = {}
    for r in records:
        by_g.setdefault(r["genre"], {})[r["model"]] = r

    lines = [
        "# Decoding artifact rates (validation-selected LoRA per genre)",
        "",
        f"Sampled {args.n_samples} continuations per (genre, model) from a",
        f"neutral header prompt (`[BOS] [KEY:Cmaj] [TIME:4/4] [GENRE:g] [BAR]`),",
        f"max_new_tokens={args.max_new_tokens}, temperature={args.temperature},",
        f"top_p={args.top_p}. Tokens after the prompt are scanned for:",
        "",
        "- **special_leak**: any of `[PAD]`, `[BOS]`, `[KEY:*]`, `[TIME:*]`,",
        "  `[GENRE:*]` appearing in the continuation",
        "- **extra_eos**: `[EOS]` reached before `max_new_tokens`",
        "- **repeat_collapse**: tail of 12 chord tokens collapses to a single",
        "  repeated chord",
        "- **low_diversity**: continuation contains ≤3 unique chord tokens",
        "",
        "Lower is better for each rate. `mean_unique_chords` is reported for",
        "context (higher = more variety).",
        "",
        "| Genre | Model | Special leak | Extra EOS | Repeat collapse | Low diversity | Mean unique chords |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for g in GENRES:
        for m in ("F1", "LoRA"):
            r = by_g.get(g, {}).get(m)
            if not r:
                continue
            lines.append(
                f"| {g} | {m} | {r['special_leak_rate']*100:.1f}% | "
                f"{r['extra_eos_rate']*100:.1f}% | {r['repeat_collapse_rate']*100:.1f}% | "
                f"{r['low_diversity_rate']*100:.1f}% | {r['mean_unique_chords']:.1f} |"
            )

    # Aggregate
    f1_rows = [r for r in records if r["model"] == "F1"]
    lora_rows = [r for r in records if r["model"] == "LoRA"]
    if f1_rows and lora_rows:
        def _avg(rows, key): return round(sum(r[key] for r in rows) / len(rows) * 100, 2)
        lines += [
            "",
            "## Aggregate",
            "",
            f"- Mean special-leak rate: F1 {_avg(f1_rows, 'special_leak_rate')}% vs LoRA {_avg(lora_rows, 'special_leak_rate')}%",
            f"- Mean extra-EOS rate: F1 {_avg(f1_rows, 'extra_eos_rate')}% vs LoRA {_avg(lora_rows, 'extra_eos_rate')}%",
            f"- Mean repeat-collapse rate: F1 {_avg(f1_rows, 'repeat_collapse_rate')}% vs LoRA {_avg(lora_rows, 'repeat_collapse_rate')}%",
            f"- Mean low-diversity rate: F1 {_avg(f1_rows, 'low_diversity_rate')}% vs LoRA {_avg(lora_rows, 'low_diversity_rate')}%",
        ]

    out_md = RESULTS_DIR / "decoding_artifacts.md"
    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
