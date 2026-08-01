"""Qualitative LoRA continuation: F1 base vs F1+<genre> LoRA for each of 11 genres.

For each prompt in workshop_prompts.yaml, generates N samples from:
  (a) F1 base (no LoRA)
  (b) F1 + that genre's LoRA adapter

Writes:
  results/lora_qualitative.csv   — full rows
  results/lora_qualitative.txt   — human-readable side-by-side

Usage:
    cd ai
    uv run python training/generate_lora_compare.py \
        --prompts training/workshop_prompts.yaml \
        --output results/lora_qualitative.csv \
        --num_samples 2 \
        --max_new_tokens 32
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from peft import PeftModel  # type: ignore[import-untyped]

from lora_train import _resize_embedding_with_init
from model import MusicTransformer
from tokenizer import ChordTokenizer

PROJECT_DIR = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = PROJECT_DIR / "checkpoints"
F1_PATH = CHECKPOINTS_DIR / "ft_jazz_pop80" / "best.pt"

GENRES = [
    "blues", "bossa", "classical", "country", "electronic",
    "folk", "funk", "gospel", "hip_hop", "rnb_soul", "rock",
]


def build_f1_base(device, new_vocab, none_id, old_vocab):
    ckpt = torch.load(F1_PATH, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model = MusicTransformer(
        vocab_size=old_vocab,
        d_model=cfg.get("d_model", 512),
        n_heads=cfg.get("n_heads", 8),
        d_ff=cfg.get("d_ff", 2048),
        n_layers=cfg.get("n_layers", 8),
        max_seq_len=cfg.get("max_seq_len", 256),
        dropout=0.0,
        pad_id=0,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    _resize_embedding_with_init(model, old_vocab, new_vocab, none_id)
    model.to(device).eval()
    return model


def apply_lora(base_model: MusicTransformer, lora_dir: Path, device) -> nn.Module:
    """Overlay embedding_extension (genre rows) + apply PEFT adapter."""
    ext_path = lora_dir / "embedding_extension.pt"
    if ext_path.exists():
        ext = torch.load(ext_path, map_location=device, weights_only=False)
        base_model.token_emb.load_state_dict(ext["token_emb_state"])
        base_model.out_proj.load_state_dict(ext["out_proj_state"])
    adapter_dir = lora_dir / "adapter"
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    peft_model.eval()
    return peft_model


@torch.no_grad()
def generate(model, tokenizer, prompt_ids, max_new_tokens=32, temperature=0.8, top_p=0.9):
    device = next(model.parameters()).device
    ids = torch.tensor([prompt_ids], device=device)
    for _ in range(max_new_tokens):
        logits = model(ids)
        next_logits = logits[:, -1, :] / temperature
        sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
        cumulative = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        mask = cumulative - torch.softmax(sorted_logits, dim=-1) >= top_p
        sorted_logits[mask] = -float("inf")
        probs = torch.softmax(sorted_logits, dim=-1)
        sampled = torch.multinomial(probs, 1)
        next_id = sorted_idx.gather(-1, sampled)
        ids = torch.cat([ids, next_id], dim=-1)
        if next_id.item() == tokenizer.eos_id:
            break
    return ids[0].tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=str, required=True)
    parser.add_argument("--output", type=str, default="results/lora_qualitative.csv")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--num_samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    old_tok = ChordTokenizer(include_extra_genres=False)
    new_tok = ChordTokenizer(include_extra_genres=True)
    old_vocab = old_tok.vocab_size
    new_vocab = new_tok.vocab_size
    none_id = new_tok.encode_genre("none")
    print(f"Vocab: {old_vocab} -> {new_vocab}")

    with open(args.prompts) as f:
        prompts = yaml.safe_load(f)

    # Build F1 base ONCE; reuse for all prompts (no LoRA branch)
    print("Building F1 base...")
    f1_base_model = build_f1_base(device, new_vocab, none_id, old_vocab)

    rows = []
    # Stage 1: F1 base (no LoRA) for every prompt
    print("\nF1 base generations:")
    for p in prompts:
        prompt_ids = new_tok.encode_sequence(p["song"])[:-1]  # drop EOS
        prompt_text = new_tok.decode(prompt_ids)
        for s in range(args.num_samples):
            torch.manual_seed(args.seed + s)
            out_ids = generate(
                f1_base_model, new_tok, prompt_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature, top_p=args.top_p,
            )
            out_text = new_tok.decode(out_ids)
            rows.append({
                "prompt": p["name"], "model": "f1_base",
                "sample": s + 1, "prompt_text": prompt_text,
                "generated_text": out_text,
            })
        print(f"  {p['name']}: done ({args.num_samples} samples)")

    del f1_base_model
    torch.cuda.empty_cache()

    # Stage 2: F1 + LoRA for each genre — rebuild base each time (peft mutates)
    print("\nLoRA generations:")
    for genre in GENRES:
        lora_dir = CHECKPOINTS_DIR / f"ft_f1_lora_{genre}"
        if not lora_dir.exists():
            print(f"  {genre}: SKIP (no checkpoint)")
            continue
        base = build_f1_base(device, new_vocab, none_id, old_vocab)
        peft_model = apply_lora(base, lora_dir, device)

        # Find prompts for this genre
        for p in prompts:
            if p["song"].get("genre") != genre:
                continue
            prompt_ids = new_tok.encode_sequence(p["song"])[:-1]
            prompt_text = new_tok.decode(prompt_ids)
            for s in range(args.num_samples):
                torch.manual_seed(args.seed + s)
                out_ids = generate(
                    peft_model, new_tok, prompt_ids,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature, top_p=args.top_p,
                )
                out_text = new_tok.decode(out_ids)
                rows.append({
                    "prompt": p["name"], "model": f"f1_lora_{genre}",
                    "sample": s + 1, "prompt_text": prompt_text,
                    "generated_text": out_text,
                })
            print(f"  {genre} on {p['name']}: done ({args.num_samples} samples)")

        del peft_model, base
        torch.cuda.empty_cache()

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = PROJECT_DIR / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["prompt", "model", "sample", "prompt_text", "generated_text"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    txt_path = out_path.with_suffix(".txt")
    with open(txt_path, "w") as f:
        for p in prompts:
            f.write(f"\n{'=' * 70}\nPROMPT: {p['name']}\n{'=' * 70}\n")
            matching = [r for r in rows if r["prompt"] == p["name"]]
            if matching:
                f.write(f"Input: {matching[0]['prompt_text']}\n\n")
            for model_name in ["f1_base", f"f1_lora_{p['song'].get('genre')}"]:
                sub = [r for r in matching if r["model"] == model_name]
                if not sub:
                    continue
                f.write(f"--- {model_name} ---\n")
                for r in sub:
                    f.write(f"  Sample {r['sample']}: {r['generated_text']}\n")
                f.write("\n")

    print(f"\nWrote {len(rows)} generations to {out_path}")
    print(f"Readable version: {txt_path}")


if __name__ == "__main__":
    main()
