"""Generate chord continuations from each experiment's best checkpoint.

Loads each best.pt, generates from the same set of prompts,
and writes comparison CSV + human-readable text.

Usage:
    cd ai
    uv run python training/generate_compare.py \
        --prompts training/eval_prompts.yaml \
        --output results/generation_comparison.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
from pathlib import Path

import torch
import yaml

from model import MusicTransformer
from tokenizer import ChordTokenizer

PROJECT_DIR = Path(__file__).resolve().parent.parent

CHECKPOINTS = {
    "phase0_pop_baseline": "checkpoints/phase0_pop_baseline/best.pt",
    "ft_jazz_pop80":       "checkpoints/ft_jazz_pop80/best.pt",
    "ft_jazz_pop80_v2":    "checkpoints/ft_jazz_pop80_v2/best.pt",
    "ft_jazz_pop67":       "checkpoints/ft_jazz_pop67/best.pt",
    "ft_jazz_pop50":       "checkpoints/ft_jazz_pop50/best.pt",
    "ft_jazz_pop29":       "checkpoints/ft_jazz_pop29/best.pt",
    "ft_jazz_only":        "checkpoints/ft_jazz_only/best.pt",
    "ft_jazz_pop50_r3":    "checkpoints/ft_jazz_pop50_r3/best.pt",
}


def load_model(ckpt_path, tokenizer, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    model = MusicTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=config.get("d_model", 512),
        n_heads=config.get("n_heads", 8),
        d_ff=config.get("d_ff", 2048),
        n_layers=config.get("n_layers", 8),
        max_seq_len=config.get("max_seq_len", 256),
        dropout=0.0,
        pad_id=tokenizer.pad_id,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt.get("epoch", "?"), ckpt.get("val_loss", "?")


def generate(model, tokenizer, prompt_ids, max_new_tokens=32, temperature=0.8, top_p=0.9):
    device = next(model.parameters()).device
    ids = torch.tensor([prompt_ids], device=device)

    with torch.no_grad():
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=str, required=True)
    parser.add_argument("--output", type=str, default="results/generation_comparison.csv")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=None,
                        help="If set, makes the sampled continuations deterministic "
                             "and reproducible (seeds torch + python RNG once before "
                             "the generation loop). Leave unset for the original "
                             "stochastic behavior.")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = ChordTokenizer()

    with open(args.prompts) as f:
        prompts = yaml.safe_load(f)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for prompt_info in prompts:
        prompt_name = prompt_info["name"]
        prompt_song = prompt_info["song"]
        prompt_ids = tokenizer.encode_sequence(prompt_song)[:-1]  # remove EOS
        prompt_text = tokenizer.decode(prompt_ids)

        for exp_name, ckpt_path in CHECKPOINTS.items():
            full_path = PROJECT_DIR / ckpt_path
            if not full_path.exists():
                print(f"SKIP {exp_name}: {full_path} not found")
                continue

            model, epoch, val_loss = load_model(full_path, tokenizer, device)

            for sample_idx in range(args.num_samples):
                if args.seed is not None:
                    # Per-example deterministic seed: each (prompt, checkpoint,
                    # sample) is reproducible independently of which other
                    # checkpoints are in the run, so quoted examples stay stable
                    # if the CHECKPOINTS set changes.
                    key = f"{prompt_name}|{exp_name}|{sample_idx}|{args.seed}"
                    sub_seed = int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2**31)
                    torch.manual_seed(sub_seed)

                output_ids = generate(
                    model, tokenizer, prompt_ids,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                output_text = tokenizer.decode(output_ids)

                rows.append({
                    "prompt": prompt_name,
                    "experiment": exp_name,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "sample": sample_idx + 1,
                    "prompt_text": prompt_text,
                    "generated_text": output_text,
                })

            del model
            torch.cuda.empty_cache()

    # CSV
    fields = ["prompt", "experiment", "epoch", "val_loss", "sample", "prompt_text", "generated_text"]
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"Saved {len(rows)} generations to {output_path}")

    # Human-readable text
    txt_path = output_path.with_suffix(".txt")
    with open(txt_path, "w") as f:
        for prompt_info in prompts:
            f.write(f"\n{'='*60}\n")
            f.write(f"PROMPT: {prompt_info['name']}\n")
            f.write(f"{'='*60}\n")
            for exp_name in CHECKPOINTS:
                matching = [r for r in rows if r["prompt"] == prompt_info["name"] and r["experiment"] == exp_name]
                if not matching:
                    continue
                f.write(f"\n--- {exp_name} (epoch {matching[0]['epoch']}) ---\n")
                for r in matching:
                    f.write(f"  Sample {r['sample']}: {r['generated_text']}\n")

    print(f"Saved readable comparison to {txt_path}")


if __name__ == "__main__":
    main()
