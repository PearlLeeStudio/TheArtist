"""Wrong-genre adapter rotation matrix (negative-control sanity check).

For each (adapter_genre, eval_genre) pair where adapter_genre ≠ eval_genre,
apply adapter_genre's LoRA to eval_genre's test split and record top-1.
Builds an 11×11 confusion-style matrix; the diagonal (adapter trained
for its own genre) should dominate the off-diagonals.

Output: ai/results/wrong_genre_rotation.csv

Paper role: §6.2 sanity-check paragraph showing each LoRA learned
genre-specific signal, not generic harmonic improvement.
"""
from __future__ import annotations

import csv
from pathlib import Path

import torch
import torch.nn as nn
from peft import PeftModel
from torch.utils.data import DataLoader

from dataset import ChordSequenceDataset
from lora_train import _resize_embedding_with_init
from model import MusicTransformer
from tokenizer import ChordTokenizer

PROJECT_DIR = Path(__file__).resolve().parent.parent
CKPT_ROOT = PROJECT_DIR / "checkpoints"
SPLITS_DIR = PROJECT_DIR / "data" / "splits"
RESULTS_DIR = PROJECT_DIR / "results"
F1_BASE = CKPT_ROOT / "ft_jazz_pop80" / "best.pt"

GENRES = [
    "blues", "bossa", "classical", "country", "electronic",
    "folk", "funk", "gospel", "hip_hop", "rnb_soul", "rock",
]


def _adapter_dir(genre: str) -> Path:
    """Resolve the F1+LoRA chosen-rank adapter directory for a genre."""
    # Try direct (released-card name) first, then fall back to selected_rank
    direct = CKPT_ROOT / f"ft_f1_lora_{genre}"
    if (direct / "adapter" / "adapter_config.json").exists():
        return direct
    # Fall back: scan ft_f1_lora_<genre>_r* dirs and pick the one matching the
    # selected rank from lora_selected_test.csv
    for sub in CKPT_ROOT.glob(f"ft_f1_lora_{genre}_r*"):
        if (sub / "adapter" / "adapter_config.json").exists():
            return sub
    return direct  # caller checks existence


def _build_test_loader(eval_genre: str, tokenizer: ChordTokenizer):
    ds = ChordSequenceDataset(
        split="test",
        max_seq_len=256,
        augment_keys=False,
        sources=[f"chordonomicon_{eval_genre}"] if eval_genre != "jazz" else None,
        tokenizer_version="v1",
        include_extra_genres=True,
    )
    if len(ds) == 0:
        return None, 0
    return DataLoader(ds, batch_size=32, shuffle=False, num_workers=2), len(ds)


def _eval_combo(adapter_genre: str, eval_genre: str, device, tokenizer):
    adir = _adapter_dir(adapter_genre)
    if not (adir / "adapter" / "adapter_config.json").exists():
        return None
    model = MusicTransformer(
        vocab_size=351, d_model=512, n_heads=8, d_ff=2048, n_layers=8,
        max_seq_len=256, dropout=0.0, pad_id=tokenizer.pad_id,
    )
    ckpt = torch.load(F1_BASE, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in sd.items()}
    model.load_state_dict(sd)
    _resize_embedding_with_init(model, 351, 359, tokenizer.token2id["[GENRE:none]"])
    model = PeftModel.from_pretrained(model, str(adir / "adapter"))
    ext_path = adir / "embedding_extension.pt"
    if ext_path.exists():
        ext = torch.load(ext_path, map_location="cpu", weights_only=False)
        with torch.no_grad():
            model.base_model.model.token_emb.load_state_dict(ext["token_emb_state"])
            model.base_model.model.out_proj.load_state_dict(ext["out_proj_state"])
    model.to(device).eval()

    loader, n = _build_test_loader(eval_genre, tokenizer)
    if loader is None:
        return None
    # Inline eval to unwrap PeftModel (avoid attention_mask passthrough to MT).
    vocab_size = tokenizer.vocab_size
    pad_id = tokenizer.pad_id
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, reduction="sum")
    total_loss = 0.0; total_tokens = 0
    top1_correct = 0; top5_correct = 0
    with torch.no_grad():
        for batch in loader:
            inp = batch["input_ids"].to(device)
            tgt = batch["target_ids"].to(device)
            if isinstance(model, PeftModel):
                logits = model.base_model.model(inp)
            else:
                logits = model(inp)
            fl = logits.reshape(-1, vocab_size); ft = tgt.reshape(-1)
            mask = ft != pad_id
            total_loss += float(criterion(fl, ft).item())
            total_tokens += int(mask.sum().item())
            t1 = fl.argmax(dim=-1)
            top1_correct += int(((t1 == ft) & mask).sum().item())
            t5 = fl.topk(5, dim=-1).indices
            top5_correct += int(((t5 == ft.unsqueeze(-1)).any(dim=-1) & mask).sum().item())
    top1 = top1_correct / max(1, total_tokens) * 100
    top5 = top5_correct / max(1, total_tokens) * 100
    return {"top1": round(top1, 2), "top5": round(top5, 2), "n_sequences": n}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = ChordTokenizer(include_extra_genres=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for adapter_genre in GENRES:
        for eval_genre in GENRES:
            label = f"{adapter_genre} → {eval_genre}"
            print(f"[{label}] evaluating")
            result = _eval_combo(adapter_genre, eval_genre, device, tokenizer)
            if result:
                rows.append({
                    "adapter_genre": adapter_genre,
                    "eval_genre": eval_genre,
                    "diagonal": adapter_genre == eval_genre,
                    **result,
                })

    out = RESULTS_DIR / "wrong_genre_rotation.csv"
    with open(out, "w", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Wrote {out}")

    # Markdown summary: 11x11 matrix of top-1
    out_md = RESULTS_DIR / "wrong_genre_rotation.md"
    with open(out_md, "w") as f:
        f.write("# Wrong-genre adapter rotation matrix (top-1)\n\n")
        f.write("Each row = adapter trained for that genre. Each column = eval test set.\n")
        f.write("Diagonal cells (adapter_genre == eval_genre) should dominate off-diagonals.\n\n")
        f.write("| adapter \\ eval | " + " | ".join(GENRES) + " |\n")
        f.write("|---|" + "---:|" * len(GENRES) + "\n")
        lookup = {(r["adapter_genre"], r["eval_genre"]): r["top1"] for r in rows}
        for a in GENRES:
            cells = [f"**{lookup.get((a, e), '—')}**" if a == e else str(lookup.get((a, e), "—")) for e in GENRES]
            f.write(f"| {a} | " + " | ".join(cells) + " |\n")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
