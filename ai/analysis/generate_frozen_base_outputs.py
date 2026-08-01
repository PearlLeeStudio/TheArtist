"""Generate frozen-base outputs for the §4E supplement (outputs/frozen_base.jsonl).

frozen base = the released F1 base checkpoint alone (no LoRA, no rerank),
generated through the production serving path (backend ModelRegistry +
generate_with_model) over the SAME 306 non-excluded contexts the other derived
outputs use (ai/humaneval/candidates_full_20260611.jsonl), one seeded
continuation per context. Rows follow the shared outputs contract.

Usage (from backend/, its venv has the serving deps):
    uv run python ../ai/analysis/generate_frozen_base_outputs.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

AI_ROOT = Path(__file__).resolve().parent.parent
ROOT = AI_ROOT.parent

os.environ.setdefault("THEARTIST_DISABLE_EXPLANATIONS", "1")
os.environ.setdefault("THEARTIST_DISABLE_RAG", "1")
os.environ.setdefault("THEARTIST_DISABLE_VOICE_LEADING", "1")

sys.path.insert(0, str(ROOT / "backend"))

from app.models.schemas import ContextMeasure  # noqa: E402
from app.services.inference import (  # noqa: E402
    ModelRegistry, build_prompt, generate_with_model,
)

CANDIDATES = AI_ROOT / "humaneval" / "candidates_full_20260611.jsonl"
OUT = AI_ROOT / "results" / "outputs" / "frozen_base.jsonl"
N_BARS = 4
SEED_BASE = 4242


def main() -> None:
    reg = ModelRegistry.get()
    lm = reg.load_model("ft_f1")
    tok = reg.tokenizer
    assert lm is not None, "ft_f1 checkpoint missing"

    n = 0
    with open(CANDIDATES) as f, open(OUT, "w") as out:
        for line in f:
            r = json.loads(line)
            if r.get("excluded"):
                continue
            context = [
                ContextMeasure(measure=i + 1, chords=bar)
                for i, bar in enumerate(r["context_bars"])
            ]
            key = r["key"].replace("maj", " major").replace("min", " minor")
            prompt = build_prompt(
                tok, key, r["genre"],
                tuple(int(x) for x in r["time_signature"].split("/")),
                context,
                selected_measures=[len(context) + 1, len(context) + N_BARS],
            )
            torch.manual_seed(SEED_BASE + n)
            bars = generate_with_model(lm, tok, prompt, n_bars=N_BARS)
            out.write(json.dumps({
                "system": "frozen_base",
                "genre": r["genre"],
                "song_id": r["context_id"],
                "context_bars": r["context_bars"],
                "generated_bars": bars,
                "meta": {
                    "model_key": "ft_f1",
                    "note": "released F1 base (= Phase-0 weights), no LoRA/rerank",
                    "seed": SEED_BASE + n,
                    "key": r["key"], "time_signature": r["time_signature"],
                    "source_contexts": CANDIDATES.name,
                },
            }) + "\n")
            n += 1
            if n % 25 == 0:
                print(f"  {n} contexts done", flush=True)
    print(f"Wrote {n} rows -> {OUT}")


if __name__ == "__main__":
    main()
