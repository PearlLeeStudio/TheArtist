"""F1 v2 vs Phase 0 — design-intent restoration check (harmonic vocabulary).

The original F1 release was weight-identical to Phase 0 (selection artifact),
so its claimed "richer harmonic vocabulary" was withdrawn. This script asks
whether the selection-corrected retrain (`ft_jazz_pop80_v2`) actually restores
that design intent: under matched prompts and seeds, do its generations carry
more harmonically complex vocabulary than Phase 0's, and does their chord
distribution move toward the held-out jazz corpus?

Prompts: the 10 jazz + 10 pop songs of the 130-song real-song eval set
(`data/eval_real_songs.jsonl`, held out from all training) — first 2 bars as
prompt, 4 sampled continuations each (fixed seeds, temperature 0.8, top-p 0.9).

Metrics per (model × prompt genre), pooled over continuations:
  - mean_quality_complexity  (fraction of chords beyond plain maj/min triads —
                              same definition as humaneval stimulus stats)
  - mean_unique_chord_ratio  (physics.diversity)
  - mean_chord_entropy_bits  (physics.diversity)
  - unique_chord_count       (pooled vocab size of generated chords)
  - chord_vocab_kl_vs_jazz / _vs_pop
                             (KL of generated chord distribution vs the eval
                              set's real jazz / pop chord distributions,
                              Laplace-smoothed over the union support)

Outputs:
    results/f1v2_vocab_comparison_20260612.csv
    results/f1v2_vocab_comparison_20260612.md            (verdict at the top)
    results/f1v2_vocab_comparison_20260612_generations.txt  (qualitative dump)
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import torch

TRAINING_DIR = Path(__file__).resolve().parent
AI_ROOT = TRAINING_DIR.parent
sys.path.insert(0, str(AI_ROOT))
sys.path.insert(0, str(TRAINING_DIR))

from model import MusicTransformer            # noqa: E402
from tokenizer import ChordTokenizer          # noqa: E402
from physics.diversity import chord_entropy, unique_chord_ratio  # noqa: E402
from humaneval._util import split_norm_token  # noqa: E402

EVAL_JSONL = AI_ROOT / "data" / "eval_real_songs.jsonl"
RESULTS_DIR = AI_ROOT / "results"
CKPT_DIR = AI_ROOT / "checkpoints"

MODELS = [
    ("Phase0", "phase0_pop_baseline"),
    ("F1v2", "ft_jazz_pop80_v2"),
    # released F1 omitted on purpose: weight-identical to Phase0 (erratum),
    # so its row would duplicate Phase0's distribution exactly.
]
PROMPT_GENRES = ["jazz", "pop"]
PROMPT_BARS = 2
N_SAMPLES = 4
MAX_NEW_TOKENS = 48          # ≈ 16 bars (BAR + 2 chords per bar)
TEMPERATURE, TOP_P = 0.8, 0.9
SIMPLE_QUALITIES = {"maj", "m"}

STAMP = "20260612"
OUT_CSV = RESULTS_DIR / f"f1v2_vocab_comparison_{STAMP}.csv"
OUT_MD = RESULTS_DIR / f"f1v2_vocab_comparison_{STAMP}.md"
OUT_TXT = RESULTS_DIR / f"f1v2_vocab_comparison_{STAMP}_generations.txt"


def load_model(ckpt_sub: str, tokenizer: ChordTokenizer) -> MusicTransformer:
    ckpt = torch.load(CKPT_DIR / ckpt_sub / "best.pt", map_location="cpu", weights_only=False)
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
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def generate(model, tokenizer, prompt_ids, seed: int) -> list[int]:
    """Nucleus sampling, same scheme as generate_compare.py, seeded."""
    torch.manual_seed(seed)
    ids = torch.tensor([prompt_ids])
    with torch.no_grad():
        for _ in range(MAX_NEW_TOKENS):
            logits = model(ids)
            next_logits = logits[:, -1, :] / TEMPERATURE
            sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
            cumulative = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            mask = cumulative - torch.softmax(sorted_logits, dim=-1) >= TOP_P
            sorted_logits[mask] = -float("inf")
            probs = torch.softmax(sorted_logits, dim=-1)
            next_id = sorted_idx.gather(-1, torch.multinomial(probs, 1))
            ids = torch.cat([ids, next_id], dim=-1)
            if next_id.item() == tokenizer.eos_id:
                break
    return ids[0].tolist()


def chord_tokens(tokenizer: ChordTokenizer, ids: list[int]) -> list[str]:
    """Normalized chord tokens from decoded ids (specials are bracketed)."""
    out = []
    for tok in tokenizer.decode(ids):
        if tok.startswith("["):
            continue
        norm = ChordTokenizer.normalize_chord(tok)
        if norm is not None:
            out.append(norm)
    return out


def quality_complexity(chords: list[str]) -> float | None:
    if not chords:
        return None
    quals = [split_norm_token(c)[1] for c in chords]
    return sum(1 for q in quals if q not in SIMPLE_QUALITIES) / len(quals)


def kl_divergence(gen: Counter, real: Counter) -> float:
    """KL(P_gen || Q_real), Laplace-smoothed over the union support."""
    support = set(gen) | set(real)
    gn, rn = sum(gen.values()), sum(real.values())
    k = len(support)
    kl = 0.0
    for c in support:
        p = (gen[c] + 1) / (gn + k)
        q = (real[c] + 1) / (rn + k)
        kl += p * math.log2(p / q)
    return kl


def main() -> None:
    tokenizer = ChordTokenizer(include_extra_genres=False)

    songs_by_genre: dict[str, list[dict]] = {g: [] for g in PROMPT_GENRES}
    real_chords: dict[str, Counter] = {g: Counter() for g in PROMPT_GENRES}
    with open(EVAL_JSONL) as f:
        for line in f:
            song = json.loads(line)
            g = song["genre"]
            if g in songs_by_genre:
                songs_by_genre[g].append(song)
                for bar in song["bars"]:
                    for ch in bar:
                        norm = ChordTokenizer.normalize_chord(ch)
                        if norm is not None:
                            real_chords[g][norm] += 1
    for g in PROMPT_GENRES:
        print(f"{g}: {len(songs_by_genre[g])} prompt songs, "
              f"{sum(real_chords[g].values())} real chords "
              f"({len(real_chords[g])} types)")

    rows: list[dict] = []
    txt_lines: list[str] = []
    for model_id, ckpt_sub in MODELS:
        print(f"\n[{model_id}] loading {ckpt_sub} ...", flush=True)
        model = load_model(ckpt_sub, tokenizer)
        for genre in PROMPT_GENRES:
            per_gen_qc, per_gen_uniq, per_gen_ent = [], [], []
            pooled: Counter = Counter()
            for s_idx, song in enumerate(songs_by_genre[genre]):
                prompt_song = {
                    "key": song["key"], "time_signature": song["time_signature"],
                    "genre": song["genre"], "bars": song["bars"][:PROMPT_BARS],
                }
                prompt_ids = tokenizer.encode_sequence(prompt_song)[:-1]
                txt_lines.append(f"\n=== {model_id} | {genre} | {song['id']} "
                                 f"(prompt: {sum(song['bars'][:PROMPT_BARS], [])}) ===")
                for sample in range(N_SAMPLES):
                    out_ids = generate(model, tokenizer, prompt_ids,
                                       seed=s_idx * 100 + sample)
                    cont = chord_tokens(tokenizer, out_ids[len(prompt_ids):])
                    txt_lines.append(f"  s{sample}: {' '.join(cont)}")
                    if not cont:
                        continue
                    qc = quality_complexity(cont)
                    if qc is not None:
                        per_gen_qc.append(qc)
                    per_gen_uniq.append(unique_chord_ratio(cont))
                    per_gen_ent.append(chord_entropy(cont))
                    pooled.update(cont)
            rows.append({
                "model": model_id, "prompt_genre": genre,
                "n_generations": len(per_gen_uniq),
                "mean_quality_complexity": round(sum(per_gen_qc) / len(per_gen_qc), 4),
                "mean_unique_chord_ratio": round(sum(per_gen_uniq) / len(per_gen_uniq), 4),
                "mean_chord_entropy_bits": round(sum(per_gen_ent) / len(per_gen_ent), 4),
                "unique_chord_count": len(pooled),
                "chord_vocab_kl_vs_jazz": round(kl_divergence(pooled, real_chords["jazz"]), 4),
                "chord_vocab_kl_vs_pop": round(kl_divergence(pooled, real_chords["pop"]), 4),
            })
            r = rows[-1]
            print(f"  {genre}: qc={r['mean_quality_complexity']} "
                  f"uniq={r['mean_unique_chord_ratio']} ent={r['mean_chord_entropy_bits']} "
                  f"KLjazz={r['chord_vocab_kl_vs_jazz']} KLpop={r['chord_vocab_kl_vs_pop']}")
        del model

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    OUT_TXT.write_text("\n".join(txt_lines) + "\n")

    by = {(r["model"], r["prompt_genre"]): r for r in rows}
    p0j, v2j = by[("Phase0", "jazz")], by[("F1v2", "jazz")]
    p0p, v2p = by[("Phase0", "pop")], by[("F1v2", "pop")]
    restored = (v2j["mean_quality_complexity"] > p0j["mean_quality_complexity"]
                and v2j["chord_vocab_kl_vs_jazz"] < p0j["chord_vocab_kl_vs_jazz"])
    verdict = (
        f"**Design intent restored: {'YES' if restored else 'NO'}.** On jazz prompts, F1 v2's "
        f"quality complexity is {v2j['mean_quality_complexity']:.3f} vs Phase 0's "
        f"{p0j['mean_quality_complexity']:.3f} "
        f"(Δ {v2j['mean_quality_complexity'] - p0j['mean_quality_complexity']:+.3f}), and its chord "
        f"distribution sits {'closer to' if restored else 'no closer to'} the held-out jazz corpus "
        f"(KL {v2j['chord_vocab_kl_vs_jazz']:.3f} vs {p0j['chord_vocab_kl_vs_jazz']:.3f}). "
        f"On pop prompts the drift is Δqc {v2p['mean_quality_complexity'] - p0p['mean_quality_complexity']:+.3f}, "
        f"KL-vs-pop {v2p['chord_vocab_kl_vs_pop']:.3f} vs {p0p['chord_vocab_kl_vs_pop']:.3f}."
    )

    header = "| " + " | ".join(rows[0].keys()) + " |"
    sep = "|" + "---|" * len(rows[0])
    table = "\n".join("| " + " | ".join(str(v) for v in r.values()) + " |" for r in rows)
    OUT_MD.write_text(
        f"# F1 v2 vs Phase 0 — harmonic-vocabulary restoration check ({STAMP})\n\n"
        f"{verdict}\n\n"
        "Matched prompts (first 2 bars of the 10 jazz + 10 pop held-out real songs), "
        f"{N_SAMPLES} seeded continuations each, temperature {TEMPERATURE} / top-p {TOP_P}, "
        f"{MAX_NEW_TOKENS} new tokens. Released F1 omitted: weight-identical to Phase 0.\n\n"
        f"{header}\n{sep}\n{table}\n\n"
        f"Generations: `{OUT_TXT.name}`. Script: `training/f1v2_vocab_comparison.py`. "
        "Companion token-level eval: `real_song_eval_f1v2_20260612.csv`.\n"
    )
    print(f"\n{verdict}")
    print(f"Wrote {OUT_CSV.name}, {OUT_MD.name}, {OUT_TXT.name}")


if __name__ == "__main__":
    main()
