#!/usr/bin/env -S uv run --with huggingface_hub python
"""Apply the F1 weight-identity erratum (2026-06-11) to the local model cards.

Background (verified 2026-06-11 on independent hardware): a full SHA-256
comparison over all 25,841,152 parameters shows the released F1 checkpoint
(`ft_jazz_pop80/best.pt`, Hub: TheArtist-MusicTransformer-ft-pop80) is
weight-identical to the Phase-0 pop-only baseline. The fine-tuning run itself
proceeded normally (training log shows jazz top-1 72.86 -> 81.03), but
best-checkpoint selection minimised validation loss on a pop-dominated
validation mix that rose monotonically during jazz fine-tuning
(0.5578 -> 0.5901), so the pre-fine-tuning initialization was retained as
"best" and the genuinely fine-tuned epoch checkpoints were never released.

Edits (all idempotent, keyed on the literal date marker "2026-06-11"):

  ft_jazz_pop80        full "## Erratum (2026-06-11)" section before ## Demo,
                       corrected bold self-description, qualifying notes on the
                       training-log eval table / generation-character claim /
                       OOD baseline section.
  phase0_pop_baseline  short cross-reference note pointing at the F1 Erratum.
  11 LoRA cards        one "Base checkpoint note (2026-06-11)" paragraph
                       inserted right before ## Adapter summary.

F2-F5 cards mention F1 only as routing pointers ("F1 for pop-leaning") — they
do not describe its training content — so they are deliberately untouched.

No reported number is changed anywhere; the marked real-song-eval blocks
(maintained by _release_f_series_cards.py) are never touched.

Usage:
    uv run python model/_sync_cards.py --pull          # ALWAYS first
    uv run python model/_patch_f1_erratum.py           # apply locally
    uv run python model/_patch_f1_erratum.py --dry-run
    uv run python model/_patch_readmes.py --repos ... --message "..."  # push
"""
from __future__ import annotations

import argparse
from pathlib import Path

from _patch_readmes import LORAS

MODEL_DIR = Path(__file__).resolve().parent
MARK = "2026-06-11"

# ----------------------------------------------------------------- F1 card --

F1_OLD_SUMMARY = (
    "**Jazz-adapted chord model with a 10,000-sequence pop rehearsal buffer. "
    "The pop-leaning endpoint of the mix-ratio sweep. Pop accuracy actually "
    "improves on the pre-fine-tune baseline; jazz reaches +8.17 points.**"
)
F1_NEW_SUMMARY = (
    "**F1 slot of the mix-ratio sweep: jazz fine-tune with a 10,000-sequence "
    "pop rehearsal buffer, the pop-leaning endpoint. In the fine-tuning run's "
    "training log, pop accuracy improves on the pre-fine-tune baseline and "
    "jazz reaches +8.17 points — but the released `best.pt` is "
    "weight-identical to the Phase-0 pop-only baseline. See the Erratum "
    "below.**"
)

F1_ERRATUM = """\
## Erratum (2026-06-11): released weights coincide with the Phase-0 pop baseline

A full SHA-256 comparison over all 25,841,152 parameters (verified 2026-06-11 on independent hardware) shows that the `best.pt` released in this repository is **weight-identical to the Phase-0 pop-only baseline** ([TheArtist-MusicTransformer-pop-baseline](https://huggingface.co/PearlLeeStudio/TheArtist-MusicTransformer-pop-baseline)).

**Mechanism.** The F1 fine-tuning run itself proceeded normally: its training log (`eval_results.csv`) shows held-out jazz top-1 rising 72.86 → 81.03 across epochs 3–8. But best-checkpoint selection used minimum validation loss on a pop-dominated validation mix, and that loss rose monotonically during jazz fine-tuning (0.5578 → 0.5901) — so the pre-fine-tuning initialization, i.e. the Phase-0 weights, was retained as "best". The genuinely fine-tuned epoch checkpoints existed only on the training machine and were not released.

**What this means for the numbers on this card.**

- The §Evaluation table below (pop 84.60 / jazz 81.03) reports the fine-tuning run's training-log metrics. The released `best.pt` does not embody them: identical weights imply identical sampling distributions, so the released checkpoint behaves exactly like the Phase-0 baseline. The per-genre tables on this card that were produced with the released artifact (the out-of-distribution baseline and the real-song eval) remain valid as measured — and accordingly match the pop-baseline card's tables.
- All adaptation gains reported over this frozen base — the 11 `lora-*` adapters and the companion reports — are gains over a **pure-pop harmonic prior**. The substantive findings are unaffected; their interpretation is in fact cleaner.
- The base-checkpoint ablation (Phase-0-base vs F1-base) is retroactively a **same-weights control**: its ±0.3 pp spread is an empirical run-to-run noise estimate, and its "essentially unchanged" result independently corroborates the weight identity.
- The earlier claim that this base "retains a richer harmonic vocabulary" than the pop-only baseline is **withdrawn** — identical weights imply identical sampling distributions.
- F2–F5 are unaffected (all weights distinct, hash-verified). All released evaluation numbers were produced with the served artifacts and reproduce on independent CPU hardware to ≤0.02 pp (model-level).

**Design intent.** When choosing the project's serving base, the author deliberately picked this jazz-adapted rehearsal-mix endpoint over the pop-only Phase-0 baseline, judged on harmonic character: the pop-only base was considered harmonically too plain for the composition tool. That design judgment stands as the intent; the released artifact simply did not embody it, because checkpoint selection silently returned the un-adapted weights. This is itself an instance of the phenomenon the project studies — metric-optimal checkpoint selection on a mismatched validation distribution can silently discard adaptation.

"""

F1_EVAL_HEADING = "## Evaluation (held-out per-genre test sets)\n"
F1_EVAL_NOTE = (
    "\n> **Erratum note (2026-06-11):** the figures in this table are the "
    "fine-tuning run's training-log metrics. The released `best.pt` does not "
    "contain the adapted weights and behaves identically to the Phase-0 "
    "baseline — see the Erratum above.\n"
)

F1_CHARACTER_OLD = (
    "(an occasional secondary dominant or ii-V detour inside an otherwise "
    "diatonic loop)."
)
F1_CHARACTER_NEW = F1_CHARACTER_OLD + (
    " *(2026-06-11: this generation character describes the trained F1 "
    "checkpoint observed during the run; the released `best.pt` samples from "
    "the same distribution as the Phase-0 baseline — see the Erratum above.)*"
)

F1_OOD_OLD = (
    "This is the no-LoRA reference reported alongside every `lora-<genre>` "
    "adapter card."
)
F1_OOD_NEW = F1_OOD_OLD + (
    " Given the Erratum above, these F1-base numbers are measurements of the "
    "released weights — i.e. the pop-only Phase-0 weights — under the "
    "extended-vocab setup; they remain the correct no-LoRA reference, now "
    "cleanly interpretable as a pure-pop prior."
)

# ------------------------------------------------------------ Phase-0 card --

PHASE0_NOTE = """\
> **Note (2026-06-11):** weight-hash verification (full SHA-256 over all 25,841,152 parameters, run on independent hardware) shows the released `ft-pop80` (F1) checkpoint is **weight-identical to this baseline** — best-checkpoint selection in the F1 fine-tuning run silently retained the pre-fine-tuning initialization. See the [Erratum on the ft-pop80 card](https://huggingface.co/PearlLeeStudio/TheArtist-MusicTransformer-ft-pop80) for the mechanism and its consequences. F2–F5 are unaffected (all weights distinct, hash-verified).

"""
PHASE0_ANCHOR = (
    "**Pop-pretrained chord generation model. The starting point that all "
    "five jazz fine-tunes resume from. Released as the no-jazz reference for "
    "measuring catastrophic forgetting in the companion paper.**\n\n"
)

# --------------------------------------------------------------- LoRA cards --

LORA_NOTE = """\
**Base checkpoint note (2026-06-11):** This adapter was trained on the released F1 base checkpoint, which full-SHA-256 weight-hash verification (2026-06-11) shows coincides with the pop-only Phase-0 baseline — best-checkpoint selection in the F1 run silently retained the pre-fine-tuning weights (see the [Erratum on the base card](https://huggingface.co/PearlLeeStudio/TheArtist-MusicTransformer-ft-pop80)). The adapter and all evaluation numbers on this card are self-consistent with that released base: every "F1 base" column was measured against the exact weights this adapter was trained on and is served with. The adaptation gains shown here are therefore gains over a **pure-pop harmonic prior**.

"""
LORA_ANCHOR = "## Adapter summary\n"


def _replace_once(text: str, old: str, new: str, label: str, log: list[str]) -> str:
    n = text.count(old)
    if n != 1:
        log.append(f"    !! {label}: anchor found {n}x (expected 1) — skipped")
        return text
    log.append(f"    + {label}")
    return text.replace(old, new, 1)


def patch_f1(text: str, log: list[str]) -> str:
    text = _replace_once(text, F1_OLD_SUMMARY, F1_NEW_SUMMARY, "bold summary corrected", log)
    text = _replace_once(text, "## Demo\n", F1_ERRATUM + "## Demo\n", "Erratum section", log)
    text = _replace_once(text, F1_EVAL_HEADING, F1_EVAL_HEADING + F1_EVAL_NOTE, "eval-table note", log)
    text = _replace_once(text, F1_CHARACTER_OLD, F1_CHARACTER_NEW, "generation-character qualifier", log)
    text = _replace_once(text, F1_OOD_OLD, F1_OOD_NEW, "OOD-baseline qualifier", log)
    return text


def patch_phase0(text: str, log: list[str]) -> str:
    return _replace_once(text, PHASE0_ANCHOR, PHASE0_ANCHOR + PHASE0_NOTE,
                         "cross-reference note", log)


def patch_lora(text: str, log: list[str]) -> str:
    return _replace_once(text, LORA_ANCHOR, LORA_NOTE + LORA_ANCHOR,
                         "base checkpoint note", log)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="report only, no writes")
    args = ap.parse_args()

    jobs = [("ft_jazz_pop80", patch_f1), ("phase0_pop_baseline", patch_phase0)]
    jobs += [(folder, patch_lora) for folder, _ in LORAS]

    changed = 0
    for folder, fn in jobs:
        p = MODEL_DIR / folder / "README.md"
        if not p.exists():
            print(f"  skip {folder}: no README.md")
            continue
        text = p.read_text()
        if MARK in text:
            print(f"  ok   {folder} (erratum mark already present)")
            continue
        log: list[str] = []
        new = fn(text, log)
        print(f"  edit {folder}")
        for line in log:
            print(line)
        if new != text:
            changed += 1
            if not args.dry_run:
                p.write_text(new)

    print(f"\n{changed} cards patched" + (" (dry-run — nothing written)" if args.dry_run else "")
          + ".\nNext: review diffs, then push with model/_patch_readmes.py.")


if __name__ == "__main__":
    main()
