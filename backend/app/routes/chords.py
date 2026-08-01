"""Chord generation endpoint for programmatic callers (TheVoyager)."""

import logging

from fastapi import APIRouter, HTTPException

from app.models.schemas import ChordGenerateRequest, ContextMeasure, ErrorResponse

log = logging.getLogger(__name__)
from app.services.inference import (
    TORCH_AVAILABLE,
    ModelRegistry,
    build_prompt,
    generate_with_model,
)
from app.services.chord_generator import generate_suggestions

router = APIRouter()


def _spread_temperatures(base: float, n: int) -> list[float]:
    """For n_candidates > 1, spread temperatures around req.temperature for diversity.

    n=2 → [base-0.2, base+0.2], n=3 → [base-0.2, base, base+0.2], n=4/5 → wider spread.
    Clamped to [0.1, 2.0].
    """
    if n == 1:
        return [base]
    if n == 2:
        offsets = [-0.2, 0.2]
    elif n == 3:
        offsets = [-0.2, 0.0, 0.2]
    elif n == 4:
        offsets = [-0.3, -0.1, 0.1, 0.3]
    else:  # n == 5
        offsets = [-0.4, -0.2, 0.0, 0.2, 0.4]
    return [max(0.1, min(2.0, base + o)) for o in offsets]


@router.post(
    "/generate/chords",
    responses={400: {"model": ErrorResponse}},
)
def generate_chords(req: ChordGenerateRequest) -> dict:
    """Generate a chord progression (or N candidates) as flat list(s) of bars.

    Designed for programmatic callers like TheVoyager.
    Falls back to rule-based generation if AI models are unavailable.

    Response branches on `n_candidates`:
      n=1 (default, backward-compat): {"bars": [...], ...}
      n>1: {"candidates": [{"bars": [...], "temperature": ..., "seed": null}], ...}
    """
    parts = req.key.strip().split()
    if len(parts) != 2 or parts[1].lower() not in ("major", "minor"):
        raise HTTPException(status_code=400, detail="Expected key format: 'C major' or 'A minor'")

    # Build context measures from context_bars
    context_measures: list[ContextMeasure] = []
    if req.context_bars:
        for i, bar in enumerate(req.context_bars):
            context_measures.append(ContextMeasure(measure=i, chords=bar))

    selected = list(range(len(context_measures), len(context_measures) + req.n_bars))
    temperatures = _spread_temperatures(req.temperature, req.n_candidates)

    # Try AI inference
    if TORCH_AVAILABLE:
        registry = ModelRegistry.get()
        tokenizer = registry.tokenizer
        prompt_ids = build_prompt(
            tokenizer, req.key, req.genre, req.time_signature,
            context_measures, selected,
            bpm=req.bpm,
        )
        loaded = registry.load_model(req.model)
        if loaded is not None:
            try:
                candidates = []
                for t in temperatures:
                    bars = generate_with_model(
                        loaded, tokenizer, prompt_ids, req.n_bars,
                        temperature=t,
                    )
                    candidates.append({
                        "bars": [{"chords": b} for b in bars],
                        "temperature": t,
                        "seed": None,
                    })
                return _format_response(candidates, req)
            except Exception:
                log.exception("AI inference failed, falling back to rules")

    # Rule-based fallback — generate N candidates by varying chord template selection
    full_context = context_measures + [
        ContextMeasure(measure=m, chords=[None, None]) for m in selected
    ]
    rule_suggestions = generate_suggestions(
        key=req.key, genre=req.genre, context=full_context,
        selected_measures=selected, time_signature=req.time_signature,
    )
    candidates = []
    for i, t in enumerate(temperatures):
        if rule_suggestions and i < len(rule_suggestions):
            sug = rule_suggestions[i]
        elif rule_suggestions:
            sug = rule_suggestions[0]  # repeat first if not enough
        else:
            sug = None
        if sug:
            sorted_keys = sorted(sug.chords.keys(), key=int)
            bars = [sug.chords[k] for k in sorted_keys]
        else:
            bars = [["Cmaj", "Cmaj"]] * req.n_bars
        candidates.append({
            "bars": [{"chords": b} for b in bars],
            "temperature": t,
            "seed": None,
        })
    return _format_response(candidates, req, model_used="rules")


def _format_response(candidates: list[dict], req: ChordGenerateRequest, *, model_used: str | None = None) -> dict:
    """Branch response shape on n_candidates. n=1 keeps legacy 'bars' field; n>1 returns 'candidates' list."""
    base = {
        "key": req.key,
        "genre": req.genre,
        "bpm": req.bpm,
        "model_used": model_used or req.model,
    }
    if req.n_candidates == 1:
        # backward-compat: single 'bars' field
        return {**base, "bars": candidates[0]["bars"]}
    return {**base, "candidates": candidates}
