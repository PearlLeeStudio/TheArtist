"""Score completed chord sequences with the served causal Music Transformer.

This is candidate reranking, not a bidirectionally trained infilling model.
Known bars on both sides are retained in written order. No mask token is invented.
"""
from __future__ import annotations

from collections import defaultdict

CONTEXT_BATCH_SIZE = 3  # Bound CPU activations on the shared two-core server.


def sequence_tokens(tokenizer, header, context, candidate, selected, *, known_only=False):
    """Return tokens and the positions whose conditional likelihood can change.

    The positions are every chord from the passage on; with ``known_only``, only the
    written chords after the passage starts (the chords the passage leads into).
    """
    selected = set(selected)
    first = min(selected)
    known = {bar.measure: bar.chords for bar in context}
    ids, positions = list(header), []
    for index in sorted(set(known) | selected):
        ids.append(tokenizer.bar_id)
        chords = candidate.chords.get(str(index), []) if index in selected else known[index]
        for symbol in chords:
            if symbol is None:
                continue
            token = tokenizer.encode_chord(symbol)
            if token is None:
                raise ValueError("A context chord is outside the Music Transformer vocabulary.")
            if index >= first and not (known_only and index in selected):
                positions.append(len(ids))
            ids.append(token)
    if not positions:
        raise ValueError("No chord tokens can be scored.")
    return ids, positions


def _scoring_model(loaded):
    # PEFT's FEATURE_EXTRACTION wrapper injects Hugging Face arguments such as
    # attention_mask. Our MusicTransformer accepts token tensors directly.
    # get_base_model keeps the installed LoRA layers and saved embedding/output
    # modules; it does not disable, merge or unload the adapter.
    from app.services import inference
    model = loaded.model
    if inference.PEFT_AVAILABLE and isinstance(model, inference.PeftModel):
        model = model.get_base_model()
    return model


def fitting_context(loaded, tokenizer, suggestions, header, context, selected):
    """The known bars the model window can hold with every candidate.

    The whole score when it fits (reach None); otherwise the nearest `reach` bars
    on each side of the passage, the widest that fits. Nothing is clipped
    silently: the caller reports the reach, and a passage whose immediate
    neighbours do not fit raises before any inference.
    """
    limit = getattr(_scoring_model(loaded), "max_seq_len", 512)

    def fits(bars):
        return all(len(sequence_tokens(tokenizer, header, bars, item, selected)[0]) <= limit for item in suggestions)

    if fits(context):
        return context, None
    first, last = min(selected), max(selected)
    known = [bar.measure for bar in context]
    widest = max(first - min(known), max(known) - last)
    for reach in range(widest, 0, -1):
        window = [bar for bar in context if first - reach <= bar.measure <= last + reach]
        if fits(window):
            return window, reach
    raise ValueError(f"Whole-score chord comparison exceeds the {limit}-token model context.")


def context_scores(loaded, tokenizer, suggestions, header, context, selected, *, known_only=False):
    """Mean log likelihood of target chords AND known later chords (only the
    known later chords with ``known_only``).

    Empty non-target bars stay empty; they are not replaced. Long/unsupported
    context raises before inference rather than silently clipping the suffix.
    """
    from app.services import inference

    model = _scoring_model(loaded)
    prepared = [sequence_tokens(tokenizer, header, context, item, selected, known_only=known_only)
                for item in suggestions]
    limit = getattr(model, "max_seq_len", 512)
    if any(len(ids) > limit for ids, _ in prepared):
        raise ValueError(f"Whole-score chord comparison exceeds the {limit}-token model context.")
    # Different exact chord patches can project to the same model vocabulary.
    # Score that token sequence once, keeping every original candidate intact.
    keys = [(tuple(ids), tuple(positions)) for ids, positions in prepared]
    groups = defaultdict(list)
    for item in dict.fromkeys(keys):
        groups[len(item[0])].append(item)
    scores = {}
    with inference._HOSTED_INFERENCE_LOCK, inference.torch.inference_mode():
        for group in groups.values():
            for start in range(0, len(group), CONTEXT_BATCH_SIZE):
                batch = group[start:start + CONTEXT_BATCH_SIZE]
                # Equal-length groups need no padding or changed attention mask.
                inputs = inference.torch.tensor([ids[:-1] for ids, _ in batch], device=loaded.device)
                logits = model(inputs)
                for row, (ids, positions) in enumerate(batch):
                    indices = inference.torch.tensor([p - 1 for p in positions], device=loaded.device)
                    targets = inference.torch.tensor([ids[p] for p in positions], device=loaded.device)
                    values = logits[row, indices].log_softmax(-1).gather(1, targets[:, None])
                    scores[(ids, positions)] = float(values.mean().item())
    return [scores[item] for item in keys]


def continuation_scores(loaded, tokenizer, suggestions, header, context, selected):
    """How well each candidate leads into the written chords after it: the mean log
    likelihood of those known chords alone, given everything before them.

    Generation picks among drafts of one temperature with this, so the written
    future decides which draft is offered while the temperature keeps deciding how
    adventurous the passage itself may be. Raises ValueError like context_scores,
    and when no written chord follows the passage inside the window.
    """
    bars, _ = fitting_context(loaded, tokenizer, suggestions, header, context, selected)
    return context_scores(loaded, tokenizer, suggestions, header, bars, selected, known_only=True)


def compare_with_context(loaded, tokenizer, suggestions, header, context, selected):
    """Candidates best first by likelihood in context, and the reach that was compared
    (None for the whole score)."""
    bars, reach = fitting_context(loaded, tokenizer, suggestions, header, context, selected)
    scores = context_scores(loaded, tokenizer, suggestions, header, bars, selected)
    return [item for _, item in sorted(zip(scores, suggestions), key=lambda pair: -pair[0])], reach


def rank_with_context(loaded, tokenizer, suggestions, header, context, selected):
    return compare_with_context(loaded, tokenizer, suggestions, header, context, selected)[0]
