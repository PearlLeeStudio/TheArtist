"""The three alternatives of a reharmonization; only server-generated patches can leave.

Every substitution family is searched within the request's bounds, the selected Music
Transformer scores the shortlist against the surrounding bars, and code offers the
best-scoring candidate of each family. `reharmonize` is what POST /api/reharmonize calls.
"""
from __future__ import annotations

from app.models.reharmonize import ReharmCandidate, ReharmRequest, ReharmResponse
from app.services.reharmonize import chord, context_window, search

FAMILIES = ("common_tones", "modal", "dominant")


def reharmonize(request: ReharmRequest) -> ReharmResponse:
    """Check the context, then offer three alternatives scored by the request's checkpoint."""
    from app.services.reharmonize import ContextScorer, model_approximations, validate_context
    validate_context(request)
    return _offer(request, ContextScorer(request), model_approximations(request))


def _offer(req: ReharmRequest, scorer, approximations) -> ReharmResponse:
    """Search every family, score the shortlist, offer the best of each family."""
    pool: dict[str, ReharmCandidate] = {}
    for family in FAMILIES:
        for candidate in search(req, family):
            pool.setdefault(candidate.id, candidate)  # a patch two families both find keeps its first family
    chosen = choose(scorer.score(list(pool.values())))
    if len(chosen) < 3:
        where = "" if req.mode == "auto" else f" in {req.mode}"
        raise ValueError(f"Only {len(chosen)} substitutes fit this passage{where}; allow more changed chords "
                         "or choose another mode.")
    return ReharmResponse(candidates=chosen, modelUsed=scorer.label,
                          contextMeasures=[bar.measure for bar in context_window(req)],
                          modelApproximations=approximations)


def _rank(candidate: ReharmCandidate):
    """Highest learned context score first, then the mathematical score, then a stable id order."""
    score = candidate.modelScore if candidate.modelScore is not None else float("-inf")
    return -score, -candidate.mathScore, candidate.id


def _roots(candidate: ReharmCandidate) -> tuple:
    """Which slots the patch changes and to which roots: two patches alike here differ only in color."""
    return tuple(sorted((c.address.measureIndex, c.address.slotIndex, chord(c.after)[0]) for c in candidate.changes))


def choose(candidates: list[ReharmCandidate], limit: int = 3) -> list[ReharmCandidate]:
    """The best-scoring candidate of each family, best family first, then the best of the rest.

    A patch that changes the same slots to the same roots as one already chosen waits until no other
    candidate is left, so the three contrast where the pool allows. Returned highest score first.
    """
    ranked = sorted(candidates, key=_rank)
    chosen: list[ReharmCandidate] = []
    taken: set[str] = set()
    roots: set[tuple] = set()

    def fits(candidate: ReharmCandidate, new_roots: bool) -> bool:
        return candidate.id not in taken and not (new_roots and _roots(candidate) in roots)

    def take(candidate: ReharmCandidate) -> None:
        chosen.append(candidate)
        taken.add(candidate.id)
        roots.add(_roots(candidate))

    for family in dict.fromkeys(c.family for c in ranked):  # families in the order of their best candidate
        best = next((c for c in ranked if c.family == family and fits(c, True)), None)
        if best is not None and len(chosen) < limit:
            take(best)
    for new_roots in (True, False):
        for candidate in ranked:
            if len(chosen) >= limit:
                break
            if fits(candidate, new_roots):
                take(candidate)
    return sorted(chosen, key=_rank)
