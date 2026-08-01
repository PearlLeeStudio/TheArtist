"""Reuse a complete chorus without presenting a copy as new model generation."""

from app.models.plan import ChorusReuse, PlanRequest, PlanSectionResult, SongPlan
from app.models.schemas import Suggestion
from app.models.sections import CHORDS_PER_BAR, section_title
from app.services.key_anchor import _chord_pcs
from app.services.song_form import apply_song_form_with_slots


def reuse_chorus(request: PlanRequest, plan: SongPlan, previous: list[PlanSectionResult],
                 index: int) -> PlanSectionResult | None:
    planned = plan.sections[index]
    if request.chorusMode != "reuse" or planned.type != "chorus":
        return None
    # Prefer the musician's locked sketch, then the first generated chorus.
    candidates: list[tuple[list[str | None], dict]] = []
    for section in request.sections:
        if section.type == "chorus" and section.length == planned.bars:
            chords = [chord for measure in request.measures[section.start:section.start + section.length]
                      for chord in measure.chords]
            candidates.append((chords, dict(origin="sketch", startMeasure=section.start, source="sketch")))
    for source_index, section in enumerate(previous[:index]):
        if section.sectionType == "chorus" and section.bars == planned.bars and section.reuse is None:
            chosen = section.suggestions[0].chords if section.suggestions else {}
            chords = [chord for bar in range(section.startMeasure, section.startMeasure + section.bars)
                      for chord in chosen.get(str(bar), [])]
            candidates.append((chords, dict(origin="section", sectionIndex=source_index,
                                           startMeasure=section.startMeasure, source=section.source,
                                           modelUsed=section.modelUsed)))
    count = planned.bars * CHORDS_PER_BAR
    for original, provenance in candidates:
        # Unknown, empty or unsupported harmony must not be multiplied through the song.
        if len(original) != count or any(not symbol or not _chord_pcs(symbol) for symbol in original):
            continue
        symbols = [str(symbol) for symbol in original]
        key, mode = (request.key or "C major").split()
        # A solo has no forced opening/loop: only the explicitly planned cadence changes.
        chords, rules = apply_song_form_with_slots(symbols, "solo", key, mode == "minor", planned.cadence)
        changed = [slot for slot in range(count) if chords[slot] != symbols[slot]]
        start = len(request.measures) + sum(section.bars for section in previous[:index])
        return PlanSectionResult(
            sectionType="chorus", title=section_title("chorus", planned.bars), startMeasure=start,
            bars=planned.bars, chordCount=count, source="reuse", modelUsed=None,
            suggestions=[Suggestion(label="Chorus reprise", chords={
                str(start + bar): chords[bar * CHORDS_PER_BAR:(bar + 1) * CHORDS_PER_BAR]
                for bar in range(planned.bars)
            })], cadence=planned.cadence, rulesSlots=rules, reason=planned.reason, cites=planned.cites,
            reuse=ChorusReuse(**provenance, retainedSlots=[slot for slot in range(count) if slot not in changed],
                              changedSlots=changed),
        )
    return None
