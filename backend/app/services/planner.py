"""Complete the form: plan the rest of a song form from the user's sketch, then write it.

One run is a deterministic controller around the author's chord model:

    plan_song          the genre-default form for the bars after the locked sketch (rules_plan)
    validate_plan      code checks the plan; its issues are machine-readable
    generate_section   the author's chord model writes every planned section in order, each
                       with the previous ones as context and the planned cadence (compose_section);
                       a same-length chorus can reuse the first one instead (chorus_memory)

The run stops with a typed failure when its time budgets would be passed. The trace
reports everything that happened, failures included; nothing is applied server-side.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from fastapi import HTTPException

from app.models.plan import (
    MAX_PLAN_SECTION_BARS,
    MAX_REASON_CHARS,
    MIN_PLAN_SECTION_BARS,
    PlanError,
    PlanIssue,
    PlanRequest,
    PlanSectionResult,
    PlanTrace,
    PlannedSection,
    SongPlan,
    ValidationAttempt,
)
from app.models.schemas import CADENCE_TYPES, ContextMeasure, SectionGenerateRequest
from app.models.sections import CHORDS_PER_BAR, SECTION_INFO, SECTION_TYPES
from app.routes.generate import compose_section
from app.services.key_anchor import _chord_pcs
from app.services.chorus_memory import reuse_chorus
from app.services.song_form import SECTION_RULES, rule_slots
from app.services.voice_leading import _NOTE_PC

log = logging.getLogger(__name__)

# --- Limits per run ---
RUN_BUDGET_S = 120.0
OWN_MODEL_BUDGET_S = 45.0     # wall clock inside the chord model
OUTRO_BARS = 4                # the rules planner's outro

# Event callback of the streaming route: (event name, payload)
OnEvent = Callable[[str, dict], None]

# Genres whose harmony usually stays in key
DIATONIC_GENRES = ("pop", "country", "folk", "gospel", "classical", "rock", "electronic", "hip_hop")

_TRAILING_ONE = 1  # a leftover of one bar is folded into the section before it


class PlanFailure(Exception):
    """A run stopped: kind and message become the trace's error (see PlanError.kind)."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message


# --- validate_plan ---


def validate_plan(plan: SongPlan, request: PlanRequest) -> list[PlanIssue]:
    """The planner's rules, as machine-readable issues (empty when the plan is valid).

    The plan holds only the sections after the sketch, so the request's existing
    sections cannot be changed by it; the rules below judge the combined form.
    """
    issues: list[PlanIssue] = []

    def issue(code: str, message: str, section: int | None = None) -> None:
        issues.append(PlanIssue(code=code, section=section, message=message))

    sections = plan.sections
    if not sections:
        issue("empty", "the plan has no sections")
        return issues
    sketch_bars = len(request.measures)
    for index, section in enumerate(sections):
        if section.type not in SECTION_INFO:
            issue("section_type", f"unknown section type {section.type!r}; use one of {', '.join(SECTION_TYPES)}", index)
        if not MIN_PLAN_SECTION_BARS <= section.bars <= MAX_PLAN_SECTION_BARS:
            issue("bars", f"{section.bars} bars; a section has {MIN_PLAN_SECTION_BARS} to {MAX_PLAN_SECTION_BARS}", index)
        if section.cadence not in CADENCE_TYPES:
            issue("cadence", f"unknown cadence {section.cadence!r}; use one of {', '.join(CADENCE_TYPES)}", index)
        if len(section.reason) > MAX_REASON_CHARS:
            issue("reason_length", f"reason has {len(section.reason)} characters; at most {MAX_REASON_CHARS}", index)
        if section.type == "intro" and (index > 0 or sketch_bars > 0):
            issue("intro_first", "an intro may only be the first section of an empty song", index)
        if section.type == "outro" and index < len(sections) - 1:
            issue("outro_last", "an outro may only be the last section", index)
        if section.type == "outro" and not request.wantOutro:
            issue("outro_unwanted", "wantOutro is false: no outro", index)
        for cite in section.cites:
            if not 0 <= cite.bar < sketch_bars:
                issue("cite_bar", f"cites bar {cite.bar}, which is not in the {sketch_bars}-bar sketch", index)
            elif cite.roman.strip() not in [roman for roman in request.measures[cite.bar].roman if roman]:
                issue("cite_roman", f"cites {cite.roman!r} at bar {cite.bar}, whose Roman numerals are "
                                    f"{[roman for roman in request.measures[cite.bar].roman]}", index)
    planned = sum(section.bars for section in sections)
    if sketch_bars + planned != request.targetBars:
        issue("total_bars", f"the sketch's {sketch_bars} bars plus the planned {planned} bars make "
                            f"{sketch_bars + planned}, not targetBars {request.targetBars}")
    last = sections[-1]
    if last.cadence in CADENCE_TYPES and last.cadence not in ("authentic", "plagal"):
        issue("final_cadence", f"the last section ends with {last.cadence}; the song must land on "
                               "authentic or plagal", len(sections) - 1)
    if request.wantOutro and last.type != "outro":
        issue("outro_wanted", "wantOutro is true: the last section must be an outro", len(sections) - 1)
    return issues


# --- The rules planner: genre-default forms ---

# (type, bars) in order; trimmed or extended to the bars left after the sketch
_POP_FORM = [("intro", 4), ("verse", 8), ("chorus", 8), ("verse", 8), ("chorus", 8), ("bridge", 8), ("chorus", 8)]
_AABA_FORM = [("a_section", 8), ("a_section", 8), ("b_section", 8), ("a_section", 8)]
RULES_FORMS: dict[str, list[tuple[str, int]]] = {
    "pop": _POP_FORM, "rock": _POP_FORM, "country": _POP_FORM,
    "rnb_soul": [("intro", 4), ("verse", 8), ("pre_chorus", 4), ("chorus", 8), ("verse", 8), ("pre_chorus", 4),
                 ("chorus", 8), ("bridge", 8), ("chorus", 8)],
    "jazz": _AABA_FORM, "bossa": _AABA_FORM,
    "blues": [("intro", 4), ("verse", 12), ("verse", 12), ("verse", 12)],  # 12-bar choruses
    "hip_hop": [("intro", 4), ("verse", 16), ("chorus", 8), ("verse", 16), ("chorus", 8)],
    "electronic": [("intro", 8), ("breakdown", 8), ("drop", 16), ("breakdown", 8), ("drop", 16)],
    "funk": [("intro", 4), ("vamp", 8), ("verse", 8), ("vamp", 8), ("chorus", 8), ("vamp", 8)],
    "gospel": [("intro", 4), ("verse", 8), ("chorus", 8), ("verse", 8), ("chorus", 8), ("vamp", 8), ("tag", 4)],
    "folk": [("intro", 4), ("verse", 8), ("verse", 8), ("verse", 8), ("verse", 8)],
    "classical": _AABA_FORM,  # 8-bar periods
}


def _fit_form(form: list[tuple[str, int]], room: int) -> list[tuple[str, int]]:
    """The form's sections in order until `room` bars are filled: a section that does not
    fit is shortened, and after the form's end its body (no intro) repeats. A leftover of
    one bar goes to the section before it, so every section keeps at least 2 bars."""
    body = [section for section in form if section[0] != "intro"] or form
    out: list[tuple[str, int]] = []
    filled = 0
    position = 0
    while filled < room:
        kind, bars = form[position] if position < len(form) else body[(position - len(form)) % len(body)]
        position += 1
        left = room - filled
        bars = min(bars, left)
        if left - bars == _TRAILING_ONE:
            bars = left
        out.append((kind, bars))
        filled += bars
    return out


def _default_cadence(section: str) -> str:
    return SECTION_RULES.get(section, {}).get("cadence") or "none"


def rules_plan(request: PlanRequest) -> SongPlan:
    """The genre-default form for the bars after the sketch; no model is involved."""
    room = request.targetBars - len(request.measures)
    form = list(RULES_FORMS.get(request.genre, _POP_FORM))
    if request.measures:
        form = [section for section in form if section[0] != "intro"]  # the sketch already opened the song
    outro = min(OUTRO_BARS, room) if request.wantOutro else 0
    if 0 < room - outro < MIN_PLAN_SECTION_BARS:
        outro = room
    sections = _fit_form(form, room - outro)
    if outro:
        sections.append(("outro", outro))
    planned: list[PlannedSection] = []
    for index, (kind, bars) in enumerate(sections):
        last = index == len(sections) - 1
        cadence = ("plagal" if kind == "outro" else "authentic") if last else _default_cadence(kind)
        planned.append(PlannedSection(type=kind, bars=bars, cadence=cadence,
                                      reason=f"{request.genre} default form (rules)", cites=[]))
    return SongPlan(sections=planned)


# --- Analyzer: facts about the model's own chords ---


def _key_collection(key: str) -> frozenset[int]:
    """Pitch classes of the key's collection: the major scale, or natural plus harmonic and
    melodic minor (so a minor key's own V7 and its raised sixth are not 'outside the key')."""
    root, mode = key.split()
    steps = (0, 2, 3, 5, 7, 8, 9, 10, 11) if mode == "minor" else (0, 2, 4, 5, 7, 9, 11)
    tonic = _NOTE_PC[root]
    return frozenset((tonic + step) % 12 for step in steps)


def _symbol(chord: str | None) -> str | None:
    """A chord without its slash bass (voice leading adds those), or None for an empty slot."""
    if not chord:
        return None
    return chord.split("/")[0].strip() or None


def _in_key(symbol: str, collection: frozenset[int]) -> bool:
    """Whether every tone of the chord sits in the key; a symbol the parser cannot read does not."""
    tones = _chord_pcs(symbol)
    return tones is not None and tones <= collection


def _flat_chords(section: PlanSectionResult) -> list[str | None]:
    """The section's applied chords (suggestion A) in slot order."""
    chords = section.suggestions[0].chords if section.suggestions else {}
    return [chord for measure in sorted(chords, key=int) for chord in chords[measure]]


# --- The run ---


@dataclass
class _Run:
    request: PlanRequest
    on_event: OnEvent | None = None
    cancelled: Callable[[], bool] | None = None
    started: float = field(default_factory=time.perf_counter)
    plan: SongPlan | None = None
    validation: list[ValidationAttempt] = field(default_factory=list)
    sections: list[PlanSectionResult] = field(default_factory=list)
    generation_ms: int = 0
    error: PlanError | None = None

    def elapsed_ms(self) -> int:
        return round((time.perf_counter() - self.started) * 1000)

    def emit(self, event: str, payload: dict[str, Any]) -> None:
        if self.on_event is not None:
            self.on_event(event, {**payload, "t_ms": self.elapsed_ms()})

    def check_clock(self) -> None:
        self.check_cancelled()
        if self.elapsed_ms() >= RUN_BUDGET_S * 1000:
            raise PlanFailure("timeout", f"the run passed its {RUN_BUDGET_S:g} s budget")

    def check_cancelled(self) -> None:
        if self.cancelled is not None and self.cancelled():
            raise PlanFailure("generation", "The song-form run was cancelled")

    def trace(self) -> PlanTrace:
        request = self.request
        return PlanTrace(
            status="error" if self.error else "ok", genre=request.genre,
            key=request.key or "", bpm=request.bpm or 0, timeSignature=request.timeSignature or (4, 4),
            plan=self.plan, validation=self.validation, sections=self.sections,
            elapsedMs=self.elapsed_ms(), generationMs=self.generation_ms, error=self.error,
        )


# --- Steps ---


def _plan(run: _Run) -> None:
    """plan_song: the genre-default form, validated."""
    run.check_clock()
    request = run.request
    plan = rules_plan(request)
    issues = validate_plan(plan, request)
    run.plan = plan
    run.validation.append(ValidationAttempt(attempt=1, ok=not issues, errors=issues))
    run.emit("plan", {"attempt": 1, "source": "rules", "plan": plan.model_dump()})
    run.emit("validate", {"attempt": 1, "ok": not issues, "errors": [i.model_dump() for i in issues]})
    if issues:  # the forms are built to pass; a failure here is a bug worth seeing
        raise PlanFailure("invalid", "the genre-default form failed validation: " + "; ".join(i.message for i in issues))


def _context(run: _Run, upto: int) -> list[ContextMeasure]:
    """The sketch plus the applied chords (suggestion A) of the sections before `upto`."""
    context = [ContextMeasure(measure=m.index, chords=list(m.chords)) for m in run.request.measures]
    for section in run.sections[:upto]:
        chords = section.suggestions[0].chords if section.suggestions else {}
        context += [ContextMeasure(measure=int(measure), chords=list(chords[measure]))
                    for measure in sorted(chords, key=int)]
    return context


def _generate_section(run: _Run, index: int) -> PlanSectionResult:
    """generate_section: the planned section after everything before it, with its cadence."""
    assert run.plan is not None
    planned = run.plan.sections[index]
    request = run.request
    run.check_clock()
    reprise = reuse_chorus(request, run.plan, run.sections, index)
    if reprise is not None:
        return reprise
    if run.generation_ms >= OWN_MODEL_BUDGET_S * 1000:
        raise PlanFailure("timeout", f"the run's {OWN_MODEL_BUDGET_S:g} s of chord-model time is used up")
    section_request = SectionGenerateRequest(
        key=request.key or "", bpm=request.bpm or 80, genre=request.genre,
        timeSignature=request.timeSignature or (4, 4), sectionType=planned.type,
        chordCount=planned.bars * CHORDS_PER_BAR, context=_context(run, index), modelKey=request.modelKey,
        cadence=planned.cadence,
        seed=None if request.seed is None else (request.seed + index * 1009) & 0xFFFFFFFF,
    )
    started = time.perf_counter()
    try:
        response = compose_section(section_request)
    except HTTPException as exc:
        raise PlanFailure("generation", str(exc.detail)) from None
    except Exception:
        log.exception("Section generation failed for planned section %d (%s)", index, planned.type)
        raise PlanFailure("generation", "Chord generation failed") from None
    finally:
        run.generation_ms += round((time.perf_counter() - started) * 1000)
    run.check_cancelled()
    return PlanSectionResult(
        **response.model_dump(), cadence=planned.cadence,
        rulesSlots=rule_slots(planned.type, response.chordCount, planned.cadence),
        reason=planned.reason, cites=planned.cites,
    )


def _generate_all(run: _Run) -> None:
    assert run.plan is not None
    for index in range(len(run.plan.sections)):
        run.emit("progress", {"label": f"Composing {run.plan.sections[index].type.replace('_', ' ')} ({index + 1}/{len(run.plan.sections)})…"})
        section = _generate_section(run, index)
        run.sections.append(section)
        run.emit("generate_section", {"index": index, **section.model_dump()})


def _finish(run: _Run, failure: PlanFailure) -> None:
    """Record the failure; the caller (route or tool) decides how to show it."""
    run.error = PlanError(kind=failure.kind, message=failure.message)
    log.warning("plan run failed (%s): %s", failure.kind, failure.message)


# --- Entry points ---


def plan_song(request: PlanRequest, on_event: OnEvent | None = None) -> PlanTrace:
    """Plan and validate only (the MCP plan_song tool): no chord is generated."""
    run = _Run(request, on_event)
    try:
        _plan(run)
    except PlanFailure as failure:
        _finish(run, failure)
    return run.trace()


def complete_form(request: PlanRequest, on_event: OnEvent | None = None,
                  cancelled: Callable[[], bool] | None = None) -> PlanTrace:
    """The whole run behind POST /api/plan/song: plan, then generate every planned section."""
    run = _Run(request, on_event, cancelled)
    try:
        run.check_clock()
        run.emit("progress", {"label": "Planning the song form from the sketch…"})
        _plan(run)
        _generate_all(run)
    except PlanFailure as failure:
        _finish(run, failure)
    return run.trace()
