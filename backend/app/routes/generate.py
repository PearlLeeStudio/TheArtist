import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.models.schemas import (
    ContextMeasure,
    ErrorResponse,
    GenerateRequest,
    GenerateResponse,
    SectionGenerateRequest,
    SectionGenerateResponse,
    Suggestion,
    VALID_SECTIONS,
)
from app.models.sections import CHORDS_PER_BAR, MAX_SECTION_BARS, bars_for_chords, section_title
from app.services.chord_generator import generate_suggestions
from app.services.inference import AIResult, generate_ai_result, generate_ai_suggestions, TORCH_AVAILABLE
from app.services.song_form import apply_song_form_to_suggestions

log = logging.getLogger(__name__)

router = APIRouter()

# Progress callback the inference pipeline calls between stages: (stage, payload).
_OnPhase = Callable[[str, dict], None]

VALID_ROOTS = {
    "C", "C#", "Db", "D", "D#", "Eb", "E", "F",
    "F#", "Gb", "G", "G#", "Ab", "A", "A#", "Bb", "B",
}
VALID_MODES = {"major", "minor"}

# Where a section usually sits in a song (0.0 = start, 1.0 = end). The
# rule-based fallback picks its cadence from this position.
SECTION_POSITION = {
    "intro": 0.0,
    "verse": 0.25,
    "pre_chorus": 0.4,
    "chorus": 0.6,
    "post_chorus": 0.65,
    "bridge": 0.7,
    "outro": 1.0,
}
DEFAULT_SECTION_POSITION = 0.5


def parse_key(key: str) -> tuple[str, bool]:
    """Return (key_root, is_minor). Raises HTTPException(400) on a bad key.

    Also the key check of /api/generate/chords, so every route rejects the
    same keys before any model is touched.
    """
    parts = key.strip().split()
    if (
        len(parts) != 2
        or parts[0] not in VALID_ROOTS
        or parts[1].lower() not in VALID_MODES
    ):
        raise HTTPException(
            status_code=400,
            detail="Expected key format: 'C major' or 'A minor'",
        )
    return parts[0], parts[1].lower().startswith("min")


def _validate_request(req: GenerateRequest) -> tuple[str, bool, str | None]:
    """Shared validation between /generate and /generate/stream.

    Returns (key_root, is_minor, section). Raises HTTPException on bad input.
    """
    key_root, is_minor = parse_key(req.key)

    if not req.selectedMeasures:
        raise HTTPException(status_code=400, detail="No measures selected")
    if len(req.selectedMeasures) > MAX_SECTION_BARS:
        raise HTTPException(
            status_code=400,
            detail=f"At most {MAX_SECTION_BARS} measures can be generated at once",
        )

    section = req.sectionType
    if section and section not in VALID_SECTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid sectionType: {section!r}. Must be one of {sorted(VALID_SECTIONS)}",
        )

    return key_root, is_minor, section


def _rule_suggestions(
    *,
    key: str,
    genre: str | None,
    context: list[ContextMeasure],
    selected_measures: list[int],
    time_signature: tuple[int, int],
    position_ratio: float,
    section: str | None,
    key_root: str,
    is_minor: bool,
    cadence: str | None = None,
    seed: int | None = None,
) -> list[Suggestion]:
    """Rule-based fallback, followed by the same song-form rules the AI path applies."""
    suggestions = generate_suggestions(
        key=key,
        genre=genre,
        context=context,
        selected_measures=selected_measures,
        time_signature=time_signature,
        position_ratio=position_ratio,
        seed=seed,
    )
    apply_song_form_to_suggestions(suggestions, section, key_root, is_minor, cadence)
    return suggestions


def _generate_ai(
    req: GenerateRequest, section: str | None, on_phase: _OnPhase | None = None,
) -> list[Suggestion] | None:
    """The AI path for /generate; None tells the caller to fall back to rules."""
    return generate_ai_suggestions(
        key=req.key,
        genre=req.genre,
        context=req.context,
        selected_measures=req.selectedMeasures,
        time_signature=req.timeSignature,
        model_key=req.modelKey,
        bpm=req.bpm,
        on_phase=on_phase,
        section_type=section,
    )


def _generate_response(
    req: GenerateRequest, suggestions: list[Suggestion] | None,
    section: str | None, key_root: str, is_minor: bool,
) -> GenerateResponse:
    """Wrap AI suggestions (song form already applied), or fall back to rules."""
    if suggestions is not None:
        return GenerateResponse(suggestions=suggestions, source="ai")
    rules = _rule_suggestions(
        key=req.key,
        genre=req.genre,
        context=req.context,
        selected_measures=req.selectedMeasures,
        time_signature=req.timeSignature,
        position_ratio=req.positionRatio,
        section=section,
        key_root=key_root,
        is_minor=is_minor,
    )
    return GenerateResponse(suggestions=rules, source="rules")


@router.post(
    "/generate",
    response_model=GenerateResponse,
    responses={400: {"model": ErrorResponse}},
)
def generate(req: GenerateRequest) -> GenerateResponse:
    key_root, is_minor, section = _validate_request(req)
    log.debug("sectionType=%s positionRatio=%s selectedMeasures=%s", section, req.positionRatio, req.selectedMeasures)

    # Try AI inference first, fall back to rule-based
    suggestions = _generate_ai(req, section) if TORCH_AVAILABLE else None
    return _generate_response(req, suggestions, section, key_root, is_minor)


def _sse(event: str, data: dict) -> bytes:
    """Format a Server-Sent Events frame."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode("utf-8")


# Disable proxy buffering so each event hits the client immediately.
_SSE_HEADERS = {"X-Accel-Buffering": "no", "Cache-Control": "no-cache"}
_STREAM_DONE = object()  # queued once the inference future has completed


def _sse_response(frames: AsyncIterator[bytes]) -> StreamingResponse:
    return StreamingResponse(frames, media_type="text/event-stream", headers=_SSE_HEADERS)


def _error_stream(detail: str) -> StreamingResponse:
    """A stream holding a single terminal `error` frame (validation failures)."""
    async def stream():
        yield _sse("error", {"detail": detail})
    return _sse_response(stream())


def _event_stream(
    run_ai: Callable[[_OnPhase], Any],
    finish: Callable[[Any], dict],
) -> StreamingResponse:
    """Stream one generation as SSE: its phase events live, then one terminal frame.

    ``run_ai(on_phase)`` runs on a worker thread (only when torch is
    available; otherwise its output is None). ``finish(output)`` builds the
    ``result`` payload from that output, falling back to rules when it is
    None. An exception in either ends the stream with an ``error`` frame,
    so exactly one of ``result`` / ``error`` closes every stream.
    """
    async def stream():
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        t0 = time.perf_counter()

        def elapsed_ms() -> int:
            return int((time.perf_counter() - t0) * 1000)

        def on_phase(stage: str, payload: dict) -> None:
            """Bridge the worker thread's phase callbacks to the event loop.

            call_soon_threadsafe keeps these puts in FIFO order with the
            executor future's completion, so the done-callback's sentinel
            always lands after the last phase event.
            """
            try:
                loop.call_soon_threadsafe(queue.put_nowait, (stage, payload))
            except RuntimeError:
                # Event loop closed (client disconnected) — drop quietly.
                pass

        if TORCH_AVAILABLE:
            future: asyncio.Future = loop.run_in_executor(None, run_ai, on_phase)
        else:
            future = loop.create_future()
            future.set_result(None)
        future.add_done_callback(lambda _f: queue.put_nowait(_STREAM_DONE))

        while (item := await queue.get()) is not _STREAM_DONE:
            stage, payload = item
            # t_ms = elapsed since request start, for latency probes; the
            # frontend ignores it.
            payload["t_ms"] = elapsed_ms()
            yield _sse(stage, payload)

        try:
            payload = finish(future.result())
        except Exception:
            log.exception("Streaming generation failed")
            yield _sse("error", {"detail": "Chord generation failed"})
            return
        payload["elapsed_ms"] = elapsed_ms()
        yield _sse("result", payload)

    return _sse_response(stream())


@router.post("/generate/stream")
async def generate_stream(req: GenerateRequest):
    """Same contract as /generate but streams progress events via SSE.

    Phases pushed to the client (each as one SSE frame, event name = phase):

        model_load   { label }                 only when checkpoint is cold
        composing    { label, step, total, creativity }   ×3
        ranking      { label }
        voicing      { label }                 unless THEARTIST_DISABLE_VOICE_LEADING
        explaining   { label }                 only when explanations enabled
        complete     { label }                 right before result
        result       { source, suggestions }   final payload, then close
        error        { detail }                terminal, on validation or generation error

    The frontend composes a progress UI from these events; the result
    event carries the same payload shape as /generate's JSON response.
    """
    # Validate up front so we can stream a nice error if needed.
    try:
        key_root, is_minor, section = _validate_request(req)
    except HTTPException as exc:
        return _error_stream(exc.detail)

    return _event_stream(
        lambda on_phase: _generate_ai(req, section, on_phase),
        lambda suggestions: _generate_response(req, suggestions, section, key_root, is_minor).model_dump(),
    )


# --- Song-form sections ---


def _section_measures(req: SectionGenerateRequest) -> tuple[list[ContextMeasure], list[int]]:
    """The request's context plus one empty row per new bar, and the new bars' indices."""
    start = len(req.context)
    selected = list(range(start, start + bars_for_chords(req.chordCount)))
    context = list(req.context) + [
        ContextMeasure(measure=m, chords=[None] * CHORDS_PER_BAR) for m in selected
    ]
    return context, selected


def _section_ai(
    req: SectionGenerateRequest, context: list[ContextMeasure], selected: list[int],
    on_phase: _OnPhase | None = None,
) -> AIResult | None:
    """The AI path for a section; None tells the caller to fall back to rules."""
    return generate_ai_result(
        key=req.key,
        genre=req.genre,
        context=context,
        selected_measures=selected,
        time_signature=req.timeSignature,
        model_key=req.modelKey,
        bpm=req.bpm,
        on_phase=on_phase,
        section_type=req.sectionType,
        cadence=req.cadence,
        seed=req.seed,
    )


def _section_response(
    req: SectionGenerateRequest, context: list[ContextMeasure], selected: list[int],
    ai: AIResult | None, key_root: str, is_minor: bool,
) -> SectionGenerateResponse:
    """Wrap the AI result, or fall back to rules cadenced for the section's
    usual place in a song (SECTION_POSITION)."""
    if ai is not None:
        suggestions, source, model_used = ai.suggestions, "ai", ai.model_key
    else:
        suggestions = _rule_suggestions(
            key=req.key,
            genre=req.genre,
            context=context,
            selected_measures=selected,
            time_signature=req.timeSignature,
            position_ratio=SECTION_POSITION.get(req.sectionType, DEFAULT_SECTION_POSITION),
            section=req.sectionType,
            key_root=key_root,
            is_minor=is_minor,
            cadence=req.cadence,
            seed=req.seed,
        )
        source, model_used = "rules", None
    bars = len(selected)
    return SectionGenerateResponse(
        sectionType=req.sectionType,
        title=section_title(req.sectionType, bars),
        startMeasure=selected[0],
        bars=bars,
        chordCount=bars * CHORDS_PER_BAR,
        suggestions=suggestions,
        source=source,
        modelUsed=model_used,
    )


def compose_section(req: SectionGenerateRequest, on_phase: _OnPhase | None = None) -> SectionGenerateResponse:
    """Compose one song-form section after the given context measures: the
    service behind POST /generate/section, also run per planned section by
    the song planner. Raises HTTPException(400) on a bad key."""
    key_root, is_minor = parse_key(req.key)
    context, selected = _section_measures(req)
    ai = _section_ai(req, context, selected, on_phase) if TORCH_AVAILABLE else None
    return _section_response(req, context, selected, ai, key_root, is_minor)


@router.post(
    "/generate/section",
    response_model=SectionGenerateResponse,
    responses={400: {"model": ErrorResponse}},
)
def generate_section(req: SectionGenerateRequest) -> SectionGenerateResponse:
    """Compose one song-form section after the given context measures."""
    return compose_section(req)


@router.post("/generate/section/stream")
async def generate_section_stream(req: SectionGenerateRequest):
    """/generate/section with the /generate/stream phase events; the
    `result` frame carries the SectionGenerateResponse plus `elapsed_ms`."""
    try:
        key_root, is_minor = parse_key(req.key)
    except HTTPException as exc:
        return _error_stream(exc.detail)
    context, selected = _section_measures(req)
    return _event_stream(
        lambda on_phase: _section_ai(req, context, selected, on_phase),
        lambda ai: _section_response(req, context, selected, ai, key_root, is_minor).model_dump(),
    )
