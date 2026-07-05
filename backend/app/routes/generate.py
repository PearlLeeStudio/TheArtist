import asyncio
import json
import logging
import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.models.schemas import GenerateRequest, GenerateResponse, ErrorResponse, VALID_SECTIONS
from app.services.chord_generator import generate_suggestions
from app.services.inference import generate_ai_suggestions, TORCH_AVAILABLE
from app.services.song_form import apply_song_form

log = logging.getLogger(__name__)

router = APIRouter()

VALID_ROOTS = {
    "C", "C#", "Db", "D", "D#", "Eb", "E", "F",
    "F#", "Gb", "G", "G#", "Ab", "A", "A#", "Bb", "B",
}
VALID_MODES = {"major", "minor"}


def _validate_request(req: GenerateRequest) -> tuple[str, bool, str | None]:
    """Shared validation between /generate and /generate/stream.

    Returns (key_root, is_minor, section). Raises HTTPException on bad input.
    """
    parts = req.key.strip().split()
    if (
        len(parts) != 2
        or parts[0] not in VALID_ROOTS
        or parts[1].lower() not in VALID_MODES
    ):
        raise HTTPException(
            status_code=400,
            detail="Expected key format: 'C major' or 'A minor'",
        )

    if not req.selectedMeasures:
        raise HTTPException(status_code=400, detail="No measures selected")

    section = req.sectionType
    if section and section not in VALID_SECTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid sectionType: {section!r}. Must be one of {sorted(VALID_SECTIONS)}",
        )

    return parts[0], parts[1].lower().startswith("min"), section


def _apply_song_form_inplace(
    suggestions: list, section: str | None, key_root: str, is_minor: bool
) -> None:
    """Run song-form post-processing on each suggestion's chord sequence in-place."""
    if not section:
        return
    for suggestion in suggestions:
        sorted_keys = sorted(suggestion.chords.keys(), key=int)
        flat_chords: list[str] = []
        for k in sorted_keys:
            flat_chords.extend(suggestion.chords[k])
        flat_chords = apply_song_form(
            chords=flat_chords,
            section=section,
            key=key_root,
            is_minor=is_minor,
        )
        idx = 0
        for k in sorted_keys:
            n = len(suggestion.chords[k])
            suggestion.chords[k] = flat_chords[idx : idx + n]
            idx += n


@router.post(
    "/generate",
    response_model=GenerateResponse,
    responses={400: {"model": ErrorResponse}},
)
def generate(req: GenerateRequest) -> GenerateResponse:
    key_root, is_minor, section = _validate_request(req)
    log.debug("sectionType=%s positionRatio=%s selectedMeasures=%s", section, req.positionRatio, req.selectedMeasures)

    # Try AI inference first, fall back to rule-based
    source = "rules"
    suggestions = None
    if TORCH_AVAILABLE:
        suggestions = generate_ai_suggestions(
            key=req.key,
            genre=req.genre,
            context=req.context,
            selected_measures=req.selectedMeasures,
            time_signature=req.timeSignature,
            model_key=req.modelKey,
            bpm=req.bpm,
        )
        if suggestions is not None:
            source = "ai"

    if suggestions is None:
        suggestions = generate_suggestions(
            key=req.key,
            genre=req.genre,
            context=req.context,
            selected_measures=req.selectedMeasures,
            time_signature=req.timeSignature,
            position_ratio=req.positionRatio,
        )

    _apply_song_form_inplace(suggestions, section, key_root, is_minor)
    return GenerateResponse(suggestions=suggestions, source=source)


def _sse(event: str, data: dict) -> bytes:
    """Format a Server-Sent Events frame."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode("utf-8")


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
        error        { detail }                terminal, on validation error

    The frontend composes a progress UI from these events; the result
    event carries the same payload shape as /generate's JSON response.
    """
    # Validate up front so we can stream a nice error if needed.
    try:
        key_root, is_minor, section = _validate_request(req)
    except HTTPException as exc:
        async def err_stream():
            yield _sse("error", {"detail": exc.detail})
        return StreamingResponse(err_stream(), media_type="text/event-stream")

    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def on_phase(stage: str, payload: dict) -> None:
        """Bridge the sync inference's phase callbacks to the async event queue.

        Called from a worker thread (run_in_executor); use thread-safe
        scheduling to push into the queue owned by the event loop.
        """
        try:
            asyncio.run_coroutine_threadsafe(
                queue.put({"event": stage, "data": payload}), loop,
            )
        except RuntimeError:
            # Event loop closed (client disconnected) — drop quietly.
            pass

    async def stream():
        # Kick off inference on a worker thread so we can interleave
        # event delivery with its progress callbacks.
        t0 = time.perf_counter()
        future: asyncio.Future
        if TORCH_AVAILABLE:
            future = loop.run_in_executor(
                None,
                lambda: generate_ai_suggestions(
                    key=req.key,
                    genre=req.genre,
                    context=req.context,
                    selected_measures=req.selectedMeasures,
                    time_signature=req.timeSignature,
                    model_key=req.modelKey,
                    bpm=req.bpm,
                    on_phase=on_phase,
                ),
            )
        else:
            done: asyncio.Future = loop.create_future()
            done.set_result(None)
            future = done

        # Drain phase events as they arrive; exit once inference is done
        # AND the queue has been emptied.
        while True:
            try:
                evt = await asyncio.wait_for(queue.get(), timeout=0.1)
                # t_ms = elapsed since request start; powers the latency
                # probe (ai/analysis/latency_probe.py) and is ignorable
                # by the frontend.
                evt["data"]["t_ms"] = int((time.perf_counter() - t0) * 1000)
                yield _sse(evt["event"], evt["data"])
            except asyncio.TimeoutError:
                if future.done() and queue.empty():
                    break

        suggestions = await future
        source = "ai" if suggestions is not None else "rules"
        if suggestions is None:
            suggestions = generate_suggestions(
                key=req.key,
                genre=req.genre,
                context=req.context,
                selected_measures=req.selectedMeasures,
                time_signature=req.timeSignature,
                position_ratio=req.positionRatio,
            )

        _apply_song_form_inplace(suggestions, section, key_root, is_minor)

        payload = GenerateResponse(suggestions=suggestions, source=source).model_dump()
        payload["elapsed_ms"] = int((time.perf_counter() - t0) * 1000)
        yield _sse("result", payload)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        # Disable proxy buffering so each event hits the client immediately.
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )
