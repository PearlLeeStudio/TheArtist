import asyncio
import json
import logging
from collections.abc import Callable

from fastapi import APIRouter, HTTPException, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import StreamingResponse
from fastapi.routing import APIRoute
from pydantic import ValidationError

from app.models.score_import import ScoreImportRequest, ScoreImportResponse
from app.services.score_import import ScoreInputError, import_score


class _NoStoreScoreRoute(APIRoute):
    """Keep transient uploaded-score results and validation errors out of HTTP caches."""

    def get_route_handler(self):
        handler = super().get_route_handler()

        async def no_store(request: Request):
            try:
                response = await handler(request)
            except HTTPException as exc:
                exc.headers = {**(exc.headers or {}), "Cache-Control": "no-store, no-transform"}
                raise
            except RequestValidationError as exc:
                response = await request_validation_exception_handler(request, exc)
            response.headers["Cache-Control"] = "no-store, no-transform"
            return response

        return no_store


router = APIRouter(route_class=_NoStoreScoreRoute)
@router.post("/import/score", response_model=ScoreImportResponse)
def import_score_route(req: ScoreImportRequest) -> ScoreImportResponse:
    return _run_import(req)


def _run_import(req: ScoreImportRequest,
                on_progress: Callable[[str], None] | None = None) -> ScoreImportResponse:
    try:
        return import_score(req, on_progress=on_progress)
    except ScoreInputError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from None
    except ValidationError:
        raise HTTPException(status_code=422, detail="The score could not be represented safely. Try a supported vector PDF.") from None


@router.post("/import/score/stream")
async def import_score_stream(req: ScoreImportRequest):
    """Send actual reader stages and keep gateways alive during the score analysis."""
    def frame(event: str, data: dict) -> bytes:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()

    async def stream():
        yield frame("progress", {"label": "Checking the uploaded score…"})
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def progress(label: str) -> None:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, ("progress", {"label": label}))
            except RuntimeError:
                pass  # The event loop is closed; nobody is left to receive the label.

        async def work():
            try:
                result = await asyncio.to_thread(_run_import, req, progress)
                await queue.put(("result", result.model_dump(mode="json")))
            except HTTPException as exc:
                await queue.put(("error", {"detail": exc.detail, "status": exc.status_code}))
            except Exception:
                logging.getLogger(__name__).exception("Score import failed")
                await queue.put(("error", {"detail": "Score analysis failed. Please try again.", "status": 500}))

        # Keep a strong reference to the worker: the reading thread cannot be
        # cancelled, so its task outlives a stream that the browser aborts.
        task = asyncio.create_task(work())
        _imports.add(task)
        task.add_done_callback(_imports.discard)
        while True:
            try:
                event, data = await asyncio.wait_for(queue.get(), timeout=5)
            except TimeoutError:
                yield b": reading score\n\n"
                continue
            yield frame(event, data)
            if event in {"result", "error"}:
                break

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-store, no-transform", "X-Accel-Buffering": "no"})


_imports: set[asyncio.Task] = set()
