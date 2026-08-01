"""POST /api/plan/song (+ /stream): complete a song form from a sketch (app.services.planner).

The JSON route answers the whole trace on success and, on a failed run, the same
trace under the HTTP status of the failure. The stream sends each step as it happens
and ends with one `complete` or `error` frame carrying the trace. Nothing is applied
server-side.
"""

import asyncio
import logging
import threading
from collections.abc import AsyncIterator, Callable

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.types import Receive, Scope, Send

from app.models.plan import PlanError, PlanRequest, PlanTrace
from app.routes.generate import _SSE_HEADERS, _sse
from app.services import planner

log = logging.getLogger(__name__)

router = APIRouter()

# The failure's HTTP status
_STATUS = {"timeout": 504, "invalid": 502, "generation": 500}
_STREAM_DONE = object()


class _PlanStreamResponse(StreamingResponse):
    """Tell the run to stop whenever the response ends, also when it ends outside the body generator."""

    def __init__(self, frames: AsyncIterator[bytes], on_close: Callable[[], None]) -> None:
        super().__init__(frames, media_type="text/event-stream", headers=_SSE_HEADERS)
        self._on_close = on_close

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self._on_close()


def error_status(error: PlanError) -> int:
    return _STATUS.get(error.kind, 500)


def error_detail(trace: PlanTrace) -> str:
    """One line for callers that take a message, not a trace (MCP): the failure."""
    assert trace.error is not None
    return f"{trace.error.message}."


def plan_only(req: PlanRequest) -> PlanTrace:
    """plan_song as the MCP tool runs it: plan and validate; a failed run is raised with
    its HTTP status so the tool result is an error, never a trace that looks like a plan."""
    trace = planner.plan_song(req)
    if trace.error is not None:
        raise HTTPException(status_code=error_status(trace.error), detail=error_detail(trace))
    return trace


@router.post(
    "/plan/song",
    response_model=PlanTrace,
    responses={code: {"model": PlanTrace} for code in sorted(set(_STATUS.values()))},
)
def plan_song_route(req: PlanRequest):
    """Plan the rest of the form, then generate every planned section."""
    trace = planner.complete_form(req)
    if trace.error is not None:
        return JSONResponse(status_code=error_status(trace.error), content=trace.model_dump(mode="json"))
    return trace


@router.post("/plan/song/stream")
async def plan_song_stream(req: PlanRequest) -> StreamingResponse:
    """The same run as SSE: progress, plan, validate, generate_section (one per section),
    then one terminal complete or error frame with the trace."""
    cancelled = threading.Event()

    async def stream():
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def on_event(event: str, payload: dict) -> None:
            # The run is on a worker thread; call_soon_threadsafe keeps the frames in order
            # ahead of the done-callback's sentinel (see routes/generate.py).
            if cancelled.is_set():
                return
            try:
                loop.call_soon_threadsafe(queue.put_nowait, (event, payload))
            except RuntimeError:
                pass  # event loop closed (client disconnected)

        def worker_done(future: asyncio.Future) -> None:
            # Retrieve exceptions even after the consumer has gone away.
            if not future.cancelled():
                future.exception()
            if not cancelled.is_set():
                queue.put_nowait(_STREAM_DONE)

        # Await only the event queue, never the executor future: cancelling this
        # generator must not discard the worker's trace.
        try:
            future: asyncio.Future = loop.run_in_executor(
                None, planner.complete_form, req, on_event, cancelled.is_set)
            future.add_done_callback(worker_done)
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=5)
                except TimeoutError:
                    yield b": composing song form\n\n"
                    continue
                if item is _STREAM_DONE:
                    break
                yield _sse(*item)
            try:
                trace = future.result()
            except Exception:
                log.exception("Song planning failed")
                yield _sse("error", {"detail": "Song planning failed"})
                return
            yield _sse("error" if trace.error is not None else "complete", trace.model_dump(mode="json"))
        finally:
            # The section being composed finishes, then the planner stops
            # before starting another one.
            cancelled.set()

    return _PlanStreamResponse(stream(), cancelled.set)
