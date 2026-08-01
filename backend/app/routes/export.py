from collections.abc import Callable, Coroutine
from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute

from app.models.export import DawOpenRequest, DawOpenResponse, DawTargetsResponse, ExportSpec
from app.services import lockdown
from app.services.daw_launcher import DawError, daw_targets, open_daw
from app.services.scripter_export import render_scripter, scripter_filename
from app.services.smf_export import ExportError, render_midi, safe_title


class ExportRoute(APIRoute):
    """Reports the first request validation error as one readable 422 message."""

    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        handler = super().get_route_handler()

        async def validated(request: Request) -> Response:
            try:
                return await handler(request)
            except RequestValidationError as exc:
                # Omit raw input values, including NaN/Infinity which cannot appear in JSON errors.
                error = exc.errors()[0]
                location = ".".join(str(part) for part in error["loc"])
                raise HTTPException(status_code=422, detail=f"Invalid export data at {location}: {error['msg']}") from None
        return validated


router = APIRouter(route_class=ExportRoute)
_daw_lockdown = lockdown.refuse_on_public_host(
    "No DAW on this server: Open in DAW works with a backend on your own Mac. Download the MIDI file "
    "(POST /api/export/midi) instead.")


@router.post("/export/midi")
def export_midi_route(spec: ExportSpec) -> Response:
    try:
        data = render_midi(spec)
    except ExportError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from None
    filename = quote(safe_title(spec.title) + ".mid", safe="")
    return Response(content=data, media_type="audio/midi", headers={
        # An ASCII fallback name, then the UTF-8 title for clients that read filename*.
        "Content-Disposition": f"attachment; filename=\"score.mid\"; filename*=UTF-8''{filename}",
        "Cache-Control": "no-store",
    })


@router.post("/export/scripter")
def export_scripter_route(spec: ExportSpec) -> Response:
    """The arrangement as a Logic Pro Scripter script; no local side effect, so a public host serves it too."""
    try:
        script = render_scripter(spec)
    except ExportError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from None
    filename = quote(scripter_filename(spec.title), safe="")
    return Response(content=script, media_type="text/javascript; charset=utf-8", headers={
        "Content-Disposition": f"attachment; filename=\"score.theartist.js\"; filename*=UTF-8''{filename}",
        "Cache-Control": "no-store",
    })


@router.get("/daw/targets", response_model=DawTargetsResponse, dependencies=[Depends(_daw_lockdown)])
def daw_targets_route() -> DawTargetsResponse:
    return daw_targets()


@router.post("/daw/open", response_model=DawOpenResponse, dependencies=[Depends(_daw_lockdown)])
def open_daw_route(req: DawOpenRequest) -> DawOpenResponse:
    try:
        return open_daw(req.target, req.spec)
    except (DawError, ExportError) as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from None
