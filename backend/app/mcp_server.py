"""MCP is an adapter over the same application contracts used by the editor."""

import base64
import ipaddress
import logging
import os
from collections.abc import Callable
from functools import partial
from typing import Annotated, Any

import anyio
from fastapi import HTTPException
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import CallToolResult, TextContent, ToolAnnotations
from pydantic import BaseModel, Field, StrictBool

from app.models.export import DawOpenRequest, ExportSpec
from app.models.plan import PlanRequest, SongPlan
from app.models.schemas import SectionGenerateRequest, SongGenerateRequest
from app.models.score_import import ScoreImportRequest
from app.routes import generate, models, score_import, song
from app.routes import plan as plan_routes
from app.routes.export import open_daw_route
from app.services import contracts, genre_tables, planner, smf_export
from app.services.contracts import ContractEnvelope

log = logging.getLogger(__name__)
READ = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)
GENERATE = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=False, openWorldHint=False)
IMPORT = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False)
LOCAL_WRITE = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False)
_INSTRUCTIONS = (
    "Shared music services for TheVoyager, the editor, and future plugin clients. "
    "Preserve source/model fields. Never claim rule-based output came from a model. "
    "Send only explicitly attached score files; never collect private photos or journals. "
    "Confirm local DAW handoff with the user before using export_daw."
)
_MAX_SECTION_CONTEXT_MEASURES = 512
_PROXY_HEADERS = ("forwarded", "x-forwarded-for", "x-real-ip")


def _result(envelope: ContractEnvelope) -> CallToolResult:
    # A brief text block avoids duplicating a multi-megabyte MIDI in the protocol response.
    return CallToolResult(content=[TextContent(type="text", text="TheArtist result (schema 1.0); see structuredContent.data.")],
                          structuredContent=envelope.model_dump(mode="json"))


async def _invoke(function: Callable[..., Any], *args: Any) -> CallToolResult:
    """Run a blocking application call in a worker thread; its failures become tool errors."""
    try:
        output = await anyio.to_thread.run_sync(partial(function, *args))
        return _result(contracts.contract_result(output))
    except HTTPException as exc:
        raise ToolError(f"HTTP {exc.status_code}: {exc.detail}") from None
    except smf_export.ExportError as exc:  # a ValueError that carries its HTTP status
        raise ToolError(f"HTTP {exc.status_code}: {exc}") from None
    except ValueError as exc:
        raise ToolError(str(exc)) from None
    except Exception:
        log.exception("TheArtist MCP operation failed")
        raise ToolError("TheArtist could not complete this operation; check the backend log.") from None


def _bounded_state(request: BaseModel) -> None:
    if contracts.json_size(request.model_dump(mode="json")) > contracts.MAX_JSON_STATE_BYTES:
        raise ToolError("Music JSON exceeds the 1 MiB state limit.")


def _local_request(ctx: Context) -> bool:
    """A direct connection from a loopback address, not one relayed by a proxy."""
    request = ctx.request_context.request
    if request is None or request.client is None:
        return False
    if any(name in request.headers for name in _PROXY_HEADERS):
        return False
    try:
        return ipaddress.ip_address(request.client.host).is_loopback
    except ValueError:
        return False


def _midi_file(spec: ExportSpec) -> dict[str, Any]:
    """The same bytes as POST /api/export/midi, as a base64 file record."""
    data = smf_export.render_midi(spec)
    return {"mimeType": "audio/midi", "filename": smf_export.safe_title(spec.title) + ".mid",
            "byteLength": len(data), "dataBase64": base64.b64encode(data).decode("ascii")}


def create_mcp_server() -> FastMCP:
    server = FastMCP(
        "TheArtist", instructions=_INSTRUCTIONS,
        streamable_http_path="/", json_response=True, stateless_http=True,
        max_request_body_size=contracts.MAX_MCP_REQUEST_BYTES,
    )

    @server.tool(annotations=READ)
    async def capabilities() -> ContractEnvelope:
        """Schema version, existing tools, reusable API contracts, limits, and unimplemented features."""
        return _result(contracts.capabilities())

    @server.tool(annotations=READ)
    async def list_models() -> ContractEnvelope:
        """List installed model availability and labels, without loading a model."""
        return await _invoke(models.list_models)

    @server.tool(annotations=READ)
    async def list_genres() -> ContractEnvelope:
        """List the supported genres and their current model dispatch."""
        return await _invoke(models.list_genres)

    @server.tool(annotations=READ)
    async def get_genre_preset(genre: Annotated[str, Field(min_length=1, max_length=40)]) -> ContractEnvelope:
        """Get existing key, tempo, meter, rhythm and instrument defaults for a supported genre."""
        if genre not in genre_tables.GENRE_DEFAULTS:
            raise ToolError("Unsupported genre; use list_genres for the supported vocabulary.")
        return _result(contracts.contract_result({
            "genre": genre, **genre_tables.genre_defaults(genre),
            "harmony_rhythm": genre_tables.genre_rhythm_preset(genre)["id"],
            "harmony_instrument": genre_tables.harmony_instrument(genre),
            "bass_instrument": genre_tables.bass_instrument(genre),
            "drum_pattern": genre_tables.genre_drum_pattern_id(genre),
        }))

    @server.tool(annotations=GENERATE)
    async def generate_section(request: SectionGenerateRequest) -> ContractEnvelope:
        """Use the same section composer as POST /api/generate/section; return explicit source/model fields."""
        _bounded_state(request)
        if len(request.context) > _MAX_SECTION_CONTEXT_MEASURES:
            raise ToolError(f"A section request supports at most {_MAX_SECTION_CONTEXT_MEASURES} context measures.")
        return await _invoke(generate.generate_section, request)

    @server.tool(annotations=GENERATE)
    async def plan_song(request: PlanRequest) -> ContractEnvelope:
        """Plan the sections after the locked sketch using genre templates. Returns a validated plan without generating chords; follow with generate_section for each section."""
        _bounded_state(request)
        return await _invoke(plan_routes.plan_only, request)

    @server.tool(annotations=READ)
    async def validate_plan(plan: SongPlan, request: PlanRequest) -> ContractEnvelope:
        """Check a song plan against the request's sketch, locked sections and chips with the planner's own validator; returns {ok, errors} with machine-readable issues and makes no model call."""
        _bounded_state(plan)
        _bounded_state(request)
        issues = planner.validate_plan(plan, request)
        return _result(contracts.contract_result({"ok": not issues, "errors": [issue.model_dump() for issue in issues]}))

    @server.tool(annotations=GENERATE)
    async def build_song(request: SongGenerateRequest) -> ContractEnvelope:
        """Use POST /api/generate/song's engine and contract. Returns one arrangement with optional request-local seed."""
        _bounded_state(request)
        if request.n_candidates != 1:
            raise ToolError("build_song currently supports n_candidates=1 only; candidate fan-out is not implemented.")
        return await _invoke(song.generate_song, request)

    @server.tool(annotations=READ)
    async def render_midi(spec: ExportSpec) -> ContractEnvelope:
        """Render the same validated ExportSpec as POST /api/export/midi; returns bounded base64 MIDI without saving files."""
        _bounded_state(spec)
        return await _invoke(_midi_file, spec)

    @server.tool(annotations=IMPORT)
    async def import_score(request: ScoreImportRequest, attachmentConfirmed: StrictBool = False) -> ContractEnvelope:
        """Read an attached vector PDF on this server. chord_symbols reads written chords; infer_harmony analyzes notes and uses our Music Transformer for sparse harmony completion. Interpretations require review; no path/URL access."""
        if not attachmentConfirmed:
            raise ToolError("Confirm that the user explicitly attached this score for analysis.")
        return await _invoke(score_import.import_score_route, request)

    @server.tool(annotations=LOCAL_WRITE)
    async def export_daw(request: DawOpenRequest, ctx: Context, confirmed: StrictBool = False) -> ContractEnvelope:
        """SIDE EFFECT: save a MIDI and open Logic or reveal for Ableton. Requires local loopback + server opt-in + user confirmation."""
        if not confirmed or os.getenv("THEARTIST_MCP_ALLOW_DAW") != "1" or not _local_request(ctx):
            raise ToolError("Local DAW handoff needs explicit user confirmation, THEARTIST_MCP_ALLOW_DAW=1, "
                            "and a direct loopback request. Remote/proxied calls are not allowed.")
        _bounded_state(request)
        return await _invoke(open_daw_route, request)

    return server
