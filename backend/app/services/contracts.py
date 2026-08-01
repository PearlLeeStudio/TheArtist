"""Version and bounded result envelope shared by HTTP discovery and MCP."""

import base64
import binascii
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from app.models.export import MAX_EXPORT_BYTES, MAX_EXPORT_MEASURES, MIN_MIDI_BYTES
from app.models.plan import MAX_PLAN_SECTION_BARS, MIN_PLAN_SECTION_BARS, TARGET_BARS
from app.models.score_import import MAX_SCORE_BYTES, MAX_SCORE_MEASURES
from app.services.planner import RUN_BUDGET_S
from app.services import theory_rag

SCHEMA_VERSION = "1.0"
MAX_JSON_STATE_BYTES = 1024 * 1024
MAX_MCP_RESULT_BYTES = 6 * 1024 * 1024
MAX_MCP_REQUEST_BYTES = 12 * 1024 * 1024
TOOLS = ("capabilities", "list_models", "list_genres", "get_genre_preset", "generate_section",
         "build_song", "render_midi", "import_score", "export_daw", "plan_song", "validate_plan")
_MIDI_BASE64_KEYS = ("midi_b64", "dataBase64")  # build_song's and render_midi's MIDI bytes


class ContractEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schemaVersion: Literal["1.0"] = SCHEMA_VERSION
    data: dict[str, Any]


def json_size(value: Any) -> int:
    """UTF-8 bytes of the compact JSON encoding; NaN and Infinity are rejected."""
    return len(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf8"))


def _base64_length(byte_count: int) -> int:
    return 4 * ((byte_count + 2) // 3)


def _check_midi_base64(encoded: Any) -> None:
    if not isinstance(encoded, str) or len(encoded) > _base64_length(MAX_EXPORT_BYTES):
        raise ValueError("MIDI exceeds the 4 MiB binary limit.")
    try:
        binary = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("The MIDI result is not valid base64.") from None
    if not MIN_MIDI_BYTES <= len(binary) <= MAX_EXPORT_BYTES or not binary.startswith(b"MThd"):
        raise ValueError("The MIDI result is invalid or exceeds 4 MiB.")


def contract_result(data: BaseModel | dict) -> ContractEnvelope:
    payload = data.model_dump(mode="json") if isinstance(data, BaseModel) else data
    # Binary content has its own limit; music JSON remains bounded independently.
    state = {key: value for key, value in payload.items() if key not in _MIDI_BASE64_KEYS}
    if json_size(state) > MAX_JSON_STATE_BYTES:
        raise ValueError("Music JSON exceeds the 1 MiB state limit.")
    for key in _MIDI_BASE64_KEYS:
        if key in payload:
            _check_midi_base64(payload[key])
    envelope = ContractEnvelope(data=payload)
    if json_size(envelope.model_dump()) > MAX_MCP_RESULT_BYTES:
        raise ValueError("The tool result exceeds the 6 MiB envelope limit.")
    return envelope


def capabilities() -> ContractEnvelope:
    return contract_result({
        "name": "TheArtist", "transport": "streamable-http", "endpoint": "/mcp/",
        "tools": list(TOOLS),
        "inference": {"provider": "theartist", "externalModels": False, "scoreFormats": ["vector-pdf"]},
        "rag": theory_rag.status(),
        "contracts": {
            "generate_section": "SectionGenerateRequest / SectionGenerateResponse",
            "build_song": "SongGenerateRequest / SongGenerateResponse",
            "render_midi": "ExportSpec / MIDI bytes",
            "import_score": "ScoreImportRequest / ScoreImportResponse",
            "plan_song": "PlanRequest / PlanTrace (plan, validation; no sections)",
            "validate_plan": "SongPlan + PlanRequest / {ok, errors: PlanIssue[]}",
        },
        "limits": {
            "jsonStateBytes": MAX_JSON_STATE_BYTES, "midiBytes": MAX_EXPORT_BYTES,
            "toolResultBytes": MAX_MCP_RESULT_BYTES, "requestBytes": MAX_MCP_REQUEST_BYTES,
            "scoreFileBytes": MAX_SCORE_BYTES, "scoreMeasures": MAX_SCORE_MEASURES,
            "scoreAssistHarmonyBars": 8,
            "exportMeasures": MAX_EXPORT_MEASURES, "generatedBars": 64,
            "planTargetBars": list(TARGET_BARS), "planSectionBars": [MIN_PLAN_SECTION_BARS, MAX_PLAN_SECTION_BARS],
            "planRunSeconds": RUN_BUDGET_S,
        },
        "sourceFields": {
            "generate_section": ["source", "modelUsed"],
            "build_song": ["model_used", "tracks.*.source", "seed"],
            "import_score": ["analysisSource", "model", "mode", "measures.*.chordBasis",
                             "measures.*.confidence", "measures.*.note", "measures.*.notes",
                             "measures.*.writtenNotes", "measures.*.bassNotes", "measures.*.noChord",
                             "sections.*.basis"],
            "plan_song": ["sections.*.source", "sections.*.modelUsed", "sections.*.rulesSlots"],
        },
        "limitations": [
            "Score import reads vector PDFs on this server. Scanned PDFs and images are unsupported.",
            "Printed chords and section labels are preserved. Sparse note-based harmony completion uses our Music Transformer; inferred chords require review.",
            "Song forms follow genre templates; section chords use the selected Music Transformer. No external model requests are made.",
            "Reharmonize and Second ear compare model likelihood, not listening quality or preference.",
            "build_song returns one arrangement; candidate fan-out is not implemented.",
            "export_daw requires a loopback request, server opt-in, and explicit per-call confirmation.",
            "MCP plan_song plans only; generate_section produces each section's chords.",
        ],
    })
