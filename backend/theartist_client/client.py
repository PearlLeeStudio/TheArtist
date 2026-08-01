"""Async MCP adapter usable by a companion app or a future plugin host."""

import asyncio
import base64
import binascii
import json
from contextlib import AsyncExitStack
from datetime import timedelta
from typing import Any

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

SCHEMA_VERSION = "1.0"
MAX_RESULT_BYTES = 6 * 1024 * 1024
MAX_MIDI_BYTES = 4 * 1024 * 1024
_MIN_MIDI_BYTES = 14  # a MIDI file holds at least its 14-byte MThd header chunk
_MAX_ERROR_CHARS = 2000


class ArtistClientError(RuntimeError):
    pass


def _is_envelope(payload: object) -> bool:
    """A {"schemaVersion": "1.0", "data": {...}} tool result."""
    return (isinstance(payload, dict) and payload.get("schemaVersion") == SCHEMA_VERSION
            and isinstance(payload.get("data"), dict))


def _base64_length(byte_count: int) -> int:
    return 4 * ((byte_count + 2) // 3)


class ArtistClient:
    def __init__(self, endpoint: str = "http://127.0.0.1:8000/mcp/", *, timeout: float = 120,
                 http_client: httpx.AsyncClient | None = None) -> None:
        self.endpoint = endpoint
        self.timeout = timeout
        self._http = http_client
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None

    async def __aenter__(self) -> "ArtistClient":
        if self._stack is not None:
            raise ArtistClientError("ArtistClient is already connected.")
        stack = AsyncExitStack()
        self._stack = stack
        try:
            http = self._http or await stack.enter_async_context(httpx.AsyncClient(timeout=self.timeout))
            read, write, _ = await stack.enter_async_context(streamable_http_client(self.endpoint, http_client=http))
            self._session = await stack.enter_async_context(ClientSession(read, write,
                read_timeout_seconds=timedelta(seconds=self.timeout)))
            await self._session.initialize()
            return self
        except BaseException as error:
            try:
                await stack.aclose()
            except Exception:
                # The transport's task group can fail while unwinding. The connection error stays on top
                # of that, unless it is the cancellation anyio used to unwind that group (already retracted,
                # so the task is not cancelling): then the group holds the real error, e.g. a refused endpoint
                if isinstance(error, asyncio.CancelledError) and not asyncio.current_task().cancelling():
                    raise
                raise error
            finally:
                self._stack = self._session = None
            raise

    async def __aexit__(self, *exc) -> None:
        if self._stack is not None:
            await self._stack.__aexit__(*exc)
            self._stack = self._session = None

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        if self._session is None:
            raise ArtistClientError("Connect with 'async with ArtistClient(...)' before calling a tool.")
        result = await self._session.call_tool(name, arguments or {})
        if result.isError:
            text = " ".join(block.text for block in result.content if block.type == "text")
            raise ArtistClientError(text[:_MAX_ERROR_CHARS] or "TheArtist tool failed.")
        payload = result.structuredContent
        if not _is_envelope(payload):
            raise ArtistClientError("TheArtist returned an unsupported or missing schema envelope.")
        if len(json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf8")) > MAX_RESULT_BYTES:
            raise ArtistClientError("TheArtist result exceeds the 6 MiB limit.")
        return payload["data"]

    async def capabilities(self) -> dict[str, Any]:
        return await self.call_tool("capabilities")

    async def build_song(self, request: dict[str, Any]) -> dict[str, Any]:
        return await self.call_tool("build_song", {"request": request})

    async def generate_section(self, request: dict[str, Any]) -> dict[str, Any]:
        return await self.call_tool("generate_section", {"request": request})

    async def render_midi(self, spec: dict[str, Any]) -> bytes:
        result = await self.call_tool("render_midi", {"spec": spec})
        encoded = result.get("dataBase64")
        if not isinstance(encoded, str) or len(encoded) > _base64_length(MAX_MIDI_BYTES):
            raise ArtistClientError("The MIDI result exceeds 4 MiB or has no bytes.")
        try:
            data = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error):
            raise ArtistClientError("The MIDI result is not valid base64.") from None
        if (not data.startswith(b"MThd") or not _MIN_MIDI_BYTES <= len(data) <= MAX_MIDI_BYTES
                or result.get("byteLength") != len(data)):
            raise ArtistClientError("The MIDI result is invalid or has an inconsistent byte length.")
        return data

    async def import_score(self, request: dict[str, Any], *, attachment_confirmed: bool) -> dict[str, Any]:
        return await self.call_tool("import_score", {"request": request, "attachmentConfirmed": attachment_confirmed})
