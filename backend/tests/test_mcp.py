"""Real MCP initialize/list/call protocol over in-process ASGI; routes and DAWs are mocked.

The client's failed-connect tests fake the transport and session instead, to make their unwinding fail on cue.
"""

import asyncio
import base64
import io
import os
import unittest
from contextlib import asynccontextmanager
from unittest import mock

import httpx
import mido
from fastapi import FastAPI, HTTPException
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

import app.main
from app import mcp_server
from app.models.export import DawOpenResponse
from app.services import contracts
from theartist_client import ArtistClient, ArtistClientError
from theartist_client import client as artist_client
from tests.test_export import spec


@asynccontextmanager
async def connection(*, client_address="127.0.0.1", headers=None):
    server = mcp_server.create_mcp_server()
    application = FastAPI()
    application.mount("/mcp", server.streamable_http_app())
    async with server.session_manager.run(), httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application, client=(client_address, 12345)),
        base_url="http://127.0.0.1:8000", headers=headers,
    ) as http:
        async with streamable_http_client("http://127.0.0.1:8000/mcp/", http_client=http) as (read, write, _):
            async with ClientSession(read, write) as session:
                initialized = await session.initialize()
                yield session, http, initialized


class MCPProtocolTest(unittest.IsolatedAsyncioTestCase):
    async def test_initialize_discover_schema_and_capabilities(self):
        async with connection() as (session, _, initialized):
            self.assertEqual(initialized.serverInfo.name, "TheArtist")
            listed = await session.list_tools()
            tools = {tool.name: tool for tool in listed.tools}
            self.assertEqual(set(tools), set(contracts.TOOLS))
            self.assertIn("plan_song", tools)
            self.assertIn("validate_plan", tools)
            self.assertNotIn("explain_progression", tools)
            self.assertFalse(tools["export_daw"].annotations.readOnlyHint)
            self.assertFalse(tools["import_score"].annotations.openWorldHint)  # the score is read on this server
            self.assertFalse(tools["plan_song"].annotations.openWorldHint)  # genre templates, no network call
            self.assertFalse(tools["plan_song"].annotations.idempotentHint)
            self.assertTrue(tools["validate_plan"].annotations.readOnlyHint)
            self.assertIn("PlanRequest", tools["validate_plan"].inputSchema["$defs"])
            self.assertIn("ExportSpec", tools["render_midi"].inputSchema["$defs"])
            self.assertIn("schemaVersion", tools["render_midi"].outputSchema["properties"])
            result = await session.call_tool("capabilities", {})
            self.assertFalse(result.isError)
            self.assertEqual(result.structuredContent["schemaVersion"], "1.0")
            self.assertEqual(result.structuredContent["data"]["limits"]["midiBytes"], 4 * 1024 * 1024)
            self.assertLess(len(result.content[0].text), 150)

    async def test_existing_api_results_and_source_fields_are_preserved(self):
        section = {"source": "rules", "modelUsed": None, "suggestions": [], "startMeasure": 0}
        song = {"model_used": "rules", "tracks": {"harmony": {"source": "rule", "events": []}}, "bars": []}
        with mock.patch.object(mcp_server.generate, "generate_section", return_value=section) as generate, \
                mock.patch.object(mcp_server.song, "generate_song", return_value=song) as build, \
                mock.patch.object(mcp_server.models, "list_models", return_value={"models": [{"key": "ft_f1", "available": False}]}):
            async with connection() as (session, _, _):
                result = await session.call_tool("generate_section", {"request": {"key": "C major", "sectionType": "verse", "chordCount": 4}})
                self.assertFalse(result.isError)
                self.assertEqual(result.structuredContent["data"], section)
                result = await session.call_tool("build_song", {"request": {"genre": "pop", "length_bars": 4, "seed": 42}})
                self.assertEqual(result.structuredContent["data"], song)
                result = await session.call_tool("list_models", {})
                self.assertFalse(result.structuredContent["data"]["models"][0]["available"])
                result = await session.call_tool("get_genre_preset", {"genre": "country"})
                self.assertEqual(result.structuredContent["data"]["harmony_rhythm"], "boom_chick")
        generate.assert_called_once()
        build.assert_called_once()
        self.assertEqual(build.call_args.args[0].length_bars, 4)
        self.assertEqual(build.call_args.args[0].seed, 42)

    async def test_http_failures_and_unsupported_features_are_mcp_errors(self):
        with mock.patch.object(mcp_server.generate, "generate_section", side_effect=HTTPException(400, "Bad key")), \
                mock.patch.object(mcp_server.song, "generate_song") as build:
            async with connection() as (session, _, _):
                result = await session.call_tool("generate_section", {"request": {"key": "C major", "sectionType": "verse", "chordCount": 4}})
                self.assertTrue(result.isError)
                self.assertIn("HTTP 400: Bad key", result.content[0].text)
                for request in ({"genre": "pop", "n_candidates": 2}, {"genre": "pop", "n_candidates": 3}):
                    result = await session.call_tool("build_song", {"request": request})
                    self.assertTrue(result.isError)
                self.assertTrue((await session.call_tool("plan_song", {})).isError)
                self.assertTrue((await session.call_tool("get_genre_preset", {"genre": "unknown"})).isError)
        build.assert_not_called()

    async def test_plan_tools_validate_and_plan_without_generating(self):
        request = {"genre": "pop", "targetBars": 16, "wantOutro": True}
        good = {"sections": [{"type": "intro", "bars": 4, "cadence": "none", "reason": "Open", "cites": []},
                             {"type": "chorus", "bars": 8, "cadence": "authentic", "reason": "Hook", "cites": []},
                             {"type": "outro", "bars": 4, "cadence": "plagal", "reason": "Close", "cites": []}]}
        bad = {"sections": [{"type": "chorus", "bars": 40, "cadence": "half", "reason": "", "cites": []}]}
        with mock.patch.object(mcp_server.planner, "compose_section") as compose:
            async with connection() as (session, _, _):
                result = await session.call_tool("validate_plan", {"plan": good, "request": request})
                self.assertFalse(result.isError)
                self.assertEqual(result.structuredContent["data"], {"ok": True, "errors": []})
                result = await session.call_tool("validate_plan", {"plan": bad, "request": request})
                self.assertFalse(result.isError)
                self.assertEqual([e["code"] for e in result.structuredContent["data"]["errors"]],
                                 ["bars", "total_bars", "final_cadence", "outro_wanted"])
                result = await session.call_tool("plan_song", {"request": request})
                self.assertFalse(result.isError)
                data = result.structuredContent["data"]
                self.assertEqual((data["status"], data["sections"]), ("ok", []))
                self.assertEqual([s["type"] for s in data["plan"]["sections"]], ["intro", "verse", "outro"])
                self.assertEqual([v["ok"] for v in data["validation"]], [True])
        compose.assert_not_called()

    async def test_midi_roundtrip_and_wrapper_do_not_duplicate_large_content(self):
        async with connection() as (session, _, _):
            result = await session.call_tool("render_midi", {"spec": spec(timeSignature=[6, 8])})
            self.assertFalse(result.isError)
            output = result.structuredContent["data"]
            binary = base64.b64decode(output["dataBase64"], validate=True)
            self.assertEqual(output["byteLength"], len(binary))
            midi = mido.MidiFile(file=io.BytesIO(binary), charset="utf8")
            self.assertEqual(midi.ticks_per_beat, 480)
            self.assertEqual(sum(message.time for message in midi.tracks[0]), 2880)
            self.assertNotIn(output["dataBase64"], result.content[0].text)

    async def test_score_import_requires_attachment_and_preserves_the_readers_failure(self):
        request = {"filename": "score.pdf", "mimeType": "application/pdf", "dataBase64": "JVBERi0="}
        refusal = HTTPException(422, "Import supports readable vector PDFs.")
        with mock.patch.object(mcp_server.score_import, "import_score_route", side_effect=refusal) as imported:
            async with connection() as (session, _, _):
                result = await session.call_tool("import_score", {"request": request})
                self.assertTrue(result.isError)
                imported.assert_not_called()
                result = await session.call_tool("import_score", {"request": request, "attachmentConfirmed": True})
                self.assertTrue(result.isError)
                self.assertIn("HTTP 422: Import supports readable vector PDFs.", result.content[0].text)
        imported.assert_called_once()

    async def test_daw_handoff_requires_three_explicit_gates(self):
        args = {"request": {"target": "logic", "spec": spec()}, "confirmed": True}
        with mock.patch.object(mcp_server, "open_daw_route", return_value=DawOpenResponse(action="open", path="/mock/score.mid")) as opened:
            with mock.patch.dict(os.environ, {"THEARTIST_MCP_ALLOW_DAW": "0"}):
                async with connection() as (session, _, _):
                    self.assertTrue((await session.call_tool("export_daw", args)).isError)
            with mock.patch.dict(os.environ, {"THEARTIST_MCP_ALLOW_DAW": "1"}):
                async with connection(client_address="198.51.100.5") as (session, _, _):
                    self.assertTrue((await session.call_tool("export_daw", args)).isError)
                async with connection(headers={"X-Forwarded-For": "127.0.0.1"}) as (session, _, _):
                    self.assertTrue((await session.call_tool("export_daw", args)).isError)
                async with connection() as (session, _, _):
                    self.assertTrue((await session.call_tool("export_daw", {**args, "confirmed": False})).isError)
                    opened.assert_not_called()
                    result = await session.call_tool("export_daw", args)
                    self.assertFalse(result.isError)
                    self.assertEqual(result.structuredContent["data"]["action"], "open")
        opened.assert_called_once()

    async def test_state_and_transport_size_limits_fail_before_work(self):
        with mock.patch.object(mcp_server.song, "generate_song") as build:
            async with connection() as (session, http, _):
                with mock.patch.object(contracts, "MAX_JSON_STATE_BYTES", 10):
                    result = await session.call_tool("build_song", {"request": {"genre": "pop"}})
                    self.assertTrue(result.isError)
                response = await http.post("/mcp/", content=b"{}", headers={"Content-Length": str(contracts.MAX_MCP_REQUEST_BYTES + 1)})
                self.assertEqual(response.status_code, 413)
        build.assert_not_called()

    async def test_main_lifespan_mount_and_public_client_roundtrip(self):
        with mock.patch.dict(os.environ, {"THEARTIST_WARMUP": "0"}):
            async with app.main.app.router.lifespan_context(app.main.app), httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app.main.app), base_url="http://127.0.0.1:8000",
            ) as http:
                self.assertEqual((await http.get("/api/health")).json(), {"status": "ok"})
                expected = (await http.get("/api/capabilities")).json()["data"]
                async with ArtistClient(http_client=http) as artist:
                    self.assertEqual(await artist.capabilities(), expected)
                    binary = await artist.render_midi(spec())
                    self.assertTrue(binary.startswith(b"MThd"))
                    with self.assertRaises(ArtistClientError):
                        await artist.call_tool("get_genre_preset", {"genre": "missing"})


class MCPResultLimitsTest(unittest.TestCase):
    def test_result_state_and_binary_validation(self):
        with mock.patch.object(contracts, "MAX_JSON_STATE_BYTES", 20):
            with self.assertRaisesRegex(ValueError, "state limit"):
                contracts.contract_result({"notes": "a" * 30})
        for midi in ("invalid!", base64.b64encode(b"not midi").decode()):
            with self.subTest(midi=midi), self.assertRaises(ValueError):
                contracts.contract_result({"midi_b64": midi})


class FakeTransport:
    """Stands in for streamable_http_client; a failing exit is the task-group error a dead endpoint leaves behind."""

    def __init__(self, exit_error=None):
        self.exit_error = exit_error
        self.exits = 0

    def __call__(self, url, *, http_client=None):
        return self

    async def __aenter__(self):
        return object(), object(), lambda: None

    async def __aexit__(self, *exc):
        self.exits += 1
        if self.exit_error is not None:
            raise self.exit_error


def fake_session(connect_error=None, *, started=None):
    """Stands in for ClientSession; a refused endpoint surfaces in initialize(), after the transport is entered."""

    class FakeSession:
        def __init__(self, read, write, *, read_timeout_seconds=None):
            self.read_timeout_seconds = read_timeout_seconds

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

        async def initialize(self):
            if started is not None:
                started.set()
                await asyncio.Event().wait()  # a connect that only a cancellation ends
            if connect_error is not None:
                raise connect_error

    return FakeSession


class ArtistClientConnectTest(unittest.IsolatedAsyncioTestCase):
    async def test_failed_connect_stays_reconnectable_when_cleanup_also_fails(self):
        connect_error = httpx.ConnectError("All connection attempts failed")
        cleanup_error = ExceptionGroup("unhandled errors in a TaskGroup", [RuntimeError("post writer died")])
        broken, working = FakeTransport(exit_error=cleanup_error), FakeTransport()
        artist = ArtistClient(timeout=5)
        with mock.patch.object(artist_client, "streamable_http_client", broken), \
                mock.patch.object(artist_client, "ClientSession", fake_session(connect_error)):
            with self.assertRaises(httpx.ConnectError) as raised:
                async with artist:
                    self.fail("a refused endpoint must not connect")
        self.assertIs(raised.exception, connect_error)
        self.assertIs(connect_error.__context__, cleanup_error)  # kept visible, but behind the real error
        self.assertEqual(broken.exits, 1)
        self.assertIsNone(artist._stack)
        self.assertIsNone(artist._session)
        with mock.patch.object(artist_client, "streamable_http_client", working), \
                mock.patch.object(artist_client, "ClientSession", fake_session()):
            async with artist as connected:
                self.assertIs(connected, artist)
                self.assertEqual(artist._session.read_timeout_seconds.total_seconds(), 5)
        self.assertEqual(working.exits, 1)
        self.assertIsNone(artist._stack)
        self.assertIsNone(artist._session)

    async def test_cancellation_used_to_unwind_the_transport_does_not_hide_its_error(self):
        # A refused endpoint fails inside the transport's task group, which cancels the connect to unwind
        cleanup_error = ExceptionGroup("unhandled errors in a TaskGroup", [httpx.ConnectError("refused")])
        broken = FakeTransport(exit_error=cleanup_error)
        artist = ArtistClient()
        with mock.patch.object(artist_client, "streamable_http_client", broken), \
                mock.patch.object(artist_client, "ClientSession", fake_session(asyncio.CancelledError("Cancelled via cancel scope"))):
            with self.assertRaises(ExceptionGroup) as raised:
                async with artist:
                    self.fail("a refused endpoint must not connect")
        self.assertIs(raised.exception, cleanup_error)
        self.assertEqual(asyncio.current_task().cancelling(), 0)
        self.assertIsNone(artist._stack)
        self.assertIsNone(artist._session)

    async def test_real_cancellation_wins_over_a_failing_cleanup(self):
        broken = FakeTransport(exit_error=ExceptionGroup("unhandled errors in a TaskGroup", [RuntimeError("post writer died")]))
        started = asyncio.Event()
        artist = ArtistClient()
        with mock.patch.object(artist_client, "streamable_http_client", broken), \
                mock.patch.object(artist_client, "ClientSession", fake_session(started=started)):
            connect = asyncio.create_task(artist.__aenter__())
            await started.wait()
            connect.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await connect
        self.assertTrue(connect.cancelled())
        self.assertEqual(broken.exits, 1)
        self.assertIsNone(artist._stack)
        self.assertIsNone(artist._session)


if __name__ == "__main__":
    unittest.main()
