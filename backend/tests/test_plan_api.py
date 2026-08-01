"""API tests for /api/plan/song (+ /stream): the genre-form run with the rules-path section
composer, the trace's fields, the SSE order, the failure status and the request checks."""

import asyncio
import logging
import unittest
from unittest import mock

from fastapi.testclient import TestClient
from starlette.requests import ClientDisconnect

import app.main
from app.models.plan import PlanRequest
from app.routes import plan as plan_routes
from app.services import planner
from tests.test_planner import SKETCH
from tests.test_section_api import sse_events

CLIENT = TestClient(app.main.app)
PLAN = "/api/plan/song"
PLAN_STREAM = "/api/plan/song/stream"
TRACE_KEYS = {"status", "genre", "key", "bpm", "timeSignature", "plan", "validation", "sections",
              "elapsedMs", "generationMs", "error"}
_QUIET = logging.NullHandler()


def setUpModule():
    for name in ("app.services.planner", "app.routes.plan"):
        logging.getLogger(name).addHandler(_QUIET)


def tearDownModule():
    for name in ("app.services.planner", "app.routes.plan"):
        logging.getLogger(name).removeHandler(_QUIET)


def body(**overrides) -> dict:
    req = {"genre": "pop", "key": "C major", "measures": SKETCH, "sections": [{"type": "verse", "start": 0, "length": 8}],
           "targetBars": 32, "wantOutro": True}
    req.update(overrides)
    return req


class UnstartedPlanStreamTest(unittest.TestCase):
    """A response that never reaches its body starts no run."""

    def test_disconnect_during_headers_starts_no_worker(self):
        async def disconnect():
            headers_started = asyncio.Event()

            async def send(message):
                if message["type"] == "http.response.start":
                    headers_started.set()
                    await asyncio.Future()  # response-header backpressure

            async def receive():
                await headers_started.wait()
                return {"type": "http.disconnect"}

            response = await plan_routes.plan_song_stream(PlanRequest(**body()))
            await asyncio.wait_for(response({"type": "http", "asgi": {"spec_version": "2.3"}}, receive, send), 2)

        with mock.patch.object(planner, "complete_form") as complete:
            asyncio.run(disconnect())
        complete.assert_not_called()

    def test_header_send_failure_starts_no_worker_on_newer_asgi_servers(self):
        async def disconnect():
            async def send(_message):
                raise OSError("client disconnected before body")

            async def receive():
                raise AssertionError("ASGI 2.4 handles disconnects through send")

            response = await plan_routes.plan_song_stream(PlanRequest(**body()))
            with self.assertRaises(ClientDisconnect):
                await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)

        with mock.patch.object(planner, "complete_form") as complete:
            asyncio.run(disconnect())
        complete.assert_not_called()

    def test_executor_rejection_fails_the_response_without_a_run(self):
        async def rejected():
            async def send(_message):
                pass

            async def receive():
                raise AssertionError("ASGI 2.4 handles disconnects through send")

            response = await plan_routes.plan_song_stream(PlanRequest(**body()))
            with mock.patch.object(asyncio.get_running_loop(), "run_in_executor",
                                   side_effect=RuntimeError("executor stopped")), \
                    self.assertRaisesRegex(RuntimeError, "executor stopped"):
                await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)

        with mock.patch.object(planner, "complete_form") as complete:
            asyncio.run(rejected())
        complete.assert_not_called()


class PlanApiTest(unittest.TestCase):
    def setUp(self):
        # The chord model is not needed: the rules composer runs the real song-form path
        patcher = mock.patch("app.routes.generate.TORCH_AVAILABLE", False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_new_song_takes_the_genre_defaults_and_the_genre_form(self):
        res = CLIENT.post(PLAN, json={"genre": "jazz", "targetBars": 32})
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()
        self.assertEqual(set(data), TRACE_KEYS)
        self.assertEqual((data["status"], data["error"]), ("ok", None))
        self.assertEqual((data["genre"], data["key"], data["bpm"], data["timeSignature"]), ("jazz", "F major", 110, [4, 4]))
        self.assertEqual([(s["sectionType"], s["startMeasure"]) for s in data["sections"]],
                         [("a_section", 0), ("a_section", 8), ("b_section", 16), ("a_section", 24)])
        self.assertEqual(data["sections"][-1]["suggestions"][0]["chords"]["31"], ["C7", "Fmaj"])

    def test_the_stream_sends_each_step_in_run_order_and_ends_with_the_trace(self):
        res = CLIENT.post(PLAN_STREAM, json={"genre": "jazz", "targetBars": 32})
        self.assertEqual(res.status_code, 200, res.text)
        self.assertTrue(res.headers["content-type"].startswith("text/event-stream"))
        events = sse_events(res.text)
        self.assertEqual([name for name, _ in events],
                         ["progress", "plan", "validate", "progress", "generate_section", "progress", "generate_section",
                          "progress", "generate_section", "progress", "generate_section", "complete"])
        for name, payload in events[:-1]:
            with self.subTest(event=name):
                self.assertIn("t_ms", payload)
        self.assertEqual([payload["label"] for name, payload in events if name == "progress"],
                         ["Planning the song form from the sketch…", "Composing a section (1/4)…",
                          "Composing a section (2/4)…", "Composing b section (3/4)…", "Composing a section (4/4)…"])
        plan, validate, final = events[1][1], events[2][1], events[-1][1]
        self.assertEqual(set(plan), {"attempt", "source", "plan", "t_ms"})
        self.assertEqual((plan["attempt"], plan["source"]), (1, "rules"))
        self.assertEqual([(s["type"], s["bars"], s["cadence"]) for s in plan["plan"]["sections"]],
                         [("a_section", 8, "authentic"), ("a_section", 8, "authentic"), ("b_section", 8, "half_iv_v"),
                          ("a_section", 8, "authentic")])
        self.assertEqual({key: value for key, value in validate.items() if key != "t_ms"},
                         {"attempt": 1, "ok": True, "errors": []})
        sections = [payload for name, payload in events if name == "generate_section"]
        self.assertEqual([(s["index"], s["sectionType"], s["startMeasure"], s["bars"]) for s in sections],
                         [(0, "a_section", 0, 8), (1, "a_section", 8, 8), (2, "b_section", 16, 8), (3, "a_section", 24, 8)])
        self.assertEqual(set(final), TRACE_KEYS)
        self.assertEqual((final["status"], final["error"], final["plan"]), ("ok", None, plan["plan"]))
        self.assertEqual([v["ok"] for v in final["validation"]], [True])
        # The terminal frame carries the sections exactly as the stream sent them
        self.assertEqual(final["sections"],
                         [{key: value for key, value in s.items() if key not in ("index", "t_ms")} for s in sections])
        self.assertEqual(final["sections"][-1]["suggestions"][0]["chords"]["31"], ["C7", "Fmaj"])

    def test_a_failed_run_answers_its_trace_under_the_failure_status(self):
        with mock.patch.object(planner, "OWN_MODEL_BUDGET_S", 0):
            res = CLIENT.post(PLAN, json=body())
            events = sse_events(CLIENT.post(PLAN_STREAM, json=body()).text)
        self.assertEqual(res.status_code, 504, res.text)
        data = res.json()
        self.assertEqual(set(data), TRACE_KEYS)
        self.assertEqual((data["status"], data["sections"]), ("error", []))
        self.assertEqual(set(data["error"]), {"kind", "message"})
        self.assertEqual(data["error"]["kind"], "timeout")
        self.assertIn("chord-model time", data["error"]["message"])
        self.assertIsNotNone(data["plan"])
        self.assertEqual([name for name, _ in events], ["progress", "plan", "validate", "progress", "error"])
        self.assertEqual((events[-1][1]["status"], events[-1][1]["error"]), ("error", data["error"]))

    def test_unexpected_worker_failure_ends_the_stream_with_one_error_frame(self):
        with mock.patch.object(planner, "complete_form", side_effect=RuntimeError("worker failure")):
            events = sse_events(CLIENT.post(PLAN_STREAM, json=body()).text)
        self.assertEqual(events, [("error", {"detail": "Song planning failed"})])

    def test_bad_bodies_are_422_on_both_routes(self):
        for route in (PLAN, PLAN_STREAM):
            for bad in ({"targetBars": 20}, {"contrast": "wild"}, {"genre": "polka"}, {"key": "H major"},
                        {"measures": SKETCH * 4}, {"sections": [{"type": "verse", "start": 4, "length": 8}]},
                        {"measures": [{"index": 0, "chords": ["C", "F", "G"], "roman": ["I", "IV", "V"]}]},
                        {"prompt": "make it sad"},
                        # Request fields of earlier builds that chose an external model
                        {"planner": "rules"}, {"chordSource": "own"}):
                with self.subTest(route=route, bad=bad):
                    self.assertEqual(CLIENT.post(route, json=body(**bad)).status_code, 422)

    def test_contrast_is_still_accepted(self):
        res = CLIENT.post(PLAN, json={"genre": "jazz", "targetBars": 16, "contrast": "high", "chorusMode": "reuse"})
        self.assertEqual(res.status_code, 200, res.text)
        self.assertEqual(res.json()["status"], "ok")


if __name__ == "__main__":
    unittest.main()
