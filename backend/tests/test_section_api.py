"""API tests for /api/generate/section (+ /stream) and the sectionType change on /api/generate.

Generated sections stay at or below 8 bars so the suite runs fast on CPU.
The AI-path assertions run only when torch and the F1 checkpoint are present.
"""

import json
import unittest
from unittest import mock

from fastapi.testclient import TestClient

import app.main
from app.services import inference
from app.services.voice_leading import _parse

CLIENT = TestClient(app.main.app)
AI_READY = inference.TORCH_AVAILABLE and inference.CHECKPOINTS["ft_f1"].exists()
C_MAJOR_PCS = {0, 2, 4, 5, 7, 9, 11}
SECTION = "/api/generate/section"
SECTION_STREAM = "/api/generate/section/stream"
RESPONSE_KEYS = {"sectionType", "title", "startMeasure", "bars", "chordCount", "suggestions", "source", "modelUsed"}


def body(**overrides) -> dict:
    req = {"key": "C major", "genre": "pop", "sectionType": "chorus", "chordCount": 8}
    req.update(overrides)
    return req


def sse_events(text: str) -> list[tuple[str, dict]]:
    events = []
    for frame in text.split("\n\n"):
        if frame.strip():
            fields = dict(line.split(": ", 1) for line in frame.split("\n"))
            events.append((fields["event"], json.loads(fields["data"])))
    return events


class SectionValidationTest(unittest.TestCase):
    def test_body_violations_are_422_on_both_routes(self):
        for route in (SECTION, SECTION_STREAM):
            for bad in ({"sectionType": "coda"}, {"chordCount": 1}, {"chordCount": 129}, {"bpm": 10},
                        {"seed": -1}, {"seed": True}, {"seed": 2**32}, {"seed": "7"}):
                with self.subTest(route=route, bad=bad):
                    self.assertEqual(CLIENT.post(route, json=body(**bad)).status_code, 422)
            with self.subTest(route=route, missing="sectionType"):
                req = body()
                del req["sectionType"]
                self.assertEqual(CLIENT.post(route, json=req).status_code, 422)

    def test_bad_key_is_400_on_json_and_one_error_event_on_stream(self):
        res = CLIENT.post(SECTION, json=body(key="H major"))
        self.assertEqual(res.status_code, 400)
        self.assertIn("Expected key format", res.json()["detail"])

        res = CLIENT.post(SECTION_STREAM, json=body(key="C dorian"))
        self.assertEqual(res.status_code, 200)
        events = sse_events(res.text)
        self.assertEqual([e for e, _ in events], ["error"])
        self.assertIn("Expected key format", events[0][1]["detail"])


class SectionShapeTest(unittest.TestCase):
    @unittest.skipUnless(AI_READY, "requires the chord-model checkpoint")
    def test_seeded_neural_section_is_reproducible_without_global_reseeding(self):
        first = CLIENT.post(SECTION, json=body(seed=1379, modelKey="ft_f1")).json()
        # Interleave another seed: the first request owns its random stream.
        CLIENT.post(SECTION, json=body(seed=1380, modelKey="ft_f1"))
        repeated = CLIENT.post(SECTION, json=body(seed=1379, modelKey="ft_f1")).json()
        self.assertEqual(first["source"], "ai")
        self.assertEqual(first, repeated)

    def assert_section_shape(self, data: dict, *, start: int, bars: int, title: str) -> None:
        self.assertEqual(data["sectionType"], "chorus")
        self.assertEqual(data["title"], title)
        self.assertEqual(data["startMeasure"], start)
        self.assertEqual(data["bars"], bars)
        self.assertEqual(data["chordCount"], bars * 2)
        self.assertIn(data["source"], ("ai", "rules"))
        self.assertIn(len(data["suggestions"]), (1, 2, 3))
        expected = [str(m) for m in range(start, start + bars)]
        for s in data["suggestions"]:
            self.assertEqual(sorted(s["chords"], key=int), expected)
            for chords in s["chords"].values():
                self.assertEqual(len(chords), 2)
        if data["source"] == "rules":
            self.assertIsNone(data["modelUsed"])
        if AI_READY:
            self.assertEqual(data["source"], "ai")
            self.assertEqual(data["modelUsed"], "ft_f1")

    def test_json_route_fresh_section(self):
        res = CLIENT.post(SECTION, json=body(chordCount=7))  # odd count rounds up to 4 bars
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(set(data), RESPONSE_KEYS)
        self.assert_section_shape(data, start=0, bars=4, title="Chorus — 4 bars")

    def test_json_route_after_context(self):
        context = [{"measure": i, "chords": pair} for i, pair in
                   enumerate([["Cmaj7", "Am7"], ["Dm7", "G7"], ["Cmaj7", "Am7"], ["Fmaj7", "G7"]])]
        res = CLIENT.post(SECTION, json=body(chordCount=8, context=context))
        self.assertEqual(res.status_code, 200)
        self.assert_section_shape(res.json(), start=4, bars=4, title="Chorus — 4 bars")

    def test_single_bar_title(self):
        res = CLIENT.post(SECTION, json=body(chordCount=2))
        self.assertEqual(res.status_code, 200)
        self.assert_section_shape(res.json(), start=0, bars=1, title="Chorus — 1 bar")

    def test_stream_route_ends_with_one_result(self):
        res = CLIENT.post(SECTION_STREAM, json=body(chordCount=8))
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.headers["content-type"].startswith("text/event-stream"))
        events = sse_events(res.text)
        names = [e for e, _ in events]
        self.assertEqual(names[-1], "result")
        self.assertEqual(names.count("result"), 1)
        self.assertNotIn("error", names)
        result = events[-1][1]
        self.assertEqual(set(result), RESPONSE_KEYS | {"elapsed_ms"})
        self.assert_section_shape(result, start=0, bars=4, title="Chorus — 4 bars")
        for name, data in events[:-1]:
            self.assertIn("label", data)
            self.assertIn("t_ms", data)
        if AI_READY:
            self.assertEqual(names.count("composing"), 3)
            self.assertEqual(names[-2], "complete")
            self.assertIn("ranking", names)

    @unittest.skipUnless(AI_READY, "needs torch and the F1 checkpoint")
    def test_explicit_model_key_is_reported(self):
        res = CLIENT.post(SECTION, json=body(chordCount=4, modelKey="ft_f4"))
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["modelUsed"], "ft_f4")

    @unittest.skipUnless(AI_READY, "needs torch and the F1 checkpoint")
    def test_resolving_sections_end_on_a_root_position_tonic(self):
        # Voice leading runs after song form; it must not invert the final I.
        for section, ending in (("chorus", ["G7", "Cmaj"]), ("outro", ["Fmaj7", "Cmaj"]), ("tag", ["G7", "Cmaj"])):
            res = CLIENT.post(SECTION, json=body(sectionType=section, chordCount=6))
            self.assertEqual(res.status_code, 200)
            for s in res.json()["suggestions"]:
                with self.subTest(section=section, label=s["label"]):
                    last_bar = s["chords"]["2"]
                    self.assertEqual(last_bar[-1], ending[-1])
                    self.assertEqual(last_bar[0].split("/")[0], ending[0])

    def test_fresh_c_major_pop_section_stays_in_c_major(self):
        res = CLIENT.post(SECTION, json=body(sectionType="verse", chordCount=16))
        self.assertEqual(res.status_code, 200)
        a = res.json()["suggestions"][0]
        roots = [_parse(c)[0] for chords in a["chords"].values() for c in chords]
        in_key = sum(pc in C_MAJOR_PCS for pc in roots)
        self.assertGreater(in_key, len(roots) / 2, a["chords"])


class RulesFallbackTest(unittest.TestCase):
    def test_rules_path_applies_song_form(self):
        with mock.patch("app.routes.generate.TORCH_AVAILABLE", False):
            res = CLIENT.post(SECTION, json=body(sectionType="outro", chordCount=8, genre=None))
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual((data["source"], data["modelUsed"]), ("rules", None))
        self.assertEqual(data["title"], "Outro — 4 bars")
        for s in data["suggestions"]:
            self.assertEqual(s["chords"]["3"], ["Fmaj7", "Cmaj"])

    def test_rules_stream_has_only_the_result(self):
        with mock.patch("app.routes.generate.TORCH_AVAILABLE", False):
            res = CLIENT.post(SECTION_STREAM, json=body(sectionType="turnaround", chordCount=4))
        events = sse_events(res.text)
        self.assertEqual([e for e, _ in events], ["result"])
        data = events[0][1]
        self.assertEqual(data["source"], "rules")
        for s in data["suggestions"]:
            self.assertEqual(s["chords"]["0"] + s["chords"]["1"], ["Cmaj7", "Am7", "Dm7", "G7"])

    def test_generation_failure_ends_the_stream_with_one_error(self):
        with mock.patch("app.routes.generate.TORCH_AVAILABLE", True), \
             mock.patch("app.routes.generate.generate_ai_result", side_effect=RuntimeError("boom")), \
             self.assertLogs("app.routes.generate", level="ERROR"):
            res = CLIENT.post(SECTION_STREAM, json=body())
        events = sse_events(res.text)
        self.assertEqual([e for e, _ in events], ["error"])
        self.assertEqual(events[0][1], {"detail": "Chord generation failed"})


class GenerateSectionTypeTest(unittest.TestCase):
    """/api/generate keeps its shape; sectionType takes the full vocabulary."""

    def generate_body(self, **overrides) -> dict:
        req = {
            "key": "C major",
            "genre": "pop",
            "context": [{"measure": 0, "chords": [None, None]}, {"measure": 1, "chords": [None, None]}],
            "selectedMeasures": [0, 1],
            "sectionType": "turnaround",
        }
        req.update(overrides)
        return req

    def test_new_section_id_is_accepted(self):
        res = CLIENT.post("/api/generate", json=self.generate_body())
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(set(data), {"suggestions", "source"})
        for s in data["suggestions"]:
            roots = [_parse(c)[0] for c in s["chords"]["0"] + s["chords"]["1"]]
            self.assertEqual(roots, [0, 9, 2, 7])  # I vi ii V, whatever the inversions

    def test_unknown_section_id_is_400(self):
        res = CLIENT.post("/api/generate", json=self.generate_body(sectionType="coda"))
        self.assertEqual(res.status_code, 400)
        events = sse_events(CLIENT.post("/api/generate/stream", json=self.generate_body(sectionType="coda")).text)
        self.assertEqual([e for e, _ in events], ["error"])

    def test_more_than_64_measures_is_400(self):
        context = [{"measure": i, "chords": [None, None]} for i in range(65)]
        res = CLIENT.post("/api/generate", json=self.generate_body(context=context, selectedMeasures=list(range(65))))
        self.assertEqual(res.status_code, 400)
        self.assertIn("64", res.json()["detail"])


if __name__ == "__main__":
    unittest.main()
