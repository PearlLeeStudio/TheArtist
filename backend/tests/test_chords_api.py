"""API tests for /api/generate/chords: the key check runs before any model
is touched, on the AI path and on the rules path alike.

The rules path is forced by patching TORCH_AVAILABLE off; the AI-path test
makes ModelRegistry.get fail loudly, so a 400 proves no model was loaded.
"""

import unittest
from unittest import mock

from fastapi.testclient import TestClient

import app.main
from app.routes.generate import VALID_ROOTS

CLIENT = TestClient(app.main.app)
CHORDS = "/api/generate/chords"
BAD_KEYS = ("H major", "C dorian", "Cmajor", "")


def body(**overrides) -> dict:
    req = {"key": "C major", "genre": "pop", "n_bars": 4}
    req.update(overrides)
    return req


class ChordsKeyValidationTest(unittest.TestCase):
    def assert_bad_key_is_400(self, key: str) -> None:
        res = CLIENT.post(CHORDS, json=body(key=key))
        self.assertEqual(res.status_code, 400)
        self.assertIn("Expected key format", res.json()["detail"])
        self.assertNotIn("bars", res.json())

    def test_bad_key_is_400_on_the_rules_path(self):
        # Before the fix, "H major" reached chord_generator and became a 500.
        with mock.patch("app.routes.chords.TORCH_AVAILABLE", False):
            for key in BAD_KEYS:
                with self.subTest(key=key):
                    self.assert_bad_key_is_400(key)

    def test_bad_key_is_400_before_any_model_is_touched(self):
        # Before the fix, the AI path accepted "H major" and echoed it back.
        with mock.patch("app.routes.chords.TORCH_AVAILABLE", True), \
             mock.patch("app.routes.chords.ModelRegistry.get",
                        side_effect=AssertionError("model touched before validation")) as get:
            for key in BAD_KEYS:
                with self.subTest(key=key):
                    self.assert_bad_key_is_400(key)
        get.assert_not_called()


class ChordsRulesPathTest(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch("app.routes.chords.TORCH_AVAILABLE", False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_valid_key_is_200(self):
        res = CLIENT.post(CHORDS, json=body())
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(set(data), {"key", "genre", "bpm", "model_used", "bars"})
        self.assertEqual((data["key"], data["genre"], data["model_used"]), ("C major", "pop", "rules"))
        self.assertEqual(len(data["bars"]), 4)
        for bar in data["bars"]:
            self.assertEqual(len(bar["chords"]), 2)

    def test_every_valid_root_passes(self):
        # The same roots /api/generate accepts; none may reach a 500 in the rules generator.
        for root in sorted(VALID_ROOTS):
            for mode in ("major", "minor"):
                with self.subTest(key=f"{root} {mode}"):
                    res = CLIENT.post(CHORDS, json=body(key=f"{root} {mode}", n_bars=2))
                    self.assertEqual(res.status_code, 200)
                    self.assertEqual(res.json()["model_used"], "rules")

    def test_candidates_shape(self):
        res = CLIENT.post(CHORDS, json=body(n_candidates=3))
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertNotIn("bars", data)
        self.assertEqual(len(data["candidates"]), 3)
        for candidate in data["candidates"]:
            self.assertEqual(set(candidate), {"bars", "temperature", "seed"})
            self.assertEqual(len(candidate["bars"]), 4)


if __name__ == "__main__":
    unittest.main()
