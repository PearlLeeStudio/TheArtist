"""Public-host lockdown (THEARTIST_PUBLIC_HOST) through the in-process API; no network."""

import os
import unittest
from unittest import mock

from fastapi.testclient import TestClient
from starlette.routing import Mount

import app.main
from app.routes import score_import as score_import_routes
from app.services import lockdown
from app.services.score_import import ScoreInputError
from tests.test_export import spec
from tests.test_score_import import body as score_body

CLIENT = TestClient(app.main.app)
IMPORT = "/api/import/score"
_LOCKDOWN_VARS = (lockdown.PUBLIC_HOST_VAR, app.main.WARMUP_VAR, "THEARTIST_WARMUP_KEYS")


class LockdownCase(unittest.TestCase):
    """The lockdown and warmup variables are unset for each test; `env` sets the test's own, undone afterwards."""

    def setUp(self):
        self.patch(mock.patch.dict(os.environ))  # restores the whole environment when the test ends
        for name in _LOCKDOWN_VARS:
            os.environ.pop(name, None)

    def patch(self, patcher):
        patcher.start()
        self.addCleanup(patcher.stop)

    def env(self, **values):
        os.environ.update(values)


class LockdownTest(LockdownCase):
    """THEARTIST_PUBLIC_HOST: 403s for the DAW and warmup, no MCP mount; MIDI download and score reading stay."""

    def setUp(self):
        super().setUp()
        self.env(THEARTIST_PUBLIC_HOST="1")

    def test_daw_and_warmup_are_403_while_score_reading_and_midi_download_stay(self):
        res = CLIENT.get("/api/daw/targets")
        self.assertEqual(res.status_code, 403, res.text)
        self.assertIn("No DAW on this server", res.json()["detail"])
        self.assertEqual(CLIENT.post("/api/daw/open", json={"target": "logic", "spec": spec()}).status_code, 403)
        res = CLIENT.post("/api/models/warmup")
        self.assertEqual(res.status_code, 403, res.text)
        self.assertIn("warmup", res.json()["detail"].lower())
        reader = mock.patch.object(score_import_routes, "import_score",
                                   side_effect=ScoreInputError("reached the reader", 418))
        with reader as import_score:
            res = CLIENT.post(IMPORT, json=score_body())
            self.assertEqual((res.status_code, res.json()["detail"]), (418, "reached the reader"))
            self.assertEqual(import_score.call_count, 1)
        self.assertEqual(CLIENT.post("/api/export/midi", json=spec()).status_code, 200)

    def test_the_mcp_mount_is_skipped_when_the_app_is_built_for_a_public_host(self):
        locked = app.main.create_app()
        self.assertEqual([route.path for route in locked.routes if isinstance(route, Mount)], [])
        self.assertIn("/mcp", [route.path for route in app.main.app.routes if isinstance(route, Mount)])
        self.assertIn("/mcp", [route.path for route in app.main.create_app({}).routes if isinstance(route, Mount)])
        with TestClient(locked) as client:  # the lifespan runs without an MCP session manager
            self.assertEqual(client.get("/api/health").json(), {"status": "ok"})
            self.assertEqual(client.post("/mcp/", json={}).status_code, 404)
            self.assertEqual(client.get("/api/daw/targets").status_code, 403)

    def test_a_pdf_the_reader_cannot_read_is_a_422_on_a_public_host(self):
        # This synthetic PDF is no vector score: the lockdown does not stand in the way of the reader's own refusal.
        res = CLIENT.post(IMPORT, json=score_body())
        self.assertEqual(res.status_code, 422, res.text)
        self.assertIn("vector PDF", res.json()["detail"])


if __name__ == "__main__":
    unittest.main()
