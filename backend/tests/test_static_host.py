import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from app.main import create_app


class StaticHostTest(unittest.TestCase):
    def test_api_does_not_expose_assets_without_explicit_switch(self):
        client = TestClient(create_app({"THEARTIST_PUBLIC_HOST": "1"}))
        self.assertEqual(client.get("/api/health").json(), {"status": "ok"})
        self.assertEqual(client.get("/").status_code, 404)

    def test_missing_build_fails_loudly(self):
        with patch("app.main.FRONTEND_DIST", Path(__file__).resolve().parent / "missing_frontend_fixture"):
            with self.assertRaisesRegex(RuntimeError, "Build the frontend"):
                create_app({"THEARTIST_SERVE_FRONTEND": "1", "THEARTIST_PUBLIC_HOST": "1"})

    def test_asset_mount_is_last_and_restricted_to_dist(self):
        with patch("pathlib.Path.is_file", return_value=True), patch("app.main.StaticFiles") as static:
            app = create_app({"THEARTIST_SERVE_FRONTEND": "1", "THEARTIST_PUBLIC_HOST": "1"})
        self.assertEqual(app.routes[-1].name, "frontend")
        self.assertEqual(Path(static.call_args.kwargs["directory"]).name, "dist")
        self.assertTrue(any(getattr(route, "path", "") == "/api/health" for route in app.routes[:-1]))


if __name__ == "__main__":
    unittest.main()
