"""THEARTIST_WARMUP_KEYS narrows the preload at startup and on POST /api/models/warmup; unknown keys are reported."""

import asyncio
import os
import unittest
from unittest import mock

from fastapi.testclient import TestClient

import app.main
from app.routes import models as models_routes
from app.services import inference
from app.services.inference import CHECKPOINTS, ModelRegistry, warmup_keys

CLIENT = TestClient(app.main.app)
VAR = inference.WARMUP_KEYS_VAR


class FakeRegistry:
    """Records what warmup is asked for; every asked key counts as loaded."""

    def __init__(self):
        self.calls = []

    def warmup(self, keys=None):
        self.calls.append(None if keys is None else list(keys))
        return {name: True for name in (CHECKPOINTS if keys is None else keys)}


class WarmupKeysTest(unittest.TestCase):
    def setUp(self):
        self.patch(mock.patch.dict(os.environ))
        for name in (VAR, app.main.WARMUP_VAR):
            os.environ.pop(name, None)
        self.registry = FakeRegistry()
        self.patch(mock.patch.object(ModelRegistry, "get", return_value=self.registry))
        self.patch(mock.patch.object(models_routes, "TORCH_AVAILABLE", True))
        self.patch(mock.patch.object(inference, "TORCH_AVAILABLE", True))

    def patch(self, patcher):
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_unset_or_blank_means_every_checkpoint(self):
        self.assertEqual(warmup_keys({}), (None, []))
        self.assertEqual(warmup_keys({VAR: " , "}), (None, []))

    def test_keys_are_filtered_in_order_and_unknown_names_reported(self):
        self.assertEqual(warmup_keys({VAR: " ft_f4, ft_f1 ,nope,ft_f1"}), (["ft_f4", "ft_f1"], ["nope"]))
        self.assertEqual(warmup_keys({VAR: "nope"}), ([], ["nope"]))

    def test_registry_warmup_loads_only_the_given_keys_and_an_unknown_one_is_false(self):
        registry = ModelRegistry()
        loads = lambda name: object() if name in CHECKPOINTS else None  # noqa: E731
        with mock.patch.object(registry, "load_model", side_effect=loads) as load:
            self.assertEqual(registry.warmup(["ft_f1", "nope"]), {"ft_f1": True, "nope": False})
            self.assertEqual([call.args[0] for call in load.call_args_list], ["ft_f1", "nope"])
            load.reset_mock()
            self.assertEqual(list(registry.warmup()), list(CHECKPOINTS))

    def test_route_honors_the_keys_and_reports_unknown_ones(self):
        os.environ[VAR] = "ft_f1,nope"
        res = CLIENT.post("/api/models/warmup")
        self.assertEqual(res.status_code, 200, res.text)
        self.assertEqual(res.json(), {"torch_available": True, "loaded": {"ft_f1": True}, "unknown": ["nope"]})
        self.assertEqual(self.registry.calls, [["ft_f1"]])

    def test_route_default_warms_every_checkpoint(self):
        res = CLIENT.post("/api/models/warmup")
        self.assertEqual(res.status_code, 200, res.text)
        self.assertEqual((list(res.json()["loaded"]), res.json()["unknown"]), (list(CHECKPOINTS), []))
        self.assertEqual(self.registry.calls, [None])

    def test_startup_preloads_with_the_switch_or_the_keys_only(self):
        asyncio.run(app.main.startup_warmup())
        self.assertEqual(self.registry.calls, [])
        os.environ[app.main.WARMUP_VAR] = "1"
        asyncio.run(app.main.startup_warmup())
        self.assertEqual(self.registry.calls, [None])  # every checkpoint, as before
        os.environ[VAR] = "ft_f4"
        asyncio.run(app.main.startup_warmup())
        self.assertEqual(self.registry.calls[-1], ["ft_f4"])
        del os.environ[app.main.WARMUP_VAR]  # the list alone preloads too
        os.environ[VAR] = "ft_f1,nope"
        with self.assertLogs("app.main", level="WARNING") as logs:
            asyncio.run(app.main.startup_warmup())
        self.assertEqual(self.registry.calls[-1], ["ft_f1"])
        self.assertIn("nope", logs.output[0])
        calls = len(self.registry.calls)
        os.environ[app.main.WARMUP_VAR] = "0"  # an explicit off still wins over a key list
        asyncio.run(app.main.startup_warmup())
        self.assertEqual(len(self.registry.calls), calls)

    def test_startup_without_torch_loads_nothing(self):
        os.environ[app.main.WARMUP_VAR] = "1"
        with mock.patch.object(inference, "TORCH_AVAILABLE", False):
            asyncio.run(app.main.startup_warmup())
        self.assertEqual(self.registry.calls, [])


if __name__ == "__main__":
    unittest.main()
