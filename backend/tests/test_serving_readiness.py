import os
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.services import inference
from app.services.serving_readiness import require_full_catalog


def asset(present=True):
    path = MagicMock()
    path.__truediv__.return_value = path
    path.is_file.return_value = present
    path.stat.return_value = SimpleNamespace(st_size=100)
    return path


class ServingReadinessTest(unittest.TestCase):
    def setUp(self):
        self.catalog = {"ft_f1": asset(), "ft_f1_lora_country": asset()}
        self.registry = MagicMock()
        self.registry.load_model.side_effect = lambda key: SimpleNamespace(name=key)
        for replacement in (
            patch.object(inference, "CHECKPOINTS", self.catalog),
            patch.object(inference, "TORCH_AVAILABLE", True),
            patch.object(inference, "PEFT_AVAILABLE", True),
            patch.object(inference.ModelRegistry, "get", return_value=self.registry),
        ):
            replacement.start()
            self.addCleanup(replacement.stop)

    def test_missing_adapter_fails_before_any_model_loading(self):
        self.catalog["ft_f1_lora_country"].is_file.return_value = False
        with self.assertRaisesRegex(RuntimeError, "Missing model assets: ft_f1_lora_country"):
            require_full_catalog()
        self.registry.load_model.assert_not_called()

    def test_f1_substitution_is_not_a_successful_lora_start(self):
        self.registry.load_model.side_effect = lambda key: SimpleNamespace(name="ft_f1")
        with self.assertRaisesRegex(RuntimeError, "did not load as itself"):
            require_full_catalog()

    def test_complete_catalog_loads_every_choice(self):
        self.assertEqual(require_full_catalog(), list(self.catalog))
        self.assertEqual([call.args[0] for call in self.registry.load_model.call_args_list], list(self.catalog))


@unittest.skipUnless(inference.TORCH_AVAILABLE, "Torch unavailable")
class HostedInferenceTest(unittest.TestCase):
    def test_public_neural_passes_are_serialized_and_released_after_error(self):
        tokenizer = inference.ChordTokenizer(include_extra_genres=True)
        entered = threading.Event()
        release = threading.Event()
        second_entered = threading.Event()
        second_requested = threading.Event()
        calls = []

        def generate(*args, **kwargs):
            calls.append(len(calls))
            if len(calls) == 1:
                entered.set()
                if not release.wait(3):
                    raise AssertionError("test failed to release first inference")
                raise RuntimeError("deliberate inference failure")
            second_entered.set()
            return inference.torch.tensor([[tokenizer.bar_id, tokenizer.bar_id,
                                            tokenizer.encode_chord("C"), tokenizer.eos_id]])

        loaded = inference.LoadedModel(SimpleNamespace(generate=generate), "test", inference.torch.device("cpu"))
        def run(started=None):
            if started is not None:
                started.set()
            return inference.generate_with_model(loaded, tokenizer, [tokenizer.bar_id], 1)

        with patch.dict(os.environ, {"THEARTIST_PUBLIC_HOST": "1"}), ThreadPoolExecutor(2) as pool:
            first = pool.submit(run)
            try:
                self.assertTrue(entered.wait(3))
                second = pool.submit(run, second_requested)
                self.assertTrue(second_requested.wait(3))
                self.assertFalse(second_entered.wait(0.1))
            finally:
                release.set()
            with self.assertRaisesRegex(RuntimeError, "deliberate inference failure"):
                first.result(timeout=3)
            self.assertEqual(second.result(timeout=3), [["Cmaj"]])
            self.assertTrue(second_entered.is_set())


if __name__ == "__main__":
    unittest.main()
