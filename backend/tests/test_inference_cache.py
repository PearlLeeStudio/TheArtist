"""Cached relative attention must preserve logits, window cropping and sampling."""

import unittest
from unittest import mock

import torch

from app.services.inference import MusicTransformer


class InferenceCacheTests(unittest.TestCase):
    def setUp(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            self.model = MusicTransformer(
                vocab_size=24, d_model=16, n_heads=2, d_ff=32,
                n_layers=2, max_seq_len=16, dropout=0.1,
            ).eval()
        self.ids = torch.tensor([[1, 4, 7, 3, 6, 9, 11, 5]])

    @torch.no_grad()
    def test_prefill_and_incremental_chunks_match_full_relative_attention(self):
        expected = self.model(self.ids)
        cache = [[] for _ in self.model.layers]
        actual = []
        for start, end in [(0, 3), (3, 4), (4, 6), (6, 8)]:
            actual.append(self.model(self.ids[:, start:end], kv_cache=cache))
        torch.testing.assert_close(torch.cat(actual, dim=1), expected, rtol=1e-5, atol=1e-6)
        self.assertTrue(all(layer[0][0].shape[2] == 8 for layer in cache))

    def test_seeded_generation_matches_with_repetition_and_sliding_window(self):
        for prompt in [self.ids[:, :3], self.ids, self.ids.repeat(1, 3)]:
            for seed in range(4):
                kwargs = dict(
                    max_new_tokens=24, eos_id=-1, top_k=12, top_p=0.9,
                    repetition_penalty=1.3, no_repeat_ngram_size=3,
                    ignore_repeat_token_ids={3},
                )
                expected = self.model.generate(prompt, **kwargs, generator=torch.Generator().manual_seed(seed))
                actual = self.model.generate(prompt, **kwargs, use_cache=True,
                                             generator=torch.Generator().manual_seed(seed))
                self.assertTrue(torch.equal(actual, expected))

    def test_only_new_token_is_evaluated_until_context_window_slides(self):
        with mock.patch.object(self.model, 'forward', wraps=self.model.forward) as forward:
            self.model.generate(self.ids, max_new_tokens=12, eos_id=-1, use_cache=True)
        widths = [call.args[0].shape[1] for call in forward.call_args_list]
        self.assertEqual(widths, [8] + [1] * 8 + [16] * 3)

    def test_cache_is_not_retained_in_the_model_or_checkpoint(self):
        keys = set(self.model.state_dict())
        first = self.model.generate(self.ids, max_new_tokens=8, eos_id=-1, use_cache=True,
                                    generator=torch.Generator().manual_seed(9))
        self.model.generate(self.ids.flip(1), max_new_tokens=8, eos_id=-1, use_cache=True)
        repeated = self.model.generate(self.ids, max_new_tokens=8, eos_id=-1, use_cache=True,
                                       generator=torch.Generator().manual_seed(9))
        self.assertTrue(torch.equal(first, repeated))
        self.assertEqual(set(self.model.state_dict()), keys)

    def test_cache_rejects_training_and_prefix_tuning(self):
        with self.assertRaisesRegex(ValueError, 'inference'):
            self.model(self.ids, kv_cache=[[], []])
        with torch.no_grad(), self.assertRaisesRegex(ValueError, 'prefix'):
            self.model(self.ids, prefix_kvs=[], kv_cache=[[], []])
