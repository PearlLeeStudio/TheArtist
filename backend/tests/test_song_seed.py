"""Seeded song sampling exercises the real sampler without weights or network calls."""

from concurrent.futures import ThreadPoolExecutor
import random
from types import SimpleNamespace
import unittest
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np
import torch

from app.models.schemas import SongGenerateRequest
from app.routes import song
from app.services import inference


class TinyTokenizer:
    tokens = ['[PAD]', '[BOS]', '[EOS]', '[BAR]', 'Cmaj', 'Dm7', 'G7', 'Am', 'Fmaj']
    token2id = {token: index for index, token in enumerate(tokens)}
    pad_id, bos_id, eos_id, bar_id = range(4)

    def decode(self, ids):
        return [self.tokens[index] for index in ids]


TOKENIZER = TinyTokenizer()


def rng_states():
    return torch.get_rng_state().clone(), random.getstate(), np.random.get_state()


def assert_rng_unchanged(test: unittest.TestCase, before):
    after = rng_states()
    test.assertTrue(torch.equal(before[0], after[0]), 'global torch sampling state changed')
    test.assertEqual(before[1], after[1], 'global Python random state changed')
    test.assertEqual(before[2][0], after[2][0])
    np.testing.assert_array_equal(before[2][1], after[2][1])
    test.assertEqual(before[2][2:], after[2][2:], 'global NumPy random state changed')


@unittest.skipUnless(inference.TORCH_AVAILABLE, 'needs the local torch runtime')
class SeededSamplerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Real attention, normalization, token filtering and multinomial run
        # on a tiny CPU model. Zero weights make its distribution broad and
        # reproducible, without training or reading production checkpoints.
        with torch.random.fork_rng(devices=[]):
            model = inference.MusicTransformer(
                vocab_size=len(TOKENIZER.tokens), d_model=8, n_heads=2,
                d_ff=8, n_layers=1, max_seq_len=64, dropout=0.0,
            )
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
        cls.loaded = inference.LoadedModel(model=model, name='tiny-local', device=torch.device('cpu'))

    def sample(self, seed):
        return inference.generate_with_model(
            self.loaded, TOKENIZER, [TOKENIZER.bos_id], 8,
            temperature=1.0, top_k=0, top_p=1.0,
            repetition_penalty=1.0, no_repeat_ngram_size=0, seed=seed,
        )

    def test_real_model_sampling_repeats_without_mutating_global_rngs(self):
        before = rng_states()
        first = self.sample(17)
        second = self.sample(17)
        other = self.sample(18)
        self.assertTrue(first)
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)
        assert_rng_unchanged(self, before)

    def test_concurrent_requests_keep_independent_seed_streams(self):
        seeds = [17, 18, 17, 0]
        expected = [self.sample(seed) for seed in seeds]
        before = rng_states()
        with ThreadPoolExecutor(max_workers=4) as executor:
            actual = list(executor.map(self.sample, seeds))
        self.assertEqual(actual, expected)
        assert_rng_unchanged(self, before)

    def test_seed_none_keeps_the_existing_unseeded_torch_path(self):
        before = torch.get_rng_state().clone()
        self.addCleanup(torch.set_rng_state, before)
        self.sample(None)
        self.assertFalse(torch.equal(before, torch.get_rng_state()))

    def test_sampler_exception_cannot_leak_the_seed_into_other_requests(self):
        class FailingSampler:
            def generate(self, prompt, **kwargs):
                torch.multinomial(torch.ones(3), 1, generator=kwargs['generator'])
                raise RuntimeError('sampling failed')

        before = rng_states()
        loaded = inference.LoadedModel(model=FailingSampler(), name='failing', device=torch.device('cpu'))
        with self.assertRaisesRegex(RuntimeError, 'sampling failed'):
            inference.generate_with_model(loaded, TOKENIZER, [TOKENIZER.bos_id], 1, seed=0)
        assert_rng_unchanged(self, before)

    def test_composite_route_reaches_the_actual_seeded_sampler_and_midi_renderer(self):
        registry = SimpleNamespace(tokenizer=TOKENIZER, load_model=lambda _name: self.loaded)
        with mock.patch.object(song.ModelRegistry, 'get', return_value=registry), \
                mock.patch.object(song, 'build_prompt', return_value=[TOKENIZER.bos_id]), \
                mock.patch.object(song, 'generate_with_model', wraps=inference.generate_with_model) as generate:
            first = song.generate_song(SongGenerateRequest(genre='pop', length_bars=4, seed=17))
            second = song.generate_song(SongGenerateRequest(genre='pop', length_bars=4, seed=17))
        self.assertEqual(generate.call_args.kwargs['seed'], 17)
        self.assertEqual(first.model_used, inference.dispatch_model_for_genre('pop'))
        self.assertEqual(first.seed, 17)
        self.assertEqual(first.bars, second.bars)
        self.assertEqual(first.tracks, second.tracks)
        self.assertEqual(first.midi_b64, second.midi_b64)
        self.assertEqual(len(first.bars), 4)


class SongSeedContractTests(unittest.TestCase):
    def test_rule_fallback_is_reproducible_and_never_reseeds_python(self):
        before = rng_states()
        with mock.patch.object(song, 'TORCH_AVAILABLE', False):
            def build(seed):
                return song.generate_song(SongGenerateRequest(genre='jazz', length_bars=8, seed=seed))

            first = build(0)
            with ThreadPoolExecutor(max_workers=3) as executor:
                repeated, other, third = list(executor.map(build, [0, 1, 0]))
        self.assertEqual(first.model_used, 'rules')
        self.assertEqual(first.seed, 0)
        self.assertEqual(first.bars, repeated.bars)
        self.assertEqual(first.bars, third.bars)
        self.assertEqual(first.midi_b64, repeated.midi_b64)
        self.assertNotEqual(first.bars, other.bars)
        assert_rng_unchanged(self, before)

    def test_http_rejects_ignored_candidate_counts_and_bad_seeds_before_generation(self):
        app = FastAPI()
        app.include_router(song.router, prefix='/api')
        with TestClient(app) as client, mock.patch.object(song, '_generate_bars') as generate:
            for payload in (
                {'n_candidates': 2}, {'n_candidates': 3}, {'n_candidates': True},
                {'seed': -1}, {'seed': 0x100000000}, {'seed': True}, {'seed': '12'}, {'seed': 1.5},
            ):
                with self.subTest(payload=payload):
                    response = client.post('/api/generate/song', json={'genre': 'pop', **payload})
                    self.assertEqual(response.status_code, 422)
            generate.assert_not_called()

    def test_omitted_seed_and_default_single_song_remain_valid(self):
        self.assertIsNone(SongGenerateRequest(genre='pop').seed)
        self.assertEqual(SongGenerateRequest(genre='pop').n_candidates, 1)
        with mock.patch.object(song, '_generate_bars', return_value=([{'chords': ['Cmaj', 'G7']}], 'test')) as generate:
            result = song.generate_song(SongGenerateRequest(genre='pop', length_bars=1))
        self.assertIsNone(generate.call_args.kwargs['seed'])
        self.assertIsNone(result.seed)
        self.assertTrue(result.midi_b64)


if __name__ == '__main__':
    unittest.main()
