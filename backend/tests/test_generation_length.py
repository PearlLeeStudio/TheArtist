"""Generation length: each sampling pass writes the bars its request asks for.

A pass used to stop at EOS or at its token budget long before a long selection
was full, and the service filled the rest by repeating the pass's own bars.
generate_with_model now holds every pass to the selection's bar count
(MusicTransformer.generate(n_bars=...)); the padding is left for a pass that
still runs out of budget. The scripted-model and service tests need no
weights. The checkpoint tests need the F1 (pop) and F4 (jazz) checkpoints in
ai/checkpoints, skip without them, and take about 30 s on CPU.
"""

import os
import unittest
from unittest import mock

import torch

from app.models.schemas import ContextMeasure
from app.services import inference

TOKENIZER = inference.ModelRegistry.get().tokenizer if inference.TORCH_AVAILABLE else None
CHECKPOINTS_READY = inference.TORCH_AVAILABLE and all(
    inference.CHECKPOINTS[inference.GENRE_MODEL_DISPATCH[genre]].exists() for genre in ("pop", "jazz")
)

# Toy vocabulary for the scripted model.
PAD, BOS, EOS, BAR, KEY, CHORD = 0, 1, 2, 3, 4, 5
TOY_CHORDS = {5, 6, 7}
TOY_VOCAB = 8

# Context bars for requests that follow chords: I vi | IV V in C, two chords a bar.
LOOP = (["Cmaj", "Am"], ["Fmaj", "Gmaj"])

if inference.TORCH_AVAILABLE:
    class ScriptedTransformer(inference.MusicTransformer):
        """Untrained stand-in with scripted scores for the next token: [EOS], then
        [PAD], always rank first; then [BAR] when the window holds none, a chord
        until the bar holds two, else [BAR]. ``empty_bars`` makes a freshly opened
        bar prefer another [BAR] to its first chord. Decoded greedily (top_k=1),
        every run is deterministic. Records the width of every window it sees."""

        def __init__(self, max_seq_len: int = 64, empty_bars: bool = False) -> None:
            super().__init__(vocab_size=TOY_VOCAB, d_model=8, n_heads=2, d_ff=8, n_layers=1,
                             max_seq_len=max_seq_len, dropout=0.0)
            self.empty_bars = empty_bars
            self.widths: list[int] = []

        def forward(self, input_ids, prefix_kvs=None):
            self.widths.append(input_ids.shape[1])
            seq = input_ids[0].tolist()
            logits = torch.zeros(1, len(seq), TOY_VOCAB)
            logits[0, -1, EOS] = 4.0
            logits[0, -1, PAD] = 3.0
            if BAR not in seq:
                logits[0, -1, BAR] = 2.0
            elif self.empty_bars and seq[-1] == BAR:
                logits[0, -1, BAR] = 2.0
                logits[0, -1, CHORD] = 1.0
            else:
                since_bar = seq[::-1].index(BAR)
                chords = sum(t in TOY_CHORDS for t in seq[len(seq) - since_bar:])
                logits[0, -1, CHORD if chords < 2 else BAR] = 2.0
            return logits


def request(bars: int, context_bars: int = 0) -> tuple[list[ContextMeasure], list[int]]:
    """The score editor's request: filled context bars, then the empty selected bars."""
    context = [ContextMeasure(measure=m, chords=list(LOOP[m % 2])) for m in range(context_bars)]
    selected = list(range(context_bars, context_bars + bars))
    context += [ContextMeasure(measure=m, chords=[None, None]) for m in selected]
    return context, selected


@unittest.skipUnless(inference.TORCH_AVAILABLE, "needs torch and ai/training")
class GenerateBarsTest(unittest.TestCase):
    """MusicTransformer.generate with and without n_bars, on the scripted model."""

    def generate(self, model, prompt: list[int], **kwargs) -> list[int]:
        out = model.generate(torch.tensor([prompt]), top_k=1, eos_id=EOS, **kwargs)
        return out[0, len(prompt):].tolist()

    def held(self, model, prompt: list[int], n_bars: int, max_new_tokens: int = 64) -> list[int]:
        return self.generate(model, prompt, max_new_tokens=max_new_tokens, n_bars=n_bars,
                             bar_id=BAR, chord_ids=TOY_CHORDS, end_ids={PAD})

    def test_defaults_still_stop_on_eos(self):
        self.assertEqual(self.generate(ScriptedTransformer(), [BOS, KEY], max_new_tokens=16), [EOS])

    def test_n_bars_writes_exactly_that_many_bars(self):
        # The first chord fills the prompt's open bar and does not count; the
        # [BAR] after the second bar closes it and ends the pass.
        got = self.held(ScriptedTransformer(), [BOS, KEY, BAR, CHORD], n_bars=2)
        self.assertEqual(got, [CHORD, BAR, CHORD, CHORD, BAR, CHORD, CHORD, BAR])

    def test_a_bar_gets_a_chord_before_the_next_bar(self):
        got = self.held(ScriptedTransformer(empty_bars=True), [BOS, KEY], n_bars=3)
        self.assertEqual(got, [BAR, CHORD, CHORD] * 3 + [BAR])

    def test_the_token_budget_still_caps_a_pass(self):
        got = self.held(ScriptedTransformer(), [BOS, KEY], n_bars=10, max_new_tokens=7)
        self.assertEqual(got, [BAR, CHORD, CHORD, BAR, CHORD, CHORD, BAR])

    def test_the_window_slides_while_bars_are_held(self):
        model = ScriptedTransformer(max_seq_len=6)
        got = self.held(model, [BOS, KEY], n_bars=20)
        self.assertEqual(got, [BAR, CHORD, CHORD] * 20 + [BAR])
        self.assertEqual(max(model.widths), 6)

    def test_n_bars_needs_chord_ids(self):
        with self.assertRaises(ValueError):
            ScriptedTransformer().generate(torch.tensor([[BOS]]), n_bars=1)


class CannedModel:
    """Returns the prompt plus fixed tokens, and keeps the keyword arguments it got."""

    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.kwargs: dict = {}

    def generate(self, prompt, **kwargs):
        self.kwargs = kwargs
        ids = torch.tensor([[TOKENIZER.token2id[t] for t in self.tokens]], dtype=prompt.dtype)
        return torch.cat([prompt, ids], dim=1)


def no_phase(*_args, **_kwargs) -> None:
    pass


@unittest.skipUnless(inference.TORCH_AVAILABLE, "needs torch and ai/training")
class ServicePassTest(unittest.TestCase):
    def test_a_pass_is_held_to_the_selection(self):
        model = CannedModel("Fmaj [BAR] Cmaj Am [BAR] Fmaj Gmaj [BAR] Em [BAR]".split())
        loaded = inference.LoadedModel(model=model, name="canned", device=torch.device("cpu"))
        bars = inference.generate_with_model(loaded, TOKENIZER, [TOKENIZER.bos_id], 2)
        # The leading Fmaj continues the prompt's last bar; the trim keeps 2 bars.
        self.assertEqual(bars, [["Cmaj", "Am"], ["Fmaj", "Gmaj"]])
        self.assertEqual(model.kwargs["n_bars"], 2)
        self.assertEqual(model.kwargs["bar_id"], TOKENIZER.bar_id)
        self.assertEqual(model.kwargs["end_ids"], {TOKENIZER.pad_id})
        chord_ids = model.kwargs["chord_ids"]
        self.assertIn(TOKENIZER.encode_chord("Cmaj"), chord_ids)
        self.assertFalse({TOKENIZER.bar_id, TOKENIZER.eos_id, TOKENIZER.pad_id, TOKENIZER.bos_id} & chord_ids)
        self.assertEqual(len(chord_ids), 312)

    def test_the_budget_has_slack_beyond_three_tokens_a_bar(self):
        for bars, budget in ((2, 14), (8, 32), (32, 112), (64, 224)):
            with self.subTest(bars=bars):
                model = CannedModel(["[BAR]"])
                loaded = inference.LoadedModel(model=model, name="canned", device=torch.device("cpu"))
                inference.generate_with_model(loaded, TOKENIZER, [TOKENIZER.bos_id], bars)
                self.assertEqual(model.kwargs["max_new_tokens"], budget)

    def test_a_short_pass_is_logged_and_padded_with_its_own_bars(self):
        written = [["Cmaj", "Am"], ["Fmaj", "Gmaj"]]
        loaded = inference.LoadedModel(model=None, name="canned", device=torch.device("cpu"))
        with mock.patch.object(inference, "generate_with_model", return_value=written), \
                self.assertLogs(inference.log, level="WARNING") as logs:
            suggestions = inference._compose_candidates(loaded, "Canned", [1], 5, list(range(5)), no_phase)
        self.assertEqual(len(logs.records), 3)
        self.assertIn("Model wrote 2 of 5 bars", logs.output[0])
        self.assertIn("temp=0.7", logs.output[0])
        self.assertEqual(suggestions[0].chords["4"], ["Cmaj", "Am"])

    def test_a_full_pass_is_not_padded(self):
        written = [["Cmaj", "Am"], ["Fmaj", "Gmaj"], ["Dm7", "G7"]]
        loaded = inference.LoadedModel(model=None, name="canned", device=torch.device("cpu"))
        with mock.patch.object(inference, "generate_with_model", return_value=written), \
                self.assertNoLogs(inference.log, level="WARNING"):
            suggestions = inference._compose_candidates(loaded, "Canned", [1], 3, [4, 5, 6], no_phase)
        self.assertEqual(suggestions[0].chords, {"4": ["Cmaj", "Am"], "5": ["Fmaj", "Gmaj"], "6": ["Dm7", "G7"]})


@unittest.skipUnless(CHECKPOINTS_READY, "needs torch and the F1 / F4 checkpoints in ai/checkpoints")
class CheckpointLengthTest(unittest.TestCase):
    """Real weights, fixed seeds, C major, the service's three temperatures in turn."""

    def passes(self, genre: str, bars: int, context_bars: int = 0, runs: int = 3) -> list[int]:
        """Bars each sampling pass writes, before any padding."""
        loaded = inference.ModelRegistry.get().load_model(inference.dispatch_model_for_genre(genre))
        prompt = inference.build_prompt(TOKENIZER, "C major", genre, (4, 4), *request(bars, context_bars))
        written = []
        for seed in range(runs):
            torch.manual_seed(seed)
            temperature = inference._TEMPS[seed % len(inference._TEMPS)][0]
            bars_written = inference.generate_with_model(loaded, TOKENIZER, prompt, bars, temperature=temperature)
            written.append(len(bars_written))
        return written

    def test_fresh_32_bar_requests_write_every_bar(self):
        for genre in ("pop", "jazz"):
            with self.subTest(genre=genre):
                written = self.passes(genre, 32, runs=10)
                self.assertGreaterEqual(written.count(32), 9, written)

    def test_8_bar_requests_still_write_8_bars(self):
        for genre in ("pop", "jazz"):
            with self.subTest(genre=genre):
                self.assertEqual(self.passes(genre, 8), [8, 8, 8])

    def run_request(self, genre: str, context: list[ContextMeasure], selected: list[int]):
        """One C-major request through generate_ai_result (seed 0, RAG off): the
        bars each pass wrote, the widest window the model saw, its max_seq_len
        and the result."""
        loaded = inference.ModelRegistry.get().load_model(inference.dispatch_model_for_genre(genre))
        real_generate = inference.generate_with_model
        written: list[int] = []
        widths: list[int] = []

        def counted(*args, **kwargs):
            bars = real_generate(*args, **kwargs)
            written.append(len(bars))
            return bars

        hook = loaded.model.register_forward_pre_hook(lambda _m, args: widths.append(args[0].shape[1]))
        torch.manual_seed(0)
        try:
            with mock.patch.dict(os.environ, {"THEARTIST_DISABLE_RAG": "1"}), \
                    mock.patch.object(inference, "generate_with_model", counted):
                result = inference.generate_ai_result("C major", genre, context, selected)
        finally:
            hook.remove()
        return written, max(widths), loaded.model.max_seq_len, result

    def test_64_bars_after_a_16_bar_context(self):
        # Three full passes (no short-pass warning), every measure filled, and no
        # window wider than the model's max_seq_len.
        context, selected = request(64, 16)
        for genre in ("pop", "jazz"):
            with self.subTest(genre=genre), self.assertNoLogs(inference.log, level="WARNING"):
                written, widest, window, result = self.run_request(genre, context, selected)
                self.assertEqual(written, [64, 64, 64])
                self.assertLessEqual(widest, window)
                for s in result.suggestions:
                    self.assertEqual(sorted(s.chords, key=int), [str(m) for m in selected])


if __name__ == "__main__":
    unittest.main()
