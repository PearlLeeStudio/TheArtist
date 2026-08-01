"""Written chords after a passage shape what generation offers, not just its order.

The causal model writes from the bars before a passage. When the score holds
chords after it, each temperature draws several drafts and keeps the one under
which those written chords are likeliest (chord_context.continuation_scores),
judged as the draft will be offered: key-anchored when nothing precedes, with the
section's song-form rules applied.
"""
import os
import unittest
from types import SimpleNamespace
from unittest import mock

from app.models.schemas import ContextMeasure, Suggestion
from app.services import chord_context, inference
from app.services.chord_context import context_scores, continuation_scores, sequence_tokens


def no_phase(*_args, **_kwargs) -> None:
    pass


class ContinuationScoreTest(unittest.TestCase):
    def test_only_the_written_chords_after_the_passage_are_scored(self):
        tokenizer = SimpleNamespace(bar_id=3, encode_chord=lambda symbol: {"C": 10, "Dm": 11, "G7": 12}[symbol])
        context = [ContextMeasure(measure=0, chords=["C"]), ContextMeasure(measure=1, chords=[None, None]),
                   ContextMeasure(measure=2, chords=["G7"])]
        candidate = Suggestion(label="draft", chords={"1": ["Dm", "G7"]})
        ids, every = sequence_tokens(tokenizer, [1, 4], context, candidate, [1])
        _, known = sequence_tokens(tokenizer, [1, 4], context, candidate, [1], known_only=True)
        self.assertEqual(ids, [1, 4, 3, 10, 3, 11, 12, 3, 12])
        self.assertEqual(every, [5, 6, 8])
        self.assertEqual(known, [8], "the passage's own chords are not scored")

    def test_the_draft_that_makes_the_written_future_likeliest_wins_whatever_its_own_likelihood(self):
        torch = inference.torch
        tokenizer = SimpleNamespace(bar_id=3, encode_chord=lambda symbol: {"C": 4, "Dm": 5, "G7": 6, "Am": 7}[symbol])

        class Model:
            max_seq_len = 32

            def __call__(self, inputs):
                logits = torch.zeros(inputs.shape[0], inputs.shape[1], 8)
                for row, ids in enumerate(inputs):
                    # Am is far likelier than Dm as the passage itself...
                    logits[row, :, 7] = 6
                    # ...but the written G7 after it is far likelier after Dm.
                    logits[row, -1, 6] = 8 if 5 in ids else -8
                return logits

        loaded = SimpleNamespace(model=Model(), device="cpu")
        drafts = [Suggestion(label="Am", chords={"1": ["Am"]}), Suggestion(label="Dm", chords={"1": ["Dm"]})]
        context = [ContextMeasure(measure=0, chords=["C"]), ContextMeasure(measure=2, chords=["G7"])]
        future = continuation_scores(loaded, tokenizer, drafts, [1], context, [1])
        self.assertGreater(future[1], future[0])
        # The whole-window score (passage and future) is a different measure.
        self.assertNotEqual(context_scores(loaded, tokenizer, drafts, [1], context, [1]), future)

    def test_no_written_chord_after_the_passage_is_an_error_not_a_score(self):
        tokenizer = SimpleNamespace(bar_id=3, encode_chord=lambda symbol: {"C": 4, "Dm": 5}[symbol])
        loaded = SimpleNamespace(model=SimpleNamespace(max_seq_len=32), device="cpu")
        drafts = [Suggestion(label="Dm", chords={"1": ["Dm"]})]
        with self.assertRaisesRegex(ValueError, "No chord tokens"):
            continuation_scores(loaded, tokenizer, drafts, [1], [ContextMeasure(measure=0, chords=["C"])], [1])


class DraftSettingsTest(unittest.TestCase):
    def test_drafts_per_temperature(self):
        self.assertEqual(inference.context_draws({}), 4)
        self.assertEqual(inference.context_draws({"THEARTIST_CONTEXT_DRAWS": "2"}), 2)
        self.assertEqual(inference.context_draws({"THEARTIST_CONTEXT_DRAWS": "0"}), 1)
        self.assertEqual(inference.context_draws({"THEARTIST_CONTEXT_DRAWS": "99"}), 8)
        with self.assertLogs(inference.log, level="WARNING"):
            self.assertEqual(inference.context_draws({"THEARTIST_CONTEXT_DRAWS": "many"}), 4)

    def test_written_chords_after_the_passage_start(self):
        def bars(**chords):
            return [ContextMeasure(measure=int(m[1:]), chords=c) for m, c in chords.items()]
        self.assertTrue(inference._has_chords_after(bars(m0=["C", None], m3=["G7", None]), [1, 2]))
        # A written bar inside a split selection counts; empty and blank bars do not
        self.assertTrue(inference._has_chords_after(bars(m2=["F", None]), [1, 3]))
        self.assertFalse(inference._has_chords_after(bars(m0=["C", None], m3=[None, " "]), [1, 2]))
        self.assertFalse(inference._has_chords_after(bars(m3=["G7", None]), []))


@unittest.skipUnless(inference.TORCH_AVAILABLE, "needs torch and ai/training")
class ComposeDraftsTest(unittest.TestCase):
    def loaded(self):
        return inference.LoadedModel(model=None, name="canned", device=inference.torch.device("cpu"))

    def test_one_draft_keeps_the_seeds_a_single_pass_had(self):
        with mock.patch.object(inference, "generate_with_model", return_value=[["Cmaj", "Am"]]) as generate:
            inference._compose_candidates(self.loaded(), "Canned", [1], 1, [4], no_phase, seed=10)
        self.assertEqual([call.kwargs["seed"] for call in generate.call_args_list], [10, 11, 12])

    def test_each_temperature_keeps_the_draft_pick_chooses(self):
        written = iter([[[f"draft{n}"]] for n in range(9)])
        seen: list[list[str]] = []

        def pick(drafts):
            seen.append([d["4"][0] for d in drafts])
            return 2

        with mock.patch.object(inference, "generate_with_model", side_effect=lambda *a, **k: next(written)) as generate:
            suggestions = inference._compose_candidates(
                self.loaded(), "Canned", [1], 1, [4], no_phase, seed=10, draws=3, pick=pick)
        # Draft d of temperature step s has seed 10 + (s - 1) + 3d: the first drafts keep 10, 11, 12
        self.assertEqual([call.kwargs["seed"] for call in generate.call_args_list], [10, 13, 16, 11, 14, 17, 12, 15, 18])
        self.assertEqual([call.kwargs["temperature"] for call in generate.call_args_list][::3], [0.7, 0.9, 1.1])
        self.assertEqual(seen, [["draft0", "draft1", "draft2"], ["draft3", "draft4", "draft5"], ["draft6", "draft7", "draft8"]])
        self.assertEqual([s.chords["4"][0] for s in suggestions], ["draft2", "draft5", "draft8"])
        self.assertEqual([s.label for s in suggestions],
                         ["Option A — Canned (conservative)", "Option B — Canned (balanced)", "Option C — Canned (creative)"])

    def test_a_pick_that_cannot_score_keeps_the_first_draft_and_says_so(self):
        def pick(_drafts):
            raise ValueError("Whole-score chord comparison exceeds the 256-token model context.")

        phases: list[dict] = []
        with mock.patch.object(inference, "generate_with_model", side_effect=[[["Cmaj"]], [["Dm"]]] * 3), \
                self.assertLogs(inference.log, level="WARNING") as logs:
            suggestions = inference._compose_candidates(self.loaded(), "Canned", [1], 1, [4],
                                                        lambda _stage, payload: phases.append(payload), draws=2, pick=pick)
        # A one-chord bar fills both slots, as every pass does
        self.assertEqual([s.chords["4"] for s in suggestions], [["Cmaj", "Cmaj"]] * 3)
        self.assertIn("Kept the first draft", logs.output[0])
        # The progress line no longer claims a choice: each temperature reports the first draft was kept
        corrections = [p for p in phases if "kept the first draft" in p["label"]]
        self.assertEqual([(p["step"], p["total"], p["creativity"]) for p in corrections],
                         [(1, 3, "conservative"), (2, 3, "balanced"), (3, 3, "creative")])


@unittest.skipUnless(inference.TORCH_AVAILABLE, "needs torch and ai/training")
class DraftCountTest(unittest.TestCase):
    """_drafts_for: the setting, the bar budget, and scores that cannot be judged."""

    def count(self, context, selected, draws="4"):
        tokenizer = inference.ModelRegistry.get().tokenizer
        loaded = SimpleNamespace(model=SimpleNamespace(max_seq_len=256), device="cpu")
        header = inference.build_prompt(tokenizer, "C major", "pop", (4, 4), [], [])
        with mock.patch.dict(os.environ, {"THEARTIST_CONTEXT_DRAWS": draws}):
            return inference._drafts_for(loaded, tokenizer, header, context, selected)

    @staticmethod
    def score(selected: list[int], after: list[str]) -> list[ContextMeasure]:
        """One written bar before the passage, the passage, then the written bars after it."""
        last = max(selected)
        bars = [ContextMeasure(measure=0, chords=["Cmaj", "Cmaj"])]
        bars += [ContextMeasure(measure=m, chords=[None, None]) for m in selected]
        bars += [ContextMeasure(measure=last + 1 + i, chords=[c, c]) for i, c in enumerate(after)]
        return bars

    def test_drafts_times_bars_stay_within_one_longest_pass(self):
        for bars, expected in ((8, 4), (16, 4), (17, 3), (32, 2), (33, 1), (64, 1)):
            with self.subTest(bars=bars):
                selected = list(range(1, bars + 1))
                self.assertEqual(self.count(self.score(selected, ["Fmaj"]), selected), expected)
        self.assertEqual(self.count(self.score([1, 2], ["Fmaj"]), [1, 2], draws="1"), 1)

    def test_a_score_that_cannot_be_judged_gets_one_draft(self):
        # A written chord outside the vocabulary anywhere in the score
        odd = self.score([1, 2], ["Fmaj"]) + [ContextMeasure(measure=9, chords=["", None])]
        with self.assertLogs(inference.log, level="WARNING") as logs:
            self.assertEqual(self.count(odd, [1, 2]), 1)
        self.assertIn("One draft per temperature", logs.output[0])
        # The only written bar after lies beyond what the 256-token window can hold next to the passage:
        # 291 empty bars separate it from an 8-bar selection
        selected = list(range(1, 9))
        far = ([ContextMeasure(measure=0, chords=["Cmaj", "Cmaj"])]
               + [ContextMeasure(measure=m, chords=[None, None]) for m in range(1, 300)]
               + [ContextMeasure(measure=300, chords=["Fmaj", "Fmaj"])])
        with self.assertLogs(inference.log, level="WARNING") as logs:
            self.assertEqual(self.count(far, selected), 1)
        self.assertIn("No chord tokens", logs.output[0])


@unittest.skipUnless(inference.TORCH_AVAILABLE, "needs torch and ai/training")
class GenerateWithWrittenFutureTest(unittest.TestCase):
    """generate_ai_result with the model, rerank, explanations and whole-score order stubbed."""

    def run_request(self, context, selected, section_type=None, draws="2"):
        loaded = inference.LoadedModel(model=None, name="ft_f1", device=inference.torch.device("cpu"))
        drafts = iter([[["Dm", "Fmaj"], ["Em", "Am"]], [["Fmaj", "Fmaj"], ["Am", "Dm"]]] * 3)
        judged: list[list[dict]] = []

        def scores(_loaded, _tokenizer, trial, _header, _context, _selected):
            judged.append([dict(t.chords) for t in trial])
            return [0.0, 1.0]  # the second draft leads best into the written bars

        with mock.patch.dict(os.environ, {"THEARTIST_CONTEXT_DRAWS": draws}), \
                mock.patch.object(inference, "_resolve_model", return_value=(loaded, "F1")), \
                mock.patch.object(inference, "generate_with_model", side_effect=lambda *a, **k: next(drafts)) as generate, \
                mock.patch.object(inference, "_score_and_rerank", side_effect=lambda s, *_a, **_k: s), \
                mock.patch.object(inference, "_inject_voice_leading", side_effect=lambda s, *_a, **_k: s), \
                mock.patch.object(inference, "_explain_each_transition", side_effect=lambda s, *_a, **_k: s), \
                mock.patch.object(chord_context, "compare_with_context", side_effect=lambda _l, _t, s, *_a: (s, None)), \
                mock.patch.object(chord_context, "continuation_scores", side_effect=scores):
            result = inference.generate_ai_result("C major", "pop", context, selected, section_type=section_type)
        return result, generate.call_count, judged

    def test_written_bars_after_the_passage_choose_each_temperatures_draft(self):
        context = [ContextMeasure(measure=0, chords=["Cmaj", "Cmaj"]), ContextMeasure(measure=1, chords=[None, None]),
                   ContextMeasure(measure=2, chords=[None, None]), ContextMeasure(measure=3, chords=["Fmaj", "Gmaj"])]
        result, calls, judged = self.run_request(context, [1, 2])
        self.assertEqual(calls, 6, "two drafts at each of the three temperatures")
        self.assertEqual(len(judged), 3)
        for suggestion in result.suggestions:
            self.assertEqual(suggestion.chords, {"1": ["Fmaj", "Fmaj"], "2": ["Am", "Dm"]})

    def test_drafts_are_judged_with_the_song_form_rules_they_will_get(self):
        context = [ContextMeasure(measure=0, chords=["Cmaj", "Cmaj"]), ContextMeasure(measure=1, chords=[None, None]),
                   ContextMeasure(measure=2, chords=[None, None]), ContextMeasure(measure=3, chords=["Fmaj", "Gmaj"])]
        result, _, judged = self.run_request(context, [1, 2], section_type="chorus")
        # The chorus ends on V7 - I: the judge saw that ending on every draft...
        for trial in judged[0]:
            self.assertEqual(trial["2"], ["G7", "Cmaj"])
        # ...the drafts kept their own chords until the pipeline applied the rules once
        self.assertEqual(result.suggestions[0].chords, {"1": ["Fmaj", "Fmaj"], "2": ["G7", "Cmaj"]})

    def test_without_written_bars_after_the_passage_each_temperature_draws_once(self):
        context = [ContextMeasure(measure=0, chords=["Cmaj", "Cmaj"]), ContextMeasure(measure=1, chords=[None, None]),
                   ContextMeasure(measure=2, chords=[None, None])]
        result, calls, judged = self.run_request(context, [1, 2])
        self.assertEqual((calls, judged), (3, []))
        self.assertEqual(result.suggestions[0].chords, {"1": ["Dm", "Fmaj"], "2": ["Em", "Am"]})

    def test_with_nothing_before_the_passage_drafts_are_judged_key_anchored(self):
        # Bars 1-2 open the song, so the model sees only the header and every draft is anchored to the key
        context = [ContextMeasure(measure=0, chords=[None, None]), ContextMeasure(measure=1, chords=[None, None]),
                   ContextMeasure(measure=2, chords=["Fmaj", "Gmaj"])]
        real_anchor = inference._anchor_candidates
        anchored: list[list[str]] = []

        def recording(suggestions, key_root, is_minor):
            anchored.append([s.label for s in suggestions])
            real_anchor(suggestions, key_root, is_minor)

        with mock.patch.object(inference, "_anchor_candidates", side_effect=recording):
            result, calls, judged = self.run_request(context, [0, 1])
        self.assertEqual(calls, 6)
        # The judge's copies (labelled by draft index) were anchored before scoring, at each temperature,
        # and the kept candidates once more in the pipeline
        self.assertEqual(anchored[:3], [["0", "1"]] * 3)
        self.assertEqual(len(anchored), 4)
        self.assertEqual(len(judged), 3)

    def test_one_draft_by_setting_turns_the_choice_off(self):
        context = [ContextMeasure(measure=0, chords=["Cmaj", "Cmaj"]), ContextMeasure(measure=1, chords=[None, None]),
                   ContextMeasure(measure=2, chords=["Fmaj", "Gmaj"])]
        _, calls, judged = self.run_request(context, [1], draws="1")
        self.assertEqual((calls, judged), (3, []))


if __name__ == "__main__":
    unittest.main()
