"""Sparse search, chord parsing and the choice of the three alternatives; no model is loaded."""
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from app.models.reharmonize import Change, ReharmCandidate, ReharmRequest, Slot
from app.services import reharmonize_choice
from app.services.reharmonize import chord, context_window, convention_preference, model_symbol, search


def request(**overrides):
    return ReharmRequest(**{
        "key": "C major", "context": [
            {"measure": 0, "chords": ["Cmaj7", "Am7"]},
            {"measure": 1, "chords": ["Dm7", "G7"]},
            {"measure": 2, "chords": ["Cmaj7", None]}],
        "selectedSlots": [{"measureIndex": 1, "slotIndex": 0}, {"measureIndex": 1, "slotIndex": 1}],
        **overrides})


class FakeScorer:
    label = "test-checkpoint"

    def score(self, candidates):
        return [c.model_copy(update={"modelScore": -i - .5}) for i, c in enumerate(candidates)]


def candidate(id, family, after, score, slot=0):
    return ReharmCandidate(id=id, family=family, changes=[Change(
        address=Slot(measureIndex=1, slotIndex=slot), before="Dm7", after=after, commonTones=2, modes=[])],
        pitchDistance=0, originalPitchDistance=0, bassMotion=0, originalBassMotion=0, mathScore=0, modelScore=score)


class SearchTests(unittest.TestCase):
    def test_augmented_sevenths_and_imported_extensions_keep_exact_pitch_classes(self):
        fixtures = {
            'Eaug7': {0, 2, 4, 8}, 'E+7': {0, 2, 4, 8}, 'E7#5': {0, 2, 4, 8},
            'Caugmaj7': {0, 4, 8, 11}, 'C7sus4': {0, 5, 7, 10},
            'C6/9': {0, 2, 4, 7, 9}, 'C6/9/Db': {0, 1, 2, 4, 7, 9},
            'C7(b9,#11)': {0, 1, 4, 6, 7, 10}, 'C7(b9#11)': {0, 1, 4, 6, 7, 10},
            'C7(#9,b9)': {0, 1, 3, 4, 7, 10}, 'C7(b9,#9)': {0, 1, 3, 4, 7, 10},
            'Cø7': {0, 3, 6, 10}, 'CΔ9': {0, 2, 4, 7, 11},
        }
        for symbol, expected in fixtures.items():
            with self.subTest(symbol=symbol):
                self.assertEqual(chord(symbol)[1], expected)
        self.assertEqual(model_symbol('Eaug7'), 'E7')
        self.assertEqual(model_symbol('C7(b9)'), 'C7b9')
        for symbol in ('Cgarbage', 'C7(b9,,#11)', 'C7(b9', 'C7/', 'C7/9', 'C7alt'):
            with self.subTest(symbol=symbol), self.assertRaises(ValueError):
                chord(symbol)

    def test_aug7_target_and_context_across_unresolved_bars_can_be_searched(self):
        req = request(context=[{'measure': i, 'chords': row} for i, row in enumerate([
            ['Cmaj7', None], [None, None], ['Eaug7', None], [None, None], ['Am7', None]])],
            selectedSlots=[{'measureIndex': 2, 'slotIndex': 0}])
        before = req.model_dump_json()
        for family in ('common_tones', 'modal', 'dominant'):
            candidates = search(req, family)
            self.assertTrue(candidates)
            self.assertTrue(all(c.changes[0].before == 'Eaug7' for c in candidates))
            self.assertTrue(all(c.changes[0].address.measureIndex == 2 for c in candidates))
        self.assertEqual(req.model_dump_json(), before)
        req.selectedSlots[0].measureIndex = 4
        self.assertTrue(search(req, 'common_tones'))

    def test_six_nine_is_not_mistaken_for_a_slash_bass(self):
        for symbol, bass in [('C6/9', None), ('C6/9/G', 'G')]:
            req = request(context=[{'measure': 0, 'chords': [symbol, 'G7']}],
                selectedSlots=[{'measureIndex': 0, 'slotIndex': 0}])
            candidates = search(req, 'common_tones')
            self.assertTrue(candidates)
            self.assertTrue(all(c.changes[0].after.endswith('/G') if bass else '/' not in c.changes[0].after
                                for c in candidates))

    def test_sparse_unique_and_original_immutable(self):
        for maximum in (1, 2, 3, 4):
            req = request(maxChanges=maximum)
            original = req.model_dump_json()
            for family in ("common_tones", "modal", "dominant"):
                candidates = search(req, family)
                self.assertGreaterEqual(len(candidates), 2)
                self.assertEqual(len({c.id for c in candidates}), len(candidates))
                for candidate in candidates:
                    self.assertLessEqual(len(candidate.changes), maximum)
                    for change in candidate.changes:
                        self.assertEqual(change.address.measureIndex, 1)
                        self.assertEqual(change.before, req.context[1].chords[change.address.slotIndex])
                        self.assertNotEqual(change.before, change.after)
            self.assertEqual(req.model_dump_json(), original)

    def test_wider_search_reaches_four_changes_without_exceeding_six_candidates(self):
        slots = [{"measureIndex": m, "slotIndex": s} for m in range(2) for s in range(2)]
        addresses = {(s['measureIndex'], s['slotIndex']) for s in slots}
        for maximum in (3, 4):
            req = request(maxChanges=maximum, selectedSlots=slots)
            original = req.model_dump_json()
            for family in ('common_tones', 'modal', 'dominant'):
                with self.subTest(maximum=maximum, family=family):
                    candidates = search(req, family)
                    self.assertEqual(len(candidates), 6)
                    self.assertIn(maximum, {len(c.changes) for c in candidates})
                    self.assertEqual(len({c.id for c in candidates}), len(candidates))
                    for candidate in candidates:
                        changed = {(c.address.measureIndex, c.address.slotIndex) for c in candidate.changes}
                        self.assertEqual(len(changed), len(candidate.changes))
                        self.assertTrue(changed <= addresses)
                        self.assertLessEqual(len(changed), maximum)
                        for change in candidate.changes:
                            self.assertEqual(change.before, req.context[change.address.measureIndex].chords[change.address.slotIndex])
                            self.assertNotEqual(change.before, change.after)
            self.assertEqual(req.model_dump_json(), original)

    def test_four_change_search_keeps_mode_constraints_and_small_selections(self):
        req = request(maxChanges=4, mode='dorian',
                      selectedSlots=[{'measureIndex': m, 'slotIndex': s} for m in range(2) for s in range(2)])
        self.assertTrue(all('dorian' in change.modes for c in search(req, 'modal') for change in c.changes))
        one = request(maxChanges=4, selectedSlots=[{'measureIndex': 1, 'slotIndex': 0}])
        candidates = search(one, 'common_tones')
        self.assertTrue(candidates)
        self.assertTrue(all(len(c.changes) == 1 for c in candidates))
        for maximum in (0, 5, 16):
            with self.assertRaises(ValidationError):
                request(maxChanges=maximum)

    def test_extension_and_slash_pitch_classes_are_not_lost(self):
        self.assertEqual(chord("C7(b9)/Db")[1], {0, 1, 4, 7, 10})
        self.assertEqual(chord("Cmaj7/Db")[1], {0, 1, 4, 7, 11})
        self.assertEqual(chord("Cmaj9")[1], {0, 2, 4, 7, 11})
        with self.assertRaises(ValueError):
            chord("Cmadeup")
        req = request(context=[{"measure": 0, "chords": ["Dm7", "G7/B"]}],
                      selectedSlots=[{"measureIndex": 0, "slotIndex": 1}])
        self.assertTrue(all(c.changes[0].after.endswith('/B') for c in search(req, 'common_tones')))

    def test_parallel_mode_is_a_hard_candidate_constraint(self):
        req = request(mode="dorian")
        candidates = search(req, "modal")
        self.assertTrue(candidates)
        self.assertTrue(all("dorian" in change.modes for c in candidates for change in c.changes))

    def test_both_neighbors_change_the_mathematical_comparison(self):
        req = request(selectedSlots=[{"measureIndex": 1, "slotIndex": 1}])
        before = {c.id: c.mathScore for c in search(req, "dominant")}
        req.context[1].chords[0] = "Abmaj7"
        middle = {c.id: c.mathScore for c in search(req, "dominant")}
        self.assertNotEqual(before, middle)
        req.context[2].chords[0] = "Fm7"
        after = {c.id: c.mathScore for c in search(req, "dominant")}
        self.assertNotEqual(before, after)
        self.assertNotEqual(middle, after)

    def test_refuse_blanks_duplicate_targets_and_wide_selection(self):
        for selection in ([{"measureIndex": 2, "slotIndex": 1}],
                          [{"measureIndex": 0, "slotIndex": 0}] * 2,
                          [{"measureIndex": 200, "slotIndex": 0}]):
            with self.assertRaises(ValidationError):
                request(selectedSlots=selection)
        with self.assertRaises(ValidationError):
            request(context=[{"measure": 3, "chords": ["C", "G"]}])

    def test_windowed_context_of_a_long_score_keeps_absolute_addresses(self):
        # The editor sends the bars around the passage of a 465-bar score; every
        # reported address refers to the score, not to the window.
        req = request(context=[{"measure": i, "chords": ["Cmaj7", "Am7"] if i % 2 else ["Dm7", "G7"]}
                               for i in range(300, 340)],
                      selectedSlots=[{"measureIndex": 321, "slotIndex": 0}, {"measureIndex": 321, "slotIndex": 1}],
                      maxChanges=2)
        candidates = search(req, "common_tones")
        self.assertTrue(candidates)
        for candidate in candidates:
            for change in candidate.changes:
                self.assertEqual(change.address.measureIndex, 321)
                self.assertIn(change.before, ("Cmaj7", "Am7"))
        self.assertTrue(any(len(c.changes) == 2 for c in candidates))
        self.assertEqual([bar.measure for bar in context_window(req)], list(range(313, 330)))

    def test_scores_up_to_512_bars_are_accepted_and_windows_need_not_start_at_zero(self):
        request(context=[{"measure": i, "chords": ["C", "G"]} for i in range(512)],
                selectedSlots=[{"measureIndex": 511, "slotIndex": 0}])
        request(context=[{"measure": i, "chords": ["C", "G"]} for i in range(100, 110)],
                selectedSlots=[{"measureIndex": 105, "slotIndex": 0}])
        with self.assertRaisesRegex(ValidationError, "within the context"):
            request(context=[{"measure": i, "chords": ["C", "G"]} for i in range(100, 110)],
                    selectedSlots=[{"measureIndex": 5, "slotIndex": 0}])
        with self.assertRaisesRegex(ValidationError, "consecutive"):
            request(context=[{"measure": 3, "chords": ["C", "G"]}, {"measure": 5, "chords": ["C", "G"]}],
                    selectedSlots=[{"measureIndex": 3, "slotIndex": 0}])

    def test_context_window_preserves_known_bars_on_both_sides(self):
        req = request(context=[{"measure": i, "chords": ["C", "G"]} for i in range(64)],
                      selectedSlots=[{"measureIndex": 30, "slotIndex": 0}])
        self.assertEqual([bar.measure for bar in context_window(req)], list(range(22, 39)))

    def test_familiar_predominants_get_contextual_support_not_a_blanket_mode_bonus(self):
        resolving, reasons = convention_preference('C major', 'Dm7', 'Dm7b5', 'Am7', 'G7')
        unrelated, _ = convention_preference('C major', 'Dm7', 'Dm7b5', 'Am7', 'Bb')
        self.assertGreater(resolving, unrelated)
        self.assertTrue(any('predominant' in label for label in reasons))
        self.assertGreater(convention_preference('C major', 'F', 'Fm', 'C', 'G7')[0], 0)
        self.assertEqual(convention_preference('C major', 'F', 'F#m7b5', 'C', 'Ab')[0], 0)

    def test_tritone_and_backdoor_preferences_require_their_resolution(self):
        self.assertGreater(convention_preference('C major', 'G7', 'Db7', 'Dm7', 'Cmaj7')[0], 0)
        self.assertEqual(convention_preference('C major', 'G7', 'Db7', 'Dm7', 'Am7')[0], 0)
        prepared = convention_preference('C major', 'G7', 'Bb7', 'Fm7', 'Cmaj7')
        unprepared = convention_preference('C major', 'G7', 'Bb7', 'Dm7', 'Cmaj7')
        self.assertGreater(prepared[0], unprepared[0])
        self.assertIn('Backdoor dominant to I', prepared[1])
        self.assertNotIn('Backdoor dominant to I', convention_preference('C major', 'G7', 'Bb7', 'Fm7', 'Em7')[1])

    def test_priors_transpose_and_support_minor_dorian_and_local_lydian(self):
        self.assertEqual(convention_preference('C major', 'Dm7', 'Dm7b5', 'Am7', 'G7'),
                         convention_preference('D major', 'Em7', 'Em7b5', 'Bm7', 'A7'))
        self.assertGreater(convention_preference('C minor', 'Fm', 'F', 'Cm', 'G7')[0], 0)
        self.assertGreater(convention_preference('C major', 'Dm', 'D', 'C', 'G')[0], 0)
        self.assertEqual(convention_preference('C major', 'Dm', 'D', 'Am', 'G')[0], 0)

    def test_familiar_options_reach_the_shortlist_with_the_contextual_prior(self):
        candidates = search(request(), 'modal')
        borrowed = next(c for c in candidates if c.changes[0].after == 'Dm7b5')
        self.assertGreater(borrowed.conventionScore, 0)
        with patch('app.services.reharmonize.convention_preference', return_value=(0, [])):
            no_prior = search(request(), 'modal', limit=100)
        self.assertNotIn(borrowed.id, {c.id for c in no_prior})
        self.assertTrue(any('Resolving tritone substitute' in change.conventions
                            for c in search(request(), 'dominant') for change in c.changes))

    def test_empty_half_bars_do_not_hide_the_previous_or_following_harmony(self):
        dense = request(context=[{'measure': 0, 'chords': ['Dm7', 'G7']}, {'measure': 1, 'chords': ['Cmaj7', None]}],
                        selectedSlots=[{'measureIndex': 0, 'slotIndex': 1}])
        sparse = request(context=[{'measure': i, 'chords': [symbol, None]}
                                  for i, symbol in enumerate(['Dm7', 'G7', 'Cmaj7'])],
                         selectedSlots=[{'measureIndex': 1, 'slotIndex': 0}])
        def facts(req):
            return {c.changes[0].after: (c.mathScore, c.conventionScore, c.changes[0].conventions)
                    for c in search(req, 'dominant')}
        self.assertEqual(facts(dense), facts(sparse))


class ChoiceTests(unittest.TestCase):
    """The product entry point with a fake scorer, and the choice among scored candidates."""

    def reharmonize(self, req):
        with patch('app.services.reharmonize.ContextScorer', return_value=FakeScorer()):
            return reharmonize_choice.reharmonize(req)

    def test_bad_context_names_its_slot_before_model_loading(self):
        req = request()
        req.context[0].chords[1] = 'Cmadeup'
        with patch('app.services.reharmonize.ContextScorer') as scorer, self.assertRaisesRegex(ValueError, 'M1.2'):
            reharmonize_choice.reharmonize(req)
        scorer.assert_not_called()

    def test_the_three_are_chosen_from_every_family_in_score_order(self):
        req = request()
        original = req.model_dump_json()
        result = self.reharmonize(req)
        self.assertEqual(len(result.candidates), 3)
        self.assertEqual(len({c.id for c in result.candidates}), 3)
        scores = [c.modelScore for c in result.candidates]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual({c.family for c in result.candidates}, {"common_tones", "modal", "dominant"})
        self.assertEqual((result.modelUsed, result.contextMeasures), ("test-checkpoint", [0, 1, 2]))
        self.assertEqual(result.modelApproximations, [])
        self.assertEqual(req.model_dump_json(), original)

    def test_chords_scored_under_a_simpler_symbol_are_reported(self):
        req = request()
        req.context[0].chords[1] = 'Eaug7'
        result = self.reharmonize(req)
        self.assertEqual(len(result.candidates), 3)
        self.assertEqual([a.model_dump() for a in result.modelApproximations],
                         [{'address': {'measureIndex': 0, 'slotIndex': 1}, 'original': 'Eaug7', 'scoredAs': 'E7'}])

    def test_too_few_substitutes_is_a_plain_error(self):
        with patch.object(reharmonize_choice, "search", return_value=[]), self.assertRaisesRegex(ValueError, "Only 0 substitutes"):
            self.reharmonize(request())

    def test_choice_spans_families_and_roots_before_repeating_one(self):
        pool = [candidate("a", "common_tones", "F6", -1.0), candidate("b", "common_tones", "Fmaj7", -1.1),
                candidate("d", "modal", "Fm", -1.2), candidate("e", "modal", "Bbmaj7", -2.0),
                candidate("c", "dominant", "Ab7", -3.0)]
        # modal's best (Fm) repeats a's root on the same slot, so its next candidate is offered instead
        self.assertEqual([c.id for c in reharmonize_choice.choose(pool)], ["a", "e", "c"])
        one_family = [candidate("a", "common_tones", "F6", -1.0), candidate("b", "common_tones", "Fmaj7", -1.1),
                      candidate("x", "common_tones", "Am7", -2.0)]
        # a repeated root is offered only when nothing else is left; the three stay in score order
        self.assertEqual([c.id for c in reharmonize_choice.choose(one_family)], ["a", "b", "x"])


class ManualSpellingTests(unittest.TestCase):
    def test_manual_builder_spellings_are_scored_by_their_tones(self):
        for symbol, tones, projected in (("Dm(add9)", {2, 4, 5, 9}, "Dm"), ("Dmadd9", {2, 4, 5, 9}, "Dm"),
                                         ("G7(no3)", {7, 2, 5}, "G7"), ("Am(add4)", {9, 0, 2, 4}, "Am")):
            with self.subTest(symbol=symbol):
                self.assertEqual(set(chord(symbol)[1]), tones)
                self.assertEqual(model_symbol(symbol), projected)
