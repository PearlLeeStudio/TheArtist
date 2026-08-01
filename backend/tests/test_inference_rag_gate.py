"""Actual retrieval with synthetic text: source opt-ins, isolation and rollback."""
import os
import unittest
from unittest import mock
from app.models.schemas import Suggestion, TheoryCitation
from app.services import inference, theory_rag as rag


def passage(source, text):
    return rag.Passage(text, "Synthetic theory", 0, 0,
        TheoryCitation(source=source, title=source, license="test fixture", passage_id=source))


class RagGateTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(mock.patch.dict(os.environ, {}, clear=True))
        self.enterContext(mock.patch.object(inference, "RAG_AVAILABLE", True))
        self.enterContext(mock.patch.object(rag, "_signature", return_value=(1, 10)))
        self.omt = self.enterContext(mock.patch.object(rag, "load_omt", return_value=[passage("omt", "Harmonic function concerns chord progression and cadence.")]))
        self.berklee = self.enterContext(mock.patch.object(rag, "load_berklee", return_value=[passage("berklee", "Dominant tonic resolution authentic cadence applied tonicization harmonic progression.")]))
        rag._runtime.cache_clear()
        self.addCleanup(rag._runtime.cache_clear)

    def test_default_omt_loads_without_berklee_or_external_models(self):
        with mock.patch.dict("sys.modules", {"rag.retrieve": None, "rag.explain": None}):
            self.assertFalse(inference.rag_disabled())
            self.assertTrue(inference._should_use_rag("pop"))
            note = inference._explain_one("G7", "Cmaj")
        self.assertEqual(note.kind, "source_excerpt")
        self.assertEqual(note.citation.source, "omt")
        self.omt.assert_called_once()
        self.berklee.assert_not_called()

    def test_explicit_berklee_never_leaks_after_disable(self):
        os.environ["THEARTIST_RAG_BERKLEE_ENABLED"] = "1"
        both = inference._explain_one("G7", "Cmaj")
        self.assertEqual(both.citation.source, "berklee")
        os.environ["THEARTIST_RAG_BERKLEE_ENABLED"] = "0"
        omt = inference._explain_one("G7", "Cmaj")
        self.assertEqual(omt.citation.source, "omt")
        self.assertNotEqual(both.explanation, omt.explanation)
        self.assertEqual(rag.status()["sources"], {"omt": "ready", "berklee": "disabled"})

    def test_berklee_only_is_separate_from_omt(self):
        os.environ.update(THEARTIST_RAG_OMT_ENABLED="0", THEARTIST_RAG_BERKLEE_ENABLED="1")
        self.assertEqual(inference._explain_one("G7", "Cmaj").citation.source, "berklee")
        self.omt.assert_not_called()

    def test_public_host_rejects_berklee_before_reading(self):
        os.environ.update(THEARTIST_PUBLIC_HOST="1", THEARTIST_RAG_BERKLEE_ENABLED="1")
        with self.assertRaisesRegex(ValueError, "private research"):
            inference._get_theory_retriever()
        self.omt.assert_not_called()
        self.berklee.assert_not_called()
        self.assertFalse(rag.status()["ready"])

    def test_public_host_omt_is_available(self):
        os.environ["THEARTIST_PUBLIC_HOST"] = "1"
        self.assertTrue(inference._should_use_rag("classical"))
        self.berklee.assert_not_called()

    def test_default_does_not_even_stat_berklee(self):
        with mock.patch.object(rag, "_signature", return_value=(2, 20)) as signature:
            rag.runtime()
        self.assertEqual([call.args[0] for call in signature.call_args_list], [str(rag.OMT_PATH), str(rag.MANIFEST_PATH)])

    def test_corpus_or_manifest_change_invalidates_cached_index(self):
        first = rag.runtime()[0]
        with mock.patch.object(rag, "_signature", return_value=(2, 20)):
            self.assertIsNot(first, rag.runtime()[0])
        self.assertEqual(self.omt.call_count, 2)

    def test_malformed_corpus_is_reported_as_invalid(self):
        self.omt.side_effect = ValueError("Not a reviewed OMT corpus")
        self.assertEqual(rag.status()["sources"], {"omt": "invalid", "berklee": "disabled"})
        self.assertIsNone(inference._get_theory_retriever())

    def test_global_rollback_stops_previously_cached_explanations(self):
        self.assertIsNotNone(inference._explain_one("G7", "Cmaj"))
        os.environ["THEARTIST_DISABLE_RAG"] = "1"
        self.assertIsNone(inference._explain_one("G7", "Cmaj"))
        self.assertIsNone(inference._theory_distance("G7", "Cmaj"))
        self.assertFalse(inference._should_use_rag("jazz"))

    def test_both_sources_off_do_not_stat_or_read_files(self):
        os.environ.update(THEARTIST_RAG_OMT_ENABLED="0", THEARTIST_RAG_BERKLEE_ENABLED="0")
        with mock.patch.object(rag, "_signature", side_effect=AssertionError("No file access")):
            self.assertTrue(inference.rag_disabled())
            self.assertFalse(rag.status()["ready"])
        self.omt.assert_not_called()
        self.berklee.assert_not_called()

    def test_bad_flags_raise_instead_of_enabling_a_source(self):
        for flag in ("THEARTIST_DISABLE_RAG", "THEARTIST_RAG_OMT_ENABLED", "THEARTIST_RAG_BERKLEE_ENABLED"):
            with self.subTest(flag=flag), mock.patch.dict(os.environ, {flag: "maybe"}), self.assertRaises(ValueError):
                inference._get_theory_retriever()

    def test_missing_omt_is_reported_without_berklee_fallback(self):
        self.omt.side_effect = FileNotFoundError()
        self.assertEqual(rag.status()["sources"], {"omt": "missing", "berklee": "disabled"})
        self.assertFalse(inference._should_use_rag("jazz"))
        self.assertIsNone(inference._explain_one("G7", "Cmaj"))
        self.berklee.assert_not_called()

    def test_disabled_notes_do_not_disable_ranking(self):
        os.environ["THEARTIST_DISABLE_EXPLANATIONS"] = "1"
        suggestions = [Suggestion(label="A", chords={"0": ["G7", "Cmaj"]})]
        inference._attach_explanations(suggestions, [0])
        self.assertIsNone(suggestions[0].explanations)
        self.assertIsNotNone(inference._theory_distance("G7", "Cmaj"))

    def test_uncovered_transition_does_not_earn_perfect_distance(self):
        self.assertEqual(inference._theory_distance("N.C.", "Cmaj"), 1.0)

    def test_repeated_pairs_share_one_note_and_results_are_independent(self):
        suggestions = [Suggestion(label="A", chords={"0": ["G7", "Cmaj"], "1": ["G7", "Cmaj"]}), Suggestion(label="B", chords={"0": ["G7", "Cmaj"]})]
        inference._attach_explanations(suggestions, [0, 1])
        self.assertEqual(len(suggestions[0].explanations), 2)
        self.assertEqual(len(suggestions[1].explanations), 1)
        suggestions[0].explanations[0].citation.authors.append("changed")
        self.assertNotIn("changed", suggestions[1].explanations[0].citation.authors)
