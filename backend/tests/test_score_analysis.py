"""Local PDF boundary -> actual deterministic note engine -> import contract.

The PDF parser is mocked here deliberately: these tests establish orchestration,
not glyph recognition or accuracy on a real score. No files or API are used.
"""

import base64
from dataclasses import asdict
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

from app.models.score_import import ScoreImportRequest
from app.services import harmony_analysis, score_analysis, score_import
from app.services.score_analysis import LocalScoreUnsupported, analyze_local_score


PDF = b"%PDF-1.7\n(synthetic parser boundary only)\n%%EOF"


def request(**overrides):
    return ScoreImportRequest(**{
        "filename": "synthetic.pdf", "mimeType": "application/pdf",
        "dataBase64": base64.b64encode(PDF).decode("ascii"),
        "mode": "infer_harmony", **overrides,
    })


def notes(pitches, *, duration=2.0, onset=0.0, staff=1, spellings=None):
    names = {0: "C", 1: "C#", 2: "D", 3: "Eb", 4: "E", 5: "F", 6: "F#",
             7: "G", 8: "Ab", 9: "A", 10: "Bb", 11: "B"}
    return [SimpleNamespace(midi=pitch, onset=onset, duration=duration, staff=staff,
                            spelling=(spellings[i] if spellings else names[pitch % 12]) + str(pitch // 12 - 1))
            for i, pitch in enumerate(pitches)]


def bar(index=0, *, read_notes=(), chords=(), meter=(4, 4), no_chord=False,
        warnings=(), repeat_markers=(), rhythm_exact=True):
    return SimpleNamespace(index=index, page=1, system=1, number=index + 1,
                           time_signature=meter, notes=list(read_notes),
                           chords=[SimpleNamespace(symbol=symbol) for symbol in chords],
                           no_chord=no_chord, warnings=list(warnings),
                           repeat_markers=list(repeat_markers), rhythm_exact=rhythm_exact)


def parsed_score(measures=None, **overrides):
    return SimpleNamespace(**{
        "supported": True, "complete": True, "title": "Synthetic notation",
        "key": "C major", "key_signature": 0, "time_signature": (4, 4),
        "measures": measures if measures is not None else [bar(read_notes=notes([48, 52, 55]))],
        "warnings": [], "unresolved": [], **overrides,
    })


class ScoreAnalysisTest(unittest.TestCase):
    def setUp(self):
        # Keeping the parser interface behind a fake module allows the reader
        # and the orchestrator to be developed independently. Its contract is
        # exercised below with realistic multi-measure note evidence.
        self.parser_module = ModuleType("app.services.score_pdf")
        self.parser_module.ScorePDFError = type("ScorePDFError", (ValueError,), {})
        self.parser = mock.Mock(return_value=parsed_score())
        self.parser_module.parse_vector_score = self.parser
        modules = mock.patch.dict(sys.modules, {"app.services.score_pdf": self.parser_module})
        modules.start()
        self.addCleanup(modules.stop)

    def test_local_pdf_reaches_real_note_engine_and_returns_local_source_without_a_model(self):
        analyses = []
        real_analyze = harmony_analysis.analyze_harmony

        def capture(*args, **kwargs):
            result = real_analyze(*args, **kwargs)
            analyses.append(result)
            return result

        with mock.patch("app.services.harmony_analysis.analyze_harmony", side_effect=capture):
            result = score_import.import_score(request())
        self.parser.assert_called_once_with(PDF)
        self.assertEqual(result.analysisSource, "local")
        self.assertIsNone(result.model)
        self.assertEqual(result.mode, "infer_harmony")
        self.assertEqual(result.measures[0].chords, ["C"])
        self.assertEqual(result.measures[0].chordBasis, "inferred")
        self.assertTrue(result.measures[0].needsReview)
        self.assertEqual(result.measures[0].notes, ["C3", "E3", "G3"])
        self.assertEqual(analyses[0].segments[0].candidates[0].roman, "I")
        self.assertIn("Active MIDI pitches", result.measures[0].note)
        self.assertEqual(result.model_dump()["analysisSource"], "local")

    def test_explicit_no_chord_measure_is_preserved_without_inventing_harmony(self):
        self.parser.return_value = parsed_score([bar(no_chord=True)])
        result = analyze_local_score(request(mode="chord_symbols"))
        self.assertTrue(result.measures[0].noChord)
        self.assertEqual(result.measures[0].chords, [])
        self.assertEqual(result.measures[0].chordBasis, "printed")
        self.assertFalse(result.measures[0].needsReview)

    def test_printed_three_change_bar_is_not_truncated_or_replaced_by_note_analysis(self):
        self.parser.return_value = parsed_score([bar(read_notes=notes([48, 52, 55]),
                                                   chords=["Cmaj7", "Dm7", "G7"])])
        with mock.patch("app.services.harmony_analysis.analyze_harmony") as analyzer:
            result = analyze_local_score(request())
        analyzer.assert_not_called()
        self.assertEqual(result.measures[0].chords, ["Cmaj7", "Dm7", "G7"])
        self.assertTrue(result.measures[0].needsReview)
        self.assertEqual(result.measures[0].chordBasis, "printed")

    def test_mixed_meter_uses_each_measure_meter_and_preserves_source_warning(self):
        self.parser.return_value = parsed_score([
            bar(read_notes=notes([48, 52, 55], duration=4), meter=(4, 4)),
            bar(1, read_notes=notes([43, 47, 50, 53], duration=2), meter=(2, 4)),
            bar(2, read_notes=notes([45, 48, 52], duration=3), meter=(3, 4)),
        ])
        with mock.patch("app.services.harmony_analysis.analyze_harmony", wraps=harmony_analysis.analyze_harmony) as analyzer:
            result = analyze_local_score(request())
        self.assertEqual([call.kwargs["time_signature"] for call in analyzer.call_args_list],
                         [(4, 4), (2, 4), (3, 4)])
        self.assertEqual([measure.timeSignature for measure in result.measures], [[4, 4], [2, 4], [3, 4]])
        self.assertEqual([measure.chords for measure in result.measures], [["C"], ["G7"], ["Am"]])
        self.assertTrue(any("changes meter" in warning for warning in result.warnings))

    def test_key_changes_reanalyze_fixed_note_evidence_without_transposing_it(self):
        analyses = []
        real_analyze = harmony_analysis.analyze_harmony

        def capture(*args, **kwargs):
            result = real_analyze(*args, **kwargs)
            analyses.append(result)
            return result

        with mock.patch("app.services.harmony_analysis.analyze_harmony", side_effect=capture):
            c_major = analyze_local_score(request())
            self.parser.return_value.key = "G major"
            g_major = analyze_local_score(request())
        self.assertEqual(c_major.key, "C major")
        self.assertEqual(g_major.key, "G major")
        self.assertEqual(c_major.measures[0].chords, g_major.measures[0].chords)
        self.assertEqual(c_major.measures[0].notes, g_major.measures[0].notes)
        self.assertEqual([result.segments[0].candidates[0].roman for result in analyses], ["I", "IV"])
        self.assertEqual(asdict(analyses[0].segments[0].notes[0]), asdict(analyses[1].segments[0].notes[0]))

    def test_unknown_key_uses_duration_profile_and_signature_without_forcing_major(self):
        self.parser.return_value = parsed_score([
            bar(read_notes=notes([50, 53, 57], duration=4)),
            bar(1, read_notes=notes([45, 49, 52, 55], duration=2), meter=(2, 4)),
            bar(2, read_notes=notes([50, 53, 57], duration=4)),
        ], key=None, key_signature=-1)
        result = analyze_local_score(request())
        self.assertEqual(result.key, "D minor")
        self.assertTrue(any("key is inferred" in warning for warning in result.warnings))

    def test_approximate_rhythm_proposals_remain_low_confidence_and_require_review(self):
        self.parser.return_value = parsed_score([bar(read_notes=notes([48, 52, 55], duration=None),
                                                   rhythm_exact=False)], key=None)
        key_results = []
        real_estimate = harmony_analysis.estimate_key

        def capture(*args, **kwargs):
            result = real_estimate(*args, **kwargs)
            key_results.append(result)
            return result

        with mock.patch("app.services.harmony_analysis.estimate_key", side_effect=capture) as estimator:
            result = analyze_local_score(request())
        self.assertEqual(result.measures[0].chords, ["C"])
        self.assertEqual(result.measures[0].confidence, "low")
        self.assertTrue(result.measures[0].needsReview)
        self.assertIn("Timing is approximate", result.measures[0].note)
        self.assertTrue(estimator.call_args.kwargs["timing_is_estimated"])
        self.assertLessEqual(key_results[0].confidence, .35)
        self.assertTrue(any("durations could not be decoded exactly" in warning for warning in result.warnings))

    def test_bass_evidence_uses_actual_sounding_pitch_when_staffs_or_voices_cross(self):
        read_notes = notes([43], duration=4, staff=0)
        read_notes += notes([60, 64, 67], duration=4, staff=2)
        read_notes += notes([40], onset=1, duration=.5, staff=2)
        self.parser.return_value = parsed_score([bar(read_notes=read_notes, chords=["C"])])
        result = analyze_local_score(request(mode="chord_symbols"))
        self.assertEqual(result.measures[0].bassNotes, ["G2", "E2"])
        self.assertNotIn("C4", result.measures[0].bassNotes)

    def test_unknown_duration_is_not_silently_sustained_as_bass_at_later_onsets(self):
        read_notes = notes([36], duration=None, staff=0)
        read_notes += notes([40, 55, 60], onset=1, duration=1, staff=1)
        self.parser.return_value = parsed_score([bar(read_notes=read_notes, chords=["C"], rhythm_exact=False)])
        result = analyze_local_score(request(mode="chord_symbols"))
        self.assertEqual(result.measures[0].bassNotes, ["C2", "E2"])

    def test_bass_change_after_a_note_release_is_retained_without_a_new_attack(self):
        read_notes = notes([36], duration=1, staff=0) + notes([40], duration=4, staff=1)
        self.parser.return_value = parsed_score([bar(read_notes=read_notes, chords=["C"])])
        result = analyze_local_score(request(mode="chord_symbols"))
        self.assertEqual(result.measures[0].bassNotes, ["C2", "E2"])

    def test_polyphonic_changes_are_retained_in_order_within_the_source_measure(self):
        read_notes = notes([48, 52, 55], onset=0, duration=1)
        read_notes += notes([53, 57, 60], onset=1, duration=1)
        read_notes += notes([43, 47, 50, 53], onset=2, duration=2)
        self.parser.return_value = parsed_score([bar(read_notes=read_notes)])
        result = analyze_local_score(request())
        self.assertEqual(result.measures[0].chords, ["C", "F", "G7"])
        self.assertTrue(result.measures[0].needsReview)
        self.assertEqual(len(result.measures), 1)

    def test_unread_or_monophonic_evidence_does_not_become_a_fake_chord(self):
        self.parser.return_value = parsed_score([bar(read_notes=[])])
        with self.assertRaisesRegex(LocalScoreUnsupported, "No supported note"):
            analyze_local_score(request())
        for read_notes in (notes([60]), notes([60, 64])):
            with self.subTest(pitches=[note.midi for note in read_notes]):
                self.parser.return_value = parsed_score([bar(read_notes=read_notes)])
                draft = analyze_local_score(request())
                self.assertEqual(draft.measures[0].chords, [])
                self.assertEqual(draft.measures[0].chordBasis, "unreadable")
                self.assertTrue(draft.measures[0].needsReview)
                self.assertTrue(draft.measures[0].notes)

    def test_sparse_read_notes_stay_as_evidence_through_harmony_completion(self):
        self.parser.return_value = parsed_score([bar(read_notes=notes([60]))])
        draft = score_import.import_score(request())
        self.assertEqual(draft.measures[0].notes, ["C4"])
        self.assertIn("no sufficiently supported harmony", draft.warnings[0])

    def test_wrong_format_and_failed_partial_or_unsupported_reader_fail_explicitly(self):
        with self.assertRaisesRegex(LocalScoreUnsupported, "vector PDF"):
            analyze_local_score(request(mimeType="image/png", filename="score.png"))
        self.parser.assert_not_called()
        self.parser.side_effect = self.parser_module.ScorePDFError("Unsupported clef evidence")
        with self.assertRaisesRegex(LocalScoreUnsupported, "Unsupported clef evidence"):
            analyze_local_score(request())
        self.parser.side_effect = None
        for parsed, message in ((parsed_score(supported=False), "No supported"),
                                (parsed_score(measures=[]), "No supported"),
                                (parsed_score(complete=False), "complete written measure order")):
            with self.subTest(message=message):
                self.parser.return_value = parsed
                with self.assertRaisesRegex(LocalScoreUnsupported, message):
                    analyze_local_score(request())

    def test_unsupported_or_incomplete_readings_are_422_errors_that_name_vector_pdfs(self):
        for parsed in (parsed_score(supported=False), parsed_score(complete=False)):
            with self.subTest(supported=parsed.supported, complete=parsed.complete):
                self.parser.return_value = parsed
                with self.assertRaises(score_import.ScoreInputError) as error:
                    score_import.import_score(request())
                self.assertEqual(error.exception.status_code, 422)
                self.assertIn("vector PDF", str(error.exception))

    def test_printed_chord_score_imports_as_a_local_reading(self):
        self.parser.return_value = parsed_score([bar(chords=["C"])])
        result = score_import.import_score(request())
        self.assertEqual(result.analysisSource, "local")
        self.assertEqual(result.measures[0].chords, ["C"])

    def test_valid_pdf_is_read_once_and_bytes_that_are_not_a_pdf_never_reach_the_parser(self):
        result = score_import.import_score(request())
        self.assertEqual(result.analysisSource, "local")
        self.parser.assert_called_once()
        self.parser.reset_mock()
        with self.assertRaises(score_import.ScoreInputError):
            score_import.import_score(request(dataBase64=base64.b64encode(b"Not a PDF").decode("ascii")))
        self.parser.assert_not_called()


if __name__ == "__main__":
    unittest.main()
