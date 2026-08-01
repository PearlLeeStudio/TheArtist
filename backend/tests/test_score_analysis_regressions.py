"""Repeat semantics through the real harmony engine, without files."""

import base64
import unittest
from unittest import mock

from app.models.score_import import ScoreImportRequest
from app.services.score_import import import_score
from app.services.score_pdf import NoteEvent, ParsedScore, PrintedChord, ScoreMeasure


class ScoreRepeatRegressionTest(unittest.TestCase):
    def analyze(self, bars, *, mode="infer_harmony"):
        score = ParsedScore(title="Original repeat regression", key="C major", key_signature=0,
                            time_signature=(4, 4), measures=bars, supported=True)
        request = ScoreImportRequest(filename="original.pdf", mimeType="application/pdf",
                                     dataBase64=base64.b64encode(b"%PDF-original parser boundary").decode(),
                                     mode=mode)
        with mock.patch("app.services.score_pdf.parse_vector_score", return_value=score):
            return import_score(request)

    def test_repeat_preserves_inferred_origin_confidence_and_required_review(self):
        for exact in (True, False):
            with self.subTest(rhythm_exact=exact):
                notes = [NoteEvent(0, midi, spelling, 0, 4, 120, 200)
                         for midi, spelling in ((60, "C4"), (64, "E4"), (67, "G4"))]
                first = ScoreMeasure(0, 1, 1, 1, (4, 4), notes=notes, rhythm_exact=exact,
                                     warnings=[] if exact else ["Source timing is approximate."])
                repeat = ScoreMeasure(1, 1, 1, 2, (4, 4), repeat_markers=["%"])
                result = self.analyze([first, repeat])
                original, repeated = result.measures
                self.assertEqual(original.chords, ["C"])
                self.assertEqual(original.chordBasis, "inferred")
                self.assertEqual(repeated.chords, original.chords)
                self.assertEqual(repeated.chordBasis, original.chordBasis)
                self.assertEqual(repeated.confidence, original.confidence)
                self.assertTrue(original.needsReview)
                self.assertTrue(repeated.needsReview)
                self.assertIn("repeat-measure", repeated.note)
                self.assertEqual(repeated.analysisSource, "local")
                if not exact:
                    self.assertEqual(repeated.confidence, "low")

    def test_unreadable_previous_measure_breaks_the_repeat_chain(self):
        bars = [ScoreMeasure(0, 1, 1, 1, (4, 4), chords=[PrintedChord("C", 0, 120)]),
                ScoreMeasure(1, 1, 1, 2, (4, 4)),
                ScoreMeasure(2, 1, 1, 3, (4, 4), repeat_markers=["%"])]
        result = self.analyze(bars)
        self.assertEqual(result.measures[0].chords, ["C"])
        for measure in result.measures[1:]:
            self.assertEqual(measure.chords, [])
            self.assertEqual(measure.chordBasis, "unreadable")
            self.assertEqual(measure.confidence, "low")
            self.assertTrue(measure.needsReview)
        self.assertIn("no preceding readable harmony", result.measures[2].note)

    def test_repeat_after_explicit_no_chord_repeats_no_chord_and_not_older_harmony(self):
        bars = [ScoreMeasure(0, 1, 1, 1, (4, 4), chords=[PrintedChord("C", 0, 120)]),
                ScoreMeasure(1, 1, 1, 2, (4, 4), no_chord=True),
                ScoreMeasure(2, 1, 1, 3, (4, 4), repeat_markers=["%"])]
        result = self.analyze(bars, mode="chord_symbols")
        self.assertEqual(result.measures[0].chords, ["C"])
        for measure in result.measures[1:]:
            self.assertTrue(measure.noChord)
            self.assertEqual(measure.chords, [])
            self.assertEqual(measure.chordBasis, "printed")
            self.assertEqual(measure.confidence, "high")
            self.assertFalse(measure.needsReview)


if __name__ == "__main__":
    unittest.main()
