"""Score import contract through an in-process API, without network or files."""

import base64
import unittest
from unittest import mock

from fastapi.testclient import TestClient

import app.main
from app.models.score_import import ScoreImportResponse

CLIENT = TestClient(app.main.app)
ROUTE = "/api/import/score"
PDF = b"%PDF-1.7\n(score with Cmaj7, Dm7, G7)\n%%EOF"


def body(raw=PDF, **overrides):
    return {"filename": "score.pdf", "mimeType": "application/pdf",
            "dataBase64": base64.b64encode(raw).decode("ascii"), **overrides}


def read_draft():
    """What the reader returns for a four-bar chart: one, two and three printed symbols, then an unread bar."""
    measures = [
        {"chords": ["Cmaj7"], "sourceLabel": "p1 m1", "needsReview": False, "note": ""},
        {"chords": ["Dm7", "G7"], "sourceLabel": "p1 m2", "needsReview": False, "note": ""},
        {"chords": ["Am7", "D7", "Gmaj7"], "sourceLabel": "p1 m3", "needsReview": True, "note": "3 symbols"},
        {"chords": [], "sourceLabel": "p1 m4", "needsReview": True, "note": "No chord symbol"},
    ]
    for measure in measures:
        measure.update(chordBasis="printed" if measure["chords"] else "unreadable",
                       confidence="high" if measure["chords"] else "low", notes=[], writtenNotes=[],
                       bassNotes=[], noChord=False, timeSignature=None)
    return ScoreImportResponse(
        title="A score", key="C major", timeSignature=[4, 4], measures=measures,
        sections=[{"section": "verse", "label": "Verse", "startMeasure": 0, "length": 4,
                   "basis": "explicit", "note": "Printed heading"}],
        warnings=[], mode="chord_symbols", analysisSource="local", model=None)


class ScoreImportTest(unittest.TestCase):
    def test_invalid_uploads_never_reach_reader_or_model(self):
        cases = [body(raw=b"wrong type"), body(dataBase64="!not-base64"),
                 body(filename="../score.pdf"), body(filename="score.png"), body(raw=b"")]
        with mock.patch("app.services.score_analysis.analyze_local_score") as read:
            for data in cases:
                response = CLIENT.post(ROUTE, json=data)
                self.assertIn(response.status_code, (400, 415, 422), response.text)
                self.assertIn("no-store", response.headers["cache-control"])
            read.assert_not_called()

    def test_unknown_modes_and_fields_are_rejected(self):
        # The request model forbids extra fields, so the removed reader choice is refused like any unknown field.
        for fields in ({"mode": "invent"}, {"engine": "local"}):
            response = CLIENT.post(ROUTE, json=body(**fields))
            self.assertEqual(response.status_code, 422)

    def test_printed_chords_and_uncertainty_survive_local_import(self):
        response = read_draft()
        with mock.patch("app.services.score_analysis.analyze_local_score", return_value=response) as reader:
            result = CLIENT.post(ROUTE, json=body())
        self.assertEqual(result.status_code, 200, result.text)
        data, expected = result.json(), response.model_dump(mode="json")
        self.assertEqual(data["measures"], expected["measures"])
        self.assertEqual(data["sections"], expected["sections"])
        self.assertEqual(set(data), {"title", "key", "timeSignature", "measures", "sections", "warnings",
                                     "model", "mode", "analysisSource"})
        self.assertEqual((data["analysisSource"], data["model"]), ("local", None))
        self.assertTrue(all(bar["analysisSource"] == "local" and "requestId" not in bar for bar in data["measures"]))
        reader.assert_called_once()
