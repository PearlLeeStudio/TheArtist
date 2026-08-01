"""Uploaded scores are decoded per request and leave no process or disk cache."""

import base64
import gc
import unittest
import weakref
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routes.score_import import router
from app.services import score_pdf
from tests.vector_score_fixture import build_vector_score_pdf


class ScoreImportRetentionTests(unittest.TestCase):
    def test_repeated_pdf_is_decoded_again_with_independent_results_and_limits(self):
        data = build_vector_score_pdf()
        with mock.patch.object(score_pdf.pdfplumber, "open", wraps=score_pdf.pdfplumber.open) as decode:
            first = score_pdf.parse_vector_score(data)
            second = score_pdf.parse_vector_score(data)
            self.assertEqual(second, first)
            self.assertIsNot(second, first)
            self.assertEqual(decode.call_count, 2)
            first.measures.clear()
            self.assertTrue(second.measures)
            self.assertTrue(score_pdf.parse_vector_score(data).measures)
            self.assertEqual(decode.call_count, 3)
            with self.assertRaises(score_pdf.ScorePDFError):
                score_pdf.parse_vector_score(data, max_pages=0)
            with self.assertRaises(score_pdf.ScorePDFError):
                score_pdf.parse_vector_score(b"bad")

    def test_repeated_http_import_does_not_retain_decoded_notation_or_write_source_files(self):
        data = build_vector_score_pdf()
        request = {"filename": "original.pdf", "mimeType": "application/pdf",
                   "mode": "infer_harmony", "dataBase64": base64.b64encode(data).decode()}
        app = FastAPI()
        app.include_router(router, prefix="/api")
        parsed_refs = []
        parse_pages = score_pdf._parse_pages

        def track_parsed(*args, **kwargs):
            parsed = parse_pages(*args, **kwargs)
            self.assertTrue(parsed.measures[0].notes)
            parsed_refs.append(weakref.ref(parsed))
            return parsed

        # The upload and decoded notation stay in memory. No parser/import file
        # handle is needed, even for the second identical request.
        with TestClient(app) as client, \
                mock.patch.object(score_pdf.pdfplumber, "open", wraps=score_pdf.pdfplumber.open) as decode, \
                mock.patch.object(score_pdf, "_parse_pages", side_effect=track_parsed), \
                mock.patch("builtins.open", side_effect=AssertionError("Score import must not open a disk file")), \
                mock.patch("io.open", side_effect=AssertionError("Score import must not open a disk file")):
            first = client.post("/api/import/score", json=request)
            second = client.post("/api/import/score", json=request)
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(second.status_code, 200, second.text)
        self.assertEqual(first.json(), second.json())
        self.assertEqual(first.json()["measures"][0]["chords"], ["C"])
        self.assertEqual(decode.call_count, 2)
        self.assertIn("no-store", first.headers["cache-control"])
        self.assertEqual(len(parsed_refs), 2)
        gc.collect()
        self.assertTrue(all(reference() is None for reference in parsed_refs),
                        "Decoded notes must not survive their import requests")

    def test_failed_pdf_decoding_is_attempted_again(self):
        with mock.patch.object(score_pdf.pdfplumber, "open", side_effect=ValueError("bad")) as decode:
            for _ in range(2):
                with self.assertRaises(score_pdf.ScorePDFError):
                    score_pdf.parse_vector_score(b"%PDF-invalid")
            self.assertEqual(decode.call_count, 2)
