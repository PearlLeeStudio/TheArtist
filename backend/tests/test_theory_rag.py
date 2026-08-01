"""Corpus integrity, actual-hit attribution, and reviewed-HTML selection."""
import copy
import hashlib
import json
import sys
import unittest
from pathlib import Path
from unittest import mock
from app.services import theory_rag as rag
from tests.test_inference_rag_gate import passage

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ai"))
from rag.omt import parse_page, prepare, download

TEXT = "A synthetic fixture discusses a dominant cadence and its tonic resolution without quoting any book."
CHECKSUM = hashlib.sha256(TEXT.encode()).hexdigest()
URL = "https://openmusictheory.github.io/cadenceTypes.html"
MANIFEST = {"edition": "fixture", "authors": ["Fixture author"], "editors": ["Fixture editor"],
    "copyright": "Fixture notice", "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
    "pages": [{"url": URL, "title": "Fixture chapter", "paragraphs": [CHECKSUM], "sections": {CHECKSUM: "Fixture section"}}]}
CORPUS = {"version": 1, "source": "omt", "passages": [{"source": "omt", "id": CHECKSUM,
    "text": TEXT, "url": URL, "section": "untrusted section", "authors": ["wrong author"]}]}


class CorpusTest(unittest.TestCase):
    def load(self, corpus):
        with mock.patch.object(rag, "_read", return_value=json.dumps(corpus)), mock.patch.object(rag, "MANIFEST_PATH") as manifest:
            manifest.read_text.return_value = json.dumps(MANIFEST)
            return rag.load_omt("synthetic-fixture")

    def test_attribution_and_section_come_from_manifest(self):
        result = self.load(CORPUS)[0]
        self.assertEqual(result.citation.authors, ["Fixture author"])
        self.assertEqual(result.citation.editors, ["Fixture editor"])
        self.assertEqual(result.section, "Fixture section")

    def test_changed_text_source_url_or_identity_cannot_pass_as_omt(self):
        for field, value in (("text", "different book text"), ("source", "berklee"), ("url", "https://example.org/"), ("id", "wrong")):
            with self.subTest(field=field):
                corpus = copy.deepcopy(CORPUS)
                corpus["passages"][0][field] = value
                with self.assertRaises(ValueError):
                    self.load(corpus)

    def test_missing_and_duplicate_passages_are_rejected(self):
        for items in ([], CORPUS["passages"] * 2):
            with self.subTest(items=len(items)), self.assertRaises(ValueError):
                self.load({**CORPUS, "passages": items})

    def test_malformed_corpus_shapes_raise_validation_errors(self):
        for corpus in ([], None, {**CORPUS, "passages": {}}, {**CORPUS, "passages": [None]}, {**CORPUS, "passages": [{"text": 12}]}):
            with self.subTest(corpus=corpus), self.assertRaises(ValueError):
                self.load(corpus)

    def test_actual_hit_supplies_excerpt_and_citation_together(self):
        retriever = rag.TheoryRetriever([passage("berklee", "passing neighbor bass"), passage("omt", "dominant tonic resolution authentic cadence applied tonicization harmonic progression")])
        note = retriever.explain("G7", "Cmaj")
        self.assertEqual(note.citation.source, "omt")
        self.assertEqual(note.citation.passage_id, "omt")
        self.assertTrue(note.explanation.startswith("dominant tonic"))
        self.assertEqual(note.kind, "source_excerpt")

    def test_unrelated_query_returns_no_hit(self):
        self.assertEqual(rag.TheoryRetriever([passage("omt", TEXT)]).query("unrelatedxyz"), [])

    def test_shortened_excerpt_is_labeled_and_bounded(self):
        note = rag.TheoryRetriever([passage("omt", "dominant tonic resolution cadence " * 40)]).explain("G7", "Cmaj")
        self.assertLessEqual(len(note.explanation), 400)
        self.assertTrue(note.explanation.endswith("…"))
        self.assertIn("shortened", note.citation.changes)

    def test_inversion_and_parallel_mode_queries(self):
        self.assertIn("change bass", rag.transition_query("Cmaj", "Cmaj/E"))
        self.assertIn("modal mixture", rag.transition_query("Cmaj", "Cm"))


class SetupTest(unittest.TestCase):
    def page(self, body):
        return '<article>' + body + '</article><a href="https://creativecommons.org/licenses/by-sa/4.0/">license</a>'

    def test_media_links_quotes_and_tables_are_not_selected(self):
        html = self.page(f'<h2>Section</h2><p>{TEXT}</p><blockquote><p>{TEXT}</p></blockquote>'
                         f'<table><tr><td><p>{TEXT}</p></td></tr></table><p>{TEXT}<img src="score.png"></p>'
                         f'<p>{TEXT}<iframe src="song"></iframe></p><p><a href="https://elsewhere.example/">{TEXT}</a></p>')
        blocks = parse_page(html)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["sha256"], CHECKSUM)
        self.assertEqual(blocks[0]["section"], "Section")

    def test_unreviewed_paragraph_is_never_added(self):
        corpus = prepare(MANIFEST, lambda _: self.page(f'<p>{TEXT}</p><p>Another synthetic paragraph with enough words to pass extraction but no review approval.</p>'))
        self.assertEqual(len(corpus["passages"]), 1)
        self.assertEqual(corpus["passages"][0]["id"], CHECKSUM)

    def test_changed_source_or_missing_license_fails(self):
        for html in (self.page('<p>A changed paragraph is not approved even if it has more than twelve words.</p>'), f'<article><p>{TEXT}</p></article>'):
            with self.subTest(html=html), self.assertRaises(ValueError):
                prepare(MANIFEST, lambda _: html)

    def test_non_publisher_url_is_rejected_before_network(self):
        with mock.patch("rag.omt.urlopen", side_effect=AssertionError("No network")), self.assertRaises(ValueError):
            download("https://example.org/book.html")
