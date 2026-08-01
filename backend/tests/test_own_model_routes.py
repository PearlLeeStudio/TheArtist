"""The public HTTP paths of Reharmonize, score import, Song Form and Second ear run on own models."""
import base64
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app
from app.services import inference
from app.models.review import ReviewRequest
from tests.test_planner import FakeComposer
from tests.test_reharmonize import FakeScorer, request as reharm_request
from tests.test_review import body as review_body


class OwnModelRoutesTest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        # The hosted server's setting: these routes answer the same there
        env = patch.dict(os.environ, {'THEARTIST_PUBLIC_HOST': '1'})
        env.start()
        self.addCleanup(env.stop)

    def test_reharm_returns_three_alternatives_scored_by_the_own_model(self):
        req = reharm_request()
        original = req.model_dump_json()
        with patch('app.services.reharmonize.ContextScorer', return_value=FakeScorer()):
            result = self.client.post('/api/reharmonize', json=req.model_dump(mode='json'))
        self.assertEqual(result.status_code, 200, result.text)
        data = result.json()
        self.assertEqual(set(data), {'candidates', 'modelUsed', 'contextMeasures', 'modelApproximations'})
        self.assertEqual((data['modelUsed'], data['contextMeasures'], data['modelApproximations']),
                         ('test-checkpoint', [0, 1, 2], []))
        self.assertEqual(len(data['candidates']), 3)
        self.assertEqual(req.model_dump_json(), original)

    def test_missing_own_model_is_an_error_without_a_synthetic_success(self):
        with patch.object(inference.ModelRegistry, 'get', return_value=SimpleNamespace(load_model=lambda _: None)):
            response = self.client.post('/api/reharmonize', json=reharm_request().model_dump(mode='json'))
        self.assertEqual(response.status_code, 422)
        self.assertNotIn('candidates', response.json())

    def test_a_real_vector_pdf_is_read_on_this_server_the_same_way_each_time(self):
        encoded = base64.b64encode((Path(__file__).parent / 'fixtures' / 'vector-score-c.pdf').read_bytes()).decode()
        outputs = []
        for _ in range(2):
            response = self.client.post('/api/import/score', json={
                'filename': 'score.pdf', 'mimeType': 'application/pdf', 'dataBase64': encoded,
                'mode': 'infer_harmony'})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertIn('no-store', response.headers['cache-control'])
            data = response.json()
            self.assertEqual(data['analysisSource'], 'local')
            self.assertNotIn('requestId', data)
            outputs.append(data)
        self.assertTrue(all(output == outputs[0] for output in outputs))
        self.assertGreater(len(outputs[0]['measures']), 0)

    def test_images_are_rejected(self):
        encoded = base64.b64encode(b'\x89PNG\r\n\x1a\nexample').decode()
        response = self.client.post('/api/import/score', json={
            'filename': 'score.png', 'mimeType': 'image/png', 'dataBase64': encoded})
        self.assertEqual(response.status_code, 422, response.text)
        self.assertIn('vector PDF', response.json()['detail'])

    def test_full_form_uses_own_chords(self):
        composer = FakeComposer()
        body = {'genre': 'pop', 'targetBars': 16, 'modelKey': 'ft_f1'}
        with patch('app.services.planner.compose_section', side_effect=composer):
            response = self.client.post('/api/plan/song', json=body)
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertFalse({'planner', 'chordSource', 'judge', 'calls', 'spendUsd', 'reservedUsd', 'metaMs'} & set(data))
        self.assertEqual(sum(section['bars'] for section in data['sections']), 16)
        self.assertTrue(all(req.modelKey == 'ft_f1' for req in composer.requests))

    def test_song_form_stream_finishes_once_without_judge_or_paid_events(self):
        with patch('app.services.planner.compose_section', side_effect=FakeComposer()):
            response = self.client.post('/api/plan/song/stream', json={'genre': 'jazz', 'targetBars': 16})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.text.count('event: complete'), 1)
        self.assertNotIn('event: judge', response.text)
        self.assertNotIn('event: error', response.text)

    def test_second_ear_ranks_the_selected_model_without_mutating_chords(self):
        req = ReviewRequest(**review_body(), modelKey='ft_f1', timeSignature=(3, 4))
        before = req.model_dump_json()
        registry = SimpleNamespace(load_model=lambda _: object(), tokenizer=object())
        with patch.object(inference.ModelRegistry, 'get', return_value=registry), \
             patch.object(inference, 'build_prompt', return_value=[1]) as prompt, \
             patch('app.services.chord_context.fitting_context', side_effect=lambda *args: (args[-2], None)), \
             patch('app.services.chord_context.context_scores', return_value=[-2.5, -1.2, -3.0]):
            response = self.client.post('/api/review/candidates', json=req.model_dump(mode='json'))
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertEqual(set(data), {'choice', 'reasons', 'facts', 'model', 'latencyMs'})
        self.assertEqual((data['choice'], data['model']), ('B', inference.MODEL_LABELS.get('ft_f1', 'ft_f1')))
        self.assertEqual([row['modelScore'] for row in data['facts']], [-2.5, -1.2, -3.0])
        self.assertEqual(data['reasons'], [{
            'fact': 'modelScore=-1.2', 'factName': 'modelScore', 'value': -1.2,
            'claim': 'Highest conditional chord likelihood among these candidates; not a quality rating.',
            'confirmed': True, 'analyzerValue': -1.2}])
        self.assertEqual(prompt.call_args.args[3], (3, 4))
        self.assertEqual(req.model_dump_json(), before)

    def test_discovery_reports_own_model_inference_only(self):
        data = self.client.get('/api/capabilities').json()['data']
        self.assertNotIn('meta', data)
        self.assertEqual(data['inference'], {'provider': 'theartist', 'externalModels': False,
                                             'scoreFormats': ['vector-pdf']})
