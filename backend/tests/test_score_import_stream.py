import base64
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient
from fastapi import FastAPI
from app.routes.score_import import router
from app.services.score_import import ScoreInputError


class ScoreImportStreamTest(unittest.TestCase):
    def request(self, run):
        app = FastAPI()
        app.include_router(router, prefix='/api')
        with patch('app.routes.score_import.import_score', side_effect=run):
            with TestClient(app) as client:
                response = client.post('/api/import/score/stream', json={
                    'filename': 'score.pdf', 'mimeType': 'application/pdf',
                    'dataBase64': base64.b64encode(b'%PDF-1.7\n%%EOF').decode(),
                })
        self.assertEqual(response.status_code, 200)
        self.assertIn('text/event-stream', response.headers['content-type'])
        self.assertIn('no-store', response.headers['cache-control'])
        return [(block.splitlines()[0][7:], json.loads(block.split('data: ', 1)[1]))
                for block in response.text.split('\n\n') if 'data: ' in block]

    def test_actual_stages_precede_one_complete_result(self):
        def run(req, *, on_progress):
            on_progress('Reading notes')
            on_progress('Preparing the score')
            return SimpleNamespace(model_dump=lambda **_: {'complete': True, 'measures': [1, 2, 3]})
        frames = self.request(run)
        self.assertEqual([event for event, _ in frames], ['progress', 'progress', 'progress', 'result'])
        self.assertEqual(frames[1][1]['label'], 'Reading notes')
        self.assertEqual(frames[-1][1]['measures'], [1, 2, 3])

    def test_invalid_input_is_terminal_error_not_a_successful_partial_score(self):
        def run(*args, **kwargs):
            raise ScoreInputError('Unsupported score encoding', 422)
        frames = self.request(run)
        self.assertEqual(frames[-1], ('error', {'detail': 'Unsupported score encoding', 'status': 422}))
        self.assertNotIn('result', [event for event, _ in frames])

    def test_reader_error_keeps_its_status_and_actionable_detail(self):
        def run(*args, **kwargs):
            raise ScoreInputError('Score analysis reached its time limit', 504)
        frames = self.request(run)
        self.assertEqual(frames[-1][0], 'error')
        self.assertEqual(frames[-1][1]['status'], 504)
        self.assertIn('time limit', frames[-1][1]['detail'])

    def test_json_reader_errors_and_request_validation_are_not_cacheable(self):
        app = FastAPI()
        app.include_router(router, prefix='/api')
        with patch('app.routes.score_import.import_score', side_effect=ScoreInputError('Unreadable score', 422)), \
                TestClient(app) as client:
            reader_error = client.post('/api/import/score', json={
                'filename': 'score.pdf', 'mimeType': 'application/pdf',
                'dataBase64': base64.b64encode(b'%PDF-1.7\n%%EOF').decode(),
            })
            self.assertEqual(reader_error.status_code, 422)
            self.assertEqual(reader_error.json(), {'detail': 'Unreadable score'})
            self.assertIn('no-store', reader_error.headers['cache-control'])
            for path in ('/api/import/score', '/api/import/score/stream'):
                invalid_request = client.post(path, json={'filename': 'score.pdf'})
                self.assertEqual(invalid_request.status_code, 422)
                self.assertIn('no-store', invalid_request.headers['cache-control'])
