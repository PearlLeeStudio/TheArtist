import importlib.util
import base64
import http.client
from pathlib import Path
import shutil
import types
import threading
import unittest
from unittest.mock import patch
import uuid


def load_bridge():
    path = Path(__file__).resolve().parents[2] / "tools" / "midi-bridge" / "bridge.py"
    spec = importlib.util.spec_from_file_location("local_midi_bridge", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MidiBridgeTest(unittest.TestCase):
    def setUp(self):
        self.bridge = load_bridge()

    def test_only_explicit_app_origins_with_bounded_nonce_open_a_connection(self):
        query = "origin=https%3A%2F%2Fartist.pearlleestudio.co&token=12345678-1234-1234-1234-123456789abc"
        self.assertEqual(self.bridge.connection(query)[0], "https://artist.pearlleestudio.co")
        for invalid in (query.replace("artist.pearlleestudio.co", "attacker.example"),
                        "origin=https%3A%2F%2Fartist.pearlleestudio.co&token=short"):
            with self.assertRaises(ValueError):
                self.bridge.connection(invalid)

    def test_helper_page_uses_same_origin_stream_and_exact_postmessage_target(self):
        page = self.bridge.page("https://artist.pearlleestudio.co", "nonce").decode()
        self.assertIn("new EventSource('/events?", page)
        self.assertIn("config.origin)", page)
        self.assertNotIn("postMessage('*'", page)
        self.assertIn("if (!window.opener)", page)

    def test_native_failure_replaces_ready_state_and_buffer_is_bounded(self):
        events = self.bridge.Events()
        events.publish({"type": "state", "devices": [{"id": "1", "name": "test"}]})
        events.publish({"type": "error", "message": "input stopped"})
        self.assertEqual(events.state["error"], "input stopped")
        self.assertEqual(events.state["devices"], [])
        for _ in range(1100):
            events.publish({"type": "midi", "data": [144, 60, 1], "port": "1"})
        self.assertEqual(len(events.events), 1024)

    def test_daw_page_only_accepts_its_exact_opener_origin_and_nonce(self):
        page = self.bridge.daw_page('https://artist.pearlleestudio.co', 'nonce').decode()
        self.assertIn('event.origin !== config.origin', page)
        self.assertIn('event.source !== window.opener', page)
        self.assertIn('value?.token !== config.token', page)
        self.assertIn("fetch('/daw/open?", page)

    def test_daw_handoff_writes_unique_midi_files_and_uses_fixed_app_arguments(self):
        folder = Path(__file__).resolve().parents[2] / 'tools' / 'midi-bridge' / '.build' / ('test-exports-' + uuid.uuid4().hex)
        midi = bytes.fromhex('4d546864000000060001000101e04d54726b0000000400ff2f00')
        payload = {'target': 'logic', 'title': '../../song;echo hi', 'midi': base64.b64encode(midi).decode()}
        try:
            with patch.object(self.bridge, 'EXPORTS', folder), patch.object(self.bridge.subprocess, 'run', return_value=types.SimpleNamespace(returncode=0)) as run:
                self.assertEqual(self.bridge.open_midi(payload), {'action': 'open'})
                self.bridge.open_midi(payload)
                files = list(folder.glob('*.mid'))
                self.assertEqual(len(files), 2)
                self.assertTrue(all(file.read_bytes() == midi for file in files))
                self.assertEqual(run.call_args.args[0][:3], ['/usr/bin/open', '-b', 'com.apple.logic10'])
                self.assertEqual(Path(run.call_args.args[0][3]).parent, folder)
                self.assertIs(run.call_args.kwargs['shell'], False)
        finally:
            if folder.exists(): shutil.rmtree(folder)

    def test_bad_midi_or_target_never_opens_a_mac_app(self):
        with patch.object(self.bridge.subprocess, 'run') as run:
            for payload in [
                {'target': 'terminal', 'midi': ''},
                {'target': 'logic', 'midi': 'not base64'},
                {'target': 'logic', 'midi': base64.b64encode(b'MThd' + bytes(10)).decode()},
                {'target': 'logic', 'midi': base64.b64encode(bytes.fromhex('4d546864000000060001000101e04d54726b00000008')).decode()},
            ]:
                with self.assertRaises(ValueError): self.bridge.open_midi(payload)
            run.assert_not_called()

    def test_web_origins_and_form_posts_cannot_invoke_the_local_daw_endpoint(self):
        server = self.bridge.ThreadingHTTPServer(('127.0.0.1', 0), self.bridge.handler(self.bridge.Events()))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with patch.object(self.bridge, 'open_midi') as launch:
                for origin, content_type in [('https://artist.pearlleestudio.co', 'application/json'),
                                             ('http://attacker.example', 'application/json'),
                                             ('http://127.0.0.1:8768', 'text/plain')]:
                    client = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=2)
                    client.request('POST', '/daw/open', '{}', {'Host':'127.0.0.1:8768', 'Origin':origin, 'Content-Type':content_type})
                    response = client.getresponse()
                    self.assertEqual(response.status, 403)
                    response.read(); client.close()
                launch.assert_not_called()
        finally:
            server.shutdown(); server.server_close(); thread.join()
