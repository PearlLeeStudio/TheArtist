"""Loopback-only MIDI receiver page for Safari/Chrome, with no third-party packages.

The native helper reads CoreMIDI. A local top-level page forwards selected MIDI
messages to its verified TheArtist opener, avoiding HTTPS-to-HTTP mixed content.
No MIDI goes to the cloud, no output ports are opened, and nothing is recorded.
"""
from __future__ import annotations

import argparse
import base64
from collections import deque
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import struct
import subprocess
import threading
import uuid
from urllib.parse import parse_qs, urlsplit

ALLOWED_ORIGINS = {"https://artist.pearlleestudio.co",
                   "http://localhost:5173", "http://127.0.0.1:5173"}
PORT = 8768
MAX_MIDI_BYTES = 4 * 1024 * 1024
EXPORTS = Path(__file__).resolve().parent / '.build' / 'exports'


def open_midi(payload):
    """Accept a bounded Standard MIDI file, then use a fixed macOS app command."""
    target = payload.get('target')
    encoded = payload.get('midi')
    if target not in ('logic', 'ableton') or not isinstance(encoded, str):
        raise ValueError('Choose Logic or Ableton and provide a MIDI file.')
    if len(encoded) > (MAX_MIDI_BYTES + 2) // 3 * 4:
        raise ValueError('The MIDI file is too large.')
    try:
        data = base64.b64decode(encoded, validate=True)
    except ValueError:
        raise ValueError('The MIDI file is invalid.') from None
    if not 14 <= len(data) <= MAX_MIDI_BYTES or data[:8] != b'MThd\x00\x00\x00\x06':
        raise ValueError('The MIDI file is invalid.')
    format_id, tracks, division = struct.unpack('>HHH', data[8:14])
    if format_id not in (0, 1) or not 1 <= tracks <= 64 or division == 0:
        raise ValueError('The MIDI header is invalid.')
    cursor = 14
    for _ in range(tracks):
        if data[cursor:cursor + 4] != b'MTrk' or cursor + 8 > len(data):
            raise ValueError('The MIDI tracks are invalid.')
        length = int.from_bytes(data[cursor + 4:cursor + 8], 'big')
        cursor += 8 + length
        if cursor > len(data):
            raise ValueError('The MIDI track is incomplete.')
    if cursor != len(data):
        raise ValueError('The MIDI length is invalid.')
    if EXPORTS.is_symlink() or EXPORTS.parent.is_symlink():
        raise ValueError('The export folder must not be a symbolic link.')
    EXPORTS.mkdir(parents=True, exist_ok=True, mode=0o700)
    title = re.sub(r'[^\w -]', '_', str(payload.get('title', 'TheArtist')))[:80].strip() or 'TheArtist'
    path = EXPORTS / f'{title}-{uuid.uuid4().hex[:12]}.mid'
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as output:
        output.write(data)
    command = ['/usr/bin/open', '-b', 'com.apple.logic10', str(path)] if target == 'logic' else ['/usr/bin/open', '-R', str(path)]
    result = subprocess.run(command, capture_output=True, timeout=10, check=False, shell=False)
    if result.returncode:
        raise ValueError('The Mac could not open the MIDI. Check that Logic Pro is installed, or download MIDI instead.')
    return {'action': 'open' if target == 'logic' else 'reveal'}


def daw_page(origin, token):
    config = json.dumps({'origin': origin, 'token': token})
    return ('''<!doctype html><meta charset="utf-8"><title>TheArtist · Open in DAW</title>
<style>body{font:16px system-ui;padding:24px;line-height:1.5}</style>
<p id="status">Preparing the MIDI for your Mac…</p><script>
const config = ''' + config + ''';
const status = document.getElementById('status');
const post = message => window.opener?.postMessage({...message,protocol:'theartist-daw-v1',token:config.token},config.origin);
let received = false;
window.addEventListener('message', async event => {
  const value = event.data;
  if(received || event.origin !== config.origin || event.source !== window.opener ||
     value?.protocol !== 'theartist-daw-v1' || value?.token !== config.token || value.type !== 'open') return;
  received = true;
  try {
    const response = await fetch('/daw/open?origin='+encodeURIComponent(config.origin)+'&token='+config.token, {
      method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({target:value.target,title:value.title,midi:value.midi})
    });
    const result = await response.json();
    if(!response.ok) throw new Error(result.error || 'The Mac could not open the MIDI.');
    post({type:'result',action:result.action});
  } catch(error) {status.textContent=error.message;post({type:'result',error:error.message});}
});
if(window.opener) post({type:'ready'});
else status.textContent='Open this window from TheArtist Export menu.';
</script>''').encode()


def connection(query):
    values = parse_qs(query)
    origin = values.get("origin", [""])[0]
    token = values.get("token", [""])[0]
    if origin not in ALLOWED_ORIGINS or not re.fullmatch(r"[a-zA-Z0-9-]{20,80}", token):
        raise ValueError("Unrecognized app origin or connection token")
    return origin, token


def page(origin, token):
    config = json.dumps({"origin": origin, "token": token})
    return ("""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>TheArtist MIDI Bridge</title><style>body{font:16px system-ui;background:white;color:#222;padding:24px;line-height:1.5}h1{font-size:20px}</style>
<h1>TheArtist MIDI Bridge</h1><p id="status">Connecting to your keyboard…</p>
<p>Keep this window open while playing. Chords are entered only in the selected score slot.</p>
<p>No audio or MIDI is uploaded or recorded.</p><script>
const config = """ + config + """;
const status = document.getElementById('status');
if (!window.opener) { status.textContent = 'Open this connection from TheArtist Input → Enable MIDI.'; }
else {
  const post = message => window.opener.postMessage({...message, protocol:'theartist-midi-v1', token:config.token}, config.origin);
  const stream = new EventSource('/events?origin='+encodeURIComponent(config.origin)+'&token='+config.token);
  stream.onmessage = event => {
    const value = JSON.parse(event.data);
    if(value.type === 'state') {
      status.textContent = value.error || (value.devices.length ? 'Connected: '+value.devices.map(d=>d.name).join(', ') : 'No MIDI input found. Connect or power on your keyboard.');
      post({type:'state', connected:!value.error && value.devices.length>0, devices:value.devices.map(d=>d.name)});
    } else if(value.type === 'midi') post(value);
  };
  stream.onerror = () => { status.textContent = 'Connection lost. Restart MIDI Bridge, then reconnect in TheArtist.'; post({type:'state',connected:false}); };
  window.addEventListener('pagehide', () => { post({type:'state',connected:false}); stream.close(); });
}
</script>""").encode()


class Events:
    def __init__(self):
        self.condition = threading.Condition()
        self.events = deque(maxlen=1024)
        self.sequence = 0
        self.state = {"type": "state", "devices": [], "error": "Waiting for CoreMIDI…"}

    def publish(self, value):
        with self.condition:
            if value.get("type") == "state":
                self.state = value
            elif value.get("type") == "error":
                value = {"type": "state", "devices": [], "error": value["message"]}
                self.state = value
            self.sequence += 1
            self.events.append((self.sequence, value))
            self.condition.notify_all()


def handler(events):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass  # Never log connection tokens or played notes.

        def do_POST(self):
            url = urlsplit(self.path)
            if (self.headers.get('Host') != f'127.0.0.1:{PORT}' or url.path != '/daw/open'
                    or self.headers.get('Origin') != f'http://127.0.0.1:{PORT}'
                    or self.headers.get('Content-Type') != 'application/json'):
                self.send_error(403); return
            try:
                connection(url.query)
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= MAX_MIDI_BYTES * 2:
                    raise ValueError('The MIDI request is too large or empty.')
                self.connection.settimeout(15)
                payload = json.loads(self.rfile.read(size))
                if not isinstance(payload, dict):
                    raise ValueError('Invalid MIDI request.')
                result, status = open_midi(payload), 200
            except (ValueError, OSError, subprocess.SubprocessError) as error:
                result, status = {'error': str(error) if isinstance(error, ValueError) else 'The Mac could not complete the MIDI handoff.'}, 400
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())

        def do_GET(self):
            if self.headers.get("Host") != f"127.0.0.1:{PORT}":
                self.send_error(403); return
            url = urlsplit(self.path)
            if url.path == "/health":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                with events.condition:
                    self.wfile.write(json.dumps({"service": "theartist-midi-v1", "devices": len(events.state["devices"]),
                                                 "ready": not bool(events.state.get("error"))}).encode())
                return
            if url.path not in ("/bridge", "/events", "/daw"):
                self.send_error(404); return
            try:
                origin, token = connection(url.query)
            except ValueError:
                self.send_error(403); return
            # The event stream is same-origin only. No CORS grants, even to the app.
            if url.path == "/events" and self.headers.get("Sec-Fetch-Site") == "cross-site":
                self.send_error(403); return
            self.send_response(200)
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Type", "text/event-stream" if url.path == "/events" else "text/html; charset=utf-8")
            self.end_headers()
            if url.path == "/bridge":
                self.wfile.write(page(origin, token)); return
            if url.path == '/daw':
                self.wfile.write(daw_page(origin, token)); return
            try:
                with events.condition:
                    cursor, state = events.sequence, dict(events.state)
                self.wfile.write(b"data: " + json.dumps(state).encode() + b"\n\n"); self.wfile.flush()
                while True:
                    with events.condition:
                        events.condition.wait_for(lambda: events.sequence > cursor, timeout=10)
                        items = [(seq, value) for seq, value in events.events if seq > cursor]
                        if events.events and cursor < events.events[0][0] - 1:
                            # Missing a release must never leave held notes stuck.
                            items.insert(0, (cursor, {"type": "state", "devices": [], "error": "MIDI stream overflow; reconnect."}))
                    for cursor, value in items:
                        self.wfile.write(b"data: " + json.dumps(value).encode() + b"\n\n")
                    if not items:
                        self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--helper", type=Path, required=True)
    args = parser.parse_args()
    if not args.helper.is_absolute() or not args.helper.is_file():
        parser.error("Use the absolute path to the compiled CoreMidiInput helper")
    events = Events()
    server = ThreadingHTTPServer(("127.0.0.1", PORT), handler(events))
    server.daemon_threads = True
    process = subprocess.Popen([str(args.helper)], stdout=subprocess.PIPE, text=True)
    def read():
        try:
            for line in process.stdout:
                events.publish(json.loads(line))
        finally:
            events.publish({"type": "error", "message": "CoreMIDI input stopped. Restart the bridge."})
            # Let the login service restart both processes if the native helper exits.
            server.shutdown()
    threading.Thread(target=read, daemon=True).start()
    print(f"TheArtist MIDI Bridge ready at http://127.0.0.1:{PORT}. Open Input → Enable MIDI in Safari or Chrome.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        process.terminate()
        process.wait(timeout=5)
        server.server_close()


if __name__ == "__main__":
    main()
