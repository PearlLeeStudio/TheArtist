"""Logic Scripter export: the script plays the MIDI export's harmony and bass, and the route serves it."""

import json
import unittest
from pathlib import Path
from unittest import mock
from urllib.parse import quote

from fastapi.testclient import TestClient

import app.main
from app.models.export import ExportSpec
from app.services import scripter_export
from app.services.scripter_export import SONG_MARKER, TEMPLATE, render_scripter, scripter_filename, scripter_song
from app.services.smf_export import PPQ, render_midi
from tests.test_export import decode, ons, spec

CLIENT = TestClient(app.main.app)
EXAMPLE_FILE = Path(__file__).resolve().parents[2] / "plugin" / "logic-scripter" / "example" / "theartist_chords.example.js"

VOICING = {"Cmaj7": [48, 52, 55, 59], "Am7": [45, 52, 55, 60], "Dm7": [50, 53, 57, 60], "G7": [43, 50, 53, 59],
           "C/E": [52, 55, 60, 64], "Fmaj7": [53, 57, 60, 64], "F": [53, 57, 60], "Em7": [52, 55, 59, 62],
           "C": [48, 52, 55, 60]}
# The committed example: eight pop bars with repeated, changing and slash chords and one empty slot
EXAMPLE = {"title": "TheArtist example", "bpm": 96, "key": "C major", "timeSignature": [4, 4], "genre": "pop",
           "measures": [{"chords": [{"symbol": s, "voicing": VOICING[s]} if s else None for s in bar]} for bar in (
               ("Cmaj7", "Cmaj7"), ("Am7", "Dm7"), ("G7", "G7"), ("Cmaj7", "C/E"),
               ("Fmaj7", "F"), ("Em7", "Am7"), ("Dm7", "G7"), ("C", None))],
           "sections": [{"label": "Verse", "startMeasure": 0, "length": 4},
                        {"label": "Chorus", "startMeasure": 4, "length": 4}],
           "play": {"harmonyOverride": None, "harmonyRhythmId": None, "bassEnabled": True, "drumMode": "auto",
                    "drumPatternId": None, "masterVolume": 1.0, "harmonyVolume": 1.0, "bassVolume": 1.0,
                    "drumVolume": 0.6}}


def song_of(script: str) -> dict:
    start = script.index(SONG_MARKER) + len(SONG_MARKER)
    return json.loads(script[start:script.index(";\n", start)])


def strikes(song: dict, part: str) -> list[tuple[int, int]]:
    return sorted((round(note["start"] * PPQ), note["pitch"]) for note in song["notes"] if note["part"] == part)


class ScripterExportTest(unittest.TestCase):
    def test_the_script_strikes_what_the_midi_export_writes(self):
        swung = spec(genre="jazz", play={**spec()["play"], "harmonyRhythmId": "charleston"})
        for body in (spec(), swung, EXAMPLE, spec(timeSignature=[6, 8])):
            with self.subTest(genre=body["genre"], meter=body["timeSignature"]):
                export = ExportSpec.model_validate(body)
                midi = decode(render_midi(export))
                harmony, bass = (next(t for t in midi.tracks if t.name.startswith(name)) for name in ("Harmony", "Bass"))
                song = song_of(render_scripter(export))
                self.assertEqual(strikes(song, "harmony"), sorted((tick, note) for tick, note, _ in ons(harmony)))
                self.assertEqual(strikes(song, "bass"), sorted((tick, note) for tick, note, _ in ons(bass)))
                self.assertNotIn("drums", {note["part"] for note in song["notes"]})

    def test_song_fields_place_bars_and_slots_on_the_quarter_note_line(self):
        song = scripter_song(ExportSpec.model_validate(spec(timeSignature=[6, 8])))
        self.assertEqual((song["quartersPerBar"], song["lengthQuarters"]), (3.0, 6.0))  # two bars of 6/8
        self.assertEqual([(c["bar"], c["slot"], c["symbol"], c["startQuarter"]) for c in song["chords"]],
                         [(0, 0, "F/C", 0.0), (0, 1, "G7", 1.5), (1, 0, "C", 3.0)])  # the empty slot is left out
        self.assertEqual((song["title"], song["key"], song["bpm"], song["timeSignature"]),
                         ("테스트 악보", "F major", 120, [6, 8]))
        # The song form the plugin plays from, in bar order
        self.assertEqual(song["sections"], [{"label": "Verse", "startBar": 0, "bars": 1},
                                            {"label": "후렴", "startBar": 1, "bars": 1}])
        reordered = spec(sections=list(reversed(spec()["sections"])))
        self.assertEqual([s["label"] for s in scripter_song(ExportSpec.model_validate(reordered))["sections"]],
                         ["Verse", "후렴"])
        bass_off = scripter_song(ExportSpec.model_validate(spec(play={**spec()["play"], "bassEnabled": False})))
        self.assertEqual({note["part"] for note in bass_off["notes"]}, {"harmony"})

    def test_only_the_placeholder_object_is_replaced(self):
        template = TEMPLATE.read_text(encoding="utf-8")
        script = render_scripter(ExportSpec.model_validate(spec(title="line break")))
        self.assertEqual(script.count(SONG_MARKER), 1)
        before, after = template.split(SONG_MARKER)
        self.assertTrue(script.startswith(before))
        self.assertTrue(script.endswith(after[after.index(";\n"):]))
        self.assertIn("line\\u2028break", script)  # escaped for older JavaScript string literals
        self.assertEqual(song_of(script)["title"], "line break")

    def test_the_committed_example_matches_the_generator(self):
        self.assertEqual(EXAMPLE_FILE.read_text(encoding="utf-8"), render_scripter(ExportSpec.model_validate(EXAMPLE)))

    def test_route_serves_a_named_javascript_file(self):
        response = CLIENT.post("/api/export/scripter", json=spec())
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.headers["content-type"], "text/javascript; charset=utf-8")
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertEqual(scripter_filename("테스트 악보"), "테스트 악보.theartist.js")
        self.assertIn(f"filename*=UTF-8''{quote('테스트 악보.theartist.js', safe='')}",
                      response.headers["content-disposition"])
        self.assertEqual(song_of(response.text)["title"], "테스트 악보")

    def test_route_errors_are_readable(self):
        bad = CLIENT.post("/api/export/scripter", json=spec(bpm=5))
        self.assertEqual(bad.status_code, 422)
        self.assertTrue(bad.json()["detail"].startswith("Invalid export data at body.bpm"))
        with mock.patch.object(scripter_export, "MAX_SCRIPT_BYTES", 1000):
            too_big = CLIENT.post("/api/export/scripter", json=spec())
        self.assertEqual((too_big.status_code, too_big.json()["detail"]),
                         (413, "The Scripter script exceeds the 4 MiB export limit."))


if __name__ == "__main__":
    unittest.main()
