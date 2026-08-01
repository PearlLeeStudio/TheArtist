"""MIDI round trips and mocked macOS handoff; no DAW or network is invoked."""

import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import mido
from fastapi.testclient import TestClient

import app.main
from app.models.export import ExportSpec
from app.services import daw_launcher, smf_export
from app.services.smf_export import ExportError, Note, _note_messages, render_midi

CLIENT = TestClient(app.main.app)


def spec(**overrides):
    # The first harmony explicitly voices F/C with C as its lowest sounding note.
    return {"title": "테스트 악보", "bpm": 120, "key": "F major", "timeSignature": [4, 4], "genre": "pop",
            "measures": [
                {"chords": [{"symbol": "F/C", "voicing": [60, 65, 69]},
                            {"symbol": "G7", "voicing": [55, 59, 62, 65]}]},
                {"chords": [{"symbol": "C", "voicing": [60, 64, 67]}, None]},
            ],
            "sections": [{"label": "Verse", "startMeasure": 0, "length": 1},
                         {"label": "후렴", "startMeasure": 1, "length": 1}],
            "play": {"harmonyOverride": None, "harmonyRhythmId": "block", "bassEnabled": True,
                     "drumMode": "auto", "drumPatternId": None,
                     "masterVolume": 1.0, "harmonyVolume": 1.0, "bassVolume": 1.0, "drumVolume": 0.6},
            **overrides}


def decode(data):
    return mido.MidiFile(file=io.BytesIO(data), charset="utf8")


def absolute(track):
    tick = 0
    for message in track:
        tick += message.time
        yield tick, message


def ons(track):
    return [(tick, message.note, message.velocity) for tick, message in absolute(track)
            if message.type == "note_on" and message.velocity > 0]


class MidiExportTest(unittest.TestCase):
    def render(self, body):
        return decode(render_midi(ExportSpec.model_validate(body)))

    def test_download_has_three_named_music_tracks_and_metadata(self):
        response = CLIENT.post("/api/export/midi", json=spec())
        self.assertEqual(response.status_code, 200, response.text if response.status_code != 200 else "")
        self.assertEqual(response.headers["content-type"], "audio/midi")
        self.assertIn("attachment", response.headers["content-disposition"])
        self.assertIn("filename*=UTF-8''", response.headers["content-disposition"])
        self.assertEqual(response.headers["cache-control"], "no-store")
        midi = decode(response.content)
        self.assertEqual((midi.type, midi.ticks_per_beat, len(midi.tracks)), (1, 480, 4))
        self.assertEqual([track.name for track in midi.tracks],
                         ["테스트 악보", "Harmony – acoustic grand piano", "Bass", "Drums"])
        meta = midi.tracks[0]
        self.assertEqual(next(m.tempo for m in meta if m.type == "set_tempo"), 500_000)
        self.assertEqual(next(m.key for m in meta if m.type == "key_signature"), "F")
        self.assertEqual([(t, m.text) for t, m in absolute(meta) if m.type == "marker"], [(0, "Verse"), (1920, "후렴")])
        for channel, track in zip((0, 1, 9), midi.tracks[1:]):
            self.assertEqual({message.channel for message in track if not message.is_meta}, {channel})
            self.assertEqual(sum(message.time for message in track), 3840)
        self.assertEqual({note for tick, note, _ in ons(midi.tracks[1]) if tick == 0}, {60, 65, 69})
        self.assertEqual(ons(midi.tracks[2])[0][1], 36)  # F/C bass is C2, not F2.

    def test_eighth_note_meter_has_correct_half_bar_and_marker_ticks(self):
        body = spec(timeSignature=[6, 8], genre="country")
        body["play"].update(drumMode="manual", drumPatternId=None)
        midi = self.render(body)
        time_signature = next(m for m in midi.tracks[0] if m.type == "time_signature")
        self.assertEqual((time_signature.numerator, time_signature.denominator), (6, 8))
        # Independent arithmetic: 6 eighths = 3 quarter beats = 1440 ticks; half = 720.
        self.assertEqual(sorted({tick for tick, _, _ in ons(midi.tracks[1])}), [0, 720, 1440])
        self.assertEqual(sorted({tick for tick, _, _ in ons(midi.tracks[2])}), [0, 720, 1440])
        self.assertEqual([(t, m.text) for t, m in absolute(midi.tracks[0]) if m.type == "marker"], [(0, "Verse"), (1440, "후렴")])
        self.assertEqual([sum(m.time for m in track) for track in midi.tracks], [2880] * 3)
        self.assertAlmostEqual(midi.length, 3.0)

    def test_every_supported_meter_uses_the_same_written_grid_scale(self):
        for numerator, denominator in ((2, 4), (3, 4), (4, 4), (5, 4), (6, 8), (7, 8), (1, 32)):
            with self.subTest(meter=(numerator, denominator)):
                body = spec(timeSignature=[numerator, denominator], genre="country")
                body["play"].update(drumMode="manual", drumPatternId=None)
                midi = self.render(body)
                half_ticks = round(numerator * 4 / denominator * 480 / 2)
                self.assertEqual(sorted({tick for tick, _, _ in ons(midi.tracks[1])}), [0, half_ticks, half_ticks * 2])

    def test_half_and_charleston_do_not_drop_a_different_second_chord(self):
        for rhythm, expected in (("half", [0, 960, 1920]), ("charleston", [0, 720, 960, 1920, 2640])):
            with self.subTest(rhythm=rhythm):
                body = spec()
                body["play"]["harmonyRhythmId"] = rhythm
                midi = self.render(body)
                self.assertEqual(sorted({tick for tick, _, _ in ons(midi.tracks[1])}), expected)
        body = spec(measures=[{"chords": [{"symbol": "C", "voicing": [60, 64, 67]}] * 2}], sections=[])
        body["play"]["harmonyRhythmId"] = "half"
        self.assertEqual(sorted({tick for tick, _, _ in ons(self.render(body).tracks[1])}), [0])

    def test_swing_delays_offbeat_harmony_bass_and_drums_in_meter_units(self):
        body = spec(timeSignature=[6, 8], genre="funk")
        body["play"].update(harmonyRhythmId="eighth_pulse", drumMode="manual", drumPatternId=4)
        midi = self.render(body)
        # Notated .5 + jazz swing 1/6, scaled to quarters by 4/8, at 480 ppq = 160 ticks.
        for track in midi.tracks[1:]:
            self.assertIn(160, {tick for tick, _, _ in ons(track)})
            self.assertNotIn(120, {tick for tick, _, _ in ons(track)})
        # Releases remain bounded and every swung onset still has exactly one later note-off.
        self.assert_valid_note_pairs(midi)

    def assert_valid_note_pairs(self, midi):
        for track in midi.tracks[1:]:
            active = {}
            for tick, message in absolute(track):
                self.assertGreaterEqual(message.time, 0)
                if message.type == "note_on" and message.velocity:
                    self.assertNotIn(message.note, active)
                    active[message.note] = tick
                    self.assertTrue(1 <= message.velocity <= 127)
                elif message.type == "note_off":
                    self.assertIn(message.note, active)
                    self.assertGreater(tick, active.pop(message.note))
            self.assertEqual(active, {})

    def test_same_pitch_retriggers_end_before_the_new_on_and_duplicate_voices_merge(self):
        notes = [Note(0, 1, 60, 0.7), Note(0, 1, 60, 0.7), Note(0.5, 1.5, 60, 0.7)]
        messages = _note_messages(notes, 0, 960)
        self.assertEqual([(tick, message.type) for tick, message in absolute(messages)],
                         [(0, "note_on"), (240, "note_off"), (240, "note_on"), (720, "note_off")])
        for genre in ("funk", "jazz", "blues", "bossa", "country"):
            body = spec(genre=genre)
            body["play"]["harmonyRhythmId"] = None
            self.assert_valid_note_pairs(self.render(body))

    def test_bass_ends_at_chord_boundaries_and_swing_keeps_original_release(self):
        body = spec(timeSignature=[3, 4])
        midi = self.render(body)
        self.assertIn((720, 36), [(tick, message.note) for tick, message in absolute(midi.tracks[2])
                                 if message.type == "note_off"])
        body = spec(genre="funk")
        body["play"].update(drumMode="manual", drumPatternId=4)
        bass = self.render(body).tracks[2]
        self.assertIn((320, 36, 76), ons(bass))
        self.assertIn((672, 36), [(tick, message.note) for tick, message in absolute(bass)
                                 if message.type == "note_off"])

    def test_invalid_generated_note_data_is_refused(self):
        invalid = [Note(-1, 1, 60, 0.5), Note(0, 0, 60, 0.5), Note(0, 1, 128, 0.5),
                   Note(0, 1, True, 0.5), Note(True, 2, 60, 0.5), Note(0, 1, 60, True),
                   Note(0, 1, 60, 0), Note(0, 1, 60, float("nan")),
                   Note(float("inf"), 1, 60, 0.5), Note(5, 6, 60, 0.5)]
        for note in invalid:
            with self.subTest(note=note), self.assertRaises(ExportError):
                _note_messages([note], 0, 960)

    def test_numeric_slash_extension_does_not_replace_slash_bass(self):
        body = spec()
        body["measures"][0]["chords"][0]["symbol"] = "F6/9/C"
        midi = self.render(body)
        self.assertEqual(ons(midi.tracks[2])[0][1], 36)
        self.assertEqual({note for tick, note, _ in ons(midi.tracks[1]) if tick == 0}, {60, 65, 69})

    def test_all_31_instrument_programs(self):
        expected = {"acoustic_grand_piano": 0, "electric_piano_1": 4, "drawbar_organ": 16, "clavinet": 7,
                    "pad_2_warm": 89, "choir_aahs": 52, "acoustic_guitar_nylon": 24, "acoustic_guitar_steel": 25,
                    "electric_guitar_clean": 27, "overdriven_guitar": 29, "distortion_guitar": 30,
                    "violin": 40, "viola": 41, "cello": 42, "string_ensemble_1": 48, "pizzicato_strings": 45,
                    "orchestral_harp": 46, "flute": 73, "clarinet": 71, "oboe": 68, "bassoon": 70,
                    "soprano_sax": 64, "alto_sax": 65, "tenor_sax": 66, "baritone_sax": 67,
                    "trumpet": 56, "trombone": 57, "french_horn": 60, "tuba": 58, "brass_section": 61, "muted_trumpet": 59}
        for instrument, program in expected.items():
            with self.subTest(instrument=instrument):
                body = spec()
                body["play"]["harmonyOverride"] = instrument
                track = self.render(body).tracks[1]
                self.assertEqual(next(message.program for message in track if message.type == "program_change"), program)

    def test_volumes_keep_zero_cc7_and_disabled_tracks_are_omitted(self):
        body = spec()
        body["play"].update(masterVolume=0.5, harmonyVolume=0.4, bassVolume=0.3, drumVolume=0.6)
        midi = self.render(body)
        self.assertEqual([next(m.value for m in track if m.type == "control_change" and m.control == 7)
                          for track in midi.tracks[1:]], [25, 19, 38])
        body["play"]["masterVolume"] = 0
        midi = self.render(body)
        self.assertEqual([next(m.value for m in track if m.type == "control_change") for track in midi.tracks[1:]], [0, 0, 0])
        body["play"].update(bassEnabled=False, drumMode="manual", drumPatternId=None)
        self.assertEqual(len(self.render(body).tracks), 2)

    def test_enharmonic_keys_export_to_representable_smf_signatures(self):
        expected = {"D# major": "Eb", "G# major": "Ab", "A# major": "Bb", "Db minor": "C#m",
                    "Gb minor": "F#m", "Cb major": "Cb", "Ab minor": "Abm"}
        for key, midi_key in expected.items():
            with self.subTest(key=key):
                track = self.render(spec(key=key)).tracks[0]
                self.assertEqual(next(m.key for m in track if m.type == "key_signature"), midi_key)

    def test_request_bounds_and_nonfinite_values_are_readable_422(self):
        invalid = []
        def bad(change):
            body = spec()
            change(body)
            invalid.append(body)
        bad(lambda b: b.update(bpm=True))
        bad(lambda b: b.update(bpm=19))
        bad(lambda b: b.update(key="C dorian"))
        bad(lambda b: b.update(timeSignature=[4, 3]))
        bad(lambda b: b.update(timeSignature=[True, 4]))
        bad(lambda b: b.update(measures=[]))
        bad(lambda b: b.update(measures=b["measures"] * 257))
        bad(lambda b: b["measures"][0].update(chords=[None]))
        bad(lambda b: b["measures"][0]["chords"][0].update(voicing=[128]))
        bad(lambda b: b["measures"][0]["chords"][0].update(voicing=[60.5]))
        bad(lambda b: b["measures"][0]["chords"][0].update(voicing=[True]))
        bad(lambda b: b["measures"][0]["chords"][0].update(symbol="C/../../file"))
        bad(lambda b: b["sections"][0].update(length=3))
        bad(lambda b: b["sections"][1].update(startMeasure=0))
        bad(lambda b: b["play"].update(harmonyOverride="missing"))
        bad(lambda b: b["play"].update(harmonyRhythmId="missing"))
        bad(lambda b: b["play"].update(drumPatternId=99))
        bad(lambda b: b["play"].update(masterVolume=-0.1))
        bad(lambda b: b["play"].update(masterVolume=float("nan")))
        bad(lambda b: b["play"].update(masterVolume=float("inf")))
        bad(lambda b: b["play"].update(masterVolume=True))
        for index, body in enumerate(invalid):
            with self.subTest(case=index):
                response = CLIENT.post("/api/export/midi", content=json.dumps(body), headers={"Content-Type": "application/json"})
                self.assertEqual(response.status_code, 422, response.text)
                self.assertIsInstance(response.json()["detail"], str)

    def test_note_and_binary_output_caps_are_enforced(self):
        for constant in ("MAX_EXPORT_NOTES", "MAX_EXPORT_BYTES"):
            with self.subTest(constant=constant), mock.patch.object(smf_export, constant, 1):
                response = CLIENT.post("/api/export/midi", json=spec())
                self.assertEqual(response.status_code, 413)


class DawLauncherTest(unittest.TestCase):
    def setUp(self):
        workspace = tempfile.TemporaryDirectory(prefix="export-test-", dir=Path(__file__).parent)
        self.addCleanup(workspace.cleanup)
        self.exports = Path(workspace.name) / "exports"
        patches = [mock.patch.object(daw_launcher, "EXPORT_DIR", self.exports),
                   mock.patch.object(daw_launcher.platform, "system", return_value="Darwin"),
                   mock.patch.object(daw_launcher, "_installed", return_value=True),
                   mock.patch.object(daw_launcher.subprocess, "run", return_value=mock.Mock(returncode=0, stdout=""))]
        self.dir_patch, self.system, self.installed, self.run = [patch.start() for patch in patches]
        for patch in patches:
            self.addCleanup(patch.stop)

    def test_targets_do_not_create_files_or_open_apps(self):
        response = CLIENT.get("/api/daw/targets")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"targets": [
            {"id": "logic", "name": "Logic Pro", "available": True, "action": "open"},
            {"id": "ableton", "name": "Ableton Live", "available": True, "action": "reveal"}]})
        self.assertFalse(self.exports.exists())
        self.run.assert_not_called()

    def test_logic_saves_unique_files_and_uses_argument_arrays(self):
        body = {"target": "logic", "spec": spec(title="../악보;$(touch secret)\n")}
        first = CLIENT.post("/api/daw/open", json=body)
        second = CLIENT.post("/api/daw/open", json=body)
        self.assertEqual((first.status_code, second.status_code), (200, 200))
        self.assertNotEqual(first.json()["path"], second.json()["path"])
        path = Path(first.json()["path"])
        self.assertEqual(path.parent, self.exports)
        self.assertNotIn("$", path.name)
        self.assertNotIn("..", path.name)
        self.assertEqual(first.json()["action"], "open")
        self.assertEqual(len(decode(path.read_bytes()).tracks), 4)
        args, kwargs = self.run.call_args_list[0]
        self.assertEqual(args[0], ["/usr/bin/open", "-b", "com.apple.logic10", str(path)])
        self.assertIs(kwargs["shell"], False)
        self.assertEqual(kwargs["timeout"], 10)

    def test_logic_fallback_and_ableton_reveal(self):
        self.run.side_effect = [mock.Mock(returncode=1), mock.Mock(returncode=0)]
        response = CLIENT.post("/api/daw/open", json={"target": "logic", "spec": spec()})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.run.call_args.args[0][:4], ["/usr/bin/open", "-a", "Logic Pro", response.json()["path"]])
        self.run.reset_mock(side_effect=True)
        self.run.return_value = mock.Mock(returncode=0)
        response = CLIENT.post("/api/daw/open", json={"target": "ableton", "spec": spec()})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["action"], "reveal")
        self.assertEqual(self.run.call_args.args[0], ["/usr/bin/open", "-R", response.json()["path"]])

    def test_missing_app_and_non_macos_do_not_save_or_execute(self):
        self.installed.return_value = False
        self.assertEqual(CLIENT.post("/api/daw/open", json={"target": "logic", "spec": spec()}).status_code, 409)
        self.system.return_value = "Linux"
        self.assertEqual(CLIENT.post("/api/daw/open", json={"target": "logic", "spec": spec()}).status_code, 501)
        response = CLIENT.get("/api/daw/targets")
        self.assertTrue(all(not target["available"] for target in response.json()["targets"]))
        self.assertFalse(self.exports.exists())
        self.run.assert_not_called()

    def test_failed_open_keeps_saved_midi_and_reports_500(self):
        self.run.side_effect = subprocess.TimeoutExpired("open", timeout=10)
        response = CLIENT.post("/api/daw/open", json={"target": "logic", "spec": spec()})
        self.assertEqual(response.status_code, 500)
        self.assertIn("saved file remains", response.json()["detail"])
        self.assertEqual(len(list(self.exports.glob("*.mid"))), 1)

    def test_filename_collision_never_overwrites_a_file(self):
        self.exports.mkdir()
        path = self.exports / "Untitled-aaaaaaaaaaaa.mid"
        path.write_bytes(b"existing user data")
        with mock.patch.object(daw_launcher.uuid, "uuid4", return_value=mock.Mock(hex="a" * 32)):
            response = CLIENT.post("/api/daw/open", json={"target": "logic", "spec": spec(title="")})
        self.assertEqual(response.status_code, 500)
        self.assertEqual(path.read_bytes(), b"existing user data")
        self.run.assert_not_called()

    def test_invalid_saved_binary_is_rejected(self):
        for data in (b"", b"not midi", "MThd"):
            with self.subTest(data=data), self.assertRaises(daw_launcher.DawError):
                daw_launcher._save(data, "score")


class DawDiscoveryTest(unittest.TestCase):
    def test_spotlight_and_applications_fallback_are_mocked(self):
        with mock.patch.object(daw_launcher.subprocess, "run", return_value=mock.Mock(returncode=0, stdout="/Applications/Logic Pro.app\n")) as run, \
                mock.patch.object(Path, "is_dir", return_value=True), \
                mock.patch.object(daw_launcher, "_APPLICATIONS") as applications:
            self.assertTrue(daw_launcher._installed("logic"))
            self.assertEqual(run.call_args.args[0], ["/usr/bin/mdfind", "kMDItemCFBundleIdentifier == 'com.apple.logic10'"])
            self.assertIs(run.call_args.kwargs["shell"], False)
            applications.glob.assert_not_called()
            run.return_value = mock.Mock(returncode=1, stdout="")
            applications.glob.return_value = [mock.Mock(is_dir=lambda: True)]
            self.assertTrue(daw_launcher._installed("logic"))
            applications.glob.assert_called_with("Logic Pro*.app")
            self.assertTrue(daw_launcher._installed("ableton"))
            applications.glob.assert_called_with("Ableton Live*.app")


if __name__ == "__main__":
    unittest.main()
