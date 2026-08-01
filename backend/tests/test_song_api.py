"""API tests for /api/generate/song: the genre defaults, the three optional
Play-panel choices (comping rhythm preset, drum pattern or off, harmony
instrument), the 422 for an unknown choice, the MIDI render, and the
fitting of a model pass's bars to length_bars.

Chord generation is patched out: every request gets the same two bars (or,
in ModelBarsTest, a fixed model pass) and no model runs.
"""

import base64
import copy
import io
import unittest
from unittest import mock

import mido
from fastapi.testclient import TestClient

import app.main
from app.services.genre_tables import (
    DRUM_PATTERNS,
    GENRE_DEFAULTS,
    GENRE_HARMONY_INSTRUMENT,
    GENRE_TO_DRUM_PATTERN_ID,
    HARMONY_RHYTHM_PRESETS,
    harmony_strikes,
)
from app.services.inference import dispatch_model_for_genre
from app.services.song_builder import _harmony_voicing, build_drum_events

CLIENT = TestClient(app.main.app)
SONG = "/api/generate/song"
BARS = [{"chords": ["Cmaj7", "Am7"]}, {"chords": ["Dm7", "G7"]}]
RESPONSE_KEYS = {"genre", "key", "bpm", "time_signature", "harmony_rhythm", "drum_pattern",
                 "bars", "tracks", "midi_b64", "model_used", "seed"}


def strikes(track: dict) -> set[tuple[int, float, float]]:
    """(bar, beat, duration) of every harmony strike (one event per chord tone)."""
    return {(e["bar"], e["beat"], round(e["duration"], 9)) for e in track["events"]}


def midi_tracks(data: dict) -> list[mido.MidiTrack]:
    return mido.MidiFile(file=io.BytesIO(base64.b64decode(data["midi_b64"]))).tracks


def midi_programs(data: dict) -> list[tuple[int, int]]:
    return [(m.channel, m.program) for track in midi_tracks(data) for m in track if m.type == "program_change"]


def midi_restrikes(data: dict) -> list[tuple[int, int, int]]:
    """(channel, note, tick) of every MIDI note_on whose pitch is still sounding on its channel."""
    found = []
    for track in midi_tracks(data):
        sounding, tick = set(), 0
        for m in track:
            tick += m.time
            if m.type == "note_on":
                if (m.channel, m.note) in sounding:
                    found.append((m.channel, m.note, tick))
                sounding.add((m.channel, m.note))
            elif m.type == "note_off":
                sounding.discard((m.channel, m.note))
    return found


class SongApiTest(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch("app.routes.song._generate_bars",
                             side_effect=lambda **_: (copy.deepcopy(BARS), "rules"))
        self.generate = patcher.start()
        self.addCleanup(patcher.stop)

    def post(self, **fields):
        return CLIENT.post(SONG, json={"length_bars": len(BARS), **fields})

    def test_defaults_follow_the_genre(self):
        res = self.post(genre="jazz")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(set(data), RESPONSE_KEYS)
        self.assertIsNone(data["seed"])
        self.assertEqual((data["harmony_rhythm"], data["drum_pattern"]), ("charleston", 4))
        harmony = data["tracks"]["harmony"]
        self.assertEqual(harmony["instrument"], "electric_piano_1")
        # A different second chord adds the half-bar strike to Charleston.
        self.assertEqual(strikes(harmony), {(bar, pos, dur) for bar in (0, 1)
                                          for pos, dur in [(0.0, 1.133333333), (1.5, 0.425), (2.0, 1.133333333)]})
        self.assertEqual(data["tracks"]["drum"]["instrument"], "gm_drum_kit")
        self.assertEqual(midi_programs(data), [(0, 4), (1, 32)])   # electric piano, upright bass

    def test_every_genre_echoes_its_own_choices(self):
        for genre in GENRE_DEFAULTS:
            with self.subTest(genre=genre):
                data = self.post(genre=genre).json()
                preset = next(p for p in HARMONY_RHYTHM_PRESETS if genre in p["genres"])
                self.assertEqual(data["harmony_rhythm"], preset["id"])
                self.assertEqual(data["drum_pattern"], GENRE_TO_DRUM_PATTERN_ID[genre])
                self.assertEqual(data["tracks"]["harmony"]["instrument"], GENRE_HARMONY_INSTRUMENT[genre])
                self.assertTrue(data["tracks"]["drum"]["events"])

    def test_picked_rhythm_replaces_the_genre_rhythm(self):
        data = self.post(genre="pop", harmony_rhythm="boom_bap").json()
        self.assertEqual(data["harmony_rhythm"], "boom_bap")
        events = data["tracks"]["harmony"]["events"]
        self.assertEqual(sorted({e["beat"] for e in events}), [0.0, 2.5])
        # [0, 2.5]: gaps 2.5 and 1.5 against an average of 2.
        self.assertEqual(strikes(data["tracks"]["harmony"]), {(bar, 0.0, 1.7) for bar in (0, 1)}
                         | {(bar, 2.5, 1.275) for bar in (0, 1)})
        # Beat 2.5 lies in the second chord slot.
        at_2_5 = sorted(e["pitch"] for e in events if e["bar"] == 0 and e["beat"] == 2.5)
        self.assertEqual(at_2_5, sorted(_harmony_voicing("Am7")))

    def test_every_rhythm_preset_strikes_its_positions(self):
        # With no chord change, sparse presets retain their literal rhythm.
        self.generate.side_effect = lambda **_: ([{"chords": ["Cmaj7", "Cmaj7"]},
                                                  {"chords": ["Dm7", "Dm7"]}], "rules")
        for preset in HARMONY_RHYTHM_PRESETS:
            with self.subTest(preset=preset["id"]):
                data = self.post(genre="pop", harmony_rhythm=preset["id"]).json()
                self.assertEqual(data["harmony_rhythm"], preset["id"])
                want = {(bar, pos, round(dur, 9)) for bar in (0, 1) for pos, dur in harmony_strikes(preset["pattern"], 4)}
                self.assertEqual(strikes(data["tracks"]["harmony"]), want)

    def test_picked_drum_pattern_plays_its_hits(self):
        data = self.post(genre="pop", drum_pattern=11).json()
        self.assertEqual(data["drum_pattern"], 11)
        got = sorted((e["bar"], e["beat"], e["voice"], e["velocity"]) for e in data["tracks"]["drum"]["events"])
        want = sorted((bar, pos16 / 4, voice, vel)
                      for bar in range(len(BARS))
                      for voice, hits in DRUM_PATTERNS[11]["voices"].items()
                      for pos16, vel in hits)
        self.assertEqual(got, want)
        self.assertIn((0, 1.0, "L", 0.9), got)   # Samba's surdo on 2; Pop has no floor tom

    def test_drums_off(self):
        data = self.post(genre="pop", drum_pattern="off").json()
        self.assertEqual(data["drum_pattern"], "off")
        self.assertEqual(data["tracks"]["drum"], {"events": [], "instrument": "gm_drum_kit", "source": "rule"})
        self.assertFalse([m for m in midi_tracks(data)[3] if m.type == "note_on"])
        self.assertTrue(data["tracks"]["harmony"]["events"])
        self.assertTrue(data["tracks"]["bass"]["events"])

    def test_picked_instrument_sets_the_track_and_midi_program(self):
        data = self.post(genre="pop", harmony_instrument="trumpet").json()
        self.assertEqual(data["tracks"]["harmony"]["instrument"], "trumpet")
        self.assertEqual(data["tracks"]["bass"]["instrument"], "electric_bass_finger")
        self.assertEqual(midi_programs(data), [(0, 56), (1, 33)])

    def test_picked_key_tempo_and_bass_replace_the_genre_own(self):
        data = self.post(genre="pop", key="D minor", bpm=72, bass_instrument="contrabass").json()
        self.assertEqual((data["key"], data["bpm"]), ("D minor", 72))
        self.assertEqual((self.generate.call_args.kwargs["key"], self.generate.call_args.kwargs["anchor"]),
                         ("D minor", True))
        self.assertEqual(data["tracks"]["bass"]["instrument"], "contrabass")
        self.assertEqual(midi_programs(data), [(0, 0), (1, 32)])   # the genre's piano, the picked upright bass
        tempos = [m.tempo for m in midi_tracks(data)[0] if m.type == "set_tempo"]
        self.assertEqual(tempos, [mido.bpm2tempo(72)])
        # Left out, each falls back to the genre's own, and the genre's key is not anchored.
        data = self.post(genre="pop").json()
        self.assertFalse(self.generate.call_args.kwargs["anchor"])
        self.assertEqual((data["key"], data["bpm"], data["tracks"]["bass"]["instrument"]),
                         (GENRE_DEFAULTS["pop"]["key"], GENRE_DEFAULTS["pop"]["bpm"], "electric_bass_finger"))

    def test_unknown_choices_are_422(self):
        cases = [
            ("harmony_rhythm", "waltz", "charleston"),
            ("harmony_rhythm", "", "charleston"),
            ("drum_pattern", 99, "1-15 or 'off'"),
            ("drum_pattern", 0, "1-15 or 'off'"),
            ("drum_pattern", "loud", "1-15 or 'off'"),
            ("drum_pattern", "3", "1-15 or 'off'"),
            ("drum_pattern", True, "1-15 or 'off'"),
            ("harmony_instrument", "kazoo", "trumpet"),
            ("key", "H major", "Db major"),
            ("key", "C# major", "Db major"),
            ("key", "d minor", "D minor"),
            ("bpm", 300, "220"),
            ("bpm", 39, "40"),
            ("bpm", "120", "valid integer"),
            ("bass_instrument", "tuba", "contrabass"),
        ]
        for field, value, listed in cases:
            with self.subTest(field=field, value=value):
                res = self.post(genre="pop", **{field: value})
                self.assertEqual(res.status_code, 422)
                (error,) = res.json()["detail"]
                self.assertEqual(error["loc"], ["body", field])
                self.assertIn(listed, error["msg"])
        self.generate.assert_not_called()

    def test_unsupported_genre_is_still_400(self):
        res = self.post(genre="polka", harmony_rhythm="block")
        self.assertEqual(res.status_code, 400)
        self.assertIn("Unsupported genre", res.json()["detail"])

    def test_midi_never_restrikes_a_sounding_pitch(self):
        for genre in GENRE_DEFAULTS:
            with self.subTest(genre=genre):
                self.assertEqual(midi_restrikes(self.post(genre=genre).json()), [])
        # Funk's bass strikes C at beats 0 and 0.5: the events keep the
        # editor's 0.9 beats, the MIDI note stops at the re-strike (480 ticks a beat).
        data = self.post(genre="funk").json()
        self.assertEqual({e["duration"] for e in data["tracks"]["bass"]["events"]}, {0.9})
        bass = [(m.type, m.note, m.time) for m in midi_tracks(data)[2] if m.type in ("note_on", "note_off")]
        self.assertEqual(bass[:4], [("note_on", 36, 0), ("note_off", 36, 240),
                                    ("note_on", 36, 0), ("note_off", 36, 432)])


class ModelBarsTest(unittest.TestCase):
    """The model path (patched) is fitted to length_bars bars of two chords
    before the three tracks are built."""

    def song(self, model_bars: list[list[str]], **fields) -> dict:
        registry = mock.Mock()
        registry.load_model.return_value = object()
        with mock.patch("app.routes.song.TORCH_AVAILABLE", True), \
                mock.patch("app.routes.song.ModelRegistry") as model_registry, \
                mock.patch("app.routes.song.build_prompt", return_value=[]), \
                mock.patch("app.routes.song.generate_with_model", return_value=model_bars):
            model_registry.get.return_value = registry
            res = CLIENT.post(SONG, json={"genre": "pop", "length_bars": 4, **fields})
        self.assertEqual(res.status_code, 200)
        return res.json()

    def assert_tracks_span(self, data: dict, n_bars: int):
        for name in ("harmony", "bass", "drum"):
            with self.subTest(track=name):
                self.assertEqual({e["bar"] for e in data["tracks"][name]["events"]}, set(range(n_bars)))

    def test_short_pass_repeats_its_bars_with_two_chords_each(self):
        # A C blues already sits on pop's C major, so the key anchor leaves it as written.
        data = self.song([["C7"], ["F7", "G7", "C7"]])
        self.assertEqual(data["model_used"], dispatch_model_for_genre("pop"))
        self.assertEqual([b["chords"] for b in data["bars"]],
                         [["C7", "C7"], ["F7", "G7"], ["C7", "C7"], ["F7", "G7"]])
        self.assert_tracks_span(data, 4)

    def test_a_model_pass_in_another_key_moves_into_the_requested_key(self):
        # The model often answers an empty-context request in C; the song is moved to the picked key.
        data = self.song([["Cmaj", "Am"], ["Fmaj", "Gmaj"]], key="D major")
        self.assertEqual(data["key"], "D major")
        self.assertEqual([b["chords"] for b in data["bars"]],
                         [["Dmaj", "Bm"], ["Gmaj", "Amaj"], ["Dmaj", "Bm"], ["Gmaj", "Amaj"]])
        # The tracks are built from the moved chords: the bass plays D first, not C.
        first_bass = min(data["tracks"]["bass"]["events"], key=lambda e: (e["bar"], e["beat"]))
        self.assertEqual(first_bass["pitch"] % 12, 2)

    def test_a_pass_is_only_anchored_when_the_request_picked_the_key(self):
        # Left to the genre, a pass comes back as written, even one in another key.
        data = self.song([["Cmaj", "Am"], ["Fmaj", "Gmaj"]], genre="country")
        self.assertEqual(data["key"], "G major")
        self.assertEqual(data["bars"][0]["chords"], ["Cmaj", "Am"])
        # A secondary dominant in the picked key stays: each written chord counts once, so a
        # one-chord bar is not counted twice before the pass is fitted to two slots.
        data = self.song([["Cmaj"], ["D7"], ["Gmaj"], ["Cmaj"]], key="C major")
        self.assertEqual([b["chords"] for b in data["bars"]],
                         [["Cmaj", "Cmaj"], ["D7", "D7"], ["Gmaj", "Gmaj"], ["Cmaj", "Cmaj"]])

    def test_empty_pass_falls_back_to_the_rules(self):
        data = self.song([])
        self.assertEqual(data["model_used"], "rules")
        self.assertEqual([len(b["chords"]) for b in data["bars"]], [2, 2, 2, 2])
        self.assert_tracks_span(data, 4)


class DrumGridTest(unittest.TestCase):
    def test_a_16th_is_a_quarter_beat_in_any_meter(self):
        # As in the editor's drums.ts: 3/4 keeps the 16th grid and drops steps 12-15.
        events = build_drum_events(1, "pop", beats_per_measure=3)
        got = sorted((e["beat"], e["voice"]) for e in events)
        want = sorted((pos16 / 4, voice) for voice, hits in DRUM_PATTERNS[1]["voices"].items()
                      for pos16, _ in hits if pos16 < 12)
        self.assertEqual(got, want)
        self.assertIn((1.0, "S"), got)   # the backbeat snare stays on beat 2
        self.assertEqual({e["duration"] for e in events}, {0.25})


class HarmonyStrikesTest(unittest.TestCase):
    def test_no_strike_runs_into_the_next(self):
        for preset in HARMONY_RHYTHM_PRESETS:
            for beats in (4, 3, 6):
                with self.subTest(preset=preset["id"], beats=beats):
                    bar_strikes = harmony_strikes(preset["pattern"], beats)
                    self.assertTrue(bar_strikes)
                    # The last strike is followed by the next bar's first.
                    nexts = [pos for pos, _ in bar_strikes[1:]] + [bar_strikes[0][0] + beats]
                    for (pos, dur), nxt in zip(bar_strikes, nexts):
                        self.assertGreater(dur, 0)
                        self.assertLess(pos + dur, nxt)

    def test_positions_outside_the_bar_are_dropped(self):
        self.assertEqual([pos for pos, _ in harmony_strikes([0, 1.5, 3, 3.5], 3)], [0.0, 1.5])
        self.assertEqual(harmony_strikes([], 4), [])


if __name__ == "__main__":
    unittest.main()
