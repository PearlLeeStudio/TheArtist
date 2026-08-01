"""Parity between app/services/genre_tables.py and the frontend tables it
mirrors: genreTables.ts, drumPatterns.ts and harmonyStrikes in
playSettings.ts (frontend/src/engine/).

node imports the TypeScript sources directly (built-in type stripping) and
prints the tables as JSON. Those tests skip when node or the frontend
folder is absent, or node is too old to strip types by default (before
22.18 / 23.6); any other node failure is an error. The backend-only checks
always run.
"""

import json
import shutil
import subprocess
import unittest
from pathlib import Path

from app.services import genre_tables as gt

ENGINE = Path(__file__).resolve().parents[2] / "frontend" / "src" / "engine"
NODE = shutil.which("node")
BEATS = (4, 3, 6)

DUMP_SCRIPT = """
const t = await import(%(tables)s);
const d = await import(%(drums)s);
const p = await import(%(play)s);
const strikes = {};
for (const preset of t.HARMONY_RHYTHM_PRESETS) {
  strikes[preset.id] = Object.fromEntries(%(beats)s.map((b) => [b, p.harmonyStrikes(preset.pattern, b)]));
}
console.log(JSON.stringify({
  presets: t.HARMONY_RHYTHM_PRESETS,
  defaultRhythm: t.DEFAULT_HARMONY_RHYTHM_ID,
  harmonyRhythm: t.GENRE_HARMONY_RHYTHM,
  bassRhythm: t.GENRE_BASS_RHYTHM,
  harmonyInstrument: t.GENRE_HARMONY_INSTRUMENT,
  bassInstrument: t.GENRE_BASS_INSTRUMENT,
  genreDrum: t.GENRE_TO_DRUM_PATTERN_ID,
  instruments: t.HARMONY_INSTRUMENTS.map((i) => i.name),
  drums: d.DRUM_PATTERNS,
  strikes,
}));
"""


def strips_types(version: str) -> bool:
    """Whether this node version (e.g. 'v22.18.0') imports .ts files with no
    flag: type stripping is on by default from 22.18 and 23.6."""
    major, minor = (int(n) for n in version.strip().lstrip("v").split(".")[:2])
    return (major, minor) >= (23, 6) or (major == 22 and minor >= 18)


def frontend_drum(p: dict) -> dict:
    """A drumPatterns.ts pattern in the backend shape: optional fields get
    their defaults and a bare-number hit n becomes [n, 1.0]."""
    return {
        "name": p["name"],
        "kit": p.get("kit", "acoustic"),
        "swing": p.get("swing", 0),
        "genre": p.get("genre"),
        "voices": {
            voice: [[hit, 1.0] if isinstance(hit, (int, float)) else hit for hit in hits]
            for voice, hits in p["voices"].items()
        },
    }


@unittest.skipUnless(NODE and ENGINE.is_dir(), "needs node and frontend/src/engine")
class FrontendParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        version = subprocess.run([NODE, "--version"], capture_output=True, text=True, timeout=30).stdout.strip()
        if not strips_types(version):
            raise unittest.SkipTest(f"node {version} cannot import .ts files (needs 22.18+ or 23.6+)")
        script = DUMP_SCRIPT % {
            "tables": json.dumps((ENGINE / "genreTables.ts").as_uri()),
            "drums": json.dumps((ENGINE / "drumPatterns.ts").as_uri()),
            "play": json.dumps((ENGINE / "playSettings.ts").as_uri()),
            "beats": json.dumps(list(BEATS)),
        }
        run = subprocess.run([NODE, "--input-type=module", "-e", script],
                             capture_output=True, text=True, timeout=60)
        if run.returncode != 0:
            raise RuntimeError(f"node could not load the frontend tables:\n{run.stderr}")
        cls.fe = json.loads(run.stdout)

    def test_rhythm_presets(self):
        presets = [{k: p[k] for k in ("id", "label", "pattern", "genres")} for p in self.fe["presets"]]
        self.assertEqual(presets, gt.HARMONY_RHYTHM_PRESETS)
        self.assertEqual(self.fe["defaultRhythm"], gt.DEFAULT_HARMONY_RHYTHM_ID)

    def test_genre_tables(self):
        for name, table in (
            ("harmonyRhythm", gt.GENRE_HARMONY_RHYTHM),
            ("bassRhythm", gt.GENRE_BASS_RHYTHM),
            ("harmonyInstrument", gt.GENRE_HARMONY_INSTRUMENT),
            ("bassInstrument", gt.GENRE_BASS_INSTRUMENT),
            ("genreDrum", gt.GENRE_TO_DRUM_PATTERN_ID),
        ):
            with self.subTest(table=name):
                self.assertEqual(self.fe[name], table)

    def test_harmony_instruments_in_order(self):
        self.assertEqual(tuple(self.fe["instruments"]), gt.HARMONY_INSTRUMENTS)

    def test_drum_patterns(self):
        patterns = {p["id"]: frontend_drum(p) for p in self.fe["drums"]}
        self.assertEqual(sorted(patterns), sorted(gt.DRUM_PATTERNS))
        for pattern_id, pattern in patterns.items():
            with self.subTest(pattern=pattern_id):
                self.assertEqual(pattern, gt.DRUM_PATTERNS[pattern_id])

    def test_harmony_strikes(self):
        for preset in gt.HARMONY_RHYTHM_PRESETS:
            for beats in BEATS:
                with self.subTest(preset=preset["id"], beats=beats):
                    fe = self.fe["strikes"][preset["id"]][str(beats)]
                    be = gt.harmony_strikes(preset["pattern"], beats)
                    self.assertEqual(len(fe), len(be))
                    for f, (pos, dur) in zip(fe, be):
                        self.assertAlmostEqual(f["pos"], pos, delta=1e-12)
                        self.assertAlmostEqual(f["dur"], dur, delta=1e-12)


class NodeGateTest(unittest.TestCase):
    def test_only_nodes_without_type_stripping_skip(self):
        for version in ("v20.19.0", "v22.17.1", "v23.5.0"):
            with self.subTest(version=version):
                self.assertFalse(strips_types(version))
        for version in ("v22.18.0", "v23.6.0", "v24.0.0", "v25.9.0\n"):
            with self.subTest(version=version):
                self.assertTrue(strips_types(version))


class BackendTablesTest(unittest.TestCase):
    def test_every_instrument_has_a_gm_program(self):
        self.assertEqual(len(set(gt.HARMONY_INSTRUMENTS)), 31)
        for name in gt.HARMONY_INSTRUMENTS + tuple(gt.GENRE_BASS_INSTRUMENT.values()):
            with self.subTest(name=name):
                self.assertIn(name, gt.GM_PROGRAM)

    def test_genre_defaults_are_pickable_instruments(self):
        self.assertLessEqual(set(gt.GENRE_HARMONY_INSTRUMENT.values()), set(gt.HARMONY_INSTRUMENTS))

    def test_each_genre_has_one_rhythm_and_its_own_drum_pattern(self):
        genres = [g for p in gt.HARMONY_RHYTHM_PRESETS for g in p["genres"]]
        self.assertEqual(sorted(genres), sorted(gt.GENRE_DEFAULTS))
        self.assertEqual(sorted(gt.GENRE_TO_DRUM_PATTERN_ID), sorted(gt.GENRE_DEFAULTS))
        for genre, pattern_id in gt.GENRE_TO_DRUM_PATTERN_ID.items():
            with self.subTest(genre=genre):
                pattern = gt.DRUM_PATTERNS[pattern_id]
                self.assertEqual(pattern["genre"], genre)
                self.assertEqual(gt.drum_pattern(genre),
                                 {k: pattern[k] for k in ("name", "kit", "swing", "voices")})


if __name__ == "__main__":
    unittest.main()
