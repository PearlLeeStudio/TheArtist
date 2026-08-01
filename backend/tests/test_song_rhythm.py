import unittest

from app.services.song_builder import build_harmony_events


class ChordBoundaryTests(unittest.TestCase):
    def positions(self, chords, pattern, beats=4):
        events = build_harmony_events([{"chords": chords}], "pop", beats_per_measure=beats, pattern=pattern)
        return sorted({event["beat"] for event in events})

    def test_sparse_patterns_keep_both_changed_chords(self):
        self.assertEqual(self.positions(["C", "F"], [0, 1.5]), [0, 1.5, 2])
        self.assertEqual(self.positions(["C", "F"], [0]), [0, 2])
        self.assertEqual(self.positions(["C", "F"], [0], 3), [0, 1.5])

    def test_repeated_chord_keeps_sparse_rhythm(self):
        self.assertEqual(self.positions(["C", "C"], [0]), [0])
        self.assertEqual(self.positions(["C"], [0]), [0])

    def test_the_models_minor_spelling_is_voiced_minor(self):
        # The vocabulary writes a minor triad as "Am"; it once fell back to a major voicing
        events = build_harmony_events([{"chords": ["Am", "Em"]}], "pop", beats_per_measure=4, pattern=[0])
        by_beat = {}
        for event in events:
            by_beat.setdefault(event["beat"], set()).add(event["pitch"] % 12)
        self.assertEqual(by_beat, {0: {9, 0, 4}, 2: {4, 7, 11}})  # A C E, then E G B

    def test_voice_leading_never_puts_a_major_third_under_a_minor_chord(self):
        from app.services.voice_leading import pick_bass_for_smooth_voice_leading
        for prev in ("D", "Db", "C#", "F", "Bb7", "Gmaj7"):
            with self.subTest(prev=prev):
                self.assertNotIn(pick_bass_for_smooth_voice_leading(prev, "Am"), ("C#", "Db"))

    def test_sustains_end_at_changed_chord_boundary(self):
        events = build_harmony_events([{"chords": ["C", "F"]}], "pop", beats_per_measure=4, pattern=[0, 1.5, 3])
        self.assertTrue(all(e["beat"] >= 2 or e["beat"] + e["duration"] <= 2 for e in events))
