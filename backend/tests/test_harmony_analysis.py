"""Deterministic note analysis, without OCR, I/O or model calls."""

from dataclasses import asdict
import math
import unittest

from app.services.harmony_analysis import NoteEvent, analyze_harmony, estimate_key


def sonority(pitches, onset=0, duration=2, measure=0, *, spellings=None):
    return [NoteEvent(pitch, onset, duration, measure, str(index), "piano",
                      spellings[index] if spellings else None) for index, pitch in enumerate(pitches)]


class HarmonyAnalysisTest(unittest.TestCase):
    def test_major_minor_and_dominant_chords_have_key_relative_romans(self):
        for pitches, symbol, roman in (([48, 52, 55], "C", "I"),
                                      ([45, 48, 52], "Am", "vi"),
                                      ([43, 47, 50, 53], "G7", "V7")):
            with self.subTest(symbol=symbol):
                result = analyze_harmony(sonority(pitches), key="C major")
                top = result.segments[0].candidates[0]
                self.assertEqual((top.symbol, top.roman), (symbol, roman))
                self.assertEqual(top.missing_pitch_classes, ())
                self.assertTrue(top.evidence)
                self.assertGreater(top.confidence, .5)

    def test_inversions_retain_root_analysis_and_actual_bass(self):
        for pitches, symbol, roman, inversion in (([40, 43, 48], "C/E", "I", 1),
                                                  ([47, 50, 53, 55], "G7/B", "V7", 1)):
            with self.subTest(symbol=symbol):
                top = analyze_harmony(sonority(pitches), key="C major").segments[0].candidates[0]
                self.assertEqual((top.symbol, top.roman, top.inversion), (symbol, roman, inversion))

    def test_two_chords_in_one_measure_stay_separate_not_a_union(self):
        notes = sonority([48, 52, 55], duration=1) + sonority([53, 57, 60], onset=1, duration=1)
        notes += sonority([43, 47, 50, 53], onset=2, duration=2)
        result = analyze_harmony(notes, key="C major")
        self.assertEqual([(s.start, s.end, s.candidates[0].symbol) for s in result.segments],
                         [(0, 1, "C"), (1, 2, "F"), (2, 4, "G7")])
        self.assertEqual({s.measure for s in result.segments}, {0})

    def test_short_stepwise_passing_tone_does_not_turn_the_chord_into_an_extension(self):
        notes = sonority([48, 52, 55], duration=4)
        notes += [NoteEvent(72, 0, 1.5, voice="melody", staff="treble"),
                  NoteEvent(74, 1.5, .25, voice="melody", staff="treble"),
                  NoteEvent(76, 1.75, 2.25, voice="melody", staff="treble")]
        result = analyze_harmony(notes, key="C major")
        passing = next(segment for segment in result.segments if segment.start == 1.5)
        self.assertEqual(passing.candidates[0].symbol, "C")
        self.assertIn(74, passing.candidates[0].non_chord_pitches)
        self.assertTrue(next(note for note in passing.notes if note.pitch == 74).ornamental)
        self.assertIn("downweighted", passing.reason)

    def test_short_added_note_without_voice_metadata_is_not_treated_as_a_sustained_ninth(self):
        notes = sonority([48, 52, 55], duration=4) + [NoteEvent(62, 1.5, .25)]
        passing = next(s for s in analyze_harmony(notes, key="C major").segments if s.start == 1.5)
        self.assertEqual(passing.candidates[0].symbol, "C")

    def test_held_tritone_and_short_neighbor_melody_never_become_a_complete_unrelated_chord(self):
        for shift in range(12):
            with self.subTest(semitone_shift=shift):
                held = [NoteEvent(50 + shift, 0, 2, voice="lower"),
                        NoteEvent(56 + shift, 0, 2, voice="inner")]
                melody = [NoteEvent(pitch + shift, onset, duration, voice="melody")
                          for pitch, onset, duration in ((72, .25, .25), (71, .5, .5),
                                                         (72, 1, .5), (71, 1.5, .5))]
                result = analyze_harmony(held + melody, key="C major", total_duration=2)
                for segment in result.segments:
                    self.assertTrue(segment.uncertain)
                    self.assertEqual(len(segment.candidates), 2)
                    self.assertTrue(all(candidate.rootless_shell and candidate.quality == "7"
                                        and candidate.confidence <= .35 and len(candidate.missing_pitch_classes) == 2
                                        for candidate in segment.candidates))
                self.assertTrue(any(note.ornamental for segment in result.segments for note in segment.notes))
                self.assertEqual({note.pitch for segment in result.segments for note in segment.notes},
                                 {note.pitch for note in held + melody})

    def test_rootless_tritone_retains_both_dominant_roots_in_all_transpositions(self):
        for shift in range(12):
            with self.subTest(semitone_shift=shift):
                low, high = 48 + shift, 54 + shift
                result = analyze_harmony(sonority([low, high]), key="C major", top_n=1)
                segment = result.segments[0]
                self.assertEqual(len(segment.candidates), 2)
                self.assertTrue(segment.uncertain)
                self.assertEqual({candidate.chord_pitch_classes[0] for candidate in segment.candidates},
                                 {((low % 12) - 4) % 12, ((high % 12) - 4) % 12})
                for candidate in segment.candidates:
                    root = candidate.chord_pitch_classes[0]
                    self.assertEqual(set(candidate.missing_pitch_classes), {root, (root + 7) % 12})
                    self.assertTrue(candidate.rootless_shell)
                    self.assertLessEqual(candidate.confidence, .35)
                    self.assertIn("Ambiguous rootless dominant shell", candidate.evidence)

    def test_key_context_ranks_rootless_alternatives_without_changing_the_observed_pitches(self):
        notes = sonority([48, 54])
        g = analyze_harmony(notes, key="G major")
        db = analyze_harmony(notes, key="Db major")
        self.assertEqual(g.segments[0].candidates[0].root, "D")
        self.assertEqual(db.segments[0].candidates[0].root, "Ab")
        self.assertEqual(asdict(g.segments[0].notes[0]), asdict(db.segments[0].notes[0]))
        self.assertEqual({candidate.chord_pitch_classes[0] for candidate in g.segments[0].candidates},
                         {candidate.chord_pitch_classes[0] for candidate in db.segments[0].candidates})

    def test_a_perfect_fifth_dyad_has_no_rootless_dominant_exception(self):
        segment = analyze_harmony(sonority([48, 55]), key="C major").segments[0]
        self.assertEqual(segment.kind, "insufficient")
        self.assertEqual(segment.candidates, ())
        self.assertTrue(segment.uncertain)

    def test_structural_triad_keeps_its_identity_under_a_passing_tone_in_all_transpositions(self):
        for shift in range(12):
            with self.subTest(semitone_shift=shift):
                held = sonority([48 + shift, 52 + shift, 55 + shift], duration=4)
                melody = [NoteEvent(pitch + shift, onset, duration, voice="melody", staff="treble")
                          for pitch, onset, duration in ((72, 0, 1.5), (74, 1.5, .25), (76, 1.75, 2.25))]
                result = analyze_harmony(held + melody, key="C major")
                passing = next(segment for segment in result.segments if segment.start == 1.5)
                top = passing.candidates[0]
                self.assertEqual(top.quality, "maj")
                self.assertEqual(set(top.chord_pitch_classes), {(shift + pc) % 12 for pc in (0, 4, 7)})
                self.assertIn(74 + shift, top.non_chord_pitches)

    def test_short_coordinated_vertical_triad_still_has_three_structural_tones(self):
        result = analyze_harmony(sonority([48, 52, 55], duration=.25), key="C major")
        self.assertEqual(result.segments[0].candidates[0].symbol, "C")

    def test_candidate_alternatives_cannot_use_an_ornament_as_their_third_structural_tone(self):
        held = sonority([48, 52, 55], duration=4)
        melody = [NoteEvent(69, 0, 1.5, voice="melody", staff="treble"),
                  NoteEvent(71, 1.5, .25, voice="melody", staff="treble"),
                  NoteEvent(72, 1.75, 2.25, voice="melody", staff="treble")]
        result = analyze_harmony(held + melody, key="C major", top_n=10)
        passing = next(segment for segment in result.segments if segment.start == 1.5)
        self.assertEqual(passing.candidates[0].symbol, "C")
        self.assertTrue(all({0, 4, 7} <= set(candidate.chord_pitch_classes) for candidate in passing.candidates))

    def test_real_sustained_extension_is_kept_in_the_candidate(self):
        result = analyze_harmony(sonority([43, 47, 50, 53, 57]), key="C major")
        self.assertEqual(result.segments[0].candidates[0].symbol, "G9")

    def test_minor_and_borrowed_chords_respect_spelling(self):
        minor = analyze_harmony(sonority([45, 48, 52]), key="A minor")
        self.assertEqual(minor.segments[0].candidates[0].roman, "i")
        borrowed = analyze_harmony(sonority([46, 50, 53], spellings=["Bb", "D", "F"]), key="C major")
        top = borrowed.segments[0].candidates[0]
        self.assertEqual((top.symbol, top.roman), ("Bb", "bVII"))
        raised = analyze_harmony(sonority([46, 50, 53], spellings=["A#", "D", "F"]), key="C major")
        self.assertEqual(raised.segments[0].candidates[0].roman, "#VI")

    def test_manual_key_overrides_inference_without_transposing_evidence(self):
        notes = sonority([48, 52, 55])
        c = analyze_harmony(notes, key="C major")
        g = analyze_harmony(notes, key="G major")
        self.assertEqual(c.key.source, "provided")
        self.assertEqual(g.key.key, "G major")
        self.assertEqual(g.key.confidence, 1)
        self.assertEqual((c.segments[0].candidates[0].roman, g.segments[0].candidates[0].roman), ("I", "IV"))
        self.assertEqual(asdict(c.segments[0].notes[0]), asdict(g.segments[0].notes[0]))

    def test_duration_histogram_infers_key_and_keeps_ambiguous_alternatives(self):
        cadence = sonority([48, 52, 55], duration=4)
        cadence += sonority([41, 45, 48], onset=4, duration=2, measure=1)
        cadence += sonority([43, 47, 50, 53], onset=6, duration=2, measure=1)
        cadence += sonority([48, 52, 55], onset=8, duration=4, measure=2)
        result = analyze_harmony(cadence)
        self.assertEqual(result.key.source, "inferred")
        self.assertEqual(result.key.key, "C major")
        self.assertEqual(len(result.key.alternatives), 5)
        self.assertGreater(result.key.pitch_class_durations[0], result.key.pitch_class_durations[1])
        self.assertGreaterEqual(result.key.confidence, 0)
        self.assertLessEqual(result.key.confidence, .95)
        ambiguous = analyze_harmony(sonority([48, 52, 55]))
        self.assertLess(ambiguous.key.confidence, .5)
        self.assertTrue(ambiguous.warnings)

    def test_rests_empty_evidence_and_melody_are_not_fake_chords(self):
        empty = analyze_harmony([], total_duration=4)
        self.assertIsNone(empty.key.key)
        self.assertEqual(empty.segments[0].kind, "rest")
        self.assertEqual(empty.segments[0].candidates, ())
        notes = [NoteEvent(60, 1, 1)]
        result = analyze_harmony(notes, key="C major", total_duration=4)
        self.assertEqual([(s.start, s.end, s.kind) for s in result.segments],
                         [(0, 1, "rest"), (1, 2, "insufficient"), (2, 4, "rest")])
        self.assertTrue(all(not segment.candidates for segment in result.segments))
        self.assertTrue(all(segment.uncertain for segment in result.segments))

    def test_signature_guides_relative_pair_without_assuming_a_major_mode(self):
        notes = sonority([50, 53, 57], duration=4)
        notes += sonority([45, 49, 52, 55], onset=4, duration=2, measure=1)
        notes += sonority([50, 53, 57], onset=6, duration=4, measure=1)
        result = analyze_harmony(notes, key_signature=-1)
        self.assertEqual(result.key.key, "D minor")
        self.assertEqual(result.segments[0].candidates[0].roman, "i")
        provided = analyze_harmony(notes, key="F major", key_signature=-1)
        self.assertEqual(provided.key.key, "F major")
        self.assertEqual(provided.key.source, "provided")

    def test_inferred_extreme_major_signatures_preserve_written_tonic_spelling_and_roman_I(self):
        cases = [(-7, [47, 51, 54], ["Cb", "Eb", "Gb"], "Cb major", "Cb"),
                 (7, [49, 53, 56], ["C#", "E#", "G#"], "C# major", "C#")]
        for signature, pitches, spellings, expected_key, expected_symbol in cases:
            with self.subTest(key_signature=signature):
                notes = sonority(pitches, duration=4, spellings=spellings)
                estimate = estimate_key(notes, key_signature=signature)
                result = analyze_harmony(notes, key_signature=signature)
                self.assertEqual(estimate.key, expected_key)
                self.assertEqual(result.key.key, expected_key)
                self.assertEqual(result.segments[0].candidates[0].symbol, expected_symbol)
                self.assertEqual(result.segments[0].candidates[0].roman, "I")
                self.assertIn(expected_key, {candidate.key for candidate in estimate.alternatives})

    def test_inferred_extreme_sharp_signatures_keep_the_relative_minor_and_manual_key_still_wins(self):
        cases = [(6, [51, 54, 58], ["D#", "F#", "A#"], "D# minor"),
                 (7, [46, 49, 53], ["A#", "C#", "E#"], "A# minor")]
        for signature, pitches, spellings, expected_key in cases:
            with self.subTest(key_signature=signature):
                notes = sonority(pitches, duration=4, spellings=spellings)
                result = analyze_harmony(notes, key_signature=signature)
                self.assertEqual(result.key.key, expected_key)
                self.assertEqual(result.segments[0].candidates[0].roman, "i")
        cb = sonority([47, 51, 54], duration=4, spellings=["Cb", "Eb", "Gb"])
        overridden = estimate_key(cb, key="B major", key_signature=-7)
        self.assertEqual(overridden.key, "B major")
        self.assertEqual(overridden.source, "provided")
        self.assertEqual(overridden.confidence, 1.0)

    def test_estimated_note_timing_remains_explicit_and_caps_confidence(self):
        result = analyze_harmony(sonority([48, 52, 55]), key="C major", timing_is_estimated=True)
        self.assertLessEqual(result.segments[0].candidates[0].confidence, .55)
        self.assertTrue(result.segments[0].uncertain)
        self.assertTrue(any("timing is estimated" in warning for warning in result.warnings))

    def test_short_melody_neighbors_do_not_flip_the_rank_of_an_unchanged_rootless_shell(self):
        names = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")
        for shift in range(12):
            with self.subTest(transposition=shift):
                notes = [NoteEvent(62 + shift, 0, 2, voice="bass", staff="lower"),
                         NoteEvent(68 + shift, 0, 2, voice="inner", staff="lower")]
                notes += [NoteEvent(pitch + shift, onset, duration, voice="melody", staff="upper")
                          for pitch, onset, duration in ((84, .25, .25), (83, .5, .5),
                                                         (84, 1, .5), (83, 1.5, .5))]
                result = analyze_harmony(notes, key=f"{names[(8 + shift) % 12]} major",
                                         time_signature=(2, 4), total_duration=2)
                roots = {segment.candidates[0].root for segment in result.segments if segment.candidates}
                self.assertEqual(len(roots), 1)
                self.assertTrue(all(segment.uncertain for segment in result.segments))
                self.assertTrue(all(len(segment.candidates) == 2 and segment.candidates[0].rootless_shell
                                    for segment in result.segments))

    def test_score_key_can_be_estimated_once_and_reused_with_local_measure_times_and_meters(self):
        global_notes = sonority([48, 52, 55], duration=4)
        global_notes += sonority([43, 47, 50, 53], onset=4, duration=2, measure=1)
        global_notes += sonority([48, 52, 55], onset=6, duration=4, measure=2)
        key = estimate_key(global_notes, key_signature=0)
        self.assertEqual(key.key, "C major")
        local = analyze_harmony(sonority([43, 47, 50, 53], duration=2, measure=1),
                                key=key.key, time_signature=(2, 4), total_duration=2)
        self.assertEqual(local.segments[0].measure, 1)
        self.assertEqual(local.segments[0].end, 2)
        self.assertEqual(local.segments[0].candidates[0].roman, "V7")
        self.assertLessEqual(estimate_key(global_notes, timing_is_estimated=True).confidence, .35)

    def test_results_are_deterministic_and_serializable_without_mutating_notes(self):
        notes = sonority([48, 52, 55])
        before = list(notes)
        first = analyze_harmony(notes, key="C major")
        second = analyze_harmony(reversed(notes), key="C major")
        self.assertEqual(asdict(first), asdict(second))
        self.assertEqual(notes, before)
        self.assertEqual(len(first.segments[0].candidates), 3)

    def test_invalid_notes_keys_and_options_fail_explicitly(self):
        for kwargs in ({"pitch": True}, {"pitch": 128}, {"onset": -1}, {"duration": 0},
                       {"duration": math.inf}, {"onset": math.nan}, {"measure": True},
                       {"voice": 2}, {"spelling": "G"}, {"spelling": "C4"}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    NoteEvent(**{"pitch": 60, "onset": 0, "duration": 1, **kwargs})
        for kwargs in ({"key": "H major"}, {"key": "C"}, {"key": True}, {"top_n": True},
                       {"top_n": 0}, {"time_signature": (4, 3)}, {"total_duration": 1},
                       {"key_signature": True}, {"key_signature": 8}, {"timing_is_estimated": 1}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    analyze_harmony(sonority([48, 52, 55]), **kwargs)
        with self.assertRaises(ValueError):
            analyze_harmony([{"pitch": 60, "onset": 0, "duration": 1}])


if __name__ == "__main__":
    unittest.main()
