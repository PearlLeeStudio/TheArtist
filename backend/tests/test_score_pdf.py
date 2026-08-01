"""Independent translated/scaled vector geometry, without copyrighted fixtures."""

import unittest
import base64
import io
from types import SimpleNamespace
from unittest import mock

from app.services.score_pdf import ScorePDFError, _parse_pages, parse_vector_score
from tests.vector_score_fixture import build_vector_score_pdf


def geometry(*, scale=1.0, dx=0.0, dy=0.0, font="Maestro", smufl=False):
    chars, lines = [], []

    def line(x0, y0, x1, y1):
        lines.append({"x0": dx + scale * x0, "y0": dy + scale * y0,
                      "x1": dx + scale * x1, "y1": dy + scale * y1})

    def glyph(text, x, y, *, family=font, size=20, width=5):
        record = {"text": text, "fontname": family, "matrix": (size * scale, 0, 0, size * scale, dx + x * scale, dy + y * scale),
                  "x0": dx + x * scale, "x1": dx + (x + width) * scale, "size": size * scale}
        chars.append(record)
        return record

    for bottom in (280, 200):
        for index in range(5):
            # Fragment each staff line, leaving a small engraving gap.
            line(50, bottom + index * 5, 249.7, bottom + index * 5)
            line(250.3, bottom + index * 5, 450, bottom + index * 5)
    for x in (50, 250, 450):
        line(x, 200, x, 300)
    glyph("\ue050" if smufl else "&", 55, 285)
    glyph("\ue062" if smufl else "?", 55, 215)
    for bottom in (280, 200):
        glyph("\ue082" if smufl else "2", 95, bottom + 15)
        glyph("\ue084" if smufl else "4", 95, bottom + 5)
    for x in (140, 200, 300, 360):
        for y in (280, 200):
            glyph("\ue0a4" if smufl else "œ", x, y)
            line(x + 5, y, x + 5, y + 20)
    page = SimpleNamespace(chars=chars, lines=lines, curves=[], rects=[], width=600 * scale + dx, height=500 * scale + dy)
    return page, glyph, line


def octave_bracket(page, glyph, line, *, label="8va", below=False, hook=True, solid=False,
                   start=120, end=220, family="TextFont", reverse_hook=False):
    y = 180 if below else 330
    glyph(label, start, y - 2, family=family, size=15, width=20)
    line(start + 25, y, end, y)
    page.lines[-1]["dash"] = ([], 0) if solid else ([2, 2], 0)
    if hook:
        direction = 1 if below else -1
        if reverse_hook:
            direction *= -1
        line(end, y, end, y + direction * 3)
        page.lines[-1]["dash"] = ([], 0)


class VectorScoreTest(unittest.TestCase):
    def test_bounded_octave_brackets_shift_only_the_owning_staff_and_preserve_written_pitches(self):
        cases = [("8va", False, 12, 0), ("8vb", True, -12, 1),
                 ("15ma", False, 24, 0), ("15mb", True, -24, 1)]
        for scale, dx, dy in ((1, 0, 0), (.65, 14, 23), (1.7, 33, 61)):
            for label, below, shift, staff in cases:
                with self.subTest(label=label, scale=scale):
                    page, glyph, line = geometry(scale=scale, dx=dx, dy=dy)
                    octave_bracket(page, glyph, line, label=label, below=below)
                    score = _parse_pages([page])
                    self.assertEqual(score.resolved_counts["ottava_spans"], 1)
                    self.assertEqual(score.resolved_counts["octave_shifted_notes"], 2)
                    self.assertNotIn("ottava", score.unsupported_counts)
                    first = score.measures[0]
                    own = [note for note in first.notes if note.staff == staff]
                    other = [note for note in first.notes if note.staff != staff]
                    written = 64 if staff == 0 else 43
                    written_name = "E4" if staff == 0 else "G2"
                    self.assertEqual([note.midi for note in own], [written + shift] * 2)
                    self.assertTrue(all(note.written_midi == written and note.written_spelling == written_name for note in own))
                    self.assertTrue(all(note.octave_shift == shift for note in own))
                    self.assertTrue(all(note.octave_shift == 0 and note.written_midi is None for note in other))
                    self.assertTrue(all(note.octave_shift == 0 for note in score.measures[1].notes))

    def test_known_maestro_and_standard_smufl_octave_marks_use_the_same_geometry(self):
        for label, family in (("√", "Maestro"), ("\ue511", "CustomNotation")):
            with self.subTest(label=label):
                page, glyph, line = geometry()
                octave_bracket(page, glyph, line, label=label, family=family)
                score = _parse_pages([page])
                self.assertEqual([note.midi for note in score.measures[0].notes if note.staff == 0], [76, 76])
                self.assertEqual(score.resolved_counts["ottava_spans"], 1)

    def test_missing_hook_solid_line_wrong_hook_or_multiple_spans_remain_unapplied(self):
        for failure in ("missing", "solid", "reverse", "ambiguous", "cross_system"):
            with self.subTest(failure=failure):
                page, glyph, line = geometry()
                octave_bracket(page, glyph, line, hook=failure != "missing", solid=failure == "solid",
                               reverse_hook=failure == "reverse", end=500 if failure == "cross_system" else 220)
                if failure == "ambiguous":
                    line(145, 330, 230, 330)
                    page.lines[-1]["dash"] = ([2, 2], 0)
                    line(230, 330, 230, 327)
                score = _parse_pages([page])
                self.assertEqual([note.midi for note in score.measures[0].notes if note.staff == 0], [64, 64])
                self.assertEqual(score.resolved_counts.get("ottava_spans", 0), 0)
                self.assertEqual(score.unsupported_counts["ottava"], 1)
                self.assertFalse(score.measures[0].rhythm_exact)

    def test_tie_within_octave_span_carries_written_accidental_and_transposes_once(self):
        page, glyph, line = geometry()
        octave_bracket(page, glyph, line, start=180, end=320)
        glyph("b", 193, 280)
        page.curves = [{"x0": 205, "x1": 301, "y0": 282.5, "y1": 283.5, "fill": True,
                        "path": [("m", (205, 217)), ("c", (230, 210), (275, 210), (301, 217))]}]
        score = _parse_pages([page])
        previous = [note for note in score.measures[0].notes if note.staff == 0][-1]
        following = [note for note in score.measures[1].notes if note.staff == 0][0]
        self.assertEqual((previous.midi, following.midi), (75, 75))
        self.assertEqual((previous.spelling, following.spelling), ("Eb5", "Eb5"))
        self.assertEqual((previous.written_spelling, following.written_spelling), ("Eb4", "Eb4"))
        self.assertTrue(previous.tie_to_next)
        self.assertTrue(following.tie_continuation)

    def test_tie_crossing_an_octave_boundary_does_not_invent_pitch_continuity(self):
        page, glyph, line = geometry()
        octave_bracket(page, glyph, line, start=180, end=240)
        page.curves = [{"x0": 205, "x1": 301, "y0": 282.5, "y1": 283.5, "fill": True,
                        "path": [("m", (205, 217)), ("c", (230, 210), (275, 210), (301, 217))]}]
        score = _parse_pages([page])
        following = [note for note in score.measures[1].notes if note.staff == 0][0]
        self.assertEqual(following.midi, 64)
        self.assertFalse(following.tie_continuation)
        self.assertEqual(score.unsupported_counts["tie_octave_conflict"], 1)

    def test_original_type3_pdf_bytes_decode_pitches_and_actual_ctm(self):
        import pdfplumber

        cases = [("C4", "treble", 1.0, 0, 0, (4, 4), [60, 64, 67], ["C4", "E4", "G4"], 0),
                 ("F2", "bass", 1.4, 37, 29, (4, 4), [41, 45, 48], ["F2", "A2", "C3"], -1),
                 ("C4", "treble", .7, 12, 31, (2, 4), [60, 64, 67], ["C4", "E4", "G4"], 0)]
        for tonic, clef, scale, dx, dy, meter, midi, names, signature in cases:
            with self.subTest(tonic=tonic, scale=scale, meter=meter):
                data = build_vector_score_pdf(tonic=tonic, clef=clef, scale=scale,
                                              offset_x=dx, offset_y=dy, time_signature=meter)
                with pdfplumber.open(io.BytesIO(data)) as pdf:
                    music_chars = [char for char in pdf.pages[0].chars if char["text"] and 0xE000 <= ord(char["text"][0]) <= 0xE0FF]
                    self.assertTrue(music_chars)
                    self.assertFalse(any("Maestro" in char["fontname"] for char in music_chars))
                score = parse_vector_score(data)
                self.assertTrue(score.supported)
                self.assertTrue(score.complete)
                self.assertEqual(score.title, "Original vector score")
                self.assertEqual(score.key_signature, signature)
                self.assertIsNone(score.key)
                self.assertEqual(score.time_signature, meter)
                self.assertEqual(len(score.measures), 1)
                bar = score.measures[0]
                self.assertTrue(bar.rhythm_exact, bar.warnings)
                self.assertEqual([note.midi for note in bar.notes], midi)
                self.assertEqual([note.spelling for note in bar.notes], names)
                self.assertTrue(all(note.onset == 0 and note.duration == meter[0] for note in bar.notes))

    def test_original_pdf_through_the_import_service_yields_tonic_and_roman(self):
        from app.models.score_import import ScoreImportRequest
        from app.services.harmony_analysis import NoteEvent, analyze_harmony
        from app.services.score_import import import_score

        for tonic, clef, key, chord in (("C4", "treble", "C major", "C"), ("F2", "bass", "F major", "F")):
            with self.subTest(tonic=tonic):
                data = build_vector_score_pdf(tonic=tonic, clef=clef, scale=1.2, offset_x=17, offset_y=23)
                request = ScoreImportRequest(filename="original.pdf", mimeType="application/pdf",
                                             dataBase64=base64.b64encode(data).decode(), mode="infer_harmony")
                response = import_score(request)
                self.assertEqual(response.analysisSource, "local")
                self.assertEqual(response.key, key)
                self.assertEqual(response.measures[0].chords, [chord])
                bar = parse_vector_score(data).measures[0]
                events = [NoteEvent(note.midi, note.onset, note.duration, bar.index, staff=str(note.staff)) for note in bar.notes]
                analysis = analyze_harmony(events, key=key, time_signature=bar.time_signature, total_duration=4)
                self.assertEqual(analysis.segments[0].candidates[0].symbol, chord)
                self.assertEqual(analysis.segments[0].candidates[0].roman, "I")

    def test_duplicated_clef_glyph_at_the_same_position_reads_like_a_single_clef(self):
        import pdfplumber

        cases = [("C4", "treble", "\ue050", ["C4", "E4", "G4"]), ("F2", "bass", "\ue062", ["F2", "A2", "C3"])]
        for tonic, clef, code, names in cases:
            with self.subTest(clef=clef):
                data = build_vector_score_pdf(tonic=tonic, clef=clef)
                with pdfplumber.open(io.BytesIO(data)) as pdf:
                    source = pdf.pages[0]
                    clefs = [char for char in source.chars if char["text"] == code]
                    self.assertEqual(len(clefs), 1)
                    # A layer that redraws the clef from its own font resource leaves a second
                    # record at the same position; an identical record would compare equal.
                    twin = dict(clefs[0], fontname=clefs[0]["fontname"] + "-Layer")
                    page = SimpleNamespace(chars=source.chars + [twin], lines=source.lines, curves=source.curves,
                                           rects=source.rects, width=source.width, height=source.height)
                single = parse_vector_score(data)
                pdf = mock.MagicMock()
                pdf.__enter__.return_value = pdf
                pdf.pages = [page]
                with mock.patch("app.services.score_pdf.pdfplumber.open", return_value=pdf):
                    duplicated = parse_vector_score(data)
                self.assertEqual([note.spelling for note in duplicated.measures[0].notes], names)
                self.assertEqual(duplicated.measures, single.measures)
                self.assertEqual(duplicated, single)

    def test_pitch_meter_and_measure_clock_survive_scale_and_translation(self):
        for scale, dx, dy in ((1, 0, 0), (1.7, 33, 81), (.65, 12, 17)):
            with self.subTest(scale=scale):
                page, _, _ = geometry(scale=scale, dx=dx, dy=dy, font="mAeStRo-Subset")
                score = _parse_pages([page])
                self.assertTrue(score.supported)
                self.assertTrue(score.complete)
                self.assertEqual(len(score.measures), 2)
                self.assertEqual(score.time_signature, (2, 4))
                self.assertIsNone(score.key)
                self.assertTrue(all(bar.rhythm_exact for bar in score.measures))
                for bar in score.measures:
                    self.assertEqual([(n.staff, n.spelling, n.midi, n.onset, n.duration) for n in bar.notes],
                                     [(0, "E4", 64, 0, 1), (1, "G2", 43, 0, 1),
                                      (0, "E4", 64, 1, 1), (1, "G2", 43, 1, 1)])

    def test_standard_smufl_symbols_do_not_depend_on_font_family(self):
        page, _, _ = geometry(font="CustomNotation", smufl=True)
        score = _parse_pages([page])
        self.assertTrue(score.supported)
        self.assertEqual(score.time_signature, (2, 4))
        self.assertEqual(score.measures[0].notes[0].spelling, "E4")
        self.assertEqual(score.measures[0].notes[1].spelling, "G2")

    def test_unknown_legacy_font_is_not_treated_as_known_music(self):
        page, _, _ = geometry(font="UnknownLegacyFont")
        score = _parse_pages([page])
        self.assertFalse(score.supported)
        self.assertFalse(score.complete)
        self.assertGreater(score.unsupported_counts["unknown_legacy_music_font"], 0)

    def test_key_accidentals_reset_at_barline_and_clef_changes_apply_locally(self):
        page, glyph, _ = geometry()
        for bottom in (280, 200):
            for index, height in enumerate((10, 17.5, 7.5)):
                glyph("b", 70 + index * 5, bottom + height)
        glyph("n", 133, 280)
        glyph("&", 265, 205)
        score = _parse_pages([page])
        self.assertEqual(score.key_signature, -3)
        self.assertIsNone(score.key)
        self.assertEqual([n.spelling for n in score.measures[0].notes if n.staff == 0], ["E4", "E4"])
        self.assertEqual([n.spelling for n in score.measures[1].notes if n.staff == 0], ["Eb4", "Eb4"])
        self.assertEqual([n.spelling for n in score.measures[1].notes if n.staff == 1], ["Eb4", "Eb4"])

    def test_ledger_lines_and_c_clef_position(self):
        page, glyph, _ = geometry()
        for char in page.chars:
            if char["text"] == "œ" and char["matrix"][4] == 140 and char["matrix"][5] == 280:
                char["matrix"] = (20, 0, 0, 20, 140, 307.5)
        glyph("B", 120, 215)  # Tenor C clef, C4 on the fourth line.
        score = _parse_pages([page])
        self.assertIn("B5", [note.spelling for note in score.measures[0].notes])
        self.assertIn("D3", [note.spelling for note in score.measures[0].notes if note.staff == 1])

    def test_chord_font_runs_slash_accidentals_and_no_chord_are_preserved(self):
        page, glyph, _ = geometry()
        glyph("N.C.", 70, 310, family="TextFont", size=10, width=25)
        glyph("G", 300, 310, family="TextFont", size=10, width=6)
        glyph("13", 308, 311, family="TextFont", size=8, width=8)
        glyph("A/C", 360, 310, family="TextFont", size=10, width=20)
        glyph("#", 381, 314, size=15, width=4)
        score = _parse_pages([page])
        self.assertTrue(score.measures[0].no_chord)
        self.assertEqual([chord.symbol for chord in score.measures[1].chords], ["G13", "A/C#"])

    def test_meter_changes_and_repeat_signs_remain_in_written_order(self):
        page, glyph, _ = geometry()
        glyph("c", 260, 290)
        glyph("%", 320, 290)
        score = _parse_pages([page])
        self.assertEqual(len(score.measures), 2)
        self.assertEqual([bar.time_signature for bar in score.measures], [(2, 4), (4, 4)])
        self.assertEqual(score.measures[1].repeat_markers, ["%"])
        self.assertIn("unexpanded_repeat", score.unsupported_counts)

    def test_unsupported_ties_tuplets_and_ottava_are_exposed(self):
        page, glyph, _ = geometry()
        page.curves = [{"x0": 150, "x1": 220, "y0": 286, "y1": 291, "fill": False,
                        "path": [("m", (150, 286)), ("c", (170, 291), (200, 291), (220, 286))]}]
        glyph("3", 180, 310)
        glyph("8va", 175, 330, family="TextFont", size=10, width=15)
        score = _parse_pages([page])
        self.assertFalse(score.measures[0].rhythm_exact)
        for name in ("tie_or_slur", "possible_tuplet", "ottava"):
            self.assertGreater(score.unsupported_counts[name], 0)
        self.assertTrue(score.measures[0].staff_warnings)

    def test_beam_counts_and_rests_produce_quarter_beat_onsets(self):
        page, glyph, _ = geometry()
        page.curves = [{"x0": 140, "x1": 205, "y0": 298, "y1": 300,
                        "fill": True, "path": [("m", (140, 298)), ("l", (205, 298)),
                                                ("l", (205, 300)), ("l", (140, 300))]}]
        glyph("Œ", 220, 290)
        score = _parse_pages([page])
        upper = [note for note in score.measures[0].notes if note.staff == 0]
        self.assertEqual([(note.onset, note.duration) for note in upper], [(0, .5), (.5, .5)])
        self.assertTrue(score.measures[0].rhythm_exact)

    def test_key_signature_changes_and_cancellation_are_applied_before_notes(self):
        page, glyph, _ = geometry()
        for bottom in (280, 200):
            glyph("b", 70, bottom + 10)
            glyph("n", 270, bottom + 10)
        # B3 in upper staff is flat in the first measure and natural after cancellation.
        for char in page.chars:
            if char["text"] == "œ" and char["matrix"][5] == 280:
                a, b, c, d, x, _ = char["matrix"]
                char["matrix"] = (a, b, c, d, x, 272.5)
        score = _parse_pages([page])
        self.assertEqual([bar.key_signature for bar in score.measures], [-1, 0])
        self.assertEqual([note.spelling for note in score.measures[0].notes if note.staff == 0], ["Bb3", "Bb3"])
        self.assertEqual([note.spelling for note in score.measures[1].notes if note.staff == 0], ["B3", "B3"])

    def test_clear_cross_bar_tie_carries_accidental_and_marks_duration_continuity(self):
        page, glyph, _ = geometry()
        glyph("b", 193, 280)
        page.curves = [{"x0": 205, "x1": 301, "y0": 282.5, "y1": 283.5, "fill": True,
                        "path": [("m", (205, 217)), ("c", (230, 210), (275, 210), (301, 217)),
                                 ("l", (300, 218)), ("c", (275, 212), (230, 212), (205, 218))]}]
        score = _parse_pages([page])
        previous = [note for note in score.measures[0].notes if note.staff == 0][-1]
        following = [note for note in score.measures[1].notes if note.staff == 0]
        self.assertEqual(previous.spelling, "Eb4")
        self.assertTrue(previous.tie_to_next)
        self.assertEqual(following[0].spelling, "Eb4")
        self.assertEqual(following[0].midi, previous.midi)
        self.assertTrue(following[0].tie_continuation)
        self.assertEqual(previous.duration, 1)
        self.assertEqual(following[0].duration, 1)
        self.assertEqual(following[1].spelling, "E4")

    def test_tie_with_conflicting_written_accidental_requires_review(self):
        page, glyph, _ = geometry()
        glyph("b", 193, 280)
        glyph("n", 293, 280)
        page.curves = [{"x0": 205, "x1": 301, "y0": 282.5, "y1": 283.5, "fill": True,
                        "path": [("m", (205, 217)), ("c", (230, 210), (275, 210), (301, 217))]}]
        score = _parse_pages([page])
        following = [note for note in score.measures[1].notes if note.staff == 0][0]
        self.assertEqual(following.spelling, "E4")
        self.assertFalse(following.tie_continuation)
        self.assertEqual(score.unsupported_counts["tie_accidental_conflict"], 1)
        self.assertFalse(score.measures[1].rhythm_exact)

    def test_slur_over_intermediate_notes_does_not_carry_an_accidental(self):
        page, glyph, _ = geometry()
        glyph("b", 193, 280)
        page.curves = [{"x0": 205, "x1": 361, "y0": 282.5, "y1": 283.5, "fill": True,
                        "path": [("m", (205, 217)), ("c", (230, 210), (325, 210), (361, 217))]}]
        score = _parse_pages([page])
        following = [note for note in score.measures[1].notes if note.staff == 0]
        self.assertEqual([note.spelling for note in following], ["E4", "E4"])
        self.assertFalse(any(note.tie_continuation for note in following))
        self.assertGreater(score.unsupported_counts["tie_or_slur"], 0)

    def test_unsupported_standard_notehead_cannot_silently_complete(self):
        page, glyph, _ = geometry(font="CustomNotation", smufl=True)
        glyph("\ue0a9", 170, 280)
        score = _parse_pages([page])
        self.assertFalse(score.complete)
        self.assertEqual(score.unsupported_counts["unsupported_notehead"], 1)

    def test_rotated_music_and_unsupported_composite_notes_cannot_silently_complete(self):
        page, glyph, _ = geometry(font="CustomNotation", smufl=True)
        glyph("\ue1d5", 170, 280)
        for char in page.chars:
            if char["text"] == "\ue0a4" and char["matrix"][4] == 140:
                char["matrix"] = (0, 20, -20, 0, char["matrix"][4], char["matrix"][5])
        score = _parse_pages([page])
        self.assertFalse(score.complete)
        self.assertGreater(score.unsupported_counts["rotated_music_glyph"], 0)
        self.assertEqual(score.unsupported_counts["unsupported_composite_note"], 1)

    def test_standard_augmentation_dot_is_not_an_unsupported_composite_note(self):
        page, glyph, _ = geometry(font="CustomNotation", smufl=True)
        glyph("\ue1e7", 148, 282.5)
        score = _parse_pages([page])
        self.assertTrue(score.complete)
        self.assertNotIn("unsupported_composite_note", score.unsupported_counts)
        self.assertEqual([note for note in score.measures[0].notes if note.staff == 0][0].duration, 1.5)

    def test_measure_limit_is_explicit_and_partial_result_is_not_complete(self):
        page, _, _ = geometry()
        score = _parse_pages([page], max_measures=1)
        self.assertEqual(len(score.measures), 1)
        self.assertFalse(score.complete)
        self.assertIn("measure_limit", score.unsupported_counts)

    def test_pdf_entrypoint_rejects_invalid_input_and_bounds(self):
        for data in (b"", b"not pdf", b"%PDF-not valid"):
            with self.subTest(data=data):
                with self.assertRaises(ScorePDFError):
                    parse_vector_score(data)
        with self.assertRaises(ScorePDFError):
            parse_vector_score(b"%PDF-", max_pages=51)
        with self.assertRaises(ScorePDFError):
            parse_vector_score(b"%PDF-", max_measures=513)
        pdf = mock.MagicMock()
        pdf.__enter__.return_value = pdf
        pdf.pages = [object(), object()]
        with mock.patch("app.services.score_pdf.pdfplumber.open", return_value=pdf):
            with self.assertRaisesRegex(ScorePDFError, "1 pages"):
                parse_vector_score(b"%PDF-", max_pages=1)


if __name__ == "__main__":
    unittest.main()
