import unittest

from app.models.schemas import Suggestion
from app.models.sections import SECTION_TYPES
from app.services.key_anchor import uses_sharps
from app.services.song_form import (
    NOTE_TO_SEMI,
    ROOT_POSITION_ENDINGS,
    SECTION_RULES,
    apply_song_form,
    apply_song_form_to_suggestions,
)

INPUTS = {
    "one": "Dm7",
    "two": "Am7 Dm7",
    "eight": "Am F C G Em Am Dm7 G7",
    "iv_first": "Fmaj7 G7 Em7 Am7",
    "slash": "C/E F/A G7/B Am",
}

# apply_song_form output for the five original sections, captured from the
# module before the section vocabulary grew: (section, key, minor, input).
LEGACY_GOLDEN = {
    ("intro", "C", False, "one"): "Cmaj7",
    ("intro", "C", False, "two"): "Cmaj7 Dm7",
    ("intro", "C", False, "eight"): "Cmaj7 F C G Em Am Dm7 G7",
    ("intro", "C", False, "iv_first"): "Cmaj7 G7 Em7 Am7",
    ("intro", "C", False, "slash"): "C/E F/A G7/B Am",
    ("intro", "A", True, "one"): "Am7",
    ("intro", "A", True, "two"): "Am7 Dm7",
    ("intro", "A", True, "eight"): "Am F C G Em Am Dm7 G7",
    ("intro", "A", True, "iv_first"): "Am7 G7 Em7 Am7",
    ("intro", "A", True, "slash"): "Am7 F/A G7/B Am",
    ("intro", "F#", False, "eight"): "F#maj7 F C G Em Am Dm7 G7",
    ("intro", "Bb", False, "eight"): "Bbmaj7 F C G Em Am Dm7 G7",
    ("intro", "E", True, "eight"): "Em7 F C G Em Am Dm7 G7",
    ("intro", "C#", True, "eight"): "C#m7 F C G Em Am Dm7 G7",
    ("verse", "C", False, "one"): "Dm7",
    ("verse", "C", False, "two"): "G7 Cmaj",
    ("verse", "C", False, "eight"): "Am F C G Em Am G7 Cmaj",
    ("verse", "C", False, "iv_first"): "Fmaj7 G7 G7 Cmaj",
    ("verse", "C", False, "slash"): "C/E F/A G7 Cmaj",
    ("verse", "A", True, "one"): "Dm7",
    ("verse", "A", True, "two"): "E7 Amin",
    ("verse", "A", True, "eight"): "Am F C G Em Am E7 Amin",
    ("verse", "A", True, "iv_first"): "Fmaj7 G7 E7 Amin",
    ("verse", "A", True, "slash"): "C/E F/A E7 Amin",
    ("verse", "F#", False, "eight"): "Am F C G Em Am C#7 F#maj",
    ("verse", "Bb", False, "eight"): "Am F C G Em Am F7 Bbmaj",
    ("verse", "E", True, "eight"): "Am F C G Em Am B7 Emin",
    ("verse", "C#", True, "eight"): "Am F C G Em Am G#7 C#min",
    ("chorus", "C", False, "one"): "Dm7",
    ("chorus", "C", False, "two"): "G7 Cmaj",
    ("chorus", "C", False, "eight"): "Am F C G Em Am G7 Cmaj",
    ("chorus", "C", False, "iv_first"): "Fmaj7 G7 G7 Cmaj",
    ("chorus", "C", False, "slash"): "C/E F/A G7 Cmaj",
    ("chorus", "A", True, "one"): "Dm7",
    ("chorus", "A", True, "two"): "E7 Amin",
    ("chorus", "A", True, "eight"): "Am F C G Em Am E7 Amin",
    ("chorus", "A", True, "iv_first"): "Fmaj7 G7 E7 Amin",
    ("chorus", "A", True, "slash"): "C/E F/A E7 Amin",
    ("chorus", "F#", False, "eight"): "Am F C G Em Am C#7 F#maj",
    ("chorus", "Bb", False, "eight"): "Am F C G Em Am F7 Bbmaj",
    ("chorus", "E", True, "eight"): "Am F C G Em Am B7 Emin",
    ("chorus", "C#", True, "eight"): "Am F C G Em Am G#7 C#min",
    ("bridge", "C", False, "one"): "Fmaj7",
    ("bridge", "C", False, "two"): "Fmaj7 G7",
    ("bridge", "C", False, "eight"): "Fmaj7 F C G Em Am Fmaj7 G7",
    ("bridge", "C", False, "iv_first"): "Fmaj7 G7 Fmaj7 G7",
    ("bridge", "C", False, "slash"): "Fmaj7 F/A Fmaj7 G7",
    ("bridge", "A", True, "one"): "Dm7",
    ("bridge", "A", True, "two"): "Dm7 E7",
    ("bridge", "A", True, "eight"): "Dm7 F C G Em Am Dm7 E7",
    ("bridge", "A", True, "iv_first"): "Dm7 G7 Dm7 E7",
    ("bridge", "A", True, "slash"): "Dm7 F/A Dm7 E7",
    ("bridge", "F#", False, "eight"): "Bmaj7 F C G Em Am Bmaj7 C#7",
    ("bridge", "Bb", False, "eight"): "Ebmaj7 F C G Em Am Ebmaj7 F7",
    ("bridge", "E", True, "eight"): "Am F C G Em Am Am7 B7",
    ("bridge", "C#", True, "eight"): "F#m7 F C G Em Am F#m7 G#7",
    ("outro", "C", False, "one"): "Cmaj",
    ("outro", "C", False, "two"): "Fmaj7 Cmaj",
    ("outro", "C", False, "eight"): "Am F C G Em Am Fmaj7 Cmaj",
    ("outro", "C", False, "iv_first"): "Fmaj7 G7 Fmaj7 Cmaj",
    ("outro", "C", False, "slash"): "C/E F/A Fmaj7 Cmaj",
    ("outro", "A", True, "one"): "Amin",
    ("outro", "A", True, "two"): "Dm7 Amin",
    ("outro", "A", True, "eight"): "Am F C G Em Am Dm7 Amin",
    ("outro", "A", True, "iv_first"): "Fmaj7 G7 Dm7 Amin",
    ("outro", "A", True, "slash"): "C/E F/A Dm7 Amin",
    ("outro", "F#", False, "eight"): "Am F C G Em Am Bmaj7 F#maj",
    ("outro", "Bb", False, "eight"): "Am F C G Em Am Ebmaj7 Bbmaj",
    ("outro", "E", True, "eight"): "Am F C G Em Am Am7 Emin",
    ("outro", "C#", True, "eight"): "Am F C G Em Am F#m7 C#min",
    # Keys where key_anchor would spell with sharps: the original sections keep their flats.
    ("intro", "G#", True, "eight"): "Abm7 F C G Em Am Dm7 G7",
    ("verse", "G#", True, "eight"): "Am F C G Em Am Eb7 Abmin",
    ("chorus", "D#", True, "eight"): "Am F C G Em Am Bb7 Ebmin",
    ("bridge", "G#", True, "eight"): "Dbm7 F C G Em Am Dbm7 Eb7",
    ("outro", "D#", True, "eight"): "Am F C G Em Am Abm7 Ebmin",
    ("verse", "A#", False, "eight"): "Am F C G Em Am F7 Bbmaj",
}
ORIGINAL_SECTIONS = {section for section, *_ in LEGACY_GOLDEN}

# A 16-chord phrase (8 bars) that starts and ends away from any forced chord.
# Every root is natural, so an accidental in a rule's output is one the rule wrote.
PHRASE = "Em7 Am7 Dm7 G7 Fmaj7 Em7 Dm7 Am7 Bm7b5 E7 Am F Dm Em F Em".split()
KEYS = [("C", False), ("A", True)]


def form(section: str, key: str, minor: bool, chords: list[str]) -> list[str]:
    return apply_song_form(list(chords), section, key, minor)


class LegacyRulesTest(unittest.TestCase):
    def test_original_sections_match_captured_output(self):
        for (section, key, minor, name), expected in LEGACY_GOLDEN.items():
            with self.subTest(section=section, key=key, minor=minor, input=name):
                got = form(section, key, minor, INPUTS[name].split())
                self.assertEqual(got, expected.split())


class ChordCountTest(unittest.TestCase):
    def test_every_section_keeps_the_chord_count(self):
        for section in SECTION_TYPES:
            for key, minor in KEYS:
                for n in (1, 2, 3, 4, 5, 8, 16):
                    with self.subTest(section=section, key=key, n=n):
                        self.assertEqual(len(form(section, key, minor, PHRASE[:n])), n)

    def test_long_sections_keep_the_chord_count(self):
        chords = (PHRASE * 8)[:128]
        for section in SECTION_TYPES:
            with self.subTest(section=section):
                self.assertEqual(len(form(section, "C", False, chords)), 128)

    def test_every_section_id_has_a_rule(self):
        self.assertEqual(set(SECTION_RULES), set(SECTION_TYPES))

    def test_unknown_and_empty_sections_pass_through(self):
        self.assertEqual(form("coda", "C", False, PHRASE), PHRASE)
        self.assertEqual(form("", "C", False, PHRASE), PHRASE)
        self.assertEqual(apply_song_form([], "chorus", "C"), [])


class NewSectionRulesTest(unittest.TestCase):
    def test_pre_chorus_ends_on_the_dominant(self):
        for (key, minor), v7 in zip(KEYS, ("G7", "E7")):
            got = form("pre_chorus", key, minor, PHRASE)
            self.assertEqual(got[-1], v7)
            self.assertEqual(got[:-1], PHRASE[:-1])

    def test_post_chorus_and_drop_loop_the_first_two_bars(self):
        for section in ("post_chorus", "drop"):
            for key, minor in KEYS:
                with self.subTest(section=section, key=key):
                    got = form(section, key, minor, PHRASE)
                    self.assertEqual(got, PHRASE[:4] * 4)
        self.assertEqual(form("drop", "C", False, PHRASE[:6]), PHRASE[:4] + PHRASE[:2])

    def test_breakdown_and_vamp_loop_the_first_bar(self):
        for section in ("breakdown", "vamp"):
            for key, minor in KEYS:
                with self.subTest(section=section, key=key):
                    self.assertEqual(form(section, key, minor, PHRASE), PHRASE[:2] * 8)

    def test_turnaround_ends_i_vi_ii_v(self):
        got = form("turnaround", "C", False, PHRASE)
        self.assertEqual(got[-4:], ["Cmaj7", "Am7", "Dm7", "G7"])
        self.assertEqual(got[:-4], PHRASE[:-4])
        got = form("turnaround", "A", True, PHRASE)
        self.assertEqual(got[-4:], ["Am7", "Fmaj7", "Bm7b5", "E7"])
        self.assertEqual(got[:-4], PHRASE[:-4])

    def test_short_turnaround_just_ends_on_v7(self):
        self.assertEqual(form("turnaround", "C", False, ["Em7", "Am7"]), ["Em7", "G7"])
        self.assertEqual(form("turnaround", "A", True, ["Em7", "Am7", "Dm7"]), ["Em7", "Am7", "E7"])

    def test_tag_closes_on_a_bare_tonic_after_v7(self):
        for (key, minor), ending in zip(KEYS, (["G7", "Cmaj"], ["E7", "Amin"])):
            got = form("tag", key, minor, PHRASE)
            self.assertEqual(got[-2:], ending)
            self.assertEqual(got[:-2], PHRASE[:-2])

    def test_head_and_a_section_cadence_like_verse(self):
        for section in ("head", "a_section"):
            for key, minor in KEYS:
                with self.subTest(section=section, key=key):
                    self.assertEqual(form(section, key, minor, PHRASE), form("verse", key, minor, PHRASE))
        self.assertEqual(form("head", "C", False, PHRASE)[-2:], ["G7", "Cmaj"])

    def test_b_section_follows_the_bridge(self):
        for key, minor in KEYS:
            with self.subTest(key=key):
                self.assertEqual(form("b_section", key, minor, PHRASE), form("bridge", key, minor, PHRASE))
        got = form("b_section", "C", False, PHRASE)
        self.assertEqual([got[0], got[-2], got[-1]], ["Fmaj7", "Fmaj7", "G7"])

    def test_interlude_and_solo_pass_through(self):
        for section in ("interlude", "solo"):
            for key, minor in KEYS:
                with self.subTest(section=section, key=key):
                    self.assertEqual(form(section, key, minor, PHRASE), PHRASE)

    def test_root_position_endings_are_the_resolving_sections(self):
        self.assertEqual(ROOT_POSITION_ENDINGS, {"verse", "chorus", "outro", "tag", "head", "a_section"})


class KeySignatureSpellingTest(unittest.TestCase):
    """The newer sections spell what they write like key_anchor (sharp keys use sharps)."""

    def test_sharp_keys_write_sharps(self):
        self.assertEqual(form("turnaround", "E", False, PHRASE[:4]), ["Emaj7", "C#m7", "F#m7", "B7"])
        self.assertEqual(form("turnaround", "A", False, PHRASE[:4]), ["Amaj7", "F#m7", "Bm7", "E7"])
        self.assertEqual(form("turnaround", "E", True, PHRASE[:4]), ["Em7", "Cmaj7", "F#m7b5", "B7"])
        self.assertEqual(form("turnaround", "B", False, PHRASE[:4]), ["Bmaj7", "G#m7", "C#m7", "F#7"])
        self.assertEqual(form("tag", "G#", True, PHRASE)[-2:], ["D#7", "G#min"])

    def test_flat_keys_stay_flat(self):
        self.assertEqual(form("turnaround", "D", True, PHRASE[:4]), ["Dm7", "Bbmaj7", "Em7b5", "A7"])
        self.assertEqual(form("turnaround", "Bb", False, PHRASE[:4]), ["Bbmaj7", "Gm7", "Cm7", "F7"])
        got = form("b_section", "F", False, PHRASE)
        self.assertEqual([got[0], got[-2], got[-1]], ["Bbmaj7", "Bbmaj7", "C7"])

    def test_every_written_chord_follows_key_anchor(self):
        for section in SECTION_TYPES:
            if section in ORIGINAL_SECTIONS:
                continue
            for key in NOTE_TO_SEMI:
                for minor in (False, True):
                    wrong = "b" if uses_sharps(key, minor) else "#"
                    with self.subTest(section=section, key=key, minor=minor):
                        for n in (1, 2, 4, 16):
                            got = form(section, key, minor, PHRASE[:n])
                            self.assertEqual([c for c in got if c[1:2] == wrong], [], got)


class SuggestionHelperTest(unittest.TestCase):
    def test_rules_run_over_the_whole_selection_and_keep_measures(self):
        s = Suggestion(label="A", chords={"10": ["Em7", "Am7"], "8": ["Dm7", "G7"], "9": ["Fmaj7", "Em7"]})
        apply_song_form_to_suggestions([s], "outro", "C", False)
        self.assertEqual(s.chords, {"8": ["Dm7", "G7"], "9": ["Fmaj7", "Em7"], "10": ["Fmaj7", "Cmaj"]})

    def test_no_section_is_a_no_op(self):
        s = Suggestion(label="A", chords={"0": ["Em7", "Am7"]})
        apply_song_form_to_suggestions([s], None, "C", False)
        self.assertEqual(s.chords, {"0": ["Em7", "Am7"]})


if __name__ == "__main__":
    unittest.main()
