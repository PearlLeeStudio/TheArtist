import unittest

from app.services.key_anchor import anchor_to_key, uses_sharps

# 12-bar blues in C, two chords a bar, ending on the V7 turnaround.
C_BLUES = (["C7"] * 2 + ["F7"] * 2 + ["C7"] * 4 + ["F7"] * 4 + ["C7"] * 4
           + ["G7"] * 2 + ["F7"] * 2 + ["C7"] * 2 + ["G7"] * 2)


class AnchorToKeyTest(unittest.TestCase):
    def test_e_major_candidate_moves_to_c_major(self):
        # The model spells C#m as Dbm (its vocabulary prefers flats).
        chords = ["Emaj", "Amaj", "Emaj", "Bmaj", "Dbm", "Amaj", "Bmaj", "Emaj"]
        self.assertEqual(
            anchor_to_key(chords, "C", False),
            ["Cmaj", "Fmaj", "Cmaj", "Gmaj", "Am", "Fmaj", "Gmaj", "Cmaj"],
        )

    def test_slash_basses_move_with_their_chords(self):
        chords = ["Emaj", "Amaj/E", "Emaj", "Bmaj/Eb", "Dbm", "Amaj", "Bmaj", "Emaj"]
        self.assertEqual(
            anchor_to_key(chords, "C", False),
            ["Cmaj", "Fmaj/C", "Cmaj", "Gmaj/B", "Am", "Fmaj", "Gmaj", "Cmaj"],
        )

    def test_c_major_candidate_is_unchanged(self):
        chords = ["Cmaj7", "Am7", "Dm7", "G7", "Em7", "Fmaj7", "G7", "Cmaj"]
        self.assertEqual(anchor_to_key(chords, "C", False), chords)

    def test_relative_minor_progression_is_unchanged(self):
        chords = ["Am", "F", "C", "G"]
        self.assertEqual(anchor_to_key(chords, "C", False), chords)
        self.assertEqual(anchor_to_key(chords, "A", True), chords)

    def test_one_out_of_key_tone_never_moves_a_progression(self):
        # I-IV-V in G has a single F# (in D) against C major: a near-tie.
        chords = ["Gmaj", "Cmaj", "Dmaj", "Gmaj"]
        self.assertEqual(anchor_to_key(chords, "C", False), chords)
        # A secondary dominant (D7) balanced by G7's F stays put too.
        chords = ["Cmaj", "Am", "D7", "G7"]
        self.assertEqual(anchor_to_key(chords, "C", False), chords)

    def test_in_key_rock_and_blues_are_unchanged(self):
        # bVII and dominant sevenths fit F major's collection better than C's,
        # but these centre on that collection's 5th degree: the requested tonic.
        for chords in (
            ["Cmaj", "Bbmaj", "Fmaj", "Cmaj"] * 2,
            ["Cmaj", "Bbmaj"] * 4,  # 17% of its chord tones outside C major
            ["C7", "F7", "C7", "G7"] * 2,
            C_BLUES,
        ):
            with self.subTest(chords=chords):
                self.assertEqual(anchor_to_key(chords, "C", False), chords)
        self.assertEqual(anchor_to_key(["Gmaj", "Fmaj", "Cmaj", "Gmaj"] * 2, "G", False),
                         ["Gmaj", "Fmaj", "Cmaj", "Gmaj"] * 2)

    def test_out_of_key_rock_and_blues_land_on_the_requested_tonic(self):
        # An E blues (A major's collection) asked for in C major: E becomes C, not G.
        chords = ["Emaj", "Amaj", "Emaj", "B7", "A7", "E7", "A7", "Emaj",
                  "B7", "A7", "E7", "A7", "Emaj", "Dmaj", "Amaj", "Emaj"]
        self.assertEqual(
            anchor_to_key(chords, "C", False),
            ["Cmaj", "Fmaj", "Cmaj", "G7", "F7", "C7", "F7", "Cmaj",
             "G7", "F7", "C7", "F7", "Cmaj", "Bbmaj", "Fmaj", "Cmaj"],
        )
        # D-C-G-D rock (G major's collection) becomes C-Bb-F-C.
        self.assertEqual(
            anchor_to_key(["Dmaj", "Cmaj", "Gmaj", "Dmaj"] * 2, "C", False),
            ["Cmaj", "Bbmaj", "Fmaj", "Cmaj"] * 2,
        )
        # The same 12-bar blues written in Eb comes back in C.
        eb_blues = [{"C7": "Eb7", "F7": "Ab7", "G7": "Bb7"}[c] for c in C_BLUES]
        self.assertEqual(anchor_to_key(eb_blues, "C", False), C_BLUES)

    def test_major_key_progression_opening_on_v_keeps_the_collection_shift(self):
        # V-I-vi-IV in G opens on D but neither closes there nor dwells on it.
        chords = ["Dmaj", "Gmaj", "Em", "Cmaj"] * 2
        self.assertEqual(anchor_to_key(chords, "C", False), ["Gmaj", "Cmaj", "Am", "Fmaj"] * 2)

    def test_minor_request_puts_a_mixolydian_centre_on_its_tonic(self):
        # E mixolydian for A minor lands on A, spelled like A major (F#m, not Gbm).
        chords = ["Emaj", "Dmaj", "Amaj", "Emaj", "Emaj", "Dbm", "Dmaj", "Emaj"]
        self.assertEqual(
            anchor_to_key(chords, "A", True),
            ["Amaj", "Gmaj", "Dmaj", "Amaj", "Amaj", "F#m", "Gmaj", "Amaj"],
        )

    def test_minor_request_moves_by_the_relative_major(self):
        # A minor candidate (C major collection) for E minor (G major collection).
        chords = ["Am", "Bm7b5", "E7", "Am", "Fmaj7", "Dm", "E7", "Am"]
        self.assertEqual(
            anchor_to_key(chords, "E", True),
            ["Em", "F#m7b5", "B7", "Em", "Cmaj7", "Am", "B7", "Em"],
        )
        # ... and for C minor (Eb major collection), spelled with flats.
        self.assertEqual(
            anchor_to_key(chords, "C", True),
            ["Cm", "Dm7b5", "G7", "Cm", "Abmaj7", "Fm", "G7", "Cm"],
        )

    def test_sharp_key_spells_with_sharps(self):
        chords = ["Fmaj", "Bbmaj", "Cmaj", "Dm", "Fmaj", "Am", "Gm", "C7"]
        self.assertEqual(
            anchor_to_key(chords, "D", False),
            ["Dmaj", "Gmaj", "Amaj", "Bm", "Dmaj", "F#m", "Em", "A7"],
        )

    def test_flat_key_spells_with_flats(self):
        chords = ["Amaj", "Dmaj", "Emaj", "F#m", "Amaj", "C#m", "Dmaj", "Emaj"]
        self.assertEqual(
            anchor_to_key(chords, "F", False),
            ["Fmaj", "Bbmaj", "Cmaj", "Dm", "Fmaj", "Am", "Bbmaj", "Cmaj"],
        )

    def test_unparseable_chords_pass_through(self):
        chords = ["N.C.", "Emaj", "Amaj", "Bmaj", "Emaj"]
        self.assertEqual(anchor_to_key(chords, "C", False), ["N.C.", "Cmaj", "Fmaj", "Gmaj", "Cmaj"])
        self.assertEqual(anchor_to_key(["N.C."], "C", False), ["N.C."])
        self.assertEqual(anchor_to_key([], "C", False), [])

    def test_unknown_key_root_is_a_no_op(self):
        chords = ["Emaj", "Amaj", "Bmaj", "Emaj"]
        self.assertEqual(anchor_to_key(chords, "H", False), chords)

    def test_pure_function(self):
        chords = ["Emaj", "Amaj", "Bmaj", "Emaj"]
        before = list(chords)
        out = anchor_to_key(chords, "C", False)
        self.assertEqual(chords, before)
        self.assertIsNot(out, chords)
        unchanged = ["Cmaj", "Fmaj"]
        self.assertIsNot(anchor_to_key(unchanged, "C", False), unchanged)
        in_key_blues = list(C_BLUES)
        self.assertIsNot(anchor_to_key(in_key_blues, "C", False), in_key_blues)


class SpellingTest(unittest.TestCase):
    def test_sharp_and_flat_keys(self):
        for root in ("G", "D", "A", "E", "B", "F#", "C#"):
            self.assertTrue(uses_sharps(root, False), root)
        for root in ("C", "F", "Bb", "Eb", "Ab", "Db", "Gb"):
            self.assertFalse(uses_sharps(root, False), root)
        for root in ("E", "B", "F#", "C#", "G#"):
            self.assertTrue(uses_sharps(root, True), root)
        for root in ("A", "D", "G", "C", "F", "Bb", "Eb"):
            self.assertFalse(uses_sharps(root, True), root)


if __name__ == "__main__":
    unittest.main()
