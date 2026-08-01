"""R1's voicing reads every chord of the model's vocabulary as written: flat roots and slash basses
(music21 spells a flat "-"), maj9 and mMaj7. Before this, Bb7 was voiced as a B chord and Bbm7, Ebmaj
or C/Eb were skipped, so a flat-key Generate was ranked on the wrong chords."""

import unittest

from app.services import inference

# The tones each vocabulary quality must sound above its root
CORE = {"maj": {0, 4, 7}, "m": {0, 3, 7}, "7": {0, 4, 10}, "maj7": {0, 4, 11}, "m7": {0, 3, 10},
        "m7b5": {0, 3, 6, 10}, "dim7": {0, 3, 6, 9}, "dim": {0, 3, 6}, "aug": {0, 4, 8}, "sus4": {0, 5, 7},
        "sus2": {0, 2, 7}, "6": {0, 4, 9}, "m6": {0, 3, 9}, "9": {0, 2, 4, 10}, "m9": {0, 2, 3, 10},
        "maj9": {0, 2, 4, 11}, "11": {0, 5, 10}, "m11": {0, 3, 5, 10}, "13": {0, 4, 9, 10}, "m13": {0, 3, 10},
        "add9": {0, 2, 4, 7}, "mMaj7": {0, 3, 11}, "7b9": {0, 1, 4, 10}, "7#9": {0, 3, 4, 10},
        "7#11": {0, 4, 6, 10}, "7b13": {0, 4, 8, 10}}


@unittest.skipUnless(inference.PHYSICS_AVAILABLE, "ai/physics (music21) is not installed")
class PhysicsVoicingTest(unittest.TestCase):
    def test_every_vocabulary_chord_is_voiced_from_its_own_root(self):
        from physics.voicing import voice_chord
        from training.tokenizer import QUALITIES, ROOTS

        self.assertEqual(set(QUALITIES), set(CORE))
        for index, root in enumerate(ROOTS):
            for quality in QUALITIES:
                for bass in ("", "/Bb", "/Eb", "/F#"):
                    token = root + quality + bass
                    with self.subTest(token=token):
                        voicing = voice_chord(token)
                        self.assertIsNotNone(voicing)
                        self.assertLessEqual(CORE[quality], {(pitch - index) % 12 for pitch in voicing})
                        if bass:  # the bass is the written one, not the natural above a flat
                            self.assertIn({"/Bb": 10, "/Eb": 3, "/F#": 6}[bass], {pitch % 12 for pitch in voicing})

    def test_a_flat_is_read_as_music21_spells_it(self):
        from physics.voicing import voice_chord

        flats = ["Bb7", "Ebmaj7", "Abm", "Dbmaj9", "Gm7/Bb", "C7/Eb"]
        spelled = ["B-7", "E-maj7", "A-m", "D-M9", "Gm7/B-", "C7/E-"]
        self.assertEqual(inference.sequence_roughness(flats), inference.sequence_roughness(spelled))
        self.assertEqual(inference.sequence_roughness(flats)[1], len(flats))
        # A flat root is its enharmonic sharp, not the natural above it
        for flat, sharp, natural in (("Bb7", "A#7", "B7"), ("Ebmaj7", "D#maj7", "Emaj7"), ("Abm", "G#m", "Am")):
            with self.subTest(chord=flat):
                self.assertEqual(voice_chord(flat), voice_chord(sharp))
                self.assertNotEqual(voice_chord(flat), voice_chord(natural))
