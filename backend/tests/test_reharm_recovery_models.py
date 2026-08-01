"""Opt-in real-checkpoint coverage for own-model reharmonization."""
import math
import os
import unittest

from app.models.reharmonize import ReharmRequest
from app.services import inference, reharmonize_choice


@unittest.skipUnless(os.getenv("THEARTIST_TEST_REHARM_MODELS") == "1", "requires local model weights")
class RealModelRecoveryTests(unittest.TestCase):
    def test_different_passages_and_models_return_three_scored_alternatives(self):
        passages = [
            ("C major", 0, [["Dm7", "G7"]], 1),
            ("E major", 24, [["E6", "C#m"], ["C#m", "C#m"], [None, None], ["C#m", "C#m"]], 4),
            ("F minor", 12, [["Fm7", "Bbm7"], ["Eb7", "Abmaj7"], [None, "Dbmaj7"],
                             ["Gm7b5", "C7b9"], ["Fm7", "Fm7"], ["Bbm7", "Eb7"],
                             ["Abmaj7", None], ["C7", "Fm7"]], 4),
        ]
        for model in ("ft_f1", "ft_f1_lora_classical"):
            for key, start, bars, cap in passages:
                with self.subTest(model=model, key=key, bars=len(bars)):
                    req = ReharmRequest(key=key, modelKey=model, maxChanges=cap,
                        context=[{"measure": start + i, "chords": chords} for i, chords in enumerate(bars)],
                        selectedSlots=[{"measureIndex": start + i, "slotIndex": j}
                                       for i, chords in enumerate(bars) for j, symbol in enumerate(chords) if symbol])
                    original = req.model_dump_json()
                    result = reharmonize_choice.reharmonize(req)
                    self.assertEqual(result.modelUsed, inference.MODEL_LABELS[model])
                    self.assertEqual(len(result.candidates), 3)
                    self.assertEqual(len({c.id for c in result.candidates}), 3)
                    allowed = {(s.measureIndex, s.slotIndex) for s in req.selectedSlots}
                    for candidate in result.candidates:
                        self.assertTrue(math.isfinite(candidate.modelScore))
                        self.assertTrue(1 <= len(candidate.changes) <= cap)
                        self.assertTrue(all((c.address.measureIndex, c.address.slotIndex) in allowed
                                            for c in candidate.changes))
                    self.assertEqual(req.model_dump_json(), original)


if __name__ == "__main__":
    unittest.main()
