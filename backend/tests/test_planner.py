"""The song planner: validate_plan rule by rule, the rules planner for every genre, the rule slots,
the key facts Second ear shares, and the plan -> generate run with a fake section composer and its
forced failures. No network, no model."""

import json
import logging
import threading
import unittest
from unittest import mock

from fastapi import HTTPException

from app.models.plan import PlanRequest, PlanSectionResult, SongPlan, TARGET_BARS
from app.models.schemas import CADENCE_TYPES, SectionGenerateRequest, SectionGenerateResponse, Suggestion
from app.models.sections import SECTION_TYPES, section_title
from app.services import planner
from app.services.genre_tables import GENRE_DEFAULTS
from app.services.song_form import apply_song_form, apply_song_form_with_slots, resolves_to_tonic, rule_slots

_QUIET = logging.NullHandler()


def setUpModule():
    # Failure paths log on purpose; keep them out of the runner output
    logging.getLogger("app.services.planner").addHandler(_QUIET)


def tearDownModule():
    logging.getLogger("app.services.planner").removeHandler(_QUIET)


def measure(index, chords, roman):
    return {"index": index, "chords": chords, "roman": roman}


# An 8-bar pop sketch in C major with the Roman numerals the editor would compute
SKETCH = [
    measure(0, ["Cmaj7", "Am7"], ["Imaj7", "vi7"]), measure(1, ["Dm7", "G7"], ["ii7", "V7"]),
    measure(2, ["Cmaj7", "Am7"], ["Imaj7", "vi7"]), measure(3, ["Dm7", "G7"], ["ii7", "V7"]),
    measure(4, ["Fmaj7", "G7"], ["IVmaj7", "V7"]), measure(5, ["Em7", "Am7"], ["iii7", "vi7"]),
    measure(6, ["Fmaj7", "G7"], ["IVmaj7", "V7"]), measure(7, ["Cmaj", None], ["I", None]),
]
VALID_PLAN = {"sections": [
    {"type": "chorus", "bars": 8, "cadence": "authentic", "reason": "Lift after the ii-V in bars 1 and 3",
     "cites": [{"bar": 1, "roman": "V7"}, {"bar": 3, "roman": "ii7"}]},
    {"type": "verse", "bars": 8, "cadence": "half", "reason": "Second verse, left open", "cites": []},
    {"type": "outro", "bars": 8, "cadence": "plagal", "reason": "IV-I close like bar 6", "cites": [{"bar": 6, "roman": "IVmaj7"}]},
]}
SHORT_PLAN = {"sections": [  # 20 bars instead of 24, and the last cadence is open
    {"type": "chorus", "bars": 8, "cadence": "authentic", "reason": "Lift", "cites": []},
    {"type": "outro", "bars": 12, "cadence": "half", "reason": "Fade", "cites": []},
]}
# 16 distinct chords, none of them a chord the rules write in C major or A minor and none on
# a forced root (I or IV) at the slots the rules compare, so every rule-decided slot changes
PHRASE = "Em7 Gm7 Bbmaj7 Ebmaj7 Ab7 Dbmaj7 Gb7 Bm7 F#m7 Bmaj7 Dm9 Abm7 Gm9 Eb9 Fm7 Bb7".split()


def request(**overrides) -> PlanRequest:
    fields = {"genre": "pop", "key": "C major", "measures": SKETCH, "sections": [{"type": "verse", "start": 0, "length": 8}],
              "targetBars": 32, "wantOutro": True}
    fields.update(overrides)
    return PlanRequest(**fields)


class FakeComposer:
    """Stands in for compose_section: one diatonic suggestion per call, every request kept."""

    def __init__(self):
        self.requests: list[SectionGenerateRequest] = []

    def __call__(self, req: SectionGenerateRequest) -> SectionGenerateResponse:
        self.requests.append(req)
        start, bars = len(req.context), req.chordCount // 2
        chords = {str(m): ["Cmaj7", "Am7"] if m % 2 == 0 else ["Dm7", "G7"] for m in range(start, start + bars)}
        return SectionGenerateResponse(sectionType=req.sectionType, title=section_title(req.sectionType, bars),
                                       startMeasure=start, bars=bars, chordCount=bars * 2,
                                       suggestions=[Suggestion(label="Option A — fake", chords=chords)],
                                       source="ai", modelUsed="ft_f1")


class ValidatePlanTest(unittest.TestCase):
    def plan(self, **changes) -> SongPlan:
        data = json.loads(json.dumps(VALID_PLAN))
        for key, value in changes.items():
            index, field = key.split("_", 1)
            data["sections"][int(index)][field] = value
        return SongPlan.model_validate(data)

    def codes(self, plan, req=None):
        return [(issue.code, issue.section) for issue in planner.validate_plan(plan, req or request())]

    def test_valid_plan_has_no_issues(self):
        self.assertEqual(planner.validate_plan(self.plan(), request()), [])

    def test_each_rule_reports_its_code(self):
        cases = {
            "section_type": (self.plan(**{"1_type": "coda"}), ("section_type", 1)),
            "bars_low": (self.plan(**{"0_bars": 1, "1_bars": 15}), ("bars", 0)),
            "bars_high": (self.plan(**{"0_bars": 33, "1_bars": 2, "2_bars": 2}), ("bars", 0)),  # 37 + 8 != 32 too
            "cadence": (self.plan(**{"1_cadence": "deceptive"}), ("cadence", 1)),
            "reason_length": (self.plan(**{"0_reason": "x" * 121}), ("reason_length", 0)),
            "intro_first": (self.plan(**{"0_type": "intro"}), ("intro_first", 0)),
            "outro_last": (self.plan(**{"0_type": "outro"}), ("outro_last", 0)),
            "final_cadence": (self.plan(**{"2_cadence": "half"}), ("final_cadence", 2)),
            "outro_wanted": (self.plan(**{"2_type": "chorus"}), ("outro_wanted", 2)),
            "cite_bar": (self.plan(**{"1_cites": [{"bar": 8, "roman": "I"}]}), ("cite_bar", 1)),
            "cite_roman": (self.plan(**{"1_cites": [{"bar": 1, "roman": "IV"}]}), ("cite_roman", 1)),
            "total_bars": (SongPlan.model_validate(SHORT_PLAN), ("total_bars", None)),
        }
        for name, (plan, expected) in cases.items():
            with self.subTest(rule=name):
                self.assertIn(expected, self.codes(plan))
        self.assertEqual(self.codes(SongPlan(sections=[])), [("empty", None)])

    def test_rules_follow_the_request(self):
        plan = self.plan(**{"2_type": "chorus", "2_cadence": "authentic"})
        self.assertEqual(self.codes(plan, request(wantOutro=False)), [])
        self.assertEqual(self.codes(self.plan(), request(wantOutro=False)), [("outro_unwanted", 2)])
        # An empty song may open with an intro, but then nothing can be cited
        new_song = request(measures=[], sections=[], targetBars=16)
        opening = SongPlan.model_validate({"sections": [
            {"type": "intro", "bars": 4, "cadence": "none", "reason": "Open", "cites": [{"bar": 0, "roman": "I"}]},
            {"type": "chorus", "bars": 8, "cadence": "authentic", "reason": "Hook", "cites": []},
            {"type": "outro", "bars": 4, "cadence": "plagal", "reason": "Close", "cites": []}]})
        self.assertEqual(self.codes(opening, new_song), [("cite_bar", 0)])
        self.assertEqual(self.codes(self.plan(), request(targetBars=48)), [("total_bars", None)])

    def test_request_shape_is_checked_before_the_validator(self):
        # planner and chordSource were request fields of earlier builds; unknown fields are rejected
        for bad in ({"targetBars": 20}, {"contrast": "wild"}, {"planner": "gpt"}, {"planner": "rules"},
                    {"chordSource": "meta"}, {"genre": "polka"},
                    {"key": "H major"}, {"measures": SKETCH[:3]}, {"measures": SKETCH[::-1]},
                    {"measures": SKETCH * 4}, {"sections": [{"type": "verse", "start": 4, "length": 8}]},
                    {"sections": [{"type": "verse", "start": 0, "length": 4}, {"type": None, "start": 2, "length": 4}]},
                    {"measures": [measure(0, ["C", "F", "G"], ["I", "IV", "V"])]}, {"prompt": "make it sad"}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                request(**bad)
        new_song = PlanRequest(genre="jazz", targetBars=32)
        self.assertEqual((new_song.key, new_song.bpm, new_song.timeSignature), ("F major", 110, (4, 4)))
        self.assertEqual(PlanRequest(genre="pop", key="a Minor", targetBars=16).key, "A minor")


class RulesPlannerTest(unittest.TestCase):
    def test_every_genre_validates_at_every_length(self):
        for genre in [*GENRE_DEFAULTS]:
            for target in TARGET_BARS:
                for outro in (False, True):
                    for sketch_bars in (0, 8, 13, target - 2):
                        sketch = [measure(i, ["Cmaj7", "Am7"], ["Imaj7", "vi7"]) for i in range(sketch_bars)]
                        req = PlanRequest(genre=genre, measures=sketch, targetBars=target, wantOutro=outro)
                        plan = planner.rules_plan(req)
                        with self.subTest(genre=genre, target=target, outro=outro, sketch=sketch_bars):
                            self.assertEqual([i.message for i in planner.validate_plan(plan, req)], [])
                            self.assertEqual(sketch_bars + sum(s.bars for s in plan.sections), target)

    def test_forms_follow_the_genre_and_keep_the_sketch_first(self):
        new_pop = planner.rules_plan(PlanRequest(genre="pop", targetBars=32))
        self.assertEqual([(s.type, s.bars) for s in new_pop.sections],
                         [("intro", 4), ("verse", 8), ("chorus", 8), ("verse", 8), ("chorus", 4)])
        self.assertEqual([s.cadence for s in new_pop.sections], ["none", "authentic", "authentic", "authentic", "authentic"])
        continued = planner.rules_plan(request())
        self.assertEqual([(s.type, s.bars) for s in continued.sections],
                         [("verse", 8), ("chorus", 8), ("verse", 4), ("outro", 4)])
        self.assertEqual([s.cadence for s in continued.sections], ["authentic", "authentic", "authentic", "plagal"])
        jazz = planner.rules_plan(PlanRequest(genre="jazz", targetBars=64))
        self.assertEqual([s.type for s in jazz.sections], ["a_section", "a_section", "b_section", "a_section"] * 2)
        blues = planner.rules_plan(PlanRequest(genre="blues", targetBars=48, wantOutro=True))
        self.assertEqual([(s.type, s.bars) for s in blues.sections],
                         [("intro", 4), ("verse", 12), ("verse", 12), ("verse", 12), ("verse", 4), ("outro", 4)])
        self.assertTrue(all(s.cites == [] and "rules" in s.reason for s in blues.sections))


class RuleSlotsTest(unittest.TestCase):
    def test_slots_match_what_apply_song_form_rewrote(self):
        for section in SECTION_TYPES:
            for cadence in (None, *CADENCE_TYPES):
                for key, minor in (("C", False), ("A", True)):
                    for n in (1, 2, 3, 4, 5, 16):
                        with self.subTest(section=section, cadence=cadence, key=key, n=n):
                            chords, slots = apply_song_form_with_slots(PHRASE[:n], section, key, minor, cadence)
                            changed = [i for i in range(n) if chords[i] != PHRASE[i]]
                            self.assertEqual(changed, slots)
                            self.assertEqual(slots, rule_slots(section, n, cadence))
                            self.assertEqual(chords, apply_song_form(PHRASE[:n], section, key, minor, cadence))

    def test_a_chosen_cadence_replaces_the_section_ending(self):
        self.assertEqual(apply_song_form(PHRASE[:4], "verse", "C")[-2:], ["G7", "Cmaj"])
        self.assertEqual(apply_song_form(PHRASE[:4], "verse", "C", cadence="half")[-1:], ["G7"])
        self.assertEqual(apply_song_form(PHRASE[:4], "verse", "C", cadence="plagal")[-2:], ["Fmaj7", "Cmaj"])
        self.assertEqual(apply_song_form(PHRASE[:4], "verse", "C", cadence="none"), PHRASE[:4])
        self.assertEqual(apply_song_form(PHRASE[:4], "outro", "C", cadence="turnaround"), ["Cmaj7", "Am7", "Dm7", "G7"])
        self.assertEqual(apply_song_form(PHRASE[:4], "intro", "C", cadence="authentic"), ["Cmaj7", "Gm7", "G7", "Cmaj"])
        self.assertEqual(rule_slots("post_chorus", 8, "half"), [4, 5, 6, 7])
        self.assertEqual(rule_slots("tag", 8), [6, 7])
        self.assertEqual(rule_slots("tag", 8, "none"), [])
        self.assertEqual(rule_slots("solo", 8, "authentic"), [6, 7])

    def test_resolves_to_tonic_follows_the_effective_cadence(self):
        # The voice-leading pass pins a root-position tonic only where the ending really lands on it
        for section, cadence, expected in (("verse", None, True), ("verse", "half", False), ("verse", "none", False),
                                           ("solo", "plagal", True), ("bridge", None, False), (None, "authentic", False)):
            with self.subTest(section=section, cadence=cadence):
                self.assertEqual(resolves_to_tonic(section, cadence), expected)


class KeyFactsTest(unittest.TestCase):
    """The chord facts the planner module shares with Second ear (app.services.review)."""

    def test_applied_chords_are_read_in_slot_order_and_judged_without_their_slash_bass(self):
        chords = {"8": ["Cmaj7", "Ebmaj7"], "9": ["F#7", "G7/B"], "10": ["Cmaj7", "Ebmaj7"], "11": ["G7", "Cmaj"]}
        section = PlanSectionResult(sectionType="chorus", title="Chorus — 4 bars", startMeasure=8, bars=4, chordCount=8,
                                    suggestions=[Suggestion(label="A", chords=chords)], source="ai", modelUsed="ft_f1",
                                    cadence="authentic", rulesSlots=[6, 7], reason="", cites=[])
        flat = planner._flat_chords(section)
        self.assertEqual(flat, ["Cmaj7", "Ebmaj7", "F#7", "G7/B", "Cmaj7", "Ebmaj7", "G7", "Cmaj"])
        self.assertEqual([planner._symbol(chord) for chord in flat[2:4]], ["F#7", "G7"])
        self.assertIsNone(planner._symbol(None))
        collection = planner._key_collection("C major")
        outside = [chord for chord in flat[:6] if not planner._in_key(planner._symbol(chord), collection)]
        self.assertEqual(outside, ["Ebmaj7", "F#7", "Ebmaj7"])

    def test_a_minor_key_owns_its_dominant_and_raised_sixth(self):
        chords = ["Am7", "E7", "Dm7", "E7", "Am7", "Fmaj7", "Bm7b5", "E7"]
        minor, major = planner._key_collection("A minor"), planner._key_collection("C major")
        self.assertEqual([chord for chord in chords if not planner._in_key(chord, minor)], [])
        self.assertEqual([chord for chord in chords if not planner._in_key(chord, major)], ["E7", "E7", "E7"])


class PlannerRunTest(unittest.TestCase):
    def setUp(self):
        self.composer = FakeComposer()
        patcher = mock.patch.object(planner, "compose_section", self.composer)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_cancel_during_generation_stops_the_other_sections(self):
        cancelled = threading.Event()

        def compose_and_cancel(req):
            result = self.composer(req)
            cancelled.set()
            return result

        with mock.patch.object(planner, "compose_section", side_effect=compose_and_cancel):
            trace = planner.complete_form(request(), cancelled=cancelled.is_set)
        self.assertEqual((trace.status, trace.error.kind), ("error", "generation"))
        self.assertIn("cancelled", trace.error.message)
        self.assertEqual(len(self.composer.requests), 1)
        self.assertEqual(trace.sections, [])  # the section composed while the cancel arrived is not kept
        self.assertIsNotNone(trace.plan)

    def test_a_validated_form_is_generated_section_by_section_with_its_cadences(self):
        events = []
        with mock.patch.object(planner, "rules_plan", return_value=SongPlan.model_validate(VALID_PLAN)):
            trace = planner.complete_form(request(), lambda name, payload: events.append((name, payload)))
        self.assertEqual((trace.status, trace.error), ("ok", None))
        self.assertEqual([(v.attempt, v.ok, [i.code for i in v.errors]) for v in trace.validation], [(1, True, [])])
        self.assertEqual(trace.plan.model_dump(), VALID_PLAN)
        self.assertEqual([(s.sectionType, s.bars, s.startMeasure, s.cadence) for s in trace.sections],
                         [("chorus", 8, 8, "authentic"), ("verse", 8, 16, "half"), ("outro", 8, 24, "plagal")])
        self.assertEqual([s.rulesSlots for s in trace.sections], [[14, 15], [15], [14, 15]])
        self.assertEqual([(s.reason, s.cites) for s in trace.sections],
                         [(p.reason, p.cites) for p in trace.plan.sections])
        self.assertEqual([(s.source, s.modelUsed, s.regenerated, s.reuse) for s in trace.sections],
                         [("ai", "ft_f1", False, None)] * 3)
        self.assertEqual([len(r.context) for r in self.composer.requests], [8, 16, 24])
        self.assertEqual([(r.cadence, r.chordCount, r.genre, r.key) for r in self.composer.requests],
                         [("authentic", 16, "pop", "C major"), ("half", 16, "pop", "C major"), ("plagal", 16, "pop", "C major")])
        self.assertEqual(self.composer.requests[1].context[8].chords, ["Cmaj7", "Am7"])  # section A applied
        self.assertEqual((trace.genre, trace.key, trace.bpm), ("pop", "C major", GENRE_DEFAULTS["pop"]["bpm"]))
        # The stream's events, in run order, each stamped with its time
        self.assertEqual([name for name, _ in events],
                         ["progress", "plan", "validate", "progress", "generate_section", "progress", "generate_section",
                          "progress", "generate_section"])
        self.assertTrue(all(isinstance(payload["t_ms"], int) for _, payload in events))
        untimed = [(name, {key: value for key, value in payload.items() if key != "t_ms"}) for name, payload in events]
        self.assertEqual(untimed[1], ("plan", {"attempt": 1, "source": "rules", "plan": VALID_PLAN}))
        self.assertEqual(untimed[2], ("validate", {"attempt": 1, "ok": True, "errors": []}))
        self.assertEqual([payload for name, payload in untimed if name == "generate_section"],
                         [{"index": index, **section.model_dump()} for index, section in enumerate(trace.sections)])
        self.assertEqual([payload["label"] for name, payload in untimed if name == "progress"],
                         ["Planning the song form from the sketch…", "Composing chorus (1/3)…",
                          "Composing verse (2/3)…", "Composing outro (3/3)…"])

    def test_the_genre_default_form_runs_from_plan_to_last_section(self):
        trace = planner.complete_form(request())
        self.assertEqual((trace.status, trace.error), ("ok", None))
        self.assertEqual([(s.sectionType, s.bars) for s in trace.sections], [("verse", 8), ("chorus", 8), ("verse", 4), ("outro", 4)])
        self.assertEqual([s.cadence for s in trace.sections], ["authentic", "authentic", "authentic", "plagal"])
        self.assertEqual([s.rulesSlots for s in trace.sections], [[14, 15], [14, 15], [6, 7], [6, 7]])
        self.assertEqual([v.ok for v in trace.validation], [True])
        self.assertEqual([len(r.context) for r in self.composer.requests], [8, 16, 24, 28])

    def test_sampling_seed_is_split_by_section(self):
        trace = planner.complete_form(request(seed=1379))
        self.assertEqual(trace.status, "ok")
        self.assertEqual([req.seed for req in self.composer.requests], [1379, 1379 + 1009, 1379 + 2018, 1379 + 3027])
        self.composer.requests.clear()
        planner.complete_form(request(seed=0xFFFFFFFF))  # every section's seed stays an unsigned 32-bit integer
        self.assertEqual([req.seed for req in self.composer.requests], [0xFFFFFFFF, 1008, 2017, 3026])
        self.composer.requests.clear()
        planner.complete_form(request())
        self.assertEqual([req.seed for req in self.composer.requests], [None] * 4)

    def test_a_form_that_fails_validation_is_an_error_before_any_chord(self):
        events = []
        with mock.patch.object(planner, "rules_plan", return_value=SongPlan.model_validate(SHORT_PLAN)):
            trace = planner.complete_form(request(), lambda name, payload: events.append(name))
        self.assertEqual((trace.status, trace.error.kind), ("error", "invalid"))
        self.assertIn("failed validation", trace.error.message)
        self.assertEqual([(v.attempt, v.ok, [i.code for i in v.errors]) for v in trace.validation],
                         [(1, False, ["total_bars", "final_cadence"])])
        self.assertEqual(trace.plan.model_dump(), SHORT_PLAN)
        self.assertEqual((trace.sections, self.composer.requests), ([], []))
        self.assertEqual(events, ["progress", "plan", "validate"])

    def test_a_section_the_chord_model_fails_ends_the_run_with_the_sections_before_it(self):
        for failure, message in ((HTTPException(status_code=400, detail="Expected key format: 'C major' or 'A minor'"),
                                  "Expected key format: 'C major' or 'A minor'"),
                                 (RuntimeError("boom"), "Chord generation failed")):
            with self.subTest(failure=type(failure).__name__):
                composer = FakeComposer()

                def compose_then_fail(req, composer=composer, failure=failure):
                    if len(composer.requests) == 1:
                        raise failure
                    return composer(req)

                with mock.patch.object(planner, "compose_section", side_effect=compose_then_fail):
                    trace = planner.complete_form(request())
                self.assertEqual((trace.status, trace.error.kind, trace.error.message), ("error", "generation", message))
                self.assertEqual([s.sectionType for s in trace.sections], ["verse"])

    def test_the_chord_model_time_budget_stops_generation(self):
        with mock.patch.object(planner, "OWN_MODEL_BUDGET_S", 0):
            trace = planner.complete_form(request())
        self.assertEqual((trace.status, trace.error.kind), ("error", "timeout"))
        self.assertIn("chord-model time", trace.error.message)
        self.assertEqual((trace.sections, self.composer.requests), ([], []))
        self.assertIsNotNone(trace.plan)

    def test_the_run_time_budget_stops_the_run_before_its_plan(self):
        with mock.patch.object(planner, "RUN_BUDGET_S", 0):
            trace = planner.complete_form(request())
        self.assertEqual((trace.status, trace.error.kind), ("error", "timeout"))
        self.assertEqual((trace.plan, trace.sections, self.composer.requests), (None, [], []))

    def test_plan_song_plans_without_generating(self):
        events = []
        trace = planner.plan_song(request(), lambda name, payload: events.append(name))
        self.assertEqual((trace.status, trace.error, trace.sections), ("ok", None, []))
        self.assertEqual([(s.type, s.bars) for s in trace.plan.sections], [("verse", 8), ("chorus", 8), ("verse", 4), ("outro", 4)])
        self.assertEqual([v.ok for v in trace.validation], [True])
        self.assertEqual(events, ["plan", "validate"])
        self.assertEqual(self.composer.requests, [])


if __name__ == "__main__":
    unittest.main()
