import unittest
from unittest.mock import patch

from app.models.plan import PlanRequest, SongPlan
from app.models.schemas import SectionGenerateResponse, Suggestion
from app.services import planner
from app.services.chorus_memory import reuse_chorus


def form():
    return SongPlan(sections=[
        dict(type="chorus", bars=4, cadence="half"),
        dict(type="verse", bars=4, cadence="half"),
        dict(type="chorus", bars=4, cadence="authentic"),
        dict(type="outro", bars=4, cadence="plagal"),
    ])


class Composer:
    def __init__(self, source="ai"):
        self.requests = []
        self.source = source

    def __call__(self, request):
        self.requests.append(request)
        start, bars = len(request.context), request.chordCount // 2
        # Distinct invocations are audibly different.
        symbol = "Am7" if len(self.requests) % 2 else "Dm7"
        return SectionGenerateResponse(
            sectionType=request.sectionType, title="Fixture", bars=bars, startMeasure=start,
            chordCount=bars * 2, source=self.source, modelUsed="ft_f1" if self.source == "ai" else None,
            suggestions=[Suggestion(label="A", chords={str(m): [symbol, "G7/B"] for m in range(start, start + bars)})],
        )


class ChorusMemoryTest(unittest.TestCase):
    def run_form(self, mode="reuse", source="ai"):
        req = PlanRequest(genre="pop", targetBars=16, wantOutro=True, chorusMode=mode)
        composer = Composer(source)
        with patch.object(planner, "rules_plan", return_value=form()), patch.object(planner, "compose_section", composer):
            trace = planner.complete_form(req)
        self.assertEqual(trace.status, "ok")
        return trace, composer

    def test_reuses_body_rekeys_bars_and_attributes_only_changed_ending(self):
        trace, composer = self.run_form()
        self.assertEqual(len(composer.requests), 3)
        first, repeated = trace.sections[0], trace.sections[2]
        self.assertEqual(repeated.source, "reuse")
        self.assertIsNone(repeated.modelUsed)
        self.assertEqual(repeated.reuse.sectionIndex, 0)
        self.assertEqual(repeated.reuse.modelUsed, "ft_f1")
        self.assertEqual(repeated.reuse.changedSlots, [6, 7])
        self.assertEqual(repeated.reuse.retainedSlots, list(range(6)))
        self.assertEqual(list(repeated.suggestions[0].chords), ["8", "9", "10", "11"])
        for bar in range(3):
            self.assertEqual(first.suggestions[0].chords[str(bar)], repeated.suggestions[0].chords[str(8 + bar)])
        self.assertEqual(repeated.suggestions[0].chords["11"], ["G7", "Cmaj"])
        self.assertIsNone(repeated.suggestions[0].explanations)

    def test_new_mode_and_old_api_default_still_generate_every_section(self):
        trace, composer = self.run_form("new")
        self.assertEqual(len(composer.requests), 4)
        self.assertTrue(all(section.reuse is None for section in trace.sections))
        self.assertEqual(PlanRequest(genre="pop", targetBars=16).chorusMode, "new")

    def test_rule_source_is_never_reported_as_new_ai(self):
        trace, _ = self.run_form(source="rules")
        self.assertEqual(trace.sections[2].reuse.source, "rules")
        self.assertIsNone(trace.sections[2].reuse.modelUsed)

    def test_sketch_is_preferred_preserved_and_keeps_slash_bass(self):
        req = PlanRequest(genre="pop", targetBars=16, chorusMode="reuse", measures=[
            dict(index=i, chords=["C/E", "Am7"], roman=["I", "vi7"]) for i in range(4)
        ], sections=[dict(type="chorus", start=0, length=4)])
        before = req.model_dump()
        plan = SongPlan(sections=[dict(type="chorus", bars=4, cadence="none")])
        result = reuse_chorus(req, plan, [], 0)
        self.assertEqual(req.model_dump(), before)
        self.assertEqual(result.source, "reuse")
        self.assertEqual(result.reuse.origin, "sketch")
        self.assertEqual(result.reuse.changedSlots, [])
        self.assertEqual(result.suggestions[0].chords["4"], ["C/E", "Am7"])

    def test_missing_unsupported_or_different_length_theme_is_not_copied(self):
        for symbol, length in ((None, 4), ("unresolved", 4), ("C", 3)):
            req = PlanRequest(genre="pop", targetBars=16, chorusMode="reuse", measures=[
                dict(index=i, chords=[symbol, "G7"], roman=[None, "V7"]) for i in range(length)
            ], sections=[dict(type="chorus", start=0, length=length)])
            self.assertIsNone(reuse_chorus(req, form(), [], 0))

    def test_minor_cadence_does_not_rewrite_the_body(self):
        req = PlanRequest(genre="pop", key="A minor", targetBars=16, chorusMode="reuse", measures=[
            dict(index=i, chords=["Am", "Dm/F"], roman=["i", "iv"]) for i in range(4)
        ], sections=[dict(type="chorus", start=0, length=4)])
        result = reuse_chorus(req, SongPlan(sections=[dict(type="chorus", bars=4, cadence="authentic")]), [], 0)
        self.assertEqual(result.suggestions[0].chords["4"], ["Am", "Dm/F"])
        self.assertEqual(result.suggestions[0].chords["7"], ["E7", "Amin"])

    def test_a_reprise_is_emitted_in_plan_order_without_invoking_the_composer(self):
        req = PlanRequest(genre="pop", targetBars=16, chorusMode="reuse")
        events = []
        run = planner._Run(req, lambda name, data: events.append((name, data)))
        run.plan = form()
        composer = Composer()
        with patch.object(planner, "compose_section", composer):
            planner._generate_all(run)
        self.assertEqual(len(composer.requests), 3)  # the reprise never invokes the composer
        self.assertEqual(run.sections[2].suggestions[0].chords["8"], run.sections[0].suggestions[0].chords["0"])
        sent = [data for name, data in events if name == "generate_section"]
        self.assertEqual([data["index"] for data in sent], [0, 1, 2, 3])
        self.assertEqual([data["source"] for data in sent], ["ai", "ai", "reuse", "ai"])


if __name__ == "__main__":
    unittest.main()
