"""Measure real local model runs with OMT ready; no working dates.

Alternates new/reuse ordering for each seed. Output includes raw times, source
counts, retained-theme checks and the first paired traces for later listening.
This measures the local pipeline, not HTTP, browser, or musical preference.
"""
import argparse
import json
import math
from pathlib import Path
import platform
import time

import torch

from app.models.plan import PlanRequest
from app.services import inference, planner
from app.services.contracts import capabilities


def percentiles(values):
    ordered = sorted(values)
    return {name: ordered[math.ceil(p * len(ordered)) - 1] for name, p in (("p50Ms", .5), ("p95Ms", .95))}


def run_case(fields, mode, seed):
    request = PlanRequest(**fields, modelKey="ft_f1", chorusMode=mode, seed=seed)
    start = time.perf_counter()
    trace = planner.complete_form(request)
    elapsed = (time.perf_counter() - start) * 1000
    if trace.status != "ok" or any(s.source not in {"ai", "reuse"} for s in trace.sections):
        raise RuntimeError("Expected a complete run using the actual checkpoint, with no rule fallback")
    reused = [s for s in trace.sections if s.reuse]
    for section in reused:
        if section.reuse.origin != "section":
            raise AssertionError("These empty-sketch cases must reuse generated material")
        source = trace.sections[section.reuse.sectionIndex]
        before = planner._flat_chords(source)
        after = planner._flat_chords(section)
        if any(before[i] != after[i] for i in section.reuse.retainedSlots):
            raise AssertionError("The retained chorus changed")
    return request, trace, dict(durationMs=round(elapsed, 3),
        generatedSections=sum(s.source == "ai" for s in trace.sections),
        reusedSections=len(reused), reusedBars=sum(s.bars for s in reused),
        retainedSlots=sum(len(s.reuse.retainedSlots) for s in reused),
        endingChanges=sum(len(s.reuse.changedSlots) for s in reused))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()
    if not args.out.is_absolute() or args.out.exists() or not 3 <= args.runs <= 20:
        parser.error("Use a new absolute output path and 3–20 runs")
    rag = capabilities().data["rag"]
    if rag["sources"] != {"omt": "ready", "berklee": "disabled"}:
        raise RuntimeError("Prepare OMT and disable Berklee before this default-runtime benchmark")
    model = inference.ModelRegistry.get().load_model("ft_f1")
    if model is None or model.device.type != "cpu":
        raise RuntimeError("Install F1 and use CPU before benchmarking")
    report = dict(runtime=dict(torch=torch.__version__, threads=torch.get_num_threads(),
                              platform=platform.system(), architecture=platform.machine(), rag=rag),
                  runs=args.runs, seeds=list(range(17, 17 + args.runs)),
                  method="Alternating paired timing; model load excluded; fresh sampling per mode; different later context is intentional",
                  exclusions=["HTTP", "browser", "audio", "human preference"], cases=[])
    for name, fields in (
        ("pop-major", dict(genre="pop", key="C major", targetBars=64, wantOutro=True)),
        ("pop-minor", dict(genre="pop", key="A minor", targetBars=64, wantOutro=True)),
        ("rock-major", dict(genre="rock", key="E major", targetBars=64, wantOutro=True)),
    ):
        for mode in ("new", "reuse"):
            run_case(fields, mode, 16)
        row = dict(name=name, samples={"new": [], "reuse": []}, listening={})
        for i in range(args.runs):
            for mode in (("new", "reuse") if i % 2 == 0 else ("reuse", "new")):
                request, trace, measurement = run_case(fields, mode, 17 + i)
                row["samples"][mode].append({"seed": 17 + i, **measurement})
                if i == 0:
                    # Retrieval text is not needed for timing/listening artifacts or redistributed here.
                    saved_trace = trace.model_dump(mode="json")
                    for section in saved_trace["sections"]:
                        for suggestion in section["suggestions"]:
                            suggestion["explanations"] = None
                    row["listening"][mode] = {"request": name, "requestBody": request.model_dump(mode="json"), "trace": saved_trace}
        row["summary"] = {mode: percentiles([s["durationMs"] for s in row["samples"][mode]]) for mode in ("new", "reuse")}
        report["cases"].append(row)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"case": name, "summary": row["summary"], "reuse": row["samples"]["reuse"][0]}), flush=True)


if __name__ == "__main__":
    main()
