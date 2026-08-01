"""Chord-generation speed probe: wall time and per-phase timing of generate_ai_suggestions.

Measurement only. Each genre x bars cell resolves its checkpoint the way the API
does (genre dispatch, no modelKey) and loads it once, outside the timed runs.
Warm-up calls run unrecorded, then every recorded call is timed. Phase timing
comes from the service's on_phase events: an event marks the start of a stage,
so the gap to the next event (or to the return) is that stage's time, and the
gap before the first event is "setup". phase_ms keeps the composing steps apart
(composing_1..3); stage_ms sums them as "composing".

Usage (from backend/):
    uv run python -m scripts.speed_probe --out /tmp/speed_probe.json
    uv run python -m scripts.speed_probe --bars 8 --genres jazz --runs 5 --context-bars 8

--out is the JSON path; the Markdown table is written next to it (same name,
.md) and always printed. Torch is reseeded with --seed at the start of every
cell, so a cell's samples do not depend on which other cells ran.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import platform
import statistics
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from app.models.schemas import ContextMeasure, GenerateRequest
from app.models.sections import SECTION_TYPES
from app.services import inference

ENV_FLAGS = (
    "THEARTIST_DISABLE_RAG",
    "THEARTIST_DISABLE_R2_RERANK",
    "THEARTIST_DISABLE_EXPLANATIONS",
    "THEARTIST_DISABLE_VOICE_LEADING",
)

# Roots the API accepts (routes/generate.py VALID_ROOTS) as pitch classes, and
# the tokenizer's spelling for the context chords built on them.
PITCH_CLASS = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5, "F#": 6,
    "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}
SPELLING = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

# --context-bars loop, two chords per bar: major I vi | IV V, minor i VI | III VII.
CONTEXT_LOOP = {
    "major": [(0, ""), (9, "m"), (5, ""), (7, "")],
    "minor": [(0, "m"), (8, ""), (3, ""), (10, "")],
}


def _log(message: str) -> None:
    print(f"[speed_probe] {message}", file=sys.stderr, flush=True)


def _ms(seconds: float) -> float:
    return round(seconds * 1000, 3)


def _percentile(values: list[float], q: float) -> float:
    """Linear interpolation between closest ranks (numpy's default method)."""
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def _csv(text: str) -> list[str]:
    items = list(dict.fromkeys(part.strip() for part in text.split(",") if part.strip()))
    if not items:
        raise argparse.ArgumentTypeError("expected a comma-separated list")
    return items


def _bar_counts(text: str) -> list[int]:
    try:
        counts = list(dict.fromkeys(int(item) for item in _csv(text)))
    except ValueError:
        raise argparse.ArgumentTypeError(f"bar counts must be integers, got {text!r}") from None
    if min(counts) < 1:
        raise argparse.ArgumentTypeError("bar counts start at 1")
    return counts


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.speed_probe",
        description="Time generate_ai_suggestions per genre and bar count (measurement only).",
    )
    parser.add_argument("--bars", type=_bar_counts, default="8,32,64",
                        help="bar counts to request, comma-separated (default 8,32,64)")
    parser.add_argument("--genres", type=_csv, default="pop,jazz,rock",
                        help="genres, comma-separated (default pop,jazz,rock)")
    parser.add_argument("--runs", type=int, default=3, help="recorded runs per cell (default 3)")
    parser.add_argument("--warmup", type=int, default=1,
                        help="unrecorded warm-up runs per cell (default 1)")
    parser.add_argument("--context-bars", type=int, default=0,
                        help="bars of a diatonic progression in the key placed before the "
                             "requested bars (default 0)")
    parser.add_argument("--key", default="C major",
                        help="key in the API's form, e.g. 'A minor' (default 'C major')")
    parser.add_argument("--out", type=Path,
                        help="JSON output path; the Markdown table is written next to it as .md")
    parser.add_argument("--threads", type=int,
                        help="torch.set_num_threads value (default: torch's own)")
    parser.add_argument("--seed", type=int, default=0,
                        help="torch seed set at the start of every cell (default 0)")
    parser.add_argument("--section-type", choices=SECTION_TYPES,
                        help="passed as section_type when generate_ai_suggestions takes it")
    args = parser.parse_args(argv)

    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if args.warmup < 0 or args.context_bars < 0:
        parser.error("--warmup and --context-bars cannot be negative")
    if args.threads is not None and args.threads < 1:
        parser.error("--threads must be at least 1")
    parts = args.key.split()
    if len(parts) != 2 or parts[0] not in PITCH_CLASS or parts[1].lower() not in ("major", "minor"):
        parser.error("--key must look like 'C major' or 'A minor'")
    if args.out is not None and args.out.suffix.lower() == ".md":
        parser.error("--out is the JSON path; the .md is written next to it")
    return args


def _request(args: argparse.Namespace, genre: str, bars: int) -> GenerateRequest:
    """The score editor's request: --context-bars filled bars, then the empty selected bars."""
    root, mode = args.key.split()
    loop = CONTEXT_LOOP[mode.lower()]
    context = []
    for measure in range(args.context_bars):
        first = 2 * (measure % 2)  # even bars take the loop's first two chords, odd bars the last two
        context.append(ContextMeasure(
            measure=measure,
            chords=[SPELLING[(PITCH_CLASS[root] + step) % 12] + quality
                    for step, quality in loop[first:first + 2]],
        ))
    selected = list(range(args.context_bars, args.context_bars + bars))
    context += [ContextMeasure(measure=m, chords=[None, None]) for m in selected]
    return GenerateRequest(key=args.key, genre=genre, context=context,
                           selectedMeasures=selected, sectionType=args.section_type)


def _service_kwargs(req: GenerateRequest, takes_section: bool) -> dict:
    """The keyword arguments routes/generate.py passes, taken from the same request fields."""
    kwargs = {
        "key": req.key,
        "genre": req.genre,
        "context": req.context,
        "selected_measures": req.selectedMeasures,
        "time_signature": req.timeSignature,
        "model_key": req.modelKey,
        "bpm": req.bpm,
    }
    if takes_section:
        kwargs["section_type"] = req.sectionType
    return kwargs


def _count_model_bars(sink: list[int]) -> Callable[..., Any] | None:
    """Wrap inference.generate_with_model to record how many bars each sampling pass
    returns before the service pads them to the request. Returns the original for
    restoring, or None when the service has no such function (model_bars stays null)."""
    original = getattr(inference, "generate_with_model", None)
    if original is None:
        return None

    def counted(*args, **kwargs):
        bars = original(*args, **kwargs)
        if isinstance(bars, list):
            sink.append(len(bars))
        return bars

    inference.generate_with_model = counted
    return original


def _load_model(registry: inference.ModelRegistry, model_key: str) -> dict:
    """Load one model, outside the timed runs, and log how that went."""
    start = time.perf_counter()
    loaded = registry.load_model(model_key)
    model = {
        "label": inference.MODEL_LABELS.get(model_key, model_key),
        "loaded": loaded is not None,
        "device": str(loaded.device) if loaded is not None else None,
        "load_ms": _ms(time.perf_counter() - start),
    }
    if model["loaded"]:
        _log(f"{model_key} loaded on {model['device']} in {model['load_ms']:.0f} ms")
    else:
        _log(f"{model_key} unavailable; its runs will return None")
    return model


def _timed_call(kwargs: dict, produced: list[int]) -> dict:
    """One generate_ai_suggestions call, its on_phase marks turned into gaps."""
    marks: list[tuple[str, object, float]] = []

    def on_phase(stage, payload=None, *_args, **_kwargs) -> None:
        step = payload.get("step") if isinstance(payload, dict) else None
        marks.append((str(stage), step, time.perf_counter()))

    produced.clear()
    start = time.perf_counter()
    result = inference.generate_ai_suggestions(**kwargs, on_phase=on_phase)
    end = time.perf_counter()

    setup = (marks[0][2] if marks else end) - start
    phase_s = {"setup": setup}
    stage_s = {"setup": setup}
    events = []
    # Each mark's stage lasts until the next mark, the last one until the return
    stage_ends = [at for _, _, at in marks[1:]] + [end]
    for (stage, step, at), stage_end in zip(marks, stage_ends):
        gap = stage_end - at
        name = stage if step is None else f"{stage}_{step}"
        phase_s[name] = phase_s.get(name, 0.0) + gap
        stage_s[stage] = stage_s.get(stage, 0.0) + gap
        events.append({"phase": stage, "step": step, "t_ms": _ms(at - start), "gap_ms": _ms(gap)})
    return {
        "total_ms": _ms(end - start),
        "result": "ai" if result is not None else "none",
        "suggestions": len(result) if result else 0,
        "model_bars": list(produced) or None,
        "events": events,
        "phase_ms": {name: _ms(s) for name, s in phase_s.items()},
        "stage_ms": {name: _ms(s) for name, s in stage_s.items()},
    }


def _recorded_runs(args: argparse.Namespace, kwargs: dict, produced: list[int]) -> list[dict]:
    """Reseed torch, make the unrecorded warm-up calls, then time the recorded ones."""
    torch.manual_seed(args.seed)
    for _ in range(args.warmup):
        _timed_call(kwargs, produced)
    return [_timed_call(kwargs, produced) for _ in range(args.runs)]


def _medians(rows: list[dict[str, float]]) -> dict[str, float]:
    """Median of every key over the runs that have it, in first-seen order."""
    collected: dict[str, list[float]] = {}
    for row in rows:
        for name, value in row.items():
            collected.setdefault(name, []).append(value)
    return {name: round(statistics.median(values), 3) for name, values in collected.items()}


def _share(runs: list[dict], stage: str) -> float:
    """Median over the runs of the stage's fraction of the run's wall time."""
    return round(statistics.median(
        run["stage_ms"].get(stage, 0.0) / (run["total_ms"] or 1.0) for run in runs
    ), 4)


def _summary(runs: list[dict]) -> dict:
    totals = [run["total_ms"] for run in runs]
    produced = [n for run in runs for n in (run["model_bars"] or [])]
    return {
        "total_ms": {
            "p50": round(_percentile(totals, 0.50), 3),
            "p95": round(_percentile(totals, 0.95), 3),
            "min": min(totals),
            "max": max(totals),
        },
        "phase_median_ms": _medians([run["phase_ms"] for run in runs]),
        "stage_median_ms": _medians([run["stage_ms"] for run in runs]),
        "composing_share": _share(runs, "composing"),
        "ranking_share": _share(runs, "ranking"),
        "ai_runs": sum(run["result"] == "ai" for run in runs),
        "suggestion_counts": sorted({run["suggestions"] for run in runs}),
        "cold_runs": sum("model_load" in run["stage_ms"] for run in runs),
        # Bars each sampling pass returned before padding; fewer than requested means the
        # model hit EOS or its token budget.
        "model_bars_stats": (
            {"min": min(produced), "median": statistics.median(produced), "max": max(produced)}
            if produced else None
        ),
    }


def _settings(args: argparse.Namespace, takes_section: bool) -> dict:
    """The probe's own settings, plus the request defaults it leaves to GenerateRequest."""
    return {
        "bars": args.bars,
        "genres": args.genres,
        "runs": args.runs,
        "warmup": args.warmup,
        "context_bars": args.context_bars,
        "key": args.key,
        "threads": args.threads,
        "seed": args.seed,
        "section_type": args.section_type,
        "service_takes_section_type": takes_section,
        "time_signature": list(GenerateRequest.model_fields["timeSignature"].default),
        "bpm": GenerateRequest.model_fields["bpm"].default,
    }


def _environment(models: dict[str, dict]) -> dict:
    devices = sorted({model["device"] for model in models.values() if model["device"]})
    return {
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "device": ", ".join(devices) or "none (no model loaded)",
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "flags": {name: os.environ.get(name) for name in ENV_FLAGS},
        "service_modules": {
            "torch": getattr(inference, "TORCH_AVAILABLE", None),
            "peft": getattr(inference, "PEFT_AVAILABLE", None),
            "physics_rerank": getattr(inference, "PHYSICS_AVAILABLE", None),
            "rag": getattr(inference, "RAG_AVAILABLE", None),
        },
    }


def _notes(cells: list[dict]) -> list[str]:
    notes = []
    for cell in cells:
        where, runs = f"{cell['genre']} x {cell['bars']} bars", len(cell["runs"])
        if cell["ai_runs"] < runs:
            notes.append(f"- {where}: {runs - cell['ai_runs']} of {runs} runs returned None "
                         "(no AI result).")
        if cell["cold_runs"]:
            notes.append(f"- {where}: a model was loaded inside {cell['cold_runs']} timed run(s).")
        stats = cell["model_bars_stats"]
        if stats is not None and stats["min"] < cell["bars"]:
            notes.append(f"- {where}: sampling passes returned {stats['min']} to {stats['max']} "
                         f"bars (median {stats['median']:g}) of {cell['bars']} requested; "
                         "the service pads the rest.")
    return ["", "Notes:", "", *notes] if notes else []


def _markdown(report: dict) -> str:
    settings, env = report["settings"], report["environment"]
    lines = [
        "# Chord-generation speed probe",
        "",
        f"Per cell: {settings['runs']} recorded and {settings['warmup']} warm-up runs; "
        f"key {settings['key']}, context bars {settings['context_bars']}, "
        f"section type {settings['section_type'] or 'none'}, seed {settings['seed']}; "
        f"created {report['created_utc']}.",
        "",
        "| genre | bars | model | p50 ms | p95 ms | composing share | ranking share |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for cell in report["cells"]:
        total = cell["total_ms"]
        lines.append(
            f"| {cell['genre']} | {cell['bars']} | {cell['model_key']} | {total['p50']:.1f} "
            f"| {total['p95']:.1f} | {cell['composing_share']:.1%} | {cell['ranking_share']:.1%} |"
        )
    loads = ", ".join(
        f"{key} {model['load_ms']:.0f} ms" if model["loaded"] else f"{key} not loaded"
        for key, model in report["models"].items()
    )
    lines += [
        "",
        "A stage runs from its on_phase event to the next event or the return; a share is the "
        "median over the recorded runs of stage time / wall time.",
        "",
        f"Model load, once per model and outside the timed runs: {loads}.",
        *_notes(report["cells"]),
        "",
        "## Environment",
        "",
    ]
    rows = [
        ("torch", env["torch"]),
        ("torch threads", f"{env['torch_threads']} (interop {env['torch_interop_threads']})"),
        ("device", env["device"]),
        ("platform.machine", env["machine"] or "(empty)"),
        ("platform.processor", env["processor"] or "(empty)"),
        ("python", env["python"]),
        ("service modules", ", ".join(
            f"{name} {'yes' if ok else 'no'}" for name, ok in env["service_modules"].items()
        )),
    ]
    rows += [(name, "(unset)" if value is None else value or "(empty)")
             for name, value in env["flags"].items()]
    width = max(len(name) for name, _ in rows)
    lines += [f"    {name.ljust(width)}  {value}" for name, value in rows]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.threads is not None:
        torch.set_num_threads(args.threads)

    # section_type is newer than the other service arguments; pass it only when taken.
    takes_section = "section_type" in inspect.signature(inference.generate_ai_suggestions).parameters
    if args.section_type is not None and not takes_section:
        sys.exit("speed_probe: generate_ai_suggestions takes no section_type; drop --section-type")

    registry = inference.ModelRegistry.get()
    produced: list[int] = []
    original_generate = _count_model_bars(produced)
    models: dict[str, dict] = {}
    cells: list[dict] = []
    try:
        for genre in args.genres:
            # Same resolution as the service with modelKey None; loads only this genre's model.
            model_key = inference.dispatch_model_for_genre(genre)
            if model_key not in models:
                models[model_key] = _load_model(registry, model_key)
            for bars in args.bars:
                kwargs = _service_kwargs(_request(args, genre, bars), takes_section)
                runs = _recorded_runs(args, kwargs, produced)
                cell = {
                    "genre": genre,
                    "bars": bars,
                    "model_key": model_key,
                    "model_label": models[model_key]["label"],
                    **_summary(runs),
                    "runs": runs,
                }
                cells.append(cell)
                total = cell["total_ms"]
                _log(f"{genre} x {bars} bars on {model_key}: p50 {total['p50']:.1f} ms, "
                     f"p95 {total['p95']:.1f} ms, {cell['ai_runs']}/{len(runs)} AI")
    finally:
        if original_generate is not None:
            inference.generate_with_model = original_generate

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "settings": _settings(args, takes_section),
        "environment": _environment(models),
        "models": models,
        "cells": cells,
    }
    markdown = _markdown(report)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
        md_path = args.out.with_suffix(".md")
        md_path.write_text(markdown, encoding="utf-8")
        _log(f"wrote {args.out} and {md_path}")
    sys.stdout.write(markdown)


if __name__ == "__main__":
    main()
