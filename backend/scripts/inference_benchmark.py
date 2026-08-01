"""Paired local inference benchmark; no downloads or working dates.

Run with the project's dev.sh py wrapper and an absolute --out JSON path.
Each pair uses the same seed and checkpoint. Model loading is excluded; all
three candidates, ranking and voicing remain enabled. Timing order alternates.
"""

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import time

import torch

from app.models.plan import PlanRequest
from app.services import inference, planner


@contextmanager
def cache_mode(enabled):
    name = "THEARTIST_DISABLE_KV_CACHE"
    previous = os.environ.get(name)
    os.environ[name] = "0" if enabled else "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def canonical(value):
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: canonical(item) for key, item in value.items()
                if key not in {"id", "elapsedMs", "generationMs"}}
    if isinstance(value, (list, tuple)):
        return [canonical(item) for item in value]
    return value


def paired(name, call, runs):
    # Warm both execution paths before measuring either.
    for enabled in (False, True):
        with cache_mode(enabled):
            call(17)
    times = {False: [], True: []}
    digests = []
    for run in range(runs):
        results = {}
        for enabled in ((False, True) if run % 2 == 0 else (True, False)):
            with cache_mode(enabled):
                start = time.perf_counter()
                result = call(17 + run)
                times[enabled].append(round((time.perf_counter() - start) * 1000, 3))
            results[enabled] = json.dumps(canonical(result), sort_keys=True)
        if results[False] != results[True]:
            raise AssertionError(f"Cached output differs: {name}, seed {17 + run}")
        digests.append(hashlib.sha256(results[True].encode()).hexdigest())
    before, after = (statistics.median(times[enabled]) for enabled in (False, True))
    row = dict(case=name, uncached_ms=times[False], cached_ms=times[True],
               median_before_ms=before, median_after_ms=after,
               speedup=round(before / after, 3), identical_outputs=True, output_sha256=digests)
    print(f"{name}: {before:.1f} -> {after:.1f} ms ({before / after:.2f}x), identical", flush=True)
    return row


def suggestions(model_key, genre, bars, seed):
    result = inference.generate_ai_result(
        key="C major", genre=genre, context=[], selected_measures=list(range(bars)),
        model_key=model_key, seed=seed,
    )
    if result is None or len(result.suggestions) != 3:
        raise AssertionError("Expected all three model candidates")
    return result.suggestions


def song_form(seed):
    result = planner.complete_form(PlanRequest(
        genre="pop", key="C major", targetBars=32, modelKey="ft_f1", seed=seed,
    ))
    if result.status != "ok":
        raise AssertionError("Expected a complete song form")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.runs < 1 or not args.out.is_absolute():
        parser.error("Use at least one run and an absolute output path")
    if not inference.rag_disabled():
        parser.error("Disable RAG for this local-only benchmark")
    rows = []
    registry = inference.ModelRegistry.get()
    for model_key, genre in (("ft_f1", "pop"), ("ft_f4", "jazz"), ("ft_f1_lora_rock", "rock")):
        loaded = registry.load_model(model_key)
        if loaded is None:
            raise RuntimeError(f"Install the checkpoint before benchmarking: {model_key}")
        if loaded.device.type != "cpu":
            raise RuntimeError("This benchmark measures the CPU runtime")
        for bars in (8, 32, 64):
            rows.append(paired(f"{model_key}/{bars} bars/3 candidates",
                               lambda seed: suggestions(model_key, genre, bars, seed), args.runs))
    rows.append(paired("song form/32 bars/local planner", song_form, args.runs))
    args.out.write_text(json.dumps(dict(
        runtime=dict(torch=torch.__version__, threads=torch.get_num_threads(),
                     platform=platform.system(), architecture=platform.machine(), device="cpu"),
        runs=args.runs, seeds=list(range(17, 17 + args.runs)), warmup_per_mode=1,
        exclusions=["checkpoint loading", "RAG", "HTTP and browser"],
        cases=rows,
    ), indent=2) + "\n")


if __name__ == "__main__":
    main()
