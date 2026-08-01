"""Measure checkpoint loading and fingerprint inference.

Use the project dev.sh py wrapper, an absolute --out path, and the same keys
before/after. The fresh process retains every requested model. File-system cache
is not flushed; imports, loading and first forward are reported separately.
"""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--keys", nargs="+", default=["ft_f1", "ft_f4", "ft_f1_lora_rock"])
    args = parser.parse_args()
    if not args.out.is_absolute():
        parser.error("--out must be absolute")
    started = time.perf_counter()
    from app.services import inference
    torch = inference.torch
    torch.set_num_threads(2)
    import_seconds = time.perf_counter() - started
    registry = inference.ModelRegistry.get()
    rows = []
    for key in args.keys:
        started = time.perf_counter()
        loaded = registry.load_model(key)
        if loaded is None or loaded.name != key:
            raise RuntimeError(f"Requested checkpoint is unavailable: {key}")
        seconds = time.perf_counter() - started
        model = loaded.model.get_base_model() if isinstance(loaded.model, inference.PeftModel) else loaded.model
        ids = inference.build_prompt(registry.tokenizer, "C major", "pop", (4, 4), [], [])
        inputs = torch.tensor([ids], device=loaded.device)
        started = time.perf_counter()
        with torch.inference_mode():
            logits = model(inputs).cpu().numpy().tobytes()
        rows.append(dict(key=key, load_seconds=round(seconds, 4),
                         first_forward_seconds=round(time.perf_counter() - started, 4),
                         logits_sha256=hashlib.sha256(logits).hexdigest()))
    # A second forward after all adapters have loaded detects shared-base mutation.
    for row in rows:
        loaded = registry.load_model(row['key'])
        model = loaded.model.get_base_model() if isinstance(loaded.model, inference.PeftModel) else loaded.model
        with torch.inference_mode():
            digest = hashlib.sha256(model(inputs).cpu().numpy().tobytes()).hexdigest()
        if digest != row['logits_sha256']:
            raise AssertionError(f"Another model load changed {row['key']}")
    report = dict(import_seconds=round(import_seconds, 4), torch=torch.__version__,
                  threads=torch.get_num_threads(), platform=sys.platform,
                  peak_mib=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss /
                                 (1024**2 if sys.platform == 'darwin' else 1024), 1),
                  all_models_retained=True, models=rows)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
