"""Shared loader helpers for eval/analysis scripts.

`strip_compile_prefix` removes the `_orig_mod.` prefix that torch.compile
adds to state_dict keys when a model is saved while still wrapped. Idempotent.
"""
from __future__ import annotations


def strip_compile_prefix(sd: dict) -> dict:
    if not any(k.startswith("_orig_mod.") for k in sd):
        return sd
    return {
        (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
        for k, v in sd.items()
    }
