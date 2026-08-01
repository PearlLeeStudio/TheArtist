"""Model management endpoints."""

from fastapi import APIRouter, Depends

from app.services import lockdown
from app.services.inference import (
    TORCH_AVAILABLE, ModelRegistry, CHECKPOINTS, MODEL_LABELS, DEFAULT_MODEL,
    GENRE_MODEL_DISPATCH, dispatch_model_for_genre, warmup_keys,
)

router = APIRouter()
_warmup_lockdown = lockdown.refuse_on_public_host(
    "Model warmup is not available on this public server; checkpoints load at startup (THEARTIST_WARMUP_KEYS) "
    "or on first use.")


def _checkpoint_available(key: str) -> bool:
    """Check checkpoint existence — supports both F-series .pt and LoRA dir."""
    p = CHECKPOINTS[key]
    if key.startswith("ft_f1_lora_"):
        return (p / "adapter").exists()
    return p.exists()


def _is_recommended(key: str) -> bool:
    """Pearl 2026-05-09 — F1 + F4 are the recommended F-series picks.
    2026-06-12: F1 v2 (selection-corrected retrain) joins them — it is the
    artifact that actually embodies the F1 design intent."""
    return key in {"ft_f1", "ft_f1_v2", "ft_f4"}


def _is_lora(key: str) -> bool:
    return key.startswith("ft_f1_lora_")


@router.get("/models/status")
def model_status() -> dict:
    if not TORCH_AVAILABLE:
        return {"torch_available": False, "models": {}}
    registry = ModelRegistry.get()
    return {
        "torch_available": True,
        "models": {
            name: registry.is_loaded(name)
            for name in CHECKPOINTS
        },
    }


@router.post("/models/warmup", dependencies=[Depends(_warmup_lockdown)])
def warmup_models() -> dict:
    """Preload every checkpoint, or the THEARTIST_WARMUP_KEYS list; names that are no checkpoint are reported."""
    keys, unknown = warmup_keys()  # a misspelled key is reported on every server, with or without torch
    if not TORCH_AVAILABLE:
        return {"torch_available": False, "loaded": {}, "unknown": unknown}
    results = ModelRegistry.get().warmup(keys)
    return {"torch_available": True, "loaded": results, "unknown": unknown}


@router.get("/models/list")
def list_models() -> dict:
    """List all models (paper F-series + LoRA per-genre) with labels.

    Two groups for UI:
    - paper: F-series + Phase 0 (research checkpoints)
    - lora:  F1 base + LoRA adapter per genre (R4 production path)
    """
    paper = []
    lora = []
    for key in CHECKPOINTS:
        entry = {
            "key": key,
            "label": MODEL_LABELS.get(key, key),
            "available": _checkpoint_available(key),
            "recommended": _is_recommended(key),
        }
        (lora if _is_lora(key) else paper).append(entry)

    return {
        "default": DEFAULT_MODEL,
        "models": paper + lora,   # backward-compat flat list
        "groups": {
            "paper": paper,
            "lora": lora,
        },
    }


@router.get("/genres/list")
def list_genres() -> dict:
    """13-genre vocabulary + which has a LoRA adapter available."""
    genres = []
    for genre, model_key in GENRE_MODEL_DISPATCH.items():
        served = dispatch_model_for_genre(genre)
        genres.append({
            "key": genre,
            "label": genre.replace("_", " "),  # rnb_soul → "rnb soul"
            "model_key": model_key,             # ideal route
            "served_by": served,                # actual route (fallback to ft_f1 if LoRA not ready)
            "lora_ready": served == model_key and model_key.startswith("ft_f1_lora_"),
        })
    return {
        "default_genre": "jazz",
        "genres": genres,
    }
