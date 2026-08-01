"""Fail closed when a hosted model catalog cannot serve every advertised choice."""

from app.services import inference


def require_full_catalog() -> list[str]:
    if not inference.TORCH_AVAILABLE or not inference.PEFT_AVAILABLE:
        raise RuntimeError("Serving the full catalog requires Torch and PEFT")
    missing = []
    for key, path in inference.CHECKPOINTS.items():
        required = (
            (path / "adapter" / "adapter_config.json",
             path / "adapter" / "adapter_model.safetensors",
             path / "embedding_extension.pt")
            if key.startswith("ft_f1_lora_") else (path,)
        )
        if any(not item.is_file() or item.stat().st_size == 0 for item in required):
            missing.append(key)
    if missing:
        raise RuntimeError("Missing model assets: " + ", ".join(missing)
                           + "; run model/_download.py without --smoke")

    # Check assets before entering the legacy LoRA loader, which permits F1 fallback.
    # A hosted catalog must never advertise such a substitution as the requested model.
    registry = inference.ModelRegistry.get()
    for key in inference.CHECKPOINTS:
        loaded = registry.load_model(key)
        if loaded is None or loaded.name != key:
            raise RuntimeError(f"Requested model {key} did not load as itself")
    return list(inference.CHECKPOINTS)
