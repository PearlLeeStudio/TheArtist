"""Custom prefix-tuning for the MusicTransformer.

HuggingFace PEFT's PrefixTuningConfig requires the base model to expose a
HF-transformers-style `past_key_values` interface. The MusicTransformer is a
custom PyTorch implementation that does not, so we implement prefix-tuning
directly here.

Design (Li & Liang 2021, "Prefix-Tuning: Optimizing Continuous Prompts for
Generation"):

- A learnable table of shape (n_layers, 2, prefix_len, n_heads, d_head)
  holds per-layer per-head key/value prefixes.
- At forward time the prefix is broadcast across the batch and prepended
  to each layer's attention K/V; queries attend over (prefix_len + L)
  positions, output stays length L.
- The base model is fully frozen; only the prefix table is trainable.

Mask handling: prefix slots are unconditionally attendable (we extend the
(L, L) causal mask by a (L, prefix_len) all-False block on the left). The
relative-position bias only spans content-content positions; prefix-key
attention scores receive content×prefix dot-product only (no rel-pos
term added), which matches the standard prefix-tuning recipe where
prefix slots are positionless virtual tokens.

Position embeddings: MusicTransformer uses relative position (no absolute
positional add to the token embeddings), so prepending prefix does not
shift content positions. This is one of the reasons the integration is
clean.

Usage:

    base = MusicTransformer(...)
    base.load_state_dict(...)
    model = PrefixTuningWrapper(base, prefix_len=20)
    # forward exactly like the base model:
    logits = model(input_ids)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PrefixEncoder(nn.Module):
    """Learnable per-layer per-head key/value prefix table.

    Parameters
    ----------
    n_layers: number of transformer blocks in the base model.
    n_heads:  attention heads per block.
    d_head:   per-head dim (d_model // n_heads).
    prefix_len: virtual tokens prepended at every layer.
    init_std: stddev for normal init of the prefix table.

    Shape of self.prefix: (n_layers, 2, prefix_len, n_heads, d_head).
    The "2" axis is [key, value].
    """

    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        d_head: int,
        prefix_len: int = 20,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_head = d_head
        self.prefix_len = prefix_len
        # (n_layers, 2, prefix_len, n_heads, d_head)
        self.prefix = nn.Parameter(
            torch.empty(n_layers, 2, prefix_len, n_heads, d_head)
        )
        nn.init.normal_(self.prefix, mean=0.0, std=init_std)

    def get_prefix(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Return one (K, V) tuple per layer, each shape (B, n_heads, prefix_len, d_head).

        The base parameter dtype may not match the AMP autocast dtype; we cast
        on the fly to whatever the activations are running in.
        """
        # cast & move once, then slice
        p = self.prefix.to(device=device, dtype=dtype)
        out: list[tuple[torch.Tensor, torch.Tensor]] = []
        for li in range(self.n_layers):
            # (2, prefix_len, n_heads, d_head)
            kv = p[li]
            # Transpose to (2, n_heads, prefix_len, d_head) for attention layout
            kv = kv.permute(0, 2, 1, 3).contiguous()
            # Expand across batch (no copy thanks to broadcast)
            K = kv[0].unsqueeze(0).expand(batch_size, -1, -1, -1)
            V = kv[1].unsqueeze(0).expand(batch_size, -1, -1, -1)
            out.append((K, V))
        return out

    def extra_repr(self) -> str:
        return (
            f"n_layers={self.n_layers}, n_heads={self.n_heads}, "
            f"d_head={self.d_head}, prefix_len={self.prefix_len}, "
            f"params={self.prefix.numel():,}"
        )


class PrefixTuningWrapper(nn.Module):
    """Wraps a frozen MusicTransformer with a learnable PrefixEncoder.

    Forwarding `input_ids` produces logits of the same shape as the base
    model. The base model parameters are frozen on construction; only the
    PrefixEncoder parameters require grad.

    Attribute naming intentionally mirrors the peft conventions used
    elsewhere in this codebase (a `.base_model` attribute reachable from
    eval / save code paths), so existing call sites that read
    `wrapper.base_model.model.token_emb` keep working — we expose a
    `.base_model` attribute with a `.model` shim that points at the base
    MusicTransformer.
    """

    def __init__(
        self,
        base_model: nn.Module,
        prefix_len: int = 20,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        # Freeze every base parameter.
        for p in base_model.parameters():
            p.requires_grad = False

        # Infer attention shape from the base model.
        n_layers = len(base_model.layers)
        first_attn = base_model.layers[0].attn
        n_heads = first_attn.n_heads
        d_head = first_attn.d_k

        self.model = base_model  # primary attribute used by .forward
        self.prefix_encoder = PrefixEncoder(
            n_layers=n_layers,
            n_heads=n_heads,
            d_head=d_head,
            prefix_len=prefix_len,
            init_std=init_std,
        )
        self.prefix_len = prefix_len

    # Shim so that `wrapper.base_model.model.token_emb` paths keep
    # working, mirroring peft's nested layout (PeftModel.base_model.model).
    @property
    def base_model(self) -> "PrefixTuningWrapper":
        return self

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B = input_ids.shape[0]
        # Match the autocast dtype of activations.
        dtype = self.model.token_emb.weight.dtype
        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_gpu_dtype()
        prefix_kvs = self.prefix_encoder.get_prefix(B, input_ids.device, dtype)
        return self.model(input_ids, prefix_kvs=prefix_kvs)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_total(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # --- Save / load helpers --------------------------------------------
    def prefix_state_dict(self) -> dict[str, torch.Tensor]:
        """Return only the PrefixEncoder weights (small file)."""
        return {"prefix": self.prefix_encoder.prefix.detach().cpu().clone()}

    def load_prefix_state_dict(self, sd: dict[str, torch.Tensor]) -> None:
        p = sd["prefix"]
        if p.shape != self.prefix_encoder.prefix.shape:
            raise ValueError(
                f"Prefix shape mismatch: saved {tuple(p.shape)} "
                f"vs wrapper {tuple(self.prefix_encoder.prefix.shape)}"
            )
        with torch.no_grad():
            self.prefix_encoder.prefix.copy_(p)
