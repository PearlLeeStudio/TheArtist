"""Music Transformer with relative attention for chord generation.

Architecture: Transformer decoder (autoregressive) with relative position
attention (Shaw et al. 2018, as in the Music Transformer of Huang et al. 2019).
There is no absolute position embedding. Each layer learns one relative
embedding table shared by its heads, and the query-key offsets are looked up
by index gather, not by the skewing trick of Huang et al. The relative term
is added to the attention logits only. Input embedding and output projection
are tied.

Default config (25,661,440 params, the tied table counted once):
    d_model=512, n_heads=8, d_ff=2048, n_layers=8
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelativeMultiHeadAttention(nn.Module):
    """Multi-head self-attention with relative position bias."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = math.sqrt(self.d_k)

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        # Learnable relative position embeddings: positions in [-max_len+1, max_len-1]
        self.max_seq_len = max_seq_len
        self.rel_emb = nn.Embedding(2 * max_seq_len - 1, self.d_k)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        prefix_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        *,
        kv_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, L, D)
            mask: (L, L) bool — True = masked (don't attend)
            prefix_kv: optional (K_pref, V_pref) each of shape
                (B, n_heads, prefix_len, d_head). When provided, the
                prefix is prepended along the key/value sequence
                dimension and the queries attend over (prefix_len + L)
                positions. Prefix slots are positionless virtual tokens
                (no relative-position bias is added for them) and are
                always attendable (causal mask is extended with False
                columns on the left).
            kv_cache: optional caller-owned slot containing the previous K/V
                pair. Updated in place with the appended keys and values.
        Returns:
            (B, L, D)
        """
        B, L, _ = x.shape
        H, dk = self.n_heads, self.d_k

        Q = self.w_q(x).view(B, L, H, dk).transpose(1, 2)  # (B, H, L, dk)
        K = self.w_k(x).view(B, L, H, dk).transpose(1, 2)
        V = self.w_v(x).view(B, L, H, dk).transpose(1, 2)
        if kv_cache is not None:
            if kv_cache:
                past_k, past_v = kv_cache[0]
                K = torch.cat([past_k, K], dim=2)
                V = torch.cat([past_v, V], dim=2)
            kv_cache[:] = [(K, V)]

        # Content attention: Q K^T
        content = torch.matmul(Q, K.transpose(-2, -1))  # (B, H, query length, key length)

        # Relative position attention: Q R^T via efficient gather
        rel = self._relative_attention(Q, L, K.shape[2])

        # Content+rel attention scores over the real keys (including cached ones).
        attn_content = (content + rel) / self.scale

        if prefix_kv is not None:
            K_pref, V_pref = prefix_kv  # (B, H, P, dk) each
            # Prefix attention: Q K_pref^T (no relative-position term —
            # prefix slots are positionless virtual tokens).
            attn_pref = torch.matmul(Q, K_pref.transpose(-2, -1)) / self.scale  # (B, H, L, P)
            # Concatenate along key dim so softmax is computed jointly.
            attn = torch.cat([attn_pref, attn_content], dim=-1)  # (B, H, L, P+L)
            if mask is not None:
                # Original (L, L) causal mask — extend to (L, P+L) by
                # left-prepending an all-False (P) block: prefix is
                # always attendable.
                P = K_pref.shape[2]
                pref_mask = torch.zeros(L, P, device=mask.device, dtype=torch.bool)
                full_mask = torch.cat([pref_mask, mask], dim=-1)  # (L, P+L)
                attn = attn.masked_fill(full_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
            attn = self.dropout(F.softmax(attn, dim=-1))
            # Split attention weights back into the prefix vs content
            # blocks and matmul against the corresponding value tensors.
            P = K_pref.shape[2]
            attn_p = attn[..., :P]              # (B, H, L, P)
            attn_c = attn[..., P:]              # (B, H, L, L)
            out = torch.matmul(attn_p, V_pref) + torch.matmul(attn_c, V)  # (B, H, L, dk)
        else:
            attn = attn_content
            if mask is not None:
                attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
            attn = self.dropout(F.softmax(attn, dim=-1))
            out = torch.matmul(attn, V)  # (B, H, L, dk)

        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.w_o(out)

    def _relative_attention(self, Q: torch.Tensor, L: int, key_length: int) -> torch.Tensor:
        """Compute Q @ R^T using relative position embeddings.

        Uses the index-gather approach: for each (i, j) pair, the relative
        position is j - i, shifted to a non-negative index.
        """
        device = Q.device
        # Relative position indices: rel[i,j] = j - i + max_seq_len - 1
        positions = torch.arange(key_length, device=device)
        queries = positions[-L:]
        rel_idx = positions.unsqueeze(0) - queries.unsqueeze(1) + self.max_seq_len - 1
        rel_idx = rel_idx.clamp(0, 2 * self.max_seq_len - 2)

        R = self.rel_emb(rel_idx)  # (query length, key length, dk)
        BH = Q.shape[0] * Q.shape[1]
        Q_flat = Q.reshape(BH, L, self.d_k)
        rel_score = torch.einsum("bld,lsd->bls", Q_flat, R)
        return rel_score.view(Q.shape[0], Q.shape[1], L, key_length)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer decoder block."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = RelativeMultiHeadAttention(d_model, n_heads, max_seq_len, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        prefix_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        *,
        kv_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x), mask, prefix_kv=prefix_kv, kv_cache=kv_cache))
        x = x + self.ffn(self.norm2(x))
        return x


class MusicTransformer(nn.Module):
    """Autoregressive Music Transformer for chord generation."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 2048,
        n_layers: int = 8,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, max_seq_len, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying (embedding ↔ output projection)
        self.out_proj.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self) -> None:
        for name, p in self.named_parameters():
            if p.dim() > 1 and "token_emb" not in name:
                nn.init.xavier_uniform_(p)
        # Embedding std=1/sqrt(d_model) so that after *sqrt(d_model) scaling
        # inputs have unit variance, and weight-tied output logits stay small
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=self.d_model ** -0.5)

    @staticmethod
    def _causal_mask(L: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular causal mask (True = masked)."""
        return torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)

    def forward(
        self,
        input_ids: torch.Tensor,
        prefix_kvs: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        *,
        kv_cache: list[list[tuple[torch.Tensor, torch.Tensor]]] | None = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (B, L) token IDs
            prefix_kvs: optional per-layer prefix (K, V) tuples used by
                the prefix-tuning wrapper. Must have length == n_layers
                or be None.
            kv_cache: inference-only caller-owned list of per-layer slots.
                Start with one empty list per layer, then pass only new tokens.
        Returns:
            logits: (B, L, vocab_size)
        """
        B, L = input_ids.shape
        x = self.token_emb(input_ids) * math.sqrt(self.d_model)
        x = self.drop(x)

        past_length = 0
        if kv_cache is not None:
            if self.training or torch.is_grad_enabled() or prefix_kvs is not None:
                raise ValueError("KV caching requires inference without prefix tuning")
            if len(kv_cache) != len(self.layers):
                raise ValueError("KV cache must have one entry per layer")
            if kv_cache[0]:
                past_length = kv_cache[0][0][0].shape[2]
            if past_length + L > self.max_seq_len:
                raise ValueError("KV cache exceeds the context window; prefill a new cache")
        # A single newly appended token may attend to every cached position.
        mask = None if past_length and L == 1 else self._causal_mask(past_length + L, input_ids.device)[-L:]
        if prefix_kvs is not None and len(prefix_kvs) != len(self.layers):
            raise ValueError(
                f"prefix_kvs has length {len(prefix_kvs)} but model has "
                f"{len(self.layers)} layers."
            )
        for i, layer in enumerate(self.layers):
            pkv = prefix_kvs[i] if prefix_kvs is not None else None
            x = layer(x, mask, prefix_kv=pkv, kv_cache=kv_cache[i] if kv_cache is not None else None)

        return self.out_proj(self.norm(x))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9,
        eos_id: int = 2,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        ignore_repeat_token_ids: set[int] | None = None,
        *,
        n_bars: int | None = None,
        bar_id: int = 3,
        chord_ids: set[int] | None = None,
        end_ids: set[int] | None = None,
        generator: torch.Generator | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """Autoregressive generation from a prompt.

        Args:
            prompt_ids: (1, L) token IDs including [BOS] and context.
            max_new_tokens: maximum tokens to generate.
            temperature: sampling temperature (lower = more deterministic).
            top_k: keep only top-k logits (0 = disabled).
            top_p: nucleus sampling threshold.
            eos_id: stop token.
            repetition_penalty: divide logits of previously-seen tokens by
                this factor (HF convention). > 1.0 discourages repeats.
                1.0 disables. Typical: 1.2–1.5.
            no_repeat_ngram_size: ban candidate tokens that would complete
                an n-gram already present in the current sequence (n =
                this value). 0 disables. Typical: 3 for chord sequences.
            ignore_repeat_token_ids: token ids exempt from the two repetition
                controls above — e.g. [BAR] or other separators that
                *should* recur. If None, no exemptions.
            n_bars: write exactly this many complete bars after the prompt.
                ``eos_id`` and ``end_ids`` are banned until they are
                complete, and generation stops as soon as they are;
                ``max_new_tokens`` still caps the pass, so it can come back
                short. Bars count as the tokenizer encodes them: ``bar_id``
                followed by one or more ``chord_ids``, complete once the
                next ``bar_id`` closes it (that closing ``bar_id`` ends the
                output). A ``bar_id`` that would close an empty bar is
                banned too, so runs of empty bars cannot use up the budget
                (they appear once the window has slid past the prompt's
                header). Chords before the first generated ``bar_id``
                continue the prompt's last bar and do not count. None (the
                default) stops on ``eos_id`` or ``max_new_tokens`` alone.
            bar_id: the bar separator ([BAR]); used with ``n_bars``.
            chord_ids: the chord token ids; required with ``n_bars``.
            end_ids: other tokens that would end the sequence (e.g. [PAD]),
                banned with ``eos_id`` under ``n_bars``.
            generator: request-local sampling RNG on the same device as the
                model. None retains the caller's normal unseeded sampling.
            use_cache: reuse request-local attention keys/values during inference.
                Rebuild after cropping the context so sliding-window semantics
                match the uncached path. Training and prefix tuning are unchanged.

        Returns:
            (1, L') full sequence including prompt and generated tokens.
        """
        self.eval()
        ids = prompt_ids.clone()
        exempt = ignore_repeat_token_ids or set()

        if n_bars is not None and chord_ids is None:
            raise ValueError("generate(n_bars=...) needs chord_ids")
        banned_ends = sorted({eos_id, *(end_ids or ())})
        bars_done = 0
        bar_chords: int | None = None  # chords in the bar being written; None before its [BAR]
        kv_cache = None

        for _ in range(max_new_tokens):
            if n_bars is not None and bars_done >= n_bars:
                break
            ctx = ids[:, -self.max_seq_len :]
            if use_cache:
                if kv_cache is None or ids.shape[1] > self.max_seq_len:
                    kv_cache = [[] for _ in self.layers]
                    logits = self(ctx, kv_cache=kv_cache)[:, -1, :]
                else:
                    logits = self(ids[:, -1:], kv_cache=kv_cache)[:, -1, :]
            else:
                logits = self(ctx)[:, -1, :]
            logits = logits / max(temperature, 1e-8)

            # Repetition penalty (HuggingFace-style): scale already-seen token
            # logits so they are less attractive. Positive logits get divided,
            # negative logits get multiplied (stays "less attractive" either sign).
            if repetition_penalty != 1.0:
                seen = set(ids[0].tolist()) - exempt
                if seen:
                    idx = torch.tensor(list(seen), device=logits.device, dtype=torch.long)
                    vals = logits[0, idx]
                    vals = torch.where(
                        vals > 0,
                        vals / repetition_penalty,
                        vals * repetition_penalty,
                    )
                    logits[0, idx] = vals

            # No-repeat n-gram: block any candidate token that would complete
            # an n-gram already present earlier in the sequence.
            if no_repeat_ngram_size > 0 and ids.shape[1] >= no_repeat_ngram_size:
                n = no_repeat_ngram_size
                seq = ids[0].tolist()
                prefix = tuple(seq[-(n - 1):]) if n > 1 else ()
                banned: set[int] = set()
                for i in range(len(seq) - n + 1):
                    if tuple(seq[i : i + n - 1]) == prefix:
                        banned.add(seq[i + n - 1])
                banned -= exempt
                if banned:
                    bidx = torch.tensor(list(banned), device=logits.device, dtype=torch.long)
                    logits[0, bidx] = float("-inf")

            # Under n_bars nothing may end the sequence (the loop itself stops
            # once the bars are complete), and a bar just opened must get a
            # chord before the next [BAR]. Banned before top-k / top-p so the
            # filters pick from the tokens that remain.
            if n_bars is not None:
                logits[0, banned_ends] = float("-inf")
                if bar_chords == 0:
                    logits[0, bar_id] = float("-inf")

            # Top-k
            if top_k > 0:
                topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < topk_vals[:, -1:]] = float("-inf")

            # Top-p (nucleus)
            if 0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cum_probs = torch.cumsum(sorted_probs, dim=-1)
                remove = cum_probs - sorted_probs > top_p
                sorted_logits[remove] = float("-inf")
                logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1, generator=generator)
            ids = torch.cat([ids, next_id], dim=-1)

            if (next_id == eos_id).all():
                break

            if n_bars is not None:
                token = int(next_id[0, 0])
                if token == bar_id:
                    if bar_chords:  # this [BAR] closes a bar that holds chords
                        bars_done += 1
                    bar_chords = 0
                elif bar_chords is not None and token in chord_ids:
                    bar_chords += 1

        return ids
