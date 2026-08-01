"""Own-model candidates, physical roughness and optional theory retrieval.

R1 measures Sethares sensory roughness. R2 measures local TF-IDF textual
relevance to reviewed theory passages: OMT is enabled by default and Berklee
is disabled. Retrieval relevance is not a musical-quality score. Source notes
are attributed excerpts; this module never calls an explanation LLM.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

from app.models.schemas import ContextMeasure, Suggestion, TheoryExplanation
from app.models.schemas import MAX_SEED
from app.services.env_flags import env_bool
from app.services.key_anchor import anchor_to_key
from app.services.song_form import (
    apply_song_form_to_suggestions,
    resolves_to_tonic,
    rewrite_chord_sequence,
)
from app.services.voice_leading import apply_voice_leading_bass

log = logging.getLogger(__name__)
_HOSTED_INFERENCE_LOCK = threading.Lock()

_AI_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "ai"

# --- Guarded torch + model imports ---
TORCH_AVAILABLE = False
try:
    _AI_TRAINING_DIR = str(_AI_ROOT / "training")
    if _AI_TRAINING_DIR not in sys.path:
        sys.path.insert(0, _AI_TRAINING_DIR)

    import torch
    import torch.nn as nn
    from model import MusicTransformer  # type: ignore[import-untyped]
    from tokenizer import ChordTokenizer  # type: ignore[import-untyped]

    TORCH_AVAILABLE = True
except ImportError:
    log.warning("PyTorch or AI modules not available — AI inference disabled")

# --- Guarded peft (LoRA serving) ---
PEFT_AVAILABLE = False
try:
    from peft import PeftModel  # type: ignore[import-untyped]
    PEFT_AVAILABLE = True
except ImportError:
    log.warning("peft not available — LoRA adapters cannot be served, will fall back to F1 base")

# --- Guarded R1 physics rerank ---
PHYSICS_AVAILABLE = False
try:
    if str(_AI_ROOT) not in sys.path:
        sys.path.insert(0, str(_AI_ROOT))
    from physics.rerank import sequence_roughness  # type: ignore[import-untyped]

    PHYSICS_AVAILABLE = True
except ImportError:
    log.warning("ai/physics not available — Sethares rerank disabled, candidates served in temperature order")

# Product theory retrieval is local and source-isolated. Research Chroma/Gemma
# modules remain separate and are never imported by this runtime.
from app.services import theory_rag
RAG_AVAILABLE = True

# --- Checkpoint paths (paper experiment models) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_AI_CKPT = PROJECT_ROOT / "ai" / "checkpoints"

CHECKPOINTS = {
    # --- Paper experiment (F-series) ---
    "phase0":   _AI_CKPT / "phase0_pop_baseline" / "best.pt",  # Pop baseline
    "ft_f1":    _AI_CKPT / "ft_jazz_pop80" / "best.pt",        # Jazz + 10K pop mix (Pearl recommend ⭐)
    "ft_f1_v2": _AI_CKPT / "ft_jazz_pop80_v2" / "best.pt",     # F1 retrain, jazz_val-selected (2026-06-12)
    "ft_f2":    _AI_CKPT / "ft_jazz_pop67" / "best.pt",        # Jazz + 5K pop mix
    "ft_f3":    _AI_CKPT / "ft_jazz_pop50" / "best.pt",        # Jazz + 2.5K pop mix
    "ft_f4":    _AI_CKPT / "ft_jazz_pop29" / "best.pt",        # Jazz + 1K pop mix (Pearl recommend ⭐)
    "ft_f5":    _AI_CKPT / "ft_jazz_only" / "best.pt",         # Jazz only
    # --- LoRA per-genre adapters (R4, 2026-05-09 — F1 base + adapter) ---
    "ft_f1_lora_country":    _AI_CKPT / "ft_f1_lora_country",
    "ft_f1_lora_funk":       _AI_CKPT / "ft_f1_lora_funk",
    "ft_f1_lora_gospel":     _AI_CKPT / "ft_f1_lora_gospel",
    "ft_f1_lora_rnb_soul":   _AI_CKPT / "ft_f1_lora_rnb_soul",
    "ft_f1_lora_hip_hop":    _AI_CKPT / "ft_f1_lora_hip_hop",
    "ft_f1_lora_electronic": _AI_CKPT / "ft_f1_lora_electronic",
    "ft_f1_lora_folk":       _AI_CKPT / "ft_f1_lora_folk",
    "ft_f1_lora_classical":  _AI_CKPT / "ft_f1_lora_classical",
    "ft_f1_lora_rock":       _AI_CKPT / "ft_f1_lora_rock",
    "ft_f1_lora_blues":      _AI_CKPT / "ft_f1_lora_blues",
    "ft_f1_lora_bossa":      _AI_CKPT / "ft_f1_lora_bossa",
}

WARMUP_KEYS_VAR = "THEARTIST_WARMUP_KEYS"


def warmup_keys(env: Mapping[str, str] | None = None) -> tuple[list[str] | None, list[str]]:
    """(checkpoint keys named by THEARTIST_WARMUP_KEYS, names that are no checkpoint); keys None = every checkpoint."""
    raw = (os.environ if env is None else env).get(WARMUP_KEYS_VAR, "")
    names = list(dict.fromkeys(name.strip() for name in raw.split(",") if name.strip()))
    if not names:
        return None, []
    return [name for name in names if name in CHECKPOINTS], [name for name in names if name not in CHECKPOINTS]


# Human-readable labels for UI
MODEL_LABELS = {
    "phase0":   "Phase 0 — Pop Baseline",
    "ft_f1":    "F1 — 10K pop mix",
    "ft_f1_v2": "F1 v2 — 10K pop mix, jazz-val selected",
    "ft_f2":    "F2 — 5K pop mix",
    "ft_f3":    "F3 — 2.5K pop mix",
    "ft_f4":    "F4 — 1K pop mix",
    "ft_f5":    "F5 — Jazz only",
    "ft_f1_lora_country":    "F1 + LoRA: country",
    "ft_f1_lora_funk":       "F1 + LoRA: funk",
    "ft_f1_lora_gospel":     "F1 + LoRA: gospel",
    "ft_f1_lora_rnb_soul":   "F1 + LoRA: R&B/soul",
    "ft_f1_lora_hip_hop":    "F1 + LoRA: hip-hop",
    "ft_f1_lora_electronic": "F1 + LoRA: electronic",
    "ft_f1_lora_folk":       "F1 + LoRA: folk",
    "ft_f1_lora_classical":  "F1 + LoRA: classical",
    "ft_f1_lora_rock":       "F1 + LoRA: rock",
    "ft_f1_lora_blues":      "F1 + LoRA: blues",
    "ft_f1_lora_bossa":      "F1 + LoRA: bossa",
}

# Pearl 2026-05-09 — default = F1 (pop-preserving). F3 was previous default but
# Pearl prefers F1+F4 split based on listening; F3 is now just one option.
DEFAULT_MODEL = "ft_f1"

# Genre → modelKey auto-dispatch. Used when caller specifies `genre` but no
# explicit `modelKey`. jazz → F4 (Pearl ⭐, jazz-leaning), pop → F1 (pop-preserving);
# new genres routed to LoRA adapter if available, fallback to F1 base if adapter
# not yet trained.
GENRE_MODEL_DISPATCH = {
    # F-series base (no LoRA)
    "jazz":       "ft_f4",
    "pop":        "ft_f1",
    # LoRA adapters (R4)
    "country":    "ft_f1_lora_country",
    "funk":       "ft_f1_lora_funk",
    "gospel":     "ft_f1_lora_gospel",
    "rnb_soul":   "ft_f1_lora_rnb_soul",
    "hip_hop":    "ft_f1_lora_hip_hop",
    "electronic": "ft_f1_lora_electronic",
    "folk":       "ft_f1_lora_folk",
    "classical":  "ft_f1_lora_classical",
    "rock":       "ft_f1_lora_rock",
    "blues":      "ft_f1_lora_blues",
    "bossa":      "ft_f1_lora_bossa",
}

def dispatch_model_for_genre(genre: str | None) -> str:
    """Map genre to served checkpoint key, with F1 base fallback if LoRA not ready."""
    if not genre:
        return DEFAULT_MODEL
    target = GENRE_MODEL_DISPATCH.get(genre.lower())
    if target is None:
        return DEFAULT_MODEL
    # If LoRA adapter directory doesn't exist yet, fall back to F1 base.
    path = CHECKPOINTS.get(target)
    if path is None or (target.startswith("ft_f1_lora_") and not (path / "adapter").exists()):
        return "ft_f1"
    return target

def rag_disabled() -> bool:
    """Global rollback or both source switches off; OMT is on by default."""
    return not theory_rag.policy().sources


def _should_use_rag(genre: str | None) -> bool:
    """Only report a theory stage when an enabled corpus is actually ready."""
    return _get_theory_retriever() is not None


# Genre aliases for the always-served 359-token tokenizer. Base checkpoints
# expand from 351 tokens with new genre rows initialized from [GENRE:none];
# LoRA embedding overlays supply learned genre rows. Unknown genres use none.
_GENRE_MAP = {
    "bossa nova": "bossa",
    "bossa": "bossa",
    "jazz": "jazz",
    "pop": "pop",
    "rock": "rock",
    "blues": "blues",
    # Extended-vocab genres (R4 LoRA, 2026-05-09). The extended tokenizer
    # (always served — see ModelRegistry.tokenizer) has [GENRE:<name>] tokens
    # for these. For untrained genres (LoRA not yet fitted), the embedding
    # row is initialized from [GENRE:none] so the prompt still encodes a
    # known token but the model behaves unconditioned. Once a LoRA adapter
    # lands, that genre's generation becomes properly conditioned.
    "classical": "classical",
    "country": "country",
    "rnb_soul": "rnb_soul",
    "hip_hop": "hip_hop",
    "electronic": "electronic",
    "funk": "funk",
    "folk": "folk",
    "gospel": "gospel",
}


@dataclass
class LoadedModel:
    model: object  # MusicTransformer
    name: str
    device: object  # torch.device


class ModelRegistry:
    """Singleton managing lazy-loaded model instances."""

    _instance: ModelRegistry | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._models: dict[str, LoadedModel] = {}
        self._tokenizer: object | None = None  # ChordTokenizer
        self._load_lock = threading.RLock()  # A missing adapter may resolve to F1 under this lock.
        self._f1_checkpoint: dict | None = None

    @classmethod
    def get(cls) -> ModelRegistry:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @property
    def tokenizer(self) -> ChordTokenizer:
        """Always extended tokenizer (vocab 359, 13 genre tokens). F-series
        models trained with vocab=351 are loaded then resized at load time;
        new genre rows initialize from the [GENRE:none] row so unknown
        genres degrade gracefully to unconditioned generation.
        LoRA models additionally load embedding_extension.pt + apply
        the peft adapter."""
        if self._tokenizer is None:
            self._tokenizer = ChordTokenizer(include_extra_genres=True)
        return self._tokenizer  # type: ignore[return-value]

    def _trained_vocab_size(self) -> int:
        """Vocab size F-series checkpoints were trained at (351 — without
        EXTRA_GENRES). Used to (a) build the model with the original
        vocab so load_state_dict succeeds, then (b) resize to extended
        vocab for unified serving."""
        return ChordTokenizer(include_extra_genres=False).vocab_size

    def _resize_embedding_inplace(
        self,
        model: "MusicTransformer",
        old_vocab: int,
        new_vocab: int,
        device: "torch.device",
    ) -> None:
        """Grow model.token_emb + model.out_proj from old_vocab → new_vocab.
        New rows are initialized from the [GENRE:none] row so requests for
        not-yet-trained genres degrade to unconditioned generation. Mirrors
        ai/training/lora_train.py:_resize_embedding_with_init."""
        if new_vocab <= old_vocab:
            return
        none_id = ChordTokenizer(include_extra_genres=False).encode_genre("none")
        d_model = model.token_emb.embedding_dim
        pad_id = model.token_emb.padding_idx

        new_emb = nn.Embedding(new_vocab, d_model, padding_idx=pad_id, device="meta").to_empty(device=device)
        with torch.no_grad():
            new_emb.weight[:old_vocab] = model.token_emb.weight
            none_row = model.token_emb.weight[none_id]
            for i in range(old_vocab, new_vocab):
                new_emb.weight[i] = none_row
        model.token_emb = new_emb

        new_out = nn.Linear(d_model, new_vocab, bias=False, device="meta").to_empty(device=device)
        with torch.no_grad():
            new_out.weight[:old_vocab] = model.out_proj.weight
            none_row = model.out_proj.weight[none_id]
            for i in range(old_vocab, new_vocab):
                new_out.weight[i] = none_row
        model.out_proj = new_out

    def _build_base_model(
        self,
        ckpt: dict,
        config: dict,
        device: "torch.device",
    ) -> "MusicTransformer":
        """Construct a MusicTransformer with the trained vocab (351), load
        the checkpoint state, then resize the embedding to extended vocab
        (359). Used for both F-series and LoRA serving paths."""
        trained_vocab = self._trained_vocab_size()
        extended_vocab = self.tokenizer.vocab_size
        # Every parameter comes from the checkpoint. Allocating and randomly
        # initializing 25M values only to overwrite them is wasted startup work.
        with torch.device("meta"):
            model = MusicTransformer(
                vocab_size=trained_vocab,
                d_model=config.get("d_model", 512),
                n_heads=config.get("n_heads", 8),
                d_ff=config.get("d_ff", 2048),
                n_layers=config.get("n_layers", 8),
                max_seq_len=config.get("max_seq_len", 256),
                dropout=0.0,
                pad_id=self.tokenizer.pad_id,
            )
        model.load_state_dict(ckpt["model_state_dict"], assign=True)
        model.to(device)
        self._resize_embedding_inplace(model, trained_vocab, extended_vocab, device)
        return model

    def _read_checkpoint(self, name: str, device: "torch.device") -> dict:
        """Map released weights; retain only F1's inference state for its adapters.

        Each LoRA has independent modules/embeddings/adapters but shares immutable
        F1 base tensor storage. Never merge adapters or train these served models.
        Training/optimizer tensors are deliberately not retained in this cache.
        """
        if name == "ft_f1" and self._f1_checkpoint is not None:
            return self._f1_checkpoint
        ckpt = torch.load(CHECKPOINTS[name], map_location=device, weights_only=False, mmap=True)
        result = {"model_state_dict": ckpt["model_state_dict"], "config": ckpt.get("config", {})}
        if name == "ft_f1":
            self._f1_checkpoint = result
        return result

    def _load_lora(self, name: str, lora_dir: Path) -> LoadedModel | None:
        """Serve a LoRA model: F1 base + embedding_extension overlay + peft adapter.
        Falls back to F1 base if peft is unavailable or the adapter directory
        is missing (matches dispatch_model_for_genre fallback semantics)."""
        adapter_dir = lora_dir / "adapter"
        embedding_ext = lora_dir / "embedding_extension.pt"
        if not adapter_dir.exists():
            log.info(f"LoRA adapter not found at {adapter_dir} — falling back to ft_f1")
            return self.load_model("ft_f1")
        if not PEFT_AVAILABLE:
            log.warning(f"peft missing — cannot serve LoRA '{name}', falling back to ft_f1")
            return self.load_model("ft_f1")

        f1_path = CHECKPOINTS["ft_f1"]
        if not f1_path.exists():
            log.warning(f"F1 base not found at {f1_path} — cannot serve LoRA '{name}'")
            return None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = self._read_checkpoint("ft_f1", device)
        config = ckpt.get("config", {})

        # Step 1+2: F1 base into a fresh model + resize to extended vocab
        # (new rows init from [GENRE:none] before being overwritten in step 3)
        base_model = self._build_base_model(ckpt, config, device)

        # Step 3: Override the resized embedding with the LoRA-trained
        # embedding_extension overlay (contains the learned new-genre rows)
        if embedding_ext.exists():
            ext = torch.load(embedding_ext, map_location=device, weights_only=False)
            base_model.token_emb.load_state_dict(ext["token_emb_state"])
            base_model.out_proj.load_state_dict(ext["out_proj_state"])
            log.info(f"Loaded embedding extension for '{name}' from {embedding_ext.name}")

        # Step 4: Apply peft LoRA adapter
        peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
        peft_model.eval().requires_grad_(False)
        loaded = LoadedModel(model=peft_model, name=name, device=device)
        self._models[name] = loaded
        log.info(f"Loaded LoRA '{name}' on {device} (F1 base + adapter)")
        return loaded

    def load_model(self, name: str) -> LoadedModel | None:
        if not TORCH_AVAILABLE:
            return None
        if name in self._models:
            return self._models[name]
        with self._load_lock:
            if name in self._models:
                return self._models[name]
            path = CHECKPOINTS.get(name)
            if path is None:
                log.warning(f"Unknown checkpoint key: {name}")
                return None
            # LoRA path — F1 base + adapter
            if name.startswith("ft_f1_lora_"):
                return self._load_lora(name, path)
            # F-series path
            if not path.exists():
                log.warning(f"Checkpoint not found: {path}")
                return None
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ckpt = self._read_checkpoint(name, device)
            config = ckpt.get("config", {})
            model = self._build_base_model(ckpt, config, device)
            model.eval().requires_grad_(False)
            loaded = LoadedModel(model=model, name=name, device=device)
            self._models[name] = loaded
            params = sum(p.numel() for p in model.parameters()) / 1e6
            log.info(f"Loaded model '{name}' on {device} ({params:.1f}M params)")
            return loaded

    def warmup(self, keys: Iterable[str] | None = None) -> dict[str, bool]:
        """Load every checkpoint, or only `keys`; a key without a checkpoint is False, never fatal."""
        results = {}
        for name in (CHECKPOINTS if keys is None else keys):
            results[name] = self.load_model(name) is not None
        return results

    def is_loaded(self, name: str) -> bool:
        return name in self._models

    @property
    def available(self) -> bool:
        return TORCH_AVAILABLE


def build_prompt(
    tokenizer: ChordTokenizer,
    key: str,
    genre: str | None,
    time_signature: tuple[int, int],
    context: list[ContextMeasure],
    selected_measures: list[int],
    bpm: int | None = None,
) -> list[int]:
    """Build token prompt from API request context.

    `bpm` is accepted but currently not encoded into the prompt — the
    trained tokenizer has no BPM-bucket tokens. It is logged for
    traceability and is reserved for a future training round.
    """
    parts = key.strip().split()
    key_token = parts[0] + ("maj" if parts[1].lower() == "major" else "min")
    ts_str = f"{time_signature[0]}/{time_signature[1]}"
    genre_str = _GENRE_MAP.get(genre, "none") if genre else "none"
    if bpm is not None:
        log.debug("build_prompt: bpm=%d (not encoded; reserved)", bpm)

    ids: list[int] = [tokenizer.bos_id]

    kid = tokenizer.encode_key(key_token)
    if kid is not None:
        ids.append(kid)
    tid = tokenizer.encode_time_sig(ts_str)
    if tid is not None:
        ids.append(tid)
    gid = tokenizer.encode_genre(genre_str)
    if gid is not None:
        ids.append(gid)

    # Add context bars before the first selected measure
    first_selected = min(selected_measures) if selected_measures else 0
    for cm in sorted(context, key=lambda c: c.measure):
        if cm.measure >= first_selected:
            break
        ids.append(tokenizer.bar_id)
        for chord in cm.chords:
            if chord is not None:
                cid = tokenizer.encode_chord(chord)
                if cid is not None:
                    ids.append(cid)

    return ids


def generate_with_model(
    loaded: LoadedModel,
    tokenizer: ChordTokenizer,
    prompt_ids: list[int],
    n_bars: int,
    temperature: float = 0.8,
    top_k: int = 20,
    top_p: float = 0.9,
    repetition_penalty: float = 1.3,
    no_repeat_ngram_size: int = 3,
    seed: int | None = None,
) -> list[list[str]]:
    """Run model inference, return list of bars (each bar = list of chord strings).

    The pass is held to ``n_bars`` (see MusicTransformer.generate): [EOS],
    [PAD] and empty bars are banned until that many bars are complete, and
    sampling stops once they are. It still comes back short when it runs out
    of token budget first.

    A seed owns a separate CPU/CUDA Generator for this pass. No global
    torch, Python or NumPy state is reseeded, so concurrent requests do not
    consume one another's seeded stream. Reproducibility is scoped to the
    same checkpoint, runtime, device and sampling parameters.
    """
    device = loaded.device
    generator = None
    if seed is not None:
        if type(seed) is not int or not 0 <= seed <= MAX_SEED:
            raise ValueError(f"seed must be an integer from 0 to {MAX_SEED}")
        generator = torch.Generator(device=device).manual_seed(seed)
    prompt = torch.tensor([prompt_ids], device=device)
    # BAR + 2 chords per bar, plus slack so a few extra chords or other
    # non-chord tokens cannot cut the section short
    max_new = n_bars * 3 + max(8, n_bars // 2)

    # Structural separators MUST be allowed to recur; exempt them from
    # repetition / no-repeat-ngram controls.
    exempt = {tokenizer.bar_id}
    if hasattr(tokenizer, "pad_id"):
        exempt.add(tokenizer.pad_id)

    # Every token that is not a bracketed special one ([BAR], [EOS], [KEY:...]) is a chord.
    chord_ids = {i for tok, i in tokenizer.token2id.items() if not tok.startswith("[")}

    # All REST/song/MCP callers share this function and this model process.
    # On the small public host, only one neural pass allocates its KV cache at a time.
    with _HOSTED_INFERENCE_LOCK if env_bool("THEARTIST_PUBLIC_HOST", False) else nullcontext():
        output = loaded.model.generate(
            prompt,
            max_new_tokens=max_new,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_id=tokenizer.eos_id,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            ignore_repeat_token_ids=exempt,
            n_bars=n_bars,
            bar_id=tokenizer.bar_id,
            chord_ids=chord_ids,
            end_ids={tokenizer.pad_id},
            generator=generator,
            use_cache=not env_bool("THEARTIST_DISABLE_KV_CACHE", False),
        )

    gen_ids = output[0, len(prompt_ids):].tolist()
    gen_tokens = tokenizer.decode(gen_ids)

    # Parse tokens into bars, counted the way generate() counts them: chords
    # before the first [BAR] continue the prompt's last bar, not a new one.
    bars: list[list[str]] = []
    current_bar: list[str] | None = None
    for tok in gen_tokens:
        if tok == "[BAR]":
            if current_bar:
                bars.append(current_bar)
            current_bar = []
        elif tok in ("[EOS]", "[PAD]"):
            break
        elif current_bar is not None and not tok.startswith("["):
            current_bar.append(tok)
    if current_bar:
        bars.append(current_bar)

    return bars[:n_bars]


def _bars_to_chord_map(
    bars: list[list[str]],
    selected_measures: list[int],
    chords_per_bar: int = 2,
) -> dict[str, list[str]]:
    """Convert model output bars to Suggestion.chords format.

    A pass normally writes one bar per selected measure (generate_with_model
    holds it to the count). One that ran out of token budget first is
    padded: the bars it did not write repeat its own bars from the start,
    so the padding stays in the progression's key.
    """
    chord_map: dict[str, list[str]] = {}
    for i, m_idx in enumerate(selected_measures):
        bar_chords = bars[i % len(bars)][:chords_per_bar] if bars else []
        while len(bar_chords) < chords_per_bar:
            bar_chords.append(bar_chords[-1] if bar_chords else "Cmaj")
        chord_map[str(m_idx)] = bar_chords
    return chord_map


def _get_theory_retriever():
    if not RAG_AVAILABLE or rag_disabled():
        return None
    return theory_rag.runtime()[0]


def _explain_one(a: str, b: str, genre: str | None = None) -> TheoryExplanation | None:
    retriever = _get_theory_retriever()
    return retriever.explain(a, b, genre) if retriever is not None else None


def _attach_explanations(
    suggestions: list[Suggestion], selected_measures: list[int], genre: str | None = None,
) -> list[Suggestion]:
    """Attach at most 16 distinct, attributed source excerpts per suggestion.

    No generated prose, external inference or per-transition worker pool. The
    active retriever owns its bounded cache, so changing sources cannot reuse a
    disabled book's hits. Repeated transitions share the same lookup.
    """
    if env_bool("THEARTIST_DISABLE_EXPLANATIONS", False):
        return suggestions
    retriever = _get_theory_retriever()
    if retriever is None:
        return suggestions
    for suggestion in suggestions:
        sequence = _flatten_suggestion(suggestion)
        pairs = dict.fromkeys(zip(sequence, sequence[1:]))
        notes = [retriever.explain(a, b, genre) for a, b in list(pairs)[:16]]
        suggestion.explanations = [note for note in notes if note is not None] or None
    return suggestions


def _flatten_suggestion(s: Suggestion) -> list[str]:
    """Flatten a suggestion's per-measure chord lists into one ordered sequence."""
    out: list[str] = []
    for m_str in sorted(s.chords.keys(), key=int):
        out.extend(s.chords[m_str])
    return out


def _theory_distance(a: str, b: str, genre: str | None = None) -> float | None:
    """Textual retrieval distance from enabled sources, not harmonic correctness."""
    retriever = _get_theory_retriever()
    if retriever is None:
        return None
    hit = retriever.transition(a, b, genre)
    # No coverage must not earn a better score than a covered transition.
    return hit.distance if hit is not None else 1.0


def _rerank_composite(
    suggestions: list[Suggestion],
    alpha: float | None = None,
    beta: float | None = None,
    genre: str | None = None,
) -> list[Suggestion]:
    """Sort by α·R1 (Sethares) + β·R2 (enabled-source textual retrieval).

    Both axes are lower-is-better (R1 = mean roughness; R2 = mean retrieval
    distance), so the composite is a weighted sum and the lowest score wins.
    Passes through unchanged if both axes are unavailable. When only one
    axis is available, that axis effectively decides the order.

    Re-letters Option A/B/C by the new rank. Logs the per-suggestion
    R1 / R2 / composite breakdown so quality regressions are diagnosable
    from the server log.

    OMT is enabled by default; Berklee requires an explicit private-research
    source switch. THEARTIST_DISABLE_RAG=1 disables both sources;
    THEARTIST_DISABLE_R2_RERANK=1 disables ranking only, and
    THEARTIST_DISABLE_EXPLANATIONS=1 disables source notes only.
    THEARTIST_RERANK_ALPHA / THEARTIST_RERANK_BETA control the two weights.
    """
    if len(suggestions) <= 1:
        return suggestions

    # Resolve weights (CLI args > env > defaults).
    if alpha is None:
        try:
            alpha = float(os.getenv("THEARTIST_RERANK_ALPHA", "1.0"))
        except ValueError:
            alpha = 1.0
    if beta is None:
        try:
            beta = float(os.getenv("THEARTIST_RERANK_BETA", "1.0"))
        except ValueError:
            beta = 1.0

    # Only score R2 when a selected source has actually loaded.
    r2_disabled = (
        env_bool("THEARTIST_DISABLE_R2_RERANK", False)
        or not _should_use_rag(genre)
    )

    def _r1(flat: list[str]) -> float:
        if not PHYSICS_AVAILABLE or len(flat) < 2:
            return 0.0
        total, n = sequence_roughness(flat)
        return total / n if n > 0 else 0.0

    def _r2(flat: list[str]) -> float:
        if r2_disabled or not RAG_AVAILABLE or len(flat) < 2:
            return 0.0
        ds = [d for d in (_theory_distance(a, b, genre) for a, b in zip(flat, flat[1:])) if d is not None]
        return sum(ds) / len(ds) if ds else 0.0

    breakdown: list[tuple[float, float, float, Suggestion]] = []
    for s in suggestions:
        flat = _flatten_suggestion(s)
        r1_val = _r1(flat)
        r2_val = _r2(flat)
        composite = alpha * r1_val + beta * r2_val
        breakdown.append((composite, r1_val, r2_val, s))

    breakdown.sort(key=lambda x: x[0])

    # One log line per /generate call so quality drift is visible.
    log.info(
        "rerank α=%.2f β=%.2f r2=%s | "
        + " | ".join(
            f"{chr(ord('A') + i)}: comp={c:.3f} r1={r1:.3f} r2={r2:.3f}"
            for i, (c, r1, r2, _s) in enumerate(breakdown)
        ),
        alpha,
        beta,
        "off" if r2_disabled else "on",
    )

    return _reletter([s for _comp, _r1v, _r2v, s in breakdown])


def _reletter(suggestions: list[Suggestion]) -> list[Suggestion]:
    """Option A/B/C follow list order after any reordering; the creativity body stays."""
    for i, s in enumerate(suggestions):
        body = s.label.split(" — ", 1)[1] if " — " in s.label else s.label
        s.label = f"Option {chr(ord('A') + i)} — {body}"
    return suggestions


# ─── Pipeline stages ───────────────────────────────────────────────────
# `generate_ai_result` orchestrates these in order (`generate_ai_suggestions`
# wraps it). Each stage owns its own phase-event emission, so the
# orchestrator just chains calls.

# Type alias for clarity at call sites; kept as a string forward-ref
# so we don't need to add a `typing.Callable` import for it.
PhaseCallback = "Callable[[str, dict], None]"
_NOOP_NOTIFY = lambda *_a, **_kw: None  # noqa: E731 (intentional one-liner)


def _resolve_model(model_key: str | None, on_phase, genre: str | None = None) -> "tuple[LoadedModel, str] | None":
    """Stage 1 — pick the requested checkpoint and warm-load it.

    Resolution order:
      1. If `model_key` given and valid → use it directly (advanced/research path).
      2. If `genre` given → dispatch via GENRE_MODEL_DISPATCH (production path).
      3. Otherwise → DEFAULT_MODEL.

    LoRA modelKeys (`ft_f1_lora_<genre>`) gracefully fall back to ft_f1 when
    the adapter is not yet trained on disk (handled by dispatch_model_for_genre).

    Emits `model_load` on a cold load. Returns (LoadedModel, label) or None
    if torch / the checkpoint are unavailable.
    """
    if not TORCH_AVAILABLE:
        return None
    if model_key and model_key in CHECKPOINTS:
        key_to_use = model_key
    elif genre:
        key_to_use = dispatch_model_for_genre(genre)
    else:
        key_to_use = DEFAULT_MODEL
    # LoRA serving (peft + embedding extension) wired into
    # ModelRegistry._load_lora. dispatch_model_for_genre already falls
    # back to "ft_f1" when an adapter dir is missing; nothing to do here.
    registry = ModelRegistry.get()
    if not registry.is_loaded(key_to_use):
        on_phase("model_load", {"label": f"Loading {MODEL_LABELS.get(key_to_use, key_to_use)}"})
    loaded = registry.load_model(key_to_use)
    if loaded is None:
        return None
    return loaded, MODEL_LABELS.get(key_to_use, key_to_use)


# Three temperatures, three "creativity" labels, one model — produces
# the A / B / C variation set the user sees before rerank.
_TEMPS: list[tuple[float, str]] = [
    (0.7, "conservative"),
    (0.9, "balanced"),
    (1.1, "creative"),
]

# The causal model writes from the bars before a passage only. When the score
# already holds chords after it, each temperature draws this many drafts and
# keeps the one that leads best into those chords (chord_context.continuation_scores),
# so the written future shapes the offered chords, not just their order.
CONTEXT_DRAWS_VAR = "THEARTIST_CONTEXT_DRAWS"
DEFAULT_CONTEXT_DRAWS = 4
MAX_CONTEXT_DRAWS = 8
# Drafts x selected bars per temperature stay within the longest single pass the
# editor allows (64 bars), so drafting never makes a request costlier than the
# longest one-pass request: 4 drafts up to 16 bars, 2 at 32, 1 at 64.
CONTEXT_DRAFT_BARS = 64


def context_draws(env: Mapping[str, str] | None = None) -> int:
    """Drafts per temperature for a passage with written chords after it:
    THEARTIST_CONTEXT_DRAWS (1 to 8), else 4. An unreadable value keeps 4."""
    raw = ((os.environ if env is None else env).get(CONTEXT_DRAWS_VAR) or "").strip()
    if not raw:
        return DEFAULT_CONTEXT_DRAWS
    try:
        value = int(raw)
    except ValueError:
        log.warning("%s=%r is not a whole number; using %d", CONTEXT_DRAWS_VAR, raw, DEFAULT_CONTEXT_DRAWS)
        return DEFAULT_CONTEXT_DRAWS
    return min(max(value, 1), MAX_CONTEXT_DRAWS)


def _draw_seed(seed: int, step: int, draw: int) -> int:
    """The seed of one draft. The first draft of each temperature keeps the seed a
    single pass had (seed + step - 1), so requests with one draft are unchanged."""
    return (seed + step - 1 + draw * len(_TEMPS)) & MAX_SEED


def _compose_candidates(
    loaded: "LoadedModel",
    model_label: str,
    prompt_ids: "torch.Tensor",
    n_bars: int,
    selected_measures: list[int],
    on_phase,
    seed: int | None = None,
    draws: int = 1,
    pick=None,
) -> list[Suggestion]:
    """Stage 2 — run the model at the three configured temperatures.
    Emits `composing` once per temperature with step / total / creativity
    in the payload. Inference failures (and passes that end before any
    chord) are logged and skipped, so a flaky pass at one temperature
    doesn't kill the whole request. A pass that comes back short of the
    selection is logged and padded with its own bars.

    With ``draws`` > 1 each temperature writes that many drafts and
    ``pick(drafts)`` returns the index of the one to keep (the drafts are
    chord maps in written order); without ``pick`` the first draft is kept."""
    registry = ModelRegistry.get()
    tokenizer = registry.tokenizer
    suggestions: list[Suggestion] = []
    for step, (temp, creativity) in enumerate(_TEMPS, start=1):
        label = f"Composing variation {step}/{len(_TEMPS)} ({creativity})"
        if draws > 1:
            label += f": {draws} drafts, keeping the one that leads best into the written bars after it"
        on_phase("composing", {
            "label": label,
            "step": step,
            "total": len(_TEMPS),
            "creativity": creativity,
        })
        drafts: list[dict[str, list[str]]] = []
        for draw in range(draws):
            try:
                seed_options = {} if seed is None else {"seed": _draw_seed(seed, step, draw)}
                bars = generate_with_model(loaded, tokenizer, prompt_ids, n_bars, temperature=temp, **seed_options)
                if not bars:
                    log.warning(f"Model ended before any chord: temp={temp}")
                    continue
                if len(bars) < n_bars:
                    log.warning(
                        f"Model wrote {len(bars)} of {n_bars} bars, repeating its own bars "
                        f"for the rest: temp={temp}"
                    )
                drafts.append(_bars_to_chord_map(bars, selected_measures))
            except Exception:
                log.exception(f"Inference failed: temp={temp}")
        if not drafts:
            continue
        chosen = 0
        if pick is not None and len(drafts) > 1:
            try:
                chosen = pick(drafts)
            except ValueError as exc:
                log.warning("Kept the first draft at temp=%s: %s", temp, exc)
                on_phase("composing", {
                    "label": f"Composing variation {step}/{len(_TEMPS)} ({creativity}): "
                             f"kept the first draft, the drafts could not be compared ({exc})",
                    "step": step,
                    "total": len(_TEMPS),
                    "creativity": creativity,
                })
        letter = chr(ord("A") + len(suggestions))
        suggestions.append(Suggestion(
            label=f"Option {letter} — {model_label} ({creativity})",
            chords=drafts[chosen],
        ))
    return suggestions


def _score_and_rerank(
    suggestions: list[Suggestion], on_phase, genre: str | None = None,
) -> list[Suggestion]:
    """Stage 3 — composite rerank (R1 Sethares + enabled-source R2 retrieval).
    Uses ready theory sources, or R1 alone when none are ready. Emits
    `ranking` and reassigns A/B/C labels by score."""
    label = "Scoring consonance and theory relevance" if _should_use_rag(genre) else "Scoring consonance"
    on_phase("ranking", {"label": label})
    return _rerank_composite(suggestions, genre=genre)


def _final_chords(suggestions: list[Suggestion]) -> list[tuple[Suggestion, str, str]]:
    """(suggestion, last measure, that measure's last chord) for every
    suggestion whose last measure holds a chord."""
    finals: list[tuple[Suggestion, str, str]] = []
    for s in suggestions:
        if not s.chords:
            continue
        last = max(s.chords, key=int)
        if s.chords[last]:
            finals.append((s, last, s.chords[last][-1]))
    return finals


def _inject_voice_leading(
    suggestions: list[Suggestion], on_phase, section_type: str | None = None,
    cadence: str | None = None,
) -> list[Suggestion]:
    """Stage 4 — pick inversions for smooth bass motion. Emits `voicing`
    unless disabled by env var. When the section's rule (or the planner's
    cadence) resolves to the tonic (outro, tag, verse, chorus, ...) that
    final tonic keeps its root position."""
    if os.getenv("THEARTIST_DISABLE_VOICE_LEADING"):
        return suggestions
    on_phase("voicing", {"label": "Picking inversions for smooth bass motion"})
    resolving_tonics = _final_chords(suggestions) if resolves_to_tonic(section_type, cadence) else []
    suggestions = apply_voice_leading_bass(suggestions)
    for s, last, tonic in resolving_tonics:
        s.chords[last][-1] = tonic
    return suggestions


def _explain_each_transition(
    suggestions: list[Suggestion],
    selected_measures: list[int],
    on_phase,
    genre: str | None = None,
) -> list[Suggestion]:
    """Stage 5 — related theory excerpts from enabled, ready sources."""
    if not _should_use_rag(genre) or env_bool("THEARTIST_DISABLE_EXPLANATIONS", False):
        return suggestions
    on_phase("explaining", {"label": "Retrieving related theory references"})
    return _attach_explanations(suggestions, selected_measures, genre)


def _has_chords_before(context: list[ContextMeasure], selected_measures: list[int]) -> bool:
    """Does any measure before the selection hold a chord for the model to continue from?"""
    first_selected = min(selected_measures) if selected_measures else 0
    return any(
        chord and chord.strip()
        for cm in context if cm.measure < first_selected
        for chord in cm.chords
    )


def _has_chords_after(context: list[ContextMeasure], selected_measures: list[int]) -> bool:
    """Does a measure after the passage's start, outside the selection, hold a written chord?"""
    if not selected_measures:
        return False
    first = min(selected_measures)
    chosen = set(selected_measures)
    return any(
        chord and chord.strip()
        for cm in context if cm.measure > first and cm.measure not in chosen
        for chord in cm.chords
    )


def _drafts_for(loaded: "LoadedModel", tokenizer, header: list[int], context: list[ContextMeasure],
                selected_measures: list[int]) -> int:
    """Drafts per temperature: context_draws() within the CONTEXT_DRAFT_BARS budget, and 1
    when the written bars after the passage cannot be scored at all (a chord outside the
    vocabulary, a score too long for the window, no written chord inside the window).
    Every draft writes two chords a bar, so a placeholder of that shape fits exactly as
    the drafts will; the check reads lengths and tokens only and runs no model pass.
    The same conditions make the closing whole-score comparison report that it was
    skipped, so the user sees the reason there."""
    from app.services.chord_context import fitting_context, sequence_tokens

    draws = min(context_draws(), max(1, CONTEXT_DRAFT_BARS // max(1, len(selected_measures))))
    if draws <= 1:
        return 1
    # Any chord token of the vocabulary: every chord is one token, so content does not matter.
    chord = next(token for token in tokenizer.token2id if not token.startswith("["))
    probe = Suggestion.model_construct(label="probe", chords={str(m): [chord, chord] for m in selected_measures})
    try:
        bars, _ = fitting_context(loaded, tokenizer, [probe], header, context, selected_measures)
        sequence_tokens(tokenizer, header, bars, probe, selected_measures, known_only=True)
    except ValueError as exc:
        log.warning("One draft per temperature: %s", exc)
        return 1
    return draws


def _anchor_candidates(suggestions: list[Suggestion], key_root: str, is_minor: bool) -> None:
    """Transpose each candidate into the requested key (see key_anchor) in place."""
    for s in suggestions:
        rewrite_chord_sequence(s, lambda chords: anchor_to_key(chords, key_root, is_minor))


def _split_key(key: str) -> tuple[str, bool]:
    """'A minor' -> ('A', True): the key's root and whether it is minor."""
    parts = key.strip().split()
    return parts[0], len(parts) > 1 and parts[1].lower().startswith("min")


@dataclass
class AIResult:
    suggestions: list[Suggestion]
    model_key: str  # checkpoint that actually produced the chords


def generate_ai_result(
    key: str,
    genre: str | None,
    context: list[ContextMeasure],
    selected_measures: list[int],
    time_signature: tuple[int, int] = (4, 4),
    model_key: str | None = None,
    bpm: int | None = None,
    on_phase=None,  # type: ignore[assignment]
    section_type: str | None = None,
    cadence: str | None = None,
    seed: int | None = None,
) -> AIResult | None:
    """Generate 3 suggestions, then re-rank by sensory consonance + theory.

    Pipeline stages (each owns its phase event):
        1. _resolve_model            → "model_load" (cold only)
        2. _compose_candidates       → "composing" ×3
           (with written chords after the selection, each temperature draws
           context_draws() drafts and keeps the one that leads best into
           them, judged anchored and with the song-form rules applied)
           _anchor_candidates          (only when no chord precedes the selection)
        3. _score_and_rerank         → "ranking"
           song-form rules for `section_type` (None = no rules), ending
           with `cadence` when the planner chose one
        4. _inject_voice_leading     → "voicing" (unless disabled)
        5. _explain_each_transition  → "explaining" (when RAG enabled)

    Song form runs before voice leading and explanations, so slash bass
    notes and theory notes describe the final chords.

    `on_phase(stage, payload)` is called between stages so streaming
    endpoints can push live progress to the client. No-op when omitted.

    Returns None if torch / the chosen checkpoint are unavailable
    (caller should fall back to rule-based generation).
    """
    notify = on_phase if on_phase is not None else _NOOP_NOTIFY

    resolved = _resolve_model(model_key, notify, genre=genre)
    if resolved is None:
        return None
    loaded, model_label = resolved

    tokenizer = ModelRegistry.get().tokenizer
    prompt_ids = build_prompt(tokenizer, key, genre, time_signature, context, selected_measures, bpm=bpm)
    n_bars = len(selected_measures)
    key_root, is_minor = _split_key(key)
    # With no chords to continue from the model ignores the KEY token.
    anchor = not _has_chords_before(context, selected_measures)

    # The prompt holds only the bars before the passage. Written bars after it
    # choose among each temperature's drafts, judged as they will be offered:
    # key-anchored when nothing precedes, with the section's song-form rules.
    draws, pick = 1, None
    written_after = _has_chords_after(context, selected_measures)
    if written_after:
        header = build_prompt(tokenizer, key, genre, time_signature, [], [], bpm=bpm)
        draws = _drafts_for(loaded, tokenizer, header, context, selected_measures)
        if draws > 1:
            from app.services.chord_context import continuation_scores

            def pick(drafts: list[dict[str, list[str]]]) -> int:
                trial = [Suggestion.model_construct(label=str(i), chords={m: list(c) for m, c in draft.items()})
                         for i, draft in enumerate(drafts)]
                if anchor:
                    _anchor_candidates(trial, key_root, is_minor)
                apply_song_form_to_suggestions(trial, section_type, key_root, is_minor, cadence)
                scores = continuation_scores(loaded, tokenizer, trial, header, context, selected_measures)
                return max(range(len(scores)), key=scores.__getitem__)

    suggestions = _compose_candidates(
        loaded, model_label, prompt_ids, n_bars, selected_measures, notify, seed=seed,
        draws=draws, pick=pick,
    )
    if not suggestions:
        return None

    if anchor:
        _anchor_candidates(suggestions, key_root, is_minor)

    suggestions = _score_and_rerank(suggestions, notify, genre=genre)
    apply_song_form_to_suggestions(suggestions, section_type, key_root, is_minor, cadence)
    suggestions = _inject_voice_leading(suggestions, notify, section_type=section_type, cadence=cadence)
    # A causal continuation prompt cannot put future bars before the hole.
    # Instead score each completed candidate in its real whole-score order,
    # including the likelihood of the already fixed chords after the hole.
    if written_after:
        from app.services.chord_context import compare_with_context
        try:
            header = build_prompt(tokenizer, key, genre, time_signature, [], [], bpm=bpm)
            suggestions, reach = compare_with_context(loaded, tokenizer, suggestions, header, context, selected_measures)
            # The list order is the ranking the editor shows, so the letters follow it.
            suggestions = _reletter(suggestions)
            notify("ranking", {"label": "Compared candidates with preceding and following chords" if reach is None
                               else f"Compared candidates with the {reach} bars before and after the passage"})
        except ValueError as exc:
            notify("ranking", {"label": str(exc) + " Whole-score comparison was skipped."})
            log.warning("Whole-score candidate comparison skipped: %s", exc)
    suggestions = _explain_each_transition(suggestions, selected_measures, notify, genre=genre)

    notify("complete", {"label": "Done"})
    return AIResult(suggestions=suggestions, model_key=loaded.name)


def generate_ai_suggestions(
    key: str,
    genre: str | None,
    context: list[ContextMeasure],
    selected_measures: list[int],
    time_signature: tuple[int, int] = (4, 4),
    model_key: str | None = None,
    bpm: int | None = None,
    on_phase=None,  # type: ignore[assignment]
    section_type: str | None = None,
    cadence: str | None = None,
) -> list[Suggestion] | None:
    """`generate_ai_result` without the checkpoint key. Returns None when the
    caller should fall back to rule-based generation."""
    result = generate_ai_result(
        key, genre, context, selected_measures, time_signature,
        model_key=model_key, bpm=bpm, on_phase=on_phase, section_type=section_type, cadence=cadence,
    )
    return result.suggestions if result is not None else None
