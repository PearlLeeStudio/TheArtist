"""AI model inference service with lazy loading and graceful fallback.

Generates 3 candidates per request from the chosen checkpoint at three
temperatures, then re-ranks them by a composite score:

  • R1 — Sethares sensory roughness across the suggestion's transitions
         (lower = more consonant).
  • R2 — Berklee Book of Jazz Harmony retrieval distance per transition
         (lower = closer to an indexed theory passage; pulls voice
         leading, ii–V–I, modal interchange, tritone substitution,
         common-tone retention, and ~25 other jazz harmony concepts
         simultaneously).

Composite = α·R1 + β·R2 (both lower-is-better, currently α=β=1).
Each axis falls through gracefully if its module is unavailable; if
both fall through, candidates ship in temperature order.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

from app.models.schemas import ContextMeasure, Suggestion, TheoryExplanation
from app.services.voice_leading import apply_voice_leading_bass

log = logging.getLogger(__name__)

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

# --- Guarded R2 RAG (theory explanations) ---
RAG_AVAILABLE = False
_rag_retriever = None  # lazy-loaded
_rag_llm_client = None
_rag_init_lock = threading.Lock()
_rag_init_failed = False
_explanation_cache: dict[tuple[str, str], TheoryExplanation] = {}
# Separate cache for the fast theory-rerank path: just the Berklee
# retrieval distance, no LLM call. Populated by `_theory_distance` /
# `_theory_rerank_warmup`. Bounded by chord-symbol-pair space so memory
# stays small in practice.
_theory_rerank_cache: dict[tuple[str, str], float] = {}
try:
    from openai import OpenAI  # type: ignore[import-untyped]
    from rag.explain import explain_transition  # type: ignore[import-untyped]
    from rag.retrieve import Retriever  # type: ignore[import-untyped]

    RAG_AVAILABLE = True
except ImportError as e:
    log.warning(f"ai/rag not available — theory explanations disabled ({e})")

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

# Berklee Book of Jazz Harmony only applies to jazz-adjacent genres. RAG
# (R2 rerank + per-transition explanations) is gated to this family — for
# pop/rock/country/hip-hop/electronic/etc. the theory book has no relevant
# passages, so running the LLM judge wastes 5-10s/transition with no quality
# signal. Pearl 2026-05-09.
JAZZ_FAMILY_GENRES: set[str] = {"jazz", "blues", "bossa", "bossa nova", "rnb_soul"}
RAG_ENABLED_GENRES: set[str] = JAZZ_FAMILY_GENRES | {"classical"}


def _should_use_rag(genre: str | None) -> bool:
    """Centralised gate for the R2 theory-rerank + explanation RAG path.

    Corpus = Berklee Book of Jazz Harmony (jazz-family genres) + Open Music
    Theory v2 (classical and general harmony). The merged Retriever queries
    both collections per call and surfaces the closest passages by cosine
    distance, so a query naturally picks the relevant book. Genres outside
    these books' coverage skip RAG to avoid noisy LLM-judge wastage.
    """
    if not RAG_AVAILABLE:
        return False
    if os.getenv("THEARTIST_DISABLE_RAG"):
        return False
    if genre and genre.lower() not in RAG_ENABLED_GENRES:
        return False
    return True


# Genre alias mapping (frontend → tokenizer).
# The tokenizer was trained on the five genres above; "classical" is
# accepted from upstream callers (TheVoyager derives it from
# serene/pastoral mood) but mapped to the unconditioned bucket so the
# prompt does not carry an unknown genre token. Unknown genres
# generally fall through to "none" via the .get default below.
# After R4 LoRA expansion, extra-genre tokens (country/funk/gospel/etc.) are
# in the extended tokenizer; tokenizer extension is loaded only when serving
# a LoRA model.
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
        self._load_lock = threading.Lock()

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

        new_emb = nn.Embedding(new_vocab, d_model, padding_idx=pad_id).to(device)
        with torch.no_grad():
            new_emb.weight[:old_vocab] = model.token_emb.weight
            none_row = model.token_emb.weight[none_id]
            for i in range(old_vocab, new_vocab):
                new_emb.weight[i] = none_row
        model.token_emb = new_emb

        new_out = nn.Linear(d_model, new_vocab, bias=False).to(device)
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
        model = MusicTransformer(
            vocab_size=trained_vocab,
            d_model=config.get("d_model", 512),
            n_heads=config.get("n_heads", 8),
            d_ff=config.get("d_ff", 2048),
            n_layers=config.get("n_layers", 8),
            max_seq_len=config.get("max_seq_len", 256),
            dropout=0.0,
            pad_id=self.tokenizer.pad_id,
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        self._resize_embedding_inplace(model, trained_vocab, extended_vocab, device)
        return model

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
        ckpt = torch.load(f1_path, map_location=device, weights_only=False)
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
        peft_model.eval()
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
            ckpt = torch.load(path, map_location=device, weights_only=False)
            config = ckpt.get("config", {})
            model = self._build_base_model(ckpt, config, device)
            model.eval()
            loaded = LoadedModel(model=model, name=name, device=device)
            self._models[name] = loaded
            params = sum(p.numel() for p in model.parameters()) / 1e6
            log.info(f"Loaded model '{name}' on {device} ({params:.1f}M params)")
            return loaded

    def warmup(self) -> dict[str, bool]:
        results = {}
        for name in CHECKPOINTS:
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
) -> list[list[str]]:
    """Run model inference, return list of bars (each bar = list of chord strings)."""
    device = loaded.device
    prompt = torch.tensor([prompt_ids], device=device)
    max_new = n_bars * 3 + 2  # BAR + 2 chords per bar + EOS

    # Structural separators MUST be allowed to recur; exempt them from
    # repetition / no-repeat-ngram controls.
    exempt = {tokenizer.bar_id}
    if hasattr(tokenizer, "pad_id"):
        exempt.add(tokenizer.pad_id)

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
    )

    gen_ids = output[0, len(prompt_ids):].tolist()
    gen_tokens = tokenizer.decode(gen_ids)

    # Parse tokens into bars
    bars: list[list[str]] = []
    current_bar: list[str] = []
    for tok in gen_tokens:
        if tok == "[BAR]":
            if current_bar:
                bars.append(current_bar)
            current_bar = []
        elif tok in ("[EOS]", "[PAD]"):
            break
        elif not tok.startswith("["):
            current_bar.append(tok)
    if current_bar:
        bars.append(current_bar)

    return bars[:n_bars]


def _bars_to_chord_map(
    bars: list[list[str]],
    selected_measures: list[int],
    chords_per_bar: int = 2,
) -> dict[str, list[str]]:
    """Convert model output bars to Suggestion.chords format."""
    chord_map: dict[str, list[str]] = {}
    for i, m_idx in enumerate(selected_measures):
        bar_chords = bars[i][:chords_per_bar] if i < len(bars) else []
        while len(bar_chords) < chords_per_bar:
            bar_chords.append(bar_chords[-1] if bar_chords else "Cmaj")
        chord_map[str(m_idx)] = bar_chords
    return chord_map


def _get_rag_services() -> tuple[object, object] | tuple[None, None]:
    """Lazy-load the retriever and OpenAI client on first use, reused thereafter.

    Serialised with a lock so concurrent worker threads don't race to construct
    chroma's PersistentClient — that race surfaces as "Could not connect to
    tenant default_tenant" / "RustBindingsAPI ... bindings" when the rust
    bindings get half-initialised.
    """
    global _rag_retriever, _rag_llm_client, _rag_init_failed
    if os.getenv("THEARTIST_DISABLE_RAG"):
        return None, None
    if not RAG_AVAILABLE or _rag_init_failed:
        return None, None
    if _rag_retriever is not None:
        return _rag_retriever, _rag_llm_client
    with _rag_init_lock:
        if _rag_retriever is not None:
            return _rag_retriever, _rag_llm_client
        if _rag_init_failed:
            return None, None
        try:
            _rag_retriever = Retriever(collections=["berklee", "openmusictheory"])  # type: ignore[name-defined]
            _rag_llm_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")  # type: ignore[name-defined]
            log.info("RAG services initialised (Retriever + ollama client)")
        except Exception as e:
            log.warning(f"RAG services failed to initialise: {e}")
            _rag_init_failed = True
            return None, None
    return _rag_retriever, _rag_llm_client


def _explain_one(a: str, b: str) -> TheoryExplanation | None:
    """Cached per-transition RAG explanation call."""
    key = (a, b)
    if key in _explanation_cache:
        return _explanation_cache[key]
    retriever, client = _get_rag_services()
    if retriever is None or client is None:
        return None
    try:
        e = explain_transition(a, b, retriever, client)  # type: ignore[name-defined]
        te = TheoryExplanation(
            chord_a=e.chord_a,
            chord_b=e.chord_b,
            concept=e.concept,
            explanation=e.explanation,
            chapter=e.top1_chapter,
            section=e.top1_section,
            page_start=e.top1_page_start,
            page_end=e.top1_page_end,
        )
        _explanation_cache[key] = te
        return te
    except Exception as exc:
        log.warning(f"RAG explanation failed for {a} → {b}: {exc}")
        return None


def _attach_explanations(suggestions: list[Suggestion], selected_measures: list[int]) -> list[Suggestion]:
    """For each suggestion, compute per-transition theory explanations and attach.

    Transitions are parallelised with a ThreadPoolExecutor; the in-memory
    cache makes repeated (chord_a, chord_b) pairs free across suggestions.
    Falls through unchanged if RAG is unavailable.
    """
    if (
        not RAG_AVAILABLE
        or os.getenv("THEARTIST_DISABLE_RAG")
        or os.getenv("THEARTIST_DISABLE_EXPLANATIONS")
    ):
        return suggestions
    from concurrent.futures import ThreadPoolExecutor

    # Gather the unique ordered chord sequence per suggestion.
    seq_per_sug: list[list[str]] = []
    for s in suggestions:
        chords: list[str] = []
        for m in sorted(s.chords.keys(), key=int):
            chords.extend(s.chords[m])
        seq_per_sug.append(chords)

    # Unique transitions across all suggestions (cache dedup).
    all_pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for seq in seq_per_sug:
        for a, b in zip(seq, seq[1:]):
            if (a, b) not in seen and (a, b) not in _explanation_cache:
                seen.add((a, b))
                all_pairs.append((a, b))

    if all_pairs:
        log.info(f"RAG: computing {len(all_pairs)} new transition explanations (cache has {len(_explanation_cache)})")
        with ThreadPoolExecutor(max_workers=4) as ex:
            list(ex.map(lambda ab: _explain_one(*ab), all_pairs))

    # Attach (pull from cache / already computed).
    for s, seq in zip(suggestions, seq_per_sug):
        explanations: list[TheoryExplanation] = []
        for a, b in zip(seq, seq[1:]):
            te = _explanation_cache.get((a, b)) or _explain_one(a, b)
            if te is not None:
                explanations.append(te)
        s.explanations = explanations or None
    return suggestions


def _flatten_suggestion(s: Suggestion) -> list[str]:
    """Flatten a suggestion's per-measure chord lists into one ordered sequence."""
    out: list[str] = []
    for m_str in sorted(s.chords.keys(), key=int):
        out.extend(s.chords[m_str])
    return out


def _theory_distance(a: str, b: str) -> float | None:
    """Berklee retrieval distance for the transition `a` → `b`.

    Lower distance = closer to an indexed Berklee passage on chord
    progression / voice leading / cadence. Returns ``None`` when RAG
    is unavailable so callers can fall through to R1-only ranking.

    Cached by (a, b) pair. Cheap on warm hits; ~50–150 ms on cold.
    """
    key = (a, b)
    if key in _theory_rerank_cache:
        return _theory_rerank_cache[key]
    retriever, _ = _get_rag_services()
    if retriever is None:
        return None
    try:
        # Query template chosen to maximise overlap with Berklee chunks
        # that discuss chord-to-chord motion (voice leading, cadence,
        # secondary dominants, modal interchange, etc.).
        query = (
            f"{a} to {b}: chord progression, voice leading, "
            "common tones, cadence, and harmonic function"
        )
        hits = retriever.query(query, k=1)  # type: ignore[union-attr]
        if not hits:
            return None
        d = float(hits[0].distance)
        _theory_rerank_cache[key] = d
        return d
    except Exception as exc:
        log.warning(f"R2 theory rerank lookup failed for {a} → {b}: {exc}")
        return None


def _theory_rerank_warmup(suggestions: list[Suggestion]) -> None:
    """Pre-populate the theory cache for all unique transitions in parallel.

    Cuts cold-start cost from ~3×4×100 ms serial to ~300 ms wall-clock
    (4-way thread pool). Free if RAG is disabled.
    """
    if not RAG_AVAILABLE or os.getenv("THEARTIST_DISABLE_RAG"):
        return
    pairs: set[tuple[str, str]] = set()
    for s in suggestions:
        flat = _flatten_suggestion(s)
        for a, b in zip(flat, flat[1:]):
            if (a, b) not in _theory_rerank_cache:
                pairs.add((a, b))
    if not pairs:
        return
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(lambda ab: _theory_distance(*ab), pairs))


def _rerank_composite(
    suggestions: list[Suggestion],
    alpha: float | None = None,
    beta: float | None = None,
    genre: str | None = None,
) -> list[Suggestion]:
    """Sort suggestions by α·R1 (Sethares) + β·R2 (Berklee retrieval).

    Both axes are lower-is-better (R1 = mean roughness; R2 = mean retrieval
    distance), so the composite is a weighted sum and the lowest score wins.
    Passes through unchanged if both axes are unavailable. When only one
    axis is available, that axis effectively decides the order.

    Re-letters Option A/B/C by the new rank. Logs the per-suggestion
    R1 / R2 / composite breakdown so quality regressions are diagnosable
    from the server log.

    Safety knobs (env vars, all optional):
      THEARTIST_DISABLE_R2_RERANK   = "1"  # ship R1-only ranking
      THEARTIST_DISABLE_EXPLANATIONS = "1"  # skip per-transition LLM
                                            # explanations (fast path)
      THEARTIST_DISABLE_RAG         = "1"  # disables both R2 rerank and
                                            # explanations
      THEARTIST_RERANK_ALPHA        = "1.0"  # weight on Sethares
      THEARTIST_RERANK_BETA         = "1.0"  # weight on Berklee
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

    # R2 (Berklee theory) gated on (a) explicit env disable, (b) RAG
    # availability, (c) genre being in the Berklee Book of Jazz Harmony's
    # coverage. Pearl 2026-05-09 — non-jazz genres skip the slow LLM judge.
    r2_disabled = (
        bool(os.getenv("THEARTIST_DISABLE_R2_RERANK"))
        or not _should_use_rag(genre)
    )

    # Warm the R2 cache in parallel so per-suggestion scoring is O(cache).
    if not r2_disabled:
        _theory_rerank_warmup(suggestions)

    def _r1(flat: list[str]) -> float:
        if not PHYSICS_AVAILABLE or len(flat) < 2:
            return 0.0
        total, n = sequence_roughness(flat)
        return total / n if n > 0 else 0.0

    def _r2(flat: list[str]) -> float:
        if r2_disabled or not RAG_AVAILABLE or len(flat) < 2:
            return 0.0
        ds = [d for d in (_theory_distance(a, b) for a, b in zip(flat, flat[1:])) if d is not None]
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

    reranked: list[Suggestion] = []
    for i, (_comp, _r1v, _r2v, s) in enumerate(breakdown):
        new_letter = chr(ord("A") + i)
        body = s.label.split(" — ", 1)[1] if " — " in s.label else s.label
        s.label = f"Option {new_letter} — {body}"
        reranked.append(s)
    return reranked


# ─── Pipeline stages ───────────────────────────────────────────────────
# `generate_ai_suggestions` orchestrates these in order. Each stage owns
# its own phase-event emission, so the orchestrator just chains calls.

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


def _compose_candidates(
    loaded: "LoadedModel",
    model_label: str,
    prompt_ids: "torch.Tensor",
    n_bars: int,
    selected_measures: list[int],
    on_phase,
) -> list[Suggestion]:
    """Stage 2 — run the model at the three configured temperatures.
    Emits `composing` once per temperature with step / total / creativity
    in the payload. Inference failures are logged and skipped, so a
    flaky pass at one temperature doesn't kill the whole request."""
    registry = ModelRegistry.get()
    tokenizer = registry.tokenizer
    suggestions: list[Suggestion] = []
    for step, (temp, creativity) in enumerate(_TEMPS, start=1):
        on_phase("composing", {
            "label": f"Composing variation {step}/{len(_TEMPS)} ({creativity})",
            "step": step,
            "total": len(_TEMPS),
            "creativity": creativity,
        })
        try:
            bars = generate_with_model(loaded, tokenizer, prompt_ids, n_bars, temperature=temp)
            chord_map = _bars_to_chord_map(bars, selected_measures)
            letter = chr(ord("A") + len(suggestions))
            suggestions.append(Suggestion(
                label=f"Option {letter} — {model_label} ({creativity})",
                chords=chord_map,
            ))
        except Exception:
            log.exception(f"Inference failed: temp={temp}")
    return suggestions


def _score_and_rerank(
    suggestions: list[Suggestion], on_phase, genre: str | None = None,
) -> list[Suggestion]:
    """Stage 3 — composite rerank (R1 Sethares + R2 Berklee retrieval).
    R2 only fires for jazz-family genres (Pearl 2026-05-09); other genres
    skip the slow LLM-judged theory axis and rerank by R1 alone. Emits
    `ranking` and reassigns A/B/C labels by score."""
    on_phase("ranking", {"label": "Scoring consonance and theory grounding"})
    return _rerank_composite(suggestions, genre=genre)


def _inject_voice_leading(suggestions: list[Suggestion], on_phase) -> list[Suggestion]:
    """Stage 4 — pick inversions for smooth bass motion. Emits `voicing`
    unless disabled by env var."""
    if os.getenv("THEARTIST_DISABLE_VOICE_LEADING"):
        return suggestions
    on_phase("voicing", {"label": "Picking inversions for smooth bass motion"})
    return apply_voice_leading_bass(suggestions)


def _explain_each_transition(
    suggestions: list[Suggestion],
    selected_measures: list[int],
    on_phase,
    genre: str | None = None,
) -> list[Suggestion]:
    """Stage 5 — per-transition theory explanations via Berklee RAG +
    Gemma judgement. Skipped for non-jazz-family genres (the Berklee book
    has no relevant passages, so the LLM judge wastes 5-10s/transition).
    Emits `explaining` only when actually running."""
    if not _should_use_rag(genre) or os.getenv("THEARTIST_DISABLE_EXPLANATIONS"):
        return suggestions
    on_phase("explaining", {"label": "Generating per-transition theory explanations"})
    return _attach_explanations(suggestions, selected_measures)


def generate_ai_suggestions(
    key: str,
    genre: str | None,
    context: list[ContextMeasure],
    selected_measures: list[int],
    time_signature: tuple[int, int] = (4, 4),
    model_key: str | None = None,
    bpm: int | None = None,
    on_phase=None,  # type: ignore[assignment]
) -> list[Suggestion] | None:
    """Generate 3 suggestions, then re-rank by sensory consonance + theory.

    Pipeline stages (each owns its phase event):
        1. _resolve_model            → "model_load" (cold only)
        2. _compose_candidates       → "composing" ×3
        3. _score_and_rerank         → "ranking"
        4. _inject_voice_leading     → "voicing" (unless disabled)
        5. _explain_each_transition  → "explaining" (when RAG enabled)

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

    suggestions = _compose_candidates(
        loaded, model_label, prompt_ids, n_bars, selected_measures, notify,
    )
    if not suggestions:
        return None

    suggestions = _score_and_rerank(suggestions, notify, genre=genre)
    suggestions = _inject_voice_leading(suggestions, notify)
    suggestions = _explain_each_transition(suggestions, selected_measures, notify, genre=genre)

    notify("complete", {"label": "Done"})
    return suggestions
