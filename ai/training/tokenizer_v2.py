"""Enriched tokenizer v2 — adds explicit functional / positional tokens.

Addresses the "model has to infer ii-V from context" limitation of v1.
Sequences now interleave three new token families alongside chord symbols:

- ``[ROMAN:1] … [ROMAN:7]`` — chromatic scale-degree of the chord root
  relative to the song key. 12 tokens (1, b2, 2, b3, 3, 4, b5, 5, b6, 6,
  b7, 7). Quality still lives in the chord token itself.
- ``[BEAT:1] … [BEAT:6]`` — beat position within the current bar. 6
  tokens covers 4/4 (1-4), 3/4 (1-3), and 6/8 (1,4 compound).
- ``[SECTION:A] [SECTION:B] [SECTION:bridge]`` — coarse formal label.
  Most chord charts have AABA / ABAC / verse-chorus-bridge structure at
  this granularity.

Vocab grows by 12 + 6 + 3 = 21 tokens (351 → 372). Typical sequence
length inflates ~3× because ROMAN + BEAT prefix every chord, so the
model's ``max_seq_len`` should be raised from 256 → 512 when training
on v2 data.

Full Phase 0 retrain is required — checkpoints trained under v1 are
incompatible. Use ``--config configs/phase0_v2.yaml`` and
``configs/ft_jazz_pop50_v2.yaml`` (staged; user kicks off).
"""
from __future__ import annotations

import json
from pathlib import Path

from tokenizer import (
    ALL_KEYS,
    GENRES,
    QUALITIES,
    ROOTS,
    ROOT_ALIASES,
    TIME_SIGS,
    ChordTokenizer,
)

# Chromatic scale-degree labels (0..11 semitones above the key tonic).
ROMAN_DEGREES = [
    "1", "b2", "2", "b3", "3", "4",
    "b5", "5", "b6", "6", "b7", "7",
]

# Beat positions — covers 4/4 (1-4), 3/4 (1-3), 6/8 (1,4 compound), 5/4, 6/4.
BEAT_POSITIONS = ["1", "2", "3", "4", "5", "6"]

# Coarse section labels.  "A"/"B"/"C" for AABA/ABAC charts, plus bridge
# and intro/outro.  Models that only see verse-chorus style can still
# reuse A (verse), B (chorus), bridge.
SECTION_LABELS = ["A", "B", "C", "bridge", "intro", "outro"]


def _semitone_distance(chord_root: str, key_tonic: str) -> int | None:
    """Return (chord_root − key_tonic) mod 12, or None if unparseable."""
    root = ROOT_ALIASES.get(chord_root, chord_root)
    tonic = ROOT_ALIASES.get(key_tonic, key_tonic)
    if root not in ROOTS or tonic not in ROOTS:
        return None
    return (ROOTS.index(root) - ROOTS.index(tonic)) % 12


def roman_token(chord: str, key: str) -> str | None:
    """Compute the ``[ROMAN:…]`` scale-degree token for a chord in a key.

    ``chord`` is expected to be a normalized token (e.g. ``Dm7``, ``F#7``).
    ``key`` is ``{Root}maj|min`` (e.g. ``Cmaj``, ``F#min``).
    Returns ``None`` if either is unparseable.
    """
    if not chord or not key:
        return None
    # Extract the root portion of the chord
    c_root = chord[0]
    rest = chord[1:]
    if rest and rest[0] in ("b", "#"):
        c_root += rest[0]
    # Key root = leading alphabetics + optional accidental, before maj/min
    mode = None
    for m in ("maj", "min"):
        if key.endswith(m):
            mode = m
            key_root = key[: -len(m)]
            break
    else:
        return None
    dist = _semitone_distance(c_root, key_root)
    if dist is None:
        return None
    # In minor keys, shift so that the "1" token refers to the tonic
    # interpreted against the natural-minor scale. We keep the chromatic
    # mapping — b3/3/b6/6/b7/7 labels stay chromatic regardless of mode.
    _ = mode  # mode carried for potential future uses; mapping is chromatic.
    return f"[ROMAN:{ROMAN_DEGREES[dist]}]"


class ChordTokenizerV2(ChordTokenizer):
    """ChordTokenizer + ROMAN / BEAT / SECTION tokens."""

    def _build_vocab(self) -> None:
        tokens: list[str] = ["[PAD]", "[BOS]", "[EOS]", "[BAR]"]
        for key in ALL_KEYS:
            tokens.append(f"[KEY:{key}]")
        for ts in TIME_SIGS:
            tokens.append(f"[TIME:{ts}]")
        for genre in GENRES:
            tokens.append(f"[GENRE:{genre}]")
        tokens.append("[GENRE:none]")
        for degree in ROMAN_DEGREES:
            tokens.append(f"[ROMAN:{degree}]")
        for beat in BEAT_POSITIONS:
            tokens.append(f"[BEAT:{beat}]")
        for sec in SECTION_LABELS:
            tokens.append(f"[SECTION:{sec}]")
        for root in ROOTS:
            for quality in QUALITIES:
                tokens.append(f"{root}{quality}")
        for i, tok in enumerate(tokens):
            self.token2id[tok] = i
            self.id2token[i] = tok

    # ------------------------------------------------------------------
    # Enriched encoders
    # ------------------------------------------------------------------

    def encode_roman(self, chord: str, key: str) -> int | None:
        tok = roman_token(chord, key)
        return self.token2id.get(tok) if tok else None

    def encode_beat(self, beat: int) -> int | None:
        return self.token2id.get(f"[BEAT:{beat}]")

    def encode_section(self, section: str) -> int | None:
        return self.token2id.get(f"[SECTION:{section}]")

    def encode_sequence(self, song: dict) -> list[int]:
        """Encode an enriched song dict into token IDs.

        Extends the v1 ``encode_sequence`` with three new per-chord
        prefixes: ``[SECTION:…]`` at section boundaries, ``[ROMAN:…]`` +
        ``[BEAT:…]`` before each chord.

        Expected *song* format (backward-compatible — all new fields
        optional, falling back to sensible defaults)::

            {
                "key": "Cmaj",
                "time_signature": "4/4",
                "genre": "jazz",
                "sections": ["A", "A", "B", "A"],   # one label per bar (optional)
                "bars": [["Dm7", "G7"], ["Cmaj7"], ...],
            }
        """
        ids: list[int] = [self.bos_id]

        if kid := self.encode_key(song.get("key", "")):
            ids.append(kid)
        if tid := self.encode_time_sig(song.get("time_signature", "")):
            ids.append(tid)
        if gid := self.encode_genre(song.get("genre", "")):
            ids.append(gid)

        key = song.get("key", "Cmaj")
        time_sig = song.get("time_signature", "4/4")
        try:
            beats_per_bar = int(time_sig.split("/")[0])
        except Exception:
            beats_per_bar = 4
        sections = song.get("sections") or []
        last_section: str | None = None

        for bar_idx, bar in enumerate(song.get("bars", [])):
            # Section token at each new-section boundary
            sec = sections[bar_idx] if bar_idx < len(sections) else None
            if sec and sec != last_section:
                sid = self.encode_section(sec)
                if sid is not None:
                    ids.append(sid)
                last_section = sec
            ids.append(self.bar_id)
            # Distribute chords evenly across the beat grid. 2 chords per
            # bar → beats 1 and 3; 1 chord → beat 1; 3 chords → beats 1, 2,
            # 3 (or 1, 3, 5 in 6/8 — we keep it simple as integer beats).
            n = len(bar)
            for i, chord in enumerate(bar):
                if n == 1:
                    beat = 1
                elif n == 2:
                    beat = 1 if i == 0 else max(3, beats_per_bar // 2 + 1)
                else:
                    beat = 1 + int(round(i * (beats_per_bar - 1) / max(n - 1, 1)))
                # ROMAN prefix
                rid = self.encode_roman(chord, key)
                if rid is not None:
                    ids.append(rid)
                # BEAT prefix
                bid = self.encode_beat(beat)
                if bid is not None:
                    ids.append(bid)
                # Chord itself
                cid = self.encode_chord(chord)
                if cid is not None:
                    ids.append(cid)

        ids.append(self.eos_id)
        return ids

    def save(self, path: Path) -> None:
        """Persist vocab map to JSON for reproducibility."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"token2id": self.token2id, "version": "v2"}, f, indent=2)
