"""Chord sequence tokenizer for Music Transformer training.

Vocabulary (~350 tokens):
  [PAD]=0, [BOS]=1, [EOS]=2, [BAR]=3
  [KEY:Cmaj] ... [KEY:Bmin]  (24 keys)
  [TIME:4/4] ... [TIME:5/4]  (5 time sigs)
  [GENRE:jazz] ... [GENRE:none]  (6 genres)
  Cmaj, Cm, C7, ... B7b13      (12 roots x 26 qualities = 312 chords)
"""

from __future__ import annotations

import json
from pathlib import Path

# Canonical root names (jazz convention: prefer flats)
ROOTS = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

# Root name aliases for normalization
ROOT_ALIASES: dict[str, str] = {
    "C#": "Db", "D#": "Eb", "E#": "F", "Fb": "E",
    "G#": "Ab", "A#": "Bb", "B#": "C", "Cb": "B",
    "Gb": "F#",
    # Lowercase
    "c": "C", "d": "D", "e": "E", "f": "F", "g": "G", "a": "A", "b": "B",
    "c#": "Db", "db": "Db", "d#": "Eb", "eb": "Eb",
    "f#": "F#", "gb": "F#", "g#": "Ab", "ab": "Ab",
    "a#": "Bb", "bb": "Bb", "cb": "B", "fb": "E",
}

# Chord qualities in our vocabulary
QUALITIES = [
    "maj", "m", "7", "maj7", "m7", "m7b5", "dim7", "dim", "aug",
    "sus4", "sus2", "6", "m6", "9", "m9", "maj9", "11", "m11",
    "13", "m13", "add9", "mMaj7", "7b9", "7#9", "7#11", "7b13",
]

# Quality alias mapping → canonical quality
_QUALITY_ALIASES: dict[str, str] = {
    # Major
    "major": "maj", "M": "maj",
    # Minor
    "min": "m", "minor": "m", "-": "m", "mi": "m",
    # Dominant 7
    "dom7": "7", "dom": "7",
    # Major 7
    "^7": "maj7", "M7": "maj7", "Maj7": "maj7", "major7": "maj7",
    "j7": "maj7", "^": "maj7", "delta": "maj7",
    # Minor 7
    "min7": "m7", "-7": "m7", "mi7": "m7",
    # Half-diminished
    "hdim7": "m7b5", "hdim": "m7b5", "h7": "m7b5",
    "%7": "m7b5", "%": "m7b5",
    # Diminished
    "o": "dim", "o7": "dim7",
    # Augmented
    "+": "aug",
    # Suspended
    "sus": "sus4",
    # 6th
    "min6": "m6", "-6": "m6",
    # 9th
    "min9": "m9", "-9": "m9", "M9": "maj9", "^9": "maj9", "Maj9": "maj9",
    # 11th
    "min11": "m11", "-11": "m11",
    # 13th
    "min13": "m13", "-13": "m13",
    # Minor-major 7
    "minmaj7": "mMaj7", "-^7": "mMaj7", "mM7": "mMaj7",
    # Altered dominants
    "7alt": "7b9",
}

# Keys and metadata
MAJOR_KEYS = [f"{r}maj" for r in ROOTS]
MINOR_KEYS = [f"{r}min" for r in ROOTS]
ALL_KEYS = MAJOR_KEYS + MINOR_KEYS
TIME_SIGS = ["4/4", "3/4", "6/8", "2/4", "5/4"]
GENRES = ["jazz", "pop", "rock", "blues", "bossa"]

EXTRA_GENRES = [
    "country", "rnb_soul", "hip_hop", "electronic",
    "funk", "folk", "gospel", "classical",
]


class ChordTokenizer:
    """Deterministic tokenizer for chord sequences."""

    PAD = 0
    BOS = 1
    EOS = 2
    BAR = 3

    def __init__(self, include_extra_genres: bool = False) -> None:
        self.include_extra_genres = include_extra_genres
        self.token2id: dict[str, int] = {}
        self.id2token: dict[int, str] = {}
        self._build_vocab()

    # ------------------------------------------------------------------
    # Vocab construction
    # ------------------------------------------------------------------

    def _build_vocab(self) -> None:
        tokens: list[str] = ["[PAD]", "[BOS]", "[EOS]", "[BAR]"]
        for key in ALL_KEYS:
            tokens.append(f"[KEY:{key}]")
        for ts in TIME_SIGS:
            tokens.append(f"[TIME:{ts}]")
        for genre in GENRES:
            tokens.append(f"[GENRE:{genre}]")
        tokens.append("[GENRE:none]")
        for root in ROOTS:
            for quality in QUALITIES:
                tokens.append(f"{root}{quality}")
        if self.include_extra_genres:
            for genre in EXTRA_GENRES:
                tokens.append(f"[GENRE:{genre}]")
        for i, tok in enumerate(tokens):
            self.token2id[tok] = i
            self.id2token[i] = tok

    @property
    def vocab_size(self) -> int:
        return len(self.token2id)

    @property
    def pad_id(self) -> int:
        return self.PAD

    @property
    def bos_id(self) -> int:
        return self.BOS

    @property
    def eos_id(self) -> int:
        return self.EOS

    @property
    def bar_id(self) -> int:
        return self.BAR

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def encode_chord(self, chord_str: str) -> int | None:
        token = self.normalize_chord(chord_str)
        return self.token2id.get(token) if token else None

    def encode_key(self, key_str: str) -> int | None:
        return self.token2id.get(f"[KEY:{key_str}]")

    def encode_time_sig(self, ts: str) -> int | None:
        return self.token2id.get(f"[TIME:{ts}]")

    def encode_genre(self, genre: str) -> int | None:
        return self.token2id.get(f"[GENRE:{genre}]")

    def encode_sequence(self, song: dict) -> list[int]:
        """Encode a unified song dict to a token-ID sequence.

        Expected *song* format::

            {
                "key": "Cmaj",
                "time_signature": "4/4",
                "genre": "jazz",
                "bars": [["Cmaj7", "Am7"], ["Dm7", "G7"], ...]
            }
        """
        ids: list[int] = [self.BOS]

        kid = self.encode_key(song.get("key", "Cmaj"))
        if kid is not None:
            ids.append(kid)

        tid = self.encode_time_sig(song.get("time_signature", "4/4"))
        if tid is not None:
            ids.append(tid)

        gid = self.encode_genre(song.get("genre", "none"))
        if gid is not None:
            ids.append(gid)

        for bar in song.get("bars", []):
            ids.append(self.BAR)
            for chord in bar:
                cid = self.encode_chord(chord)
                if cid is not None:
                    ids.append(cid)

        ids.append(self.EOS)
        return ids

    def decode(self, ids: list[int]) -> list[str]:
        return [self.id2token.get(i, "[UNK]") for i in ids]

    # ------------------------------------------------------------------
    # Chord normalization
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_root(root: str) -> str | None:
        """Normalize a root note name to canonical form."""
        if root in ROOTS:
            return root
        if root in ROOT_ALIASES:
            return ROOT_ALIASES[root]
        # Try capitalize first letter
        cap = root[0].upper() + root[1:] if len(root) > 1 else root.upper()
        if cap in ROOTS:
            return cap
        if cap in ROOT_ALIASES:
            return ROOT_ALIASES[cap]
        return None

    @staticmethod
    def normalize_chord(chord_str: str) -> str | None:
        """Normalize any chord notation to ``{Root}{quality}`` in our vocab."""
        if not chord_str or chord_str in (
            "N", "NC", "N.C.", "X", "x",
            "pause", "silence", "&pause", "end",
        ):
            return None

        # Strip slash-chord bass
        if "/" in chord_str:
            chord_str = chord_str.split("/")[0]

        # Billboard colon format  Root:Quality
        if ":" in chord_str:
            root_part, qual_part = chord_str.split(":", 1)
            # qual_part may also have /bass — already stripped above
        else:
            root_part = chord_str[0]
            qual_part = chord_str[1:]
            if qual_part and qual_part[0] in ("b", "#"):
                root_part += qual_part[0]
                qual_part = qual_part[1:]

        norm_root = ChordTokenizer.normalize_root(root_part)
        if norm_root is None:
            return None

        quality = ChordTokenizer._normalize_quality(qual_part)
        if quality is None or quality not in QUALITIES:
            return None

        return f"{norm_root}{quality}"

    @staticmethod
    def _normalize_quality(q: str) -> str | None:
        """Map various quality notations to our canonical set."""
        if not q:
            return "maj"

        # Direct hit
        if q in QUALITIES:
            return q

        # Alias table
        if q in _QUALITY_ALIASES:
            return _QUALITY_ALIASES[q]

        # Case-insensitive alias search
        for alias, canon in _QUALITY_ALIASES.items():
            if q.lower() == alias.lower():
                return canon

        # ---- Heuristic fallbacks for unusual notations ----

        # WJazzD altered dominants: "79b" → 7b9, "79#" → 7#9, etc.
        if q.startswith("7"):
            tail = q[1:]
            if "b9" in tail or "9b" in tail:
                return "7b9"
            if "#9" in tail or "9#" in tail:
                return "7#9"
            if "#11" in tail or "11#" in tail:
                return "7#11"
            if "b13" in tail or "13b" in tail:
                return "7b13"

        # Compound minor qualities
        if q.startswith("m") or q.startswith("-"):
            inner = q.lstrip("m").lstrip("-")
            if "7" in inner and ("b5" in inner or "b5" in q):
                return "m7b5"
            if "7" in inner:
                return "m7"
            if "9" in inner:
                return "m9"
            if "11" in inner:
                return "m11"
            if "13" in inner:
                return "m13"
            if "6" in inner:
                return "m6"
            return "m"

        # Bare numbers
        if q in ("7",):
            return "7"
        if q in ("9",):
            return "9"
        if q in ("6",):
            return "6"
        if q in ("11",):
            return "11"
        if q in ("13",):
            return "13"

        # If nothing matched, approximate as major
        return "maj"

    # ------------------------------------------------------------------
    # Transposition
    # ------------------------------------------------------------------

    def transpose_chord_token(self, token: str, semitones: int) -> str | None:
        """Transpose a chord token string by *semitones*."""
        if token.startswith("["):
            return None
        root = token[0]
        rest = token[1:]
        if rest and rest[0] in ("b", "#"):
            root += rest[0]
            rest = rest[1:]
        norm_root = self.normalize_root(root)
        if norm_root is None:
            return None
        new_root = ROOTS[(ROOTS.index(norm_root) + semitones) % 12]
        return f"{new_root}{rest}"

    def transpose_key_token(self, token: str, semitones: int) -> str:
        """Transpose a key token like ``[KEY:Cmaj]``."""
        inner = token[5:-1]  # strip [KEY: and ]
        if inner.endswith("maj"):
            root, mode = inner[:-3], "maj"
        elif inner.endswith("min"):
            root, mode = inner[:-3], "min"
        else:
            return token
        norm = self.normalize_root(root)
        if norm is None:
            return token
        new_root = ROOTS[(ROOTS.index(norm) + semitones) % 12]
        return f"[KEY:{new_root}{mode}]"

    def transpose_sequence(self, ids: list[int], semitones: int) -> list[int]:
        """Transpose every chord & key token in *ids* by *semitones*."""
        if semitones % 12 == 0:
            return list(ids)
        out: list[int] = []
        for tid in ids:
            tok = self.id2token.get(tid)
            if tok is None:
                out.append(tid)
            elif tok.startswith("[KEY:"):
                new = self.transpose_key_token(tok, semitones)
                out.append(self.token2id.get(new, tid))
            elif tok.startswith("[") or tid <= self.BAR:
                out.append(tid)
            else:
                new = self.transpose_chord_token(tok, semitones)
                out.append(self.token2id[new] if new and new in self.token2id else tid)
        return out

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "token2id": self.token2id,
            "vocab_size": self.vocab_size,
        }, indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: str | Path) -> ChordTokenizer:
        tok = cls()
        data = json.loads(Path(path).read_text())
        assert data["vocab_size"] == tok.vocab_size, (
            f"Vocab mismatch: file={data['vocab_size']}, current={tok.vocab_size}"
        )
        return tok
