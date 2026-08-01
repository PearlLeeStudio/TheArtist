from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from app.models.sections import CHORDS_PER_BAR, MAX_SECTION_CHORDS, SECTION_TYPES, SectionType
from app.services.genre_tables import (
    BASS_INSTRUMENTS,
    DRUM_PATTERNS,
    HARMONY_INSTRUMENTS,
    HARMONY_RHYTHM_PRESETS,
    SONG_BPM_MAX,
    SONG_BPM_MIN,
    SONG_KEYS,
)


class ContextMeasure(BaseModel):
    measure: int
    chords: list[str | None]


VALID_SECTIONS = set(SECTION_TYPES)
# Sampling seeds are unsigned 32-bit integers in every generation endpoint.
MAX_SEED = 0xFFFFFFFF


class GenerateRequest(BaseModel):
    key: str = Field(..., examples=["G major", "A minor"])
    bpm: int = Field(default=80, ge=20, le=300)
    genre: str | None = None
    timeSignature: tuple[int, int] = (4, 4)
    context: list[ContextMeasure]
    selectedMeasures: list[int]
    sectionType: str | None = Field(
        default=None,
        description="Song-form section id from app/models/sections.py (intro, verse, pre_chorus, "
                    "chorus, ..., head, a_section, b_section); null applies no song-form rules",
    )
    modelKey: str | None = Field(
        default=None,
        description="Which model to use: phase0, ft_f1, ft_f1_v2, ft_f2..f5, ft_f1_lora_<genre>. Default: ft_f1.",
    )

    # Computed fields for model context — populated automatically
    totalMeasures: int = 0
    positionRatio: float = 0.0  # 0.0=start, 0.5=middle, 1.0=end

    @model_validator(mode="after")
    def _compute_position_context(self) -> "GenerateRequest":
        self.totalMeasures = len(self.context)
        if self.totalMeasures > 0 and self.selectedMeasures:
            valid = [m for m in self.selectedMeasures if 0 <= m < self.totalMeasures]
            if valid:
                mid_selected = sum(valid) / len(valid)
                self.positionRatio = round(
                    min(1.0, max(0.0, mid_selected / self.totalMeasures)), 3
                )
        return self


class TheoryCitation(BaseModel):
    source: Literal["omt", "berklee"]
    title: str
    url: str = ""
    authors: list[str] = Field(default_factory=list)
    editors: list[str] = Field(default_factory=list)
    license: str
    license_url: str = ""
    copyright: str = ""
    changes: str = ""
    passage_id: str


class TheoryExplanation(BaseModel):
    """Per-transition theory pointer produced by the R2 RAG layer."""
    chord_a: str
    chord_b: str
    concept: str                     # e.g., "ii-V", "V-I", "secondary dominant", "uncovered"
    explanation: str                 # Product retrieval returns a labeled source excerpt
    chapter: str = ""
    section: str = ""
    page_start: int = 0
    page_end: int = 0
    kind: Literal["generated", "source_excerpt"] = "generated"
    citation: TheoryCitation | None = None


class Suggestion(BaseModel):
    label: str
    chords: dict[str, list[str]]
    explanations: list[TheoryExplanation] | None = None


class GenerateResponse(BaseModel):
    suggestions: list[Suggestion]
    source: str = "rules"  # "ai", "rules", or "ai+rules"


# The endings the song-form rules can write (app/services/song_form.py); "none"
# keeps the model's own ending. Also the planner's per-section cadence vocabulary.
CADENCE_TYPES = ("authentic", "plagal", "half", "half_iv_v", "turnaround", "none")
CadenceType = Literal["authentic", "plagal", "half", "half_iv_v", "turnaround", "none"]


class SectionGenerateRequest(BaseModel):
    """One song-form section composed after the measures in `context`."""
    key: str = Field(..., examples=["C major", "A minor"])
    bpm: int = Field(default=80, ge=20, le=300)
    genre: str | None = None
    timeSignature: tuple[int, int] = (4, 4)
    sectionType: SectionType
    chordCount: int = Field(
        ..., ge=CHORDS_PER_BAR, le=MAX_SECTION_CHORDS,
        description="Chord slots to fill, two per measure; the section spans ceil(chordCount / 2) bars",
    )
    context: list[ContextMeasure] = Field(
        default_factory=list,
        description="The measures before the new section, indices 0..k-1 in order",
    )
    modelKey: str | None = Field(
        default=None,
        description="Same checkpoint keys as GenerateRequest.modelKey; null = genre dispatch",
    )
    seed: int | None = Field(default=None, ge=0, le=MAX_SEED, strict=True,
                             description="Optional request-local seed for chord candidates; null stays stochastic")
    cadence: CadenceType | None = Field(
        default=None,
        description="Ending the song-form rules write instead of the section's own (the planner's "
                    "choice): authentic, plagal, half, half_iv_v, turnaround, or none to keep the "
                    "model's ending; null = the section's default",
    )


class SectionGenerateResponse(BaseModel):
    sectionType: SectionType
    title: str                       # e.g. "Chorus — 8 bars"
    startMeasure: int                # = len(context); first suggestion key
    bars: int                        # = ceil(chordCount / 2)
    chordCount: int                  # = bars * 2
    suggestions: list[Suggestion]    # chords keyed by absolute measure index
    source: Literal["ai", "rules"]
    modelUsed: str | None = None     # checkpoint that produced the chords; null for rules


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None


# --- TheVoyager endpoint schemas ---


class ChordGenerateRequest(BaseModel):
    """Simplified request for programmatic callers (TheVoyager)."""
    key: str = Field(..., examples=["G major", "A minor"])
    genre: str | None = None
    time_signature: tuple[int, int] = (4, 4)
    bpm: int | None = Field(
        default=None, ge=20, le=300,
        description="Optional BPM hint. Currently echoed in response for caller-side traceability; not encoded into the prompt (the trained tokenizer has no BPM-bucket tokens). Reserved for a future training round.",
    )
    n_bars: int = Field(default=4, ge=1, le=32)
    context_bars: list[list[str]] | None = Field(
        default=None,
        description="Preceding bars as context, e.g. [['Cmaj7', 'Am7'], ['Dm7', 'G7']]",
    )
    temperature: float = Field(default=0.8, ge=0.1, le=2.0)
    model: str = Field(
        default="ft_f1",
        description="Checkpoint key: 'phase0', 'ft_f1'..'ft_f5', 'ft_f1_v2', or 'ft_f1_lora_<genre>'",
    )
    n_candidates: int = Field(
        default=1, ge=1, le=5,
        description="Number of chord progression candidates to return. n=1 returns 'bars' field (backward-compat); n>1 returns 'candidates' list with varied temperature for diversity.",
    )


class ChordCandidate(BaseModel):
    """One chord progression candidate, used when n_candidates > 1."""
    bars: list[dict]                      # [{"chords": [...]}]
    temperature: float                    # actual sampling temperature used
    seed: int | None = None               # reserved for deterministic sampling


# --- TheVoyager composite endpoint /api/generate/song ---


class TrackEvent(BaseModel):
    """One symbolic event in a multi-track render. For pitched layers
    (harmony/bass), `pitch` is a MIDI note number. For drum events,
    `voice` is the kit voice ('K' kick, 'S' snare, 'H' hi-hat closed,
    'O' hi-hat open, 'X' cross-stick / cowbell, 'C' crash, 'R' ride,
    'T'/'M'/'L' toms hi/mid/low). Times are bar-relative beats."""
    bar: int
    beat: float
    pitch: int | None = None
    voice: str | None = None
    duration: float
    velocity: float = 1.0


class Track(BaseModel):
    events: list[TrackEvent]
    instrument: str
    source: str = "rule"   # "rule" | "learned"


_RHYTHM_IDS = tuple(p["id"] for p in HARMONY_RHYTHM_PRESETS)
_RHYTHM_ID_LIST = ", ".join(_RHYTHM_IDS)
_INSTRUMENT_LIST = ", ".join(HARMONY_INSTRUMENTS)
_DRUM_ID_RANGE = f"{min(DRUM_PATTERNS)}-{max(DRUM_PATTERNS)}"   # ids are contiguous
_KEY_LIST = ", ".join(SONG_KEYS)
_BASS_LIST = ", ".join(BASS_INSTRUMENTS)


class SongGenerateRequest(BaseModel):
    """Voyager primary endpoint: a genre and a length are enough, and every other
    choice falls back to the genre's own. The optional key, tempo and bass, and the
    three Play-panel choices (comping rhythm, drum pattern, harmony instrument),
    replace the genre's; null keeps it. The time signature stays the genre's."""
    genre: str = Field(..., description="One of the 13 supported genres")
    length_bars: int = Field(default=8, ge=1, le=64)
    n_candidates: int = Field(
        default=1, ge=1, le=1, strict=True,
        description="This composite endpoint returns one arrangement and supports only 1. "
                    "Use /api/generate/chords for multiple chord candidates.",
    )
    seed: int | None = Field(
        default=None, ge=0, le=MAX_SEED, strict=True,
        description=f"Optional request-local sampling seed (0..{MAX_SEED}), including rule fallback. "
                    "Reproducible for the same model, device, runtime and request; null stays stochastic.",
    )
    harmony_rhythm: str | None = Field(
        default=None,
        description=f"Harmony comping rhythm preset id, one of: {_RHYTHM_ID_LIST}. "
                    "Null = the genre's own",
    )
    drum_pattern: int | Literal["off"] | None = Field(
        default=None,
        description=f"Drum pattern id {_DRUM_ID_RANGE}, or 'off' for no drum events. Null = the genre's own",
    )
    harmony_instrument: str | None = Field(
        default=None,
        description=f"Harmony GM instrument, one of: {_INSTRUMENT_LIST}. Null = the genre's own",
    )
    key: str | None = Field(
        default=None,
        description=f"Key, one of the model's 24 keys: {_KEY_LIST}. The model is prompted with it, and a "
                    "pass that clearly sits in another key is moved into it (a near tie stays as written). "
                    "Null = the genre's own, with the pass as written",
    )
    bpm: int | None = Field(
        default=None, ge=SONG_BPM_MIN, le=SONG_BPM_MAX, strict=True,
        description=f"Tempo in quarter notes per minute, {SONG_BPM_MIN}-{SONG_BPM_MAX}. Null = the genre's own",
    )
    bass_instrument: str | None = Field(
        default=None,
        description=f"Bass GM instrument, one of: {_BASS_LIST}. Null = the genre's own",
    )

    @field_validator("key")
    @classmethod
    def _known_key(cls, v: str | None) -> str | None:
        if v is not None and v not in SONG_KEYS:
            raise ValueError(f"unknown key {v!r}; expected one of: {_KEY_LIST}")
        return v

    @field_validator("bass_instrument")
    @classmethod
    def _known_bass_instrument(cls, v: str | None) -> str | None:
        if v is not None and v not in BASS_INSTRUMENTS:
            raise ValueError(f"unknown bass_instrument {v!r}; expected one of: {_BASS_LIST}")
        return v

    @field_validator("harmony_rhythm")
    @classmethod
    def _known_harmony_rhythm(cls, v: str | None) -> str | None:
        if v is not None and v not in _RHYTHM_IDS:
            raise ValueError(f"unknown harmony_rhythm {v!r}; expected one of: {_RHYTHM_ID_LIST}")
        return v

    @field_validator("drum_pattern", mode="before")
    @classmethod
    def _known_drum_pattern(cls, v: object) -> object:
        # Checked on the raw value so true, "3" or 3.0 are rejected, not read as ids.
        if v is None or v == "off" or (type(v) is int and v in DRUM_PATTERNS):
            return v
        raise ValueError(f"unknown drum_pattern {v!r}; expected a pattern id {_DRUM_ID_RANGE} or 'off'")

    @field_validator("harmony_instrument")
    @classmethod
    def _known_harmony_instrument(cls, v: str | None) -> str | None:
        if v is not None and v not in HARMONY_INSTRUMENTS:
            raise ValueError(f"unknown harmony_instrument {v!r}; expected one of: {_INSTRUMENT_LIST}")
        return v


class SongGenerateResponse(BaseModel):
    genre: str
    key: str
    bpm: int
    time_signature: tuple[int, int]
    harmony_rhythm: str                    # effective rhythm preset id (the genre's when not picked)
    drum_pattern: int | Literal["off"]     # effective drum pattern id, or "off"; sent back, it replays the same drums
    bars: list[dict]                       # [{"chords": [...]}]
    tracks: dict[str, Track]               # 3 tracks: harmony, bass, drum
    midi_b64: str                          # MIDI render for caller convenience
    model_used: str
    seed: int | None = None              # actual sampling seed; null means stochastic
