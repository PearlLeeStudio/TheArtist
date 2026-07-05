from pydantic import BaseModel, Field, model_validator


class ContextMeasure(BaseModel):
    measure: int
    chords: list[str | None]


VALID_SECTIONS = {"intro", "verse", "chorus", "bridge", "outro"}


class GenerateRequest(BaseModel):
    key: str = Field(..., examples=["G major", "A minor"])
    bpm: int = Field(default=80, ge=20, le=300)
    genre: str | None = None
    timeSignature: tuple[int, int] = (4, 4)
    context: list[ContextMeasure]
    selectedMeasures: list[int]
    sectionType: str | None = Field(
        default=None,
        description="Song section: intro, verse, chorus, bridge, outro",
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


class TheoryExplanation(BaseModel):
    """Per-transition theory pointer produced by the R2 RAG layer."""
    chord_a: str
    chord_b: str
    concept: str                     # e.g., "ii-V", "V-I", "secondary dominant", "uncovered"
    explanation: str                 # 2-4 sentence prose
    chapter: str = ""
    section: str = ""
    page_start: int = 0
    page_end: int = 0


class Suggestion(BaseModel):
    label: str
    chords: dict[str, list[str]]
    explanations: list[TheoryExplanation] | None = None


class GenerateResponse(BaseModel):
    suggestions: list[Suggestion]
    source: str = "rules"  # "ai", "rules", or "ai+rules"


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


class SongGenerateRequest(BaseModel):
    """Voyager primary endpoint. Voyager sends only (genre, length_bars);
    artist owns key / bpm / time-signature defaults internally."""
    genre: str = Field(..., description="One of the 13 supported genres")
    length_bars: int = Field(default=8, ge=1, le=64)
    n_candidates: int = Field(default=1, ge=1, le=3)
    seed: int | None = None


class SongGenerateResponse(BaseModel):
    genre: str
    key: str
    bpm: int
    time_signature: tuple[int, int]
    bars: list[dict]                       # [{"chords": [...]}]
    tracks: dict[str, Track]               # 3 tracks: harmony, bass, drum
    midi_b64: str                          # MIDI render for caller convenience
    model_used: str
