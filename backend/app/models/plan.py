"""Request, plan and trace models of the song planner (POST /api/plan/song, MCP plan_song / validate_plan).

Strict like the other models: unknown fields are rejected and a request's shape is
checked here (HTTP 422) before the planner's validator judges a plan, whose issues
are machine-readable.
"""

from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.models.schemas import MAX_SEED, CadenceType, SectionGenerateResponse
from app.models.sections import CHORDS_PER_BAR, MAX_SECTION_BARS, SectionType
from app.services.genre_tables import GENRE_DEFAULTS

TARGET_BARS = (16, 32, 48, 64)
TargetBars = Literal[16, 32, 48, 64]
MIN_PLAN_SECTION_BARS = 2
MAX_PLAN_SECTION_BARS = 32
MAX_PLAN_SECTIONS = 16  # 64 bars of 4-bar sections
MAX_REASON_CHARS = 120
MAX_CITES = 4
CONTRASTS = ("low", "medium", "high")
_KEY = re.compile(r"(C|C#|Db|D|D#|Eb|E|F|F#|Gb|G|G#|Ab|A|A#|Bb|B) (major|minor)")
# Chord symbols and Roman numerals as the editor writes them; bounded so a sketch stays a sketch
_Symbol = Annotated[str, Field(max_length=32)]


class SketchMeasure(BaseModel):
    model_config = ConfigDict(extra="forbid")
    index: int = Field(ge=0, lt=MAX_SECTION_BARS)
    chords: list[_Symbol | None] = Field(
        min_length=CHORDS_PER_BAR, max_length=CHORDS_PER_BAR,
        description="The measure's two chord slots; null = empty",
    )
    roman: list[_Symbol | None] = Field(
        min_length=CHORDS_PER_BAR, max_length=CHORDS_PER_BAR,
        description="Roman numeral per slot as the staff shows it (frontend romanNumeral.ts); null for an empty slot",
    )


class ExistingSection(BaseModel):
    """A section already laid over the sketch. Locked: the planner never changes it."""
    model_config = ConfigDict(extra="forbid")
    type: SectionType | None = Field(default=None, description="null keeps an imported region's role unknown")
    start: int = Field(ge=0, lt=MAX_SECTION_BARS)
    length: int = Field(ge=1, le=MAX_SECTION_BARS)
    label: str | None = Field(default=None, max_length=80)


class PlanRequest(BaseModel):
    """What the planner sees: one fixed key, meter, genre and tempo, the whole sketch
    with its locked sections, and the user's chips. No free text."""
    model_config = ConfigDict(extra="forbid")
    genre: str = Field(..., description="One of the 13 supported genres; its defaults fill key, bpm and "
                                        "timeSignature when they are null (a new song)")
    key: str | None = Field(default=None, examples=["C major", "A minor"])
    timeSignature: tuple[int, int] | None = None
    bpm: int | None = Field(default=None, ge=20, le=300)
    measures: list[SketchMeasure] = Field(
        default_factory=list, max_length=MAX_SECTION_BARS,
        description="The whole sketch in order, indexed 0..n-1; empty for a new song",
    )
    sections: list[ExistingSection] = Field(
        default_factory=list, max_length=MAX_SECTION_BARS,
        description="Existing sections over the sketch, in order and without overlap; locked",
    )
    targetBars: TargetBars = Field(..., description="Total length of the song, sketch included")
    wantOutro: bool = False
    contrast: Literal["low", "medium", "high"] = "medium"
    chorusMode: Literal["reuse", "new"] = Field(
        default="new", description="reuse keeps the first complete same-length chorus, adapting only its planned ending; new composes each chorus independently",
    )
    modelKey: str | None = Field(default=None, description="Checkpoint for every section, as SectionGenerateRequest.modelKey")
    seed: int | None = Field(default=None, ge=0, le=MAX_SEED, strict=True,
                             description="Optional chord-sampling seed, split by section")

    @field_validator("genre")
    @classmethod
    def _known_genre(cls, v: str) -> str:
        if v not in GENRE_DEFAULTS:
            raise ValueError(f"unknown genre {v!r}; expected one of: {', '.join(GENRE_DEFAULTS)}")
        return v

    @field_validator("key")
    @classmethod
    def _key_format(cls, v: str | None) -> str | None:
        if v is None:
            return None
        parts = v.strip().split()
        key = f"{parts[0][:1].upper()}{parts[0][1:]} {parts[1].lower()}" if len(parts) == 2 else v
        if not _KEY.fullmatch(key):
            raise ValueError("expected key format: 'C major' or 'A minor'")
        return key

    @model_validator(mode="after")
    def _sketch_and_defaults(self) -> "PlanRequest":
        if [m.index for m in self.measures] != list(range(len(self.measures))):
            raise ValueError("measures must be the whole sketch in order, indexed 0..n-1")
        if len(self.measures) + MIN_PLAN_SECTION_BARS > self.targetBars:
            raise ValueError(f"targetBars must leave at least {MIN_PLAN_SECTION_BARS} bars after the "
                             f"{len(self.measures)}-bar sketch")
        end = 0
        for section in self.sections:
            if section.start < end:
                raise ValueError("sections must be in order and must not overlap")
            if section.start + section.length > len(self.measures):
                raise ValueError("sections must lie within the sketch")
            end = section.start + section.length
        defaults = GENRE_DEFAULTS[self.genre]
        self.key = self.key or defaults["key"]
        self.bpm = self.bpm or defaults["bpm"]
        self.timeSignature = self.timeSignature or tuple(defaults["time_signature"])
        return self


# --- The plan (the rules planner's form, or a plan handed to validate_plan) ---


class PlanCite(BaseModel):
    """A sketch bar and one of its Roman numerals that a section's reason relies on."""
    model_config = ConfigDict(extra="forbid")
    bar: int
    roman: str = Field(max_length=32)


class PlannedSection(BaseModel):
    """type and cadence stay plain strings so validate_plan, not the parser, reports a
    wrong vocabulary; a long reason is likewise an issue, not a 422."""
    model_config = ConfigDict(extra="forbid")
    type: str = Field(max_length=40)
    bars: int
    cadence: str = Field(max_length=40)
    reason: str = Field(default="", max_length=2000)
    cites: list[PlanCite] = Field(default_factory=list, max_length=MAX_CITES)


class SongPlan(BaseModel):
    """The sections that follow the sketch, in order; the sketch's own sections are not repeated."""
    model_config = ConfigDict(extra="forbid")
    sections: list[PlannedSection] = Field(max_length=MAX_SECTION_BARS)


class PlanIssue(BaseModel):
    """One validate_plan finding. code: empty | section_type | bars | cadence | reason_length |
    intro_first | outro_last | outro_wanted | outro_unwanted | final_cadence | total_bars |
    cite_bar | cite_roman. section indexes plan.sections; null for the whole plan."""
    model_config = ConfigDict(extra="forbid")
    code: str
    section: int | None = None
    message: str


# --- The trace ---


class ValidationAttempt(BaseModel):
    attempt: int
    ok: bool
    errors: list[PlanIssue]


class ChorusReuse(BaseModel):
    origin: Literal["sketch", "section"]
    sectionIndex: int | None = None
    startMeasure: int
    source: Literal["sketch", "ai", "rules"]
    modelUsed: str | None = None
    retainedSlots: list[int]
    changedSlots: list[int]


class PlanSectionResult(SectionGenerateResponse):
    """A planned section as /api/generate/section returned it, plus the planner's
    cadence and the slots the song-form rules decided."""
    cadence: CadenceType
    rulesSlots: list[int] = Field(description="Section-relative chord slot indices (0..chordCount-1) "
                                              "written by the rules, not the model")
    reason: str
    cites: list[PlanCite]
    regenerated: bool = False
    source: Literal["ai", "rules", "reuse"]
    reuse: ChorusReuse | None = None


class PlanError(BaseModel):
    """kind: timeout (a time budget of the run was passed) | invalid (the genre-default
    form failed validation) | generation (the chord model failed, or the run was cancelled)."""
    kind: Literal["timeout", "invalid", "generation"]
    message: str


class PlanTrace(BaseModel):
    """Everything a run did, in order. Stateless: nothing is applied server-side."""
    status: Literal["ok", "error"]
    genre: str
    key: str
    bpm: int
    timeSignature: tuple[int, int]
    plan: SongPlan | None
    validation: list[ValidationAttempt]
    sections: list[PlanSectionResult]
    elapsedMs: int
    generationMs: int            # wall clock spent in the chord model
    error: PlanError | None = None
