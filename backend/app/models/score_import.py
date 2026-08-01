"""A score transcription draft, before the user chooses the app's two chord slots."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.models.sections import SectionType

MAX_SCORE_BYTES = 8 * 1024 * 1024
MAX_SCORE_BASE64 = 4 * ((MAX_SCORE_BYTES + 2) // 3)
MAX_SCORE_MEASURES = 512
MAX_SCORE_SECTIONS = 64
MAX_SCORE_WARNINGS = 32
MAX_CHORDS_PER_MEASURE = 16
MAX_PITCH_NAMES = 128
MAX_NOTE_CHARS = 400
MAX_TITLE_CHARS = 120
ScoreMime = Literal["application/pdf", "image/png", "image/jpeg", "image/webp"]
ScoreImportMode = Literal["chord_symbols", "infer_harmony"]
ChordBasis = Literal["printed", "inferred", "unreadable"]
Confidence = Literal["high", "medium", "low"]
ShortText = Annotated[str, Field(max_length=80)]
Note = Annotated[str, Field(max_length=MAX_NOTE_CHARS)]
ChordSymbol = Annotated[str, Field(min_length=1, max_length=80)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ScoreImportRequest(StrictModel):
    filename: str = Field(min_length=1, max_length=180)
    mimeType: str
    dataBase64: str
    mode: ScoreImportMode = "chord_symbols"


class ImportedMeasure(StrictModel):
    chords: list[ChordSymbol] = Field(max_length=MAX_CHORDS_PER_MEASURE)
    sourceLabel: ShortText
    needsReview: bool
    note: Note
    chordBasis: ChordBasis = "printed"
    confidence: Confidence = "high"
    notes: list[ShortText] = Field(default_factory=list, max_length=MAX_PITCH_NAMES)
    writtenNotes: list[ShortText] = Field(default_factory=list, max_length=MAX_PITCH_NAMES)
    bassNotes: list[ShortText] = Field(default_factory=list, max_length=MAX_PITCH_NAMES)
    noChord: bool = False
    timeSignature: list[int] | None = Field(default=None, min_length=2, max_length=2)
    analysisSource: Literal["local"] = "local"
    model: str | None = Field(default=None, max_length=200)

    @field_validator("chords")
    @classmethod
    def nonblank_chords(cls, value: list[str]) -> list[str]:
        if any(not chord.strip() for chord in value):
            raise ValueError("a chord symbol cannot be blank")
        return value

    @model_validator(mode="after")
    def require_review_and_evidence(self) -> "ImportedMeasure":
        if self.noChord and self.chords:
            raise ValueError("a no-chord measure cannot contain chords")
        if not self.chords and not self.noChord:
            self.chordBasis = "unreadable"
            self.confidence = "low"
        elif self.chordBasis == "unreadable":
            raise ValueError("unreadable measures cannot contain proposed chords")
        if self.chordBasis == "inferred" and not self.note.strip():
            raise ValueError("inferred chords need visible-note evidence or a harmonization rationale")
        if self.chordBasis != "printed" or self.confidence == "low":
            self.needsReview = True
        return self


class ImportedSection(StrictModel):
    section: SectionType | None
    label: ShortText
    startMeasure: int = Field(ge=0, lt=MAX_SCORE_MEASURES)
    length: int = Field(ge=1, le=MAX_SCORE_MEASURES)
    basis: Literal["explicit", "inferred"]
    note: Note


class ScoreDraft(StrictModel):
    title: str = Field(max_length=MAX_TITLE_CHARS)
    key: Annotated[str, Field(max_length=40)] | None
    timeSignature: list[int] | None = Field(min_length=2, max_length=2)
    measures: list[ImportedMeasure] = Field(max_length=MAX_SCORE_MEASURES)
    sections: list[ImportedSection] = Field(max_length=MAX_SCORE_SECTIONS)
    warnings: list[Note] = Field(max_length=MAX_SCORE_WARNINGS)

    @field_validator("timeSignature")
    @classmethod
    def valid_meter(cls, value: list[int] | None) -> list[int] | None:
        if value is not None and (not 1 <= value[0] <= 32 or value[1] not in (1, 2, 4, 8, 16, 32)):
            raise ValueError("timeSignature must have a numerator 1..32 and a power-of-two denominator 1..32")
        return value

    @model_validator(mode="after")
    def section_ranges(self) -> "ScoreDraft":
        occupied: set[int] = set()
        for section in self.sections:
            end = section.startMeasure + section.length
            if end > len(self.measures):
                raise ValueError("section extends beyond the imported measures")
            measures = set(range(section.startMeasure, end))
            if occupied & measures:
                raise ValueError("imported sections cannot overlap")
            occupied.update(measures)
        return self


class ScoreImportResponse(ScoreDraft):
    model: str | None
    mode: ScoreImportMode = "chord_symbols"
    analysisSource: Literal["local"] = "local"
