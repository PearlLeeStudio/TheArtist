"""Validated editor snapshot for MIDI rendering and local DAW handoff."""

import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.services.genre_tables import DRUM_PATTERNS, HARMONY_INSTRUMENTS, HARMONY_RHYTHM_PRESETS

MAX_EXPORT_MEASURES = 512
MAX_EXPORT_NOTES = 100_000
MAX_EXPORT_BYTES = 4 * 1024 * 1024
MIN_MIDI_BYTES = 14  # a MIDI file holds at least its 14-byte MThd header chunk
MIDI_PITCH = Annotated[int, Field(ge=0, le=127)]
VOLUME = Annotated[float, Field(ge=0, le=1)]
_ROOT = r"[A-G](?:#|b)?"
# A numeric extension such as 6/9 is part of the quality, not a slash bass.
_CHORD = re.compile(rf"(?P<root>{_ROOT})(?P<quality>[^/\s]*(?:/[0-9]+)?)(?:/(?P<bass>{_ROOT}))?")
_KEY = re.compile(rf"({_ROOT}) (major|minor)")
_RHYTHM_IDS = {preset["id"] for preset in HARMONY_RHYTHM_PRESETS}
_BASS_OCTAVE_C = 36  # C2; exported bass notes are C2..B2
NOTE_PC = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5,
           "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10,
           "B": 11, "Cb": 11, "B#": 0, "Fb": 4, "E#": 5}


class ExportModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)


class ExportChord(ExportModel):
    symbol: str = Field(min_length=1, max_length=80)
    voicing: list[MIDI_PITCH] = Field(min_length=1, max_length=16)

    @field_validator("symbol")
    @classmethod
    def valid_symbol(cls, value: str) -> str:
        if not _CHORD.fullmatch(value):
            raise ValueError("symbol must start with a note name and use a valid slash bass")
        return value

    @property
    def bass_pitch(self) -> int:
        """MIDI pitch of the slash bass, else the root, in the C2 octave."""
        match = _CHORD.fullmatch(self.symbol)
        assert match is not None  # validated above
        return _BASS_OCTAVE_C + NOTE_PC[match["bass"] or match["root"]]


class ExportMeasure(ExportModel):
    chords: list[ExportChord | None] = Field(min_length=2, max_length=2)


class ExportSection(ExportModel):
    label: str = Field(min_length=1, max_length=80)
    startMeasure: int = Field(ge=0, lt=MAX_EXPORT_MEASURES)
    length: int = Field(ge=1, le=MAX_EXPORT_MEASURES)


class ExportPlay(ExportModel):
    harmonyOverride: str | None
    harmonyRhythmId: str | None
    bassEnabled: bool
    drumMode: Literal["auto", "manual"]
    drumPatternId: int | None
    masterVolume: VOLUME
    harmonyVolume: VOLUME
    bassVolume: VOLUME
    drumVolume: VOLUME

    @field_validator("harmonyOverride")
    @classmethod
    def known_instrument(cls, value: str | None) -> str | None:
        if value is not None and value not in HARMONY_INSTRUMENTS:
            raise ValueError("unknown harmony instrument")
        return value

    @field_validator("harmonyRhythmId")
    @classmethod
    def known_rhythm(cls, value: str | None) -> str | None:
        if value is not None and value not in _RHYTHM_IDS:
            raise ValueError("unknown harmony rhythm")
        return value

    @field_validator("drumPatternId")
    @classmethod
    def known_drums(cls, value: int | None) -> int | None:
        if value is not None and value not in DRUM_PATTERNS:
            raise ValueError("unknown drum pattern")
        return value


class ExportSpec(ExportModel):
    title: str = Field(max_length=120)
    bpm: int = Field(ge=20, le=300)
    key: str = Field(min_length=1, max_length=40)
    timeSignature: list[int] = Field(min_length=2, max_length=2)
    genre: Annotated[str, Field(max_length=40)] | None
    measures: list[ExportMeasure] = Field(min_length=1, max_length=MAX_EXPORT_MEASURES)
    sections: list[ExportSection] = Field(max_length=128)
    play: ExportPlay

    @field_validator("key")
    @classmethod
    def valid_key(cls, value: str) -> str:
        if not _KEY.fullmatch(value):
            raise ValueError("key must be a note name followed by major or minor")
        return value

    @field_validator("timeSignature")
    @classmethod
    def valid_meter(cls, value: list[int]) -> list[int]:
        if not 1 <= value[0] <= 32 or value[1] not in (1, 2, 4, 8, 16, 32):
            raise ValueError("timeSignature needs numerator 1..32 and denominator 1, 2, 4, 8, 16 or 32")
        return value

    @model_validator(mode="after")
    def section_ranges(self) -> "ExportSpec":
        end = 0
        for section in sorted(self.sections, key=lambda part: part.startMeasure):
            if section.startMeasure < end or section.startMeasure + section.length > len(self.measures):
                raise ValueError("sections must be non-overlapping and within the measures")
            end = section.startMeasure + section.length
        return self


class DawOpenRequest(ExportModel):
    target: Literal["logic", "ableton"]
    spec: ExportSpec


class DawTarget(ExportModel):
    id: Literal["logic", "ableton"]
    name: str
    available: bool
    action: Literal["open", "reveal"]


class DawTargetsResponse(ExportModel):
    targets: list[DawTarget]


class DawOpenResponse(ExportModel):
    action: Literal["open", "reveal"]
    path: str
