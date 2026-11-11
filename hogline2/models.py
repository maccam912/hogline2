import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class End(Model):
    number: int = Field(ge=1, le=99)
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    evidence: str = ""
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def ordered(self):
        if self.end_seconds - self.start_seconds < 1:
            raise ValueError("An end must span at least one second")
        return self


def validate_ends(ends, duration):
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Source duration is unavailable")
    parsed = [End.model_validate(e) for e in ends]
    if not parsed:
        raise ValueError("Add at least one end")
    for i, end in enumerate(parsed):
        if end.number != i + 1:
            raise ValueError("End numbers must be consecutive from 1")
        if end.end_seconds > duration + 0.001:
            raise ValueError("An end extends beyond the recording")
        if i and end.start_seconds < parsed[i - 1].end_seconds:
            raise ValueError("Requested end intervals cannot overlap")
    return [e.model_dump() for e in parsed]


class FrameEvent(Model):
    seconds: float
    phase: Literal[
        "before_game", "playing", "deciding", "clearing", "between_ends", "after_game", "uncertain"
    ]
    delivery_view: Literal["left", "right", "other", "unknown"]
    note: str


class Survey(Model):
    frames: list[FrameEvent]


class DraftEnd(Model):
    first_delivery_guess: float
    clearing_finished_guess: float
    evidence: str
    partial_start: bool
    partial_end: bool


class Draft(Model):
    ends: list[DraftEnd]
    warnings: list[str]


class Boundary(Model):
    earliest_seconds: float
    latest_seconds: float
    found: bool
    evidence: str
