"""Message contracts (spec section 1.5): what crosses a process or component boundary.

Frozen pydantic models. The queue messages between kin-audio, kin-core and
kin-ui are plain dicts (cheap to pickle); the models here are what the spec
names, and every message that leaves a process boundary in a typed way
(the gateway's requests and responses above all) is validated through them.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, conint, constr, field_validator


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class SpeechEvent(_Frozen):  # E10  kin-audio -> kin-core
    seq: int
    start_offset: int  # sample index in the shared ring
    end_offset: int
    mode: Literal["ambient", "command"]
    t_start: float  # unix time
    t_end: float


class Transcript(_Frozen):  # E11, E36
    seq: int
    text: str
    confidence: float
    speaker: Literal["you", "other", "kin", "unknown"] = "unknown"
    t_start: float
    t_end: float


class Episode(_Frozen):  # E14, E15
    id: str  # "e-<int>"
    text: str
    t_start: float
    t_end: float
    place: str | None = None  # manual label from the UI
    speakers: list[str] = []


class TaskCandidate(_Frozen):  # E17
    what: str
    due: datetime | None = None
    source_episode: str
    confirmed: bool = False


class ContextEvent(_Frozen):  # E31
    on_ac: bool
    battery_pct: int | None
    idle_s: int
    in_meeting: bool


class Intent(_Frozen):  # E20
    label: str  # one of the Sheet 6 intents
    score: float
    runner_up: str | None = None
    runner_up_score: float | None = None


class FactRequest(BaseModel):  # E25 (Sheet 4): the only thing that may leave the laptop
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["weather", "search", "news"]
    lat: float | None = None  # rounded to 0.1 degree
    lon: float | None = None
    hours: conint(ge=0, le=48) = 24
    q: constr(max_length=80) | None = None  # PII-scrubbed
    topic: Literal["local", "tech", "business", "sport"] | None = None

    @field_validator("lat", "lon")
    @classmethod
    def _round(cls, v: float | None) -> float | None:
        return None if v is None else round(v, 1)

    def payload(self) -> dict:
        """Exactly the fields that are sent for this kind."""
        keep = {"weather": ("kind", "lat", "lon", "hours"), "search": ("kind", "q"), "news": ("kind", "topic")}[
            self.kind
        ]
        return {k: getattr(self, k) for k in keep}


class LedgerEntry(_Frozen):  # E26
    ts: float
    host: str
    payload: str  # exact bytes sent, as text
    bytes_out: int
    bytes_in: int
    status: str


class FactResponse(_Frozen):  # E25 reply
    ok: bool
    kind: str
    data: dict  # fixed structs: weather fields, 5 headlines, 5 snippets
    error: Literal["offline", "timeout", "rejected", None] = None
    ledger: LedgerEntry | None = None  # None only when no call was even attempted


class Answer(_Frozen):  # E29, E30
    text: str
    voice_text: str
    citations: list[str]  # episode ids / "forecast"
    abstained: bool


class StatusEvent(_Frozen):  # E37
    state: Literal["muted", "listening", "capturing", "thinking", "fetching", "error"]


class MuteState(_Frozen):  # E05
    muted: bool
    source: Literal["clip", "os", "ui"]
