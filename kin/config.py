"""Runtime settings shared by every Kin process.

The profile decides thread and memory budgets. `phone` mirrors the Android
limits (4 threads in kin-core, 1.1 GB resident ceiling); `laptop` is looser.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent.parent
SPEC_DIR = Path(os.environ.get("KIN_SPEC_DIR", ROOT / "spec"))
MODELS_DIR = Path(os.environ.get("KIN_MODELS_DIR", ROOT / "models"))
RUNS_DIR = Path(os.environ.get("KIN_RUNS_DIR", ROOT / "runs"))

SAMPLE_RATE = 16_000
FRAME = 512  # 32 ms: the Silero v5 window at 16 kHz (VAD and gate work in these)
CHUNK = 1280  # 80 ms: the capture block, and the openWakeWord step
RING_SECONDS = 30

# Libraries must never phone home (privacy invariant, sheet 9).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")


@dataclass(frozen=True)
class Profile:
    name: str
    core_threads: int  # total compute threads kin-core may use
    asr_threads: int
    core_rss_limit_mb: int | None  # fail loudly above this


def _physical_cores() -> int:
    return psutil.cpu_count(logical=False) or os.cpu_count() or 4


def get_profile(name: str | None = None) -> Profile:
    name = name or os.environ.get("KIN_PROFILE", "phone")
    if name == "phone":
        return Profile("phone", core_threads=4, asr_threads=2, core_rss_limit_mb=1100)
    if name == "laptop":
        cores = max(1, _physical_cores() - 2)
        return Profile("laptop", core_threads=cores, asr_threads=min(4, cores), core_rss_limit_mb=None)
    raise ValueError(f"unknown profile {name!r} (expected 'phone' or 'laptop')")


@dataclass(frozen=True)
class AudioSettings:
    device: str | int | None = None  # name substring or index; None = OS default
    gate_margin_db: float = 9.0  # open the gate this far above the noise floor
    gate_min_dbfs: float = -55.0  # never open below this absolute level
    gate_hangover_ms: int = 800
    vad_on: float = 0.5
    vad_off: float = 0.35
    speech_start_ms: int = 64  # speech must last this long to start a segment
    speech_end_ms: int = 600  # silence that closes a segment
    pre_roll_ms: int = 300
    min_segment_ms: int = 250
    max_segment_ms: int = 20_000
    wake_threshold: float = 0.5
    wake_patience: int = 2  # consecutive 80 ms chunks at or above the threshold
    wake_refractory_ms: int = 2_000
    wake_span_ms: int = 1_300  # how far back from detection the wake phrase is taken to start
    wake_noise_suppression: bool = True  # SpeexDSP, where the speexdsp_ns package is available
    command_wait_ms: int = 1_500  # silence allowed after the wake word before the command starts
    command_max_ms: int = 8_000  # a command ends this long after the wake word, whatever happens
    status_hz: float = 5.0


@dataclass(frozen=True)
class Settings:
    profile: Profile = field(default_factory=get_profile)
    audio: AudioSettings = field(default_factory=AudioSettings)
    ui_host: str = "127.0.0.1"
    ui_port: int = 8765
    memory_db: str | None = None  # default: data/kin.db
    asr_model: str | None = None  # default: per profile, see asr_model_name()
    calendar_path: str | None = None  # default: data/calendar.ics if present (or $KIN_CALENDAR)
    warm_llm: bool = False  # load the language model at start-up instead of on first use
    voice: bool | None = None  # spoken replies; None = on for the microphone, off for replays
    use_clip: bool = True  # prefer the Kin clip's microphone when it is plugged in
    serial_port: str | None = None  # the clip's serial port; None = find it
    gateway: bool = True  # start kin-gateway (the only process that may go online)


def load_model_spec() -> dict:
    with open(SPEC_DIR / "models.toml", "rb") as f:
        return tomllib.load(f)


def default_calendar_path() -> str | None:
    env = os.environ.get("KIN_CALENDAR")
    if env:
        return env
    p = Path(os.environ.get("KIN_DATA_DIR", ROOT / "data")) / "calendar.ics"
    return str(p) if p.exists() else None


def asr_model_name(profile: str, override: str | None = None) -> str:
    """--asr beats $KIN_ASR beats the profile default in spec/models.toml."""
    spec = load_model_spec()
    name = override or os.environ.get("KIN_ASR") or spec["asr"][profile]
    if name not in spec["asr_models"]:
        raise ValueError(f"unknown ASR model {name!r}; choose from {', '.join(spec['asr_models'])}")
    return name


def model_path(section: str, filename: str) -> Path:
    spec = load_model_spec()[section]
    return MODELS_DIR / spec["dir"] / filename
