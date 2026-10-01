"""P2 speaker tag: is this you or someone else?

A small speaker-embedding model (WeSpeaker ResNet34 on sherpa-onnx) turns each
speech segment into a 256-number voiceprint. The owner's voiceprint is learned
once, from about ten seconds of their speech (`start_enrolment`), and kept in
the encrypted store. A segment is tagged `you` when it is close enough to that
print, `other` when it is not, and `unknown` when there is no print yet or the
segment is too short to judge.

The voiceprint is derived data about the owner's voice, never audio; it is
deleted with everything else on a wipe.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from kin.config import SAMPLE_RATE, model_path

log = logging.getLogger("kin.speaker")

META_KEY = "owner_voice"
MIN_SEGMENT_S = 0.8  # shorter than this says too little about who is speaking
MIN_THRESHOLD = 0.5  # cosine similarity: never accept a voice this far from the print
MAX_THRESHOLD = 0.75
ENROL_SECONDS = 10.0


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n else v


class SpeakerTagger:
    def __init__(self, threads: int = 1, path: Path | None = None, get_meta=None, set_meta=None):
        self.path = Path(path or model_path("speaker", "wespeaker_en_voxceleb_resnet34.onnx"))
        self.threads = threads
        self._get, self._set = get_meta, set_meta
        self._extractor = None
        self.owner: np.ndarray | None = None
        self.threshold = MIN_THRESHOLD
        self._enrol: list[tuple[np.ndarray, float]] | None = None  # (embedding, seconds) while enrolling
        self.enrol_seconds = 0.0
        self.load_owner()

    def installed(self) -> bool:
        return self.path.exists()

    def load_owner(self) -> None:
        raw = self._get(META_KEY) if self._get else None
        if not raw:
            self.owner = None
            return
        saved = json.loads(raw)
        self.owner = _unit(np.array(saved["print"], dtype=np.float32))
        self.threshold = float(saved["threshold"])

    @property
    def enrolled(self) -> bool:
        return self.owner is not None

    @property
    def enrolling(self) -> bool:
        return self._enrol is not None

    def _ext(self):
        if self._extractor is None:
            from kin.native import preload_onnxruntime

            preload_onnxruntime()  # Windows: not the old System32 onnxruntime.dll
            import sherpa_onnx

            cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(self.path), num_threads=self.threads)
            self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
        return self._extractor

    def embed(self, audio: np.ndarray) -> np.ndarray | None:
        """A unit-length voiceprint, or None when the segment is too short."""
        if len(audio) < MIN_SEGMENT_S * SAMPLE_RATE:
            return None
        ext = self._ext()
        stream = ext.create_stream()
        stream.accept_waveform(SAMPLE_RATE, (audio.astype(np.float32) / 32768.0))
        stream.input_finished()
        if not ext.is_ready(stream):
            return None
        return _unit(np.array(ext.compute(stream), dtype=np.float32))

    # -- enrolment --------------------------------------------------------------------

    def start_enrolment(self) -> None:
        self._enrol = []
        self.enrol_seconds = 0.0

    def cancel_enrolment(self) -> None:
        self._enrol = None

    def enrolment_progress(self) -> float:
        return sum(s for _, s in (self._enrol or []))

    def clear_owner(self) -> None:
        self.owner = None
        self._enrol = None
        if self._set:
            self._set(META_KEY, "")

    def _finish_enrolment(self) -> None:
        embs, weights = zip(*self._enrol, strict=True)
        self.owner = _unit(np.average(np.stack(embs), axis=0, weights=weights).astype(np.float32))
        # How alike the owner's own segments are decides how strict "you" is: a steady voice in a
        # quiet room gets a high bar, a noisy enrolment a lower one (never below MIN_THRESHOLD).
        own = float(np.mean([e @ self.owner for e in embs]))
        self.threshold = round(min(MAX_THRESHOLD, max(MIN_THRESHOLD, 0.75 * own)), 3)
        self._enrol = None
        if self._set:
            saved = {"print": [round(float(x), 6) for x in self.owner], "threshold": self.threshold}
            self._set(META_KEY, json.dumps(saved))
        log.info("owner voice enrolled")

    # -- tagging ----------------------------------------------------------------------

    def tag(self, audio: np.ndarray) -> tuple[str, float | None]:
        """("you" | "other" | "unknown", similarity). While enrolling, segments teach the print."""
        emb = self.embed(audio)
        if emb is None:
            return "unknown", None
        if self._enrol is not None:
            self._enrol.append((emb, len(audio) / SAMPLE_RATE))
            self.enrol_seconds = self.enrolment_progress()
            if self.enrolment_progress() >= ENROL_SECONDS:
                self._finish_enrolment()
            return "you", None
        if self.owner is None:
            return "unknown", None
        score = float(emb @ self.owner)
        return ("you" if score >= self.threshold else "other"), round(score, 3)
