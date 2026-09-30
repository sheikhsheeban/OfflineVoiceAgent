"""kin-audio: capture into the shared ring and emit speech events to kin-core.

This process has no network and writes no files. It owns the only
time-critical code in Kin (the PortAudio callback), which is why it does not
share an interpreter, and a GIL, with the models in kin-core.
"""

from __future__ import annotations

import logging
import time

from kin.audio.ring import Ring
from kin.config import FRAME, AudioSettings, load_model_spec, model_path

log = logging.getLogger("kin.audio")


def run(ring_name: str, events_q, stop_evt, settings: AudioSettings, source: str, speed: float = 1.0) -> None:
    from kin.logs import setup

    setup("kin-audio")
    events_q.cancel_join_thread()

    from kin.audio.capture import MicSource, WavSource
    from kin.audio.detector import Detector
    from kin.audio.vad import SileroVAD
    from kin.audio.wakeword import WakeWord

    ring = Ring.attach(ring_name)
    spec = load_model_spec()
    vad = SileroVAD(model_path("vad", "silero_vad.onnx"))
    wake = WakeWord.from_spec(settings.wake_threshold)
    det = Detector(ring, vad, wake, settings)

    if source == "mic":
        src = MicSource(ring, settings.device)
    else:
        src = WavSource(ring, source, speed=speed)
    cursor = ring.write_pos  # only process audio captured from now on
    src.start()
    log.info("listening on %s (wake word: %s)", src.device_name, spec["wake_word"]["phrase"])
    events_q.put({"type": "audio_ready", "device": src.device_name, "t": time.time()})

    status_every = 1.0 / settings.status_hz
    next_status = time.monotonic()
    eos_sent = False
    try:
        while not stop_evt.is_set():
            wp = ring.write_pos
            if wp - cursor > ring.capacity // 2:
                # Detector fell far behind (machine stalled): skip ahead rather than overrun.
                skipped = wp - FRAME - cursor
                cursor = wp - FRAME
                log.warning("detector behind by %.1f s, skipping", skipped / 16000)
            if wp - cursor >= FRAME:
                for ev in det.process(cursor, ring.read(cursor, cursor + FRAME)):
                    events_q.put(ev)
                cursor += FRAME
                continue

            now = time.monotonic()
            if now >= next_status:
                st = det.status()
                st["overflows"] = src.overflows
                events_q.put(st)
                next_status = now + status_every

            if src.finished.is_set() and not eos_sent:
                if isinstance(src, WavSource):
                    for ev in det.flush(cursor):
                        events_q.put(ev)
                    events_q.put({"type": "eos", "t": time.time()})
                    eos_sent = True
                else:
                    raise RuntimeError("input device stopped (unplugged?)")
            time.sleep(0.01)
    finally:
        src.stop()
        ring.close()
        log.info("stopped")
