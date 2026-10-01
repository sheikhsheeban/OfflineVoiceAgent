"""kin-audio: capture into the shared ring and emit speech events to kin-core.

This process has no network and writes no files. It owns the only
time-critical code in Kin (the PortAudio callback), which is why it does not
share an interpreter, and a GIL, with the models in kin-core.

Mute (flow F4). The clip's switch cuts the microphone's power in hardware and
the clip reports it over serial (A6, E05). The UI can also mute, as a fallback
that works with the laptop's own mic. Either way this process stops capturing
and does nothing else; it never tries to turn a hardware mute back off. Only a
mute that came from the UI can be lifted from the UI.
"""

from __future__ import annotations

import logging
import os
import queue
import time

from kin.audio.ring import Ring
from kin.config import CHUNK, AudioSettings, load_model_spec, model_path

log = logging.getLogger("kin.audio")


def run(
    ring_name: str,
    events_q,
    stop_evt,
    settings: AudioSettings,
    source: str,
    speed: float = 1.0,
    cmd_q=None,
    serial_port: str | None = None,
    use_clip: bool = True,
) -> None:
    from kin import guard

    guard.install()  # before anything else: kin-audio has no network (INV3)

    from kin.logs import setup

    setup("kin-audio")
    events_q.cancel_join_thread()

    from kin.audio.capture import MicSource, WavSource, fall_back_to_laptop_mic
    from kin.audio.clip_serial import ClipLink, MuteLatch
    from kin.audio.detector import Detector
    from kin.audio.endpoint import Endpointer
    from kin.audio.vad import SileroVAD
    from kin.audio.wakeword import WakeWord

    ring = Ring.attach(ring_name)
    spec = load_model_spec()
    vad = SileroVAD(model_path("vad", "silero_vad.onnx"))
    wake = WakeWord.from_spec(settings.wake_threshold, settings.wake_noise_suppression)
    if settings.wake_noise_suppression and not wake.noise_suppression:
        log.info("wake word noise suppression unavailable here (needs the speexdsp_ns package)")
    det = Detector(ring, vad, wake, settings)

    if source == "mic":
        src = MicSource(ring, settings.device, prefer_clip=use_clip)
    else:
        src = WavSource(ring, source, speed=speed)
    cursor = ring.write_pos  # only process audio captured from now on

    latch = MuteLatch()

    def set_muted(on: bool, who: str) -> None:
        if not latch.request(on, who):
            if not on and latch.source == "clip":
                log.info("ignoring %s unmute: the clip's switch is on", who)
            return
        events_q.put({"type": "mute", "muted": on, "source": who, "t": time.time()})
        if on:
            det.endpoint = Endpointer(settings)  # an unfinished utterance is dropped, not transcribed
            wake.reset()
            if isinstance(src, MicSource):
                src.pause()
            log.info("muted by %s; nothing is being heard", who)
        else:
            if isinstance(src, MicSource):
                src.resume()
            log.info("unmuted by %s", who)

    link = None
    if source == "mic" and use_clip:
        link = ClipLink(lambda m: set_muted(m, "clip"), port=serial_port)
        link.start()

    src.start()
    probe = guard.probe()
    log.info("listening on %s (wake word: %s, threshold %.2f, noise suppression %s; network: %s)",
             src.device_name, spec["wake_word"]["phrase"], settings.wake_threshold,
             "on" if wake.noise_suppression else "off", probe["method"])  # fmt: skip
    events_q.put({"type": "audio_ready", "device": src.device_name, "t": time.time(), "net": probe,
                  "clip": bool(getattr(src, "is_clip", False)),
                  "env": {k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "DO_NOT_TRACK")}})  # fmt: skip

    status_every = 1.0 / settings.status_hz
    next_status = time.monotonic()
    eos_sent = False
    try:
        while not stop_evt.is_set():
            if cmd_q is not None:
                try:
                    while True:
                        cmd = cmd_q.get_nowait()
                        if cmd.get("type") == "mute":
                            set_muted(bool(cmd["muted"]), cmd.get("source", "ui"))
                        elif cmd.get("type") == "led" and link is not None:
                            link.set_led(cmd["pattern"])
                except queue.Empty:
                    pass

            wp = ring.write_pos
            if wp - cursor > ring.capacity // 2:
                # Detector fell far behind (machine stalled): skip ahead rather than overrun.
                skipped = wp - CHUNK - cursor
                cursor = wp - CHUNK
                log.warning("detector behind by %.1f s, skipping", skipped / 16000)
            if wp - cursor >= CHUNK:
                # Every 80 ms chunk goes to the wake word and, in 512-sample windows, to gate and VAD.
                for ev in det.process(cursor, ring.read(cursor, cursor + CHUNK)):
                    events_q.put(ev)
                cursor += CHUNK
                continue

            now = time.monotonic()
            if now >= next_status:
                st = det.status()
                st["overflows"] = src.overflows
                st["muted"] = latch.muted
                events_q.put(st)
                next_status = now + status_every

            if src.finished.is_set() and not eos_sent:
                if isinstance(src, WavSource):
                    for ev in det.flush(cursor):
                        events_q.put(ev)
                    events_q.put({"type": "eos", "t": time.time()})
                    eos_sent = True
                elif latch.muted:
                    src.finished.clear()  # the mic lost its power because of the switch: expected
                elif getattr(src, "is_clip", False):
                    # Clip unplugged mid-demo: carry on with the laptop's own mic and say so.
                    src, ev = fall_back_to_laptop_mic(src, ring)
                    cursor = ring.write_pos
                    events_q.put(ev)
                    log.warning("clip lost; using %s", src.device_name)
                else:
                    raise RuntimeError("input device stopped (unplugged?)")
            time.sleep(0.01)
    finally:
        if link is not None:
            link.stop()
        src.stop()
        ring.close()
        log.info("stopped")
