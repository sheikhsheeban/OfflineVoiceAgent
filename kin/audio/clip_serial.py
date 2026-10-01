"""A6 clip serial link: the clip's mute switch in, its status LED out.

The clip (an ESP32-S3) is a USB microphone and also a USB serial port. Its
protocol is one ASCII line per message (firmware/README.md):

  clip -> host   `M1`  the mute switch is on (the mic has no power)
                 `M0`  the mute switch is off
                 `HELLO kin-clip 1`  sent when the port opens
  host -> clip   `L<pattern>`  set the status LED, e.g. `Lgreen_solid`

The switch state comes from the clip's hardware (E05); this code only reports
it. Nothing here can turn the microphone back on: with the switch on, the mic
has no power (HW2), and kin-audio does not try.
"""

from __future__ import annotations

import logging
import threading
import time

log = logging.getLogger("kin.clip")

CLIP_NAME = "Kin Clip"  # the USB product string of the clip; also its audio device name
USB_VID_PID = ("303A", None)  # Espressif; any product


def parse_line(line: str) -> dict | None:
    """One line from the clip -> {"muted": bool} for mute reports, {"hello": version} for the greeting."""
    line = line.strip()
    if line == "M1":
        return {"muted": True}
    if line == "M0":
        return {"muted": False}
    if line.startswith("HELLO kin-clip"):
        return {"hello": line.split()[-1]}
    return None


def led_command(pattern: str) -> bytes:
    return f"L{pattern}\n".encode("ascii")


class MuteLatch:
    """Who muted the microphone, and who may lift it (flow F4).

    The clip's switch is hardware: while it is on, only the switch can turn the microphone
    back on. Nothing else in Kin, the UI included, may try to. A UI mute (the laptop-mic
    fallback) can be lifted from the UI.
    """

    def __init__(self):
        self.muted = False
        self.source: str | None = None

    def request(self, muted: bool, who: str) -> bool:
        """Apply a mute or unmute request; returns True if the state changed."""
        if muted == self.muted:
            return False
        if not muted and self.source == "clip" and who != "clip":
            return False  # refused: the switch is the only thing that unmutes the clip
        self.muted, self.source = muted, (who if muted else None)
        return True


def find_port() -> str | None:
    """The clip's serial port: by USB name, else by Espressif's vendor id."""
    from serial.tools import list_ports

    for p in list_ports.comports():
        if CLIP_NAME.lower() in (p.product or p.description or "").lower():
            return p.device
        if p.vid is not None and f"{p.vid:04X}" == USB_VID_PID[0]:
            return p.device
    return None


class ClipLink:
    """Reads mute reports and writes LED patterns. `opener(port) -> serial-like` is replaceable for tests."""

    def __init__(self, on_mute, port: str | None = None, opener=None):
        self.on_mute = on_mute  # called with (muted: bool)
        self.port = port
        self._open = opener or self._open_real
        self._ser = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="clip-serial", daemon=True)
        self.connected = False
        self.muted: bool | None = None
        self.pending_led: bytes | None = None

    @staticmethod
    def _open_real(port: str):
        import serial

        return serial.Serial(port, 115200, timeout=0.2)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2)

    def set_led(self, pattern: str) -> None:
        data = led_command(pattern)
        with self._lock:
            self.pending_led = data  # the latest pattern is re-sent after a reconnect
            if self._ser is not None:
                try:
                    self._ser.write(data)
                except Exception:  # noqa: BLE001 - the read loop notices the loss and reconnects
                    pass

    def _run(self) -> None:
        while not self._stop.is_set():
            port = self.port or find_port()
            if port is None:
                self._stop.wait(2.0)  # no clip: the laptop's own mic is in use (CL4)
                continue
            try:
                ser = self._open(port)
            except Exception:  # noqa: BLE001
                self._stop.wait(2.0)
                continue
            with self._lock:
                self._ser = ser
                self.connected = True
                if self.pending_led:
                    ser.write(self.pending_led)
            log.info("clip connected on %s", port)
            try:
                buf = b""
                while not self._stop.is_set():
                    chunk = ser.read(64)
                    if not chunk:
                        continue
                    buf += chunk
                    while b"\n" in buf:
                        raw, buf = buf.split(b"\n", 1)
                        msg = parse_line(raw.decode("ascii", errors="ignore"))
                        if msg and "muted" in msg:
                            self.muted = msg["muted"]
                            self.on_mute(msg["muted"])
            except Exception:  # noqa: BLE001 - unplugged
                log.warning("clip serial lost")
            finally:
                with self._lock:
                    self._ser = None
                    self.connected = False
                try:
                    ser.close()
                except Exception:  # noqa: BLE001
                    pass
            time.sleep(0.5)
