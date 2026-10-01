"""O1 status controller: one state, shown on screen and on the clip's LED.

States (contract StatusEvent): muted, listening, capturing, thinking, fetching, error.
The highest-priority reason wins: muted beats error beats fetching beats thinking
beats capturing beats listening. Each change is sent to the UI (E36 path) and, as
an LED pattern, to kin-audio which forwards it to the clip over serial (E37).
"""

from __future__ import annotations

import time

from kin.contracts import StatusEvent

PRIORITY = ["muted", "error", "fetching", "thinking", "capturing", "listening"]
# Patterns the firmware understands (firmware/README.md): colour and behaviour.
LED_PATTERN = {
    "muted": "off",
    "listening": "green_solid",
    "capturing": "green_pulse",
    "thinking": "blue_pulse",
    "fetching": "amber_blink",
    "error": "red_solid",
}
CAPTURE_HOLD_S = 6.0  # "capturing" lasts until the command is transcribed, or this long


class StatusController:
    def __init__(self, ui_q, led_q=None, clock=time.monotonic):
        self._ui_q, self._led_q, self._clock = ui_q, led_q, clock
        self._reasons: dict[str, float | bool] = {}
        self.state = "listening"
        self.history: list[str] = []

    def set(self, reason: str, active: bool = True) -> None:
        """`reason` is one of PRIORITY; `active` turns it on or off."""
        if reason not in PRIORITY:
            raise ValueError(reason)
        if active:
            self._reasons[reason] = self._clock()
        else:
            self._reasons.pop(reason, None)
        self._publish()

    def capture_started(self) -> None:
        self.set("capturing")

    def tick(self) -> None:
        t = self._clock()
        if "capturing" in self._reasons and t - self._reasons["capturing"] > CAPTURE_HOLD_S:
            self._reasons.pop("capturing")
            self._publish()

    def _publish(self) -> None:
        new = next((r for r in PRIORITY if r in self._reasons), "listening")
        if new == self.state:
            return
        self.state = new
        self.history.append(new)
        ev = StatusEvent(state=new)
        self._ui_q.put({"type": "kin_state", "state": ev.state, "t": time.time()})
        if self._led_q is not None:
            self._led_q.put({"type": "led", "pattern": LED_PATTERN[new], "state": new})
