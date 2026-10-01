"""C1 context fusion: what the laptop is doing, as one ContextEvent, every 30 s.

Runs as a light timer inside kin-core (spec CL1), never in kin-audio.
  - AC power and battery level: psutil
  - user idle time: the OS's last-input clock on Windows; elsewhere (a headless
    Raspberry Pi has no desktop to ask) the time since Kin last heard speech or
    was used
  - in a meeting: an event in the local calendar covers now
Only the ContextEvent leaves this module; audio never does (E31).
"""

from __future__ import annotations

import datetime as dt
import sys
import time

import psutil

from kin.contracts import ContextEvent
from kin.reasoning.calendar import Calendar

SAMPLE_EVERY_S = 30.0


def _windows_idle_s() -> float | None:
    if sys.platform != "win32":
        return None
    import ctypes

    class LASTINPUTINFO(ctypes.Structure):
        _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]

    info = LASTINPUTINFO()
    info.cbSize = ctypes.sizeof(info)
    if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):
        return None
    return max(0.0, (ctypes.windll.kernel32.GetTickCount() - info.dwTime) / 1000.0)


class ContextFusion:
    def __init__(self, calendar: Calendar | None = None, clock=time.time):
        self.calendar = calendar or Calendar(None)
        self._clock = clock
        self._last_activity = clock()
        self._last: ContextEvent | None = None
        self._next = 0.0

    def touch(self) -> None:
        """Kin heard speech or was used: the user is not idle."""
        self._last_activity = self._clock()

    def sample(self, now: float | None = None) -> ContextEvent:
        now = self._clock() if now is None else now
        batt = None
        try:
            batt = psutil.sensors_battery()
        except Exception:  # noqa: BLE001 - no battery sensor (desktop, Pi)
            pass
        os_idle = _windows_idle_s()
        own_idle = max(0.0, now - self._last_activity)
        idle = os_idle if os_idle is not None else own_idle
        local = dt.datetime.fromtimestamp(now)
        in_meeting = any(not e.all_day and e.start <= local < e.end for e in self.calendar.events())
        return ContextEvent(
            on_ac=True if batt is None else bool(batt.power_plugged),  # no battery: mains power
            battery_pct=None if batt is None else int(batt.percent),
            idle_s=int(min(idle, own_idle) if os_idle is None else idle),
            in_meeting=in_meeting,
        )

    def poll(self, now: float | None = None) -> tuple[ContextEvent | None, ContextEvent]:
        """Sample every 30 s. Returns (event if it changed else None, latest sample)."""
        now = self._clock() if now is None else now
        if now < self._next and self._last is not None:
            return None, self._last
        self._next = now + SAMPLE_EVERY_S
        ev = self.sample(now)
        prev, self._last = self._last, ev
        key = lambda e: (e.on_ac, e.in_meeting, e.battery_pct)  # noqa: E731
        changed = prev is None or key(ev) != key(prev)
        return (ev if changed else None), ev

    @property
    def previous(self) -> ContextEvent | None:
        return self._last
