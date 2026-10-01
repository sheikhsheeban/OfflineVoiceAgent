"""Waiting on the supervisor's shared stop event without deadlocking it.

`multiprocessing.Event.wait()` registers the caller as a sleeper on a shared condition;
if that process is then killed (a crash, exactly what the supervisor exists to survive),
the supervisor's later `stop_evt.set()` waits forever for the dead sleeper. Polling
`is_set()` never registers, so a crashed child cannot block shutdown.
"""

from __future__ import annotations

import time


def wait_for(event, timeout: float | None = None, poll: float = 0.05) -> bool:
    """Like Event.wait(): True as soon as the event is set, False if `timeout` passes first."""
    end = None if timeout is None else time.monotonic() + timeout
    while not event.is_set():
        if end is not None and time.monotonic() >= end:
            return False
        time.sleep(poll)
    return True
