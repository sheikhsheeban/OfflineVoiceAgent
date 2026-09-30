"""Needs a real microphone: `uv run pytest -m mic`."""

import time

import pytest

from kin.audio.capture import MicSource
from kin.audio.ring import Ring

pytestmark = pytest.mark.mic


def test_microphone_fills_ring_in_real_time():
    ring = Ring.create(16000 * 5)
    src = MicSource(ring)
    try:
        src.start()
        time.sleep(0.3)
        p0, t0 = ring.write_pos, time.perf_counter()
        time.sleep(2.0)
        rate = (ring.write_pos - p0) / (time.perf_counter() - t0)
    finally:
        src.stop()
        ring.close()
    assert 15000 < rate < 17000, f"captured at {rate:.0f} samples/s"
    assert src.overflows == 0
