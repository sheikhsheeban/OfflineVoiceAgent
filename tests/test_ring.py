import multiprocessing as mp

import numpy as np
import pytest

from kin.audio.ring import Overrun, Ring


@pytest.fixture
def ring():
    r = Ring.create(1000)
    yield r
    r.close()


def test_write_then_read(ring):
    ring.write(np.arange(300, dtype=np.int16))
    assert ring.write_pos == 300
    assert np.array_equal(ring.read(100, 200), np.arange(100, 200, dtype=np.int16))


def test_wraparound_keeps_order(ring):
    for i in range(0, 2500, 250):
        ring.write(np.arange(i, i + 250, dtype=np.int16))
    assert ring.write_pos == 2500
    assert np.array_equal(ring.read(1600, 2500), np.arange(1600, 2500, dtype=np.int16))


def test_overwritten_samples_raise(ring):
    ring.write(np.zeros(1500, dtype=np.int16))
    with pytest.raises(Overrun):
        ring.read(0, 100)


def test_cannot_read_the_future(ring):
    ring.write(np.zeros(10, dtype=np.int16))
    with pytest.raises(ValueError):
        ring.read(0, 20)


def test_zero_wipes_decoded_audio(ring):
    ring.write(np.full(600, 7, dtype=np.int16))
    ring.zero(100, 300, keep_margin=0)
    out = ring.read(0, 600)
    assert (out[100:300] == 0).all()
    assert (out[:100] == 7).all() and (out[300:] == 7).all()


def test_zero_skips_slots_about_to_be_overwritten(ring):
    ring.write(np.full(1000, 7, dtype=np.int16))
    ring.zero(0, 1000, keep_margin=200)  # the oldest 200 are left for the writer
    out = ring.read(0, 1000)
    assert (out[:200] == 7).all() and (out[200:] == 0).all()


def _child_writer(name):
    r = Ring.attach(name)
    r.write(np.arange(500, dtype=np.int16))
    r.close()


def test_shared_across_processes(ring):
    p = mp.get_context("spawn").Process(target=_child_writer, args=(ring.name,))
    p.start()
    p.join(20)
    assert p.exitcode == 0
    assert ring.write_pos == 500
    assert np.array_equal(ring.read(0, 500), np.arange(500, dtype=np.int16))
