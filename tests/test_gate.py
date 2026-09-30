import numpy as np

from kin.audio.gate import EnergyGate, frame_dbfs
from kin.config import FRAME

rng = np.random.default_rng(0)


def noise(dbfs):
    amp = 32768 * 10 ** (dbfs / 20)
    return (rng.standard_normal(FRAME) * amp).clip(-32768, 32767).astype(np.int16)


def test_dbfs_of_known_level():
    assert abs(frame_dbfs(noise(-30)) - -30) < 1.0
    assert frame_dbfs(np.zeros(FRAME, np.int16)) < -100


def test_silence_keeps_gate_closed():
    g = EnergyGate()
    assert not any(g.update(noise(-70)) for _ in range(200))


def test_speech_level_opens_and_hangover_closes():
    g = EnergyGate(hangover_frames=10, calibration_frames=0)
    for _ in range(100):
        g.update(noise(-65))
    assert g.update(noise(-25))
    states = [g.update(noise(-65)) for _ in range(15)]
    assert all(states[:9]) and not states[-1]


def test_floor_adapts_to_steady_room_noise():
    g = EnergyGate()
    opened = [g.update(noise(-40)) for _ in range(600)]  # ~19 s of fan noise
    assert not any(opened[-100:]), "steady noise must stop opening the gate"
    assert abs(g.floor - -40) < 4
    assert g.update(noise(-20))  # speech above the fan still opens it


def test_calibration_ignores_device_start_silence():
    g = EnergyGate()
    g.update(np.zeros(FRAME, np.int16))  # devices often start with digital zeros
    g.update(np.zeros(FRAME, np.int16))
    opened = [g.update(noise(-47)) for _ in range(90)]  # 3 s of a quiet office
    assert not any(opened[31:]), "room noise must not hold the gate open after ~1 s"
    assert abs(g.floor - -47) < 3


def test_gate_is_closed_while_calibrating():
    g = EnergyGate(calibration_frames=5)
    assert not any(g.update(noise(-20)) for _ in range(5))
