"""Raspberry Pi 4 (Linux, aarch64) readiness, checked as far as a Windows laptop allows.

Nothing here runs on a Pi. These tests pin the parts that differ from Windows: the Unix-socket IPC path,
the network namespace layer, the key-file fallback for a headless machine without a keyring, and the
memory and thread budgets the phone profile enforces. The real Pi run is in docs/RASPBERRY_PI.md."""

import os
import stat
import sys

import pytest

from kin import guard, ipc


def test_linux_ipc_uses_a_private_unix_socket_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(os, "getuid", lambda: 1000, raising=False)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    addr = ipc.address("gateway", "abc123")
    assert addr == str(tmp_path / "kin-1000" / "gateway-abc123.sock")
    assert (tmp_path / "kin-1000").is_dir()
    if os.name != "nt":
        assert stat.S_IMODE((tmp_path / "kin-1000").stat().st_mode) == 0o700


def test_windows_ipc_uses_a_named_pipe(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    assert ipc.address("gateway", "abc123") == "\\\\.\\pipe\\kin-gateway-abc123"


class FakeLibc:
    def __init__(self, result=0):
        self.result, self.flags = result, None

    def unshare(self, flags):
        self.flags = flags
        return self.result


@pytest.mark.parametrize(
    "uid,result,flags,expected",
    [
        (1000, 0, guard.CLONE_NEWUSER | guard.CLONE_NEWNET, True),  # a normal user: user namespace + network namespace
        (0, 0, guard.CLONE_NEWNET, True),  # root needs no user namespace
        (1000, -1, guard.CLONE_NEWUSER | guard.CLONE_NEWNET, False),  # kernel refuses: fall back to the socket guard
    ],
)
def test_INV3_linux_namespace_layer_asks_for_the_right_isolation(monkeypatch, uid, result, flags, expected):
    import ctypes

    libc = FakeLibc(result)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(os, "geteuid", lambda: uid, raising=False)
    monkeypatch.setattr(ctypes, "CDLL", lambda *a, **k: libc)
    assert guard._isolate_linux() is expected and libc.flags == flags


def test_INV3_a_machine_without_unshare_still_gets_the_socket_guard(monkeypatch):
    import ctypes

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(ctypes, "CDLL", lambda *a, **k: (_ for _ in ()).throw(OSError("no libc")))
    assert guard._isolate_linux() is False  # install() then reports "guard", and `kin doctor --net` says so


def test_headless_pi_without_a_keyring_keeps_the_key_in_an_owner_only_file(monkeypatch, tmp_path):
    from kin.memory import keys

    monkeypatch.delenv("KIN_DB_KEY", raising=False)
    monkeypatch.setenv("KIN_KEY_BACKEND", "file")
    db = str(tmp_path / "kin.db")
    k1 = keys.get_key(db, create=True)
    assert len(k1) == 64 and keys.get_key(db, create=False) == k1  # stable across restarts
    if os.name != "nt":
        assert stat.S_IMODE(os.stat(db + ".key").st_mode) == 0o600
    with pytest.raises(keys.KeyMissing):
        keys.get_key(str(tmp_path / "other.db"), create=False)  # a database with no key cannot be opened
    keys.delete_key(db)  # crypto-shred: the key file goes with the wipe
    assert not os.path.exists(db + ".key")


def test_keyring_can_be_required_so_a_missing_one_is_an_error_not_a_silent_file(monkeypatch, tmp_path):
    from kin.memory import keys

    monkeypatch.delenv("KIN_DB_KEY", raising=False)
    monkeypatch.setenv("KIN_KEY_BACKEND", "keyring")
    monkeypatch.setattr(keys, "_keyring_usable", lambda: False)
    with pytest.raises(keys.KeyMissing, match="forbids a key file"):
        keys.get_key(str(tmp_path / "kin.db"), create=True)


def test_phone_profile_is_the_pi_profile_four_threads_and_a_1_1_gb_ceiling():
    from kin.config import get_profile

    p = get_profile("phone")
    assert (p.core_threads, p.core_rss_limit_mb) == (4, 1100)  # a Pi 4 has four cores and 4 GB
    assert p.asr_threads <= 2  # leaves two cores for the language model while it is thinking


def test_the_voice_and_speaker_models_fit_a_pi():
    from kin.config import MODELS_DIR, load_model_spec

    spec = load_model_spec()
    total = 0
    for section in ("voice", "speaker"):
        for name in spec[section]["files"]:
            total += (MODELS_DIR / spec[section]["dir"] / name).stat().st_size if (MODELS_DIR / spec[section]["dir"] / name).exists() else 0
    assert total < 100 * 2**20  # about 90 MB on disk for both
