"""The memory store's encryption key, held by the OS keyring.

Windows Credential Manager, macOS Keychain or Secret Service on Linux. The key
never touches disk in plain form. `KIN_DB_KEY` (64 hex chars) overrides the
keyring, for tests and CI only.

A headless Raspberry Pi usually has no Secret Service. There, and only there,
the key falls back to a file next to the database, readable by its owner only
(mode 0600), and Kin logs a warning: anyone who can read the user's files can
read the memory. Full-disk encryption on the Pi covers that gap. Set
`KIN_KEY_BACKEND=file` to choose the file on purpose, `keyring` to forbid it.
"""

from __future__ import annotations

import logging
import os
import secrets
from pathlib import Path

SERVICE = "kin-memory"
log = logging.getLogger("kin.keys")


class KeyMissing(RuntimeError):
    pass


def _account(db_path: str) -> str:
    return "db-key:" + os.path.normcase(os.path.abspath(db_path))


def _file_path(db_path: str) -> Path:
    return Path(str(db_path) + ".key")


def _keyring_usable() -> bool:
    if os.environ.get("KIN_KEY_BACKEND") == "file":
        return False
    try:
        import keyring
        from keyring.backends import fail

        return not isinstance(keyring.get_keyring(), fail.Keyring)
    except Exception:  # noqa: BLE001 - no usable backend on this machine
        return False


def _use_file() -> bool:
    if _keyring_usable():
        return False
    if os.environ.get("KIN_KEY_BACKEND") == "keyring":
        raise KeyMissing("no OS keyring is available and KIN_KEY_BACKEND=keyring forbids a key file")
    return True


def get_key(db_path: str, create: bool) -> str:
    env = os.environ.get("KIN_DB_KEY")
    if env:
        return env
    if _use_file():
        path = _file_path(db_path)
        if path.exists():
            return path.read_text().strip()
        if not create:
            raise KeyMissing(f"{db_path} exists but its key file {path} is missing; it cannot be opened.")
        log.warning("no OS keyring here: keeping the database key in %s (owner-only file)", path)
        key = secrets.token_hex(32)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(key)
        return key
    import keyring

    key = keyring.get_password(SERVICE, _account(db_path))
    if key:
        return key
    if not create:
        raise KeyMissing(
            f"{db_path} exists but its key is not in the OS keyring; it cannot be opened. "
            "Delete the file (all memory is lost) or restore the keyring entry."
        )
    key = secrets.token_hex(32)
    keyring.set_password(SERVICE, _account(db_path), key)
    return key


def delete_key(db_path: str) -> None:
    """Crypto-shred: without the key, any leftover copy of the file is unreadable."""
    if os.environ.get("KIN_DB_KEY"):
        return
    if _use_file():
        _file_path(db_path).unlink(missing_ok=True)
        return
    import keyring
    from keyring.errors import PasswordDeleteError

    try:
        keyring.delete_password(SERVICE, _account(db_path))
    except PasswordDeleteError:
        pass
