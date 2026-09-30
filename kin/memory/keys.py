"""The memory store's encryption key, held by the OS keyring.

Windows Credential Manager, macOS Keychain or Secret Service on Linux. The key
never touches disk in plain form. `KIN_DB_KEY` (64 hex chars) overrides the
keyring, for tests and CI only.
"""

from __future__ import annotations

import os
import secrets

SERVICE = "kin-memory"


class KeyMissing(RuntimeError):
    pass


def _account(db_path: str) -> str:
    return "db-key:" + os.path.normcase(os.path.abspath(db_path))


def get_key(db_path: str, create: bool) -> str:
    env = os.environ.get("KIN_DB_KEY")
    if env:
        return env
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
    import keyring
    from keyring.errors import PasswordDeleteError

    try:
        keyring.delete_password(SERVICE, _account(db_path))
    except PasswordDeleteError:
        pass
