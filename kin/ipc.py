"""Local IPC between kin-core, kin-gateway and kin-ui (spec CL2).

`multiprocessing.connection` everywhere: a named pipe on Windows, an AF_UNIX
socket in a private directory on Linux and macOS. Neither is a network socket,
so kin-core's network guard leaves them alone, and a Unix socket on the
filesystem is reachable from inside a Linux network namespace.
"""

from __future__ import annotations

import os
import sys
import tempfile
from multiprocessing.connection import Client, Listener
from pathlib import Path


def address(name: str, session: str) -> str:
    if sys.platform == "win32":
        return "\\\\.\\pipe\\kin-" + f"{name}-{session}"
    base = os.environ.get("XDG_RUNTIME_DIR") or tempfile.gettempdir()
    d = Path(base) / f"kin-{os.getuid()}"
    d.mkdir(mode=0o700, exist_ok=True)
    return str(d / f"{name}-{session}.sock")


def listen(name: str, session: str, authkey: bytes) -> Listener:
    addr = address(name, session)
    if sys.platform != "win32":
        Path(addr).unlink(missing_ok=True)
    family = "AF_PIPE" if sys.platform == "win32" else "AF_UNIX"
    return Listener(addr, family=family, authkey=authkey)


def connect(name: str, session: str, authkey: bytes):
    addr = address(name, session)
    family = "AF_PIPE" if sys.platform == "win32" else "AF_UNIX"
    return Client(addr, family=family, authkey=authkey)
