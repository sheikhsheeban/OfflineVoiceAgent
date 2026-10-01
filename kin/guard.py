"""Network guard for kin-audio and kin-core (invariant INV3, sheet 4).

Layers, strongest first:
  1. Linux: `isolate()` moves the process into a fresh network namespace, so it has
     no network interfaces at all (unprivileged, through a user namespace). Local IPC
     is unaffected: Unix sockets on the filesystem still connect.
  2. Everywhere: `install()` replaces `socket.socket` and name resolution so that
     any attempt to open an IPv4 or IPv6 socket raises immediately. This is a
     tripwire against mistakes and against libraries that phone home, not a sandbox:
     code that reaches for the C-level `_socket` module bypasses it, which is why
     layer 1 exists on Linux.
  3. Windows / macOS: an optional per-application firewall rule (docs/NETWORK.md);
     it needs administrator rights, so Kin does not add it by itself.

Import this module first in a process, before anything else opens a socket.
"""

from __future__ import annotations

import ctypes
import os
import socket
import sys

CLONE_NEWUSER = 0x10000000
CLONE_NEWNET = 0x40000000

_installed = False
_method = "none"


class NetworkDisabled(PermissionError):
    """Raised when code in a no-network process tries to use the network."""


def _isolate_linux() -> bool:
    if not sys.platform.startswith("linux"):
        return False
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        if os.geteuid() == 0:
            ok = libc.unshare(CLONE_NEWNET) == 0
        else:
            ok = libc.unshare(CLONE_NEWUSER | CLONE_NEWNET) == 0
        return bool(ok)
    except Exception:  # noqa: BLE001 - no unshare, seccomp forbids it, etc.
        return False


def install(namespace: bool = True) -> str:
    """Block networking in this process. Returns the strongest method that took effect:
    "netns+guard", "guard"."""
    global _installed, _method
    if _installed:
        return _method
    ns = namespace and _isolate_linux()
    real_socket = socket.socket

    class GuardedSocket(real_socket):  # type: ignore[misc, valid-type]
        def __init__(self, family=-1, type=-1, proto=-1, fileno=None):  # noqa: A002
            fam = socket.AF_INET if family == -1 and fileno is None else family
            if fam in (socket.AF_INET, socket.AF_INET6):
                raise NetworkDisabled("this Kin process has no network access")
            super().__init__(family, type, proto, fileno)

    def _no_resolve(*_a, **_k):
        raise NetworkDisabled("this Kin process has no network access (name lookup blocked)")

    socket.socket = GuardedSocket  # type: ignore[misc]
    for fn in ("getaddrinfo", "gethostbyname", "gethostbyname_ex", "gethostbyaddr", "getnameinfo"):
        setattr(socket, fn, _no_resolve)
    _installed = True
    _method = "netns+guard" if ns else "guard"
    return _method


def probe() -> dict:
    """Try to reach the internet; report what happened. Used by `kin doctor --net` and the tests."""
    blocked, why = False, ""
    try:
        socket.create_connection(("1.1.1.1", 443), timeout=1.0).close()
    except OSError as e:  # NetworkDisabled is a PermissionError, so also an OSError
        blocked, why = True, type(e).__name__
    try:
        socket.getaddrinfo("example.com", 443)
        resolves = True
    except OSError:
        resolves = False
    return {"blocked": blocked, "dns_blocked": not resolves, "method": _method, "detail": why}
