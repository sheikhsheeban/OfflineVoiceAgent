# Network isolation (sheet 4, invariant INV3)

kin-audio and kin-core must not be able to reach the internet. Kin uses layers, strongest first.

| Layer | Where | What it does | Checked by |
|---|---|---|---|
| Network namespace | Linux, including a Raspberry Pi | the process has no network interfaces at all; only local Unix sockets work | `kin doctor --net` shows `netns+guard` |
| Socket guard | every OS (`kin/guard.py`) | opening an IPv4/IPv6 socket or resolving a name raises `NetworkDisabled` | `test_INV3_*` |
| Import contract | CI (`.importlinter`) | only `kin.gateway` may import `httpx`, `urllib`, ...; audio and core code may not import `socket` | `test_INV4_*` |
| Firewall rule | Windows / macOS, optional | blocks a copy of the interpreter used only for kin-core | manual |

The guard is a tripwire, not a sandbox: code that deliberately reaches for the C-level `_socket` module
can go around it. That is why the Linux namespace exists, and why the gateway is a separate process with an
allowlist of three hosts.

Check it at any time:

```
uv run kin doctor --net
```

## Optional Windows firewall rule (needs an administrator; Kin never runs this itself)

Windows firewall rules match programs, and all Kin processes share `python.exe`. To block only kin-core,
copy the interpreter, run kin-core with the copy, and block that copy:

```
Copy-Item .venv\Scripts\python.exe .venv\Scripts\python-kincore.exe
New-NetFirewallRule -DisplayName "Kin core: no network" -Program "$PWD\.venv\Scripts\python-kincore.exe" `
    -Direction Outbound -Action Block
```

This is not wired into `kin run` in this build. Without it, the socket guard is the only Windows layer.

## Linux notes

`unshare(CLONE_NEWUSER | CLONE_NEWNET)` needs unprivileged user namespaces
(`sysctl kernel.unprivileged_userns_clone=1` on Debian-based systems that disable them). If the kernel
refuses, Kin runs with the socket guard only and `kin doctor --net` says so. This path is written for the
Raspberry Pi but has not been run on one.
