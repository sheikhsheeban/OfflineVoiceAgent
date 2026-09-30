"""Static checks for the week-1 privacy invariants (sheet 9)."""

import ast
import re
from pathlib import Path

import pytest

KIN = Path(__file__).parent.parent / "kin"
NETWORK_MODULES = {"socket", "httpx", "requests", "urllib", "http", "aiohttp", "websockets"}


def imports(path: Path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


@pytest.mark.parametrize("pkg", ["audio", "perception", "memory", "reasoning", "core"])
def test_offline_packages_import_no_network_modules(pkg):
    for f in (KIN / pkg).rglob("*.py"):
        bad = imports(f) & NETWORK_MODULES
        assert not bad, f"{f.relative_to(KIN)} imports {bad}"


def test_audio_never_writes_files():
    for f in (KIN / "audio").rglob("*.py"):
        src = f.read_text(encoding="utf-8")
        # The only file access allowed is reading a demo WAV for replay.
        src = src.replace('wave.open(str(path), "rb")', "")
        assert not re.search(r"\bopen\(|write_bytes|write_text|soundfile|\"wb\"", src), f.name


def test_ui_binds_loopback_only():
    from kin.config import Settings

    assert Settings().ui_host == "127.0.0.1"
