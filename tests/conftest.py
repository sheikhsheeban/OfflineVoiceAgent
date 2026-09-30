from __future__ import annotations

import os
import tempfile
from pathlib import Path

# Tests never touch the real memory store or the OS keyring. Set before any kin
# import; spawned kin processes inherit it.
os.environ["KIN_DATA_DIR"] = tempfile.mkdtemp(prefix="kin-test-data-")
os.environ["KIN_DB_KEY"] = "ab" * 32

import pytest  # noqa: E402

from kin.cli import verify_models  # noqa: E402
from kin.perception.textmetrics import wer  # noqa: E402, F401  (re-exported for tests)

DATA = Path(__file__).parent / "data"
ASR_WAVS = Path(__file__).parent / "data" / "asr_ref"  # LibriSpeech clips shipped with sherpa-onnx

_models_ok = not verify_models()


def pytest_collection_modifyitems(config, items):
    skip = pytest.mark.skip(reason="models missing: run `uv run kin fetch-models`")
    for item in items:
        if "models" in item.keywords and not _models_ok:
            item.add_marker(skip)
