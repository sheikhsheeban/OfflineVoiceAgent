"""Native-library loading fixes.

Windows 11 ships an old onnxruntime.dll in System32 (ORT 1.17). sherpa-onnx
links onnxruntime.dll by name, and without help the loader picks the System32
copy, which lacks the API version sherpa-onnx needs and crashes the process.
Loading the pip onnxruntime DLL first makes every later by-name lookup resolve
to it.
"""

from __future__ import annotations

import os
import sys

_done = False


def preload_onnxruntime() -> None:
    global _done
    if _done or sys.platform != "win32":
        return
    import ctypes
    import importlib.util

    spec = importlib.util.find_spec("onnxruntime")
    if spec and spec.origin:
        dll = os.path.join(os.path.dirname(spec.origin), "capi", "onnxruntime.dll")
        if os.path.exists(dll):
            ctypes.WinDLL(dll)
    _done = True
