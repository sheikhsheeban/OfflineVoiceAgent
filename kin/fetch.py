"""`kin fetch-models`: the only code in Kin that downloads models.

Kept apart from the running app on purpose: the app never downloads anything
(sheet 4), and the import contract lets only this module (and kin.gateway) use
`urllib`. Every file is checked against the sha256 pinned in spec/models.toml.
"""

from __future__ import annotations

import tarfile
import tempfile
import urllib.request
from collections.abc import Callable, Iterator
from pathlib import Path

from kin.config import MODELS_DIR


def fetch_entries(entries: list[tuple[str, dict]], sha256: Callable[[Path], str]) -> Iterator[str]:
    for section, spec in entries:
        d = MODELS_DIR / spec["dir"]
        missing = [n for n, h in spec["files"].items() if not (d / n).exists() or sha256(d / n) != h]
        if not missing:
            yield f"ok       {section}: {spec['name']}"
            continue
        yield f"fetching {section}: {spec['name']}"
        d.mkdir(parents=True, exist_ok=True)
        if "urls" in spec:
            for n in missing:
                urllib.request.urlretrieve(spec["urls"][n], d / n)
        elif "url_base" in spec:
            for n in missing:
                urllib.request.urlretrieve(spec["url_base"] + n, d / n)
        elif spec["url"].endswith((".tar.bz2", ".tar.gz")):
            dest = MODELS_DIR / spec["extract_to"]
            with tempfile.TemporaryDirectory() as tmp:
                arc = Path(tmp) / "model.tar"
                urllib.request.urlretrieve(spec["url"], arc)
                with tarfile.open(arc) as tf:
                    for m in tf.getmembers():  # refuse path traversal
                        target = (dest / m.name).resolve()
                        if not str(target).startswith(str(dest.resolve())) or m.issym() or m.islnk():
                            raise SystemExit(f"unsafe path in archive: {m.name}")
                    tf.extractall(dest)
            if spec.get("prune"):
                for f in d.iterdir():
                    if f.is_file() and f.name not in spec["files"]:
                        f.unlink()
        else:
            urllib.request.urlretrieve(spec["url"], d / missing[0])
