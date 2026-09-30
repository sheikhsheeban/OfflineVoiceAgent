"""Plain stderr logging, tagged with the Kin process name."""

from __future__ import annotations

import logging
import os


def setup(process_name: str) -> None:
    logging.basicConfig(
        level=os.environ.get("KIN_LOG", "INFO"),
        format=f"%(asctime)s {process_name:<11} %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
