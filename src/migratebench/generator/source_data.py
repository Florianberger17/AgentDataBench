"""Utility for removing real/raw company data before a benchmark package is
published. Real data should never leave the local `source_data/` directory."""

from __future__ import annotations

import shutil
from pathlib import Path


def purge_source_data(root: Path) -> None:
    shutil.rmtree(root / "source_data", ignore_errors=True)
