"""Locate LM Studio's `lms` command-line tool."""

from __future__ import annotations

import shutil
from pathlib import Path


def find_lms(configured: str | None) -> str | None:
    """The configured path if it is a file, else `lms` on PATH, else LM Studio's own bin folder, else None."""
    if configured and Path(configured).is_file():
        return configured
    found = shutil.which("lms")
    if found:
        return found
    folder = Path.home() / ".lmstudio" / "bin"
    for name in ("lms.exe", "lms"):
        candidate = folder / name
        if candidate.is_file():
            return str(candidate)
    return None
