"""General-purpose utilities shared across modules."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

# Absolute path to the project root (the folder containing app.py).
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return _PROJECT_ROOT
