"""Core package — session state & shared utilities."""

from src.core.state import get, init_state, set as set_state
from src.core.utils import project_root

__all__ = ["init_state", "get", "set_state", "project_root"]
