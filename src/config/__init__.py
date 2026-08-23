"""Configuration package — app metadata & module registry."""

from src.config.settings import (
    APP_NAME,
    APP_TAGLINE,
    APP_VERSION,
    MODULES,
    get_module,
)

__all__ = [
    "APP_NAME",
    "APP_TAGLINE",
    "APP_VERSION",
    "MODULES",
    "get_module",
]
