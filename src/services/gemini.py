"""Gemini client abstraction.

Centralizes all Gemini API access so the rest of the app talks to a
single, testable service. The API key is read from Streamlit secrets or
environment variables — it is **never** hardcoded, printed, or logged.

:class:`GeminiClient` lazily imports ``google.genai`` so this module
imports cleanly even when the SDK is not installed; the import only
happens when a request is actually made.
"""

from __future__ import annotations

import os
from typing import Any, Optional


class GeminiError(Exception):
    """Base error for Gemini client failures."""


class GeminiConfigError(GeminiError):
    """Raised when the API key is missing / obviously invalid."""


class GeminiAPIError(GeminiError):
    """Raised when a Gemini API call fails (network, auth, rate limit)."""


DEFAULT_MODEL: str = "gemini-3.6-flash"


def get_api_key() -> str:
    """Read the Gemini API key from Streamlit secrets, then env.

    Order of resolution:
      1. ``st.secrets["gemini"]["api_key"]`` (Streamlit secrets file).
      2. ``st.secrets["GEMINI_API_KEY"]`` (Streamlit secrets, top-level).
      3. ``GEMINI_API_KEY`` environment variable.
      4. ``GOOGLE_API_KEY`` environment variable.

    Raises:
        GeminiConfigError: if no key is configured anywhere.

    The key is never printed or logged.
    """
    key = _from_streamlit_secrets() or _from_env()
    if not key:
        raise GeminiConfigError(
            "Gemini API key is not configured. Set GEMINI_API_KEY in your "
            "environment or add [gemini] api_key = \"...\" to "
            ".streamlit/secrets.toml (see secrets.toml.example)."
        )
    return key


def _from_streamlit_secrets() -> Optional[str]:
    """Try to read the key from Streamlit secrets.

    Returns ``None`` if Streamlit isn't running, no ``secrets.toml`` file
    exists, or the key isn't configured there. This function MUST never
    raise — indexing ``st.secrets`` triggers a lazy parse that can throw
    ``StreamlitSecretNotFoundError`` (not a ``KeyError``) when no secrets
    file is present, and we want to fall back to env vars in that case.
    """
    try:
        import streamlit as st  # local import: not all callers are Streamlit
        secrets = st.secrets
    except Exception:
        return None
    # Each index access can raise (no secrets file / missing key); tolerate all.
    try:
        return secrets["gemini"]["api_key"]
    except Exception:
        pass
    try:
        return secrets["GEMINI_API_KEY"]
    except Exception:
        return None


def _from_env() -> Optional[str]:
    """Read the key from environment variables."""
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


class GeminiClient:
    """Thin wrapper around the ``google-genai`` SDK with JSON output."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
    ) -> None:
        # ``api_key`` is resolved lazily on the first request if not given,
        # so constructing the client never requires a configured key.
        self._api_key = api_key
        self._model_name = model
        self._client: Any = None

    def _resolve_key(self) -> str:
        return self._api_key or get_api_key()

    def _ensure_client(self) -> Any:
        """Lazily create the google-genai client."""
        if self._client is None:
            try:
                from google import genai
            except ImportError as e:
                raise GeminiConfigError(
                    "The 'google-genai' package is required to call Gemini. "
                    "Install it (pip install google-genai) and configure an "
                    "API key."
                ) from e
            try:
                self._client = genai.Client(api_key=self._resolve_key())
            except Exception as e:
                raise GeminiConfigError(
                    f"Gemini client could not be initialized: {e}"
                ) from e
        return self._client

    def generate_json(
        self,
        system_prompt: str,
        user_text: str,
        response_schema: Optional[dict] = None,
        *,
        temperature: float = 0.2,
    ) -> str:
        """Generate content constrained to JSON; return the raw JSON string.

        Args:
            system_prompt: instructions shaping model behavior.
            user_text: user-supplied content (e.g. resume text).
            response_schema: optional OpenAPI-subset dict constraining shape.
            temperature: sampling temperature.

        Raises:
            GeminiConfigError: missing key / SDK not installed.
            GeminiAPIError: the API call failed or returned nothing.
        """
        from google import genai  # noqa: F401  (asserts SDK is importable)
        from google.genai import types

        client = self._ensure_client()
        config_kwargs: dict[str, Any] = {
            "system_instruction": system_prompt,
            "response_mime_type": "application/json",
            "temperature": temperature,
        }
        if response_schema is not None:
            config_kwargs["response_schema"] = response_schema
        config = types.GenerateContentConfig(**config_kwargs)
        try:
            response = client.models.generate_content(
                model=self._model_name,
                contents=user_text,
                config=config,
            )
        except Exception as e:
            raise GeminiAPIError(f"Gemini API request failed: {e}") from e
        text = getattr(response, "text", None)
        if not text:
            raise GeminiAPIError("Gemini returned an empty response.")
        return text
