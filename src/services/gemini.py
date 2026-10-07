"""Gemini client abstraction.

Centralizes all Gemini API access so the rest of the app talks to a
single, testable service. The API key is read from Streamlit secrets or
environment variables — it is **never** hardcoded, printed, or logged.

The client also protects the application from transient Gemini outages:
retryable 5xx/429 failures are retried by the SDK and, if the primary
model remains unavailable, a lightweight fallback model is attempted.
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


# gemini-2.0-flash was shut down by Google on June 1, 2026.
# Keep the primary model on a currently supported stable Flash model.
DEFAULT_MODEL: str = "gemini-3.5-flash"
# A lower-cost/high-throughput fallback for temporary capacity problems.
FALLBACK_MODELS: tuple[str, ...] = ("gemini-3.5-flash-lite",)


_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


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
    """Try to read the key from Streamlit secrets."""
    try:
        import streamlit as st
        secrets = st.secrets
    except Exception:
        return None
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


def _status_code(exc: Exception) -> Optional[int]:
    """Best-effort extraction of an HTTP/API status code from an SDK error."""
    for attr in ("code", "status_code", "http_status_code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)

    # google-genai APIError messages commonly contain ``503`` / ``429``.
    message = str(exc)
    for code in _RETRYABLE_STATUS_CODES:
        if str(code) in message:
            return code
    return None


def _is_retryable(exc: Exception) -> bool:
    """Return True for transient errors worth retrying/falling back from."""
    code = _status_code(exc)
    if code in _RETRYABLE_STATUS_CODES:
        return True
    message = str(exc).upper()
    return any(
        token in message
        for token in (
            "UNAVAILABLE",
            "RESOURCE_EXHAUSTED",
            "TOO MANY REQUESTS",
            "TIMEOUT",
            "TIMED OUT",
            "SERVICE UNAVAILABLE",
        )
    )


class GeminiClient:
    """Thin wrapper around the ``google-genai`` SDK with JSON output."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        *,
        fallback_models: tuple[str, ...] = FALLBACK_MODELS,
    ) -> None:
        self._api_key = api_key
        self._model_name = model
        self._fallback_models = tuple(fallback_models)
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

    def _generate_once(
        self,
        *,
        model: str,
        system_prompt: str,
        user_text: str,
        response_schema: Optional[dict],
        temperature: float,
    ) -> str:
        """Make one SDK request for a specific model."""
        from google.genai import types

        config_kwargs: dict[str, Any] = {
            "system_instruction": system_prompt,
            "response_mime_type": "application/json",
            "temperature": temperature,
        }
        if response_schema is not None:
            config_kwargs["response_schema"] = response_schema
        config = types.GenerateContentConfig(**config_kwargs)

        response = self._ensure_client().models.generate_content(
            model=model,
            contents=user_text,
            config=config,
        )
        text = getattr(response, "text", None)
        if not text:
            raise GeminiAPIError("Gemini returned an empty response.")
        return text

    def generate_json(
        self,
        system_prompt: str,
        user_text: str,
        response_schema: Optional[dict] = None,
        *,
        temperature: float = 0.2,
    ) -> str:
        """Generate JSON, with a supported-model fallback for transient failures.

        The google-genai Python SDK already retries transient 429/5xx errors.
        If the primary model still fails with a retryable error, this wrapper
        tries the configured fallback model(s). Non-retryable errors are raised
        immediately because changing models would not fix them.
        """
        # Ensure the SDK is installed before constructing the request.
        try:
            from google import genai  # noqa: F401
        except ImportError as e:
            raise GeminiConfigError(
                "The 'google-genai' package is required to call Gemini. "
                "Install it (pip install google-genai) and configure an API key."
            ) from e

        models = (self._model_name,) + tuple(
            model for model in self._fallback_models if model != self._model_name
        )
        last_error: Optional[Exception] = None

        for model in models:
            try:
                return self._generate_once(
                    model=model,
                    system_prompt=system_prompt,
                    user_text=user_text,
                    response_schema=response_schema,
                    temperature=temperature,
                )
            except GeminiAPIError as exc:
                last_error = exc
                if not _is_retryable(exc):
                    raise
            except GeminiConfigError:
                # Configuration errors must remain configuration errors.
                raise

            except Exception as exc:
                # Convert SDK errors to our stable application error type.
                wrapped = GeminiAPIError(f"Gemini API request failed: {exc}")
                last_error = wrapped
                if not _is_retryable(exc):
                    raise wrapped from exc

        raise GeminiAPIError(
            "Gemini API request failed after trying the configured models. "
            "The service may be temporarily unavailable; please try again."
            + (f" Last error: {last_error}" if last_error else "")
        ) from last_error
