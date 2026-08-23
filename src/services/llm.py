"""LLM service interface (placeholder).

Defines the *contract* for interacting with large language models.
No provider is wired up yet — concrete implementations (OpenAI,
Anthropic, local, etc.) will be added in later iterations. This keeps
the dependency surface minimal in skeleton mode.
"""

from __future__ import annotations

from typing import Protocol


class LLMClient(Protocol):
    """Minimal LLM interface consumed by application modules."""

    def chat(self, system: str, user: str) -> str:
        """Return a chat completion for the given system + user prompt."""
        ...


class StubLLMClient:
    """No-op client used until a real provider is configured.

    Raises ``NotImplementedError`` so that calling code fails loudly
    instead of silently returning empty results.
    """

    def chat(self, system: str, user: str) -> str:  # noqa: D401
        raise NotImplementedError(
            "LLM provider not configured. Implement `chat` on a concrete "
            "LLMClient or wire up an API key in .streamlit/secrets.toml."
        )


_default_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    """Return the currently configured LLM client (singleton).

    Returns a :class:`StubLLMClient` until a real provider is registered.
    """
    if _default_client is None:
        return StubLLMClient()
    return _default_client
