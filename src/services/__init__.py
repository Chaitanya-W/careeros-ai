"""Services package — cross-cutting service interfaces (LLM, Gemini, etc.)."""

from src.services.gemini import (
    DEFAULT_MODEL,
    GeminiAPIError,
    GeminiClient,
    GeminiConfigError,
    GeminiError,
    get_api_key,
)
from src.services.llm import LLMClient, StubLLMClient, get_llm_client

__all__ = [
    "GeminiClient",
    "GeminiError",
    "GeminiConfigError",
    "GeminiAPIError",
    "DEFAULT_MODEL",
    "get_api_key",
    "LLMClient",
    "StubLLMClient",
    "get_llm_client",
]
