# backend/ai/provider.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Protocol

Message = Dict[str, Any]


class AIProvider(Protocol):
    def chat(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        ...


@dataclass
class ProviderConfig:
    provider: str = "openai"


def get_ai_provider(provider: str = "openai") -> AIProvider:
    """
    Factory to return an AI provider instance.
    """
    if provider == "openai":
        from .openai_provider import OpenAIProvider  # lazy import
        return OpenAIProvider()
    raise ValueError(f"Unknown provider: {provider}")
