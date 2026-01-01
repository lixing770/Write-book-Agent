# backend/ai/openai_provider.py
from __future__ import annotations
from typing import List, Dict, Any
from openai import OpenAI

from .config import OPENAI_API_KEY, OPENAI_MODEL

Message = Dict[str, Any]

class OpenAIProvider:
    def __init__(self) -> None:
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = OPENAI_MODEL

    def chat(self, messages: List[Message], temperature: float = 0.7, max_tokens: int = 2000) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""
