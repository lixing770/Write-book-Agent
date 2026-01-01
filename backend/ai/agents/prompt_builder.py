# backend/ai/agents/prompt_builder.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any

from ..provider import AIProvider


def render_template(tpl: str, **kwargs: str) -> str:
    for k, v in kwargs.items():
        tpl = tpl.replace("{{" + k + "}}", v)
    return tpl

class PromptBuilderAgent:
    def __init__(self, provider: AIProvider, templates_dir: Path) -> None:
        self.provider = provider
        self.tpl_path = templates_dir / "prompt_builder.md"

    def build(self, spec: Dict[str, Any], chapter: Dict[str, Any], knowledge: str, memory: str = "") -> str:
        tpl = self.tpl_path.read_text(encoding="utf-8")
        filled = render_template(
            tpl,
            SPEC=str(spec),
            OUTLINE=str(chapter),
            KNOWLEDGE=knowledge,
            MEMORY=memory or "None",
        )
        messages = [
            {"role": "system", "content": "You are a professional fiction prompt engineer."},
            {"role": "user", "content": filled},
        ]
        return self.provider.chat(messages, temperature=0.4, max_tokens=1800)
