# backend/ai/agents/writer.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
from ..provider import AIProvider

class WriterAgent:
    def __init__(self, provider: AIProvider, templates_dir: Path) -> None:
        self.provider = provider
        self.tpl_path = templates_dir / "writer.md"

    def write(self, spec: Dict[str, Any], chapter: Dict[str, Any], writing_prompt: str) -> str:
        tpl = self.tpl_path.read_text(encoding="utf-8")
        prompt = tpl.replace("{{SPEC}}", str(spec)).replace("{{OUTLINE}}", str(chapter)).replace("{{PROMPT}}", writing_prompt)
        messages = [
            {"role": "system", "content": "You are a bestselling novelist. Output strictly in required format."},
            {"role": "user", "content": prompt},
        ]
        return self.provider.chat(messages, temperature=0.8, max_tokens=3500)
