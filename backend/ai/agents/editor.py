# backend/ai/agents/editor.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
from ..provider import AIProvider

class EditorAgent:
    def __init__(self, provider: AIProvider, templates_dir: Path) -> None:
        self.provider = provider
        self.tpl_path = templates_dir / "editor.md"

    def review(self, spec: Dict[str, Any], chapter: Dict[str, Any], draft: str) -> str:
        tpl = self.tpl_path.read_text(encoding="utf-8")
        prompt = tpl.replace("{{SPEC}}", str(spec)).replace("{{OUTLINE}}", str(chapter)).replace("{{DRAFT}}", draft)
        messages = [
            {"role": "system", "content": "You are a strict fiction editor."},
            {"role": "user", "content": prompt},
        ]
        return self.provider.chat(messages, temperature=0.3, max_tokens=1200)
