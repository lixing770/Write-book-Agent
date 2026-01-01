# backend/ai/agents/guard.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
from ..provider import AIProvider

REQUIRED_MARKERS = ["# ", "## 摘要", "## 正文", "## 结尾钩子", "## 角色状态"]

class GuardAgent:
    def __init__(self, provider: AIProvider, templates_dir: Path) -> None:
        self.provider = provider
        self.tpl_path = templates_dir / "guard.md"

    def check(self, spec: Dict[str, Any], draft: str) -> str:
        missing = [m for m in REQUIRED_MARKERS if m not in draft]
        if not missing:
            return "PASS"

        tpl = self.tpl_path.read_text(encoding="utf-8")
        prompt = tpl.replace("{{SPEC}}", str(spec)).replace("{{DRAFT}}", draft).replace("{{MISSING}}", str(missing))
        messages = [
            {"role": "system", "content": "You are a safety & formatting checker."},
            {"role": "user", "content": prompt},
        ]
        return self.provider.chat(messages, temperature=0.2, max_tokens=900)
