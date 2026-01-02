from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


class PromptBuilderAgent:
    """
    Build a writing prompt for a chapter.

    - Accepts provider with .chat(messages, temperature=..., max_tokens=...) method.
    - tpl_path can be:
        * a file path (writer.md), or
        * a directory path (templates/) -> auto use writer.md inside.
    """

    def __init__(
        self,
        provider,
        tpl_path: Optional[Path] = None,
        style_bible_path: Optional[Path] = None,
        use_llm_refine: bool = False,
    ):
        self.provider = provider
        self.use_llm_refine = use_llm_refine

        # ---- template path ----
        if tpl_path is None:
            # default: backend/ai/prompts/templates/writer.md
            self.tpl_path = Path(__file__).resolve().parents[1] / "prompts" / "templates" / "writer.md"
        else:
            self.tpl_path = Path(tpl_path)

        # 容错：如果给的是 templates 目录，则自动使用 writer.md
        if self.tpl_path.exists() and self.tpl_path.is_dir():
            self.tpl_path = self.tpl_path / "writer.md"

        # 再容错：如果传错成 templates（但目录不存在），也尽量兜底
        if not self.tpl_path.exists():
            # try common locations
            candidates = [
                Path("backend/ai/prompts/templates/writer.md"),
                Path("backend/prompts/templates/writer.md"),
            ]
            for c in candidates:
                if c.exists():
                    self.tpl_path = c
                    break

        # ---- style bible path ----
        if style_bible_path is None:
            self.style_bible_path = Path("data") / "knowledge" / "style_bible.json"
        else:
            self.style_bible_path = Path(style_bible_path)

    def _read_text(self, p: Path) -> str:
        if p.is_dir():
            raise IsADirectoryError(f"Template path is a directory, expected file: {p}")
        return p.read_text(encoding="utf-8")

    def _load_style_bible_text(self) -> str:
        """
        Return pretty JSON string (or fallback text).
        """
        if not self.style_bible_path.exists():
            return "(style_bible.json not found; follow default style guidance in template)"

        s = self.style_bible_path.read_text(encoding="utf-8", errors="strict").strip()
        obj = json.loads(s)  # ensure valid JSON
        return json.dumps(obj, ensure_ascii=False, indent=2)

    @staticmethod
    def _chapter_outline(chapter: Dict[str, Any]) -> str:
        """
        Convert chapter dict into outline text.
        Supports keys:
          - beats: list[str]
          - hook: str
          - title: str
          - id / chapter: numeric or str
        """
        beats = chapter.get("beats") or chapter.get("outline") or []
        hook = chapter.get("hook") or ""
        title = chapter.get("title") or ""
        cid = chapter.get("id", chapter.get("chapter", ""))

        lines = []
        if cid or title:
            header = []
            if cid != "":
                header.append(f"第{cid}章")
            if title:
                header.append(str(title))
            lines.append(" - ".join(header))

        if isinstance(beats, str):
            # allow outline string
            beats = [b.strip() for b in beats.splitlines() if b.strip()]

        if isinstance(beats, list):
            for i, b in enumerate(beats, start=1):
                lines.append(f"{i}. {str(b).strip()}")
        else:
            lines.append(str(beats))

        if hook:
            lines.append(f"结尾钩子：{hook}")

        return "\n".join(lines).strip()

    def build(
        self,
        spec: str,
        chapter: Dict[str, Any],
        knowledge: str = "",
        memory: str = "",
    ) -> str:
        # 1) load template
        if not self.tpl_path.exists():
            raise FileNotFoundError(
                f"writer template not found.\n"
                f"Expected: {self.tpl_path}\n"
                f"Try create: backend/ai/prompts/templates/writer.md"
            )
        tpl = self._read_text(self.tpl_path)

        # 2) load style bible
        style_bible = self._load_style_bible_text()

        # 3) build outline text
        chapter_outline = self._chapter_outline(chapter)

        # 4) format
        # template can reference these placeholders:
        # {style_bible} {spec} {chapter_outline} {memory} {knowledge}
        prompt = tpl.format(
            style_bible=style_bible,
            spec=spec or "",
            chapter_outline=chapter_outline or "",
            memory=memory or "(none)",
            knowledge=knowledge or "",
        )

        # 5) optional refine by LLM (default off; more stable if off)
        if not self.use_llm_refine:
            return prompt

        messages = [
            {"role": "system", "content": "你是 PromptBuilder。只输出最终写作指令文本，不要解释。"},
            {"role": "user", "content": prompt},
        ]
        return self.provider.chat(messages, temperature=0.2, max_tokens=1200)
