# backend/ai/workflows/run_chapter.py
from __future__ import annotations
import argparse
from pathlib import Path
import yaml

from dotenv import load_dotenv
load_dotenv()

from ..openai_provider import OpenAIProvider
from ..memory.retrieval import load_knowledge
from ..agents.prompt_builder import PromptBuilderAgent
from ..agents.writer import WriterAgent
from ..agents.editor import EditorAgent
from ..agents.guard import GuardAgent

def read_yaml(p: Path):
    return yaml.safe_load(p.read_text(encoding="utf-8"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True, help="e.g. data/projects/book_001")
    ap.add_argument("--chapter", type=int, required=True)
    args = ap.parse_args()

    project_dir = Path(args.project)
    spec = read_yaml(project_dir / "spec.yml")
    outline = read_yaml(project_dir / "outline.yml")

    ch = None
    for c in outline.get("chapters", []):
        if int(c.get("id")) == args.chapter:
            ch = c
            break
    if ch is None:
        raise RuntimeError(f"Chapter {args.chapter} not found in outline.yml")

    base_dir = Path(__file__).resolve().parents[1]  # backend/ai
    templates_dir = base_dir / "prompts" / "templates"

    provider = OpenAIProvider()

    knowledge = load_knowledge(
        base_dir=base_dir,
        genre=spec.get("genre", "urban"),
        style=spec.get("style", "noir"),
        world=spec.get("world", "default_world"),
    )

    prompt_builder = PromptBuilderAgent(provider, templates_dir)
    writer = WriterAgent(provider, templates_dir)
    editor = EditorAgent(provider, templates_dir)
    guard = GuardAgent(provider, templates_dir)

    writing_prompt = prompt_builder.build(spec=spec, chapter=ch, knowledge=knowledge, memory="")
    draft = writer.write(spec=spec, chapter=ch, writing_prompt=writing_prompt)

    guard_result = guard.check(spec=spec, draft=draft)
    if guard_result != "PASS":
        # 让 writer 按修复指令重写一次
        fix_prompt = f"你刚才输出缺少结构或不合规。请按以下修复指令重写：\n{guard_result}\n\n务必严格输出完整结构。"
        draft = writer.write(spec=spec, chapter=ch, writing_prompt=writing_prompt + "\n\n" + fix_prompt)

    edit_notes = editor.review(spec=spec, chapter=ch, draft=draft)

    out_dir = project_dir / "chapters"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ch{args.chapter:02d}.md"

    out_path.write_text(
        draft.strip() + "\n\n---\n\n## 编辑建议\n" + edit_notes.strip() + "\n",
        encoding="utf-8",
    )

    print(f"[OK] Wrote: {out_path}")

if __name__ == "__main__":
    main()
