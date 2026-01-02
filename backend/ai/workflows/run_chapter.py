#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

# 你项目里已有 provider
from ..provider import get_ai_provider


# -------------------------
# Utils
# -------------------------

def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")

def write_text(p: Path, s: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8")

def read_json(p: Path) -> Any:
    return json.loads(read_text(p))

def write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def safe_yaml_load(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError("Missing dependency: pyyaml. Install: pip install pyyaml") from e
    return yaml.safe_load(read_text(path)) or {}

def yaml_dump(obj: Any) -> str:
    try:
        import yaml  # type: ignore
    except Exception:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    return yaml.safe_dump(obj, allow_unicode=True, sort_keys=False)

def ch_id_to_slug(ch: int) -> str:
    return f"ch{ch:02d}"

def extract_outline_for_chapter(outline_yml: Dict[str, Any], chapter: int) -> str:
    """
    兼容 outline.yml 的多种写法：
    - chapters: [{id:1,title:"",beats:[...]}]
    - chapters: {1: "..."} 或 {ch01: "..."}
    - 或者直接是 list
    """
    chs = outline_yml.get("chapters", outline_yml)
    if isinstance(chs, list):
        for it in chs:
            if isinstance(it, dict) and int(it.get("id", -1)) == int(chapter):
                return yaml_dump(it)
    if isinstance(chs, dict):
        # key could be "1" / 1 / "ch01"
        for k in (str(chapter), chapter, ch_id_to_slug(chapter)):
            if k in chs:
                v = chs[k]
                return yaml_dump(v) if not isinstance(v, str) else v
    # fallback: dump whole outline (not ideal but usable)
    return yaml_dump(outline_yml)

def normalize_whitespace(s: str) -> str:
    # 轻微清理：避免模型输出前后空行太多
    s = s.strip()
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s + "\n"

# -------------------------
# Memory builder (LLM-based)
# -------------------------

MEMORY_PROMPT = """你是小说连载编辑，请把“本章正文”总结成下一章可直接续写的【记忆 JSON】。
要求：
- 只输出 JSON，不能有任何额外文字。
- JSON 字段必须包含：
  - "chapter": 章节号(int)
  - "characters": 主要人物列表（每项包含 name, status, goals, relationships）
  - "last_scene": 用 4-8 句描述本章结尾的具体场景（地点、在场人物、紧张点、下一步马上要发生什么）
  - "open_loops": 未解悬念列表（不少于 3 条）
  - "inventory_or_clues": 关键物品/线索（可空列表）
  - "tone": 本章基调（例如 冷、压抑、紧张、狠）
- 不要编造超出正文的设定。

本章正文如下：
"""


def build_memory_with_llm(provider, chapter: int, chapter_text: str) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": MEMORY_PROMPT + "\n\n" + chapter_text}
    ]
    resp = provider.chat(messages, temperature=0.2, max_tokens=1200)
    # 尝试提取 JSON
    txt = resp.strip()
    # 有些模型会包 ```json
    txt = re.sub(r"^```json\s*", "", txt)
    txt = re.sub(r"\s*```$", "", txt)
    try:
        obj = json.loads(txt)
    except Exception:
        # fallback：尽量截取第一个 {...}
        m = re.search(r"\{.*\}", txt, flags=re.S)
        if not m:
            return {
                "chapter": chapter,
                "characters": [],
                "last_scene": ["（记忆生成失败）"],
                "open_loops": ["（记忆生成失败）"],
                "inventory_or_clues": [],
                "tone": "unknown",
            }
        obj = json.loads(m.group(0))
    obj["chapter"] = int(chapter)
    return obj


# -------------------------
# Main: render writer.md and write chapter
# -------------------------

@dataclass
class Paths:
    project_dir: Path
    spec_yml: Path
    outline_yml: Path
    chapters_dir: Path
    style_bible_json: Path
    writer_tpl: Path

def resolve_paths(project: Path) -> Paths:
    project_dir = project.resolve()
    spec_yml = project_dir / "spec.yml"
    if not spec_yml.exists():
        spec_yml = project_dir / "spec.yaml"
    outline_yml = project_dir / "outline.yml"
    if not outline_yml.exists():
        outline_yml = project_dir / "outline.yaml"

    chapters_dir = project_dir / "chapters"

    # 你现在的 style_bible 在截图里是 data/knowledge/style_bible.json
    style_bible_json = Path("data/knowledge/style_bible.json").resolve()

    writer_tpl = Path("backend/ai/prompts/templates/writer.md").resolve()

    return Paths(
        project_dir=project_dir,
        spec_yml=spec_yml,
        outline_yml=outline_yml,
        chapters_dir=chapters_dir,
        style_bible_json=style_bible_json,
        writer_tpl=writer_tpl,
    )

def load_prev_memory(chapters_dir: Path, chapter: int) -> str:
    if chapter <= 1:
        return ""
    prev_slug = ch_id_to_slug(chapter - 1)
    prev_mem = chapters_dir / f"{prev_slug}_memory.json"
    if prev_mem.exists():
        return read_text(prev_mem)
    return ""

def render_writer_prompt(writer_md: str, style_bible: str, spec: str, chapter_outline: str, memory: str, knowledge: str) -> str:
    # 兼容你模板里的占位符 {style_bible} {spec} {chapter_outline} {memory} {knowledge}
    prompt = writer_md
    prompt = prompt.replace("{style_bible}", style_bible)
    prompt = prompt.replace("{spec}", spec)
    prompt = prompt.replace("{chapter_outline}", chapter_outline)
    prompt = prompt.replace("{memory}", memory if memory else "（无）")
    prompt = prompt.replace("{knowledge}", knowledge if knowledge else "（无）")
    return prompt

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True, help="project dir, e.g. data/projects/book_001")
    ap.add_argument("--chapter", required=True, type=int, help="chapter id, e.g. 1")
    ap.add_argument("--temperature", type=float, default=0.75)
    ap.add_argument("--max_tokens", type=int, default=3500)
    ap.add_argument("--dry_run", action="store_true", help="only render prompt to .prompt.txt without calling LLM")
    args = ap.parse_args()

    paths = resolve_paths(Path(args.project))
    ch = int(args.chapter)
    slug = ch_id_to_slug(ch)

    # 必要文件检查
    if not paths.writer_tpl.exists():
        raise FileNotFoundError(f"Missing writer template: {paths.writer_tpl}")
    if not paths.style_bible_json.exists():
        raise FileNotFoundError(f"Missing style_bible: {paths.style_bible_json}")
    if not paths.spec_yml.exists():
        raise FileNotFoundError(f"Missing spec.yml in project: {paths.spec_yml}")
    if not paths.outline_yml.exists():
        raise FileNotFoundError(f"Missing outline.yml in project: {paths.outline_yml}")

    writer_md = read_text(paths.writer_tpl)
    style_bible_obj = read_json(paths.style_bible_json)
    style_bible = json.dumps(style_bible_obj, ensure_ascii=False, indent=2)

    spec_obj = safe_yaml_load(paths.spec_yml)
    spec_txt = yaml_dump(spec_obj)

    outline_obj = safe_yaml_load(paths.outline_yml)
    chapter_outline = extract_outline_for_chapter(outline_obj, ch)

    prev_memory = load_prev_memory(paths.chapters_dir, ch)

    # knowledge：如果你以后还想塞“世界设定补充”，可以从 data/knowledge/worlds/default_world.yml 读，这里先空着
    knowledge = ""

    full_prompt = render_writer_prompt(
        writer_md=writer_md,
        style_bible=style_bible,
        spec=spec_txt,
        chapter_outline=chapter_outline,
        memory=prev_memory,
        knowledge=knowledge,
    )

    prompt_dump_path = paths.chapters_dir / f"{slug}.prompt.txt"
    write_text(prompt_dump_path, full_prompt)

    if args.dry_run:
        print(f"[DRY RUN] rendered prompt -> {prompt_dump_path}")
        return

    provider = get_ai_provider()

    # 关键：system 里放完整 writer 规则 + 章节材料
    messages = [{"role": "system", "content": full_prompt}]

    chapter_text = provider.chat(messages, temperature=args.temperature, max_tokens=args.max_tokens)
    chapter_text = normalize_whitespace(chapter_text)

    out_md = paths.chapters_dir / f"{slug}.md"
    write_text(out_md, chapter_text)

    # 生成记忆（用于下一章续写）
    mem_obj = build_memory_with_llm(provider, ch, chapter_text)
    out_mem = paths.chapters_dir / f"{slug}_memory.json"
    write_json(out_mem, mem_obj)

    print(f"[OK] chapter -> {out_md}")
    print(f"[OK] memory  -> {out_mem}")
    print(f"[OK] prompt  -> {prompt_dump_path}")


if __name__ == "__main__":
    main()
