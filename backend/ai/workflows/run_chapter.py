#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from backend.ai.agents.writer import WriterAgent

# provider 获取（项目里一般二选一）
try:
    from backend.ai.provider import get_ai_provider
except Exception:
    from backend.ai.openai_provider import get_ai_provider  # type: ignore


YamlObj = Union[Dict[str, Any], list]


@dataclass
class ChapterObj:
    number: int
    title: str
    outline: Dict[str, Any]


# =========================
# IO helpers
# =========================
def _read_text(p: Path) -> str:
    if not p.exists():
        raise RuntimeError(f"File not found: {p}")
    return p.read_text(encoding="utf-8").strip()


def _read_yaml(p: Path) -> YamlObj:
    if not p.exists():
        raise RuntimeError(f"YAML file not found: {p}")
    txt = p.read_text(encoding="utf-8").strip()
    if not txt:
        raise RuntimeError(f"YAML file is empty: {p}")
    obj = yaml.safe_load(txt)
    if not isinstance(obj, (dict, list)):
        raise RuntimeError(f"Invalid YAML root (expected mapping/list): {p}")
    return obj


def _yaml_dump(obj: Any) -> str:
    if isinstance(obj, str):
        return obj
    return yaml.safe_dump(obj, allow_unicode=True, sort_keys=False)


def _read_json(p: Path) -> Dict[str, Any]:
    if not p or not p.exists():
        return {}
    try:
        v = json.loads(p.read_text(encoding="utf-8"))
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}


def _write_text(p: Path, s: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8")


# =========================
# Outline parsing
# =========================
def _load_chapter_outline(outline_yml: YamlObj, n: int) -> ChapterObj:
    """
    支持：
      1) {chapters: [ {number:1,...}, ... ]}
      2) {outline:  [ {number:1,...}, ... ]}
      3) [ {number:1,...}, ... ]
    """
    ch: Optional[Dict[str, Any]] = None

    if isinstance(outline_yml, dict):
        if "chapters" in outline_yml:
            chapters = outline_yml["chapters"]
            if isinstance(chapters, list):
                for item in chapters:
                    if isinstance(item, dict) and int(item.get("number", -1)) == n:
                        ch = item
                        break
            elif isinstance(chapters, dict):
                v = chapters.get(str(n))
                if isinstance(v, dict):
                    ch = v

        elif isinstance(outline_yml.get("outline"), list):
            for item in outline_yml["outline"]:
                if isinstance(item, dict) and int(item.get("number", -1)) == n:
                    ch = item
                    break

    elif isinstance(outline_yml, list):
        for item in outline_yml:
            if isinstance(item, dict) and int(item.get("number", -1)) == n:
                ch = item
                break

    if ch is None:
        raise RuntimeError(f"Chapter {n} not found in outline.yml")

    title = str(ch.get("title") or ch.get("name") or f"第{n}章")
    return ChapterObj(number=n, title=title, outline=ch)


def _format_chapter_outline_for_prompt(ch: ChapterObj) -> str:
    keep: Dict[str, Any] = {}
    for k in ("title", "summary", "beats", "scenes", "goals", "conflicts", "reveal", "hook", "notes", "pov"):
        if k in ch.outline:
            keep[k] = ch.outline[k]
    if not keep:
        keep = dict(ch.outline)

    keep["number"] = ch.number
    keep["title"] = ch.title
    return yaml.safe_dump(keep, allow_unicode=True, sort_keys=False)


# =========================
# Cast bible helpers
# =========================
def _extract_protagonist_name(cast_bible_text: str) -> str:
    """
    从 cast_bible.yml 提取 protagonist.name
    """
    try:
        obj = yaml.safe_load(cast_bible_text)
        if isinstance(obj, dict):
            pro = obj.get("protagonist")
            if isinstance(pro, dict):
                name = pro.get("name")
                if isinstance(name, str) and name.strip():
                    return name.strip()
    except Exception:
        pass
    return "主角"


def _first_n_sentences(text: str, n: int = 3) -> str:
    parts = re.split(r"(?<=[。！？!?])\s*", (text or "").strip())
    return "".join(parts[:n]).strip()


def _protagonist_ok(text: str, protagonist: str) -> bool:
    if not text or not protagonist:
        return False
    head3 = _first_n_sentences(text, 3)
    # 规则：前三句必须出现主角名；全文至少出现 2 次
    return (protagonist in head3) and (text.count(protagonist) >= 2)


def _contains_banned_name(text: str, protagonist: str) -> bool:
    """
    硬杀你截图里出现过的误名：沈墨
    你如果还有其他常见漂移名，可以往 blacklist 里加。
    """
    if not text:
        return True
    blacklist = ["沈墨"]
    head = (text[:400] or "")
    # 如果开头没有主角且出现黑名单名，直接判定漂移
    for b in blacklist:
        if b in head and protagonist not in head:
            return True
    return False


def _enforce_opening(text: str, opening: str, protagonist: str) -> str:
    """
    如果第一句不是以主角开头，则强行前置 opening 作为锚点。
    """
    body = (text or "").strip()
    if not body:
        return opening
    first = _first_n_sentences(body, 1)
    if not first.startswith(protagonist):
        return opening + "\n\n" + body
    return body


# =========================
# Template render (writer.md)
# =========================
def _render_writer_template(
    writer_md: str,
    *,
    style_bible: str,
    spec: str,
    chapter_outline: str,
    memory: str,
    knowledge: str,
) -> str:
    out = writer_md
    out = out.replace("{style_bible}", style_bible)
    out = out.replace("{spec}", spec)
    out = out.replace("{chapter_outline}", chapter_outline)
    out = out.replace("{memory}", memory)
    out = out.replace("{knowledge}", knowledge)
    return out


# =========================
# Memory builder (optional but useful)
# =========================
def _build_memory(provider, chapter_text: str, max_tokens: int = 900) -> Dict[str, Any]:
    system = (
        "You are a writing assistant.\n"
        "Summarize the chapter into a compact JSON object for continuing the story.\n"
        "Return JSON only.\n"
        "Keys: summary, characters, plot_points, world_facts, open_threads.\n"
        "Keep it concise."
    )
    user = f"CHAPTER:\n{(chapter_text or '')[:12000]}"
    resp = provider.chat(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.1,
        max_tokens=max_tokens,
    )
    try:
        obj = json.loads(resp)
        return obj if isinstance(obj, dict) else {"summary": resp}
    except Exception:
        return {"summary": resp}


# =========================
# Main
# =========================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True, help="e.g. data/projects/book_001")
    ap.add_argument("--chapter", required=True, type=int)
    ap.add_argument("--templates_dir", default="backend/ai/prompts/templates")
    ap.add_argument("--style_bible", required=True, help="e.g. data/knowledge/style_profiles/style_bible.json")
    ap.add_argument("--knowledge_dir", default="data/knowledge")

    # WriterAgent.write 不吃这两个参数，但留着以后升级用
    ap.add_argument("--max_rewrite", type=int, default=3)
    args = ap.parse_args()

    project_dir = Path(args.project)
    templates_dir = Path(args.templates_dir)

    # 项目文件
    spec_path = project_dir / "spec.yml"
    outline_path = project_dir / "outline.yml"
    cast_bible_path = project_dir / "cast_bible.yml"
    proj_outline_path = project_dir / "project_spec_outline.yml"

    # 模板文件
    writer_md_path = templates_dir / "writer.md"

    # 必需检查
    if not spec_path.exists():
        raise RuntimeError(f"Missing spec.yml: {spec_path}")
    if not outline_path.exists():
        raise RuntimeError(f"Missing outline.yml: {outline_path}")
    if not cast_bible_path.exists():
        raise RuntimeError(f"Missing cast_bible.yml: {cast_bible_path}")
    if not writer_md_path.exists():
        raise RuntimeError(f"Missing writer.md: {writer_md_path}")

    # 读取输入
    spec_obj = _read_yaml(spec_path)
    spec_text = _yaml_dump(spec_obj)

    outline_obj = _read_yaml(outline_path)
    ch = _load_chapter_outline(outline_obj, args.chapter)
    chapter_outline_str = _format_chapter_outline_for_prompt(ch)

    cast_bible_text = _read_text(cast_bible_path)
    protagonist = _extract_protagonist_name(cast_bible_text)

    style_bible_text = _read_text(Path(args.style_bible))
    writer_md = _read_text(writer_md_path)

    # 记忆
    memory_dir = project_dir / "memory"
    prev_mem_path = memory_dir / f"ch{args.chapter-1:02d}.json" if args.chapter > 1 else None
    prev_memory = _read_json(prev_mem_path) if prev_mem_path else {}
    memory_str = json.dumps(prev_memory, ensure_ascii=False, indent=2) if prev_memory else ""

    # project_spec_outline 工具箱（可选，但你说你要用）
    proj_outline_text = ""
    if proj_outline_path.exists():
        proj_outline_text = _yaml_dump(_read_yaml(proj_outline_path))

    # extra knowledge（可选）
    knowledge_str = ""
    kb_dir = Path(args.knowledge_dir)
    for candidate in [kb_dir / "world.md", kb_dir / "lore.md", kb_dir / "glossary.yml"]:
        if candidate.exists():
            knowledge_str += f"\n\n=== EXTRA_KNOWLEDGE ({candidate.name}) ===\n{candidate.read_text(encoding='utf-8').strip()}\n"

    # 强约束 + cast + toolkit
    knowledge_pack = (
        "### HARD CONSTRAINTS (MUST FOLLOW)\n"
        f"1) Protagonist must be EXACTLY: {protagonist}. Do NOT rename. Do NOT replace protagonist.\n"
        f"2) In the first 3 sentences, MUST mention '{protagonist}'.\n"
        f"3) POV stays with '{protagonist}' unless chapter outline explicitly requests POV switch.\n"
        "If violated, output is invalid.\n\n"
        "=== CAST_BIBLE (YAML) ===\n"
        f"{cast_bible_text}\n"
    )

    if proj_outline_text.strip():
        knowledge_pack += "\n=== PROJECT_SPEC_OUTLINE TOOLKIT (YAML) ===\n" + proj_outline_text + "\n"
    if knowledge_str.strip():
        knowledge_pack += "\n" + knowledge_str.strip() + "\n"

    # 强制开头锚点（必须逐字使用）——保证开头即锁主角
    forced_opening = (
        f"{protagonist}踏入夜色，风里有潮湿的铁锈味。"
        f"他抬眼望去，远处的光像被谁掐住了喉咙般忽明忽暗。"
        f"{protagonist}知道，今夜之后，一切都会变得不同。"
    )

    # 组装最终 prompt：填满 writer.md 的 5 个占位符
    final_prompt = _render_writer_template(
        writer_md,
        style_bible=style_bible_text,
        spec=spec_text,
        chapter_outline=chapter_outline_str,
        memory=memory_str,
        knowledge=knowledge_pack + "\n\n【必须逐字使用的开头锚点】\n" + forced_opening + "\n",
    )

    # provider + writer
    provider = get_ai_provider()
    writer = WriterAgent(provider, templates_dir)

    chapter_payload = {
        "number": ch.number,
        "title": ch.title,
        "outline": chapter_outline_str,
        "protagonist": protagonist,
    }

    # 第一次生成
    chapter_text = writer.write(
        spec=spec_obj if isinstance(spec_obj, dict) else {"spec": spec_obj},
        chapter=chapter_payload,
        writing_prompt=final_prompt,
    )

    # 开头锚点强制前置（即便模型写对也不影响，只会更稳）
    chapter_text = _enforce_opening(chapter_text, forced_opening, protagonist)

    # 纠偏重写循环：拿“错误正文”回灌让它纠偏（不是自由再写）
    for attempt in range(1, int(args.max_rewrite) + 1):
        drift = (not _protagonist_ok(chapter_text, protagonist)) or _contains_banned_name(chapter_text, protagonist)
        if not drift:
            break

        print(f"[WARN] protagonist drift detected. Rewrite attempt {attempt}/{args.max_rewrite}...")

        fix_instruction = (
            "\n\n### 强制纠偏重写（必须执行）\n"
            f"你必须将本章的主角统一为：{protagonist}。\n"
            "1) 删除或降级任何“新主角”（例如：沈墨），他们只能作为配角出现。\n"
            f"2) 正文第一句必须以“{protagonist}”开头，前三句必须出现“{protagonist}”。\n"
            "3) 保留本章发生的事件逻辑与氛围，但叙事中心必须回到主角。\n"
            "4) 不要道歉，不要解释，直接输出重写后的正文。\n"
            "\n"
            "【必须逐字使用的开头】\n"
            f"{forced_opening}\n"
            "\n"
            "【你上一版的错误正文如下（用于纠偏重写）】\n"
            "-----\n"
            f"{(chapter_text or '')[:12000]}\n"
            "-----\n"
        )

        chapter_text = writer.write(
            spec=spec_obj if isinstance(spec_obj, dict) else {"spec": spec_obj},
            chapter=chapter_payload,
            writing_prompt=final_prompt + fix_instruction,
        )
        chapter_text = _enforce_opening(chapter_text, forced_opening, protagonist)

    # 保存章节
    chapter_out = project_dir / "chapters" / f"ch{args.chapter:02d}.md"
    _write_text(chapter_out, f"# {ch.title}\n\n{(chapter_text or '').strip()}\n")
    print(f"[OK] chapter  -> {chapter_out}")

    # 保存记忆
    mem = _build_memory(provider, chapter_text)
    memory_dir.mkdir(parents=True, exist_ok=True)
    mem_path = memory_dir / f"ch{args.chapter:02d}.json"
    _write_text(mem_path, json.dumps(mem, ensure_ascii=False, indent=2))
    print(f"[OK] memory   -> {mem_path}")


if __name__ == "__main__":
    main()
