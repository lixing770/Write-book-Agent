#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml

from .llm_client import OpenAILLM


DEFAULT_WRITE_PROMPT = """你是小说学习笔记作者。你将基于结构化的人物与关系数据，输出一份“人物关系学习文档”。
要求：中文，条理清晰，可用于复习记忆。
结构必须包含：
1) 核心人物总览（主角/核心圈/关键对立）
2) 关系网络总览（用文字总结，不要画图也行）
3) 关系演化时间线（按阶段/章节概括）
4) 重点关系学习卡片（每对关系一小节：如何建立、关键事件、当前状态、证据）
5) 易混人物与别名对照表（如有）

注意：不要杜撰，只能基于给定数据。
"""

SYSTEM = "You write study notes from structured character-relation graphs."


def _load_yaml(p: Path) -> Any:
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, help="out/")
    ap.add_argument("--prompt", default="prompts/write_study_notes.md")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    chars_path = out_dir / "characters.yml"
    rels_path = out_dir / "relations.yml"

    if not chars_path.exists() or not rels_path.exists():
        raise RuntimeError("Missing characters.yml or relations.yml. Run merge.py first.")

    characters: List[Dict[str, Any]] = _load_yaml(chars_path) or []
    relations: List[Dict[str, Any]] = _load_yaml(rels_path) or []

    prompt_path = Path(args.prompt)
    user_prompt = prompt_path.read_text(encoding="utf-8").strip() if prompt_path.exists() else DEFAULT_WRITE_PROMPT

    # Build a compact input package (avoid huge tokens)
    # Keep top characters by evidence_refs length
    characters_sorted = sorted(characters, key=lambda x: len(x.get("evidence_refs") or []), reverse=True)
    characters_compact = characters_sorted[:180]

    relations_sorted = sorted(relations, key=lambda x: float(x.get("confidence", 0.5)), reverse=True)
    relations_compact = relations_sorted[:450]

    payload = {
        "characters": characters_compact,
        "relations": relations_compact,
        "note": "字段说明：relations[].evidence 为证据片段（chunk_id, quote, span）。只可基于这些信息写作。",
    }

    llm = OpenAILLM()
    user = user_prompt + "\n\n=== DATA(JSON) ===\n" + json.dumps(payload, ensure_ascii=False, indent=2)

    md = llm.respond_text(system=SYSTEM, user=user)

    # timeline (cheap heuristic)
    timeline_lines = ["# 关系演化时间线（粗略）", ""]
    for r in relations_sorted[:120]:
        a, b, t = r.get("from"), r.get("to"), r.get("type")
        status = r.get("status", "")
        first = r.get("first_seen_chunk", "")
        timeline_lines.append(f"- {a} — {t} → {b}（{status}；首次片段 {first}）")

    (out_dir / "study_notes.md").write_text(md.strip() + "\n", encoding="utf-8")
    (out_dir / "timeline.md").write_text("\n".join(timeline_lines) + "\n", encoding="utf-8")

    print(f"[OK] study_notes -> {out_dir / 'study_notes.md'}")
    print(f"[OK] timeline    -> {out_dir / 'timeline.md'}")


if __name__ == "__main__":
    main()
