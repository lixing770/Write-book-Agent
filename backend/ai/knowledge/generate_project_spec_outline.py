#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract a CHAPTER OUTLINE TEMPLATE (structure/pacing/hook patterns) from a JSONL corpus.
Robust YAML post-processing:
- Fix common YAML format issues from LLM outputs
- If still invalid, ask LLM to "repair" into valid YAML

Output: YAML (outline_template.yml)

Usage:
  python -m backend.ai.knowledge.extract_outline_template \
    --jsonl "data/corpora/jsonl/仙逆_utf8.jsonl" \
    --out  "data/knowledge/outline_template.yml"
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List

import yaml

from ..provider import get_ai_provider


# -----------------------
# IO helpers
# -----------------------

def _read_jsonl_texts(jsonl_path: Path) -> List[str]:
    texts: List[str] = []
    with jsonl_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                texts.append(line)
                continue

            if isinstance(obj, dict):
                for k in ("text", "content", "paragraph"):
                    v = obj.get(k)
                    if isinstance(v, str) and v.strip():
                        texts.append(v)
                        break
                else:
                    v = obj.get("data", {}).get("text") if isinstance(obj.get("data"), dict) else None
                    if isinstance(v, str) and v.strip():
                        texts.append(v)
            elif isinstance(obj, str):
                texts.append(obj)
    return texts


def _sanitize(s: str) -> str:
    s = re.sub(r"sk-[A-Za-z0-9]{10,}", "[REDACTED_KEY]", s)
    return s


def _excerpt(texts: List[str], max_chars: int, seed: int) -> str:
    rng = random.Random(seed)
    idxs = list(range(len(texts)))
    rng.shuffle(idxs)

    buf: List[str] = []
    total = 0
    for i in idxs:
        t = texts[i].strip()
        if len(t) < 80:
            continue
        if len(t) > 3500:
            t = t[:3500]
        t = _sanitize(t)
        if total + len(t) + 2 > max_chars:
            break
        buf.append(t)
        total += len(t) + 2

    if not buf:
        return "\n".join(_sanitize(x[:1000]) for x in texts[:30])
    return "\n\n".join(buf)


# -----------------------
# Prompts
# -----------------------

SYSTEM_PROMPT = """你是一名“网文连载章节结构”分析师。
只允许输出“结构模板与套路”，严禁复述/改写任何原著句子，严禁输出原著具体剧情链条、人物名、地名、宗门名。
输出必须是【严格 YAML】，能被 yaml.safe_load 解析。
不要输出 Markdown 代码块。
"""

USER_PROMPT = """以下是语料片段集合（仅用于统计章节结构的共性，不允许引用原文）。

【语料片段】
{CORPUS}

请抽取“章节大纲模板（outline template）”，目标是用来生成同类型新书的大纲。
要求输出 YAML（顶层必须是 mapping），至少包含：

chapter_skeleton: 单章结构槽位列表。每个槽位包含：
  - slot
  - purpose
  - common_methods (list)
  - optional_variants (list, >=3)
  - typical_length_ratio (例如 "10-15%")

pacing_profile: 用【mapping】表示（不要用 list），包含：
  overall_rhythm
  conflict_density
  hook_frequency
  reveal_rhythm
  tempo_adjustment

hook_types: list（>=15）
reveal_patterns: list
conflict_catalog: list
constraints: list（包含：不要写前序、不要出现现代网络词、不要抄原著等）

注意：
- YAML 语法必须正确（冒号、缩进、列表都要合规）
- 不要出现任何原著专有名词，只能输出“泛化模板 + 占位符”。
"""


REPAIR_SYSTEM = """你是 YAML 修复器。你只做一件事：把输入修复成【严格合法 YAML】。
要求：
- 顶层必须是 mapping
- 不要输出 Markdown 代码块
- 只输出 YAML
"""

REPAIR_USER = """下面这段文本 intended as YAML，但无法解析。请修复成严格合法 YAML（保持语义不变）。

【BROKEN】
{BROKEN}
"""


# -----------------------
# YAML fixing
# -----------------------

def _strip_fences(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```yaml\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^```\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _fix_common_yaml_issues(s: str) -> str:
    """
    Fix common LLM YAML mistakes:
    - "-信息" -> "- 信息" (missing space after dash)
    - Weird fullwidth colon usage is usually fine in YAML as part of string,
      but for keys we keep ASCII ":" already. The model mostly uses ":" already.
    - Tabs -> spaces
    """
    s = s.replace("\t", "  ")

    # Ensure "-<nonspace>" becomes "- <nonspace>"
    s = re.sub(r"(?m)^(\s*)-(\S)", r"\1- \2", s)

    # Sometimes keys like " -信息揭露节奏:" appear; above fixes it.
    # Also normalize stray "："
    # If line looks like "key： value" convert to "key: value"
    s = re.sub(r"(?m)^(\s*[\w\u4e00-\u9fff][\w\u4e00-\u9fff _-]*)：\s*", r"\1: ", s)

    return s


def _parse_yaml_or_raise(s: str) -> Dict[str, Any]:
    obj = yaml.safe_load(s)
    if not isinstance(obj, dict):
        raise ValueError("YAML root must be mapping")
    return obj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_chars", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--temperature", type=float, default=0.25)
    ap.add_argument("--max_tokens", type=int, default=2500)
    ap.add_argument("--repair_max_tokens", type=int, default=1800)
    args = ap.parse_args()

    texts = _read_jsonl_texts(Path(args.jsonl))
    if not texts:
        raise RuntimeError("No texts read from jsonl.")

    corpus = _excerpt(texts, args.max_chars, args.seed)

    provider = get_ai_provider()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT.replace("{CORPUS}", corpus)},
    ]
    resp = provider.chat(messages, temperature=args.temperature, max_tokens=args.max_tokens)

    raw = _strip_fences(resp)
    fixed = _fix_common_yaml_issues(raw)

    # 1) try parse after local fixes
    try:
        data = _parse_yaml_or_raise(fixed)
    except Exception:
        # 2) ask model to repair YAML
        repair_messages = [
            {"role": "system", "content": REPAIR_SYSTEM},
            {"role": "user", "content": REPAIR_USER.replace("{BROKEN}", fixed[:12000])},
        ]
        repaired = provider.chat(repair_messages, temperature=0.0, max_tokens=args.repair_max_tokens)
        repaired = _fix_common_yaml_issues(_strip_fences(repaired))

        # 3) parse repaired or raise with good debugging
        try:
            data = _parse_yaml_or_raise(repaired)
            fixed = repaired
        except Exception as e2:
            raise RuntimeError(
                f"Invalid YAML even after repair: {e2}\n\nRAW(head):\n{raw[:1200]}\n\nFIXED(head):\n{fixed[:1200]}"
            ) from e2

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # write canonical YAML (re-dump) to guarantee validity
    out_path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"[OK] Wrote outline template: {out_path}")


if __name__ == "__main__":
    main()
