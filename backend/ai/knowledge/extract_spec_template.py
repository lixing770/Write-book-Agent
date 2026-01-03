#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract a Xianxia-style SPEC TEMPLATE from a JSONL corpus (e.g., 仙逆),
focusing on STRUCTURE (systems, factions, resources, conflicts, pacing),
NOT copying plot or proprietary text.

Output: YAML (spec_template.yml)

Usage:
  python -m backend.ai.knowledge.extract_spec_template \
    --jsonl "data/corpora/jsonl/仙逆_utf8.jsonl" \
    --out  "data/knowledge/spec_template.yml" \
    --max_chars 60000 \
    --temperature 0.2 \
    --max_tokens 2500
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from ..provider import get_ai_provider


def _read_jsonl_texts(jsonl_path: Path) -> List[str]:
    """
    Compat with various jsonl schemas:
      - {"text": "..."}
      - {"content": "..."}
      - {"paragraph": "..."}
      - {"data": {"text": "..."}}
      - {"messages":[...]} (fallback: join assistant/user content)
    Returns list of text blocks.
    """
    texts: List[str] = []
    with jsonl_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                # raw text line
                texts.append(line)
                continue

            def pick_text(o: Any) -> Optional[str]:
                if isinstance(o, str):
                    return o
                if isinstance(o, dict):
                    for k in ("text", "content", "paragraph"):
                        if k in o and isinstance(o[k], str) and o[k].strip():
                            return o[k]
                    if "data" in o and isinstance(o["data"], dict):
                        t = o["data"].get("text")
                        if isinstance(t, str) and t.strip():
                            return t
                    if "messages" in o and isinstance(o["messages"], list):
                        parts = []
                        for m in o["messages"]:
                            if isinstance(m, dict):
                                c = m.get("content")
                                if isinstance(c, str) and c.strip():
                                    parts.append(c.strip())
                        if parts:
                            return "\n".join(parts)
                return None

            t = pick_text(obj)
            if t:
                texts.append(t)
    return texts


def _sanitize_for_prompt(s: str) -> str:
    # Remove obviously sensitive tokens-like strings (avoid leaking keys)
    s = re.sub(r"sk-[A-Za-z0-9]{10,}", "[REDACTED_KEY]", s)
    return s


def _build_corpus_excerpt(texts: List[str], max_chars: int, seed: int) -> str:
    rng = random.Random(seed)
    # sample and concatenate until max_chars
    idxs = list(range(len(texts)))
    rng.shuffle(idxs)

    buf: List[str] = []
    total = 0
    for i in idxs:
        t = texts[i].strip()
        if not t:
            continue
        t = _sanitize_for_prompt(t)
        # prefer mid-length paragraphs (not too short/too huge)
        if len(t) < 80:
            continue
        if len(t) > 4000:
            t = t[:4000]
        if total + len(t) + 2 > max_chars:
            break
        buf.append(t)
        total += len(t) + 2

    # fallback if nothing selected
    if not buf:
        joined = "\n".join(_sanitize_for_prompt(x[:1000]) for x in texts[:50] if x.strip())
        return joined[:max_chars]
    return "\n\n".join(buf)


SPEC_TEMPLATE_SYSTEM_PROMPT = """你是一名“修真长篇/仙侠升级流”的结构分析师。
你只允许输出“抽象结构/套路/模板”，严禁复述或改写任何原著段落，严禁输出原著具体情节链条、具体人物/地名/宗门名。
你的目标是从给定语料中抽取一种“可迁移模板”，用于生成同类型新故事的 spec（世界观设定）。
输出必须是 YAML，字段固定且完整，便于机器读取。
"""

SPEC_TEMPLATE_USER_PROMPT = """下面是小说语料的片段集合（仅用于统计写法与世界观结构的共性，不允许引用原文）。

【语料片段】
{CORPUS}

任务：请从中抽取“修真/仙侠升级流”世界观 spec 的通用模板，必须包含但不限于这些维度：
1) 修炼体系（境界层级模板、晋升条件、常见瓶颈、寿元/天劫/心魔等机制）
2) 资源体系（灵石、丹药、法宝、功法、材料、洞府、阵法、秘境、传承等）
3) 势力结构（宗门/家族/朝廷/坊市/禁地势力/上界势力等层级与互动）
4) 社会规则（交易、规矩、因果、恩怨、门规、利益格局、灰色地带）
5) 常见冲突类型库（夺宝、追杀、试炼、宗门倾轧、秘境争夺、因果反噬等）
6) 叙事推进引擎（小目标→升级→更大谜团；信息揭露策略；章末钩子类型）
7) 主角成长模板（出身/性格/执念/底线；“狠”与“守”的平衡；成长阶段）
8) 风格约束（但只写“规则”，不要写任何原著专有名词）

输出 YAML，顶层字段建议：
- meta
- cultivation_system
- resource_system
- factions
- society_rules
- conflict_library
- narrative_engine
- protagonist_template
- hook_library
- taboo (禁止项：避免抄袭的提醒)

注意：不要出现任何原著专有名词、人物名、地名、宗门名。只能用“占位符/泛化描述”。
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="Path to corpus jsonl")
    ap.add_argument("--out", required=True, help="Output YAML path (spec_template.yml)")
    ap.add_argument("--max_chars", type=int, default=60000, help="Max chars of excerpt for analysis")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max_tokens", type=int, default=2500)
    args = ap.parse_args()

    jsonl_path = Path(args.jsonl)
    out_path = Path(args.out)

    texts = _read_jsonl_texts(jsonl_path)
    if not texts:
        raise RuntimeError(f"No texts found in jsonl: {jsonl_path}")

    excerpt = _build_corpus_excerpt(texts, max_chars=args.max_chars, seed=args.seed)

    provider = get_ai_provider()
    messages = [
        {"role": "system", "content": SPEC_TEMPLATE_SYSTEM_PROMPT},
        {"role": "user", "content": SPEC_TEMPLATE_USER_PROMPT.replace("{CORPUS}", excerpt)},
    ]
    resp = provider.chat(messages, temperature=args.temperature, max_tokens=args.max_tokens)

    # Strip markdown fences if present
    content = resp.strip()
    content = re.sub(r"^```yaml\s*", "", content, flags=re.IGNORECASE)
    content = re.sub(r"^```\s*", "", content)
    content = re.sub(r"\s*```$", "", content)

    # Validate YAML parse
    try:
        data = yaml.safe_load(content)
        if not isinstance(data, dict):
            raise ValueError("YAML root must be a mapping")
    except Exception as e:
        raise RuntimeError(f"Model output is not valid YAML. Error: {e}\n\nRAW:\n{content[:2000]}") from e

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"[OK] Wrote spec template: {out_path}")


if __name__ == "__main__":
    main()
