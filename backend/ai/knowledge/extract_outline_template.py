#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract outline template YAML from jsonl corpus, with aggressive YAML repair.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from ..provider import get_ai_provider


SYSTEM_PROMPT = """你是一名“网文连载章节结构”分析师。
只允许输出“结构模板与套路”，严禁复述/改写任何原著句子，严禁输出原著具体剧情链条、人物名、地名、宗门名。
输出必须是【严格 YAML】，能被 yaml.safe_load 解析。
不要输出 Markdown 代码块。
"""

USER_PROMPT = """以下是语料片段集合（仅用于统计章节结构的共性，不允许引用原文）。

【语料片段】
{CORPUS}

请抽取“章节大纲模板（outline template）”，用于生成同类型新书的大纲。
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

注意：YAML 语法必须正确（冒号、缩进、列表都要合规）。
"""

REPAIR_SYSTEM = """你是 YAML 修复器。你只做一件事：把输入修复成【严格合法 YAML】。
要求：
- 顶层必须是 mapping
- 不要输出 Markdown 代码块
- 只输出 YAML
"""

REPAIR_USER = """下面这段文本 intended as YAML，但无法解析。请修复成严格合法 YAML（尽量保持语义不变）。

【BROKEN】
{BROKEN}
"""


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
            elif isinstance(obj, str):
                texts.append(obj)
    return texts


def _excerpt(texts: List[str], max_chars: int, seed: int) -> str:
    rng = random.Random(seed)
    idxs = list(range(len(texts)))
    rng.shuffle(idxs)
    buf: List[str] = []
    total = 0
    for i in idxs:
        t = texts[i].strip()
        if len(t) < 120:
            continue
        if len(t) > 3500:
            t = t[:3500]
        # redact obvious secrets patterns
        t = re.sub(r"sk-[A-Za-z0-9]{10,}", "[REDACTED_KEY]", t)
        if total + len(t) + 2 > max_chars:
            break
        buf.append(t)
        total += len(t) + 2
    return "\n\n".join(buf) if buf else "\n\n".join(texts[:50])


def _strip_fences(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```(?:yaml)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _fix_dash_space(s: str) -> str:
    # "-信息" -> "- 信息"
    return re.sub(r"(?m)^(\s*)-(\S)", r"\1- \2", s)


def _fix_fullwidth_colon_for_keys(s: str) -> str:
    """
    Convert 'key： value' -> 'key: value' when it looks like a mapping entry.
    Don't blindly replace all '：' because it may appear in prose.
    """
    return re.sub(r"(?m)^(\s*[\w\u4e00-\u9fff][\w\u4e00-\u9fff _-]*)：\s*", r"\1: ", s)


def _normalize_length_keys(s: str) -> str:
    """
    typical_length_percent: 10-15  -> typical_length_ratio: "10-15%"
    """
    # percent numeric range
    def repl(m: re.Match) -> str:
        rng = m.group(1).strip()
        rng = rng.replace("%", "")
        if "-" in rng:
            return f'typical_length_ratio: "{rng}%"'
        return f'typical_length_ratio: "{rng}%"'

    s = re.sub(r"(?m)^\s*typical_length_percent:\s*([0-9]+(?:\s*-\s*[0-9]+)?%?)\s*$", lambda m: re.sub(r"\s+", "", repl(m)), s)
    return s


def _force_pacing_profile_mapping(s: str) -> str:
    """
    If pacing_profile is a list, convert into mapping.
    Example:
      pacing_profile:
        - overall_structure: ...
        - conflict_density: ...
        -节奏快慢结合：...
    becomes:
      pacing_profile:
        overall_rhythm: ...
        conflict_density: ...
        hook_frequency: ...
        reveal_rhythm: ...
        tempo_adjustment: ...
    We keep any extra keys too.
    """
    lines = s.splitlines()
    out: List[str] = []
    i = 0

    def is_top_key(line: str) -> bool:
        return bool(re.match(r"^[A-Za-z0-9_\u4e00-\u9fff].*:\s*$", line))

    while i < len(lines):
        line = lines[i]
        out.append(line)
        if line.strip() == "pacing_profile:":
            # collect indented block
            i += 1
            block: List[str] = []
            while i < len(lines) and (lines[i].startswith("  ") or lines[i].strip() == ""):
                block.append(lines[i])
                i += 1

            # If any list item exists under pacing_profile, convert
            has_list = any(re.match(r"^\s*-\s*", b) for b in block)
            if has_list:
                items: List[Tuple[str, str]] = []
                for b in block:
                    b0 = b.rstrip()
                    if not b0.strip():
                        continue
                    b0 = _fix_dash_space(b0)
                    b0 = _fix_fullwidth_colon_for_keys(b0)
                    # remove leading "-" if present
                    b0 = re.sub(r"^\s*-\s*", "  ", b0)
                    # now expect "  key: value"
                    m = re.match(r"^\s{2}([^:]+):\s*(.*)$", b0)
                    if m:
                        k = m.group(1).strip()
                        v = m.group(2).strip()
                        items.append((k, v))
                    else:
                        # If it's still a plain sentence, put it under tempo_adjustment as list later
                        items.append(("tempo_adjustment", b0.strip()))

                # Build mapping with canonical keys
                mp: Dict[str, Any] = {}
                extras: List[str] = []
                for k, v in items:
                    if k in mp:
                        # duplicates -> extras
                        extras.append(f"{k}: {v}")
                    else:
                        mp[k] = v

                # Try to rename keys into required schema
                rename_map = {
                    "overall_structure": "overall_rhythm",
                    "overall_rhythm": "overall_rhythm",
                    "conflict_density": "conflict_density",
                    "hook_frequency": "hook_frequency",
                    "信息揭露节奏": "reveal_rhythm",
                    "节奏调整": "tempo_adjustment",
                    "节奏快慢结合": "tempo_adjustment",
                    "reveal_rhythm": "reveal_rhythm",
                    "tempo_adjustment": "tempo_adjustment",
                }
                normalized: Dict[str, Any] = {}
                for k, v in mp.items():
                    nk = rename_map.get(k, k)
                    normalized[nk] = v

                # ensure required keys exist (fallback strings)
                normalized.setdefault("overall_rhythm", "起伏分明：先铺垫→冲突加速→信息揭露→阶段收束→章末钩子")
                normalized.setdefault("conflict_density", "中高（每章至少1-2段冲突/危机）")
                normalized.setdefault("hook_frequency", "章末必有钩子，部分章节中段可有小钩子")
                normalized.setdefault("reveal_rhythm", "递进式揭露：碎片线索→局部真相→反转/伏笔回收")
                normalized.setdefault("tempo_adjustment", "冲突段落紧凑，揭露段落稍缓，收束段落平稳并引出悬念")

                # write mapping block (2 spaces indent)
                out.pop()  # remove pacing_profile: line already added
                out.append("pacing_profile:")
                for k in ("overall_rhythm", "conflict_density", "hook_frequency", "reveal_rhythm", "tempo_adjustment"):
                    out.append(f"  {k}: {normalized[k]}")
                # write remaining unknown keys
                for k, v in normalized.items():
                    if k in ("overall_rhythm", "conflict_density", "hook_frequency", "reveal_rhythm", "tempo_adjustment"):
                        continue
                    out.append(f"  {k}: {v}")
                # also keep extras if any
                if extras:
                    out.append("  notes:")
                    for e in extras:
                        out.append(f"    - {e}")
            else:
                # keep block as-is
                out.extend(block)
            continue

        i += 1

    return "\n".join(out)


def _postprocess_yaml_text(s: str) -> str:
    s = _strip_fences(s)
    s = s.replace("\t", "  ")
    s = _fix_dash_space(s)
    s = _fix_fullwidth_colon_for_keys(s)
    s = _normalize_length_keys(s)
    s = _force_pacing_profile_mapping(s)
    # ensure typical_length_ratio exists even if model used percent key
    s = s.replace("typical_length_percent:", "typical_length_ratio:")
    return s.strip() + "\n"


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
    ap.add_argument("--max_tokens", type=int, default=2600)
    ap.add_argument("--repair_max_tokens", type=int, default=1800)
    args = ap.parse_args()

    jsonl_path = Path(args.jsonl)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    texts = _read_jsonl_texts(jsonl_path)
    if not texts:
        raise RuntimeError("No texts read from jsonl.")

    corpus = _excerpt(texts, args.max_chars, args.seed)

    provider = get_ai_provider()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT.replace("{CORPUS}", corpus)},
    ]
    resp = provider.chat(messages, temperature=args.temperature, max_tokens=args.max_tokens)

    fixed = _postprocess_yaml_text(resp)

    # 1) parse after aggressive postprocess
    try:
        data = _parse_yaml_or_raise(fixed)
    except Exception:
        # 2) model repair
        repair_messages = [
            {"role": "system", "content": REPAIR_SYSTEM},
            {"role": "user", "content": REPAIR_USER.replace("{BROKEN}", fixed[:12000])},
        ]
        repaired = provider.chat(repair_messages, temperature=0.0, max_tokens=args.repair_max_tokens)
        repaired = _postprocess_yaml_text(repaired)
        try:
            data = _parse_yaml_or_raise(repaired)
            fixed = repaired
        except Exception as e2:
            raise RuntimeError(
                f"Invalid YAML even after repair: {e2}\n\nFIXED(head):\n{fixed[:2000]}"
            ) from e2

    # dump canonical yaml to ensure validity
    out_path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"[OK] Wrote outline template: {out_path}")


if __name__ == "__main__":
    main()
