#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a style profile JSON from a novel jsonl (paragraph-level).

Input jsonl line format:
  {"doc_id": "...", "chapter": 1, "para_id": 12, "text": "...", "meta": {...}}

Output:
  data/knowledge/style_profiles/<doc_id>.json

Design:
- sample representative paragraphs (not too long)
- ask LLM to extract writing rules into strict JSON schema
- validate JSON and write file
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

# If you already have provider factory, use it.
# Otherwise fallback to OpenAIProvider.
try:
    from ..provider import get_ai_provider  # type: ignore
except Exception:
    get_ai_provider = None  # type: ignore

try:
    from ..openai_provider import OpenAIProvider  # type: ignore
except Exception:
    OpenAIProvider = None  # type: ignore


STYLE_SCHEMA_HINT = {
    "source": "string (novel name/doc_id)",
    "narrative": {
        "person": "first/third/multi",
        "tense": "past/present/mixed",
        "inner_monologue": "low/medium/high",
        "pov_notes": "string"
    },
    "pacing": {
        "hook_frequency_words": "e.g. 800-1200",
        "conflict_rhythm": "string",
        "chapter_end_pattern": "string",
        "reveal_strategy": "string"
    },
    "language": {
        "sentence_style": "short/medium/long/mixed",
        "tone": "string",
        "emotion_curve": ["strings"],
        "keywords": ["strings"],
        "taboo_words": ["strings"]
    },
    "plot_patterns": ["strings"],
    "character_patterns": ["strings"],
    "scene_patterns": ["strings"],
    "forbidden": ["strings"],
    "dos": ["strings"],
    "example_snippets": ["<= 6 short snippets, each <= 120 chars, no verbatim long quotes"]
}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # ignore broken line
                continue
    if not rows:
        raise RuntimeError(f"Empty or invalid jsonl: {path}")
    return rows


def _bucket_sample(
    rows: List[Dict[str, Any]],
    max_chapters: int,
    paras_per_chapter: int,
    max_total_chars: int,
    seed: int
) -> Tuple[str, str]:
    """
    Sample paragraphs across chapters to represent style.
    Return (doc_id, sampled_text)
    """
    random.seed(seed)

    doc_id = str(rows[0].get("doc_id") or "unknown")
    by_ch: Dict[int, List[str]] = {}
    for r in rows:
        ch = int(r.get("chapter") or 1)
        txt = str(r.get("text") or "").strip()
        if not txt:
            continue
        by_ch.setdefault(ch, []).append(txt)

    chapters = sorted(by_ch.keys())
    if len(chapters) > max_chapters:
        # take evenly spaced chapters
        idxs = [int(i * (len(chapters) - 1) / (max_chapters - 1)) for i in range(max_chapters)]
        chosen = [chapters[i] for i in sorted(set(idxs))]
    else:
        chosen = chapters

    parts: List[str] = []
    total = 0
    for ch in chosen:
        paras = by_ch[ch]
        if not paras:
            continue
        # sample some paragraphs
        take = paras_per_chapter if len(paras) >= paras_per_chapter else len(paras)
        picked = random.sample(paras, take)
        block = f"\n[CHAPTER {ch}]\n" + "\n".join(f"- {p}" for p in picked)
        if total + len(block) > max_total_chars:
            break
        parts.append(block)
        total += len(block)

    sampled_text = "\n".join(parts).strip()
    if not sampled_text:
        raise RuntimeError("Sampling produced empty text. Check jsonl content.")
    return doc_id, sampled_text


def _extract_json(text: str) -> Dict[str, Any]:
    """
    Robust JSON extraction:
    - try direct json.loads
    - else find first {...} block and parse
    """
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    # try locate a JSON object in text
    l = text.find("{")
    r = text.rfind("}")
    if l != -1 and r != -1 and r > l:
        candidate = text[l:r+1]
        return json.loads(candidate)

    raise RuntimeError("Model output is not valid JSON.")


def _validate_profile(obj: Dict[str, Any], source: str) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        raise RuntimeError("Profile must be a JSON object.")

    obj.setdefault("source", source)

    # very light validation
    for k in ["narrative", "pacing", "language", "plot_patterns", "forbidden", "dos", "example_snippets"]:
        if k not in obj:
            raise RuntimeError(f"Missing required key: {k}")

    if not isinstance(obj["example_snippets"], list):
        raise RuntimeError("example_snippets must be a list")

    # enforce snippet length to reduce copyright risk
    fixed_snips = []
    for s in obj["example_snippets"][:6]:
        s = str(s)
        if len(s) > 120:
            s = s[:120]
        fixed_snips.append(s)
    obj["example_snippets"] = fixed_snips

    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="Input novel jsonl path")
    ap.add_argument("--out", default="", help="Output profile json path. Default: data/knowledge/style_profiles/<doc_id>.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_chapters", type=int, default=14)
    ap.add_argument("--paras_per_chapter", type=int, default=6)
    ap.add_argument("--max_total_chars", type=int, default=22000)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max_tokens", type=int, default=1800)
    args = ap.parse_args()

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        raise SystemExit(f"Not found: {jsonl_path}")

    rows = _read_jsonl(jsonl_path)
    doc_id, sampled = _bucket_sample(
        rows,
        max_chapters=args.max_chapters,
        paras_per_chapter=args.paras_per_chapter,
        max_total_chars=args.max_total_chars,
        seed=args.seed
    )

    # provider
    if get_ai_provider is not None:
        provider = get_ai_provider()
    else:
        if OpenAIProvider is None:
            raise RuntimeError("No provider available. Ensure backend/ai/provider.py or openai_provider.py exists.")
        provider = OpenAIProvider()

    system = (
        "你是一个“小说风格分析器”。"
        "你只做风格与结构抽取，不复述原文，不生成与原文相似的内容。"
        "输出必须是严格 JSON（不要 Markdown、不要代码块）。"
        "example_snippets 必须是你自己总结改写的短句示例（<=120字），不能复制原文长句。"
    )

    user = f"""
请基于以下小说片段样本，抽取其写作风格与结构规则，输出为严格 JSON。
要求：
- 只输出 JSON，不要多余文字
- 字段尽量覆盖 schema hint
- example_snippets 给 4-6 条“你自己改写的示例句”，不能照抄原文
- forbidden 里写清楚哪些写法会破坏该风格
- dos 里写清楚写作时必须做的事（节奏/钩子/情绪）

schema_hint:
{json.dumps(STYLE_SCHEMA_HINT, ensure_ascii=False)}

samples:
{sampled}
""".strip()

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    resp = provider.chat(messages, temperature=args.temperature, max_tokens=args.max_tokens)
    profile = _extract_json(resp)
    profile = _validate_profile(profile, source=doc_id)

    out_path = Path(args.out) if args.out else Path("data/knowledge/style_profiles") / f"{doc_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] style_profile saved: {out_path}")


if __name__ == "__main__":
    main()
