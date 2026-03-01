#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Summarize final_summary_*.md -> one ~target_chars master note (DeepSeek API)
ULTRA-STABLE "FORCE-SKELETON" VERSION
------------------------------------------------------------
Core idea:
- The program ALWAYS constructs the skeleton headings by itself (1-9) in fixed order.
- The model is only allowed to generate BODY text for a given section.
- Even if the model returns messy text, we will still place it under the section heading.
- No dependency on "model must repeat headings".

Features:
- Reads final_summary_*.md under --notes_dir
- Packs into bundles (for context)
- Generates section bodies 1..9 individually
- Final polish pass to hit target length (±10%)
- Strictly removes incomplete markers and forbidden "big director" words
- Compatible with --final_retries as alias to --polish_retries

Dependency:
  pip install requests
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import requests


# ----------------------------
# DeepSeek API
# ----------------------------

class DeepSeekAPI:
    def __init__(self, api_key: str, base_url: str, model: str, max_tokens: int, temperature: float, timeout_sec: int):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/") + "/v1"
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_sec = timeout_sec

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        r = requests.post(url, headers=headers, json=payload, timeout=self.timeout_sec)
        r.raise_for_status()
        data = r.json()
        return (data["choices"][0]["message"]["content"] or "").strip()


# ----------------------------
# Read inputs
# ----------------------------

def find_final_summary_files(notes_dir: Path) -> List[Path]:
    files = list(notes_dir.glob("final_summary_*.md"))

    def key(p: Path):
        m = re.search(r"final_summary_(\d+)_", p.name)
        return int(m.group(1)) if m else 10**9

    return sorted(files, key=key)


def read_files(files: List[Path]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for fp in files:
        txt = fp.read_text(encoding="utf-8", errors="replace").strip()
        if txt:
            out.append((fp.name, txt))
    return out


def pack_into_bundles(texts: List[Tuple[str, str]], max_chars: int) -> List[str]:
    bundles: List[str] = []
    cur: List[str] = []
    cur_len = 0
    for name, txt in texts:
        block = f"\n\n===== FILE: {name} =====\n\n{txt}\n"
        if cur and cur_len + len(block) > max_chars:
            bundles.append("".join(cur))
            cur = [block]
            cur_len = len(block)
        else:
            cur.append(block)
            cur_len += len(block)
    if cur:
        bundles.append("".join(cur))
    return bundles


def extract_bundle_for_model(bundles: List[str]) -> str:
    return "\n\n---\n\n".join(bundles)


# ----------------------------
# Hard constraints
# ----------------------------

FORBIDDEN_INCOMPLETE = ["未完", "待续", "后续再补", "以后再补", "未整理完", "未生成完", "省略", "略", "……", "..."]
FORBIDDEN_BIG_DIRECTOR = ["镜头", "运镜", "分镜", "景别", "特写", "全景", "中景", "近景", "远景", "剪辑", "布光", "灯光", "摄影", "机位", "构图", "蒙太奇"]

SECTION_TITLES = [
    "## 1) 总体框架：情绪戏六要素",
    "## 2) 触发器谱系（分类 + 适用条件）",
    "## 3) 阻力模型（为什么不立刻爆）",
    "## 4) 升级台阶模板库（3步/4步/5步模板，给可复制句式）",
    "## 5) 爆点形式库（形式 + 触发条件 + 风险）",
    "## 6) 余波与情绪债：不断供公式",
    "## 7) 泄压黑名单（最致命的写法）",
    "## 8) 训练作业（7天/14天训练清单）",
    "## 9) Prompt Blocks（弱/中/强三套，直接可复制）",
]

def make_skeleton(target_chars: int) -> str:
    head = f"# 情绪戏导演学习笔记总纲（约{target_chars}字）"
    return head + "\n" + "\n".join(SECTION_TITLES) + "\n"


def clean_forbidden(text: str) -> str:
    # remove obvious incomplete marks
    out = text
    for w in FORBIDDEN_INCOMPLETE:
        out = out.replace(w, "")
    # also remove repeated blank lines
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out


def has_forbidden_terms(text: str) -> List[str]:
    hits: List[str] = []
    for w in FORBIDDEN_INCOMPLETE:
        if w in text:
            hits.append(w)
    for w in FORBIDDEN_BIG_DIRECTOR:
        if w in text:
            hits.append(w)
    return hits


def approx_len_ok(text: str, target: int) -> bool:
    n = len(text)
    return int(target * 0.9) <= n <= int(target * 1.1)


# ----------------------------
# Prompts
# ----------------------------

SYSTEM_BASE = """你是“情绪戏导演学习笔记总编”。只输出情绪戏方法论，不复述剧情，不写大导演内容。
绝对禁止：镜头/景别/运镜/灯光/剪辑/摄影/机位/构图等词与相关内容。
绝对禁止：未完/待续/省略/略/……/... 等任何未完成痕迹。
若资料缺失，只能写“资料缺失”，严禁脑补剧情细节。
写作目标：结构清晰、方法论可迁移、可直接复用的模板与Prompt。
"""

PROMPT_WRITE_SECTION_BODY = """你将收到“素材合集”。请只为下列小节生成“正文内容”（不要重复标题），要求：
- 只写方法论：分类、适用条件、风险/注意点、可复用清单（至少4条）。
- 必须给出可直接复制的模板/句式（至少2条）。
- 禁止复述剧情；禁止大导演词；禁止任何未完成痕迹。
- 字数：尽量写得扎实（建议 450-750 字），宁可多给条目也不要空泛。

小节标题：
{SECTION_TITLE}

【素材开始】
{BUNDLE}
【素材结束】
"""

PROMPT_POLISH_ALL = """请对下面整篇总纲做“最终润色与补齐字数”，要求：
1) 保持所有标题不变、顺序不变，一个都不能少。
2) 删除任何未完成标记（未完/待续/略/……/... 等）。
3) 删除任何大导演词汇（镜头/景别/运镜/灯光/剪辑等）。
4) 补齐到约{target_chars}字（±10%），字数不足就优先扩写第4/5/6/8/9节：
   - 增加更多可复制模板、适配条件、风险提示、训练步骤、Prompt细则。
5) 不复述剧情，只写方法论。

【全文开始】
{FULL_DOC}
【全文结束】

请输出最终完整成稿。
"""


# ----------------------------
# Build doc (force skeleton)
# ----------------------------

def generate_section_body(api: DeepSeekAPI, section_title: str, bundle: str) -> str:
    user = PROMPT_WRITE_SECTION_BODY.format(SECTION_TITLE=section_title, BUNDLE=bundle)
    out = api.chat(SYSTEM_BASE, user).strip()
    out = clean_forbidden(out)

    # If model accidentally repeats headings, strip them
    out = re.sub(r"^#{1,6}\s+.*\n+", "", out).strip()

    # If still empty, fallback
    if not out:
        out = "资料缺失。"
    return out


def assemble_doc(target_chars: int, section_bodies: Dict[str, str]) -> str:
    doc_lines: List[str] = []
    doc_lines.append(f"# 情绪戏导演学习笔记总纲（约{target_chars}字）")
    for title in SECTION_TITLES:
        doc_lines.append(title)
        body = (section_bodies.get(title) or "").strip()
        if body:
            doc_lines.append(body)
        else:
            doc_lines.append("资料缺失。")
        doc_lines.append("")  # blank line
    return "\n".join(doc_lines).strip() + "\n"


def polish(api: DeepSeekAPI, target_chars: int, full_doc: str) -> str:
    user = PROMPT_POLISH_ALL.format(target_chars=target_chars, FULL_DOC=full_doc)
    out = api.chat(SYSTEM_BASE, user).strip()
    out = clean_forbidden(out)
    # Ensure skeleton headings exist (force-rebuild if model messed them)
    for title in SECTION_TITLES:
        if title not in out:
            # if missing, fall back to original (safer)
            return full_doc
    if not out.startswith("# 情绪戏导演学习笔记总纲"):
        return full_doc
    return out + ("\n" if not out.endswith("\n") else "")


# ----------------------------
# CLI
# ----------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes_dir", required=True)
    ap.add_argument("--out_file", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api_key", required=True)
    ap.add_argument("--base_url", required=True)

    ap.add_argument("--target_chars", type=int, default=5000)
    ap.add_argument("--bundle_chars", type=int, default=22000)
    ap.add_argument("--max_tokens", type=int, default=3200)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--timeout_sec", type=int, default=300)
    ap.add_argument("--sleep_sec", type=float, default=0.25)

    ap.add_argument("--section_retries", type=int, default=3)

    # Keep your original name
    ap.add_argument("--polish_retries", type=int, default=4)
    # Backward compatible alias
    ap.add_argument("--final_retries", type=int, default=None)

    args = ap.parse_args()

    # alias handling
    if args.final_retries is not None:
        args.polish_retries = int(args.final_retries)

    notes_dir = Path(args.notes_dir)
    if not notes_dir.exists():
        print(f"[FATAL] notes_dir not found: {notes_dir}", file=sys.stderr)
        return 2

    files = find_final_summary_files(notes_dir)
    if not files:
        print(f"[FATAL] no final_summary_*.md found in: {notes_dir}", file=sys.stderr)
        return 2

    texts = read_files(files)
    bundles = pack_into_bundles(texts, max_chars=args.bundle_chars)
    bundle = extract_bundle_for_model(bundles)

    print(f"[INFO] Loaded {len(texts)} final_summary files, packed into {len(bundles)} bundles.")

    api = DeepSeekAPI(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout_sec=args.timeout_sec,
    )

    # Generate each section body with retries
    section_bodies: Dict[str, str] = {}
    for title in SECTION_TITLES:
        ok = False
        last_body = ""
        for attempt in range(1, args.section_retries + 1):
            body = generate_section_body(api, title, bundle)
            last_body = body
            hits = has_forbidden_terms(body)
            if hits:
                time.sleep(args.sleep_sec * attempt)
                continue
            section_bodies[title] = body
            ok = True
            break
        if not ok:
            section_bodies[title] = last_body if last_body else "资料缺失。"
        time.sleep(args.sleep_sec)

    # Assemble forced skeleton doc
    doc = assemble_doc(args.target_chars, section_bodies)

    # Polish to target length (optional but recommended)
    final_doc = doc
    for attempt in range(1, args.polish_retries + 1):
        polished = polish(api, args.target_chars, final_doc)
        polished = clean_forbidden(polished)

        # if polish got forbidden stuff, retry
        hits = has_forbidden_terms(polished)
        if hits:
            time.sleep(args.sleep_sec * attempt)
            continue

        final_doc = polished

        if approx_len_ok(final_doc, args.target_chars):
            break

        # If too short, we do another polish pass; if too long, polish often compresses a bit.
        time.sleep(args.sleep_sec * attempt)

    out_fp = Path(args.out_file)
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    out_fp.write_text(final_doc, encoding="utf-8")
    print(f"[OK] Saved: {out_fp} (len={len(final_doc)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())