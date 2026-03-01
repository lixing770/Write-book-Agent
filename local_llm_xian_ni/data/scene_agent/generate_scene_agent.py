#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scene Rendering Agent (DeepSeek) - STABLE FULL VERSION
-----------------------------------------------------
目标：
- 输入：split_dir 下的小说原文（按章 txt）
- 输出：每50章一份《场景渲染学习笔记》scene_notes_xxxx_xxxx.md
- 最终：汇总成 scene_master_notes.md

稳定性增强：
1) 读取 finish_reason；若 length 截断 -> 自动续写直到完整（chat_until_complete）
2) 先骨架，再逐小节填充，再总润色
3) 每小节做“最小长度校验”，太短/缺失则自动补写（repair loop）
4) 断点续跑：已存在则跳过（--overwrite 强制重跑）
5) 输入自动分块 chunk_chars，避免输入挤压输出

依赖：
  pip install requests
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import requests


# ----------------------------
# DeepSeek API Client
# ----------------------------

@dataclass
class DeepSeekConfig:
    api_key: str
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-reasoner"
    max_tokens: int = 8000
    temperature: float = 0.3
    timeout_sec: int = 180


class DeepSeekAPI:
    def __init__(self, cfg: DeepSeekConfig):
        self.cfg = cfg

    def chat(self, system: str, user: str) -> tuple[str, str]:
        """
        Returns: (content, finish_reason)
        finish_reason commonly: "stop" or "length"
        """
        url = f"{self.cfg.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
        }
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=self.cfg.timeout_sec)
        resp.raise_for_status()
        data = resp.json()

        choice0 = data["choices"][0]
        content = choice0["message"]["content"]
        finish_reason = choice0.get("finish_reason", "")
        return content, finish_reason


def sleep_backoff(attempt: int):
    time.sleep(min(2.0 * attempt, 12.0))


def chat_until_complete(api: DeepSeekAPI, system: str, user: str, max_rounds: int = 8) -> str:
    """
    Auto-continue if truncated (finish_reason == 'length').
    """
    full = ""
    cur_user = user

    for r in range(1, max_rounds + 1):
        content, finish = api.chat(system, cur_user)
        full += (content.rstrip() + "\n")

        if finish != "length":
            break

        tail = full[-1400:]  # 给一点尾巴定位续写点
        cur_user = (
            "你刚才的输出被截断了（finish_reason=length）。请从中断处继续，"
            "保持同样的标题结构/编号/格式，不要重复已写内容。\n\n"
            f"以下是你刚才输出的结尾片段（仅供续写定位）：\n<<<TAIL\n{tail}\nTAIL>>>"
        )

    return full.strip() + "\n"


# ----------------------------
# Prompts
# ----------------------------

HEADINGS = [
    "A) 场景类型库（可复用分类）",
    "B) 视觉要素清单（光/色/材质/空间/气候/动线/视角）",
    "C) 镜头语言与节奏（远景/中景/特写/切换策略）",
    "D) 氛围与情绪渲染（声场/触感/嗅觉/冷暖/压迫感）",
    "E) 高频可复用句式模板（可直接粘贴写作）",
    "F) 场景如何驱动剧情（冲突/升级/转折/信息揭示）",
    "G) 禁忌清单（避免流水账与重复套路）",
    "H) 标志性场景卡（3-8张：一句话定位 + 关键元素 + 适用场景）",
]

SYSTEM_PROMPT = """你是“小说场景渲染主编（Scene Rendering Chief Editor）”。
你的任务：从用户提供的小说原文中，提炼“可复用的场景渲染方法论”和“可直接复制的描写模板”，用于指导后续创作。

严格要求：
1) 必须按用户给定的标题结构输出，任何标题都不能缺失；若原文信息不足，也必须补齐并提供“通用写法”。
2) 不要复述剧情，不要长篇总结故事；重点是“渲染技法、构图要素、镜头语言、氛围塑造、句式模板、可复用规则”。
3) 输出必须完整，不能在某个小节中途结束；如内容过长，优先压缩冗余而不是缺标题。
4) 语言：中文；风格：专业但像写作教练；条理清晰，多用要点列表与模板块。"""

MASTER_SYSTEM = """你是“场景渲染知识库主编（Master Scene Rendering Editor）”。
你的任务：把多个批次的《场景渲染学习笔记》合并为一份“总知识库”，要求结构更抽象、更可复用。

严格要求：
1) 输出必须完整，不能中途截断；标题不能缺失。
2) 去重、合并同义项、把零散技巧抽象成规则、清单、模板。
3) 不要复述小说剧情，只保留写作方法论与模板。
语言：中文；风格：专业、可执行、像一本写作手册。"""

MASTER_HEADINGS = [
    "1) 场景渲染总公式（最小闭环）",
    "2) 场景类型与“元素包”库（可直接套用）",
    "3) 光影与色彩：常用组合与适用情绪",
    "4) 空间与动线：读者视线如何被你控制",
    "5) 材质细节：让画面“可触摸”的写法",
    "6) 声场/触感/嗅觉：多感官渲染模板",
    "7) 镜头语言：段落节奏与信息密度",
    "8) 情绪与冲突：场景如何推动剧情升级",
    "9) 句式与模板库（按场景/情绪分类）",
    "10) 禁忌清单 + 自检清单（写完就对照）",
]


def build_skeleton(title: str, headings: List[str]) -> str:
    parts = [f"# {title}", ""]
    for h in headings:
        parts.append(f"## {h}\n- （待生成）\n")
    return "\n".join(parts)


def build_fill_section_prompt(batch_range: str, section: str, novel_text: str, draft_doc: str) -> str:
    return f"""下面是【{batch_range}】的小说原文（可能较长）：
<<<NOVEL_TEXT
{novel_text}
NOVEL_TEXT>>>

下面是当前笔记草稿（可能某些小节还是“待生成”）：
<<<DRAFT
{draft_doc}
DRAFT>>>

任务：只填充/完善指定的小节：{section}
要求：
- 必须围绕“场景渲染技法”，不要复述剧情。
- 尽量给可复用模板/句式/清单。
- 输出时只返回该小节的完整内容（必须以“## {section}”开头），不要输出其他小节。
- 如果你觉得本小节内容仍然不够充实，请优先补“可复用模板/规则/检查清单”，而不是增加剧情复述。
"""


def build_polish_prompt(batch_range: str, draft_doc: str) -> str:
    return f"""下面是【{batch_range}】的《场景渲染学习笔记》草稿：
<<<DRAFT
{draft_doc}
DRAFT>>>

任务：做最终润色与一致性校对：
- 保证每个小节内容都不为空，不含“待生成”
- 列表结构清晰；句式模板要像“可直接粘贴”的写作素材
- 总体不要过度冗长（但不能删标题）
- 输出完整Markdown全文
"""


def build_repair_section_prompt(batch_range: str, section: str, novel_text: str, current_section: str) -> str:
    return f"""你正在修复【{batch_range}】笔记的小节：{section}。

原文片段（可能较长）：
<<<NOVEL_TEXT
{novel_text}
NOVEL_TEXT>>>

当前该小节内容（可能过短/空泛）：
<<<SECTION
{current_section}
SECTION>>>

要求：在不复述剧情的前提下，把该小节扩写到“足够可复用”的程度：
- 至少包含：清单/规则/模板（其中模板不少于5条；如果是H场景卡，至少3张卡）
- 尽量避免抽象空话，用“可直接套用”的写作指令表达
- 只输出该小节完整内容（必须以“## {section}”开头）
"""


def build_master_merge_prompt(current_master: str, batch_text: str) -> str:
    return f"""下面是当前的“总知识库草稿”：
<<<MASTER
{current_master}
MASTER>>>

下面新增一个批次学习笔记，请把它的有效内容融入总知识库（去重、抽象、合并），并输出完整总知识库Markdown：
<<<BATCH
{batch_text}
BATCH>>>
"""


def build_master_polish_prompt(master_doc: str) -> str:
    return f"""请对下面的总知识库做最终润色：
- 所有小节不能为空、不含“待生成”
- 去重、合并同义项，结构更像“写作手册”
- 增加“可复用模板/自检清单”的密度
- 输出完整Markdown

<<<MASTER
{master_doc}
MASTER>>>
"""


# ----------------------------
# Document helpers
# ----------------------------

def find_chapter_files(split_dir: Path) -> List[Path]:
    files = list(split_dir.glob("*.txt"))

    def chap_num(p: Path) -> int:
        m = re.search(r"(\d+)", p.stem)
        return int(m.group(1)) if m else 10**9

    return sorted(files, key=chap_num)


def read_chapters(split_dir: Path, start: int, end: int) -> Tuple[str, List[Path]]:
    files = find_chapter_files(split_dir)
    picked: List[Path] = []
    for p in files:
        m = re.search(r"(\d+)", p.stem)
        if not m:
            continue
        n = int(m.group(1))
        if start <= n <= end:
            picked.append(p)

    if not picked:
        raise FileNotFoundError(f"No chapter files found in range [{start}, {end}] under {split_dir}")

    texts = []
    for p in picked:
        t = p.read_text(encoding="utf-8", errors="ignore").strip()
        if t:
            texts.append(f"\n\n===== {p.name} =====\n{t}")
    return "\n".join(texts).strip(), picked


def chunk_text(text: str, chunk_chars: int) -> List[str]:
    if len(text) <= chunk_chars:
        return [text]
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i + chunk_chars])
        i += chunk_chars
    return chunks


def has_all_headings(doc: str, headings: List[str]) -> bool:
    return all(f"## {h}" in doc for h in headings)


def has_placeholders(doc: str) -> bool:
    return "待生成" in doc or "（待生成）" in doc


def extract_section(doc: str, section_title: str) -> Optional[str]:
    # return markdown from "## section" until next "## "
    pattern = re.compile(rf"^## {re.escape(section_title)}\s*$", re.M)
    m = pattern.search(doc)
    if not m:
        return None
    start = m.start()
    m2 = re.search(r"^##\s+", doc[m.end():], flags=re.M)
    end = (m.end() + m2.start()) if m2 else len(doc)
    return doc[start:end].strip() + "\n"


def replace_section(doc: str, section_title: str, new_section_md: str) -> str:
    old = extract_section(doc, section_title)
    if old is None:
        return doc.rstrip() + "\n\n" + new_section_md.strip() + "\n"
    return doc.replace(old, new_section_md.strip() + "\n")


def section_effective_length(section_md: str) -> int:
    # remove heading line, count remaining chars
    lines = section_md.strip().splitlines()
    if not lines:
        return 0
    if lines[0].startswith("## "):
        lines = lines[1:]
    body = "\n".join(lines).strip()
    # treat placeholder as zero
    if "待生成" in body:
        return 0
    return len(body)


# ----------------------------
# Batch generation logic
# ----------------------------

def call_with_retries(fn, max_retries: int):
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except Exception:
            if attempt == max_retries:
                raise
            sleep_backoff(attempt)


def ensure_heading_wrapped(section: str, resp: str) -> str:
    txt = resp.strip()
    if f"## {section}" not in txt:
        txt = f"## {section}\n" + txt
    # 强制保证以该标题开头
    if not txt.startswith(f"## {section}"):
        # 找到第一次出现的位置后截断前面
        idx = txt.find(f"## {section}")
        if idx >= 0:
            txt = txt[idx:]
        else:
            txt = f"## {section}\n" + txt
    return txt.strip() + "\n"


def generate_one_batch(
    api: DeepSeekAPI,
    batch_range: str,
    novel_text: str,
    out_path: Path,
    chunk_chars: int,
    max_retries: int,
    section_min_chars: int,
    repair_max_rounds: int,
) -> None:
    doc = build_skeleton("场景渲染学习笔记（按50章批次）", HEADINGS)
    chunks = chunk_text(novel_text, chunk_chars)

    # 逐小节生成（跨chunk累加）
    for section in HEADINGS:
        section_md = f"## {section}\n- （待生成）\n"

        for ck in chunks:
            draft_with_section = replace_section(doc, section, section_md)
            user_prompt = build_fill_section_prompt(batch_range, section, ck, draft_with_section)

            resp = call_with_retries(
                lambda: chat_until_complete(api, SYSTEM_PROMPT, user_prompt, max_rounds=8),
                max_retries=max_retries,
            )
            section_md = ensure_heading_wrapped(section, resp)

        doc = replace_section(doc, section, section_md)

        # 小节最小长度校验，不够就修复补写（用全量原文或截取片段）
        repaired = 0
        while repaired < repair_max_rounds:
            cur = extract_section(doc, section) or f"## {section}\n- （待生成）\n"
            if section_effective_length(cur) >= section_min_chars:
                break

            # 修复时不要用全量原文太大：取前中后片段拼接
            nt = novel_text
            if len(nt) > 36000:
                head = nt[:12000]
                mid = nt[len(nt)//2 - 6000: len(nt)//2 + 6000]
                tail = nt[-12000:]
                nt = head + "\n\n...（中间略）...\n\n" + mid + "\n\n...（后略）...\n\n" + tail

            repair_prompt = build_repair_section_prompt(batch_range, section, nt, cur)
            resp2 = call_with_retries(
                lambda: chat_until_complete(api, SYSTEM_PROMPT, repair_prompt, max_rounds=8),
                max_retries=max_retries,
            )
            section_md2 = ensure_heading_wrapped(section, resp2)
            doc = replace_section(doc, section, section_md2)
            repaired += 1

    # 总润色（直到无 placeholder 且标题齐全）
    for _ in range(3):
        if has_all_headings(doc, HEADINGS) and not has_placeholders(doc):
            break
        polish_prompt = build_polish_prompt(batch_range, doc)
        doc = call_with_retries(
            lambda: chat_until_complete(api, SYSTEM_PROMPT, polish_prompt, max_rounds=10),
            max_retries=max_retries,
        )

    # 终极兜底：强制补齐缺失标题
    if not has_all_headings(doc, HEADINGS):
        enforced = build_skeleton("场景渲染学习笔记（按50章批次）", HEADINGS)
        for section in HEADINGS:
            sec = extract_section(doc, section)
            if sec:
                enforced = replace_section(enforced, section, sec)
            else:
                enforced = replace_section(enforced, section, f"## {section}\n- 通用写法：用“空间层级 + 光影 + 材质细节 + 动线 + 情绪”补齐。\n")
        doc = enforced

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc.strip() + "\n", encoding="utf-8")


# ----------------------------
# Merge + Summarize master
# ----------------------------

def merge_batches_and_summarize(api: DeepSeekAPI, out_dir: Path, master_path: Path, max_retries: int) -> None:
    files = sorted(out_dir.glob("scene_notes_*.md"))
    if not files:
        raise FileNotFoundError(f"No scene_notes_*.md found under {out_dir}")

    master = build_skeleton("场景渲染总知识库（汇总版）", MASTER_HEADINGS)

    for f in files:
        batch_text = f.read_text(encoding="utf-8", errors="ignore").strip()
        if not batch_text:
            continue

        prompt = build_master_merge_prompt(master, batch_text)
        master = call_with_retries(
            lambda: chat_until_complete(api, MASTER_SYSTEM, prompt, max_rounds=12),
            max_retries=max_retries,
        )

        # 轻量保证结构不丢
        if not has_all_headings(master, MASTER_HEADINGS):
            enforced = build_skeleton("场景渲染总知识库（汇总版）", MASTER_HEADINGS)
            for h in MASTER_HEADINGS:
                sec = extract_section(master, h)
                if sec:
                    enforced = replace_section(enforced, h, sec)
            master = enforced

    # 最终润色直到无 placeholder
    for _ in range(4):
        if has_all_headings(master, MASTER_HEADINGS) and not has_placeholders(master):
            break
        master = call_with_retries(
            lambda: chat_until_complete(api, MASTER_SYSTEM, build_master_polish_prompt(master), max_rounds=14),
            max_retries=max_retries,
        )

    master_path.parent.mkdir(parents=True, exist_ok=True)
    master_path.write_text(master.strip() + "\n", encoding="utf-8")


# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_dir", required=True, help="Directory containing chapter txt files")
    ap.add_argument("--out_dir", required=True, help="Output directory for scene notes batches")
    ap.add_argument("--final_dir", required=True, help="Directory for final merged notes")
    ap.add_argument("--start_chapter", type=int, required=True)
    ap.add_argument("--end_chapter", type=int, required=True)
    ap.add_argument("--batch_chapters", type=int, default=50)

    ap.add_argument("--chunk_chars", type=int, default=12000)

    ap.add_argument("--model", default="deepseek-reasoner")
    ap.add_argument("--api_key", default=os.getenv("DEEPSEEK_API_KEY", ""))
    ap.add_argument("--base_url", default="https://api.deepseek.com")
    ap.add_argument("--max_tokens", type=int, default=8000)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--timeout_sec", type=int, default=180)
    ap.add_argument("--max_retries", type=int, default=5)

    # 稳定性参数（关键）
    ap.add_argument("--section_min_chars", type=int, default=600, help="Minimum chars per section body (excluding heading)")
    ap.add_argument("--repair_max_rounds", type=int, default=2, help="Max repair rounds per section if too short")

    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if not args.api_key:
        print("ERROR: --api_key is empty and DEEPSEEK_API_KEY env not set.", file=sys.stderr)
        sys.exit(1)

    split_dir = Path(args.split_dir)
    out_dir = Path(args.out_dir)
    final_dir = Path(args.final_dir)

    cfg = DeepSeekConfig(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout_sec=args.timeout_sec,
    )
    api = DeepSeekAPI(cfg)

    start = args.start_chapter
    end = args.end_chapter
    step = args.batch_chapters

    for bstart in range(start, end + 1, step):
        bend = min(end, bstart + step - 1)
        batch_range = f"{bstart:04d}-{bend:04d}"
        out_path = out_dir / f"scene_notes_{bstart:04d}_{bend:04d}.md"

        if out_path.exists() and not args.overwrite:
            print(f"[SKIP] {out_path.name} already exists.")
            continue

        print(f"[RUN] Generating scene notes for chapters {batch_range} ...")
        novel_text, picked = read_chapters(split_dir, bstart, bend)
        print(f"      Loaded {len(picked)} chapter files, chars={len(novel_text)}")

        generate_one_batch(
            api=api,
            batch_range=batch_range,
            novel_text=novel_text,
            out_path=out_path,
            chunk_chars=args.chunk_chars,
            max_retries=args.max_retries,
            section_min_chars=args.section_min_chars,
            repair_max_rounds=args.repair_max_rounds,
        )
        print(f"[OK] Wrote: {out_path}")

    # Merge to master
    master_path = final_dir / "scene_master_notes.md"
    merge_batches_and_summarize(api, out_dir, master_path, args.max_retries)
    print(f"[OK] Final master note: {master_path}")


if __name__ == "__main__":
    main()