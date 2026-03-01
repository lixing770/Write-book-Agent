#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import glob
import time
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

from openai import OpenAI

# =========================
# CLI
# =========================
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--split_dir", required=True, help="dir containing split files like 1_10.txt")
    p.add_argument("--out_dir", required=True, help="output root dir")
    p.add_argument("--model", required=True, help="model name, e.g. deepseek-reasoner")

    # range control (CLI)
    p.add_argument("--start_chapter", type=int, default=None, help="start chapter (inclusive)")
    p.add_argument("--end_chapter", type=int, default=None, help="end chapter (inclusive)")

    # batching
    p.add_argument("--batch_chapters", type=int, default=50, help="chapters per note (default 50)")

    # LLM config
    p.add_argument("--api_key", default=None, help="API key; if omitted, use OPENAI_API_KEY")
    p.add_argument("--base_url", default=None, help="Base URL; if omitted, use OPENAI_BASE_URL if set")
    p.add_argument("--max_tokens", type=int, default=3200)
    p.add_argument("--chunk_chars", type=int, default=22000, help="only chunk when too long")
    p.add_argument("--sleep", type=float, default=0.2)
    p.add_argument("--force", action="store_true", help="overwrite existing output")

    # index output
    p.add_argument("--write_index", action="store_true", help="write batches_index.csv into out_dir")

    return p.parse_args()


# =========================
# Parse split filenames
# =========================
_RANGE_RE = re.compile(r"^(\d+)_(\d+)$")

def parse_range_from_filename(fp: str) -> Tuple[int, int]:
    stem = Path(fp).stem  # "11_20"
    m = _RANGE_RE.match(stem)
    if not m:
        raise ValueError(f"Unexpected split filename format: {fp} (expect like 11_20.txt)")
    a, b = int(m.group(1)), int(m.group(2))
    if b < a:
        raise ValueError(f"Invalid range in filename: {fp}")
    return a, b

def get_sorted_files(split_dir: str) -> List[str]:
    files = glob.glob(os.path.join(split_dir, "*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found in: {split_dir}")
    files.sort(key=lambda x: parse_range_from_filename(x)[0])
    return files


# =========================
# Filter files by chapter range
# =========================
def filter_files_by_range(files: List[str], start_ch: Optional[int], end_ch: Optional[int]) -> List[str]:
    """
    Keep files that overlap with [start_ch, end_ch].
    If start_ch/end_ch is None, treat as unbounded.
    """
    out = []
    for fp in files:
        a, b = parse_range_from_filename(fp)
        if start_ch is not None and b < start_ch:
            continue
        if end_ch is not None and a > end_ch:
            continue
        out.append(fp)
    return out


# =========================
# Group by chapter count
# =========================
def group_files_by_chapters(files: List[str], batch_chapters: int,
                            start_ch: Optional[int], end_ch: Optional[int]) -> List[List[str]]:
    """
    Aggregate consecutive files into batches close to batch_chapters,
    but ALSO name batches by real [start,end] chapters.
    """
    if not files:
        return []

    batches = []
    buf = []
    count = 0

    for fp in files:
        a, b = parse_range_from_filename(fp)

        # clamp the actual chapter contribution if user provided start/end
        eff_a = max(a, start_ch) if start_ch is not None else a
        eff_b = min(b, end_ch) if end_ch is not None else b
        if eff_b < eff_a:
            continue

        chapters = eff_b - eff_a + 1

        # if new file would exceed and we already have something, close current
        if buf and count + chapters > batch_chapters:
            batches.append(buf)
            buf = []
            count = 0

        buf.append(fp)
        count += chapters

        if count == batch_chapters:
            batches.append(buf)
            buf = []
            count = 0

    if buf:
        batches.append(buf)

    return batches


def batch_start_end(batch: List[str], start_ch: Optional[int], end_ch: Optional[int]) -> Tuple[int, int]:
    a1, _ = parse_range_from_filename(batch[0])
    _, b2 = parse_range_from_filename(batch[-1])
    s = max(a1, start_ch) if start_ch is not None else a1
    e = min(b2, end_ch) if end_ch is not None else b2
    return s, e


# =========================
# Read + chunk
# =========================
def read_files_concat(files: List[str]) -> str:
    parts = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            parts.append(f.read())
    return "\n\n".join(parts)

def chunk_text(text: str, chunk_chars: int) -> List[str]:
    if len(text) <= chunk_chars:
        return [text]
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_chars, n)
        boundary = text.rfind("\n\n", start, end)
        if boundary == -1 or boundary <= start + int(chunk_chars * 0.6):
            boundary = end
        ck = text[start:boundary].strip()
        if ck:
            chunks.append(ck)
        start = boundary
    return chunks


# =========================
# Prompts
# =========================
SYSTEM_PROMPT = """
你是一名顶级网文主编（Chief Editor），擅长从成熟长篇网文中抽取“可复用的通用写作规律”，并将其整理成主编知识库。

硬性规则（必须严格遵守）：
- 禁止复述剧情、禁止讲“发生了什么”
- 禁止出现具体人物名、具体地名、具体门派/势力名
- 禁止按章节流水总结
- 只输出“通用可迁移”的写作结构/节奏/冲突/爽点/伏笔/世界观投放/人物关系的规则与模板
- 输出必须完整，不能突然中断；如果内容较多，优先保证结构完整，再保证细节丰富
- 输出为 Markdown，使用标签式条目，例如 <张弛交替>：... / <代价破门>：...
- 语言：中文；风格：简洁、有主编味、可直接被 AI 复用
""".strip()

USER_TEMPLATE = """
下面是小说正文片段（约 {START}-{END} 章范围）。请基于文本抽取【通用主编学习笔记】。

必须输出以下结构（每节必须有足够条目，且不能中途停止）：

# 主编知识库总纲｜通用网文写作（样本章范围：{START}-{END}）

## 1) 主线推进与阶段节点（模板）
- 至少 10 条

## 2) 节奏结构与卡点（规则库）
- 至少 10 条

## 3) 章末钩子库（可直接抄）
- 至少 12 条（短、狠、可复用）

## 4) 冲突模板与推进套路
- 至少 10 条

## 5) 升级/爽点/回报机制（公式）
- 至少 10 条

## 6) 世界观/规则投放手法（边写边讲）
- 至少 10 条

## 7) 人物功能与关系推进（分工）
- 至少 9 条

## 8) 伏笔：埋点→回收（规则）
- 至少 9 条

## 9) 风险避坑 + 下一轮编辑指令
- 风险避坑至少 10 条
- 下一轮编辑指令至少 6 条（偏流程化、可执行）

再次强调：
- 不许出现具体人名/地名/势力名
- 不许复述剧情
- 输出必须完整，不要突然中断
- 结尾必须输出【END】

【正文片段开始】
{TEXT}
【正文片段结束】
""".strip()

MERGE_SYSTEM = """
你是一名顶级网文主编，负责把多份“通用主编学习笔记片段”融合成一份最终版。

规则：
- 去重合并：相似条目合并强化，不要堆重复
- 保留结构：1-9 模块必须齐全
- 强化可执行：尽量给出触发条件/检查点/编辑动作
- 完整输出，结尾必须【END】
""".strip()

MERGE_USER = """
下面是同一批次（约 {START}-{END} 章范围）切块抽取的多个片段笔记。
请融合为单一最终版《主编知识库总纲｜通用网文写作》。

【片段开始】
{PARTS}
【片段结束】
""".strip()

def build_messages(text: str, start_ch: int, end_ch: int) -> List[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(TEXT=text, START=start_ch, END=end_ch)},
    ]

def build_merge_messages(parts: str, start_ch: int, end_ch: int) -> List[dict]:
    return [
        {"role": "system", "content": MERGE_SYSTEM},
        {"role": "user", "content": MERGE_USER.format(PARTS=parts, START=start_ch, END=end_ch)},
    ]


# =========================
# LLM client
# =========================
def make_client(api_key: Optional[str], base_url: Optional[str]) -> OpenAI:
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Missing API key: provide --api_key or set OPENAI_API_KEY")
    url = base_url or os.getenv("OPENAI_BASE_URL")
    if url:
        return OpenAI(api_key=key, base_url=url)
    return OpenAI(api_key=key)

def call_llm(client: OpenAI, model: str, messages: List[dict], max_tokens: int, retries: int = 3) -> str:
    last = None
    for i in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.6,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content
        except Exception as e:
            last = e
            time.sleep(1.5 * i)
    raise RuntimeError(f"LLM call failed after {retries} retries: {last}")


# =========================
# Index writer
# =========================
def write_batches_index(batches: List[List[str]], out_dir: str,
                        start_ch: Optional[int], end_ch: Optional[int]) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "batches_index.csv")
    with open(path, "w", encoding="utf-8") as f:
        f.write("batch_id,start_chapter,end_chapter,file_count,first_file,last_file\n")
        for i, batch in enumerate(batches, start=1):
            s, e = batch_start_end(batch, start_ch, end_ch)
            f.write(f"{i},{s},{e},{len(batch)},{Path(batch[0]).name},{Path(batch[-1]).name}\n")
    return path


# =========================
# Output
# =========================
def save_final(out_dir: str, start_ch: int, end_ch: int, content: str) -> str:
    name = f"{start_ch}_{end_ch}"
    root = os.path.join(out_dir, name)
    os.makedirs(root, exist_ok=True)
    out_path = os.path.join(root, "outline_notes.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
    return out_path


# =========================
# Main
# =========================
def main():
    args = parse_args()
    client = make_client(args.api_key, args.base_url)

    files = get_sorted_files(args.split_dir)
    files = filter_files_by_range(files, args.start_chapter, args.end_chapter)

    if not files:
        raise RuntimeError("No split files left after applying --start_chapter/--end_chapter")

    batches = group_files_by_chapters(files, args.batch_chapters, args.start_chapter, args.end_chapter)

    print(f"✅ split files after range filter: {len(files)}")
    print(f"✅ batches (~{args.batch_chapters} chapters each): {len(batches)}")

    if args.write_index:
        idx_path = write_batches_index(batches, args.out_dir, args.start_chapter, args.end_chapter)
        print(f"🧾 wrote: {idx_path}")

    for bi, batch in enumerate(batches, start=1):
        s, e = batch_start_end(batch, args.start_chapter, args.end_chapter)
        out_path = os.path.join(args.out_dir, f"{s}_{e}", "outline_notes.md")

        if os.path.exists(out_path) and not args.force:
            print(f"⏭️ Skip {bi}/{len(batches)}: chapters {s}-{e} (exists)")
            continue

        raw = read_files_concat(batch)

        # --- Most cost-saving: single call for each 50-ch batch ---
        if len(raw) <= args.chunk_chars:
            print(f"🚀 Batch {bi}/{len(batches)} chapters {s}-{e}: single-call (chars={len(raw)})")
            result = call_llm(client, args.model, build_messages(raw, s, e), args.max_tokens)

        # --- Safety only when too long: chunk + merge ---
        else:
            chunks = chunk_text(raw, args.chunk_chars)
            print(f"🚀 Batch {bi}/{len(batches)} chapters {s}-{e}: chunked={len(chunks)} (chars={len(raw)})")
            parts = []
            for ci, ck in enumerate(chunks, start=1):
                print(f"   🧩 part {ci}/{len(chunks)} ...")
                part = call_llm(client, args.model, build_messages(ck, s, e), args.max_tokens)
                parts.append(part)
                time.sleep(args.sleep)

            merged = "\n\n---\n\n".join(parts)
            result = call_llm(client, args.model, build_merge_messages(merged, s, e), args.max_tokens)

        if "【END】" not in result:
            result = result.rstrip() + "\n\n【END】\n"

        saved = save_final(args.out_dir, s, e, result)
        print(f"✅ Saved: {saved}")
        time.sleep(args.sleep)

    print("🎉 Done.")


if __name__ == "__main__":
    main()