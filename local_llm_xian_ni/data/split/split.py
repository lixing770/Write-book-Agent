#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

# 兼容：
# - 第一卷 平庸少年 第001章 离乡
# - 第002章 仙人
# - 第3章 xxx
CHAPTER_RE = re.compile(
    r"^\s*(?:第[一二三四五六七八九十百千0-9０-９]+卷[^\n]*)?\s*第\s*([0-9０-９]{1,4}|[一二三四五六七八九十百千]+)\s*章[^\n]*$",
    re.MULTILINE,
)

def _normalize_digits(s: str) -> str:
    return s.translate(str.maketrans("０１２３４５６７８９", "0123456789"))

def safe_filename(s: str, max_len: int = 60) -> str:
    s = s.strip()
    s = re.sub(r"[\\/:*?\"<>|]+", "_", s)
    s = re.sub(r"\s+", " ", s)
    return (s[:max_len].strip() or "part")

def find_chapter_spans(text: str) -> List[Tuple[int, int, str]]:
    matches = list(CHAPTER_RE.finditer(text))
    if not matches:
        raise ValueError("No chapter headers matched. 请检查章标题格式或调整 CHAPTER_RE。")
    spans: List[Tuple[int, int, str]] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        header = m.group(0).strip()
        spans.append((start, end, header))
    return spans

def split_every_k_chapters(in_path: Path, out_dir: Path, k: int = 5) -> None:
    text = in_path.read_text(encoding="utf-8")
    spans = find_chapter_spans(text)

    out_dir.mkdir(parents=True, exist_ok=True)

    group_idx = 0
    for i in range(0, len(spans), k):
        group_idx += 1
        group = spans[i:i + k]

        start_i, _, header_first = group[0]
        _, end_j, header_last = group[-1]
        chunk = text[start_i:end_j].lstrip("\n")

        m1 = CHAPTER_RE.match(header_first)
        m2 = CHAPTER_RE.match(header_last)
        c1 = _normalize_digits(m1.group(1)) if m1 else str(i + 1)
        c2 = _normalize_digits(m2.group(1)) if m2 else str(min(i + k, len(spans)))

        filename = f"part_{group_idx:03d}_chap_{c1}-{c2}_{safe_filename(header_first)}__{safe_filename(header_last)}.txt"
        out_path = out_dir / filename
        out_path.write_text(chunk, encoding="utf-8")
        print(f"✅ {out_path} (chapters {i+1}-{min(i+k, len(spans))})")

def main() -> None:
    in_file = Path("/Users/50pai/Desktop/Writing book agent/local_llm_xian_ni/data/仙逆.txt")
    out_dir = in_file.parent / "/Users/50pai/Desktop/Writing book agent/local_llm_xian_ni/data/split"
    split_every_k_chapters(in_file, out_dir, k=10)

if __name__ == "__main__":
    main()
