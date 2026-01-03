#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional


# 兼容：第一卷 平庸少年、第二卷 XXX（如果你未来想按卷切）
VOLUME_RE = re.compile(r"^(第[一二三四五六七八九十百千0-9]+卷[^\n]*)\s*$", re.M)

# ✅ 兼容：
# - 第一卷 平庸少年 第001章 离乡
# - 第001章 离乡
# - 第1章 xxx
CHAPTER_LINE_RE = re.compile(
    r"^(?P<title>.*?(第\s*\d{1,5}\s*章[^\n]*))\s*$",
    re.M
)


def _load_text(p: Path) -> str:
    if not p.exists():
        raise RuntimeError(f"Input not found: {p}")
    return p.read_text(encoding="utf-8", errors="ignore")


def _find_volume_span(text: str, volume: int) -> Optional[Tuple[str, int, int]]:
    """
    Find span for the N-th volume in the text.
    Return (volume_title, start_idx, end_idx). If not found, return None.
    """
    matches = list(VOLUME_RE.finditer(text))
    if not matches:
        return None

    idx = volume - 1  # 1-based
    if idx < 0 or idx >= len(matches):
        return None

    start = matches[idx].start()
    end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
    title = matches[idx].group(1).strip()
    return (title, start, end)


def _find_chapter_spans(text: str) -> List[Tuple[str, int, int]]:
    """
    Return list of (chapter_title, start_idx, end_idx) spans.
    start_idx points to the beginning of the chapter title line.
    """
    matches = list(CHAPTER_LINE_RE.finditer(text))
    if not matches:
        return [("全文", 0, len(text))]

    spans: List[Tuple[str, int, int]] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        title = m.group("title").strip()
        spans.append((title, start, end))
    return spans


def _chunk_text(body: str, max_chars: int, overlap: int) -> List[Dict[str, Any]]:
    body = body.strip()
    if not body:
        return []
    chunks: List[Dict[str, Any]] = []
    step = max(1, max_chars - overlap)
    i = 0
    while i < len(body):
        j = min(len(body), i + max_chars)
        chunks.append({"start_char": i, "end_char": j, "text": body[i:j]})
        i += step
    return chunks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="data/仙逆.txt")
    ap.add_argument("--out", required=True, help="out/chunks.jsonl")

    ap.add_argument("--max_chars", type=int, default=2800)
    ap.add_argument("--overlap", type=int, default=200)

    # ✅ 你需要的：只抽取前 N 章（0=不限制）
    ap.add_argument("--max_chapters", type=int, default=0, help="0=all, otherwise only first N chapters")

    # 可选：限制最多输出多少个 chunk（防止量太大）
    ap.add_argument("--max_chunks", type=int, default=0, help="0=all, otherwise only first N chunks")

    # 可选：未来按卷抽取（你现在不用）
    ap.add_argument("--volume", type=int, default=0, help="0=all, 1=第一卷, 2=第二卷 ...")

    args = ap.parse_args()

    src = Path(args.input)
    dst = Path(args.out)
    dst.parent.mkdir(parents=True, exist_ok=True)

    full_text = _load_text(src)

    # 1) 按卷截取（可选）
    volume_title = ""
    text = full_text
    if args.volume and args.volume > 0:
        span = _find_volume_span(full_text, args.volume)
        if span is None:
            print(f"[WARN] volume={args.volume} not found. Fallback to full text.")
        else:
            volume_title, s, e = span
            text = full_text[s:e].strip()
            print(f"[OK] using volume: {volume_title}")

    # 2) 按章切分
    chapter_spans = _find_chapter_spans(text)

    # ✅ 只取前 N 章（你要 100 就传 100）
    if args.max_chapters and args.max_chapters > 0:
        chapter_spans = chapter_spans[: args.max_chapters]

    # 3) 章内切 chunk 并写 jsonl
    out_lines: List[Dict[str, Any]] = []
    chunk_id = 0

    for ci, (ch_title, s, e) in enumerate(chapter_spans, start=1):
        body = text[s:e].strip()
        chunks = _chunk_text(body, max_chars=args.max_chars, overlap=args.overlap)

        for k, c in enumerate(chunks, start=1):
            out_lines.append(
                {
                    "chunk_id": f"{chunk_id:06d}",
                    "volume": args.volume if args.volume else None,
                    "volume_title": volume_title if volume_title else None,
                    "chapter_index": ci,
                    "chapter_title": ch_title,
                    "part_index": k,
                    "start_char": c["start_char"],
                    "end_char": c["end_char"],
                    "text": c["text"],
                }
            )
            chunk_id += 1

            if args.max_chunks and len(out_lines) >= args.max_chunks:
                break

        if args.max_chunks and len(out_lines) >= args.max_chunks:
            break

    with dst.open("w", encoding="utf-8") as f:
        for obj in out_lines:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"[OK] wrote {len(out_lines)} chunks -> {dst}")
    print(f"[OK] chapters used: {len(chapter_spans)} (max_chapters={args.max_chapters or 'all'})")


if __name__ == "__main__":
    main()
