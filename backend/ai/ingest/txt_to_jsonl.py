#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert novel .txt files into paragraph-level jsonl for downstream analysis/RAG.

Input:
  data/corpora/raw_txt/*.txt

Output:
  data/corpora/jsonl/<same_name>.jsonl

Each line:
  {"doc_id": "...", "chapter": <int>, "para_id": <int>, "text": "...", "meta": {...}}

Heuristics:
- Detect chapter headers like: "第1章", "第十二章", "CHAPTER 1", "Chapter 1"
- Split by blank lines; if no blank lines, split by punctuation-based chunking.
- Keep paragraph length roughly within [min_chars, max_chars] by merging/splitting.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple

CHAPTER_PATTERNS = [
    re.compile(r"^\s*第\s*([0-9一二三四五六七八九十百千两]+)\s*章.*$"),
    re.compile(r"^\s*第\s*([0-9一二三四五六七八九十百千两]+)\s*节.*$"),
    re.compile(r"^\s*(chapter|CHAPTER)\s+(\d+)\b.*$"),
]

CN_NUM = {"零":0,"一":1,"二":2,"两":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9,"十":10,"百":100,"千":1000}

def cn_to_int(s: str) -> int:
    s = s.strip()
    if s.isdigit():
        return int(s)
    # very small CN numeral converter (good for 1-9999)
    total, unit, num = 0, 1, 0
    for ch in reversed(s):
        if ch in ("十","百","千"):
            unit = CN_NUM[ch]
            if num == 0:
                num = 1
            total += num * unit
            num = 0
            unit = 1
        elif ch in CN_NUM:
            num = CN_NUM[ch]
        else:
            # unknown char -> ignore
            continue
    total += num * unit
    return total if total > 0 else 0

def normalize_text(t: str) -> str:
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    # drop common junk lines
    lines = []
    for line in t.split("\n"):
        line = line.strip()
        if not line:
            lines.append("")
            continue
        if "本书来自" in line or "更多精彩" in line or "请收藏" in line:
            continue
        lines.append(line)
    # collapse multiple blank lines
    out = []
    blank = 0
    for line in lines:
        if line == "":
            blank += 1
            if blank <= 1:
                out.append("")
        else:
            blank = 0
            out.append(line)
    return "\n".join(out).strip()

def detect_chapter(line: str) -> Tuple[bool, int]:
    for pat in CHAPTER_PATTERNS:
        m = pat.match(line)
        if not m:
            continue
        if len(m.groups()) >= 2 and m.group(1).lower() == "chapter":
            return True, int(m.group(2))
        # Chinese: first group is cn number
        return True, cn_to_int(m.group(1))
    return False, 0

def split_paragraphs(block: str) -> List[str]:
    # split by blank lines first
    paras = [p.strip() for p in re.split(r"\n\s*\n+", block) if p.strip()]
    return paras

def split_by_punct(text: str, max_chars: int) -> List[str]:
    # fallback: split by sentence punctuation
    sents = re.split(r"(?<=[。！？!?])\s*", text)
    chunks, cur = [], ""
    for s in sents:
        if not s:
            continue
        if len(cur) + len(s) <= max_chars:
            cur += ("" if not cur else " ") + s
        else:
            if cur:
                chunks.append(cur.strip())
            cur = s
    if cur:
        chunks.append(cur.strip())
    return chunks

def pack_paras(paras: List[str], min_chars: int, max_chars: int) -> List[str]:
    out = []
    buf = ""
    for p in paras:
        if not p:
            continue
        if len(p) > max_chars:
            # split long para
            pieces = split_by_punct(p, max_chars)
            for piece in pieces:
                if len(piece) < min_chars and out:
                    out[-1] = (out[-1] + " " + piece).strip()
                else:
                    out.append(piece)
            continue

        if not buf:
            buf = p
        elif len(buf) + 1 + len(p) <= max_chars:
            buf = (buf + " " + p).strip()
        else:
            if len(buf) < min_chars and out:
                out[-1] = (out[-1] + " " + buf).strip()
            else:
                out.append(buf.strip())
            buf = p

    if buf:
        if len(buf) < min_chars and out:
            out[-1] = (out[-1] + " " + buf).strip()
        else:
            out.append(buf.strip())
    return out

def convert_file(in_path: Path, out_path: Path, min_chars: int, max_chars: int) -> int:
    raw = in_path.read_text(encoding="utf-8", errors="ignore")
    text = normalize_text(raw)

    # parse chapters by headers
    lines = text.split("\n")
    chapters: List[Tuple[int, List[str]]] = []
    cur_ch = 0
    cur_lines: List[str] = []

    for line in lines:
        is_ch, ch_num = detect_chapter(line)
        if is_ch:
            # flush previous
            if cur_lines:
                chapters.append((cur_ch, cur_lines))
            cur_ch = ch_num if ch_num > 0 else (cur_ch + 1 if cur_ch >= 0 else 1)
            cur_lines = []
            continue
        cur_lines.append(line)

    if cur_lines:
        chapters.append((cur_ch, cur_lines))

    # if no chapter header detected, treat as chapter 1
    if len(chapters) == 1 and chapters[0][0] == 0:
        chapters = [(1, chapters[0][1])]

    doc_id = in_path.stem
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for ch_num, ch_lines in chapters:
            block = "\n".join(ch_lines).strip()
            if not block:
                continue
            paras = split_paragraphs(block)
            if len(paras) <= 1:
                paras = split_by_punct(block, max_chars=max_chars)

            paras = pack_paras(paras, min_chars=min_chars, max_chars=max_chars)

            for i, p in enumerate(paras, start=1):
                rec = {
                    "doc_id": doc_id,
                    "chapter": int(ch_num) if ch_num else 1,
                    "para_id": i,
                    "text": p,
                    "meta": {
                        "source_file": in_path.name,
                        "lang": "zh" if re.search(r"[\u4e00-\u9fff]", p) else "en",
                    },
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
    return n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="data/corpora/raw_txt", help="input folder of .txt novels")
    ap.add_argument("--out_dir", default="data/corpora/jsonl", help="output folder for .jsonl")
    ap.add_argument("--min_chars", type=int, default=200, help="min chars per paragraph chunk")
    ap.add_argument("--max_chars", type=int, default=900, help="max chars per paragraph chunk")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    txts = sorted(in_dir.glob("*.txt"))
    if not txts:
        raise SystemExit(f"No .txt files found in: {in_dir}")

    total = 0
    for p in txts:
        out_path = out_dir / f"{p.stem}.jsonl"
        n = convert_file(p, out_path, args.min_chars, args.max_chars)
        print(f"[OK] {p.name} -> {out_path} ({n} lines)")
        total += n
    print(f"[DONE] total lines: {total}")

if __name__ == "__main__":
    main()
