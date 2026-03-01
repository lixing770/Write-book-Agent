#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
工程化：对 study_outline_{start}_{end}.txt 做多轮“每5个合并一次”，直到只剩 1 个最终文件。

输入：--in_dir 目录（例如 outline/merged_5）
中间输出：--out_root/merged_level_01, merged_level_02, ...
最终输出：--out_root/final/
  - study_outline_{minStart}_{maxEnd}.txt
  - FINAL_STUDY_OUTLINE.txt （固定名便于后续读取）
日志：--out_root/logs/merge_until_one.log

依赖：merge_5_outlines.py（你之前用来“每5个合并”的脚本）
"""

from __future__ import annotations

import re
import os
import sys
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional


RANGE_RE = re.compile(r"study_outline_(\d+)_(\d+)\.txt$")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_line(log_path: Path, msg: str) -> None:
    ensure_dir(log_path.parent)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{now_str()}] {msg}\n")


def scan_outlines(d: Path) -> List[Tuple[int, int, Path]]:
    items: List[Tuple[int, int, Path]] = []
    for p in sorted(d.glob("study_outline_*.txt")):
        m = RANGE_RE.search(p.name)
        if not m:
            continue
        a, b = int(m.group(1)), int(m.group(2))
        if a > b:
            a, b = b, a
        items.append((a, b, p))
    items.sort(key=lambda x: (x[0], x[1]))
    return items


def pick_final_file(d: Path) -> Optional[Tuple[int, int, Path]]:
    items = scan_outlines(d)
    if not items:
        return None
    if len(items) == 1:
        return items[0]
    # 如果意外多于1个，取覆盖范围最大的那个兜底
    items.sort(key=lambda x: (x[1] - x[0], x[0]), reverse=True)
    return items[0]


def run_merge_once(
    merge_script: Path,
    in_dir: Path,
    out_dir: Path,
    batch_size: int,
    chunk_chars: int,
    base_url: str,
    model: str,
    temperature: float,
    max_tokens_update: int,
    max_tokens_section: int,
    include_training_plan: bool,
    log_path: Path,
) -> None:
    ensure_dir(out_dir)

    cmd = [
        sys.executable,
        str(merge_script),
        "--in_dir", str(in_dir),
        "--out_dir", str(out_dir),
        "--batch_size", str(batch_size),
        "--chunk_chars", str(chunk_chars),
        "--base_url", base_url,
        "--model", model,
        "--temperature", str(temperature),
        "--max_tokens_update", str(max_tokens_update),
        "--max_tokens_section", str(max_tokens_section),
    ]
    if include_training_plan:
        cmd.append("--include_training_plan")

    log_line(log_path, f"RUN merge_once: in_dir={in_dir} out_dir={out_dir} batch_size={batch_size}")
    log_line(log_path, f"CMD: {' '.join(cmd)}")

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log_line(log_path, "----- merge_5_outlines.py OUTPUT BEGIN -----")
    log_line(log_path, proc.stdout.rstrip() if proc.stdout else "")
    log_line(log_path, "----- merge_5_outlines.py OUTPUT END -----")

    if proc.returncode != 0:
        raise RuntimeError(f"merge_5_outlines.py 运行失败，returncode={proc.returncode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge_script", type=str, required=True, help="merge_5_outlines.py 的路径")
    ap.add_argument("--in_dir", type=str, required=True, help="起始输入目录（含 study_outline_*.txt）")
    ap.add_argument("--out_root", type=str, required=True, help="多轮合并输出根目录")
    ap.add_argument("--batch_size", type=int, default=5)

    ap.add_argument("--chunk_chars", type=int, default=9000)
    ap.add_argument("--base_url", type=str, default="https://api.deepseek.com")
    ap.add_argument("--model", type=str, default="deepseek-reasoner")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max_tokens_update", type=int, default=2000)
    ap.add_argument("--max_tokens_section", type=int, default=1400)
    ap.add_argument("--include_training_plan", action="store_true")

    args = ap.parse_args()

    merge_script = Path(args.merge_script).expanduser()
    in_dir = Path(args.in_dir).expanduser()
    out_root = Path(args.out_root).expanduser()
    ensure_dir(out_root)

    log_path = out_root / "logs" / "merge_until_one.log"
    ensure_dir(log_path.parent)

    if not merge_script.exists():
        raise RuntimeError(f"merge_script 不存在：{merge_script}")
    if not in_dir.exists():
        raise RuntimeError(f"in_dir 不存在：{in_dir}")

    # 初始扫描
    items0 = scan_outlines(in_dir)
    if not items0:
        raise RuntimeError(f"在 {in_dir} 没找到 study_outline_*.txt")

    global_start = min(x[0] for x in items0)
    global_end = max(x[1] for x in items0)

    log_line(log_path, f"START: in_dir={in_dir} files={len(items0)} range={global_start}-{global_end}")
    log_line(log_path, f"merge_script={merge_script}")
    log_line(log_path, f"model={args.model} base_url={args.base_url} batch_size={args.batch_size}")

    current_dir = in_dir
    level = 1

    while True:
        items = scan_outlines(current_dir)
        if len(items) <= 1:
            break

        next_dir = out_root / f"merged_level_{level:02d}"
        run_merge_once(
            merge_script=merge_script,
            in_dir=current_dir,
            out_dir=next_dir,
            batch_size=int(args.batch_size),
            chunk_chars=int(args.chunk_chars),
            base_url=str(args.base_url),
            model=str(args.model),
            temperature=float(args.temperature),
            max_tokens_update=int(args.max_tokens_update),
            max_tokens_section=int(args.max_tokens_section),
            include_training_plan=bool(args.include_training_plan),
            log_path=log_path,
        )

        # 防止无进展死循环
        after = scan_outlines(next_dir)
        log_line(log_path, f"LEVEL {level:02d} DONE: out_files={len(after)} dir={next_dir}")
        if len(after) >= len(items):
            raise RuntimeError(
                f"合并后文件数没有减少（可能命名不匹配或输出未生成）。before={len(items)} after={len(after)}"
            )

        current_dir = next_dir
        level += 1

    final_item = pick_final_file(current_dir)
    if not final_item:
        raise RuntimeError("最终没有找到任何 study_outline 文件")

    a, b, final_path = final_item
    final_dir = out_root / "final"
    ensure_dir(final_dir)

    dst1 = final_dir / f"study_outline_{global_start}_{global_end}.txt"
    dst2 = final_dir / "FINAL_STUDY_OUTLINE.txt"

    shutil.copy2(final_path, dst1)
    shutil.copy2(final_path, dst2)

    log_line(log_path, f"FINAL: copied {final_path} -> {dst1}")
    log_line(log_path, f"FINAL: copied {final_path} -> {dst2}")

    print("✅ DONE")
    print(f"- Final (range): {dst1}")
    print(f"- Final (fixed): {dst2}")
    print(f"- Log: {log_path}")


if __name__ == "__main__":
    main()
