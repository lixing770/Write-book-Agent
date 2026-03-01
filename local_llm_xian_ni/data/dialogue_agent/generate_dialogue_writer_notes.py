#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate Dialogue Writer learning notes from novel raw text via DeepSeek API.

Supports input file modes:
1) Range-files mode: split_dir contains files like "1_10.txt", "11_20.txt", "51_75.txt", ...
2) Per-chapter mode: split_dir contains files like "chapter_1.txt", "chapter_2.txt", ...

Default mode: auto (prefer range-files if present)

Key features:
- Prevents API 400 due to huge prompts by trimming input via --input_max_chars
- Prints server error body for HTTP >= 400 to locate root cause quickly
- Guarantees output completeness by requiring end marker: 【END_OF_NOTE】
- Auto-continues if marker is missing
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import requests

END_MARK = "【END_OF_NOTE】"


# ---------------------------
# DeepSeek Client
# ---------------------------
@dataclass
class DeepSeekConfig:
    api_key: str
    base_url: str
    model: str
    temperature: float
    max_tokens: int
    timeout_sec: int


class DeepSeekClient:
    def __init__(self, cfg: DeepSeekConfig):
        self.cfg = cfg

    def chat(self, messages: List[dict]) -> str:
        """
        OpenAI-style chat schema:
        POST {base_url}/chat/completions
        """
        url = self.cfg.base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "stream": False,
        }

        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=self.cfg.timeout_sec)

        # IMPORTANT: print server details on errors (400 etc.)
        if resp.status_code >= 400:
            raise RuntimeError(
                f"DeepSeek HTTP {resp.status_code} Error\n"
                f"URL: {url}\n"
                f"Response: {resp.text[:4000]}\n"
                f"Payload(model/max_tokens/temp): {payload.get('model')}/{payload.get('max_tokens')}/{payload.get('temperature')}\n"
            )

        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except Exception:
            raise RuntimeError(f"Unexpected response schema: {data}")


# ---------------------------
# Prompts
# ---------------------------
SYSTEM_PROMPT = """你是一名“小说对话编剧总监/Showrunner Dialogue Writer”，擅长把长篇网文训练成可复用的对话编剧方法论与模板。
你的任务不是复述剧情，而是从输入原文中提炼可迁移的“对话写作规律、角色说话方式、冲突推进、潜台词设计、节奏控制、信息披露、笑点泪点触发、场景对话结构”。

硬性要求：
1) 输出必须完整：不允许半句结束、不允许列表断尾、不允许在未写完的章节结构中停下。
2) 你必须在文末输出唯一结束标记：【END_OF_NOTE】。没有这个标记就表示内容未完成。
3) 不要长段引用原文（避免逐字抄写）。可以用“抽象化示例/自造示例/改写示例”说明，不超过 2 句的短引用可以接受。
4) 输出中文，风格偏“主编学习笔记”，条理清晰，可直接当作知识库被下游 agent 使用。
5) 你必须严格按照我给定的【笔记结构大纲】输出（标题与小节必须齐全）。"""


def build_user_prompt(start_ch: int, end_ch: int, novel_text: str) -> str:
    return f"""下面是小说原文（第{start_ch}-{end_ch}章），请你学习这些文本，并按【笔记结构大纲】生成“对话编剧学习笔记”。

【笔记结构大纲】（标题必须一字不改，且每节都要写满）
# 【对话编剧学习笔记｜第{start_ch}-{end_ch}章】

## 1) 角色声音库（Voice Bible）
- 主角/核心角色：口头禅、句式习惯、情绪阈值、价值观触发点
- 配角/反派：语言特征与权力姿态（压迫/试探/讨好/威胁/交易）
- 群像区分技巧：如何用“长度、逻辑密度、词汇温度、攻击性”区分角色
- 可复用“角色说话参数表”（用表格）

## 2) 对话的功能拆解（每句台词为什么存在）
- 推剧情：推动事件/任务/交易/冲突升级
- 塑角色：立场、欲望、底线、软肋
- 铺设伏笔：信息延迟、误导、反转
- 控节奏：快问快答、打断、沉默、换话题
- 交代世界观：用“误会/争执/授课/审讯/谈判”自然塞信息

## 3) 冲突与张力：四种常用对话对抗模型
- 目标冲突（我要X你要Y）
- 信息不对称（我知道你不知道）
- 道德/立场冲突（价值观互斥）
- 权力压制（地位差/筹码差）
每种模型都要给：触发条件 → 结构模板 → 常见收束方式

## 4) 潜台词与“表里不一”的写法
- 表层话术 vs 真正意图
- 试探、套话、激将、借题发挥
- “反向承诺”“先否后肯”“假设式攻击”
给出 6 条可直接复用的句式模板

## 5) 场景化对话结构（把对话写成戏）
- 进入：开场钩子（误会/质问/突发消息/交易条件）
- 交锋：节拍（beat）递进的写法
- 变招：新的信息/新的筹码/新的情绪爆点
- 收束：留悬念/立约定/埋伏笔/制造代价
提供 3 套“场景对话骨架”，分别用于：谈判、审讯/对峙、暧昧/情绪戏

## 6) 高级技巧清单（从原文抽象出来的）
- 10 条技巧，每条包括：适用场景、操作步骤、常见翻车点、修正方式

## 7) 可直接给下游 agent 用的 Prompt（非常重要）
- Dialogue-Scene Generator Prompt（生成单场景）
- Character Voice Imitation Prompt（角色口吻）
- Conflict Escalation Prompt（冲突升级）
每个 prompt 都要可直接复制使用（含输入输出格式）。

【原文开始】
{novel_text}
【原文结束】

注意：你必须写到内容自然收束，并在最后输出{END_MARK}。
"""


# ---------------------------
# Utilities
# ---------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def trim_tail(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def shrink_novel_text(novel_text: str, max_chars: int) -> str:
    """
    Reduce prompt size to avoid context limit.
    Keep head + tail with a separator so model sees both.
    """
    if max_chars <= 0:
        return novel_text
    if len(novel_text) <= max_chars:
        return novel_text

    head_len = max_chars // 2
    head = novel_text[:head_len]
    tail = novel_text[-(max_chars - head_len):]
    return (
        head
        + "\n\n...[中间内容过长已截断，为避免超过上下文长度]...\n\n"
        + tail
    )


# ---------------------------
# Range-file detection (your split style)
# ---------------------------
_RANGE_RE = re.compile(r"^\s*(\d+)\s*_\s*(\d+)\s*\.txt\s*$")


def list_range_files(split_dir: Path) -> List[Tuple[int, int, Path]]:
    items: List[Tuple[int, int, Path]] = []
    for p in split_dir.iterdir():
        if not p.is_file():
            continue
        m = _RANGE_RE.match(p.name)
        if not m:
            continue
        a, b = int(m.group(1)), int(m.group(2))
        if a > b:
            a, b = b, a
        items.append((a, b, p))
    items.sort(key=lambda x: (x[0], x[1], x[2].name))
    return items


def pick_range_files(range_files: List[Tuple[int, int, Path]], start_ch: int, end_ch: int) -> List[Tuple[int, int, Path]]:
    chosen = []
    for a, b, p in range_files:
        if b < start_ch:
            continue
        if a > end_ch:
            break
        chosen.append((a, b, p))
    return chosen


def validate_coverage(chosen: List[Tuple[int, int, Path]], start_ch: int, end_ch: int) -> None:
    covered = [False] * (end_ch - start_ch + 1)
    for a, b, _ in chosen:
        lo = max(a, start_ch)
        hi = min(b, end_ch)
        for ch in range(lo, hi + 1):
            covered[ch - start_ch] = True

    if not all(covered):
        missing = [start_ch + i for i, ok in enumerate(covered) if not ok]
        gaps = []
        s = None
        prev = None
        for ch in missing:
            if s is None:
                s = ch
                prev = ch
            elif ch == prev + 1:
                prev = ch
            else:
                gaps.append((s, prev))
                s = ch
                prev = ch
        if s is not None:
            gaps.append((s, prev))
        gap_str = ", ".join([f"{a}-{b}" if a != b else f"{a}" for a, b in gaps])
        raise RuntimeError(f"Range files do not fully cover chapters {start_ch}-{end_ch}. Missing: {gap_str}")


def read_range_files(split_dir: Path, start_ch: int, end_ch: int) -> Tuple[str, List[Path]]:
    rf = list_range_files(split_dir)
    if not rf:
        raise RuntimeError("No range files like '1_10.txt' found in split_dir.")

    chosen = pick_range_files(rf, start_ch, end_ch)
    if not chosen:
        raise RuntimeError(f"No range files overlap with chapters {start_ch}-{end_ch}.")

    validate_coverage(chosen, start_ch, end_ch)

    texts: List[str] = []
    used: List[Path] = []
    for a, b, p in chosen:
        raw = p.read_text(encoding="utf-8", errors="replace").strip()
        texts.append(f"\n\n===== 文件 {p.name}（覆盖 {a}-{b}）=====\n{raw}\n")
        used.append(p)

    return "\n".join(texts).strip(), used


# ---------------------------
# Per-chapter mode (optional)
# ---------------------------
def format_glob(chapter_glob: str, num: int) -> str:
    return chapter_glob.format(num=num)


def read_per_chapter_files(split_dir: Path, start_ch: int, end_ch: int, chapter_glob: str) -> Tuple[str, List[Path]]:
    texts: List[str] = []
    used: List[Path] = []
    for ch in range(start_ch, end_ch + 1):
        fname = format_glob(chapter_glob, ch)
        fpath = split_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Missing chapter file: {fpath}")
        raw = fpath.read_text(encoding="utf-8", errors="replace").strip()
        texts.append(f"\n\n===== 第{ch}章 =====\n{raw}\n")
        used.append(fpath)
    return "\n".join(texts).strip(), used


# ---------------------------
# Generation logic
# ---------------------------
def generate_complete_note(
    client: DeepSeekClient,
    start_ch: int,
    end_ch: int,
    novel_text: str,
    max_continue_rounds: int,
    continue_context_chars: int,
    sleep_sec: float,
) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(start_ch, end_ch, novel_text)},
    ]

    out = client.chat(messages).strip()
    time.sleep(sleep_sec)

    if END_MARK in out:
        return out

    for _ in range(max_continue_rounds):
        tail = trim_tail(out, continue_context_chars)
        cont_user = f"""你刚才的输出没有以 {END_MARK} 收尾，说明未完成。
请从你未写完的位置继续，保持原有结构，不要重复已写完的内容。
强制要求：补齐断尾的列表/小节，并在全文最后输出 {END_MARK}。

【你上一次输出的末尾上下文】
{tail}
"""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": cont_user},
        ]
        more = client.chat(messages).strip()
        time.sleep(sleep_sec)

        out = out.rstrip() + "\n\n" + more.lstrip()
        if END_MARK in out:
            return out

    out = out.rstrip() + "\n\n> [WARN] 多轮续写仍未返回结束标记，已强制截断。\n" + END_MARK
    return out


def write_index(out_dir: Path, items: List[Tuple[int, int, str]]) -> None:
    lines = ["# Dialogue Writer Notes Index\n"]
    for s, e, fname in items:
        lines.append(f"- 第{s}-{e}章：{fname}")
    (out_dir / "INDEX.md").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


# ---------------------------
# Main
# ---------------------------
def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--start_chapter", type=int, required=True)
    ap.add_argument("--end_chapter", type=int, default=None)
    ap.add_argument("--batch_chapters", type=int, default=50)

    ap.add_argument("--input_mode", choices=["auto", "range", "chapter"], default="auto",
                    help="auto: prefer range-files like 1_10.txt; range: force range-files; chapter: force per-chapter files")

    ap.add_argument("--chapter_glob", type=str, default="chapter_{num}.txt",
                    help='Only used in chapter mode. e.g. "chapter_{num}.txt" or "ch_{num:04d}.md"')

    ap.add_argument("--input_max_chars", type=int, default=120000,
                    help="Max chars of novel text per batch to avoid API 400/context limit. 0=disable.")

    ap.add_argument("--api_key", type=str, required=True)
    ap.add_argument("--base_url", type=str, default="https://api.deepseek.com")
    ap.add_argument("--model", type=str, default="deepseek-reasoner")

    ap.add_argument("--temperature", type=float, default=0.25)
    ap.add_argument("--max_tokens", type=int, default=2500)  # safer default than 5000
    ap.add_argument("--timeout_sec", type=int, default=300)

    ap.add_argument("--max_continue_rounds", type=int, default=10)
    ap.add_argument("--continue_context_chars", type=int, default=4000)
    ap.add_argument("--sleep_sec", type=float, default=0.4)

    args = ap.parse_args()

    split_dir = Path(args.split_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    ensure_dir(out_dir)

    cfg = DeepSeekConfig(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout_sec=args.timeout_sec,
    )
    client = DeepSeekClient(cfg)

    start = args.start_chapter
    end = args.end_chapter if args.end_chapter is not None else (start + args.batch_chapters - 1)

    # detect range files once
    has_range_files = bool(list_range_files(split_dir))

    index_items: List[Tuple[int, int, str]] = []
    cur = start

    while cur <= end:
        batch_start = cur
        batch_end = min(cur + args.batch_chapters - 1, end)

        mode = args.input_mode
        if mode == "auto":
            mode = "range" if has_range_files else "chapter"

        if mode == "range":
            novel_text, used_files = read_range_files(split_dir, batch_start, batch_end)
        else:
            novel_text, used_files = read_per_chapter_files(split_dir, batch_start, batch_end, args.chapter_glob)

        # shrink to avoid context blow-up -> HTTP 400
        novel_text = shrink_novel_text(novel_text, args.input_max_chars)

        note = generate_complete_note(
            client=client,
            start_ch=batch_start,
            end_ch=batch_end,
            novel_text=novel_text,
            max_continue_rounds=args.max_continue_rounds,
            continue_context_chars=args.continue_context_chars,
            sleep_sec=args.sleep_sec,
        )

        out_name = f"dialogue_writer_notes_{batch_start}_{batch_end}.md"
        (out_dir / out_name).write_text(note.rstrip() + "\n", encoding="utf-8")

        meta_name = f"dialogue_writer_notes_{batch_start}_{batch_end}.meta.json"
        meta = {
            "batch_start": batch_start,
            "batch_end": batch_end,
            "input_mode": mode,
            "input_max_chars": args.input_max_chars,
            "source_files": [str(p) for p in used_files],
            "model": args.model,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        }
        (out_dir / meta_name).write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        print(f"[OK] Wrote {out_name}")
        index_items.append((batch_start, batch_end, out_name))

        cur = batch_end + 1

    write_index(out_dir, index_items)
    print("[DONE] INDEX.md generated.")


if __name__ == "__main__":
    main()