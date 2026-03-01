#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Summarize/merge dialogue_writer_notes_*.md into ONE master note (~8000 Chinese characters)
using DeepSeek chat/completions API.

Features:
- Reads all notes in a dir (glob pattern)
- Bundles notes into manageable chunks for context limit
- Multi-stage summarization:
  Stage A: summarize each file -> "mini summary"
  Stage B: merge all mini summaries -> master note (~8000 chars)
- Completeness guaranteed with end marker: 【END_OF_MASTER_NOTE】
- Auto-continue if output is truncated
- Length control: target_chars ± tolerance_percent (default 8%)
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

END_MARK = "【END_OF_MASTER_NOTE】"


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
SYSTEM_MASTER = """你是“小说对话编剧总监（总编）”，你的任务是：把多份【对话编剧学习笔记】整合成一份“总学习笔记（知识库版）”。
你必须做到：
1) 不复述原文剧情；只抽象方法论、模板、参数表、可复用 Prompt。
2) 输出必须完整，不能半句结束、不能列表断尾、不能突然停在某一节。
3) 必须在文末输出唯一结束标记：【END_OF_MASTER_NOTE】（没有该标记视为未完成）。
4) 输出中文，风格偏“主编知识库”，层级清晰、可直接喂给下游 agent。
5) 目标长度：约 8000 字（允许 ±8% 浮动）。过短必须补充；过长必须压缩，但结构不能丢。
"""

USER_MASTER_TEMPLATE = """下面是多份《对话编剧学习笔记》（不同章节批次产物）。请你先学习所有内容，然后整合出一份新的《总对话编剧学习笔记（知识库版）》。

【输出结构必须严格如下（标题一字不改）】
# 总对话编剧学习笔记（知识库版｜整合汇总）

## 1) 核心总原则（10条）
- 每条：一句原则 + 一句解释 + 常见误区

## 2) 角色声音库（Voice Bible）通用建模
- 角色说话参数表（必须表格）
- 角色区分的“可量化维度”（至少8个维度）
- 常见角色类型：主角/导师/反派/群像（各给模板）

## 3) 对话的功能地图（从句子层到场景层）
- 句子层：台词功能分类 + 触发条件
- 段落层：对话段落的推进结构（至少3种）
- 场景层：对话场景的目标、筹码、代价、转折

## 4) 冲突与张力：四大对抗模型（模板化）
- 每个模型：触发条件 → 结构模板 → 升级策略 → 收束方式 → 示例（自造示例）

## 5) 潜台词与话术武器库
- 试探/套话/激将/借题发挥/反向承诺/先否后肯/假设式攻击
- 12条可直接复用句式模板（必须编号）

## 6) 场景化对话骨架（写成戏）
- 谈判骨架（beat 递进）
- 审讯/对峙骨架
- 情绪戏/暧昧骨架
- 每个骨架：进入→交锋→变招→收束，并给“可替换槽位”

## 7) 高级技巧清单（最少12条）
- 每条：适用场景｜操作步骤｜翻车点｜修正方式

## 8) 下游可直接用的 Prompt 套件（非常重要）
- Dialogue-Scene Generator Prompt
- Character Voice Imitation Prompt
- Conflict Escalation Prompt
- Subtext Rewriter Prompt（把直白台词改成有潜台词）
每个 Prompt 必须给：输入字段、输出格式、约束规则、示例

【输入笔记开始】
{NOTES_TEXT}
【输入笔记结束】

要求：全文自然收束，最后必须输出【END_OF_MASTER_NOTE】。
目标长度约{TARGET_CHARS}字（±{TOL}%）。
"""

SYSTEM_MINI = """你是一名“小说对话编剧总监助理”。你将收到一份《对话编剧学习笔记》。
你的任务：抽取其中的“可复用知识”，输出一份“精炼二级摘要”，用于后续总编整合。
要求：
- 不复述剧情，不抄原文。
- 保留：方法论、模板、参数表、可复用 prompt 的关键点。
- 输出必须完整，最后输出标记：【END_OF_MINI】。
"""

USER_MINI_TEMPLATE = """请将下面这份学习笔记提炼成“精炼二级摘要”，方便后续合并。
输出格式：
# MINI_SUMMARY｜{NAME}
- 核心原则要点（最多12条）
- 角色声音建模要点（参数维度）
- 冲突模型/潜台词/场景骨架：各自最关键模板
- 下游 prompt 组件：字段与约束（简述）

【输入开始】
{TEXT}
【输入结束】

最后必须输出：【END_OF_MINI】
"""


# ---------------------------
# Helpers
# ---------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def trim_tail(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[-max_chars:]


def shrink_text_keep_head_tail(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head_len = max_chars // 2
    head = text[:head_len]
    tail = text[-(max_chars - head_len):]
    return head + "\n\n...[中间过长已截断]...\n\n" + tail


def complete_with_marker(
    client: DeepSeekClient,
    system_prompt: str,
    user_prompt: str,
    end_marker: str,
    max_continue_rounds: int,
    continue_context_chars: int,
    sleep_sec: float,
) -> str:
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    out = client.chat(messages).strip()
    time.sleep(sleep_sec)

    if end_marker in out:
        return out

    for _ in range(max_continue_rounds):
        tail = trim_tail(out, continue_context_chars)
        cont_user = f"""你刚才的输出没有以 {end_marker} 收尾，说明未完成。
请从未写完的位置继续，保持结构，不要重复已写完内容；补齐断尾列表/小节；
并在全文最后输出 {end_marker}。

【末尾上下文】
{tail}
"""
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": cont_user}]
        more = client.chat(messages).strip()
        time.sleep(sleep_sec)
        out = out.rstrip() + "\n\n" + more.lstrip()
        if end_marker in out:
            return out

    # hard fail-safe
    return out.rstrip() + "\n\n> [WARN] 多轮续写仍未返回结束标记，已强制截断。\n" + end_marker


def estimate_len(s: str) -> int:
    # 近似：中文字符+标点+字母都算 1
    return len(s)


def adjust_length_pass(
    client: DeepSeekClient,
    text: str,
    target_chars: int,
    tolerance_percent: int,
    max_tokens: int,
    temperature: float,
    timeout_sec: int,
    base_url: str,
    api_key: str,
    model: str,
) -> str:
    """
    One polishing pass to expand/compress into target range.
    Keeps END_MARK requirement.
    """
    low = int(target_chars * (1 - tolerance_percent / 100))
    high = int(target_chars * (1 + tolerance_percent / 100))
    cur = estimate_len(text)

    if low <= cur <= high and END_MARK in text:
        return text

    direction = "扩写补充细节" if cur < low else "压缩精炼但不丢结构"
    sys_p = SYSTEM_MASTER
    user_p = f"""下面是你已经生成的《总对话编剧学习笔记》草稿。
当前长度约 {cur} 字，目标 {target_chars} 字（±{tolerance_percent}%）。
请在不改变既定结构标题的前提下，进行一次{direction}，使最终长度落入区间 [{low}, {high}]。
要求：
- 结构标题必须保留且完整
- 任何列表不能断尾
- Prompt 套件必须保留并可直接复制
- 最后必须输出 {END_MARK}

【草稿开始】
{text}
【草稿结束】
"""
    cfg = DeepSeekConfig(
        api_key=api_key, base_url=base_url, model=model,
        temperature=temperature, max_tokens=max_tokens, timeout_sec=timeout_sec
    )
    polish_client = DeepSeekClient(cfg)
    return complete_with_marker(
        polish_client, sys_p, user_p, END_MARK,
        max_continue_rounds=8, continue_context_chars=4000, sleep_sec=0.4
    )


# ---------------------------
# Pipeline
# ---------------------------
def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--notes_dir", required=True, help="Directory containing dialogue_writer_notes_*.md")
    ap.add_argument("--out_file", required=True, help="Output master md file")
    ap.add_argument("--glob", default="dialogue_writer_notes_*.md", help="Glob pattern in notes_dir")

    ap.add_argument("--api_key", required=True)
    ap.add_argument("--base_url", default="https://api.deepseek.com")
    ap.add_argument("--model", default="deepseek-reasoner")

    ap.add_argument("--temperature", type=float, default=0.25)
    ap.add_argument("--max_tokens", type=int, default=3200)
    ap.add_argument("--timeout_sec", type=int, default=300)

    ap.add_argument("--target_chars", type=int, default=8000)
    ap.add_argument("--tolerance_percent", type=int, default=8)

    ap.add_argument("--mini_input_max_chars", type=int, default=80000, help="Trim each note before mini-summary")
    ap.add_argument("--master_input_max_chars", type=int, default=140000, help="Trim all-mini-summary bundle before master merge")

    ap.add_argument("--max_continue_rounds", type=int, default=12)
    ap.add_argument("--continue_context_chars", type=int, default=5000)
    ap.add_argument("--sleep_sec", type=float, default=0.4)

    args = ap.parse_args()

    notes_dir = Path(args.notes_dir).expanduser().resolve()
    out_file = Path(args.out_file).expanduser().resolve()
    ensure_dir(out_file.parent)

    files = sorted(notes_dir.glob(args.glob))
    if not files:
        raise RuntimeError(f"No files matched: {notes_dir}/{args.glob}")

    cfg = DeepSeekConfig(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout_sec=args.timeout_sec,
    )
    client = DeepSeekClient(cfg)

    # Stage A: mini summaries per file
    minis: List[str] = []
    mini_meta: List[dict] = []

    for p in files:
        raw = read_text(p)
        raw = shrink_text_keep_head_tail(raw, args.mini_input_max_chars)

        user_p = USER_MINI_TEMPLATE.format(NAME=p.name, TEXT=raw)
        mini = complete_with_marker(
            client=client,
            system_prompt=SYSTEM_MINI,
            user_prompt=user_p,
            end_marker="【END_OF_MINI】",
            max_continue_rounds=args.max_continue_rounds,
            continue_context_chars=args.continue_context_chars,
            sleep_sec=args.sleep_sec,
        )
        minis.append(mini)
        mini_meta.append({"file": str(p), "mini_len": estimate_len(mini)})
        print(f"[MINI OK] {p.name} -> {estimate_len(mini)} chars")

    # Bundle minis for master
    bundle = "\n\n" + ("\n\n" + ("=" * 60) + "\n\n").join(minis)
    bundle = shrink_text_keep_head_tail(bundle, args.master_input_max_chars)

    user_master = USER_MASTER_TEMPLATE.format(
        NOTES_TEXT=bundle,
        TARGET_CHARS=args.target_chars,
        TOL=args.tolerance_percent,
    )

    # Stage B: master merge
    master = complete_with_marker(
        client=client,
        system_prompt=SYSTEM_MASTER,
        user_prompt=user_master,
        end_marker=END_MARK,
        max_continue_rounds=args.max_continue_rounds,
        continue_context_chars=args.continue_context_chars,
        sleep_sec=args.sleep_sec,
    )

    # Stage C: length adjust if needed
    master = adjust_length_pass(
        client=client,
        text=master,
        target_chars=args.target_chars,
        tolerance_percent=args.tolerance_percent,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout_sec=args.timeout_sec,
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
    )

    out_file.write_text(master.rstrip() + "\n", encoding="utf-8")

    meta_path = out_file.with_suffix(".meta.json")
    meta = {
        "notes_dir": str(notes_dir),
        "glob": args.glob,
        "files": [str(p) for p in files],
        "mini_meta": mini_meta,
        "target_chars": args.target_chars,
        "tolerance_percent": args.tolerance_percent,
        "model": args.model,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[DONE] Master note written: {out_file}")
    print(f"[DONE] Meta written: {meta_path}")


if __name__ == "__main__":
    main()