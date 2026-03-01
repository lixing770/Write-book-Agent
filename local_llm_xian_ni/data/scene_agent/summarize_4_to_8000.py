#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Summarize 4 batch notes -> one ~8000 Chinese chars master note (DeepSeek) [STRONG VERSION]
---------------------------------------------------------------------------------------
解决点：
- 自动续写（finish_reason=length）
- 逐份融合，避免一次性输入过长导致缩水
- 强制每节最小字数 + 句式模板数量 + 卡片数量
- 多轮扩写校准直到达标

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
from typing import List, Tuple

import requests


# ----------------------------
# DeepSeek API
# ----------------------------

@dataclass
class DeepSeekConfig:
    api_key: str
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-reasoner"
    max_tokens: int = 10000
    temperature: float = 0.25
    timeout_sec: int = 180


class DeepSeekAPI:
    def __init__(self, cfg: DeepSeekConfig):
        self.cfg = cfg

    def chat(self, system: str, user: str) -> tuple[str, str]:
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


def call_with_retries(fn, max_retries: int):
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except Exception:
            if attempt == max_retries:
                raise
            sleep_backoff(attempt)


def chat_until_complete(api: DeepSeekAPI, system: str, user: str, max_rounds: int = 12) -> str:
    full = ""
    cur_user = user
    for _ in range(max_rounds):
        content, finish = api.chat(system, cur_user)
        full += content.rstrip() + "\n"
        if finish != "length":
            break
        tail = full[-1800:]
        cur_user = (
            "你刚才的输出被截断了（finish_reason=length）。请从中断处继续，"
            "保持相同结构与编号，不要重复已写内容；必须写到完整结束。\n\n"
            f"以下是你刚才输出的结尾片段（仅用于定位续写点）：\n<<<TAIL\n{tail}\nTAIL>>>"
        )
    return full.strip() + "\n"


# ----------------------------
# Prompts
# ----------------------------

SYSTEM_MASTER = """你是“情绪与场景氛围渲染主编（Master Editor）”。
你将收到多个批次的学习笔记，请合并为一份“可复用写作手册”。

硬性要求：
1) 输出必须完整，不能半截结束；如内容过长请压缩冗余但不能缺结构。
2) 不要复述剧情；只保留方法论、套路拆解、清单、模板、禁忌与自检。
3) 写作口吻：专业、可执行、像写作教程；多用条列与小标题。
4) 目标长度：约 {TARGET} 字（允许±{TOL}%）。
5) 输出必须是 Markdown，且包含固定结构标题（10个二级标题）。"""

MASTER_HEADINGS = [
    "1) 总公式：情绪/氛围渲染的最小闭环",
    "2) 情绪谱系与触发器：从“冷暖、松紧、明暗、动静”建模",
    "3) 多感官渲染：声场/触感/嗅觉/温度/重量感的模板",
    "4) 场景与情绪绑定：空间层级、动线、视角如何制造情绪",
    "5) 节奏与镜头：段落长度、信息密度、远中近切换",
    "6) 可复用句式库：按情绪与场景分类（可直接粘贴）",
    "7) 常见高级套路：对比、留白、反差、延迟揭示、意象复用",
    "8) 禁忌清单：最容易写成流水账/假情绪的坑",
    "9) 自检清单：写完一段如何判定“情绪到位”",
    "10) 速用卡片：12张“场景×情绪”即插即用配方",
]


def build_skeleton() -> str:
    out = ["# 情绪与场景氛围渲染·汇总学习笔记（完整版）", ""]
    for h in MASTER_HEADINGS:
        out.append(f"## {h}\n- （待生成）\n")
    return "\n".join(out)


def normalize_md(md: str) -> str:
    out = md.strip()
    if not out.startswith("# "):
        out = "# 情绪与场景氛围渲染·汇总学习笔记（完整版）\n\n" + out
    for h in MASTER_HEADINGS:
        if f"## {h}" not in out:
            out += f"\n\n## {h}\n- （待生成）\n"
    return out.strip() + "\n"


def extract_section(doc: str, heading: str) -> str:
    pat = re.compile(rf"^## {re.escape(heading)}\s*$", re.M)
    m = pat.search(doc)
    if not m:
        return ""
    start = m.start()
    m2 = re.search(r"^##\s+", doc[m.end():], flags=re.M)
    end = (m.end() + m2.start()) if m2 else len(doc)
    return doc[start:end].strip() + "\n"


def char_count(text: str) -> int:
    return len(re.sub(r"\s+", "", text))


def within_target(n: int, target: int, tol: int) -> bool:
    low = int(target * (1 - tol / 100.0))
    high = int(target * (1 + tol / 100.0))
    return low <= n <= high


def section_body_len(doc: str, heading: str) -> int:
    sec = extract_section(doc, heading)
    if not sec:
        return 0
    lines = sec.splitlines()
    if lines and lines[0].startswith("## "):
        lines = lines[1:]
    body = "\n".join(lines).strip()
    body = body.replace("（待生成）", "").replace("待生成", "")
    return len(re.sub(r"\s+", "", body))


def check_short_sections(doc: str, per_section_min: int) -> List[Tuple[str, int]]:
    shorts = []
    for h in MASTER_HEADINGS:
        bl = section_body_len(doc, h)
        if bl < per_section_min:
            shorts.append((h, bl))
    return shorts


def reduce_merge(api: DeepSeekAPI, current_master: str, batch_text: str, target: int, tol: int, per_section_min: int) -> str:
    system = SYSTEM_MASTER.format(TARGET=target, TOL=tol)

    user = f"""你将把“新增批次笔记”融合进“当前总稿”。

【当前总稿】：
<<<MASTER
{current_master}
MASTER>>>

【新增批次笔记】：
<<<BATCH
{batch_text}
BATCH>>>

任务（硬性）：
- 去重、合并同义项、抽象为规则/清单/模板
- 必须保留以下10个二级标题（不可缺失）
- 每个小节正文（不含标题）至少 {per_section_min} 字
- 第6节：句式模板不少于 60 条（按情绪/场景分组）
- 第10节：速用卡片不少于 12 张（每张含：定位/元素包/镜头/情绪推进/可粘贴句式）
- 输出完整 Markdown 全文
- 总长度尽量靠近 {target} 字（±{tol}%）

二级标题列表：
{chr(10).join([f"- {h}" for h in MASTER_HEADINGS])}
"""
    merged = chat_until_complete(api, system, user, max_rounds=14)
    return normalize_md(merged)


def final_length_adjust(api: DeepSeekAPI, master: str, target: int, tol: int, per_section_min: int) -> str:
    master = normalize_md(master)
    total = char_count(master)
    shorts = check_short_sections(master, per_section_min)

    need_expand = (total < int(target * (1 - tol / 100.0))) or (len(shorts) > 0)
    need_compress = (total > int(target * (1 + tol / 100.0))) and (len(shorts) == 0)

    if not need_expand and not need_compress:
        return master

    if need_expand:
        direction = "扩写"
        short_str = "\n".join([f"- {h}（当前约{bl}字，需要≥{per_section_min}字）" for h, bl in shorts]) or "- 无（但总字数仍偏短）"
        instruction = f"""你必须把总稿扩写到更接近目标长度，并保证每个小节正文不少于 {per_section_min} 字。
硬性指标（必须满足）：
- 第6节「可复用句式库」：至少 60 条句式/模板（按情绪/场景分组），并补“使用条件/替换槽位”
- 第10节「速用卡片」：至少 12 张卡片；每张必须包含：
  1) 一句话定位（场景×情绪）
  2) 场景元素包（光影/色彩/材质/空间/气候/声场至少覆盖3项）
  3) 镜头与节奏（远中近/信息密度/切换）
  4) 情绪推进（起→承→转→收）
  5) 2-3句可直接粘贴的描写句式

扩写优先级（按顺序执行）：
1) 补第6节句式数量与“替换槽位”
2) 补第10节卡片数量与细节
3) 补第3/4/5节“清单+模板+示例句式”
4) 补第9节“自检问题列表（不少于20条）”
禁止：复述剧情、堆空话。
当前过短小节如下：
{short_str}
"""
    else:
        direction = "压缩"
        instruction = """整体偏长，请压缩重复与空话：保留最有用的规则/模板/清单，删除同义重复，
但不能删结构标题，也不能让任何小节变成空话；第6节与第10节硬性指标仍需保留。"""

    system = SYSTEM_MASTER.format(TARGET=target, TOL=tol)
    user = f"""下面是一份总学习笔记草稿：
<<<MASTER
{master}
MASTER>>>

任务：做一次“{direction}”以贴近长度目标。
- 目标长度：{target} 字（±{tol}%）
- 每个小节正文至少 {per_section_min} 字
- {instruction}
- 输出必须完整，且保留全部10个二级标题
- 输出完整 Markdown 全文
"""
    out = chat_until_complete(api, system, user, max_rounds=18)
    return normalize_md(out)


def read_batches(batches_dir: Path, take_n: int) -> List[Path]:
    files = sorted(batches_dir.glob("scene_notes_*.md"))
    if not files:
        raise FileNotFoundError(f"No scene_notes_*.md found under {batches_dir}")
    return files[:take_n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches_dir", required=True, help="Directory containing scene_notes_*.md")
    ap.add_argument("--out_file", required=True, help="Output master markdown file path")
    ap.add_argument("--take_n", type=int, default=4)

    ap.add_argument("--target_chars", type=int, default=8000)
    ap.add_argument("--tolerance_percent", type=int, default=8)
    ap.add_argument("--per_section_min", type=int, default=700, help="Min chars per section body (no whitespace)")

    ap.add_argument("--api_key", default=os.getenv("DEEPSEEK_API_KEY", ""))
    ap.add_argument("--base_url", default="https://api.deepseek.com")
    ap.add_argument("--model", default="deepseek-reasoner")
    ap.add_argument("--max_tokens", type=int, default=10000)
    ap.add_argument("--temperature", type=float, default=0.25)
    ap.add_argument("--timeout_sec", type=int, default=180)
    ap.add_argument("--max_retries", type=int, default=5)

    args = ap.parse_args()

    if not args.api_key:
        print("ERROR: --api_key is empty and DEEPSEEK_API_KEY env not set.", file=sys.stderr)
        sys.exit(1)

    cfg = DeepSeekConfig(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout_sec=args.timeout_sec,
    )
    api = DeepSeekAPI(cfg)

    batches_dir = Path(args.batches_dir)
    out_file = Path(args.out_file)

    files = read_batches(batches_dir, args.take_n)
    print("[INFO] Using batch files:")
    for f in files:
        print("  -", f.name)

    master = build_skeleton()
    print(f"[INFO] Init chars≈{char_count(master)}")

    # Progressive merge
    for i, f in enumerate(files, start=1):
        batch_text = f.read_text(encoding="utf-8", errors="ignore").strip()
        if not batch_text:
            continue
        print(f"[RUN] Merging {i}/{len(files)}: {f.name}")
        master = call_with_retries(
            lambda: reduce_merge(api, master, batch_text, args.target_chars, args.tolerance_percent, args.per_section_min),
            max_retries=args.max_retries,
        )
        print(f"[INFO] After merge chars≈{char_count(master)}")

    # Final calibration loop (up to 8)
    for round_i in range(1, 9):
        n = char_count(master)
        shorts = check_short_sections(master, args.per_section_min)
        ok = within_target(n, args.target_chars, args.tolerance_percent) and not shorts
        print(f"[INFO] Calib round {round_i}: chars≈{n}, shorts={len(shorts)}, ok={ok}")
        if ok:
            break
        master = call_with_retries(
            lambda: final_length_adjust(api, master, args.target_chars, args.tolerance_percent, args.per_section_min),
            max_retries=args.max_retries,
        )

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(master, encoding="utf-8")
    print(f"[OK] Wrote master note: {out_file}")
    print(f"[OK] Final chars≈{char_count(master)}")

    print("[INFO] Per-section body chars:")
    for h in MASTER_HEADINGS:
        print(f"  - {h}: {section_body_len(master, h)}")


if __name__ == "__main__":
    main()