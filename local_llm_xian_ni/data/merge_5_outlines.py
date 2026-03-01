#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
把现有的 study_outline_{start}_{end}.txt 按每 batch_size(默认5) 个合并成更大范围的学习笔记。

输入：in_dir 下的 study_outline_*.txt
输出：out_dir/study_outline_{batchStart}_{batchEnd}.txt
日志：out_dir/logs/merge_outlines.log

策略（稳 token）：
1) 先把每个 outline 递归整合到统一《Learning Notes》（9节，强制压缩、去重）
2) 再从《Learning Notes》分节生成最终“学习大纲/学习笔记”（默认1-8节，不含训练计划）
"""

from __future__ import annotations

import os
import re
import time
import argparse
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Optional

try:
    from openai import OpenAI
except Exception as e:
    raise RuntimeError("缺少 openai SDK。请先安装：pip install -U openai") from e


# =========================
# Logging
# =========================

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log_line(log_path: Path, msg: str) -> None:
    ensure_dir(log_path.parent)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{now_str()}] {msg}\n")


# =========================
# Text utils
# =========================

def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")

def chunk_text_by_paragraph(text: str, max_chars: int) -> List[str]:
    """按段落切块，避免一次输入过长。"""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    paras = re.split(r"\n\s*\n", text)
    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0
    for para in paras:
        para = para.strip()
        if not para:
            continue
        add_len = len(para) + 2
        if cur and cur_len + add_len > max_chars:
            chunks.append("\n\n".join(cur).strip())
            cur = [para]
            cur_len = len(para)
        else:
            cur.append(para)
            cur_len += add_len
    if cur:
        chunks.append("\n\n".join(cur).strip())
    return chunks


# =========================
# Learning Notes normalization (容错：保证永远能回到标准模板)
# =========================

FIXED_HEADINGS = {
    1: "1) 结构与节奏",
    2: "2) 冲突模板",
    3: "3) 人物功能与推进法",
    4: "4) 世界观/规则投放方式",
    5: "5) 爽点与回报机制",
    6: "6) 对话与信息投放",
    7: "7) 描写镜头/意象/段落骨架（仅全文可提炼）",
    8: "8) 可复用模板碎片（仅全文可提炼）",
    9: "9) 常见问题与避坑",
}

SECTION_ANY_RE = re.compile(r"(?m)^\s*([1-9])\)\s*(.*)$")

def parse_notes_sections(text: str) -> Dict[int, str]:
    t = (text or "").strip()
    matches = list(SECTION_ANY_RE.finditer(t))
    if not matches:
        return {}
    spans: List[Tuple[int, int, int]] = []
    for i, m in enumerate(matches):
        sec = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(t)
        spans.append((sec, start, end))
    out: Dict[int, str] = {}
    for sec, s, e in spans:
        out[sec] = t[s:e].strip()
    return out

def normalize_learning_notes(raw: str, coverage_line: str) -> str:
    raw = (raw or "").strip()
    secs = parse_notes_sections(raw)
    if not secs:
        secs = {9: raw if raw else "【未知】"}
    parts: List[str] = ["【Learning Notes】"]
    for i in range(1, 10):
        parts.append(FIXED_HEADINGS[i])
        c = (secs.get(i) or "").strip()
        parts.append(c if c else "【未知】")
    parts.append(coverage_line.strip())
    return "\n".join(parts).strip()

def validate_learning_notes_format(text: str) -> bool:
    t = (text or "").strip()
    if not t.startswith("【Learning Notes】"):
        return False
    for i in range(1, 10):
        if not re.search(rf"(?m)^\s*{i}\)\s*", t):
            return False
    if not re.search(r"(?m)^\s*【覆盖范围】", t):
        return False
    return True


# =========================
# LLM prompts
# =========================

MERGE_SYSTEM = """你是“学习笔记整合器（递归更新版）”。

目标：把输入的“学习大纲/学习笔记文本（study_outline）”提炼为统一的《Learning Notes》缓存，持续整合去重。

硬规则：
1) 只依据输入文本，不得编造未出现的信息；不确定写【未知】。
2) 输出必须是纯文本TXT，禁止JSON/YAML/Markdown表格。
3) 只允许输出《Learning Notes》本体（包含1)-9)九节），不要解释。
4) 必须强制压缩：总长度尽量控制在 2500-4000 中文字符以内；宁可合并同类项，不要越写越长。
5) 每条必须“可操作”，用要点短句；去重合并；把分散内容归并到正确小节。
"""

MERGE_TEMPLATE = """你将做递归整合：我给你
A) 当前《Learning Notes》（可能为空）
B) 新输入文档（它是学习大纲/学习笔记/总结）

你只输出更新后的《Learning Notes》，并严格保持以下模板结构（必须保留 1)-9) 编号行）：

【Learning Notes】
1) 结构与节奏
2) 冲突模板
3) 人物功能与推进法
4) 世界观/规则投放方式
5) 爽点与回报机制
6) 对话与信息投放
7) 描写镜头/意象/段落骨架（若输入没涉及就写【未知】）
8) 可复用模板碎片（若输入没给模板就写【未知】）
9) 常见问题与避坑

额外要求：
- 合并去重，保持“可操作要点”，避免长散文。
- 末尾必须以单独一行结束：{coverage_line}
"""

STUDY_SYSTEM = """你是“小说写作学习大纲整理器”。
只依据输入的《Learning Notes》，不编造原文细节。
输出纯文本TXT，不要JSON/YAML/表格。
"""

def llm_chat_once(client: OpenAI, model: str, system: str, user: str,
                  temperature: float, max_tokens: int) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


def update_merge_notes(
    client: OpenAI,
    model: str,
    current_notes: str,
    doc_label: str,
    doc_text: str,
    coverage_line: str,
    temperature: float,
    max_tokens: int,
    log_path: Path,
    retries: int = 4,
) -> str:
    user_prompt = MERGE_TEMPLATE.format(coverage_line=coverage_line) + f"""

【A 当前Learning Notes】
{current_notes.strip() if (current_notes or "").strip() else "（空）"}

【B 新输入文档标签】
{doc_label}

【B 新输入文档正文】
{doc_text}
"""
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            raw = llm_chat_once(client, model, MERGE_SYSTEM, user_prompt, temperature, max_tokens)
            norm = normalize_learning_notes(raw, coverage_line)
            if validate_learning_notes_format(norm):
                return norm
            raise RuntimeError("normalize 后格式仍不合规")
        except Exception as e:
            last_err = e
            log_line(log_path, f"update_merge_notes失败 attempt={attempt}/{retries} label={doc_label} err={repr(e)}")
            time.sleep(min(2 ** attempt, 10))
    log_line(log_path, f"WARNING: 跳过该文档 label={doc_label} reason={repr(last_err)}")
    return current_notes


def make_study_outline_sectioned(
    client: OpenAI,
    model: str,
    learning_notes: str,
    temperature: float,
    max_tokens_per_section: int,
    include_training_plan: bool,
) -> str:
    header = "【学习大纲｜Writing Study Outline】\n"
    sections = [
        ("1) 结构与节奏（分阶段/章末钩子/升级频率）",
         "给出阶段拆分方法、升级间隔规律、章末钩子类型与触发点。"),
        ("2) 冲突设计（目标-阻力-转折-代价 的常用套路）",
         "总结常见冲突模板，至少3种变体，标注适用场景。"),
        ("3) 人物塑造（主角成长曲线/配角功能/反派推进法）",
         "用‘成长阶段/关键选择/代价回收’描述主角；配角按功能分类；反派按升级推动分类。"),
        ("4) 世界观与规则讲解方式（怎么“边写边讲”不无聊）",
         "总结规则投放渠道（对话/任务/惩罚/展示）并给可照抄步骤。"),
        ("5) 爽点与回报机制（打脸/奖励/资源/身份跃迁的触发点）",
         "按‘触发条件->爆发方式->回报形式’给至少5条。"),
        ("6) 对话与信息投放（称呼体系/暗示/误导/回收）",
         "总结对话冲突套路、信息埋伏笔与回收方式。"),
        ("7) 描写与意象（高频意象、段落模板、常见镜头）",
         "列镜头/意象清单，并给2-3个段落骨架。"),
        ("8) 可复用模板库",
         "必须输出：开章模板(3)、章末钩子(5)、修炼/突破(3)、战斗/对决(3)。"),
    ]
    if include_training_plan:
        sections.append(("9) 训练计划（7天或14天）", "每天都有：输入->输出->复盘（检查表）。"))

    out_parts: List[str] = [header]

    for title, req in sections:
        base_prompt = f"""只输出这一节：{title}
{req}

硬性要求：
- 内容必须可操作，避免空话。
- 必须以“===END===”单独一行结尾。

《Learning Notes》如下：
{learning_notes}
"""
        text = llm_chat_once(
            client=client,
            model=model,
            system=STUDY_SYSTEM,
            user=base_prompt,
            temperature=temperature,
            max_tokens=max_tokens_per_section,
        )

        tries = 0
        while "===END===" not in text and tries < 2:
            tries += 1
            tail = text[-800:] if len(text) > 800 else text
            cont = f"""继续把这一节写完：{title}
你上一次输出可能被截断。请从末尾自然续写，少重复。
仍要求：最后必须以“===END===”单独一行结尾。

上一次末尾参考：
{tail}

《Learning Notes》仍是：
{learning_notes}
"""
            more = llm_chat_once(
                client=client,
                model=model,
                system=STUDY_SYSTEM,
                user=cont,
                temperature=temperature,
                max_tokens=max_tokens_per_section,
            )
            text = (text.rstrip() + "\n" + more.lstrip()).strip()

        text = text.replace("===END===", "").strip()
        out_parts.append(text + "\n")

    return "\n".join(out_parts).strip()


# =========================
# File scanning
# =========================

RANGE_RE = re.compile(r"study_outline_(\d+)_(\d+)\.txt$")

@dataclass
class OutlineFile:
    path: Path
    start: int
    end: int

def scan_outline_files(in_dir: Path) -> List[OutlineFile]:
    files: List[OutlineFile] = []
    for p in sorted(in_dir.glob("study_outline_*.txt")):
        m = RANGE_RE.search(p.name)
        if not m:
            continue
        a, b = int(m.group(1)), int(m.group(2))
        if a > b:
            a, b = b, a
        files.append(OutlineFile(path=p, start=a, end=b))
    files.sort(key=lambda x: (x.start, x.end))
    return files

def batch_list(items: List[OutlineFile], batch_size: int) -> List[List[OutlineFile]]:
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


# =========================
# Main
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, required=True, help="已有 study_outline_*.txt 所在目录")
    ap.add_argument("--out_dir", type=str, required=True, help="输出目录（建议 outline/merged）")
    ap.add_argument("--batch_size", type=int, default=5, help="每多少个文件合成1个（默认5）")

    ap.add_argument("--chunk_chars", type=int, default=9000, help="单次喂给模型的最大字符数（按段落切块）")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max_tokens_update", type=int, default=2000, help="整合Learning Notes输出上限")
    ap.add_argument("--max_tokens_section", type=int, default=1400, help="最终学习大纲每节输出上限")

    ap.add_argument("--include_training_plan", action="store_true", help="输出训练计划（默认不输出）")

    ap.add_argument("--base_url", type=str, default="https://api.deepseek.com")
    ap.add_argument("--model", type=str, default="deepseek-reasoner")
    ap.add_argument("--api_key_env", type=str, default="DEEPSEEK_API_KEY")

    args = ap.parse_args()

    in_dir = Path(args.in_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    ensure_dir(out_dir)
    log_path = out_dir / "logs" / "merge_outlines.log"
    ensure_dir(log_path.parent)

    api_key = os.getenv(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"未检测到环境变量 {args.api_key_env}，请先 export {args.api_key_env}='你的key'")

    client = OpenAI(api_key=api_key, base_url=args.base_url)

    files = scan_outline_files(in_dir)
    if not files:
        raise RuntimeError(f"在 {in_dir} 没找到 study_outline_*.txt（文件名需类似 study_outline_1_50.txt）")

    log_line(log_path, f"扫描到文件数={len(files)} batch_size={args.batch_size}")
    log_line(log_path, f"model={args.model} base_url={args.base_url}")

    batches = batch_list(files, args.batch_size)
    for bi, batch in enumerate(batches, start=1):
        batch_start = min(x.start for x in batch)
        batch_end = max(x.end for x in batch)

        out_file = out_dir / f"study_outline_{batch_start}_{batch_end}.txt"
        if out_file.exists():
            log_line(log_path, f"SKIP: 已存在 {out_file.name}")
            continue

        log_line(log_path, f"=== Batch {bi}/{len(batches)}: {batch_start}-{batch_end} files={len(batch)} ===")

        # 1) 递归整合 -> Learning Notes
        merged_notes = ""
        coverage_line = f"【覆盖范围】第{batch_start}-第{batch_end}"

        for f in batch:
            txt = read_text(f.path)
            parts = chunk_text_by_paragraph(txt, max_chars=int(args.chunk_chars))
            if not parts:
                log_line(log_path, f"WARNING: 空文件 {f.path.name} 跳过")
                continue

            for pi, part in enumerate(parts, start=1):
                label = f"{f.path.name}"
                if len(parts) > 1:
                    label += f" part {pi}/{len(parts)}"
                log_line(log_path, f"合并 <- {label} chars={len(part)}")

                merged_notes = update_merge_notes(
                    client=client,
                    model=args.model,
                    current_notes=merged_notes,
                    doc_label=label,
                    doc_text=part,
                    coverage_line=coverage_line,
                    temperature=args.temperature,
                    max_tokens=int(args.max_tokens_update),
                    log_path=log_path,
                )

        # 2) 从 Learning Notes 生成更大范围学习大纲（你熟悉的 study_outline 格式）
        log_line(log_path, f"生成学习大纲 <- {batch_start}-{batch_end}")
        merged_outline = make_study_outline_sectioned(
            client=client,
            model=args.model,
            learning_notes=merged_notes,
            temperature=0.2,
            max_tokens_per_section=int(args.max_tokens_section),
            include_training_plan=bool(args.include_training_plan),
        )

        out_file.write_text(merged_outline, encoding="utf-8")
        log_line(log_path, f"写出：{out_file}")

    print("✅ DONE")
    print(f"- 输出目录：{out_dir}")
    print(f"- 日志：{log_path}")


if __name__ == "__main__":
    main()
