#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
只生成：学习大纲 + 日志
输入模式（每 group_size 章一组，默认20）：
- 前 summary_k 章：从 summary_dir 读取 summary（优先区间文件，否则拼单章文件）
- 后 group_size-summary_k 章：从 split_dir 读取全文（按“第xxx章”标题切章）

输出：
- out_dir/logs/run_study.log
- out_dir/study_outline_{start}_{end}.txt
"""

from __future__ import annotations

import os
import re
import time
import argparse
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

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
# Read utils
# =========================

def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")

def chunk_text_by_paragraph(text: str, max_chars: int) -> List[str]:
    """按段落切块，避免一次输入过长"""
    text = text.strip()
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
# Summary scanning
# =========================

@dataclass
class SummaryIndex:
    # 单章：ch -> text
    single: Dict[int, str]
    # 区间：(a,b) -> text
    ranges: Dict[Tuple[int, int], str]

def build_summary_index(summary_dir: Path) -> SummaryIndex:
    single: Dict[int, str] = {}
    ranges: Dict[Tuple[int, int], str] = {}

    for p in sorted(summary_dir.glob("*.txt")):
        name = p.name
        nums = [int(x) for x in re.findall(r"\d+", name)]
        if not nums:
            continue

        txt = read_text(p).strip()
        if len(txt) < 10:
            continue

        # 判定：如果文件名里出现两个数字并且包含 '_' 或 '-'，更像区间
        if len(nums) >= 2 and (("_" in name) or ("-" in name)):
            a, b = nums[0], nums[1]
            if a > b:
                a, b = b, a
            # 限制一下范围，避免误把日期当章号
            if 1 <= a <= 10000 and 1 <= b <= 10000 and (b - a) <= 2000:
                ranges[(a, b)] = txt
        else:
            # 单章：取第一个数字
            ch = nums[0]
            if 1 <= ch <= 10000:
                single[ch] = txt

    return SummaryIndex(single=single, ranges=ranges)

def get_summary_block(idx: SummaryIndex, start_ch: int, end_ch: int) -> str:
    """优先找 exact 区间文件，否则拼单章"""
    if (start_ch, end_ch) in idx.ranges:
        return idx.ranges[(start_ch, end_ch)].strip()

    parts: List[str] = []
    for ch in range(start_ch, end_ch + 1):
        t = idx.single.get(ch)
        if t:
            parts.append(f"【第{ch}章 summary】\n{t.strip()}")
    return "\n\n".join(parts).strip()


# =========================
# Full text scanning (split)
# =========================

@dataclass
class Chapter:
    number: int
    title: str
    content: str
    source_file: str

CHAPTER_HEADER_RE = re.compile(r"^\s*(?P<prefix>.*?第\s*0*(?P<num>\d{1,4})\s*章\s*(?P<title>.*))\s*$")

def parse_chapters_from_text(text: str, source_file: str) -> List[Chapter]:
    lines = text.splitlines()
    hits: List[Tuple[int, int, str]] = []
    for i, line in enumerate(lines):
        m = CHAPTER_HEADER_RE.match(line)
        if m:
            num = int(m.group("num"))
            title = (m.group("title") or "").strip()
            if 1 <= num <= 5000:
                hits.append((i, num, title))

    if not hits:
        return []

    chs: List[Chapter] = []
    for idx, (start_i, num, title) in enumerate(hits):
        end_i = hits[idx + 1][0] if idx + 1 < len(hits) else len(lines)
        body = "\n".join(lines[start_i:end_i]).strip()
        if len(body) < 40:
            continue
        chs.append(Chapter(number=num, title=title or f"第{num}章", content=body, source_file=source_file))
    return chs

def scan_split_dir(split_dir: Path) -> Dict[int, Chapter]:
    chap_map: Dict[int, Chapter] = {}
    for p in sorted(split_dir.glob("*.txt")):
        txt = read_text(p)
        chs = parse_chapters_from_text(txt, source_file=str(p))
        for ch in chs:
            if ch.number not in chap_map:
                chap_map[ch.number] = ch
    return chap_map


# =========================
# LLM prompts
# =========================

LEARNING_SYSTEM = """你是“小说写作学习提炼器（递归版）”。
你只依据用户输入（summary 或 章节全文），提炼出“可学习的写作套路”，并更新到一个《Learning Notes》缓存中。

硬规则：
1) 只依据输入，不得编造原文细节；不确定写【未知】。
2) 输出必须是纯文本TXT，禁止JSON/YAML/Markdown表格。
3) 你的输出只允许是《Learning Notes》本体，不要任何解释。
4) 必须强制压缩：总长度尽量控制在 2500-4000 中文字符以内；宁可合并同类项，不要越写越长。
5) 从全文可以提炼：节奏、钩子、段落模板、对话套路、信息投放、意象、战斗/修炼模板等。
6) 从summary只能提炼：结构、冲突模板、升级节奏、人物功能、世界观讲解方式（禁止凭空推测文笔细节）。
"""

LEARNING_UPDATE_PROMPT = """你将做递归更新：我给你
A) 当前《Learning Notes》（可能为空）
B) 新输入块（可能是 summary 或 全文）

你只输出更新后的《Learning Notes》，并遵循这个结构（标题不能改）：

【Learning Notes】
1) 结构与节奏
2) 冲突模板
3) 人物功能与推进法
4) 世界观/规则投放方式
5) 爽点与回报机制
6) 对话与信息投放
7) 描写镜头/意象/段落骨架（仅全文可提炼）
8) 可复用模板碎片（仅全文可提炼）
9) 常见问题与避坑

要求：
- 每条都是“可操作要点”，不写长散文。
- 避免重复；同类合并。
- 末尾附一行：【覆盖范围】第X-第Y（或本次块标签）
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


def update_learning_notes(
    client: OpenAI,
    model: str,
    current_notes: str,
    block_label: str,
    block_type: str,
    block_text: str,
    temperature: float,
    max_tokens: int,
    log_path: Path,
    retries: int = 5,
) -> str:
    """
    优化输出，确保每个块的输出不超过token限制，减少细节量。
    """
    strict_system = LEARNING_SYSTEM + "\n\n额外硬规则：\n- 你输出的第一行必须是：【Learning Notes】\n- 只输出Learning Notes本体，不要前言/解释/推理过程。\n"

    user = f"""{LEARNING_UPDATE_PROMPT}

【A 当前Learning Notes】
{current_notes.strip() if current_notes.strip() else "（空）"}

【B 新输入块元信息】
- 类型：{block_type}
- 标签：{block_label}

【B 新输入块正文】
{block_text}
"""

    def _call(u: str) -> str:
        out = llm_chat_once(client, model, strict_system, u, temperature, max_tokens)
        return out.lstrip("\ufeff \n\r\t")

    def _repair(raw: str) -> str:
        repair_prompt = f"""你上一条输出没有按模板给出《Learning Notes》，现在请把“上一条输出”重排为严格模板。

硬性要求：
- 第一行必须是：【Learning Notes】
- 必须包含 1) 到 9) 九个小节（标题按模板原样）
- 内容可为空时写【未知】，不要编造
- 末尾必须包含一行：【覆盖范围】{block_label}
- 只输出重排后的 Learning Notes 本体，不要解释

【模板标题（必须照抄）】
【Learning Notes】
1) 结构与节奏
2) 冲突模板
3) 人物功能与推进法
4) 世界观/规则投放方式
5) 爽点与回报机制
6) 对话与信息投放
7) 描写镜头/意象/段落骨架（仅全文可提炼）
8) 可复用模板碎片（仅全文可提炼）
9) 常见问题与避坑

【上一条输出如下】
{raw}
"""
        return _call(repair_prompt)

    last_err: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        try:
            out = _call(user)

            if "【Learning Notes】" in out:
                return out.strip()

            fixed = _repair(out)
            if "【Learning Notes】" in fixed:
                log_line(log_path, f"格式修复成功 label={block_label} attempt={attempt}/{retries}")
                return fixed.strip()

            raise RuntimeError("Learning Notes 输出缺少标题，且修复失败")

        except Exception as e:
            last_err = e
            log_line(log_path, f"update_learning_notes失败 attempt={attempt}/{retries} label={block_label} err={repr(e)}")
            time.sleep(min(2 ** attempt, 20))

    log_line(log_path, f"WARNING: 跳过该块 label={block_label} reason={repr(last_err)}")
    return current_notes




def make_study_outline_sectioned(
    client: OpenAI,
    model: str,
    learning_notes: str,
    temperature: float = 0.2,
    max_tokens_per_section: int = 1600,
) -> str:
    """分9节生成，强制 END 标记，不完整自动续写"""
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
        ("9) 训练计划（7天或14天）",
         "每天都有：输入->输出->复盘（检查表）。"),
    ]

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
# Main
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_dir", type=str, required=True, help="summary文件夹（前10章摘要来源）")
    ap.add_argument("--split_dir", type=str, required=True, help="split文件夹（后10章全文来源）")
    ap.add_argument("--out_dir", type=str, required=True, help="输出目录（只输出学习大纲+日志）")
    ap.add_argument("--start_chapter", type=int, required=True)
    ap.add_argument("--end_chapter", type=int, required=True)

    ap.add_argument("--group_size", type=int, default=20)
    ap.add_argument("--summary_k", type=int, default=10)
    ap.add_argument("--chapter_chunk_chars", type=int, default=12000)

    ap.add_argument("--base_url", type=str, default="https://api.deepseek.com")
    ap.add_argument("--model", type=str, default="deepseek-reasoner")
    ap.add_argument("--api_key_env", type=str, default="DEEPSEEK_API_KEY")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max_tokens_update", type=int, default=2500, help="更新learning_notes的输出上限")
    ap.add_argument("--max_tokens_section", type=int, default=1600, help="学习大纲每小节输出上限")

    args = ap.parse_args()

    summary_dir = Path(args.summary_dir).expanduser()
    split_dir = Path(args.split_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    ensure_dir(out_dir)

    log_path = out_dir / "logs" / "run_study.log"
    ensure_dir(log_path.parent)

    api_key = os.getenv(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"未检测到环境变量 {args.api_key_env}，请先 export {args.api_key_env}='你的key'")

    client = OpenAI(api_key=api_key, base_url=args.base_url)

    start_ch = args.start_chapter
    end_ch = args.end_chapter
    group_size = args.group_size
    summary_k = args.summary_k

    if summary_k <= 0 or summary_k >= group_size:
        raise ValueError("--summary_k 必须在 (0, group_size) 之间（例如 group_size=20, summary_k=10）")

    log_line(log_path, f"启动 start={start_ch} end={end_ch} group={group_size} summary_k={summary_k}")
    log_line(log_path, f"summary_dir={summary_dir}")
    log_line(log_path, f"split_dir={split_dir}")
    log_line(log_path, f"out_dir={out_dir}")
    log_line(log_path, f"model={args.model} base_url={args.base_url}")

    # 建索引
    sum_idx = build_summary_index(summary_dir)
    log_line(log_path, f"summary 单章数={len(sum_idx.single)} 区间数={len(sum_idx.ranges)}")

    chap_map = scan_split_dir(split_dir)
    log_line(log_path, f"split 切出章节数={len(chap_map)}")

    learning_notes = ""  # 递归缓存

    # 逐组处理
    g = start_ch
    while g <= end_ch:
        g_end = min(g + group_size - 1, end_ch)

        s_start = g
        s_end = min(g + summary_k - 1, g_end)

        t_start = s_end + 1
        t_end = g_end

        # 1) 读 summary（前10章）
        if s_start <= s_end:
            s_text = get_summary_block(sum_idx, s_start, s_end)
            label = f"第{s_start}-{s_end}章 summary"
            if not s_text:
                log_line(log_path, f"WARNING: {label} 未找到匹配summary（将跳过该段）")
            else:
                log_line(log_path, f"更新 learning_notes <- {label} chars={len(s_text)}")
                learning_notes = update_learning_notes(
                    client=client,
                    model=args.model,
                    current_notes=learning_notes,
                    block_label=label,
                    block_type="SUMMARY",
                    block_text=s_text,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens_update,
                    log_path=log_path,
                )

        # 2) 读全文（后10章）
        if t_start <= t_end:
            for ch_no in range(t_start, t_end + 1):
                ch = chap_map.get(ch_no)
                if not ch:
                    log_line(log_path, f"WARNING: 未找到第{ch_no}章全文（split无匹配标题），跳过")
                    continue

                parts = chunk_text_by_paragraph(ch.content, max_chars=args.chapter_chunk_chars)
                for pi, part in enumerate(parts, start=1):
                    label = f"第{ch_no}章 {ch.title}".strip()
                    if len(parts) > 1:
                        label += f" part {pi}/{len(parts)}"
                    log_line(log_path, f"更新 learning_notes <- {label} chars={len(part)} file={ch.source_file}")
                    learning_notes = update_learning_notes(
                        client=client,
                        model=args.model,
                        current_notes=learning_notes,
                        block_label=label,
                        block_type="FULL_TEXT",
                        block_text=part,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens_update,
                        log_path=log_path,
                    )

        g += group_size

    # 3) 分节生成学习大纲（防截断）
    log_line(log_path, "开始生成分节学习大纲（sectioned）")
    study = make_study_outline_sectioned(
        client=client,
        model=args.model,
        learning_notes=learning_notes,
        temperature=0.2,
        max_tokens_per_section=args.max_tokens_section,
    )

    out_path = out_dir / f"study_outline_{start_ch}_{end_ch}.txt"
    out_path.write_text(study, encoding="utf-8")
    log_line(log_path, f"完成：写入学习大纲 {out_path}")

    print("✅ DONE")
    print(f"- 学习大纲：{out_path}")
    print(f"- 日志：{log_path}")


if __name__ == "__main__":
    main()
