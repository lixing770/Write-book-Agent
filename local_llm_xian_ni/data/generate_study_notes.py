#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
只生成：学习大纲 + 日志

输入模式（每 group_size 章一组，默认20）：
- 前 summary_k 章：从 summary_dir 读取 summary（优先exact区间，否则找“覆盖区间”，再不行拼单章）
- 后 group_size-summary_k 章：从 split_dir 读取全文（按“第xxx章”标题切章）

输出：
- out_dir/logs/run_study.log
- out_dir/study_outline_{start}_{end}.txt

增强保证：
1) 不同章节（summary / 全文 / 分块 part）更新《Learning Notes》的提示词逻辑完全一致：
   所有块统一走 update_learning_notes()。
2) summary 文件支持“覆盖区间”匹配：请求 1-5 可自动用 1-10_summary.txt。
3) Learning Notes 输出做本地归一化：DeepSeek 偶尔标题变形/缺行也不会报错中断。
4) SUMMARY 块强制不污染第7/8节（全文专属），会保留旧内容。
5) 学习大纲默认不含训练计划（第9节），需要时用 --include_training_plan。
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
    """按段落切块，避免一次输入过长；尽量不打断段落。"""
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
# Summary scanning
# =========================

@dataclass
class SummaryIndex:
    single: Dict[int, str]                    # 单章：ch -> text
    ranges: Dict[Tuple[int, int], str]        # 区间：(a,b) -> text

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

        # 文件名两个数字 + '_' 或 '-' => 区间
        if len(nums) >= 2 and (("_" in name) or ("-" in name)):
            a, b = nums[0], nums[1]
            if a > b:
                a, b = b, a
            if 1 <= a <= 10000 and 1 <= b <= 10000 and (b - a) <= 5000:
                ranges[(a, b)] = txt
        else:
            ch = nums[0]
            if 1 <= ch <= 10000:
                single[ch] = txt

    return SummaryIndex(single=single, ranges=ranges)

def _best_covering_range(ranges: Dict[Tuple[int, int], str], start_ch: int, end_ch: int) -> Optional[Tuple[int, int]]:
    """找能覆盖[start,end]的最短区间；如无则None。"""
    candidates = [(a, b) for (a, b) in ranges.keys() if a <= start_ch and b >= end_ch]
    if not candidates:
        return None
    # 最短优先，其次起点更小优先
    candidates.sort(key=lambda x: (x[1] - x[0], x[0], x[1]))
    return candidates[0]

def _best_overlap_range(ranges: Dict[Tuple[int, int], str], start_ch: int, end_ch: int) -> Optional[Tuple[int, int]]:
    """找与[start,end]重叠最多的区间；如无则None。"""
    best = None
    best_ov = 0
    for (a, b) in ranges.keys():
        ov = max(0, min(b, end_ch) - max(a, start_ch) + 1)
        if ov > best_ov:
            best_ov = ov
            best = (a, b)
    return best if best_ov > 0 else None

def get_summary_block(idx: SummaryIndex, start_ch: int, end_ch: int) -> Tuple[str, Optional[Tuple[int, int]]]:
    """
    取summary：
    1) exact 区间
    2) 覆盖区间（例如请求1-5，用1-10）
    3) 最大重叠区间（兜底）
    4) 拼单章
    返回：(text, used_range)；used_range=None 表示拼单章。
    """
    if (start_ch, end_ch) in idx.ranges:
        return idx.ranges[(start_ch, end_ch)].strip(), (start_ch, end_ch)

    cov = _best_covering_range(idx.ranges, start_ch, end_ch)
    if cov:
        return idx.ranges[cov].strip(), cov

    ov = _best_overlap_range(idx.ranges, start_ch, end_ch)
    if ov:
        return idx.ranges[ov].strip(), ov

    parts: List[str] = []
    for ch in range(start_ch, end_ch + 1):
        t = idx.single.get(ch)
        if t:
            parts.append(f"【第{ch}章 summary】\n{t.strip()}")
    return "\n\n".join(parts).strip(), None


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
    lines = (text or "").splitlines()
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
# Learning Notes: 本地归一化（解决“格式不合规”）
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
    """
    容错解析：只认 1)~9) 开头，不要求标题完全一致。
    返回 section -> content（不含标题行）。
    """
    t = (text or "").strip()
    matches = list(SECTION_ANY_RE.finditer(t))
    if not matches:
        return {}

    # 记录每节起止
    spans: List[Tuple[int, int, int]] = []  # (sec, start_pos, end_pos)
    for i, m in enumerate(matches):
        sec = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(t)
        spans.append((sec, start, end))

    out: Dict[int, str] = {}
    for sec, s, e in spans:
        content = t[s:e].strip()
        out[sec] = content
    return out

def normalize_learning_notes(raw: str, coverage_line: str) -> str:
    """
    把模型输出（可能标题变形/缺节/乱序）强制整理成标准模板。
    """
    raw = (raw or "").strip()
    secs = parse_notes_sections(raw)

    # 如果完全解析不到 1)~9)，就把全部塞到9)兜底，其余未知
    if not secs:
        secs = {9: raw.strip() if raw.strip() else "【未知】"}

    parts: List[str] = ["【Learning Notes】"]
    for i in range(1, 10):
        parts.append(FIXED_HEADINGS[i])
        c = (secs.get(i) or "").strip()
        if not c:
            c = "【未知】"
        parts.append(c)
    parts.append(coverage_line.strip())
    return "\n".join(parts).strip()

def is_unknown(s: str) -> bool:
    s = (s or "").strip()
    return (not s) or ("【未知】" in s and len(s) <= 6)

def enforce_summary_no_78(current_notes: str, new_notes: str) -> str:
    """
    SUMMARY 块强制不污染 7/8：
    - 若 current 里 7/8 有“非未知”内容，保留 current 的
    - 否则接受 new 的（但通常 new 会是未知）
    """
    cur = parse_notes_sections(current_notes)
    new = parse_notes_sections(new_notes)

    for sec in (7, 8):
        cur_c = (cur.get(sec) or "").strip()
        new_c = (new.get(sec) or "").strip()
        if not is_unknown(cur_c):
            new[sec] = cur_c
        else:
            # current 没货，new 就算写了也可能是编的，但我们不强行抹掉
            # 如果你想更严格：直接 new[sec] = "【未知】"
            pass

    # 重渲染（保留 new 的覆盖范围行）
    cov_line = "【覆盖范围】"
    m = re.search(r"(?m)^\s*【覆盖范围】.*$", new_notes)
    coverage_line = m.group(0).strip() if m else "【覆盖范围】未知"
    parts: List[str] = ["【Learning Notes】"]
    for i in range(1, 10):
        parts.append(FIXED_HEADINGS[i])
        c = (new.get(i) or "").strip()
        if not c:
            c = "【未知】"
        parts.append(c)
    parts.append(coverage_line)
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
# LLM prompts（逻辑清晰 + 全块一致）
# =========================

LEARNING_SYSTEM = """你是“小说写作学习提炼器（递归更新版）”。

目标：把输入块（SUMMARY 或 FULL_TEXT）里的“可学习写作套路”，合并进《Learning Notes》缓存，输出更新后的《Learning Notes》。

硬规则（必须遵守）：
1) 只依据输入，不得编造原文细节；不确定写【未知】。
2) 输出必须是纯文本TXT，禁止JSON/YAML/Markdown表格。
3) 只允许输出《Learning Notes》本体：必须包含模板标题与1)-9)九节，不能多任何解释。
4) 必须强制压缩：总长度尽量控制在 2500-4000 中文字符以内；宁可合并同类项，不要越写越长。
5) 每条必须“可操作”，用要点短句，不写长散文。
6) 去重合并：同类合并、避免重复；新信息补充到对应节，旧信息可被“归纳升级”。
"""

def build_allowed_scope(block_type: str) -> str:
    bt = (block_type or "").strip().upper()
    if bt == "FULL_TEXT":
        return "1)-9)均可提炼（含第7/8节）"
    return "只允许提炼1)-6)、9)；第7/8节不得新增细节"

LEARNING_UPDATE_TEMPLATE = """你将做递归更新：我给你
A) 当前《Learning Notes》（可能为空）
B) 新输入块（SUMMARY 或 FULL_TEXT）

你只输出更新后的《Learning Notes》，并严格遵循下面模板（请务必保留1)-9)编号行）：

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

额外要求（本次块强制约束）：
- 本次输入块类型：{block_type}
- 允许提炼范围：{allowed_scope}
- 若本次块不允许提炼第7/8节：第7/8节写【未知】或保留旧内容，不得凭空新增“文笔/意象/段落模板”细节。
- 末尾必须以单独一行结束：{coverage_line}
"""

STUDY_SYSTEM = """你是“小说写作学习大纲整理器”。
只依据输入的《Learning Notes》，不编造原文细节。
输出纯文本TXT，不要JSON/YAML/表格。
"""


# =========================
# LLM call helpers
# =========================

def llm_chat_once(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int
) -> str:
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
    coverage_line: str,
    retries: int = 4,
) -> str:
    """
    ✅ 统一入口：所有章节/summary/part 都走这里，提示词逻辑一致。
    并做本地归一化，确保永不因“格式不合规”中断。
    """
    strict_system = LEARNING_SYSTEM + "\n\n额外硬规则：\n- 输出第一行必须是：【Learning Notes】\n- 只输出Learning Notes本体，不要前言/解释/推理。\n"
    allowed_scope = build_allowed_scope(block_type)

    user_prompt = LEARNING_UPDATE_TEMPLATE.format(
        block_type=block_type,
        allowed_scope=allowed_scope,
        coverage_line=coverage_line
    ) + f"""

【A 当前Learning Notes】
{current_notes.strip() if (current_notes or "").strip() else "（空）"}

【B 新输入块元信息】
- 标签：{block_label}

【B 新输入块正文】
{block_text}
"""

    last_err: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        try:
            raw = llm_chat_once(client, model, strict_system, user_prompt, temperature, max_tokens)
            norm = normalize_learning_notes(raw, coverage_line)

            # SUMMARY 强制不污染 7/8
            if (block_type or "").strip().upper() != "FULL_TEXT":
                norm = enforce_summary_no_78(current_notes, norm)

            if validate_learning_notes_format(norm):
                return norm

            # 极少数情况下 normalize 后仍异常（比如 coverage_line 空），兜底
            raise RuntimeError("normalize 后仍不合规")

        except Exception as e:
            last_err = e
            log_line(log_path, f"update_learning_notes失败 attempt={attempt}/{retries} label={block_label} err={repr(e)}")
            time.sleep(min(2 ** attempt, 10))

    log_line(log_path, f"WARNING: 跳过该块 label={block_label} reason={repr(last_err)}")
    return current_notes


# =========================
# Study outline generation (默认不含训练计划)
# =========================

def make_study_outline_sectioned(
    client: OpenAI,
    model: str,
    learning_notes: str,
    temperature: float = 0.2,
    max_tokens_per_section: int = 1400,
    include_training_plan: bool = False,
) -> str:
    """分节生成学习大纲，强制 END 标记，不完整自动续写。默认不输出训练计划。"""
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
# Coverage helper
# =========================

def update_covered_range(covered_min: int, covered_max: int, a: int, b: int) -> Tuple[int, int]:
    mn = min(covered_min, a) if covered_min > 0 else a
    mx = max(covered_max, b)
    return mn, mx


# =========================
# Main
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_dir", type=str, required=True)
    ap.add_argument("--split_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--start_chapter", type=int, required=True)
    ap.add_argument("--end_chapter", type=int, required=True)

    ap.add_argument("--group_size", type=int, default=20)
    ap.add_argument("--summary_k", type=int, default=10)

    ap.add_argument("--chapter_chunk_chars", type=int, default=8000)
    ap.add_argument("--summary_chunk_chars", type=int, default=8000)

    ap.add_argument("--base_url", type=str, default="https://api.deepseek.com")
    ap.add_argument("--model", type=str, default="deepseek-reasoner")
    ap.add_argument("--api_key_env", type=str, default="DEEPSEEK_API_KEY")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max_tokens_update", type=int, default=2000)
    ap.add_argument("--max_tokens_section", type=int, default=1400)

    ap.add_argument("--include_training_plan", action="store_true")

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

    start_ch = int(args.start_chapter)
    end_ch = int(args.end_chapter)
    group_size = int(args.group_size)
    summary_k = int(args.summary_k)

    if start_ch <= 0 or end_ch <= 0 or start_ch > end_ch:
        raise ValueError("start/end chapter 不合法")
    if summary_k <= 0 or summary_k >= group_size:
        raise ValueError("--summary_k 必须在 (0, group_size) 之间（推荐 group_size=20 summary_k=10）")

    log_line(log_path, f"启动 start={start_ch} end={end_ch} group={group_size} summary_k={summary_k}")
    log_line(log_path, f"summary_dir={summary_dir}")
    log_line(log_path, f"split_dir={split_dir}")
    log_line(log_path, f"out_dir={out_dir}")
    log_line(log_path, f"model={args.model} base_url={args.base_url}")
    log_line(log_path, f"include_training_plan={args.include_training_plan}")

    sum_idx = build_summary_index(summary_dir)
    log_line(log_path, f"summary 单章数={len(sum_idx.single)} 区间数={len(sum_idx.ranges)}")

    chap_map = scan_split_dir(split_dir)
    log_line(log_path, f"split 切出章节数={len(chap_map)}")

    learning_notes = ""
    covered_min, covered_max = start_ch, start_ch - 1

    g = start_ch
    while g <= end_ch:
        g_end = min(g + group_size - 1, end_ch)

        s_start = g
        s_end = min(g + summary_k - 1, g_end)

        t_start = s_end + 1
        t_end = g_end

        # 1) summary块
        if s_start <= s_end:
            s_text, used_rng = get_summary_block(sum_idx, s_start, s_end)
            used_label = f"{used_rng[0]}-{used_rng[1]}" if used_rng else f"{s_start}-{s_end}"
            label_base = f"第{s_start}-{s_end}章 summary(used {used_label})"

            if not s_text:
                log_line(log_path, f"WARNING: {label_base} 未找到匹配summary（将跳过该段）")
            else:
                s_parts = chunk_text_by_paragraph(s_text, max_chars=int(args.summary_chunk_chars))
                for pi, part in enumerate(s_parts, start=1):
                    label = label_base + (f" part {pi}/{len(s_parts)}" if len(s_parts) > 1 else "")
                    covered_min, covered_max = update_covered_range(covered_min, covered_max, s_start, s_end)
                    coverage_line = f"【覆盖范围】第{covered_min}-第{covered_max}"
                    log_line(log_path, f"更新 learning_notes <- {label} chars={len(part)}")

                    learning_notes = update_learning_notes(
                        client=client,
                        model=args.model,
                        current_notes=learning_notes,
                        block_label=label,
                        block_type="SUMMARY",
                        block_text=part,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens_update,
                        log_path=log_path,
                        coverage_line=coverage_line,
                    )

        # 2) 全文块
        if t_start <= t_end:
            for ch_no in range(t_start, t_end + 1):
                ch = chap_map.get(ch_no)
                if not ch:
                    log_line(log_path, f"WARNING: 未找到第{ch_no}章全文（split无匹配标题），跳过")
                    continue

                parts = chunk_text_by_paragraph(ch.content, max_chars=int(args.chapter_chunk_chars))
                for pi, part in enumerate(parts, start=1):
                    label = f"第{ch_no}章 {ch.title}".strip()
                    if len(parts) > 1:
                        label += f" part {pi}/{len(parts)}"

                    covered_min, covered_max = update_covered_range(covered_min, covered_max, ch_no, ch_no)
                    coverage_line = f"【覆盖范围】第{covered_min}-第{covered_max}"
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
                        coverage_line=coverage_line,
                    )

        g += group_size

    # 3) 生成学习大纲
    log_line(log_path, "开始生成分节学习大纲（sectioned）")
    study = make_study_outline_sectioned(
        client=client,
        model=args.model,
        learning_notes=learning_notes,
        temperature=0.2,
        max_tokens_per_section=int(args.max_tokens_section),
        include_training_plan=bool(args.include_training_plan),
    )

    out_path = out_dir / f"study_outline_{start_ch}_{end_ch}.txt"
    out_path.write_text(study, encoding="utf-8")
    log_line(log_path, f"完成：写入学习大纲 {out_path}")

    print("✅ DONE")
    print(f"- 学习大纲：{out_path}")
    print(f"- 日志：{log_path}")


if __name__ == "__main__":
    main()
