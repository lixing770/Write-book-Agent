#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
emotion.py - Emotion Scene Director Notes (DeepSeek API) - SKIP MISSING

✅ Input:
  --split_dir contains multiple txt files like 1_10.txt, 11_20.txt ...
  Chapter heading format supported (your screenshot):
    "第一卷 平庸少年 第001章 离乡"
  (any prefix before 第xxx章 is allowed)

✅ Output (under --out_dir):
  chapter_notes/0001.md ... 0800.md  (missing chapters will be absent)
  batch_notes_50/0001_0050.md ... 0751_0800.md (ALWAYS produced, but may note missing chapters)
  final_summary_1_800.md
  meta/parse_report.json
  meta/missing_chapters.json
  meta/index.json (if --write_index)

✅ Skip missing:
  - Missing chapters in parsing => WARN only (unless --strict_complete)
  - Generation: missing_raw => skip
  - Batch merge: if some chapter md missing => skip them, and tell model "禁止脑补"
  - If an entire 50-chapter range has no notes => write placeholder file to keep pipeline complete

✅ Completeness for generated notes:
  - validation + auto repair + retries:
      * required sections exist
      * >= 3 upgrade steps
      * forbid "big-director/camera" terms
      * <=3 quote bullets in section 8

✅ Chunking:
  - If chapter text is too long, chunk by --chunk_chars
  - Generate per-chunk notes then merge into final chapter note (still validated)

DeepSeek API:
  - base_url: https://api.deepseek.com
  - endpoint used: {base_url}/v1/chat/completions

Dependencies:
  pip install requests
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import requests


# =========================
# Prompts (Emotion Scene ONLY)
# =========================

def chapter_system_prompt() -> str:
    return """你是“情绪戏导演 Agent”，你只负责情绪戏（emotion scene）的拆解与学习笔记。
你不负责镜头、调度、摄影、美术、世界观、打斗设计、宏大叙事，不要写这些。

必须抓住的6类要素（每章都要写）：
1) 情绪目标
2) 情绪触发器
3) 情绪阻力
4) 情绪升级台阶（至少3步：每步=手段+结果）
5) 情绪爆点
6) 情绪余波（情绪债）

强约束：
- 只允许 1-2 句剧情定位（为了说明情绪场景发生在哪）
- 不要长引用：最多3条短引用，每条<=20字
- 禁止镜头/摄影术语（推拉摇移、景别、特写、镜头、运镜、分镜、布光、剪辑等）
- 原文没写就写“原文未给出”，不得脑补
- 输出必须严格按【输出格式】

【输出格式】
# 第{chapter_id}章｜情绪戏学习笔记（导演视角）
## 0) 情绪场景定位（剧情1-2句）
...
## 1) 情绪目标（谁想让谁产生什么感受）
- 角色A：
- 角色B：
## 2) 情绪触发器（触发变化的具体点）
- 触发点：
- 触发形式：信息 / 行为 / 对话 / 环境事实
## 3) 情绪阻力（为什么不能立刻爆）
- 阻力1：
- 阻力2：
## 4) 情绪升级台阶（至少3步）
1) 手段：
   结果：
2) 手段：
   结果：
3) 手段：
   结果：
## 5) 爆点（爆发的形式与落点）
- 爆点发生处：
- 爆点形式：
- 爆点为何有效：
## 6) 情绪余波（爆后走向）
- 当下余波：
- 埋下的后续情绪债：
## 7) 可复用“情绪戏配方”（3-7条）
1. ...
2. ...
## 8) 短引用/句式样本（最多3条，<=20字）
- ...
"""


def chapter_user_prompt(chapter_id: int, chapter_text: str) -> str:
    return f"""下面是《仙逆》第{chapter_id}章原文。
请严格按System的【输出格式】输出“情绪戏学习笔记（导演视角）”。

【原文开始】
{chapter_text}
【原文结束】
"""


def repair_user_prompt(chapter_id: int, chapter_text: str, problems: List[str], prev_md: str) -> str:
    prob_text = "\n".join(f"- {p}" for p in problems)
    return f"""你上一版输出未通过质量检查，需要你在不编造原文信息的前提下修复。
问题列表：
{prob_text}

修复要求：
- 严格保留System的输出格式与标题
- 补齐缺失小节
- 删除任何“大导演/镜头/摄影”类词汇
- 情绪升级台阶必须至少3步，且每步要有“手段/结果”
- 短引用最多3条，每条<=20字

【原文开始】
{chapter_text}
【原文结束】

【上一版开始】
{prev_md}
【上一版结束】

请输出“修复后的最终版”，不要附加解释。
"""


def chunk_merge_system_prompt() -> str:
    return """你是“情绪戏导演 Agent（分块合并器）”。
你会把同一章的多个分块笔记合并为“完整的一章笔记”，必须严格输出同一个【输出格式】。
禁止剧情复述，禁止镜头术语。若分块信息冲突，以“更贴近原文细节、更具体”的表述为准。
保持结构完整，升级台阶至少3步。短引用<=3条，每条<=20字。
"""


def chunk_merge_user_prompt(chapter_id: int, partial_notes: str) -> str:
    return f"""请把以下“同一章的分块笔记”合并成一份完整的一章笔记。

章节：第{chapter_id}章

【分块笔记开始】
{partial_notes}
【分块笔记结束】

请直接输出最终“一章笔记”，严格按标准格式。
"""


def batch_merge_system_prompt() -> str:
    return """你是“情绪戏套路整理员”。只整理情绪戏，不做剧情复述。

必须输出：
A) 高频触发器（分类）
B) 高频阻力模型
C) 升级台阶模板（3/4/5步）
D) 爆点形式库
E) 余波与情绪债（不断供公式）
F) 泄压黑名单
G) Prompt Blocks（可复制）

输出格式：
# 第{start}-{end}章｜情绪戏套路库
## A) 触发器类型库
## B) 阻力模型库
## C) 升级台阶模板库
## D) 爆点形式库
## E) 余波与情绪债（不断供的方法）
## F) 泄压黑名单（避免写法）
## G) Prompt Blocks（可直接复制）
"""


def final_merge_system_prompt() -> str:
    return """你是“情绪戏总纲编纂者”，只输出情绪戏方法论总纲，不讲剧情。

必须包含：
1) 触发器谱系
2) 阻力模型
3) 升级台阶模板（可复制）
4) 爆点形式与适配条件
5) 余波与情绪债：不断供公式
6) 泄压黑名单
7) Prompt Blocks（弱/中/强）

输出结构：
# 仙逆 {start}-{end}｜情绪戏方法论总纲
## 1) 触发器谱系
## 2) 阻力模型
## 3) 升级台阶模板
## 4) 爆点形式与适配条件
## 5) 余波与情绪债：不断供公式
## 6) 泄压黑名单
## 7) Prompt Blocks（弱/中/强 三套）
"""


# =========================
# Validation (STRICT for produced notes)
# =========================

FORBIDDEN_BIG_DIRECTOR_TERMS = [
    "推拉", "摇移", "景别", "特写", "全景", "中景", "近景", "远景",
    "镜头", "运镜", "分镜", "光色", "布光", "灯光", "摄影", "机位",
    "构图", "剪辑", "蒙太奇",
]

REQUIRED_SECTIONS = [
    r"^#\s*第\d+章｜情绪戏学习笔记（导演视角）\s*$",
    r"^##\s*0\)\s*情绪场景定位（剧情1-2句）\s*$",
    r"^##\s*1\)\s*情绪目标（谁想让谁产生什么感受）\s*$",
    r"^##\s*2\)\s*情绪触发器（触发变化的具体点）\s*$",
    r"^##\s*3\)\s*情绪阻力（为什么不能立刻爆）\s*$",
    r"^##\s*4\)\s*情绪升级台阶（至少3步）\s*$",
    r"^##\s*5\)\s*爆点（爆发的形式与落点）\s*$",
    r"^##\s*6\)\s*情绪余波（爆后走向）\s*$",
    r"^##\s*7\)\s*可复用“情绪戏配方”（3-7条）\s*$",
    r"^##\s*8\)\s*短引用/句式样本（最多3条，<=20字）\s*$",
]


def validate_chapter_note(md: str) -> Tuple[bool, List[str]]:
    problems: List[str] = []
    for pat in REQUIRED_SECTIONS:
        if not re.search(pat, md, flags=re.MULTILINE):
            problems.append(f"missing_section: {pat}")

    if len(re.findall(r"^\s*\d\)\s*手段：", md, flags=re.MULTILINE)) < 3:
        problems.append("missing_upgrade_steps: need >=3 'n) 手段：' lines")

    for t in FORBIDDEN_BIG_DIRECTOR_TERMS:
        if t in md:
            problems.append(f"forbidden_term: {t}")

    m = re.search(r"^##\s*8\)\s*短引用/句式样本.*?\n(.*)$", md, flags=re.MULTILINE | re.DOTALL)
    if m:
        tail = m.group(1)
        tail = re.split(r"^##\s+", tail, maxsplit=1, flags=re.MULTILINE)[0]
        quote_lines = [ln for ln in tail.splitlines() if ln.strip().startswith("-")]
        if len(quote_lines) > 3:
            problems.append("too_many_quotes: section 8 bullet lines > 3")

    return (len(problems) == 0, problems)


# =========================
# Chapter parsing (supports "第一卷 ... 第001章 ...")
# =========================

CHAPTER_HEADER_PATTERNS = [
    re.compile(r"^\s*.*?第\s*0*(\d{1,4})\s*章[^\n]*$", re.MULTILINE),
    re.compile(r"^\s*.*?第\s*0*(\d{1,4})\s*回[^\n]*$", re.MULTILINE),
    re.compile(r"^\s*(?:Chapter|CHAPTER)\s+0*(\d{1,4})\b[^\n]*$", re.MULTILINE),
]


def parse_chapters_from_text(text: str) -> Dict[int, str]:
    matches: List[Tuple[int, int, int]] = []
    for pat in CHAPTER_HEADER_PATTERNS:
        for m in pat.finditer(text):
            cid = int(m.group(1))
            start = m.start()
            line_end = text.find("\n", m.end())
            if line_end == -1:
                line_end = m.end()
            matches.append((cid, start, line_end))

    matches.sort(key=lambda x: x[1])
    if not matches:
        return {}

    seen = set()
    uniq: List[Tuple[int, int, int]] = []
    for cid, s, le in matches:
        if cid in seen:
            continue
        seen.add(cid)
        uniq.append((cid, s, le))

    out: Dict[int, str] = {}
    for i, (cid, s, le) in enumerate(uniq):
        next_start = uniq[i + 1][1] if i + 1 < len(uniq) else len(text)
        heading = text[s:le].strip()
        body = text[le:next_start].strip()
        out[cid] = (heading + "\n" + body).strip()
    return out


def load_all_chapters(split_dir: Path) -> Tuple[Dict[int, str], Dict[str, int]]:
    files = sorted(split_dir.glob("*.txt"), key=lambda p: p.name)
    all_ch: Dict[int, str] = {}
    report: Dict[str, int] = {}
    for fp in files:
        text = fp.read_text(encoding="utf-8", errors="replace")
        parsed = parse_chapters_from_text(text)
        report[fp.name] = len(parsed)
        for cid, ctext in parsed.items():
            if cid not in all_ch:
                all_ch[cid] = ctext
    return all_ch, report


# =========================
# DeepSeek API client
# =========================

class DeepSeekAPI:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        max_tokens: int,
        temperature: float,
        timeout_sec: int,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/") + "/v1"
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_sec = timeout_sec

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        r = requests.post(url, headers=headers, json=payload, timeout=self.timeout_sec)
        r.raise_for_status()
        data = r.json()
        return (data["choices"][0]["message"]["content"] or "").strip()


# =========================
# Config + helpers
# =========================

@dataclass
class Config:
    split_dir: Path
    out_dir: Path
    model: str
    api_key: str
    base_url: str
    start: int
    end: int
    batch: int
    max_tokens: int
    chunk_chars: int
    write_index: bool
    strict_complete: bool
    max_retries: int
    sleep_sec: float
    overwrite: bool
    validate: bool
    reask_on_fail: bool


def ensure_dirs(out_dir: Path, batch: int) -> Dict[str, Path]:
    paths = {
        "chapter_notes": out_dir / "chapter_notes",
        "batch_notes": out_dir / f"batch_notes_{batch}",
        "meta": out_dir / "meta",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def backoff_sleep(base: float, attempt: int) -> None:
    time.sleep(min(base * (2 ** (attempt - 1)), 6.0))


def split_into_chunks(text: str, chunk_chars: int) -> List[str]:
    if chunk_chars <= 0 or len(text) <= chunk_chars:
        return [text]

    chunks: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + chunk_chars, n)
        if j < n:
            k = text.rfind("\n", i, j)
            if k > i + int(chunk_chars * 0.6):
                j = k
        chunks.append(text[i:j].strip())
        i = j
    return [c for c in chunks if c]


# =========================
# Generation
# =========================

def generate_note_with_repair(api: DeepSeekAPI, cfg: Config, cid: int, source_text: str, sys_prompt: str, user_prompt: str) -> str:
    last_md = ""
    last_probs: List[str] = []

    for attempt in range(1, cfg.max_retries + 1):
        md = api.chat(sys_prompt, user_prompt)
        last_md = md

        if cfg.validate:
            ok, probs = validate_chapter_note(md)
            last_probs = probs
            if not ok:
                if cfg.reask_on_fail and attempt < cfg.max_retries:
                    user_prompt = repair_user_prompt(cid, source_text, probs, md)
                    backoff_sleep(cfg.sleep_sec, attempt)
                    continue
                raise RuntimeError(f"validation failed: {probs}")

        return md

    raise RuntimeError(f"failed after retries: {last_probs}\nlast_md={last_md[:200]}")


def generate_chapter(api: DeepSeekAPI, cfg: Config, paths: Dict[str, Path], cid: int, raw_text: str) -> None:
    out_fp = paths["chapter_notes"] / f"{cid:04d}.md"
    if out_fp.exists() and not cfg.overwrite:
        return

    chunks = split_into_chunks(raw_text, cfg.chunk_chars)

    partials: List[str] = []
    for ch in chunks:
        sys_prompt = chapter_system_prompt().format(chapter_id=cid)
        user_prompt = chapter_user_prompt(cid, ch)
        md = generate_note_with_repair(api, cfg, cid, ch, sys_prompt, user_prompt)
        partials.append(md)
        time.sleep(cfg.sleep_sec)

    if len(partials) == 1:
        final_md = partials[0]
    else:
        merge_sys = chunk_merge_system_prompt()
        merge_user = chunk_merge_user_prompt(cid, "\n\n---\n\n".join(partials))
        final_md = generate_note_with_repair(api, cfg, cid, raw_text, merge_sys, merge_user)

    out_fp.write_text(final_md, encoding="utf-8")


def merge_batch(api: DeepSeekAPI, cfg: Config, paths: Dict[str, Path], start: int, end: int) -> Path:
    out_fp = paths["batch_notes"] / f"{start:04d}_{end:04d}.md"
    if out_fp.exists() and not cfg.overwrite:
        return out_fp

    parts: List[str] = []
    present: List[int] = []
    missing_md: List[int] = []

    for cid in range(start, end + 1):
        fp = paths["chapter_notes"] / f"{cid:04d}.md"
        if fp.exists():
            parts.append(fp.read_text(encoding="utf-8"))
            present.append(cid)
        else:
            missing_md.append(cid)

    # Entire range empty => placeholder file (keep pipeline complete)
    if not parts:
        placeholder = (
            f"# 第{start}-{end}章｜情绪戏套路库\n"
            f"## A) 触发器类型库\n"
            f"- 本范围无可用章节笔记（全部缺失或未生成）。\n\n"
            f"## B) 阻力模型库\n- 同上。\n\n"
            f"## C) 升级台阶模板库\n- 同上。\n\n"
            f"## D) 爆点形式库\n- 同上。\n\n"
            f"## E) 余波与情绪债（不断供的方法）\n- 同上。\n\n"
            f"## F) 泄压黑名单（避免写法）\n- 同上。\n\n"
            f"## G) Prompt Blocks（可直接复制）\n- 同上。\n"
        )
        out_fp.write_text(placeholder, encoding="utf-8")
        return out_fp

    notes_bundle = "\n\n---\n\n".join(parts)
    sys_prompt = batch_merge_system_prompt().format(start=start, end=end)

    missing_str = ", ".join(str(x) for x in missing_md) if missing_md else "无"
    present_str = ", ".join(str(x) for x in present)

    user_prompt = (
        f"请把以下范围内的“情绪戏学习笔记”合并成套路库：\n"
        f"范围：{start}-{end}\n"
        f"注意：以下章节缺失或未生成，严禁脑补：{missing_str}\n"
        f"本次可用章节：{present_str}\n\n"
        f"【笔记集合开始】\n{notes_bundle}\n【笔记集合结束】"
    )

    md = api.chat(sys_prompt, user_prompt)
    if not md.startswith(f"# 第{start}-{end}章｜情绪戏套路库"):
        md = f"# 第{start}-{end}章｜情绪戏套路库\n\n" + md

    out_fp.write_text(md, encoding="utf-8")
    return out_fp


def merge_final(api: DeepSeekAPI, cfg: Config, batch_files: List[Path]) -> Path:
    out_fp = cfg.out_dir / f"final_summary_{cfg.start}_{cfg.end}.md"
    if out_fp.exists() and not cfg.overwrite:
        return out_fp

    bundle = "\n\n---\n\n".join([p.read_text(encoding="utf-8") for p in batch_files])
    sys_prompt = final_merge_system_prompt().format(start=cfg.start, end=cfg.end)
    user_prompt = (
        f"以下是多个“50章套路库”，其中部分范围可能缺章（已经在各范围里声明）。"
        f"请只基于已有内容总结，不要脑补缺章。\n\n"
        f"【50章套路库开始】\n{bundle}\n【50章套路库结束】"
    )
    md = api.chat(sys_prompt, user_prompt)
    out_fp.write_text(md, encoding="utf-8")
    return out_fp


# =========================
# CLI
# =========================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--split_dir", required=True)
    p.add_argument("--out_dir", required=True)

    p.add_argument("--model", required=True)
    p.add_argument("--api_key", required=True)
    p.add_argument("--base_url", required=True)

    p.add_argument("--start_chapter", type=int, default=1)
    p.add_argument("--end_chapter", type=int, default=800)
    p.add_argument("--batch_chapters", type=int, default=50)

    p.add_argument("--max_tokens", type=int, default=3200)
    p.add_argument("--chunk_chars", type=int, default=22000)

    p.add_argument("--max_retries", type=int, default=3)
    p.add_argument("--sleep_sec", type=float, default=0.2)

    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no_validate", action="store_true")
    p.add_argument("--no_reask", action="store_true")

    p.add_argument("--write_index", action="store_true")
    p.add_argument("--strict_complete", action="store_true")

    p.add_argument("--mode", choices=["all", "chapters", "merge", "final"], default="all")

    return p.parse_args()


def main() -> int:
    args = parse_args()

    cfg = Config(
        split_dir=Path(args.split_dir),
        out_dir=Path(args.out_dir),
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        start=args.start_chapter,
        end=args.end_chapter,
        batch=args.batch_chapters,
        max_tokens=args.max_tokens,
        chunk_chars=args.chunk_chars,
        write_index=args.write_index,
        strict_complete=args.strict_complete,
        max_retries=args.max_retries,
        sleep_sec=args.sleep_sec,
        overwrite=args.overwrite,
        validate=(not args.no_validate),
        reask_on_fail=(not args.no_reask),
    )

    paths = ensure_dirs(cfg.out_dir, cfg.batch)

    # Parse chapters
    all_ch, report = load_all_chapters(cfg.split_dir)
    (paths["meta"] / "parse_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    missing = [cid for cid in range(cfg.start, cfg.end + 1) if cid not in all_ch]
    (paths["meta"] / "missing_chapters.json").write_text(json.dumps(missing, ensure_ascii=False, indent=2), encoding="utf-8")

    if missing:
        if cfg.strict_complete:
            print(f"[FATAL] Missing {len(missing)} chapters in input. See: {paths['meta'] / 'missing_chapters.json'}", file=sys.stderr)
            return 2
        else:
            print(f"[WARN] Missing {len(missing)} chapters in input. Will skip them. See missing_chapters.json", file=sys.stderr)

    timeout_sec = 300 if "reasoner" in cfg.model else 180
    api = DeepSeekAPI(
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        model=cfg.model,
        max_tokens=cfg.max_tokens,
        temperature=0.4 if "reasoner" in cfg.model else 0.3,
        timeout_sec=timeout_sec,
    )

    index: Dict[str, dict] = {}

    # Generate chapters
    if args.mode in ("all", "chapters"):
        for cid in range(cfg.start, cfg.end + 1):
            if cid not in all_ch:
                index[str(cid)] = {"chapter": cid, "status": "missing_raw"}
                continue

            try:
                generate_chapter(api, cfg, paths, cid, all_ch[cid])
                index[str(cid)] = {"chapter": cid, "status": "ok"}
            except Exception as e:
                index[str(cid)] = {"chapter": cid, "status": "failed", "error": str(e)}
                (paths["meta"] / f"failed_{cid:04d}.json").write_text(
                    json.dumps(index[str(cid)], ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(f"[WARN] Chapter {cid} failed (skipping): {e}", file=sys.stderr)
                # skip in non-strict; fail-fast in strict
                if cfg.strict_complete:
                    return 3

            if cfg.write_index:
                (paths["meta"] / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

    # Merge batches (always produce batch file per range)
    batch_files: List[Path] = []
    if args.mode in ("all", "merge", "final"):
        s = cfg.start
        while s <= cfg.end:
            e = min(s + cfg.batch - 1, cfg.end)
            try:
                bf = merge_batch(api, cfg, paths, s, e)
                batch_files.append(bf)
            except Exception as e2:
                print(f"[FATAL] Batch merge {s}-{e} failed: {e2}", file=sys.stderr)
                return 4
            s = e + 1

    # Final summary
    if args.mode in ("all", "final"):
        try:
            merge_final(api, cfg, batch_files)
        except Exception as e3:
            print(f"[FATAL] Final merge failed: {e3}", file=sys.stderr)
            return 5

    print("[DONE] Pipeline finished (skip-missing mode).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())