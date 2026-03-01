#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build Chief Editor Study Notes per batch (e.g. every 50 chapters) from split/*.txt.

✅ 解决两件事：
1) 学习笔记“主编化”：输出的是可复用的编辑套路/规则库（不是剧情复述）
2) DeepSeek 输出被截断：自动“续写”直到出现【END】，保证每节完整

Input:
  - split_dir: split/*.txt, e.g. 1_10.txt, 11_20.txt...
Output (out_dir):
  - batches/chief_editor_notes_{start}_{end}.md   # 每batch一份完整学习笔记
  - micro/{file}.micro.txt                        # 每split文件一个micro缓存（模板化）
  - state/progress.json                           # 断点续跑
  - logs/run.log

DeepSeek API via openai SDK:
  export DEEPSEEK_API_KEY="sk-..."
"""

from __future__ import annotations

import os
import re
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional


try:
    from openai import OpenAI
except Exception as e:
    raise RuntimeError("Missing dependency: openai. Install: pip install -U openai") from e


# ======================
# IO / LOG
# ======================

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log_line(log_path: Path, msg: str) -> None:
    ensure_dir(log_path.parent)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{now_str()}] {msg}\n")

def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")

def write_text(p: Path, s: str) -> None:
    ensure_dir(p.parent)
    p.write_text(s, encoding="utf-8")

def load_json(p: Path, default):
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default

def save_json(p: Path, obj) -> None:
    ensure_dir(p.parent)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# ======================
# FILE ORDERING / RANGE
# ======================

def parse_range_from_filename(name: str) -> Optional[Tuple[int, int]]:
    nums = [int(x) for x in re.findall(r"\d+", name)]
    if not nums:
        return None
    if len(nums) == 1:
        return (nums[0], nums[0])
    a, b = nums[0], nums[1]
    if a > b:
        a, b = b, a
    return (a, b)

def overlap(a1: int, a2: int, b1: int, b2: int) -> bool:
    return not (a2 < b1 or b2 < a1)

def list_split_files(split_dir: Path) -> List[Path]:
    items = []
    for fp in split_dir.glob("*.txt"):
        r = parse_range_from_filename(fp.name)
        if r:
            items.append((r[0], r[1], fp))
    items.sort(key=lambda x: (x[0], x[1], x[2].name))
    return [x[2] for x in items]


# ======================
# ROBUST CHUNKING (NO blank line needed)
# ======================

def chunk_text(text: str, max_chars: int, overlap_chars: int = 200) -> List[str]:
    """
    Robust chunking even if file has NO blank lines.
    - hard slice by chars
    - try cut at newline near tail
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + max_chars, n)

        cut = j
        window = text[i:j]
        k = window.rfind("\n")
        if k != -1 and (j - (i + k)) <= 400:
            cut = i + k

        if cut <= i + 200:
            cut = j

        part = text[i:cut].strip()
        if part:
            chunks.append(part)

        if cut >= n:
            break
        i = max(cut - overlap_chars, i + 1)

    return chunks


# ======================
# LLM CORE
# ======================

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

def retry_call(fn, retries: int, sleep_base: float, log_path: Path, label: str) -> str:
    last: Optional[Exception] = None
    for i in range(1, retries + 1):
        try:
            return fn()
        except Exception as e:
            last = e
            log_line(log_path, f"{label} failed attempt={i}/{retries} err={repr(e)}")
            time.sleep(min(sleep_base * (2 ** (i - 1)), 20))
    raise RuntimeError(f"{label} failed after {retries} retries: {repr(last)}")


# ======================
# CONTINUATION (guarantee END)
# ======================

def _max_suffix_prefix_overlap(a: str, b: str, min_k: int = 30, max_k: int = 400) -> int:
    """Find overlap length where suffix of a == prefix of b"""
    a = a or ""
    b = b or ""
    max_k = min(max_k, len(a), len(b))
    for k in range(max_k, min_k - 1, -1):
        if a[-k:] == b[:k]:
            return k
    return 0

def ensure_end_complete(
    client: OpenAI,
    model: str,
    system: str,
    first_user: str,
    temperature: float,
    max_tokens: int,
    log_path: Path,
    label: str,
    end_marker: str = "【END】",
    max_continuations: int = 4,
    tail_chars: int = 800,
) -> str:
    """
    Call LLM; if missing end_marker, keep asking to continue until end_marker appears.
    """
    def _call0():
        return llm_chat_once(client, model, system, first_user, temperature=temperature, max_tokens=max_tokens)

    out = retry_call(_call0, retries=4, sleep_base=1.5, log_path=log_path, label=f"{label} call0")

    acc = out.strip()
    if end_marker in acc:
        return acc

    for t in range(1, max_continuations + 1):
        tail = acc[-tail_chars:]
        cont_user = (
            "你上条输出被截断/缺少结尾。\n"
            f"请【直接续写】（不要重复已写内容），直到输出 {end_marker} 为止。\n"
            "要求：继续保持同样的格式与约束。\n\n"
            f"【已输出末尾（仅供定位，不要重复）】\n{tail}\n"
        )

        def _call_cont():
            return llm_chat_once(client, model, system, cont_user, temperature=0.0, max_tokens=max_tokens)

        nxt = retry_call(_call_cont, retries=3, sleep_base=1.2, log_path=log_path, label=f"{label} cont{t}")
        nxt = (nxt or "").strip()

        # merge with simple de-dup overlap
        ov = _max_suffix_prefix_overlap(acc, nxt)
        if ov > 0:
            nxt = nxt[ov:].lstrip()

        acc = (acc + "\n" + nxt).strip()

        if end_marker in acc:
            return acc

    # fallback append marker
    log_line(log_path, f"{label} WARNING: still missing END, force append.")
    return acc + f"\n{end_marker}"


# ======================
# PROMPTS (主编化：模板/规则，不准剧情复述)
# ======================
MICRO_SYSTEM = """你是“主编知识库-微提炼器（通用写作规则）”。

目标：把输入小说文本压缩为【通用可复用】的编辑规则/写作套路卡片。
产物将用于训练“主编Agent”，所以必须：短、硬、可执行。

禁止：
- 禁止剧情复述（不要写发生了什么、不要按章概括）
- 禁止空泛正确话（如“保持张力”“注意节奏”）
- 禁止长句堆叠

必须输出（硬性）：
- 第一行必须是【MICRO】
- 只用要点（- 开头）
- 每条格式：<标签>：<可复用规则/模板/检查点>
- 每条尽量具体：带条件/触发点/动作/结果（至少包含其中2个）
- 12-16条；每条<=32字；总长<=900中文字符
- 末尾必须是【END】
只输出micro本体，不要解释。
"""

MICRO_USER_TMPL = """【文件标签】{label}
【正文】
{text}

只输出micro，末尾必须【END】。
"""

SECTION_SYSTEM = """你是“主编学习笔记-分节生成器（模板化/可抄）”。
你只生成指定小节内容：输出一组“可复用套路/规则库”。

禁止：
- 禁止剧情复述（不要写剧情节点）
- 避免具体角色名/地名（最多2个）
- 不允许输出【未知】（宁可给通用模板）

必须：
- 要点式（- 开头）
- 每条格式：<标签>：<可复用规则/套路>
- 条数<=用户指定
- 每条<=用户指定字数
- 末尾必须输出【END】（否则视为不完整）

只输出该小节内容，不要额外解释。
"""


SECTION_USER_TMPL = """你在写《仙逆》主编学习笔记的一个小节。

【覆盖范围】第{start}-{end}章
【小节名称】{section_name}

【输出限制】
- 最多 {max_items} 条
- 每条<= {max_len} 字
- 末尾必须【END】

【输入材料：micro-notes 合集】
{micro_bundle}
"""

# 每节条数/字数（你想“多点字数显示”主要调这两个 + section_max_tokens）

SECTIONS = [
    ("1) 主线推进与阶段节点（模板）", 10, 36),
    ("2) 节奏结构与卡点（规则库）", 10, 36),
    ("3) 章末钩子库（可直接抄）", 12, 30),
    ("4) 冲突模板与推进套路", 10, 36),
    ("5) 升级/爽点/回报机制（公式）", 10, 36),
    ("6) 世界观/规则投放手法（边写边讲）", 10, 36),
    ("7) 人物功能与关系推进（分工）", 10, 36),
    ("8) 伏笔：埋点→回收（规则）", 10, 38),
    ("9) 风险避坑 + 下一轮编辑指令", 12, 36),
]


# ======================
# VALIDATION
# ======================

def is_bad_micro(s: str) -> bool:
    t = (s or "").strip()
    if not t:
        return True
    if t in ("【END】", "[END]", "END"):
        return True
    if "【MICRO】" not in t:
        return True
    if "【END】" not in t:
        return True
    core = t.replace("【MICRO】", "").replace("【END】", "").strip()
    return len(core) < 20

def strip_end_mark(s: str) -> str:
    return (s or "").replace("【END】", "").strip()


# ======================
# MICRO BUILD (complete + merge)
# ======================

def build_micro_for_file(
    client: OpenAI,
    model: str,
    fp: Path,
    chunk_chars: int,
    temperature: float,
    micro_max_tokens: int,
    log_path: Path,
) -> str:
    r = parse_range_from_filename(fp.name) or (0, 0)
    label = f"{fp.name} (range {r[0]}-{r[1]})"

    text = read_text(fp)
    parts = chunk_text(text, max_chars=chunk_chars, overlap_chars=200)
    if not parts:
        return "【MICRO】\n- 空文本：跳过\n【END】"

    micros: List[str] = []
    for idx, part in enumerate(parts, 1):
        u = MICRO_USER_TMPL.format(label=f"{label} part {idx}/{len(parts)}", text=part)

        out = ensure_end_complete(
            client=client,
            model=model,
            system=MICRO_SYSTEM,
            first_user=u,
            temperature=temperature,
            max_tokens=micro_max_tokens,
            log_path=log_path,
            label=f"MICRO {label} {idx}",
            end_marker="【END】",
            max_continuations=4,
            tail_chars=700,
        )

        if is_bad_micro(out):
            out = "【MICRO】\n- 兜底：生成失败\n【END】"

        micros.append(out.strip())

    # merge into one micro (still template rules)
    merged = "\n\n".join(micros)
    merge_user = (
        f"【文件标签】{label}\n"
        "下面是多段micro，请合并为一个更短micro（<=700字），保持“模板化规则”风格。\n"
        "必须【MICRO】开头，末尾【END】。\n\n"
        f"{merged}"
    )

    outm = ensure_end_complete(
        client=client,
        model=model,
        system=MICRO_SYSTEM,
        first_user=merge_user,
        temperature=0.0,
        max_tokens=micro_max_tokens,
        log_path=log_path,
        label=f"MICRO_MERGE {label}",
        end_marker="【END】",
        max_continuations=4,
        tail_chars=700,
    )

    if is_bad_micro(outm):
        outm = "【MICRO】\n- 兜底：合并失败\n【END】"

    return outm.strip()


# ======================
# SECTION / BATCH NOTE
# ======================

def gen_section(
    client: OpenAI,
    model: str,
    start: int,
    end: int,
    section_name: str,
    max_items: int,
    max_len: int,
    micro_bundle: str,
    temperature: float,
    section_max_tokens: int,
    log_path: Path,
) -> str:
    u = SECTION_USER_TMPL.format(
        start=start, end=end,
        section_name=section_name,
        max_items=max_items,
        max_len=max_len,
        micro_bundle=micro_bundle
    )

    out = ensure_end_complete(
        client=client,
        model=model,
        system=SECTION_SYSTEM,
        first_user=u,
        temperature=temperature,
        max_tokens=section_max_tokens,
        log_path=log_path,
        label=f"SECTION {start}-{end} {section_name}",
        end_marker="【END】",
        max_continuations=5,
        tail_chars=700,
    )

    cleaned = strip_end_mark(out)
    # 兜底：避免空
    if not cleaned.strip():
        cleaned = "- 兜底：本节输出为空"
    return cleaned.strip()

def generate_batch_note(
    client: OpenAI,
    model: str,
    start: int,
    end: int,
    micro_bundle: str,
    temperature: float,
    section_max_tokens: int,
    log_path: Path,
) -> str:
    blocks: List[str] = []
    blocks.append(f"【主编学习笔记｜仙逆｜第{start}-{end}章】\n")
    for sec_name, max_items, max_len in SECTIONS:
        sec = gen_section(
            client=client,
            model=model,
            start=start,
            end=end,
            section_name=sec_name,
            max_items=max_items,
            max_len=max_len,
            micro_bundle=micro_bundle,
            temperature=temperature,
            section_max_tokens=section_max_tokens,
            log_path=log_path,
        )
        blocks.append(f"### {sec_name}\n{sec}\n")
    return "\n".join(blocks).strip() + "\n"


# ======================
# MAIN
# ======================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--start_chapter", type=int, default=1)
    ap.add_argument("--end_chapter", type=int, required=True)
    ap.add_argument("--batch_size", type=int, default=50)

    ap.add_argument("--base_url", type=str, default="https://api.deepseek.com")
    ap.add_argument("--model", type=str, default="deepseek-reasoner")
    ap.add_argument("--api_key_env", type=str, default="DEEPSEEK_API_KEY")

    ap.add_argument("--temperature", type=float, default=0.0)

    # 输入切块大小（解决你的“原文没空行读不动/太长”问题）
    ap.add_argument("--chunk_chars", type=int, default=9000)

    # 你要“多点字数显示”就主要调这两个（建议 micro 900-1200，section 1400-2200）
    ap.add_argument("--micro_max_tokens", type=int, default=1100)
    ap.add_argument("--section_max_tokens", type=int, default=1800)

    # micro_bundle（所有micro合到一起）太大也会压爆上下文，这里限制一下
    ap.add_argument("--micro_bundle_max_chars", type=int, default=18000)

    # 是否强制重做micro（你想重跑某段时用）
    ap.add_argument("--force_regen_micro", action="store_true")

    args = ap.parse_args()

    split_dir = Path(args.split_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    ensure_dir(out_dir)

    log_path = out_dir / "logs" / "run.log"
    ensure_dir(log_path.parent)

    api_key = os.getenv(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"Missing env var {args.api_key_env}. Do: export {args.api_key_env}='sk-...'")
    if api_key.lower().startswith("bearer "):
        api_key = api_key[7:].strip()

    client = OpenAI(api_key=api_key, base_url=args.base_url)

    files = list_split_files(split_dir)
    if not files:
        raise RuntimeError(f"No .txt files found in {split_dir}")

    micro_dir = out_dir / "micro"
    batch_dir = out_dir / "batches"
    state_dir = out_dir / "state"
    ensure_dir(micro_dir)
    ensure_dir(batch_dir)
    ensure_dir(state_dir)

    state_path = state_dir / "progress.json"
    state = load_json(state_path, default={"done_batches": [], "bad_micro_regen": 0})
    done_batches = set(state.get("done_batches", []))

    start = args.start_chapter
    end = args.end_chapter
    bs = args.batch_size

    log_line(log_path, f"START split_dir={split_dir} range={start}-{end} batch_size={bs} model={args.model}")

    for s in range(start, end + 1, bs):
        e = min(s + bs - 1, end)
        batch_key = f"{s}_{e}"
        out_file = batch_dir / f"chief_editor_notes_{s}_{e}.md"

        if batch_key in done_batches and out_file.exists():
            log_line(log_path, f"SKIP batch {batch_key} exists")
            continue

        log_line(log_path, f"PROCESS batch {batch_key}")

        # split files overlapping [s,e]
        batch_files: List[Path] = []
        for fp in files:
            r = parse_range_from_filename(fp.name)
            if not r:
                continue
            a, b = r
            if overlap(a, b, s, e):
                batch_files.append(fp)

        if not batch_files:
            log_line(log_path, f"WARNING batch {batch_key} has no files")
            write_text(out_file, f"【主编学习笔记｜仙逆｜第{s}-{e}章】\n\n（未找到对应 split 文件）\n")
            done_batches.add(batch_key)
            state["done_batches"] = sorted(done_batches)
            save_json(state_path, state)
            continue

        micros: List[str] = []
        for fp in batch_files:
            micro_path = micro_dir / f"{fp.stem}.micro.txt"
            micro = ""

            if (not args.force_regen_micro) and micro_path.exists():
                micro = read_text(micro_path).strip()
                if is_bad_micro(micro):
                    log_line(log_path, f"BAD micro cache -> regen {micro_path}")
                    micro = ""
            else:
                micro = ""

            if not micro:
                micro = build_micro_for_file(
                    client=client,
                    model=args.model,
                    fp=fp,
                    chunk_chars=args.chunk_chars,
                    temperature=args.temperature,
                    micro_max_tokens=args.micro_max_tokens,
                    log_path=log_path,
                )
                write_text(micro_path, micro)

            micros.append(micro)

        micro_bundle = "\n\n".join(micros)
        if len(micro_bundle) > args.micro_bundle_max_chars:
            micro_bundle = micro_bundle[:args.micro_bundle_max_chars] + "\n（micro_bundle截断）"

        note = generate_batch_note(
            client=client,
            model=args.model,
            start=s,
            end=e,
            micro_bundle=micro_bundle,
            temperature=args.temperature,
            section_max_tokens=args.section_max_tokens,
            log_path=log_path,
        )

        write_text(out_file, note)
        log_line(log_path, f"WROTE {out_file}")

        done_batches.add(batch_key)
        state["done_batches"] = sorted(done_batches)
        save_json(state_path, state)

    log_line(log_path, "FINISH")
    print("✅ DONE")
    print(f"- batches: {batch_dir}")
    print(f"- micro: {micro_dir}")
    print(f"- log: {log_path}")
    print(f"- state: {state_path}")


if __name__ == "__main__":
    main()
