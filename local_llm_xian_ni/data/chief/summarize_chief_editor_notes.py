#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Summarize Chief Editor batch notes into one consolidated KB (generic + a bit of style).

Input:
  - in_dir: folder containing chief_editor_notes_*.md (e.g. chief/batches/)
Output (out_dir):
  - chief_editor_kb_summary.md
  - logs/summarize.log
  - state/progress.json (resume)

DeepSeek API via OpenAI SDK:
  export DEEPSEEK_API_KEY="sk-..."

Key features:
  - correct ordering by chapter range in filename
  - chunking to avoid too-long inputs
  - iterative KB update (map-reduce style)
  - ensure output completeness (auto-continue until 【END】)
"""

from __future__ import annotations
import os, re, json, time, argparse
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional

try:
    from openai import OpenAI
except Exception as e:
    raise RuntimeError("Missing dependency: openai. Install: pip install -U openai") from e


# -----------------------
# IO / LOG
# -----------------------
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


# -----------------------
# Ordering
# -----------------------
def parse_range_from_filename(name: str) -> Tuple[int, int]:
    nums = [int(x) for x in re.findall(r"\d+", name)]
    if len(nums) >= 2:
        a, b = nums[-2], nums[-1]
        return (a, b) if a <= b else (b, a)
    if len(nums) == 1:
        return (nums[0], nums[0])
    return (10**9, 10**9)

def list_note_files(in_dir: Path) -> List[Path]:
    files = list(in_dir.glob("chief_editor_notes_*.md"))
    files.sort(key=lambda p: parse_range_from_filename(p.name))
    return files


# -----------------------
# Chunking
# -----------------------
def chunk_by_size(text: str, max_chars: int) -> List[str]:
    """Simple size-based chunking (prefer splitting at blank lines)."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    parts = re.split(r"\n\s*\n", text)
    chunks: List[str] = []
    buf: List[str] = []
    size = 0

    for para in parts:
        para = para.strip()
        if not para:
            continue
        add = len(para) + 2
        if buf and size + add > max_chars:
            chunks.append("\n\n".join(buf).strip())
            buf = [para]
            size = len(para)
        else:
            buf.append(para)
            size += add

    if buf:
        chunks.append("\n\n".join(buf).strip())
    return chunks


# -----------------------
# LLM calls + retry + ensure END
# -----------------------
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

def retry_call(fn, retries: int, sleep_base: float, log_path: Path, label: str):
    last: Optional[Exception] = None
    for i in range(1, retries + 1):
        try:
            return fn()
        except Exception as e:
            last = e
            log_line(log_path, f"{label} failed attempt={i}/{retries} err={repr(e)}")
            time.sleep(min(sleep_base * (2 ** (i - 1)), 18))
    raise RuntimeError(f"{label} failed after {retries} retries: {repr(last)}")

def _merge_no_dup(existing: str, new: str) -> str:
    """Remove overlap if model repeats tail/head."""
    if not existing:
        return new
    a = existing[-800:]  # tail
    b = new[:800]        # head
    # find max overlap
    max_k = 0
    max_len = min(len(a), len(b))
    for k in range(40, max_len + 1):
        if a[-k:] == b[:k]:
            max_k = k
    if max_k > 0:
        return existing + new[max_k:]
    return existing + new

def ensure_end_complete(client: OpenAI, model: str, system: str,
                        base_user: str, draft: str,
                        temperature: float, max_tokens: int,
                        log_path: Path, label: str,
                        end_marker: str = "【END】",
                        max_rounds: int = 6) -> str:
    """Auto-continue until end_marker appears."""
    if end_marker in draft:
        return draft

    out = draft
    for r in range(1, max_rounds + 1):
        tail = out[-900:] if len(out) > 900 else out
        cont_system = "你是续写器。继续补全用户要求的输出，不要重复已写内容，最后必须输出【END】。只输出续写部分。"
        cont_user = (
            f"请继续补全并结束。\n\n"
            f"【原始任务】\n{base_user}\n\n"
            f"【已输出末尾（供定位，不要重复）】\n{tail}\n"
        )

        def _call():
            return llm_chat_once(client, model, cont_system, cont_user, temperature=0.0, max_tokens=max_tokens)

        add = retry_call(_call, retries=3, sleep_base=1.5, log_path=log_path, label=f"{label} continue r={r}")
        out = _merge_no_dup(out, "\n" + add.strip())

        if end_marker in out:
            return out

    log_line(log_path, f"WARNING: {label} still missing {end_marker} after {max_rounds} rounds")
    return out


# -----------------------
# Summarization Prompts (generic KB)
# -----------------------
KB_SYSTEM = """你是“主编学习笔记总编（通用写作规则KB）”。
你将把多份主编学习笔记归纳成“一份总纲KB”（通用、可复用、略带主编风格）。

硬约束（必须遵守）：
- 禁止剧情复述、禁止按章概括
- 禁止空泛正确话（每条必须可执行：像条款/模板/检查点）
- 必须用 <标签>：<规则/模板/检查点> 的形式
- 只输出9个小节（标题必须完全照抄）
- 每节 8-12 条（不要超过12）
- 每条 <= 36 个中文字符（尽量短硬）
- 总长度建议 <= 6500 中文字符（宁可删减去重）
- 末尾必须输出【END】

输出结构（标题照抄）：
【主编知识库总纲｜通用网文写作】
### 1) 主线推进与阶段节点（模板）
### 2) 节奏结构与卡点（规则库）
### 3) 章末钩子库（可直接抄）
### 4) 冲突模板与推进套路
### 5) 升级/爽点/回报机制（公式）
### 6) 世界观/规则投放手法（边写边讲）
### 7) 人物功能与关系推进（分工）
### 8) 伏笔：埋点→回收（规则）
### 9) 风险避坑 + 下一轮编辑指令
"""

KB_UPDATE_USER_TMPL = """你在做“递归更新”：给你
A) 当前KB（可能为空）
B) 新输入材料（来自多份主编学习笔记）

你只输出“更新后的KB”，必须继续满足 KB_SYSTEM 的所有硬约束，并且去重、压缩、保留最有用条款。

【A 当前KB】
{kb}

【B 新材料标签】
{label}

【B 新材料正文】
{text}
"""


# -----------------------
# Resume
# -----------------------
def load_progress(p: Path) -> dict:
    if not p.exists():
        return {"done": []}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"done": []}

def save_progress(p: Path, data: dict) -> None:
    ensure_dir(p.parent)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, type=str, help="chief/batches folder containing chief_editor_notes_*.md")
    ap.add_argument("--out_dir", required=True, type=str, help="output folder")

    ap.add_argument("--base_url", default="https://api.deepseek.com", type=str)
    ap.add_argument("--model", default="deepseek-reasoner", type=str)
    ap.add_argument("--api_key_env", default="DEEPSEEK_API_KEY", type=str)

    ap.add_argument("--temperature", default=0.0, type=float)
    ap.add_argument("--max_tokens", default=3200, type=int, help="per call output cap")
    ap.add_argument("--chunk_chars", default=20000, type=int, help="input chunk size")
    ap.add_argument("--force", action="store_true", help="force re-run even if progress exists")

    args = ap.parse_args()

    in_dir = Path(args.in_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    ensure_dir(out_dir)

    log_path = out_dir / "logs" / "summarize.log"
    state_path = out_dir / "state" / "progress.json"
    kb_path = out_dir / "chief_editor_kb_summary.md"

    ensure_dir(log_path.parent)
    ensure_dir(state_path.parent)

    api_key = os.getenv(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"Missing env var {args.api_key_env}. Do: export {args.api_key_env}='sk-...'")
    if api_key.lower().startswith("bearer "):
        api_key = api_key[7:].strip()

    client = OpenAI(api_key=api_key, base_url=args.base_url)

    files = list_note_files(in_dir)
    if not files:
        raise RuntimeError(f"No chief_editor_notes_*.md found in {in_dir}")

    progress = load_progress(state_path)
    done = set(progress.get("done", []))

    if args.force:
        done = set()
        progress = {"done": []}
        log_line(log_path, "FORCE enabled: reset progress")

    kb = read_text(kb_path).strip() if kb_path.exists() and not args.force else ""
    log_line(log_path, f"START in_dir={in_dir} files={len(files)} done={len(done)} model={args.model}")

    for fp in files:
        if fp.name in done:
            continue

        a, b = parse_range_from_filename(fp.name)
        label = f"{fp.name} ({a}-{b})"
        raw = read_text(fp).strip()
        if not raw:
            log_line(log_path, f"SKIP empty {label}")
            done.add(fp.name)
            progress["done"] = sorted(done)
            save_progress(state_path, progress)
            continue

        # chunk material for safety
        chunks = chunk_by_size(raw, max_chars=args.chunk_chars)
        log_line(log_path, f"PROCESS {label} chunks={len(chunks)}")

        for i, ck in enumerate(chunks, 1):
            sub_label = f"{label} part {i}/{len(chunks)}"
            user = KB_UPDATE_USER_TMPL.format(
                kb=kb if kb else "（空）",
                label=sub_label,
                text=ck
            )

            def _call():
                return llm_chat_once(client, args.model, KB_SYSTEM, user,
                                    temperature=args.temperature, max_tokens=args.max_tokens)

            draft = retry_call(_call, retries=4, sleep_base=1.5, log_path=log_path, label=f"KB update {sub_label}")
            draft = ensure_end_complete(
                client=client,
                model=args.model,
                system=KB_SYSTEM,
                base_user=user,
                draft=draft,
                temperature=args.temperature,
                max_tokens=min(args.max_tokens, 2200),
                log_path=log_path,
                label=f"KB ensure_end {sub_label}",
                end_marker="【END】",
                max_rounds=6
            )

            # remove END marker before next update to avoid growing noise
            kb = draft.replace("【END】", "").strip()
            write_text(kb_path, kb)

        done.add(fp.name)
        progress["done"] = sorted(done)
        save_progress(state_path, progress)
        log_line(log_path, f"DONE {label}")

    # Final: add END marker for final output file
    final_out = kb.strip()
    if "【主编知识库总纲｜通用网文写作】" not in final_out:
        # if somehow drifted, wrap it minimally
        final_out = "【主编知识库总纲｜通用网文写作】\n" + final_out
    final_out = final_out.strip() + "\n\n【END】\n"
    write_text(kb_path, final_out)

    log_line(log_path, f"FINISH out={kb_path}")
    print("✅ DONE")
    print(f"- Summary KB: {kb_path}")
    print(f"- Log: {log_path}")
    print(f"- Progress: {state_path}")


if __name__ == "__main__":
    main()
