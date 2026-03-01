#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import httpx


# =========================
# 固定路径配置
# =========================
SPLIT_DIR = Path(
    "/Users/50pai/Desktop/Writing book agent/local_llm_xian_ni/data/split"
)
OUTPUT_ROOT = Path(
    "/Users/50pai/Desktop/Writing book agent/local_llm_xian_ni/data/summary_all"
)

FACTS_DIR = OUTPUT_ROOT / "facts_txt"
MASTER_DIR = OUTPUT_ROOT / "master_txt"

START_FROM_NAME = "266_275.txt"   # <<< 从这里开始跑（包含）


# =========================
# 工具函数
# =========================
def ensureDir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def readText(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return p.read_text(encoding="utf-8-sig")


def naturalKey(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def listTxtFiles(folder: Path) -> List[Path]:
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix == ".txt"]
    files.sort(key=lambda p: naturalKey(p.name))
    return files


# =========================
# 章节切分
# =========================
def splitChapters(text: str) -> List[str]:
    starts = [m.start() for m in re.finditer(r"第\s*\d+\s*章", text)]
    if not starts:
        return [text.strip()]

    chapters = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(text)
        chunk = text[s:e].strip()
        if chunk:
            chapters.append(chunk)
    return chapters


# =========================
# Prompt
# =========================
def buildPrompt(chapters: List[str], filename: str) -> str:
    first8 = "\n\n".join(chapters[:8])
    last2 = "\n\n".join(chapters[8:10])

    return f"""你是“长篇小说知识库构建器”。请严格按区块输出纯文本：

[FILE]
{filename}

[SUMMARY200]
两段中文，总字数180-220字：
第一段（20%）：概括前8章
第二段（80%）：详细概括后2章，突出冲突与承接点

[CHARACTERS_DELTA]
- 人物名：状态/目标/变化

[RELATIONS_DELTA]
- A -> B：关系变化（原因；影响）

[EVENTS]
- E1：起因｜行动｜结果｜影响
- E2：...

[SETTINGS]
新设定/规则/物品/地点（无则写“无新增”）

[THREADS]
未回收悬念/伏笔（无则写“无”）

前8章内容：
{first8}

后2章内容：
{last2}
"""


# =========================
# DeepSeek 客户端
# =========================
@dataclass
class DeepSeekClient:
    model: str = "deepseek-chat"
    temperature: float = 0.2
    maxTokens: int = 1200
    timeout: float = 90.0
    retries: int = 3

    def apiKey(self) -> str:
        key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not key:
            raise RuntimeError("DEEPSEEK_API_KEY 未设置")
        return key

    def chat(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "严格遵守区块格式输出"},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.maxTokens,
        }

        headers = {
            "Authorization": f"Bearer {self.apiKey()}",
            "Content-Type": "application/json",
        }

        last_err = None
        with httpx.Client(timeout=self.timeout) as client:
            for i in range(self.retries):
                try:
                    r = client.post(
                        "https://api.deepseek.com/chat/completions",
                        json=payload,
                        headers=headers,
                    )
                    r.raise_for_status()
                    data = r.json()
                    return data["choices"][0]["message"]["content"].strip()
                except Exception as e:
                    last_err = e
                    time.sleep(1.5 * (i + 1))

        raise RuntimeError(last_err)


# =========================
# 主流程
# =========================
def main():
    ensureDir(FACTS_DIR)
    ensureDir(MASTER_DIR)

    client = DeepSeekClient()
    files = listTxtFiles(SPLIT_DIR)

    started = False
    START_FROM_NAME = "266_275.txt"
    facts_files: List[Path] = []

    for p in files:
        # ===== 关键：从 266_275.txt 开始 =====
        if not started:
            if p.name == START_FROM_NAME:
                started = True
            else:
                continue
        # ===================================

        print(f"[RUN] {p.name}")

        try:
            raw = readText(p)
            chapters = splitChapters(raw)
            if len(chapters) < 10:
                print(f"[SKIP] 章节不足 10：{p.name}")
                continue

            prompt = buildPrompt(chapters[:10], p.name)
            result = client.chat(prompt)

            out_path = FACTS_DIR / f"{p.stem}_facts.txt"
            out_path.write_text(result + "\n", encoding="utf-8")
            facts_files.append(out_path)

            print(f"[OK]  -> {out_path.name}")

        except Exception as e:
            print(f"[FAIL] {p.name} : {e}", file=sys.stderr)
            print("⛔ 遇到错误，立即停止，防止烧钱", file=sys.stderr)
            break

    # ===== 汇总 =====
    if not facts_files:
        print("没有生成任何 facts 文件")
        return

    all_facts = []
    relations = []

    for fp in FACTS_DIR.glob("*_facts.txt"):
        txt = readText(fp).strip()
        if not txt:
            continue
        all_facts.append(txt)

        def grab(block):
            m = re.search(
                rf"\[{block}\]\n(.*?)(?=\n\[[A-Z_]+\]|\Z)",
                txt,
                re.S,
            )
            return m.group(1).strip() if m else ""

        relations.append(
            f"[SOURCE]\n{fp.name}\n\n"
            f"[CHARACTERS_DELTA]\n{grab('CHARACTERS_DELTA')}\n\n"
            f"[RELATIONS_DELTA]\n{grab('RELATIONS_DELTA')}"
        )

    (MASTER_DIR / "ALL_FACTS.txt").write_text(
        "\n\n".join(all_facts), encoding="utf-8"
    )
    (MASTER_DIR / "CHARACTER_RELATIONS_MASTER.txt").write_text(
        "\n\n".join(relations), encoding="utf-8"
    )

    print("\n✅ 汇总完成")
    print(" - ALL_FACTS.txt")
    print(" - CHARACTER_RELATIONS_MASTER.txt")


if __name__ == "__main__":
    main()
