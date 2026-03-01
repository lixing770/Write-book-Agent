#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import httpx


SCRIPT_DIR = Path(__file__).resolve().parent

# ✅ 你指定的输出根目录
OUTPUT_ROOT_DEFAULT = Path(
    "/Users/50pai/Desktop/Writing book agent/local_llm_xian_ni/data/summary_all"
)

# ✅ 你 split 的默认位置（你若不同就运行时传 --split_dir）
SPLIT_DIR_DEFAULT = Path(
    "/Users/50pai/Desktop/Writing book agent/local_llm_xian_ni/data/split"
)


def readText(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8-sig")


def ensureDir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def naturalKey(s: str) -> List:
    # sort like part_2 before part_10
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def listTxtFiles(folder: Path) -> List[Path]:
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".txt"]
    files.sort(key=lambda p: naturalKey(p.name))
    return files


def findChapterStarts(text: str) -> List[int]:
    pattern = re.compile(r"(?:^|\n).*?(第\s*\d{1,5}\s*章)", re.MULTILINE)
    starts: List[int] = []
    for m in pattern.finditer(text):
        starts.append(m.start())
    return sorted(set(starts))


def splitChapters(text: str) -> List[str]:
    starts = findChapterStarts(text)
    if len(starts) < 2:
        pattern2 = re.compile(r"第\s*\d{1,5}\s*章")
        starts = sorted(set([m.start() for m in pattern2.finditer(text)]))

    if not starts:
        return [text.strip()]

    chapters: List[str] = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(text)
        chunk = text[s:e].strip()
        if chunk:
            chapters.append(chunk)
    return chapters


def clampTenChapters(chapters: List[str]) -> List[str]:
    return chapters[:10] if len(chapters) >= 10 else chapters


def buildDevNextPromptForTxtOutput(chapters10: List[str], filename: str) -> str:
    first8 = "\n\n".join(chapters10[:8])
    last2 = "\n\n".join(chapters10[8:10])

    prompt = f"""你是“长篇小说知识库构建器”。请只输出纯文本，严格按下面区块格式输出（区块标题必须一字不差）：

[FILE]
{filename}

[SUMMARY200]
要求：总长度约200字（180-220），只分两段：
- 第一段：概括前8章（信息量约20%，更短更概括）
- 第二段：概括后2章（信息量约80%，更长更细，强调可承接后续的钩子/冲突/状态变化/关键设定）
禁止：引用原文句子、出现“第X章/本章”等字样、项目符号。

[CHARACTERS_DELTA]
列出本批10章中“人物新增/状态变化/动机变化/能力变化”，用短句即可。格式：
- 人物名：变化/状态/目标/关键动作

[RELATIONS_DELTA]
列出本批10章中“人物关系变化”，格式：
- A -> B：关系变化（原因：...；影响：...）

[EVENTS]
列出5-12条关键事件，用短句。格式：
- E1：起因...｜行动...｜结果...｜影响...
（必须体现因果，不要流水账）

[SETTINGS]
列出新出现或被强调的设定/规则/物品/地点（若无写“无新增”）。

[THREADS]
列出未解决悬念/伏笔/待回收点（若无写“无”）。

只输出以上区块，不要额外解释。

以下是前8章内容：
{first8}

以下是后2章内容：
{last2}
"""
    return prompt


@dataclass
class DeepSeekClient:
    model: str = "deepseek-chat"
    temperature: float = 0.2
    maxTokens: int = 1200
    timeoutSeconds: float = 90.0
    maxRetries: int = 3
    baseUrl: str = "https://api.deepseek.com"

    def getApiKey(self, cliKey: Optional[str]) -> str:
        if cliKey and cliKey.strip():
            return cliKey.strip()
        envKey = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if envKey:
            return envKey
        raise RuntimeError(
            "Missing DeepSeek API key.\n"
            "Use env:\n"
            "  export DEEPSEEK_API_KEY='YOUR_KEY'\n"
            "or pass:\n"
            "  --api_key YOUR_KEY"
        )

    def createHttpClient(self) -> httpx.Client:
        return httpx.Client(
            base_url=self.baseUrl,
            timeout=httpx.Timeout(self.timeoutSeconds),
            headers={"Content-Type": "application/json"},
        )

    def chat(self, apiKey: str, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "你必须严格按用户要求的区块格式输出。"},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.maxTokens,
        }

        lastErr: Optional[Exception] = None
        with self.createHttpClient() as client:
            for attempt in range(1, self.maxRetries + 1):
                try:
                    resp = client.post(
                        "/chat/completions",
                        json=payload,
                        headers={"Authorization": f"Bearer {apiKey}"},
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    content = (
                        data.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                    )
                    if not isinstance(content, str) or not content.strip():
                        raise RuntimeError(f"Empty response content: {data}")
                    return content.strip()
                except Exception as e:
                    lastErr = e
                    if attempt < self.maxRetries:
                        time.sleep(1.2 * attempt)
                        continue
                    break

        raise RuntimeError(f"DeepSeek request failed after retries: {lastErr}")


def extractBlock(text: str, blockName: str) -> str:
    pattern = re.compile(
        rf"\[{re.escape(blockName)}\]\s*\n(.*?)(?=\n\[[A-Z_]+\]\s*\n|\Z)",
        re.DOTALL,
    )
    m = pattern.search(text)
    if not m:
        return ""
    return m.group(1).strip()


def runOneFile(
    client: DeepSeekClient,
    apiKey: str,
    inputPath: Path,
    outFactsDir: Path,
) -> Path:
    raw = readText(inputPath)
    chapters = clampTenChapters(splitChapters(raw))

    if len(chapters) < 10:
        raise RuntimeError(
            f"Expected 10 chapters in file, but detected {len(chapters)}: {inputPath.name}"
        )

    prompt = buildDevNextPromptForTxtOutput(chapters, filename=inputPath.name)
    result = client.chat(apiKey=apiKey, prompt=prompt)

    ensureDir(outFactsDir)
    outPath = outFactsDir / f"{inputPath.stem}_facts.txt"
    outPath.write_text(result + "\n", encoding="utf-8")
    return outPath


def buildMasterFiles(factsFiles: List[Path], outDir: Path) -> Tuple[Path, Path]:
    ensureDir(outDir)

    allFactsPath = outDir / "ALL_FACTS.txt"
    masterCRPath = outDir / "CHARACTER_RELATIONS_MASTER.txt"

    allParts: List[str] = []
    crParts: List[str] = []

    for fp in factsFiles:
        txt = readText(fp).strip()
        if not txt:
            continue
        allParts.append(txt)

        fileBlock = extractBlock(txt, "FILE").strip()
        chars = extractBlock(txt, "CHARACTERS_DELTA").strip()
        rels = extractBlock(txt, "RELATIONS_DELTA").strip()

        cr = []
        cr.append(f"[SOURCE]\n{fp.name}")
        if fileBlock:
            cr.append(f"[FILE]\n{fileBlock}")
        cr.append("[CHARACTERS_DELTA]")
        cr.append(chars if chars else "（缺失）")
        cr.append("\n[RELATIONS_DELTA]")
        cr.append(rels if rels else "（缺失）")
        crParts.append("\n".join(cr).strip())

    allFactsPath.write_text("\n\n".join(allParts) + "\n", encoding="utf-8")
    masterCRPath.write_text("\n\n".join(crParts) + "\n", encoding="utf-8")
    return allFactsPath, masterCRPath


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Summarize all split/*.txt into txt facts, then build master txt files."
    )
    ap.add_argument(
        "--split_dir",
        default=str(SPLIT_DIR_DEFAULT),
        help="Folder containing split txt files (each file should contain 10 chapters).",
    )
    ap.add_argument(
        "--out_root",
        default=str(OUTPUT_ROOT_DEFAULT),
        help="Output root folder. Will create subfolders inside.",
    )
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--api_key", default=None, help="DeepSeek API key (or env DEEPSEEK_API_KEY).")
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max_tokens", type=int, default=1200)
    args = ap.parse_args()

    splitDir = Path(args.split_dir).expanduser()
    if not splitDir.exists():
        raise SystemExit(f"split_dir not found: {splitDir}")

    outRoot = Path(args.out_root).expanduser()
    outFactsDir = outRoot / "facts_txt"
    outMasterDir = outRoot / "master_txt"

    client = DeepSeekClient(
        model=args.model,
        timeoutSeconds=float(args.timeout),
        maxRetries=int(args.retries),
        temperature=float(args.temperature),
        maxTokens=int(args.max_tokens),
    )
    apiKey = client.getApiKey(args.api_key)

    inputs = listTxtFiles(splitDir)
    if not inputs:
        raise SystemExit(f"No .txt files found in: {splitDir}")

    ensureDir(outFactsDir)
    ensureDir(outMasterDir)

    factsFiles: List[Path] = []
    for p in inputs:
        try:
            out = runOneFile(client=client, apiKey=apiKey, inputPath=p, outFactsDir=outFactsDir)
            print(f"[OK] {p.name} -> {out.name}")
            factsFiles.append(out)
        except Exception as e:
            print(f"[FAIL] {p.name}: {e}", file=sys.stderr)

    if not factsFiles:
        raise SystemExit("No facts files generated. Check errors above.")

    allFactsPath, masterCRPath = buildMasterFiles(factsFiles=factsFiles, outDir=outMasterDir)
    print(f"\n[MASTER] {allFactsPath}")
    print(f"[MASTER] {masterCRPath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
