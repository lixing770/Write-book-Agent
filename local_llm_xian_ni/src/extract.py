#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Set

from tqdm import tqdm

from .llm_client import OpenAILLM


DEFAULT_PROMPT = """你是中文长篇小说信息抽取专家。
你将从给定小说片段中抽取“人物实体”和“人物关系”，用于制作学习文档与关系图谱。

要求：只输出JSON（不要Markdown，不要解释）。
JSON schema:
{
  "entities":[
    {"name":"人物名","aliases":["别名1","别名2"],"type":"person","notes":"身份/特征/势力(若有)","confidence":0.0~1.0}
  ],
  "relations":[
    {"from":"人物A","to":"人物B","type":"师徒/亲友/敌对/同盟/交易/利用/情感/家族/上下级/其他",
     "status":"稳定/变化中/结束/不确定",
     "evidence":{"quote":"原文证据(<=80字)","start_char":123,"end_char":456},
     "confidence":0.0~1.0,
     "notes":"为什么判定为该关系(<=40字)"}
  ],
  "events":[
    {"summary":"触发关系变化的事件(<=50字)","involved":["A","B"],"confidence":0.0~1.0}
  ]
}

注意：
1) 只抽取片段中“明确出现/强烈暗示”的人物与关系；不要脑补。
2) 人物可能有别名/称呼；尽量写入aliases。
3) evidence.start_char/end_char 指相对本chunk的字符位置（从0开始）。
"""

SYSTEM = "You extract structured knowledge from Chinese novel text. Output JSON only."


def _read_jsonl(p: Path) -> List[Dict[str, Any]]:
    if not p.exists():
        raise RuntimeError(f"Missing: {p}")
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _existing_chunk_ids(out_path: Path) -> Set[str]:
    if not out_path.exists():
        return set()
    s: Set[str] = set()
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                cid = obj.get("chunk_id")
                if cid:
                    s.add(str(cid))
            except Exception:
                continue
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True, help="out/chunks.jsonl")
    ap.add_argument("--out", required=True, help="out/extractions.jsonl")
    ap.add_argument("--prompt", default="prompts/extract_chunk.md")
    ap.add_argument("--workers", type=int, default=1, help="keep 1 for safety; you can parallelize later")
    args = ap.parse_args()

    chunks_path = Path(args.chunks)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prompt_path = Path(args.prompt)
    user_prompt = prompt_path.read_text(encoding="utf-8").strip() if prompt_path.exists() else DEFAULT_PROMPT

    rows = _read_jsonl(chunks_path)
    done = _existing_chunk_ids(out_path)

    llm = OpenAILLM()

    with out_path.open("a", encoding="utf-8") as fout:
        for r in tqdm(rows, desc="extract"):
            cid = str(r["chunk_id"])
            if cid in done:
                continue

            text = r["text"]
            user = (
                user_prompt
                + "\n\n"
                + "=== META ===\n"
                + f"chunk_id: {cid}\n"
                + f"chapter: {r.get('chapter_title','')}\n"
                + f"part: {r.get('part_index','')}\n"
                + "=== TEXT ===\n"
                + text
            )

            obj = llm.respond_json(system=SYSTEM, user=user)

            # attach meta
            out_obj = {
                "chunk_id": cid,
                "chapter_index": r.get("chapter_index"),
                "chapter_title": r.get("chapter_title"),
                "part_index": r.get("part_index"),
                "start_char": r.get("start_char"),
                "end_char": r.get("end_char"),
                "extraction": obj,
            }

            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"[OK] extractions -> {out_path}")


if __name__ == "__main__":
    main()
