#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

mkdir -p out

echo "[1/4] chunking..."
python -m src.chunker --input data/仙逆.txt --out out/chunks.jsonl --max_chapters 100 --max_chars 2800 --overlap 200


echo "[2/4] extracting (OpenAI)..."
python -m src.extract --chunks out/chunks.jsonl --out out/extractions.jsonl --prompt prompts/extract_chunk.md --workers 1

echo "[3/4] merging..."
python -m src.merge --extractions out/extractions.jsonl --out_dir out

echo "[4/4] building docs..."
python -m src.build_docs --out_dir out --prompt prompts/write_study_notes.md

echo "DONE. See: out/study_notes.md / out/characters.yml / out/relations.yml"
