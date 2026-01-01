# backend/ai/memory/retrieval.py
from __future__ import annotations
from pathlib import Path

def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")

def load_knowledge(base_dir: Path, genre: str, style: str, world: str) -> str:
    parts = []
    g = base_dir / "knowledge" / "genres" / f"{genre}.yml"
    s = base_dir / "knowledge" / "styles" / f"{style}.yml"
    w = base_dir / "knowledge" / "worlds" / f"{world}.yml"
    for p in (g, s, w):
        if p.exists():
            parts.append(f"### {p.name}\n{read_text(p)}")
    return "\n\n".join(parts).strip()
