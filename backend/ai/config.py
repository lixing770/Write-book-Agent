# backend/ai/config.py
from __future__ import annotations
import os
from pathlib import Path

from dotenv import load_dotenv

# 从项目根目录加载 .env（兼容你从任何目录运行）
PROJECT_ROOT = Path(__file__).resolve().parents[2]   # backend/ai -> project root
load_dotenv(PROJECT_ROOT / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY env var (check your .env or shell env)")
