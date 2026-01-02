from __future__ import annotations

import os
from pathlib import Path

# Load .env if present (backend/ai/.env or project root .env)
try:
    from dotenv import load_dotenv
    here = Path(__file__).resolve().parent
    load_dotenv(here / ".env", override=False)
    load_dotenv(Path.cwd() / ".env", override=False)
except Exception:
    pass

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY env var (check your .env or shell env)")
