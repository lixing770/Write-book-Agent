#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from openai import OpenAI


@dataclass
class LLMConfig:
    model: str
    temperature: float = 0.2
    max_output_tokens: int = 1800
    retries: int = 4
    backoff_base_s: float = 0.8


class OpenAILLM:
    """
    Minimal OpenAI client wrapper using Responses API.

    Docs:
      - Responses API reference :contentReference[oaicite:1]{index=1}
      - Quickstart :contentReference[oaicite:2]{index=2}
    """

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        model = os.getenv("OPENAI_MODEL", "gpt-5.2")
        self.cfg = config or LLMConfig(model=model)
        self.client = OpenAI()

    def _safe_json(self, text: str) -> Dict[str, Any]:
        """
        Best-effort JSON extraction:
        - prefer pure JSON
        - otherwise extract first {...} block
        """
        text = (text or "").strip()
        if not text:
            return {"_error": "empty_response"}

        # direct
        try:
            obj = json.loads(text)
            return obj if isinstance(obj, dict) else {"_raw": obj}
        except Exception:
            pass

        # extract first {...}
        start = text.find("{")
        end = text.rfind("}")
        if 0 <= start < end:
            cand = text[start : end + 1]
            try:
                obj = json.loads(cand)
                return obj if isinstance(obj, dict) else {"_raw": obj}
            except Exception:
                return {"_error": "invalid_json", "_text": text[:2000]}

        return {"_error": "no_json_object", "_text": text[:2000]}

    def respond_text(self, *, system: str, user: str) -> str:
        """
        Returns plain text output.
        """
        last_err: Optional[Exception] = None
        for i in range(self.cfg.retries):
            try:
                resp = self.client.responses.create(
                    model=self.cfg.model,
                    input=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=self.cfg.temperature,
                    max_output_tokens=self.cfg.max_output_tokens,
                )
                # Python SDK exposes output_text as convenience in docs :contentReference[oaicite:3]{index=3}
                out = getattr(resp, "output_text", None)
                if out is None:
                    # fallback: best-effort stringify
                    out = str(resp)
                return str(out)
            except Exception as e:
                last_err = e
                time.sleep(self.cfg.backoff_base_s * (2**i))
        raise RuntimeError(f"OpenAI call failed after retries: {last_err}")

    def respond_json(self, *, system: str, user: str) -> Dict[str, Any]:
        """
        Asks model to output JSON only; then parses it robustly.
        """
        txt = self.respond_text(system=system, user=user)
        return self._safe_json(txt)
