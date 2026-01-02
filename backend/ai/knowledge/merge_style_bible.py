#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles_dir", default="data/knowledge/style_profiles")
    ap.add_argument("--out", default="data/knowledge/style_bible.json")
    args = ap.parse_args()

    d = Path(args.profiles_dir)
    out = Path(args.out)
    files = sorted(d.glob("*.json"))
    if not files:
        raise SystemExit(f"No profiles found in {d}")

    profiles: List[Dict[str, Any]] = []
    for f in files:
        profiles.append(json.loads(f.read_text(encoding="utf-8")))

    # Very simple merge strategy:
    # - sources: list
    # - for list fields: union
    # - for dict fields: keep as list of dicts under a key (avoid incorrect overwrite)
    merged: Dict[str, Any] = {
        "sources": [p.get("source", f.stem) for p, f in zip(profiles, files)],
        "profiles": profiles
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] style_bible saved: {out}")


if __name__ == "__main__":
    main()
