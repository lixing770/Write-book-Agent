#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


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


def _norm_name(s: str) -> str:
    s = (s or "").strip()
    # 去掉常见空白与括号内容
    s = s.replace("　", " ").strip()
    return s


def _merge_alias_map(entities: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Very simple alias resolution:
    - canonical = entity.name
    - map aliases -> canonical
    - if alias appears multiple times, keep first (you can improve later)
    """
    amap: Dict[str, str] = {}
    for e in entities:
        name = _norm_name(str(e.get("name", "")))
        if not name:
            continue
        if name not in amap:
            amap[name] = name
        aliases = e.get("aliases") or []
        if isinstance(aliases, list):
            for a in aliases:
                a = _norm_name(str(a))
                if a and a not in amap:
                    amap[a] = name
    return amap


def _canon(amap: Dict[str, str], name: str) -> str:
    name = _norm_name(name)
    return amap.get(name, name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extractions", required=True, help="out/extractions.jsonl")
    ap.add_argument("--out_dir", required=True, help="out/")
    args = ap.parse_args()

    in_path = Path(args.extractions)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_jsonl(in_path)

    # collect raw entities + relations
    all_entities: List[Dict[str, Any]] = []
    rel_rows: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []

    for r in rows:
        ex = r.get("extraction") or {}
        ents = ex.get("entities") or []
        rels = ex.get("relations") or []
        if isinstance(ents, list):
            for e in ents:
                if isinstance(e, dict):
                    e2 = dict(e)
                    e2["_chunk_id"] = r.get("chunk_id")
                    all_entities.append(e2)
        if isinstance(rels, list):
            for rel in rels:
                if isinstance(rel, dict):
                    rel_rows.append((r, rel))

    # build alias map (single-pass)
    alias_map = _merge_alias_map(all_entities)

    # build character registry
    char_map: Dict[str, Dict[str, Any]] = {}
    for e in all_entities:
        name = _canon(alias_map, str(e.get("name", "")))
        if not name:
            continue
        obj = char_map.get(name) or {
            "name": name,
            "aliases": set(),
            "notes": set(),
            "type": e.get("type", "person"),
            "evidence_refs": set(),
        }
        # aliases
        aliases = e.get("aliases") or []
        if isinstance(aliases, list):
            for a in aliases:
                a = _norm_name(str(a))
                if a and a != name:
                    obj["aliases"].add(a)
        # notes
        notes = e.get("notes")
        if isinstance(notes, str) and notes.strip():
            obj["notes"].add(notes.strip())
        # evidence refs
        cid = e.get("_chunk_id")
        if cid:
            obj["evidence_refs"].add(str(cid))
        char_map[name] = obj

    # normalize characters.yml
    characters = []
    for name, obj in sorted(char_map.items(), key=lambda x: x[0]):
        characters.append(
            {
                "name": name,
                "aliases": sorted(list(obj["aliases"]))[:50],
                "type": obj.get("type", "person"),
                "notes": "; ".join(sorted(list(obj["notes"]))[:20]),
                "evidence_refs": sorted(list(obj["evidence_refs"]))[:200],
            }
        )

    # relations: dedupe by (from,to,type,status) + aggregate evidence
    rel_key_map: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}

    evidence_jsonl_path = out_dir / "evidence.jsonl"
    ev_f = evidence_jsonl_path.open("w", encoding="utf-8")

    for meta, rel in rel_rows:
        a = _canon(alias_map, str(rel.get("from", "")))
        b = _canon(alias_map, str(rel.get("to", "")))
        t = _norm_name(str(rel.get("type", "其他")))
        status = _norm_name(str(rel.get("status", "不确定")))
        conf = rel.get("confidence", 0.5)
        try:
            conf = float(conf)
        except Exception:
            conf = 0.5

        ev = rel.get("evidence") or {}
        quote = _norm_name(str(ev.get("quote", "")))
        st = ev.get("start_char")
        ed = ev.get("end_char")

        key = (a, b, t, status)
        item = rel_key_map.get(key)
        if item is None:
            item = {
                "from": a,
                "to": b,
                "type": t,
                "status": status,
                "confidence": conf,
                "notes": _norm_name(str(rel.get("notes", ""))),
                "first_seen_chunk": meta.get("chunk_id"),
                "evidence": [],
            }
            rel_key_map[key] = item
        else:
            item["confidence"] = max(item["confidence"], conf)

        if quote:
            ev_obj = {
                "chunk_id": meta.get("chunk_id"),
                "chapter_title": meta.get("chapter_title"),
                "quote": quote[:120],
                "span": [st, ed],
            }
            item["evidence"].append(ev_obj)
            ev_f.write(json.dumps({"key": list(key), **ev_obj}, ensure_ascii=False) + "\n")

    ev_f.close()

    relations = []
    for _, item in sorted(rel_key_map.items(), key=lambda x: (x[1]["from"], x[1]["to"], x[1]["type"])):
        # cap evidence
        item["evidence"] = item["evidence"][:12]
        relations.append(item)

    # graph.json
    graph = {
        "nodes": [{"id": c["name"], "aliases": c["aliases"], "notes": c["notes"]} for c in characters],
        "edges": [
            {
                "source": r["from"],
                "target": r["to"],
                "type": r["type"],
                "status": r["status"],
                "confidence": r["confidence"],
            }
            for r in relations
        ],
    }

    # write outputs
    (out_dir / "characters.yml").write_text(yaml.safe_dump(characters, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (out_dir / "relations.yml").write_text(yaml.safe_dump(relations, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (out_dir / "graph.json").write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] characters -> {out_dir / 'characters.yml'}")
    print(f"[OK] relations   -> {out_dir / 'relations.yml'}")
    print(f"[OK] graph       -> {out_dir / 'graph.json'}")
    print(f"[OK] evidence    -> {evidence_jsonl_path}")


if __name__ == "__main__":
    main()
