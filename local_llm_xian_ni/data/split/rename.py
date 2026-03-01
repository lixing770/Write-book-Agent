#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
from pathlib import Path


def parseChapRange(name: str) -> tuple[int, int] | None:
    """
    Extract (start, end) from filenames containing:
      chap_2086-2088
    Also tolerates spaces around dash.
    """
    m = re.search(r"chap_(\d+)\s*-\s*(\d+)", name)
    if not m:
        return None
    start = int(m.group(1))
    end = int(m.group(2))
    if start <= 0 or end <= 0 or end < start:
        return None
    return start, end


def targetName(start: int, end: int) -> str:
    return f"{start}_{end}.txt"


def renameOneFile(file_path: Path, apply: bool) -> int:
    if not file_path.exists() or not file_path.is_file():
        raise SystemExit(f"File not found: {file_path}")

    parsed = parseChapRange(file_path.name)
    if not parsed:
        raise SystemExit(f"No 'chap_START-END' pattern found in filename: {file_path.name}")

    start, end = parsed
    dst = file_path.with_name(targetName(start, end))

    if dst.exists():
        print(f"[SKIP] target exists: {dst}")
        return 0

    if file_path.name == dst.name:
        print(f"[OK] already named: {file_path.name}")
        return 0

    print(f"{'[APPLY]' if apply else '[DRY]  '} {file_path.name} -> {dst.name}")

    if apply:
        file_path.rename(dst)
        print(f"[DONE] {dst}")
    else:
        print("Dry-run only. Re-run with --apply to perform rename.")

    return 0


def renameDir(dir_path: Path, apply: bool) -> int:
    if not dir_path.exists() or not dir_path.is_dir():
        raise SystemExit(f"Directory not found: {dir_path}")

    files = sorted([p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() == ".txt"])
    planned: list[tuple[Path, Path]] = []
    collisions = 0

    for p in files:
        parsed = parseChapRange(p.name)
        if not parsed:
            continue
        start, end = parsed
        dst = p.with_name(targetName(start, end))
        if p.name == dst.name:
            continue
        if dst.exists():
            collisions += 1
            print(f"[SKIP] target exists: {p.name} -> {dst.name}")
            continue
        planned.append((p, dst))

    if not planned:
        print("No matching files found in directory.")
        return 0

    print(f"Matched {len(planned)} file(s).")
    for src, dst in planned:
        print(f"{'[APPLY]' if apply else '[DRY]  '} {src.name} -> {dst.name}")

    if not apply:
        print("\nDry-run only. Re-run with --apply to perform renames.")
        return 0

    for src, dst in planned:
        src.rename(dst)

    print(f"\nDone. Renamed {len(planned)} file(s). Collisions skipped: {collisions}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Rename chap_START-END files to START_END.txt")
    ap.add_argument("--file", default=None, help="Single file path to rename.")
    ap.add_argument("--dir", default=None, help="Directory path to batch rename.")
    ap.add_argument("--apply", action="store_true", help="Actually rename. Default is dry-run.")
    args = ap.parse_args()

    if not args.file and not args.dir:
        raise SystemExit("Must provide --file or --dir")

    if args.file:
        return renameOneFile(Path(args.file).expanduser().resolve(), apply=args.apply)

    return renameDir(Path(args.dir).expanduser().resolve(), apply=args.apply)


if __name__ == "__main__":
    raise SystemExit(main())
