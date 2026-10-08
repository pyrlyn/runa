#!/usr/bin/env python3
# Copyright (c) 2026 Ivan Tugay
# SPDX-License-Identifier: GPL-3.0-or-later
# Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

"""Remove downloaded weights from tests/fixtures, keep hand-made files.

Downloaded weights are git-ignored (`tests/fixtures/*.gguf` in .gitignore);
hand-made fixtures stay: `synthetic-*.gguf` (tracked, header-only),
`audio/`, `video/`, `api/`, `shapes.png`, `mcp-echo.py`, docs.

Usage:
  python3 scripts/clean-fixtures.py [--dry-run] [--keep NAME,...]

Exits 0 even when there is nothing to clean.
"""

import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
FIXTURES = os.path.join(ROOT, "tests", "fixtures")

# Tracked, hand-made GGUFs: never delete.
PROTECTED_PREFIX = "synthetic-"


def parse_args(argv):
    dry_run = False
    keep = set()
    for arg in argv[1:]:
        if arg == "--dry-run":
            dry_run = True
        elif arg.startswith("--keep="):
            keep.update(n for n in arg.split("=", 1)[1].split(",") if n)
        else:
            print(f"unknown arg: {arg}", file=sys.stderr)
            return None, None
    return dry_run, keep


def main(argv):
    parsed = parse_args(argv)
    if parsed[0] is None:
        return 2
    dry_run, keep = parsed
    if not os.path.isdir(FIXTURES):
        print(f"no fixtures dir at {FIXTURES}, nothing to clean")
        return 0
    removed = 0
    freed = 0
    for name in sorted(os.listdir(FIXTURES)):
        if not name.endswith(".gguf"):
            continue
        if name.startswith(PROTECTED_PREFIX):
            continue
        if name in keep:
            continue
        path = os.path.join(FIXTURES, name)
        if not os.path.isfile(path):
            continue
        size = os.path.getsize(path)
        if dry_run:
            print(f"would remove {name} ({size} bytes)")
        else:
            os.remove(path)
            print(f"removed {name} ({size} bytes)")
        removed += 1
        freed += size
    print(f"{'would free' if dry_run else 'freed'} {freed} bytes in {removed} file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
