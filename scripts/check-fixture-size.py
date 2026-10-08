#!/usr/bin/env python3
# Copyright (c) 2026 Ivan Tugay
# SPDX-License-Identifier: GPL-3.0-or-later
# Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

"""Fail when any file in tests/fixtures exceeds the size limit (default 3 GiB).

Usage:
  python3 scripts/check-fixture-size.py [--limit-bytes N]

Override: RUNA_FIXTURE_MAX_BYTES (bytes). Exit 0 when everything fits,
exit 1 listing every violator otherwise.
"""

import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
FIXTURES = os.path.join(ROOT, "tests", "fixtures")

DEFAULT_LIMIT_BYTES = 3 * 1024 * 1024 * 1024  # 3 GiB


def limit_bytes(argv):
    for arg in argv[1:]:
        if arg.startswith("--limit-bytes="):
            try:
                return int(arg.split("=", 1)[1])
            except ValueError:
                print(f"bad --limit-bytes value: {arg}", file=sys.stderr)
                sys.exit(2)
        else:
            print(f"unknown arg: {arg}", file=sys.stderr)
            sys.exit(2)
    try:
        return int(os.environ.get("RUNA_FIXTURE_MAX_BYTES", DEFAULT_LIMIT_BYTES))
    except ValueError:
        print("bad RUNA_FIXTURE_MAX_BYTES, want integer bytes", file=sys.stderr)
        sys.exit(2)


def main(argv):
    limit = limit_bytes(argv)
    if not os.path.isdir(FIXTURES):
        print(f"no fixtures dir at {FIXTURES}, ok")
        return 0
    violators = []
    biggest = ("", 0)
    checked = 0
    for dirpath, _dirnames, filenames in os.walk(FIXTURES):
        for name in filenames:
            path = os.path.join(dirpath, name)
            if not os.path.isfile(path):
                continue
            size = os.path.getsize(path)
            checked += 1
            if size > biggest[1]:
                biggest = (os.path.relpath(path, ROOT), size)
            if size > limit:
                violators.append((os.path.relpath(path, ROOT), size))
    print(f"checked {checked} file(s), limit {limit} bytes")
    if biggest[0]:
        print(f"largest: {biggest[0]} ({biggest[1]} bytes)")
    if violators:
        for path, size in sorted(violators):
            print(f"too large: {path} ({size} bytes > {limit})", file=sys.stderr)
        print("run `python3 scripts/clean-fixtures.py` to drop downloaded weights", file=sys.stderr)
        return 1
    print("all fixtures within the limit")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
