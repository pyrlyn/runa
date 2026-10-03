#!/usr/bin/env python3
# Copyright (c) 2026 Ivan Tugay
# SPDX-License-Identifier: GPL-3.0-or-later
# Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

"""P5.8: fail if runa bench pp/tg dropped more than --max-drop (default 3 %).

Usage:
  perf-regress.py --current now.json --baseline docs/perf-baseline.json \\
      --key macos-14-cpu [--max-drop 0.03] [--write-missing]
  perf-regress.py --self-test
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path


def load_bench(path: Path) -> dict:
    text = path.read_text()
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{") and "pp_tok_s" in line:
            return json.loads(line)
    return json.loads(text)


def compare(cur: dict, table: dict, key: str, max_drop: float, write_missing: bool, baseline: Path) -> int:
    base = table.get(key)
    if base is None:
        print(f"no baseline for {key}; recording current as first report")
        if write_missing:
            table[key] = {
                "pp_tok_s": cur["pp_tok_s"],
                "tg_tok_s": cur["tg_tok_s"],
                "model": cur.get("model"),
                "placement": cur.get("placement"),
                "n_prompt": cur.get("n_prompt"),
                "n_gen": cur.get("n_gen"),
            }
            baseline.parent.mkdir(parents=True, exist_ok=True)
            baseline.write_text(json.dumps(table, indent=2) + "\n")
        return 0

    failed = False
    for metric in ("pp_tok_s", "tg_tok_s"):
        old = float(base[metric])
        new = float(cur[metric])
        if old <= 0:
            continue
        drop = (old - new) / old
        print(f"{metric}: {old:.3f} -> {new:.3f} (drop {drop * 100:.2f}%)")
        if drop > max_drop:
            print(
                f"FAIL: {metric} dropped {drop * 100:.2f}% > {max_drop * 100:.1f}%",
                file=sys.stderr,
            )
            failed = True
    return 1 if failed else 0


def self_test() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        current = tmp_path / "now.json"
        baseline = tmp_path / "base.json"
        sample = {
            "pp_tok_s": 100.0,
            "tg_tok_s": 50.0,
            "model": "m.gguf",
            "placement": "cpu",
            "n_prompt": 512,
            "n_gen": 128,
        }
        current.write_text(json.dumps(sample) + "\n")

        # missing key: first report, exit 0
        rc = compare(sample, {}, "macos-14-cpu", 0.03, False, baseline)
        assert rc == 0, rc

        # write-missing records
        table: dict = {}
        rc = compare(sample, table, "macos-14-cpu", 0.03, True, baseline)
        assert rc == 0, rc
        written = json.loads(baseline.read_text())
        assert written["macos-14-cpu"]["pp_tok_s"] == 100.0

        # 2 % drop: pass
        mild = dict(sample, pp_tok_s=98.0, tg_tok_s=49.0)
        rc = compare(mild, written, "macos-14-cpu", 0.03, False, baseline)
        assert rc == 0, rc

        # 10 % drop: fail
        bad = dict(sample, pp_tok_s=90.0, tg_tok_s=50.0)
        rc = compare(bad, written, "macos-14-cpu", 0.03, False, baseline)
        assert rc == 1, rc

    print("self-test ok")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--current", type=Path)
    p.add_argument("--baseline", type=Path)
    p.add_argument("--key")
    p.add_argument("--max-drop", type=float, default=0.03)
    p.add_argument("--write-missing", action="store_true")
    args = p.parse_args()

    if args.self_test:
        return self_test()
    if args.current is None or args.baseline is None or args.key is None:
        p.error("--current, --baseline, and --key are required (or pass --self-test)")

    cur = load_bench(args.current)
    table: dict = {}
    if args.baseline.is_file():
        table = json.loads(args.baseline.read_text())
    return compare(cur, table, args.key, args.max_drop, args.write_missing, args.baseline)


if __name__ == "__main__":
    raise SystemExit(main())
