#!/usr/bin/env python3
# Copyright (c) 2026 Ivan Tugay
# SPDX-License-Identifier: GPL-3.0-or-later
# Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

"""Lint docs/tasks.md claim registry (plan P0.9/P7.6, protocol AGENTS.md §6).

Fail (exit 1):
  - unknown status (must be `free` or `in progress`)
An empty table (header only) is valid when no plan tasks remain.
  - `in progress` row without agent or without RFC 3339 UTC `started_at`
  - `started_at` not parseable as %Y-%m-%dT%H:%M:%SZ
  - `free` row with non-empty agent/started cells
  - duplicate task IDs
  - AGENTS.md §2 claim-first wording missing (P11.3, marker below)
Warn (exit 0): claims older than 7 days (stale; ask human per AGENTS.md §3).

P11.3 note — why the "no tree edits without a claim" rule is only partly
lintable here: this script sees a single registry snapshot. It cannot see
the worktree diff, who authored it, or which claim (if any) an edit belongs
to, so detecting a source file modified without a matching in-progress
claim is impossible for a static registry lint; that enforcement belongs to
the claim process itself (AGENTS.md §2), not to a file-format check. What
IS mechanically checkable — and what this script enforces — is that the
normative claim-first wording is present in AGENTS.md §2. If the rule is
ever deleted or reworded past recognition, the lint fails closed until the
marker is restored and this script updated together. `--self-test` proves
both the checkable behavior and the documented boundary.

Usage: python3 scripts/lint-tasks.py [path/to/tasks.md] [--agents-path PATH]
       python3 scripts/lint-tasks.py --self-test
"""

import re
import sys
from datetime import datetime, timezone
from pathlib import Path

STALE_DAYS = 7
ROW = re.compile(r"^\|\s*((?:P\d+\.\d+|K\d+))\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|?\s*$")
TS_FMT = "%Y-%m-%dT%H:%M:%SZ"
# Stable marker for the AGENTS.md §2 claim-first rule (P11.3). Keep in sync
# with the §2 bullet; the lint matches this substring, not the full line.
CLAIM_FIRST_MARKER = "No tree edits without a claim row"
DEFAULT_AGENTS = Path(__file__).resolve().parent.parent / "AGENTS.md"
BAD_AGENTS_FIXTURE = (
    Path(__file__).resolve().parent.parent
    / "tests"
    / "fixtures"
    / "agents-without-claim-first.md"
)


def lint_registry(path: str) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    seen: set[str] = set()
    now = datetime.now(timezone.utc)

    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()

    rows = [ROW.match(l) for l in lines]
    rows = [m for m in rows if m]

    for m in rows:
        task, status, agent, started = (g.strip() for g in m.groups())
        if task in seen:
            errors.append(f"{task}: duplicate row")
        seen.add(task)

        if status == "free":
            if agent or started:
                errors.append(f"{task}: free row must have empty agent/started")
        elif status == "in progress":
            if not agent:
                errors.append(f"{task}: in-progress claim without agent")
            if not started:
                errors.append(f"{task}: in-progress claim without started_at")
            else:
                try:
                    ts = datetime.strptime(started, TS_FMT).replace(tzinfo=timezone.utc)
                except ValueError:
                    errors.append(f"{task}: started_at not RFC 3339 UTC ({started!r})")
                else:
                    age_days = (now - ts).total_seconds() / 86400
                    if age_days > STALE_DAYS:
                        warnings.append(
                            f"{task}: claim by {agent} is {age_days:.1f} days old — ask human"
                        )
        else:
            errors.append(f"{task}: unknown status {status!r}")

    return errors, warnings


def check_claim_first_wording(agents_path: Path) -> list[str]:
    """Fail closed when the AGENTS.md §2 claim-first rule is missing (P11.3)."""
    try:
        text = agents_path.read_text(encoding="utf-8")
    except OSError as e:
        return [f"claim-first rule: cannot read {agents_path} ({e})"]
    if CLAIM_FIRST_MARKER not in text:
        return [
            f"claim-first rule: {agents_path} lacks AGENTS.md §2 wording "
            f"{CLAIM_FIRST_MARKER!r} (P11.3)"
        ]
    return []


def main(tasks_path: str, agents_path: Path = DEFAULT_AGENTS) -> int:
    errors, warnings = lint_registry(tasks_path)
    errors.extend(check_claim_first_wording(agents_path))
    for w in warnings:
        print(f"warning: {w}")
    return fail(errors)


def fail(errors: list[str]) -> int:
    for e in errors:
        print(f"error: {e}")
    print(f"lint-tasks: {len(errors)} error(s)")
    return 1 if errors else 0


def self_test() -> int:
    """Prove the P11.3 behavior: claim-first check works, exit contract holds."""
    import subprocess
    import tempfile

    failures: list[str] = []

    def check(name: str, cond: bool) -> None:
        print(f"{'ok' if cond else 'FAIL'}: {name}")
        if not cond:
            failures.append(name)

    def run(*args: str) -> "subprocess.CompletedProcess[str]":
        return subprocess.run(
            [sys.executable, __file__, *args],
            capture_output=True,
            text=True,
        )

    with tempfile.TemporaryDirectory() as d:
        ok = Path(d) / "tasks-ok.md"
        ok.write_text(
            "| Task | Status | Agent | Started (UTC) |\n"
            "|---|---|---|---|\n"
            "| P11.3 | in progress | Test / model | 2026-09-16T05:19:54Z |\n"
            "| P1.1 | free | | |\n",
            encoding="utf-8",
        )
        bad = Path(d) / "tasks-bad.md"
        bad.write_text(
            "| Task | Status | Agent | Started (UTC) |\n"
            "|---|---|---|---|\n"
            "| P9.9 | in progress | | 2026-09-08T12:00:00Z |\n",
            encoding="utf-8",
        )
        stale = Path(d) / "tasks-stale.md"
        stale.write_text(
            "| Task | Status | Agent | Started (UTC) |\n"
            "|---|---|---|---|\n"
            "| P1.2 | in progress | Test / model | 2020-01-01T00:00:00Z |\n",
            encoding="utf-8",
        )

        r = run(str(ok))
        check("valid registry + real AGENTS.md exits 0", r.returncode == 0)
        r = run(str(bad))
        check(
            "nameless in-progress claim exits 1",
            r.returncode == 1 and "without agent" in r.stdout,
        )
        r = run(str(stale))
        check(
            "stale claim warns but exits 0 (exit contract)",
            r.returncode == 0 and "warning:" in r.stdout,
        )
        check(
            "bad AGENTS.md fixture is checked in",
            BAD_AGENTS_FIXTURE.is_file(),
        )
        r = run(str(ok), "--agents-path", str(BAD_AGENTS_FIXTURE))
        check(
            "AGENTS.md without claim-first wording exits 1",
            r.returncode == 1 and "claim-first rule" in r.stdout,
        )
        r = run(str(bad), "--agents-path", str(BAD_AGENTS_FIXTURE))
        check("registry + wording errors combine (exit 1)", r.returncode == 1)
        # Documented boundary (see docstring): with a valid table and the
        # wording present, the lint passes — it cannot know about worktree
        # edits, so unclaimed-edit detection stays a process rule, not a
        # lint result. This assertion pins that boundary.
        errors, _ = lint_registry(str(ok))
        check("valid table yields no registry errors", errors == [])

    print(f"self-test: {len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--self-test" in argv:
        sys.exit(self_test())
    agents = DEFAULT_AGENTS
    positional: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--agents-path="):
            agents = Path(a.split("=", 1)[1])
        elif a == "--agents-path" and i + 1 < len(argv):
            agents = Path(argv[i + 1])
            i += 1
        elif a.startswith("--"):
            print(f"error: unknown flag {a!r}")
            sys.exit(2)
        else:
            positional.append(a)
        i += 1
    sys.exit(main(positional[0] if positional else "docs/tasks.md", agents))
