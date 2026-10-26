#!/usr/bin/env python3
# Copyright (c) 2026 Ivan Tugay
# SPDX-License-Identifier: GPL-3.0-or-later
# Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

"""Fail when the module graph of a crate has a cycle (issue #31).

A cycle means no module in it can move to another crate on its own, and a
module that calls back into the crate root ties the whole library to it.

Nodes: the crate root and every module another module can name. A private
child of a non-root module (`mod infill;` inside `serve/mod.rs`) is an
implementation split of its parent and merges into the parent's node; the
crate root's children are visible crate-wide, so each is its own node.

Edges:
  - parent -> declared child (`mod x;`): a child that reaches back into its
    parent, or a module that reaches into the root, closes a cycle;
  - every `crate::`/`super::`/`self::` path, `use` groups included, to the
    module that owns the item. Root-level module re-exports
    (`pub(crate) use cli::config;`, `use daemon::proto as daemon_proto;`)
    resolve to the module they name, so the alias is not a root reference.

Comments, string and char literals are skipped. `super::` inside an inline
`mod tests { .. }` resolves to the file's own module, so the usual
`use super::*;` does not count.

Usage: python3 scripts/check-module-cycles.py [crate/src ...] [--self-test]
Default: crates/runa/src. Exit 1 when any cycle exists.
"""

from __future__ import annotations

import io
import re
import sys
import tempfile
from pathlib import Path

DEFAULT_ROOTS = ["crates/runa/src"]
ROOT = "crate"

MOD_DECL = re.compile(
    r"(?<![\w])(pub(?:\s*\([^)]*\))?\s+)?mod\s+([A-Za-z_]\w*)\s*([;{])"
)
PATH_HEAD = re.compile(
    r"(?<![\w$:])(crate|self|super)\b((?:\s*::\s*[A-Za-z_]\w*)*)(\s*::\s*\{|\s*::\s*\*)?"
)
RAW_STR = re.compile(r'b?r(#*)"')
CHAR_LIT = re.compile(r"'(\\(?:u\{[0-9A-Fa-f]+\}|x[0-9A-Fa-f]{2}|.)|[^\\'])'")
IDENT_PATH = re.compile(r"\s*([A-Za-z_]\w*(?:\s*::\s*[A-Za-z_]\w*)*)(\s*::\s*\{)?")


def strip(src: str) -> str:
    """Blank out comments and string/char literals, keeping offsets and newlines."""
    out = list(src)
    i, n = 0, len(src)

    def blank(a: int, b: int) -> None:
        for k in range(a, min(b, n)):
            if out[k] != "\n":
                out[k] = " "

    while i < n:
        c = src[i]
        if src.startswith("//", i):
            j = src.find("\n", i)
            j = n if j < 0 else j
            blank(i, j)
            i = j
        elif src.startswith("/*", i):
            depth, j = 1, i + 2
            while j < n and depth:
                if src.startswith("/*", j):
                    depth, j = depth + 1, j + 2
                elif src.startswith("*/", j):
                    depth, j = depth - 1, j + 2
                else:
                    j += 1
            blank(i, j)
            i = j
        elif (m := RAW_STR.match(src, i)) and (i == 0 or not (src[i - 1].isalnum() or src[i - 1] == "_")):
            end = src.find('"' + m.group(1), m.end())
            j = n if end < 0 else end + 1 + len(m.group(1))
            blank(i, j)
            i = j
        elif c == '"':
            j = i + 1
            while j < n and src[j] != '"':
                j += 2 if src[j] == "\\" else 1
            blank(i, j + 1)
            i = j + 1
        elif c == "'":
            # A char literal closes within a few chars; a lifetime does not.
            m = CHAR_LIT.match(src, i)
            if m:
                blank(i, m.end())
                i = m.end()
            else:
                i += 1
        else:
            i += 1
    return "".join(out)


def module_of(rel: Path) -> tuple[str, ...]:
    parts = list(rel.with_suffix("").parts)
    if parts in (["lib"], ["main"]):
        return ()
    if parts[-1] == "mod":
        parts.pop()
    return tuple(parts)


def split_group(text: str, start: int) -> tuple[list[str], int]:
    """Top-level comma-separated entries of the `{...}` opening at `start`."""
    depth, j, last, items = 0, start, start + 1, []
    while j < len(text):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                items.append(text[last:j])
                return items, j + 1
        elif text[j] == "," and depth == 1:
            items.append(text[last:j])
            last = j + 1
        j += 1
    items.append(text[last:])
    return items, len(text)


def segs(path: str) -> list[str]:
    return [s for s in re.split(r"\s*::\s*", path.strip()) if s]


def expand(prefix: list[str], entry: str) -> list[list[str]]:
    """Full paths named by one `use` group entry (nested groups included)."""
    m = IDENT_PATH.match(entry)
    if not m:
        return [prefix]
    head = prefix + segs(m.group(1))
    if not m.group(2):
        return [head]
    items, _ = split_group(entry, entry.index("{", m.end(1)))
    paths: list[list[str]] = []
    for item in items:
        paths.extend(expand(head, item))
    return paths or [head]


class Crate:
    def __init__(self, src_dir: Path):
        self.src_dir = src_dir
        self.files: dict[tuple[str, ...], tuple[Path, str]] = {}
        for path in sorted(src_dir.rglob("*.rs")):
            rel = path.relative_to(src_dir)
            if rel.parts == ("main.rs",) or "bin" in rel.parts[:1]:
                continue  # separate crate roots: they reach the library as `runa::`
            self.files[module_of(rel)] = (path, strip(path.read_text(encoding="utf-8")))
        self.public: set[tuple[str, ...]] = {()}
        self.children: dict[tuple[str, ...], list[tuple[str, ...]]] = {}
        for mod, (_, text) in self.files.items():
            for m in MOD_DECL.finditer(text):
                child = mod + (m.group(2),)
                if m.group(3) == ";" and child in self.files:
                    self.children.setdefault(mod, []).append(child)
                    if mod == () or m.group(1):
                        self.public.add(child)
        self.aliases = self._root_aliases()

    def node(self, mod: tuple[str, ...]) -> tuple[str, ...]:
        """The visible module owning `mod` (private descendants merge up)."""
        while mod not in self.public:
            mod = mod[:-1]
        return mod

    def _root_aliases(self) -> dict[str, list[str]]:
        aliases: dict[str, list[str]] = {}
        root = self.files.get(())
        if not root:
            return aliases
        for m in re.finditer(r"\buse\s+([^;{]+?)(?:\s+as\s+([A-Za-z_]\w*))?\s*;", root[1]):
            target = [s for s in segs(m.group(1)) if s != "crate"]
            if tuple(target) in self.files:
                aliases[m.group(2) or target[-1]] = target
        return aliases

    def resolve(self, segments: list[str]) -> tuple[str, ...]:
        if segments and segments[0] in self.aliases:
            segments = self.aliases[segments[0]] + segments[1:]
        best: tuple[str, ...] = ()
        for k in range(1, len(segments) + 1):
            if tuple(segments[:k]) in self.files:
                best = tuple(segments[:k])
        return self.node(best)

    def edges(self) -> dict[tuple[str, ...], dict[tuple[str, ...], str]]:
        """node -> {target node: first `file:line` that creates the edge}."""
        graph: dict[tuple[str, ...], dict[tuple[str, ...], str]] = {
            self.node(m): {} for m in self.files
        }
        for parent, kids in self.children.items():
            for kid in kids:
                a, b = self.node(parent), self.node(kid)
                if a != b:
                    path = self.files[parent][0]
                    graph[a].setdefault(b, f"{path}: mod {kid[-1]}")
        for mod, (path, text) in self.files.items():
            src = self.node(mod)
            inline = self._inline_spans(text)
            for m in PATH_HEAD.finditer(text):
                if not (m.group(2) or m.group(3)):
                    continue  # bare `self`, `pub(super)`: not a path into a module
                here = list(mod) + [n for a, b, n in inline if a <= m.start() < b]
                for full in self._paths(text, m, here):
                    dst = self.resolve(full)
                    if dst != src:
                        line = text.count("\n", 0, m.start()) + 1
                        graph[src].setdefault(dst, f"{path}:{line}")
        return graph

    @staticmethod
    def _inline_spans(text: str) -> list[tuple[int, int, str]]:
        """Body spans of the inline `mod x { .. }` blocks in a file."""
        spans: list[tuple[int, int, str]] = []
        for m in MOD_DECL.finditer(text):
            if m.group(3) != "{":
                continue
            depth, j = 0, m.end() - 1
            while j < len(text):
                depth += {"{": 1, "}": -1}.get(text[j], 0)
                if depth == 0:
                    break
                j += 1
            spans.append((m.end(), j, m.group(2)))
        return spans

    def _paths(self, text: str, m: re.Match, here: list[str]) -> list[list[str]]:
        rel = [m.group(1)] + segs(m.group(2))
        base = [] if rel[0] == "crate" else list(here)
        k = 0 if rel[0] == "super" else 1
        while k < len(rel) and rel[k] == "super":
            base = base[:-1]
            k += 1
        prefix = base + rel[k:]
        if not m.group(3) or "{" not in m.group(3):
            return [prefix]
        items, _ = split_group(text, m.end() - 1)
        paths: list[list[str]] = []
        for item in items:
            paths.extend(expand(prefix, item))
        return paths or [prefix]


def cycles(graph: dict) -> list[list]:
    """Strongly connected components with more than one node (Tarjan)."""
    index: dict = {}
    low: dict = {}
    stack: list = []
    on: set = set()
    out: list[list] = []
    counter = [0]

    def visit(v) -> None:
        index[v] = low[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        on.add(v)
        for w in graph.get(v, {}):
            if w not in index:
                visit(w)
                low[v] = min(low[v], low[w])
            elif w in on:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp = []
            while True:
                w = stack.pop()
                on.discard(w)
                comp.append(w)
                if w == v:
                    break
            if len(comp) > 1:
                out.append(sorted(comp))

    sys.setrecursionlimit(max(1000, 4 * len(graph)))
    for v in sorted(graph):
        if v not in index:
            visit(v)
    return out


def name(mod: tuple[str, ...]) -> str:
    return "::".join((ROOT,) + mod)


def check(src_dir: Path, out=sys.stdout) -> int:
    graph = Crate(src_dir).edges()
    found = cycles(graph)
    for comp in found:
        members = set(comp)
        print(f"{src_dir}: module cycle: {', '.join(name(m) for m in comp)}", file=out)
        for a in comp:
            for b, where in sorted(graph[a].items()):
                if b in members:
                    print(f"  {name(a)} -> {name(b)}  ({where})", file=out)
    if not found:
        print(f"{src_dir}: {len(graph)} modules, no cycles", file=out)
    return 1 if found else 0


def self_test() -> int:
    cases = {
        "root item back-reference": (
            {"lib.rs": "mod a;\npub fn helper() {}\n", "a.rs": "fn f() { crate::helper(); }\n"},
            1,
        ),
        "two-module cycle via use group": (
            {
                "lib.rs": "mod a;\nmod b;\n",
                "a.rs": "use crate::{b::B, x};\npub struct A;\n",
                "b.rs": "use crate::a::A;\npub struct B;\n",
            },
            1,
        ),
        "visible child calling its parent": (
            {
                "lib.rs": "mod cli;\npub(crate) use cli::config;\n",
                "cli/mod.rs": "pub(crate) mod config;\npub(crate) fn run() {}\n",
                "cli/config.rs": "fn f() { super::run(); }\n",
            },
            1,
        ),
        "acyclic: aliases, private child, inline tests, comments, strings": (
            {
                "lib.rs": "mod cli;\nmod mcp;\nmod serve;\npub(crate) use cli::config;\n",
                "cli/mod.rs": "pub(crate) mod config;\nuse crate::mcp;\nuse crate::serve;\n",
                "cli/config.rs": (
                    "// crate::mcp::Gone\n"
                    "/* crate::serve */\n"
                    "const S: &str = \"crate::mcp\";\n"
                    "const R: &str = r#\"crate::serve\"#;\n"
                    "fn g<'a>(c: char) -> bool { c == '\"' }\n"
                    "pub struct McpServer;\n"
                    "#[cfg(test)]\nmod tests { use super::*; }\n"
                ),
                "mcp.rs": "use crate::config::McpServer;\n",
                "serve/mod.rs": "mod infill;\nuse crate::config;\npub(crate) struct AppState;\n",
                "serve/infill.rs": "use super::AppState;\nuse crate::mcp;\n",
            },
            0,
        ),
        "nested super inside an inline module": (
            {
                "lib.rs": "mod a;\nmod b;\n",
                "a.rs": "mod inner { mod deeper { use super::super::super::b::B; } }\npub struct A;\n",
                "b.rs": "use crate::a::A;\npub struct B;\n",
            },
            1,
        ),
    }
    failed = 0
    for label, (files, want) in cases.items():
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rel, body in files.items():
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                (root / rel).write_text(body, encoding="utf-8")
            got = check(root, out=io.StringIO())
        status = "ok" if got == want else "FAIL"
        failed += got != want
        print(f"self-test {status}: {label} (exit {got}, want {want})")
    return 1 if failed else 0


def main(argv: list[str]) -> int:
    if "--self-test" in argv:
        return self_test()
    roots = [a for a in argv if not a.startswith("-")] or DEFAULT_ROOTS
    status = 0
    for root in roots:
        status |= check(Path(root))
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
