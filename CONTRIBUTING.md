# Contributing

Repository prose is English. Read [`AGENTS.md`](AGENTS.md) / [`CLAUDE.md`](CLAUDE.md)
before changing code: claim protocol, Zig vs C kernels, and workspace layout.

## Checks

```sh
mise install
moon run root:test-fast
moon run :test
moon run root:lint-tasks
```

`moon run root:test-fast` is the cheap path. `moon run :test` is the full gate.

Use the pinned toolchain via mise. Do not invent public APIs in docs — document
what the tree already ships. User guides live under [`docs/`](docs/).
