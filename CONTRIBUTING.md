# Contributing

Repository prose is English (except the `docs/ru/` and `docs/uk/` translations). Read
[`AGENTS.md`](AGENTS.md) / [`CLAUDE.md`](CLAUDE.md) before changing code:
claim protocol, Zig vs C kernels, and workspace layout.

## Checks

```sh
mise install
moon run :test
moon run root:lint-tasks
```

Use the pinned toolchain via mise. Do not invent public APIs in docs — document
what the tree already ships. User guides live under [`docs/`](docs/). English docs are the source; a change to one
updates its `docs/ru/` and `docs/uk/` translations in the same change (see `AGENTS.md` §13).
