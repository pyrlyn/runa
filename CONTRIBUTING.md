# Contributing

Repository prose is English. Read [`AGENTS.md`](AGENTS.md) / [`CLAUDE.md`](CLAUDE.md)
before changing code: claim protocol, Zig vs C kernels, and workspace layout.

## Checks

```sh
mise install
moon run :test
moon run root:lint-tasks
```

Use the pinned toolchain via mise. Do not invent public APIs in docs — document
what the tree already ships. User guides live under [`docs/`](docs/).

## Contributor License Agreement

Before a pull request can be merged, every committer must sign the
[Contributor License Agreement](https://github.com/pyrlyn/infra/blob/main/CLA.md)
([Russian translation](https://github.com/pyrlyn/infra/blob/main/CLA.ru.md); the English text
prevails). You keep the copyright in your work; the agreement lets the project be offered under
the GPL and under its royalty-free and commercial licenses. The `cla` check on your pull request
explains how to sign: post the comment `I have read the CLA Document and I hereby sign the CLA`.
One signature covers all pyrlyn repositories.
