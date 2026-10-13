# Development

Repository layout, the local gates, CI and releases. Read [AGENTS.md](../AGENTS.md) and [CONTRIBUTING.md](../CONTRIBUTING.md) before claiming a task or changing code.

## Repository layout

| Path | What |
|------|------|
| `crates/runa` | CLI: config, TUI, serve, daemon, MCP, pull, bench, fit |
| `crates/runa-core` | `Backend` trait, `Request`/`Event`, `ThinkConfig`, `Mode` |
| `crates/runa-engine` | `llama-cpp-2` wrapper: load, placement, sampling, mtmd |
| `crates/runa-pool` | Engine dispatch and model pool, usable in-process without the binary |
| `crates/runa-fit` | GGUF header, hardware probe, estimator, calibration DB |
| `crates/runa-memory` | Adaptive memory + task-claim registry |
| `crates/runa-media` | Audio/video decode, frame sampling, ASR (`whisper-rs`) |
| `crates/runa-cloud` | OpenAI + Anthropic adapters, price table |
| `crates/runa-kernels` | Own kernels (Zig preferred); dispatch, refs, benches |
| `docs/` | All documentation — start at [`docs/README.md`](README.md) |
| `scripts/` | Registry lint, fixture guards, perf-regress, release helpers |

Toolchain: `rust-toolchain.toml` + `mise.toml` (see
[`docs/versions.md`](versions.md)).

## Develop

```sh
mise install                      # toolchain from mise.toml
moon run root:test-fast           # library tests plus weight-free integration tests
moon run :test                    # cargo test per crate, including integration tests
moon run root:lint-tasks          # registry lint over docs/tasks.md
moon run root:test-with-cleanup   # full workspace test, then drop downloaded weights and compact target/
```

`moon run root:test-fast` skips GGUF downloads and engine e2e (serve, generate,
bench). `moon run :test` and `moon run root:test-with-cleanup` stay the full gate.

CI: `cargo fmt --check`, `clippy -D warnings`, build + test on macOS / Linux /
Windows, registry lint, fixture-size guard. Read
[`AGENTS.md`](../AGENTS.md) / [`CONTRIBUTING.md`](../CONTRIBUTING.md) before claiming
a task or changing code.

## Release

Cutting a release: Actions → Bump and release (or `bash scripts/release.sh`, which starts it) — see [release.md](release.md).

## The `license` field

The `license` field in `Cargo.toml` (`[workspace.package]`) stays
`GPL-3.0-or-later`. Of the three choices above, only the GPL has an identifier
on the SPDX License List. crates.io accepts only those identifiers in `license`
and rejects `LicenseRef-*` custom names, so the royalty-free and commercial
terms cannot be written there. `license` and `license-file` cannot be set
together; replacing the SPDX identifier with `license-file` would hide the GPL
from license scanners. The field is the open-source choice. This section is
the choice among all three.
