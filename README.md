# runa

[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=listepo_runa&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=listepo_runa) [![Coverage](https://sonarcloud.io/api/project_badges/measure?project=listepo_runa&metric=coverage)](https://sonarcloud.io/component_measures?id=listepo_runa&metric=coverage) [![Tests](https://img.shields.io/sonar/tests/listepo_runa?server=https%3A%2F%2Fsonarcloud.io&compact_message)](https://sonarcloud.io/component_measures?id=listepo_runa&metric=tests)

Local-first AI CLI: fit-check a GGUF before you download it, run locally via
ggml / llama.cpp, or hit OpenAI and Anthropic — one binary, one thinking model,
one OpenAI-compatible `serve`.

**Website:** https://listepo.github.io/runa/

## What it does

- **Fit before fetch** — `runa fit` says whether a model runs on this machine
  and how fast (memory + speed forecast, calibrated by real `bench` runs).
  `runa fit --recommend` ranks a curated catalog for your hardware.
- **Local + cloud, same controls** — deep-thinking (`off` / `on` / budget /
  effort), structured output, and tools work the same way for GGUF and APIs.
- **Media in the loop** — audio and video via native mmproj, ASR
  (whisper.cpp), or cloud; `runa media probe|video|transcribe`.
- **Serve & stay warm** — OpenAI-compatible HTTP (`runa serve`) and a local
  daemon socket for snappy `run` / `chat`; compute modes `cpu` / `gpu` /
  `hybrid` plus `auto` placement.
- **Adaptive memory** — shrink toward a floor when idle, bounded pre-grow when
  a task is heavy; MCP tool loop in `run` / `chat`.

## Install

From a GitHub Release:

```sh
curl --proto '=https' --tlsv1.2 -LsSf \
  https://github.com/listepo/runa/releases/latest/download/runa-installer.sh | sh
```

Or Homebrew from the formula on that Release: `brew install ./runa.rb`.

Host-tuned local build:

```sh
RUSTFLAGS='-C target-cpu=native' cargo build --release --features native
```

GPU variant artifacts (Metal / Vulkan / CUDA) and feature pins:
[`docs/versions.md`](docs/versions.md).

## Getting started

```sh
runa fit hf:unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M --ctx 16384 --kv q8_0
runa fit --recommend --use code   # best catalog models for this machine
runa run qwen "explain KV-cache quantization in one paragraph"
runa serve --port 8080            # OpenAI-compatible HTTP server
runa chat qwen                    # REPL; add --tui for the full-screen UI
```

Config search order (later wins): `~/.config/runa/config.toml`, then
`./runa.toml`. Flags beat `RUNA_*` env vars, which beat the files.

Full walkthrough with copy-paste examples:
[`docs/getting-started.md`](docs/getting-started.md) →
[`docs/guide.md`](docs/guide.md).

## Usage / CLI

| Command | What |
|---------|------|
| `runa fit <model\|hf:repo:file>` | Verdict before any download. `--recommend [--use USE] [--top N] [--offline]` ranks the catalog |
| `runa run [prompt]` | One-shot generation (stdin when piped) |
| `runa chat [model]` | Interactive REPL (`/think`, `/mode`, `/model`, `/reset`); `--tui` for full-screen |
| `runa serve [model]` | OpenAI-compatible HTTP server |
| `runa daemon [model]` | Keep models warm over `~/.cache/runa/runa.sock` |
| `runa pull hf:<repo>:<file>` | Download a model; `runa models` lists cache + aliases |
| `runa bench <model>` | Prefill/decode throughput; records into the calibration DB |
| `runa doctor` | Compiled-in backends and native-build flags |
| `runa media probe\|video\|transcribe` | Inspect / sample / ASR via whisper.cpp |
| `runa tasks list\|claim\|release` | Claim registry in `docs/tasks.md` (contributors) |

Man pages: [`docs/runa.1`](docs/runa.1), [`docs/runa-run.1`](docs/runa-run.1)
(`man ./docs/runa.1`). Every flag and `runa.toml` key:
[`docs/config.md`](docs/config.md).

## Concepts

| Topic | One-liner | Doc |
|-------|-----------|-----|
| **Fit** | Memory + speed forecast before download; `--recommend` ranks the catalog | [`docs/fit.md`](docs/fit.md) |
| **Thinking** | `ThinkConfig`: off / on / budget / effort; same mapping for local and cloud | [`docs/thinking.md`](docs/thinking.md) |
| **Serve** | OpenAI + Anthropic routes; load progress on `/health`; tools & `response_format` | [`docs/structured.md`](docs/structured.md) |
| **MCP / tools** | `--mcp` stdio servers; tool loop in `run` / `chat`; serve tool calling | [`docs/structured.md`](docs/structured.md) |
| **Media** | native / asr / auto audio routes; `--image` / `--video`; whisper ASR | [`docs/media.md`](docs/media.md) |
| **Memory** | Idle shrink, heavy grow; `MemoryManager` / `TaskRegistry` API | [`docs/memory.md`](docs/memory.md) |

## Docs

| File | What |
|------|------|
| [`docs/README.md`](docs/README.md) | Documentation index — audience map and how to add docs |
| [`docs/getting-started.md`](docs/getting-started.md) | Install, first commands, where to go next |
| [`docs/guide.md`](docs/guide.md) | Capability-by-capability how-to with real examples |
| [`docs/config.md`](docs/config.md) | Every `runa.toml` / `RUNA_*` key |
| [`docs/fit.md`](docs/fit.md) | `fit` / `--recommend`, formulas, media context |
| [`docs/thinking.md`](docs/thinking.md) | `ThinkConfig` modes, show/hide, cloud mapping |
| [`docs/media.md`](docs/media.md) | Audio routes, vision, ASR |
| [`docs/structured.md`](docs/structured.md) | JSON Schema / GBNF, serve `response_format`, tools, MCP |
| [`docs/memory.md`](docs/memory.md) | `MemoryManager`, `TaskRegistry` public methods |
| [`docs/profiles.md`](docs/profiles.md) | Profiling and polyglot escape-hatch evaluation |
| [`docs/versions.md`](docs/versions.md) | Version pins, cargo features, release artifacts |
| [`docs/prices.toml`](docs/prices.toml) | User-editable cloud price table (USD / 1M tokens) |
| [`docs/kernels.md`](docs/kernels.md) | `runa-kernels` benchmarks and adoption gates |
| [`docs/baselines.md`](docs/baselines.md) | `llama-bench` reference numbers |
| [`docs/perf-nightly.md`](docs/perf-nightly.md) | Nightly perf workflow and > 3 % regression gate |
| [`docs/perf-baseline.json`](docs/perf-baseline.json) | Baseline data the gate reads |
| [`docs/release.md`](docs/release.md) | How a release runs (`scripts/release.sh`, tag, artifacts) |
| [`docs/release-1.0.md`](docs/release-1.0.md) | v1.0 metric checklist with evidence |
| [`docs/adr/`](docs/adr/) | ADRs for D1–D18 and D23 |
| [`docs/runa.1`](docs/runa.1) / [`docs/runa-run.1`](docs/runa-run.1) | Man pages |

### Contributors

| File | What |
|------|------|
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Contribution checks |
| [`AGENTS.md`](AGENTS.md) | Agent claim protocol |
| [`docs/tasks.md`](docs/tasks.md) | Live claim registry |
| [`plan.md`](plan.md) / [`done.md`](done.md) | Active plan and finished work |

## Repository layout

| Path | What |
|------|------|
| `crates/runa` | CLI: config, TUI, serve, daemon, MCP, pull, bench, fit |
| `crates/runa-core` | `Backend` trait, `Request`/`Event`, `ThinkConfig`, `Mode` |
| `crates/runa-engine` | `llama-cpp-2` wrapper: load, placement, sampling, mtmd |
| `crates/runa-fit` | GGUF header, hardware probe, estimator, calibration DB |
| `crates/runa-memory` | Adaptive memory + task-claim registry |
| `crates/runa-media` | Audio/video decode, frame sampling, ASR (`whisper-rs`) |
| `crates/runa-cloud` | OpenAI + Anthropic adapters, price table |
| `crates/runa-kernels` | Own kernels (Zig preferred); dispatch, refs, benches |
| `docs/` | All documentation — start at [`docs/README.md`](docs/README.md) |
| `site/` | Hugo site (`baseURL` → https://listepo.github.io/runa/) |
| `scripts/` | Registry lint, fixture guards, perf-regress, release helpers |

Toolchain: `rust-toolchain.toml` + `mise.toml` (see
[`docs/versions.md`](docs/versions.md)).

## Develop

```sh
mise install                      # toolchain from mise.toml
moon run :test                    # cargo test across the workspace
moon run root:lint-tasks          # registry lint over docs/tasks.md
moon run root:test-with-cleanup   # full test, then drop downloaded weights and compact target/
```

CI: `cargo fmt --check`, `clippy -D warnings`, build + test on macOS / Linux /
Windows, registry lint, fixture-size guard. Read
[`AGENTS.md`](AGENTS.md) / [`CONTRIBUTING.md`](CONTRIBUTING.md) before claiming
a task or changing code.

## Status / license

Active work lives in [`plan.md`](plan.md); finished tasks in [`done.md`](done.md);
after-1.0 ideas in [`roadmap.md`](roadmap.md).

Release binaries are **portable** (ggml runtime CPU dispatch). GitHub Releases
upload macOS arm64, Linux x86_64, and Windows x86_64 CPU archives plus
installers; Metal / Vulkan / CUDA builds are extra artifacts. Cutting a release:
`bash scripts/release.sh` — see [`docs/release.md`](docs/release.md).

License: see [License](#license) below. No telemetry.

## License

You can use this project under **any** of the following licenses, at your choice:

1. [GNU GPLv3](LICENSE): free for open source applications on any platform, including embedded systems.
2. [Royalty-free License](LICENSE-ROYALTY-FREE.md): free for proprietary desktop, mobile, and web applications, as long as you disclose that your application uses this project. Embedded systems are not covered.
3. [Commercial license](PRICING.md): for proprietary applications, including embedded systems, without the attribution requirement.
