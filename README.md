> [!NOTE]
> This project is under active development. We need "testing volunteers": try it and report what breaks.

# runa

[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=listepo_runa&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=listepo_runa) [![Coverage](https://sonarcloud.io/api/project_badges/measure?project=listepo_runa&metric=coverage)](https://sonarcloud.io/component_measures?id=listepo_runa&metric=coverage) [![Tests](https://img.shields.io/sonar/tests/listepo_runa?server=https%3A%2F%2Fsonarcloud.io&compact_message)](https://sonarcloud.io/component_measures?id=listepo_runa&metric=tests)

Local-first AI CLI: fit-check a GGUF before you download it, run locally via
ggml / llama.cpp, or hit OpenAI and Anthropic — one binary, one thinking model,
one OpenAI-compatible `serve`.

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

runa has no published GitHub release yet (the `v0.1.0` tag exists without one), so build from
source for now:

```sh
RUSTFLAGS='-C target-cpu=native' cargo build --release --features native
```

From the first release on:

```sh
curl --proto '=https' --tlsv1.2 -LsSf \
  https://github.com/pyrlyn/runa/releases/latest/download/runa-installer.sh | sh
```

or Homebrew from the formula on that release: `brew install ./runa.rb`. GPU variants
(Metal / Vulkan / CUDA) and feature pins: [`docs/versions.md`](docs/versions.md).

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
| `runa serve [model]` | OpenAI-compatible HTTP server; live graphs at `/dashboard` |
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

Start at [`docs/README.md`](docs/README.md), the documentation index. Most used:
[`getting-started.md`](docs/getting-started.md), [`guide.md`](docs/guide.md),
[`config.md`](docs/config.md), [`fit.md`](docs/fit.md), [`thinking.md`](docs/thinking.md),
[`media.md`](docs/media.md), [`structured.md`](docs/structured.md),
[`dashboard.md`](docs/dashboard.md). Contributors: [`docs/development.md`](docs/development.md),
[`CONTRIBUTING.md`](CONTRIBUTING.md), [`AGENTS.md`](AGENTS.md).

## Status

In development: active work lives in [`plan.md`](plan.md), finished tasks in
[`done.md`](done.md), after-1.0 ideas in [`roadmap.md`](roadmap.md). Release binaries are
portable CPU builds (ggml runtime dispatch) for macOS arm64, Linux x86_64 and Windows x86_64;
Metal / Vulkan / CUDA builds are extra artifacts. No telemetry.

## License

You can use this project under **any** of the following licenses, at your choice:

1. [GNU GPLv3](LICENSE): free for open source applications on any platform, including embedded systems.
2. [Royalty-free License](LICENSE-ROYALTY-FREE.md): free for proprietary desktop, mobile, and web applications, as long as you disclose that your application uses this project. Embedded systems are not covered.
3. [Commercial license](PRICING.md): for proprietary applications, including embedded systems, without the attribution requirement.

<!-- license-sync:start -->
Commercial use not covered by the GPLv3 or the Royalty-free License requires a separate paid
license — see [PRICING.md](PRICING.md).
<!-- license-sync:end -->

The `license` field in `Cargo.toml` (`[workspace.package]`) stays
`GPL-3.0-or-later`. Of the three choices above, only the GPL has an identifier
on the SPDX License List. crates.io accepts only those identifiers in `license`
and rejects `LicenseRef-*` custom names, so the royalty-free and commercial
terms cannot be written there. `license` and `license-file` cannot be set
together; replacing the SPDX identifier with `license-file` would hide the GPL
from license scanners. The field is the open-source choice. This section is
the choice among all three.
