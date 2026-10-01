---
title: runa
tagline: Local-first AI CLI — fit-check a GGUF before you download it, run it locally via ggml / llama.cpp, or hit OpenAI and Anthropic.
repo: https://github.com/pyrlyn/runa
install: "curl --proto '=https' --tlsv1.2 -LsSf https://github.com/listepo/runa/releases/latest/download/runa-installer.sh | sh"
install_alternatives:
  - 'brew install ./runa.rb'
  - "RUSTFLAGS='-C target-cpu=native' cargo build --release --features native"
version: "0.1.0"
accent: "#F07050"
accent2: "#FFB088"
accentLight: "#E85A3C"
order: 4
---

<!-- Website copy for the listepo project site. The sync-docs workflow copies this file to
pyrlyn/landing (main) as content/projects/runa.md on every change to main and on every v*
tag; front matter follows CONTENT_CONTRACT.md in that repository.
Sources (checked 2026-10-01): README.md, docs/getting-started.md and docs/guide.md; version
from the latest tag (v0.1.0); accent, accent2 and accentLight are the dark-theme --runa-accent,
the dark-theme --runa-ember and the light-theme --runa-accent in docs/brand/tokens.css. -->

## Overview

`runa` is one CLI binary that runs AI models locally (GGUF via ggml / llama.cpp) or through the
OpenAI and Anthropic APIs. Thinking controls, structured output, tools and `runa serve` work the
same way on both sides.

Before you download a model, `runa fit` tells you whether it runs on this machine and how fast,
and the forecast calibrates to your device as you record `runa bench` runs. Release binaries are
portable CPU builds for macOS arm64, Linux x86_64 and Windows x86_64; Metal, Vulkan and CUDA
builds are extra artifacts. No telemetry.

## Features

- **Fit before fetch.** `runa fit` gives a memory and speed forecast before any download,
  calibrated by real `runa bench` runs. `runa fit --recommend` ranks a curated catalog for your
  hardware, offline too.
- **Local and cloud, same controls.** Deep thinking (`off` / `on` / budget / effort), structured
  output and tools work the same way for GGUF models and for the OpenAI and Anthropic APIs.
- **Structured output.** Force an answer into a JSON Schema or a GBNF grammar; local models use
  llama.cpp grammar sampling, so the output always matches.
- **MCP tools.** `--mcp` starts stdio MCP servers and the model calls them in a loop in `run` and
  `chat`; `runa serve` supports tool calling.
- **Media in the loop.** Audio and video through native mmproj, ASR (whisper.cpp) or the cloud;
  `runa media probe|video|transcribe`.
- **Serve and stay warm.** An OpenAI-compatible HTTP server (`runa serve`) and a local daemon
  socket for snappy `run` / `chat`; compute modes `cpu` / `gpu` / `hybrid` plus `auto` placement.
- **Adaptive memory.** Shrinks toward a floor when idle, with a bounded pre-grow when a task is
  heavy.

## Install

From a GitHub Release (portable CPU archives; Metal / Vulkan / CUDA are extra artifacts):

```sh
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/listepo/runa/releases/latest/download/runa-installer.sh | sh
```

Or Homebrew from the formula on that Release:

```sh
brew install ./runa.rb
```

Host-tuned local build:

```sh
RUSTFLAGS='-C target-cpu=native' cargo build --release --features native
```

## Usage examples

Fit-check a model before downloading it, and find the best catalog models for this machine:

```sh
runa fit hf:unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M --ctx 16384 --kv q8_0
runa fit --recommend --use code
runa fit --recommend --offline --top 5
```

Run, chat, or serve an OpenAI-compatible HTTP API:

```sh
runa run qwen "explain KV-cache quantization in one paragraph"
runa chat qwen --tui
runa serve --port 8080
```

Calibrate speed predictions with a measured run:

```sh
runa bench qwen2.gguf --mode cpu --pp 512 --tg 128
runa fit qwen2.gguf --json
```

Cap reasoning tokens, or let the model call MCP tools:

```sh
runa run qwen3-8b.gguf "What is 2+2?" --think-budget 32 --max-tokens 48 --json
runa run qwen3 "What is the weather in Paris?" --mcp 'python3 weather_server.py'
```

Transcribe audio with whisper.cpp:

```sh
runa media transcribe --model base --lang auto clip.wav
```

## Links

- Repository: <https://github.com/pyrlyn/runa>
- Documentation: <https://github.com/pyrlyn/runa/tree/main/docs>
- Getting started: <https://github.com/pyrlyn/runa/blob/main/docs/getting-started.md>
- Configuration reference: <https://github.com/pyrlyn/runa/blob/main/docs/config.md>
- Releases: <https://github.com/pyrlyn/runa/releases>
- License: your choice of GNU GPLv3, a royalty-free license for proprietary desktop, mobile and web
  apps (with attribution), or a commercial license (see <https://github.com/pyrlyn/runa#license>)
