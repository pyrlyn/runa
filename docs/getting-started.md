# Getting started with runa

`runa` is one CLI binary that runs AI models locally (GGUF via ggml / llama.cpp)
or through the OpenAI and Anthropic APIs. Thinking controls, tools, and
`runa serve` work the same way on both sides.

## Install

From a GitHub Release (portable CPU archives; Metal / Vulkan / CUDA are extra
artifacts):

```sh
curl --proto '=https' --tlsv1.2 -LsSf \
  https://github.com/pyrlyn/runa/releases/latest/download/runa-installer.sh | sh
```

Or Homebrew from the formula on that Release: `brew install ./runa.rb`.

Host-tuned local build:

```sh
RUSTFLAGS='-C target-cpu=native' cargo build --release --features native
```

See [`versions.md`](versions.md) for feature and artifact details.

## First commands

Fit-check a model before downloading, then run / serve / chat:

```sh
runa fit hf:unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M --ctx 16384 --kv q8_0
runa fit --recommend --use code   # best catalog models for this machine
runa run qwen "explain KV-cache quantization in one paragraph"
runa serve --port 8080            # OpenAI-compatible HTTP server (loopback)
runa chat qwen                    # REPL; add --tui for the full-screen UI
```

`runa fit` says whether a model runs on this machine and how fast, before you
download it. After a few `runa bench` runs, predictions calibrate to your
device (see [`guide.md`](guide.md)). `runa serve` listens on `127.0.0.1`
unless you pass `--api-key` with a non-loopback `--host`.

## Useful next steps

**Check before you download** — `fit` reads the model's header remotely (no weights) and prints a verdict. For `hf:ggml-org/gpt-oss-20b-GGUF:gpt-oss-20b-MXFP4.gguf` at `--ctx 16384 --kv q8_0` on a 64 GiB Apple-silicon Mac it printed:

```sh
runa fit hf:ggml-org/gpt-oss-20b-GGUF:gpt-oss-20b-MXFP4.gguf --ctx 16384 --kv q8_0
```

```text
Verdict: FITS GPU
Exit code: 0

Memory:
  GPU weights:  1854.1 MiB
  Compute buf:  582.7 MiB
  KV cache:     408.0 MiB
  GPU total:    2844.8 MiB

GPU decode:  224.8 tok/s  |  prefill: 224.8 tok/s  |  TTFT: 4.55s
CPU decode:  11.7 tok/s  |  prefill: 11.7 tok/s  |  TTFT: 87.45s
```

(Rows with zero bytes are left out above; the speeds are estimates until `runa bench` calibrates them.)

**Calibrate speed predictions** — each `runa bench` appends to the calibration
DB; later `fit` / `--recommend` scale by measured history:

```sh
runa bench qwen2.gguf --mode cpu --pp 512 --tg 128
runa fit qwen2.gguf --json
```

**Offline recommend** — rank the catalog with no network (KV + compute buffer
counted, not just weights):

```sh
runa fit --recommend --offline --top 5
```

**Thinking budget** — hard-cap reasoning tokens and see the split in JSON:

```sh
runa run qwen3-8b.gguf "What is 2+2?" --think-budget 32 --max-tokens 48 --json
```

**Structured output** — JSON Schema or GBNF on local / OpenAI / Anthropic:

```sh
runa run qwen "Capital of France and its population?" \
  --json-schema '{"type":"object","properties":{"city":{"type":"string"},"population":{"type":"integer"}},"required":["city","population"]}'
```

**MCP tools** — start stdio MCP servers; the model can call them in a loop:

```sh
runa run qwen3 "What is the weather in Paris?" --mcp 'python3 weather_server.py'
runa chat qwen3 --mcp 'npx -y @modelcontextprotocol/server-filesystem .'
```

**See what you spent** — `run` / `chat` / `serve` append to a local journal.
`runa stats` prints it; `runa web` follows it live on `127.0.0.1:3939`:

```sh
runa stats --since 7d
runa web
```

**Transcribe audio** — whisper.cpp ASR (`--lang auto` detects then decodes):

```sh
runa media transcribe --model base --lang auto clip.wav
```

## Configuration

Search order (later wins): `~/.config/runa/config.toml`, then `./runa.toml`.
CLI flags beat `RUNA_*` env vars, which beat the files. Full key list:
[`config.md`](config.md).

Example `runa.toml` knobs:

```toml
[defaults]
threads = 8

[think]
show = true

[audio]
route = "auto"
```

## What to read next

| Doc | Contents |
| --- | --- |
| [guide.md](guide.md) | Capability-by-capability how-to with examples |
| [fit.md](fit.md) | Fit formulas, `--recommend`, media context |
| [thinking.md](thinking.md) | Deep-thinking modes and cloud mapping |
| [structured.md](structured.md) | JSON Schema / GBNF, tools, MCP |
| [media.md](media.md) | Audio, vision, ASR |
| [memory.md](memory.md) | Adaptive memory and task registry API |
| [README.md](README.md) | Full docs index |

License: your choice of GPL-3.0-or-later, the royalty-free license, or a commercial license (see the README). Cargo metadata names only `GPL-3.0-or-later`, the only one of the three with an SPDX identifier; crates.io rejects custom `LicenseRef` names.
No telemetry.
