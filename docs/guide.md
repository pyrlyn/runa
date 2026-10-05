# User guide

Copy-pasteable examples for the flags and behaviors. Unless noted,
outputs below come from macOS CPU runs on the fixtures in
`tests/fixtures/`. Reference docs: `config.md` (every key),
`fit.md` (formulas), `structured.md` (tools/MCP), `media.md`
(audio/vision), `thinking.md`, `memory.md` (public API).

## Calibrated speed predictions (`runa bench`, `runa fit`)

**What.** Decode/prefill predictions scale by measured history:
after a `runa bench` run is recorded, later predictions for the same
`(device, backend, quant)` multiply by median measured/predicted.

**Why.** Cold model errors reached 265% on CPU (M3). Three
calibration runs on your device pull typical errors into the ±15%
band instead.

**How.** Every `runa bench` appends to `~/.runa/calibration.json`
(unless `--no-calibrate`); `RUNA_CALIBRATION` points elsewhere.
`bench`, `fit`, and `fit --recommend` all read it; an empty or
missing DB leaves raw model output untouched, and one bad sample
(zero/NaN factor) is ignored per axis.

```sh
runa bench qwen2.gguf --mode cpu --pp 512 --tg 128   # records a sample
runa fit qwen2.gguf --json                            # predictions now calibrated
```

Proven by e2e `fit_calibration_db_scales_predictions`: a 2.0×
measured/predicted sample exactly doubles the reported decode.

## Worker threads (`--threads`)

**What.** `--threads N` on `run` / `chat` / `bench` / `serve`;
`RUNA_THREADS`; `[defaults] threads` in `runa.toml`. Precedence:
flag > env > file > built-in default.

**Why.** The old default (all logical CPUs incl. E-cores) lost to
`llama-bench` auto-threading (M4, 27%). The built-in default now
matches llama.cpp: P-cores on Apple Silicon
(`hw.perflevel0.logicalcpu`), logical CPUs elsewhere.

```sh
runa run model.gguf "hi" --threads 4
# verdict line shows it:
# runa load ... threads 4 mmap:true mlock:false
```

```toml
# runa.toml
[defaults]
threads = 8
```

`--threads 0` (or a bad `RUNA_THREADS` / `[defaults] threads`)
fails before any load. GGUF-only: `--backend mistral` rejects it
explicitly, and `run --threads …` skips the daemon (load-shaping
flag, like `--device`).

## Speech transcription language (`--lang auto`)

**What.** `runa media transcribe --lang auto` (the default)
detects the language and transcribes in one pass.

**Why.** It used to detect correctly yet return an empty transcript
(M7 blocker): whisper.cpp's `detect_language` flag means
*detect-only, return without decoding*. Auto now passes language
`"auto"` with the flag unset.

```sh
runa media transcribe --model base --lang auto clip.wav
# language: en
# (transcript…)
```

Verified live: `--lang auto` output is byte-identical to
`--lang en` on the same audio.

## Reasoning token counts (`--json`, serve, `/usage`)

**What.** Every generation counts reasoning-body tokens
(`Usage.reasoning_tokens`), so budget compliance (M6) is checkable
from outside:

```sh
runa run qwen3-8b.gguf "What is 2+2?" --think-budget 32 --max-tokens 48 --json
```

```json
{"text":"\n\n2 + 2 equals 4. Here's","usage":{"prompt_tokens":15,"generated_tokens":48,"reasoning_tokens":35,"pp_toks_per_s":77.1,"tg_toks_per_s":15.5},"stop":"max_tokens"}
```

35 ≤ 32 budget + grace — enforced *and* reported. Serve exposes
the same split on non-streaming chat completions:

```json
"usage": {
    "prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13,
    "completion_tokens_details": {"reasoning_tokens": 0}
}
```

Chat `/usage` appends `(reasoning N)` when N > 0. Mistral/cloud
backends report 0 (no reasoning split there).

## Idle shrink in `serve` / `daemon` (M11)

**What.** After `idle_timeout_s` (default 300, `[memory]`,
`RUNA_MEMORY_IDLE_TIMEOUT_S`) without requests, loaded engines get
an idle job: the LMDB prompt cache is unmapped, the model stays
loaded:

```text
serve: idle qwen2-0_5b-instruct-q4_0: prompt cache released (model kept)
```

**Why / limits.** Idle CLIs/servers should not sit on cache
mappings. Measured: RSS 772.9 MiB before *and* after on a 0.5B
model — the release is real but negligible next to resident
weights, so the M11 gate (RSS ≤ floor + 10%) stays infeasible
without model unload, which the design forbids. Only generation
endpoints (`chat/completions`, `messages`, `embeddings`,
`transcriptions`) reset the idle timer; `/health` and `/v1/models`
polls deliberately do not.

## Anthropic streaming tool calls (client)

**What.** The `reqwest`+SSE client reassembles streamed `tool_use`
blocks (`content_block_start` + `input_json_delta` fragments) into
the same `ToolUse` event the non-stream reply produces, ahead of
`Done`, so the CLI tool loop drains both paths identically.

**Why not stream by default.** Only whole-message replies preserve
thinking signatures for the next tool round; streamed thinking
blocks carry none to send back. Hence the CLI uses non-stream
calls and the SSE path stays as the correct fallback.

## Offline recommendations with KV + compute

**What.** `runa fit --recommend --offline` fits catalog file sizes
with no network — now counting KV cache (linear in `--ctx`,
`--kv q8_0` halves / `q4_0` quarters the f16 number) and the
compute buffer (linear in ubatch), not just weights:

```sh
runa fit --recommend --offline --top 5
# runa fit --recommend · VRAM 12 GiB (RUNA_FAKE_VRAM) · RAM 16 GiB · ctx 8192 · kv f16 · offline: sizes only
#  1. gpt-oss 20B                tier 4    ~11 tok/s  FITS CPU
#     runa pull hf:ggml-org/gpt-oss-20b-GGUF:gpt-oss-20b-MXFP4.gguf
#  2. Qwen2.5 VL 7B Instruct     tier 3    ~98 tok/s  FITS GPU
#     ...
```

A long context on a big model no longer "fits" offline while
failing online. `~` marks size-based estimates.

## MCP server quoting (`--mcp`)

**What.** `--mcp '<command args>'` splits with shell quoting:
`'...'` is literal, `"..."` allows backslash escapes, a backslash
outside quotes escapes the next character:

```sh
runa run qwen3 "hi" --mcp "python3 'my dir/s.py' --root /tmp"
```

Unterminated quotes are an error. Anything fancier (env vars)
belongs in `[mcp.servers]` (`docs/config.md`).

## Chat history across turns

**What.** `runa chat` (REPL and `--tui`) sends the transcript so
far with every turn, tool rounds included:

```text
> My name is Ada.
Hello! How can I assist you today?
> What is my name? Answer with the name only.
Ada
```

History trims oldest whole turns past ~75% of `--ctx` with a
`(history trimmed to fit ctx)` notice. Local loads count real tokens
through the model's own tokenizer; over the daemon socket (no
tokenizer there) the trim falls back to a chars/4 estimate — a guard
rail either way, since the engine still fails loudly past real ctx.
`/reset` and `/model` clear it (`/mode` keeps it: same weights).
Failed turns record nothing, so retry resends the same history.

## Split-GGUF catalog entries

**What.** A catalog entry may list shard refs 2..N in `parts`
(`ref` stays part 1); the probe sums tensor bytes across every
part header before fitting. `size` is then the parts' byte sum:

```toml
[[model]]
name = "Example 70B split"
ref = "hf:org/model:model-00001-of-00003.gguf"
parts = ["hf:org/model:model-00002-of-00003.gguf",
         "hf:org/model:model-00003-of-00003.gguf"]
size = 42520398432
```

No split entry ships yet (everything catalogued fits in one
file); the schema and probe support are covered by tests, waiting
for the first sharded model that fits Tier-1/2 hardware.

## Harmony (gpt-oss) tool calls

**What.** Tool requests render through the model's own template,
so gpt-oss speaks Harmony (`<|channel|>commentary
to=functions.…`) while Qwen3 speaks Hermes `<tool_call>` and
template-less models fall back to generic JSON. Covered against
the real 20B template in both header orders; `thinking_forced_open`
templates need no client special-casing (llama.cpp owns the flag
from our `enable_thinking` — proven by a required Qwen3-8B tool
call completing inside a 64+8 thinking budget).

## System load cap (`--max-load-percent`)

**What.** Warning-only cap (default 80%) on the share of total
system resources a run may use: `--max-load-percent N` on `run` /
`chat` / `bench` / `serve` / `daemon`, `RUNA_MAX_LOAD_PERCENT`,
`[system] max_load_percent`. Precedence: flag > env > file.

**Why.** A heads-up before a big model squeezes the machine: each
breach prints an actionable `warning:` naming the value to set.
The run always proceeds — never fatal.

```sh
runa run model.gguf "hi" --threads 16
# warning: system already uses ~924 MiB RAM (91% of 1024 MiB), above
#   max_load_percent=80 — set max_load_percent to at least 91 (...)
# warning: --threads 16 wants ~100% of 16 CPUs, above
#   max_load_percent=80 — set max_load_percent to at least 100 (...)
#   or lower --threads to 12
```

Three lines exist: model RAM demand (after the header is read),
ambient RAM pressure, and `--threads` CPU share. Out-of-range
values fail before any load. The flag opts `run` out of the daemon
(the daemon knows no per-request cap) and is rejected on
`--backend mistral` (never warned there).
