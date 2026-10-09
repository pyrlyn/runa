# Configuration: `runa.toml` and environment keys

Search order (later files win): `~/.config/runa/config.toml`, then `./runa.toml`.
Home is `$HOME`, or `$USERPROFILE` on Windows when `HOME` is unset. CLI flags
beat `RUNA_*` env vars, which beat the files. Inline API keys in TOML are
rejected (P3.8). Later `[memory]` files overlay only the keys they set; earlier
keys stay. Cloud prices follow the same later-wins overlay:
`docs/prices.toml` (when the process cwd is the repo), then
`~/.config/runa/prices.toml`.

## Top level

| Key | Values | Env | CLI |
|-----|--------|-----|-----|
| `on_unfit` | `error` (default) \| `cpu` \| `cloud:<backend>:<model>` | `RUNA_ON_UNFIT` | `--on-unfit` |

## `[models.<alias>]`

| Key | Values |
|-----|--------|
| `source` | Local path or `hf:org/repo:quant` (resolved by `runa pull`, never fetched by `run`) |
| `lora` | String or array of `path[:scale]` LoRA adapters for this alias (P8.5; appended after `[model] lora`) |

## `[model]`

Defaults for whichever model is being run (P8.5).

| Key | Values | CLI |
|-----|--------|-----|
| `lora` | String or array of `path[:scale]` LoRA adapters | `--lora` |

```toml
[model]
lora = ["adapter.gguf:0.5", "other.gguf"]
```

## `[defaults]`

Run-wide defaults for inference commands (P10.2).

| Key | Values | Env | CLI |
|-----|--------|-----|-----|
| `threads` | worker threads, integer >= 1 (default: P-cores on macOS, else all logical CPUs) | `RUNA_THREADS` | `--threads` |

```toml
[defaults]
threads = 8
```

## `[system]`

Cap on the share of total system resources this app may use, in percent
of the total (default: 80). Applies to the RAM budget (model demand must
fit `total RAM × max_load_percent / 100`) and the CPU thread share
(`--threads` must fit `CPUs × max_load_percent / 100`). Every command
that loads a model (`run`, `chat`, `bench`, `serve`, `daemon`, `fit`)
checks the cap at startup and prints a `warning:` to stderr when system
RAM is already over the cap or the model/threads do not fit — each
warning names the value to set so the run fits. Warnings never fail the
run; the user decides.

| Key | Values | Env | CLI |
|-----|--------|-----|-----|
| `max_load_percent` | integer 1..=100 (default: 80) | `RUNA_MAX_LOAD_PERCENT` | `--max-load-percent` |

```toml
[system]
max_load_percent = 80
```

## `[mcp.servers.<name>]`

Stdio MCP servers whose tools `runa run` / `runa chat` offer to the model
(P8.3). `--mcp '<command args>'` adds more for one run. See `docs/structured.md`.

| Key | Values |
|-----|--------|
| `command` | Program to start (required) |
| `args` | Array of argument strings |
| `env` | Table of extra environment variables |

```toml
[mcp.servers.fs]
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "."]
```

## `[think]`

| Key | Values | Env | CLI |
|-----|--------|-----|-----|
| `mode` | `on` \| `off` \| `budget` \| `effort` | `RUNA_THINK` | `--think` |
| `budget` | reasoning token cap (required when `mode = "budget"`) | `RUNA_THINK_BUDGET` | `--think-budget` |
| `grace` | extra tokens after the budget (default 0) | `RUNA_THINK_GRACE` | (slash `/think grace`) |
| `effort` | `low` \| `medium` \| `high` \| `max` | `RUNA_EFFORT` | `--effort` |
| `show` | bool — print reasoning | `RUNA_SHOW_REASONING` | `--show-reasoning` / `--no-show-reasoning` |

`mode = "budget"` needs `budget`. `--think off` cannot combine with budget/effort.

## `[audio]`

| Key | Values | Env | CLI |
|-----|--------|-----|-----|
| `route` | `auto` (default) \| `native` \| `asr` | `RUNA_AUDIO_ROUTE` | `--audio-route` |

See `docs/media.md`.

## `[memory]`

Later files overlay only the keys they set (a project `floor_mib` does not
reset `idle_timeout_s` from the user file).

| Key | Values | Env |
|-----|--------|-----|
| `idle_timeout_s` | seconds before idle shrink | `RUNA_MEMORY_IDLE_TIMEOUT_S` |
| `floor_mib` | idle floor | `RUNA_MEMORY_FLOOR_MIB` |
| `max_growth_mib` | cap on `grow_for` | `RUNA_MEMORY_MAX_GROWTH_MIB` |

See `docs/memory.md`.

## Serve

`runa serve` binds `127.0.0.1` by default. `--host` must be a loopback
address (`127.0.0.0/8` or `::1`) unless `--api-key KEY` is set. With a
key, every `/v1` route requires `Authorization: Bearer <key>`; `/health`
stays open. There is no `runa.toml` key for this token (inline API keys
in TOML are rejected).

## Cloud env (not TOML)

| Env | Use |
|-----|-----|
| `OPENAI_API_KEY` | OpenAI |
| `ANTHROPIC_API_KEY` | Anthropic |
| `OPENAI_BASE_URL` | OpenAI-compatible base URL |
