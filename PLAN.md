# runa: Kotlin/Native rewrite plan

Status: in progress. Each phase in [section 8](#8-implementation-phases) carries a `Status:` line;
a phase marked done has landed. The previous implementation was removed in the pull request that
added this file (see [Repository reset](#12-repository-reset)).

## Contents

1. [Goal](#1-goal)
2. [Scope and non-goals](#2-scope-and-non-goals)
3. [Analysis of lmstudio-ai/lms](#3-analysis-of-lmstudio-ailms)
4. [Feature mapping: lms to runa](#4-feature-mapping-lms-to-runa)
5. [Architecture](#5-architecture)
6. [Daemon API (what the CLI talks to)](#6-daemon-api-what-the-cli-talks-to)
7. [Files, configuration and environment](#7-files-configuration-and-environment)
8. [Implementation phases](#8-implementation-phases)
9. [Testing strategy](#9-testing-strategy)
10. [CI](#10-ci)
11. [Packaging and release](#11-packaging-and-release)
12. [Repository reset](#12-repository-reset)
13. [Risks and open questions](#13-risks-and-open-questions)

## 1. Goal

`runa` is a single native binary, written in Kotlin/Native, that does for local models what
[`lms`](https://github.com/lmstudio-ai/lms) does for LM Studio, without needing LM Studio:

- find and download GGUF models (`runa get`), list and import them (`runa ls`, `runa import`);
- load and unload them in a background daemon (`runa load`, `runa unload`, `runa ps`);
- serve them over an OpenAI-compatible HTTP API (`runa server start|stop|status`);
- chat with them in the terminal (`runa chat`, `runa chat -p "..."`);
- stream logs (`runa log stream`) and report status (`runa status`);
- optionally drive a running LM Studio through its documented REST API (`--backend lmstudio`).

Inference runs in-process in the daemon through [llama.cpp](https://github.com/ggml-org/llama.cpp),
linked as a static C library through Kotlin/Native cinterop.

Targets, in priority order: `macosArm64` (primary, Metal), `linuxX64`, `linuxArm64`; `mingwX64`
is optional and comes last.

## 2. Scope and non-goals

In scope for 1.0:

- the commands in [section 4](#4-feature-mapping-lms-to-runa) marked 1.0;
- GGUF models only, via llama.cpp (CPU everywhere, Metal on macOS; CUDA/Vulkan builds later);
- LLM text chat and embeddings; tool calls and JSON-schema output through llama.cpp grammars;
- one machine, plus `--host` to reach a daemon on another machine over the HTTP API.

Not in scope (explicitly dropped from lms):

- LM Studio Hub publishing (`clone`, `push`, `login`, `logout`, `whoami`), plugins (`dev`,
  `create`), LM Link (`link *`), experiment flags (`flags`), MLX models.
- A GUI. The `brand/` assets are kept for the website, icons and the terminal colour palette.
- Re-implementing LM Studio's private WebSocket RPC (lmstudio-js); only its public REST API is
  supported in the compatibility backend.

## 3. Analysis of lmstudio-ai/lms

Snapshot analysed: `lmstudio-ai/lms` at commit `138dcff` (2026-11-10), package
`@lmstudio/lms-cli` 0.4.1, MIT licence, about 17 000 lines of TypeScript in `src/`.

### 3.1 What it is

`lms` is a thin client. It holds no models and runs no inference: every command connects to a
running LM Studio app (or the headless `llmster` daemon, or Bionic) and calls it through the
`@lmstudio/sdk` (lmstudio-js) over a WebSocket at `ws://<host>:<port>`. It ships inside LM Studio
and is put on `PATH` by `lms bootstrap` / `npx lmstudio install-cli`. It cannot be built on its
own: it lives as a submodule of the lmstudio-js monorepo (Node >= 25).

### 3.2 Command surface

Grouped as in `src/index.ts`:

| Group | Commands | Notes |
|---|---|---|
| Local models | `chat`, `get`, `load`, `unload`, `ls`, `ps`, `import` | the core of the tool |
| Serve | `server start\|stop\|status`, `log stream` | REST server inside the app; log tail over the SDK |
| Remote instances | `link status\|enable\|disable\|agent\|set-device-name\|set-preferred-device` | LM Link, a paid networking feature |
| Runtime | `runtime ls\|get\|update\|remove\|select\|survey` | engine "extension packs" (llama.cpp CPU/CUDA/Vulkan/Metal, MLX) |
| Develop and publish | `clone`, `push`, `dev`, `login`, `logout`, `whoami` | LM Studio Hub and plugins |
| Hidden | `create`, `bootstrap`, `daemon up\|down\|status\|update`, `flags`, `status`, `version` | |

Notable options worth copying:

- `load [model-key]`: interactive picker when no key; `--gpu off|max|0..1`; `-c/--context-length`;
  `--parallel`; `--ttl <s>` (auto-unload when idle); `--identifier` (instance id used by the API);
  `--estimate-only` (memory estimate, no load); `--auto` (placement and context chosen for you);
  speculative decoding (`--speculative-draft-model`, min/max draft tokens); `-y/--yes`.
- `get [name]`: `owner/model`, `owner/model@quant`, or a full Hugging Face URL; `--gguf`/`--mlx`
  filters; `--select` to pick a variant; picks a quantization that fits the hardware by default.
- `ls [modelKey]`: `--llm`, `--embedding`, `--variants`, `--json`. `ps --json`.
- `unload [id]` / `unload --all`.
- `import <file>`: move by default, or `--copy`, `--hard-link`, `--symbolic-link`; `--user-repo`;
  `--dry-run`.
- `server start -p <port> --bind <addr> --cors`; `server status --json`.
- `log stream --source model|server|runtime --filter input|output --json --stats`.
- `chat [model] -p <prompt> -s <system> --stats --ttl --reasoning`; slash commands `/help`,
  `/exit`, `/model`, `/clear`, `/system-prompt`, `/stats`, `/reasoning`, `/download`.
- Global hidden options: `--host`, `--port`, `--log-level`, `--quiet`, `--verbose`; `-v/--version`.

### 3.3 How it finds and talks to the backend

From `src/createClient.ts` and `src/localAPIServer.ts`:

1. Discovery reads `~/.lmstudio/.internal/http-server.json` (`{port, pid}`), checks the pid is
   alive, then probes the port; falls back to scanning a fixed list of legacy ports; then tries
   Bionic's own folder. `LMS_API_SERVER_INFO_PATH` pins one exact instance.
2. If nothing runs, it starts the app recorded in `*-install-location.json`
   (`llmster`, then LM Studio with `--run-as-service`, then Bionic), detached, and polls for up to
   60 s until the API appears; a process that exits early is a failed launch.
3. It authenticates with a client identifier (`lms-cli`) and a passkey built from a key compiled
   into the binary plus `~/.lmstudio/.internal/lms-key-2`; remote hosts get a random,
   unprivileged identifier.
4. Everything else is SDK calls: `client.system.listDownloadedModels()`,
   `client.llm.listLoaded()`, `client.repository.createModelDownloadPlanner()`,
   `client.system.startHttpServer()`, `client.runtime.surveyHardware()`, and so on.

`status`, `server status` and `server stop` never start an app. That rule is worth keeping.

### 3.4 Configuration and state

- Home: `~/.lmstudio` (overridable), models in `~/.lmstudio/models/<publisher>/<repo>/<file>`.
- CLI preferences: `~/.lmstudio/.internal/cli-pref.json` (zod-validated, defaults filled in):
  last loaded models, auto-start server, catalog fetching, one-time warnings.
- Environment: `LMS_API_SERVER_INFO_PATH`, `LMS_SERVER_HOST`, `LMS_FORCE_PROD`, plugin variables.

### 3.5 UX and engineering patterns worth keeping

- Every listing command has `--json`; human output is a coloured table (`columnify`, `chalk`).
- Progress bars and spinners for downloads and loads; Ctrl-C cancels a download or load cleanly.
- Interactive pickers (fuzzy search) when an argument is missing, and `-y` to make every prompt
  non-interactive for scripts.
- `UserInputError` prints a one-line message without a stack trace; anything else prints the stack.
- The chat UI is an Ink (React for terminals) app with an input reducer that has 1 300 lines of
  unit tests. runa's REPL gets the same treatment: a pure, tested input state machine.

### 3.6 What not to copy

- The dependency on a private, versioned WebSocket RPC and an embedded passkey.
- Hidden global options (`--host`, `--port` are hidden in lms; runa documents them).
- Node as a runtime: start-up cost and a large install. runa is one static-ish binary.

## 4. Feature mapping: lms to runa

| lms | runa | Phase | Release |
|---|---|---|---|
| `lms version`, `-v` | `runa version [--json]`, `--version` | 1 | 1.0 |
| `lms status` | `runa status [--json]` | 5 | 1.0 |
| `lms ls [key] --json --llm --embedding --variants` | `runa ls [key] [--json] [--llm\|--embedding] [--variants]` | 2 | 1.0 |
| `lms import <file> -c -L -l --user-repo --dry-run` | `runa import <file> [--copy\|--hard-link\|--symlink] [--user-repo] [--dry-run]` | 2 | 1.0 |
| `lms get <name>[@quant] --select -y` | `runa get <hf-repo\|url>[@quant] [--select] [-y]` (Hugging Face) | 3 | 1.0 |
| (none: needs a daemon) | `runa run <model> "<prompt>"` one-shot, in-process | 4 | 1.0 |
| `lms daemon up\|down\|status` | `runa daemon up\|down\|status [--json]` | 5 | 1.0 |
| `lms load <key> --gpu -c --parallel --ttl --identifier --estimate-only -y` | `runa load` with the same flags | 5 | 1.0 |
| `lms unload [id] --all` | `runa unload [id] [--all]` | 5 | 1.0 |
| `lms ps --json` | `runa ps [--json]` | 5 | 1.0 |
| `lms server start -p --bind --cors`, `stop`, `status` | `runa server start\|stop\|status` | 6 | 1.0 |
| `lms chat [model] -p -s --stats --ttl` | `runa chat [model] [-p] [-s] [--stats] [--ttl]` | 6 | 1.0 |
| `lms log stream --source --filter --json --stats` | `runa log stream [...]` | 7 | 1.0 |
| `lms runtime survey` | `runa runtime survey [--json]` (CPU, RAM, GPU, Metal/CUDA) | 9 | 1.0 |
| `lms runtime ls\|get\|update\|select\|remove` | `runa runtime ls` (compiled-in backends only) | 9 | later |
| `lms load --auto`, speculative decoding | `runa load --auto`, `--draft-model` | 9 | later |
| `lms chat --reasoning` | `runa chat --reasoning off\|on\|<budget>` | 6 | later |
| `--host`, `--port` | documented global options, plus `--backend runa\|lmstudio` | 5, 8 | 1.0 |
| `lms bootstrap` | install script / Homebrew put `runa` on `PATH` | 10 | 1.0 |
| `link *`, `clone`, `push`, `dev`, `create`, `login`, `logout`, `whoami`, `flags` | not planned | - | - |

## 5. Architecture

### 5.1 Process model

```
runa (CLI process)                         runa daemon (background process, same binary)
  clikt commands                              Ktor server (CIO) on 127.0.0.1:<port>
  RunaClient (Ktor client) ---- HTTP/SSE ---> /api/v1/*  native API (LM Studio-shaped)
     |                                        /v1/*      OpenAI-compatible API
     |                                        ModelRegistry: loaded instances, TTL, JIT load
     |                                        Engine interface -> LlamaEngine (llama.cpp via
     |                                                            cinterop, Metal on macOS)
     +-- local-only commands (ls, import, get, run, runtime survey) work without the daemon
```

- One binary. `runa daemon up` re-executes itself as `runa __daemon` detached
  (`posix_spawn` + `setsid` on Unix; `CreateProcess` with `DETACHED_PROCESS` on Windows) and
  writes `~/.runa/run/daemon.json` with `{pid, port, token, version}`.
- Commands that need the daemon (`load`, `unload`, `ps`, `chat`, `server`, `log stream`) start
  it on demand, like lms starts LM Studio, and poll up to 30 s for the discovery file and
  `GET /api/v1/status`. `status`, `daemon status`, `server status` and `server stop` never start it.
- The "local server" of lms (`server start`) maps to exposing the daemon's HTTP API on a
  user-chosen bind address and port. With `server stop` the daemon keeps running on its private
  loopback port so `load`/`ps` keep working.

### 5.2 Gradle modules

Kotlin Multiplatform project, Gradle Kotlin DSL, version catalog in `gradle/libs.versions.toml`.

| Module | Responsibility | Depends on |
|---|---|---|
| `:core` | domain types (`ModelKey`, `ModelInfo`, `LoadConfig`, `LoadedInstance`, `HardwareInfo`), error types (`UserInputError`), `Result` helpers | kotlinx.serialization |
| `:platform` | paths (`~/.runa`, XDG on Linux, `%LOCALAPPDATA%` on Windows), process spawn/detach, pid liveness, signals, terminal size, file locking; `expect/actual` per OS | okio, platform.posix |
| `:config` | config file (`config.json`), env overrides, CLI preferences, discovery file read/write | `:core`, `:platform` |
| `:gguf` | pure-Kotlin GGUF header and metadata parser (architecture, parameter count, context length, quantization, tensor sizes) | okio |
| `:store` | on-disk model store: layout, index, `ls`, `import` (move/copy/hard link/symlink), integrity | `:gguf`, `:config` |
| `:hub` | Hugging Face client: repo info, file list, GGUF variant selection, resumable downloads with SHA-256 check, auth token | `:core`, `:gguf`, Ktor client |
| `:engine` | backend-agnostic inference API: `Engine` (memory estimate, load) and `LoadedModel` (tokenize, generate as a `Flow<GenerationEvent>`, embeddings), request, sampling and stats types | `:core`, okio, kotlinx.coroutines |
| `:engine-llama` | the llama.cpp implementation of `Engine`: load model, context, sampling, chat templates, grammars; the only module with cinterop | `:engine`, `:gguf`, `:platform`, cinterop `llama` |
| `:engine-testing` | `FakeEngine`, an `Engine` without weights or llama.cpp, for the tests of every module that takes an engine | `:engine` |
| `:testing` | `assertGolden`, the golden file check shared by the tests of `:cli` and `:daemon` | `:platform`, okio, kotlin.test |
| `:daemon` | model registry (instances, TTL eviction, JIT load, parallel slots), HTTP server, SSE, log bus | `:engine` (interface only), `:store`, `:hub`, Ktor server |
| `:client` | typed client for the daemon API; `LmStudioBackend` for LM Studio's REST API; backend selection | Ktor client |
| `:cli` | clikt command tree, output formatting (tables, `--json`), progress bars, prompts, chat REPL | clikt, mordant, `:client` and local modules |
| `:app` | `main()`, the `runa` executable for every target; picks the `Engine` implementation | `:cli`, `:daemon`, `:engine-llama` |

Rule: `:cli` never touches `:engine` directly except through `runa run`, which embeds the engine
for one-shot use without a daemon.

Rule: `:daemon` and `:cli` depend on the `Engine` interface, never on `:engine-llama`. Only `:app`
names the implementation and passes it in. That keeps the C API in one module, lets every other
module build and test without the llama.cpp static libraries, and leaves room for another backend
(MLX, a remote engine) without touching the daemon.

### 5.3 Libraries

Exact versions are pinned in phase 0 in `gradle/libs.versions.toml` (latest stable at that time).

| Need | Library | Native note |
|---|---|---|
| CLI parsing, help, completion | [Clikt](https://github.com/ajalt/clikt) | supports Kotlin/Native; chosen over kotlinx-cli (unmaintained, no subcommand ergonomics) |
| Terminal colours, tables, progress, prompts | [Mordant](https://github.com/ajalt/mordant) | Clikt's own renderer, native targets supported |
| HTTP client | Ktor client: `Darwin` engine on macOS, `Curl` on Linux, `WinHttp` on Windows | CIO client on native lacks TLS for HTTPS to Hugging Face; verify in phase 0 |
| HTTP server | Ktor server `CIO` engine, SSE plugin, ContentNegotiation | native-capable; plain HTTP on loopback, TLS left to a reverse proxy |
| JSON | kotlinx.serialization (json) | |
| Concurrency | kotlinx.coroutines | `Dispatchers.IO` exists on native; every loaded model gets a serial view of it (`limitedParallelism(1)`) and all its llama.cpp calls run there |
| Files | Okio (`FileSystem.SYSTEM`) | |
| Time | kotlinx-datetime | |
| Logging | small own logger over a `SharedFlow` log bus (feeds `log stream`) | avoid JVM-only logging libs |
| Inference | llama.cpp (C API `llama.h`), static libs per target, Metal on macOS | built with CMake, consumed with cinterop `.def` |
| Tests | kotlin.test, kotlinx-coroutines-test, Ktor `MockEngine` and `testApplication` | |

### 5.4 Native interop (llama.cpp)

All of this lives in `:engine-llama`, behind the `Engine` and `LoadedModel` interfaces of `:engine`.

- Vendor llama.cpp as a git submodule at `third_party/llama.cpp`, pinned to a release tag.
- A Gradle task per target (`CmakeStaticLibraries` in `build-logic`) runs CMake with the clang and
  sysroot of the Kotlin/Native distribution, so the Linux libraries also build on macOS
  (`-DBUILD_SHARED_LIBS=OFF`, only the library, Metal on for macOS with the shader library
  embedded, OpenMP off, `GGML_NATIVE` off with an AVX2 baseline on x64) and produces `libllama.a`,
  `libggml*.a`. `mingwX64` compiles against the bindings but has no static libraries yet.
- `src/nativeInterop/cinterop/llama.def` declares the headers (`llama.h`, `runa_llama.h`) and
  `staticLibraries`; link the C++ runtime (`-lc++` on macOS, `-lstdc++` on Linux) and the
  Metal/Foundation frameworks.
- Chat templates, tool calls and JSON answers are in llama.cpp's `common` library, which is C++
  and so out of cinterop's reach. `src/nativeInterop/shim` is a C API over it (`runa_llama.h`:
  the templates of a model, and a turn that renders the chat, samples and parses the answer as
  it is written); it speaks JSON where llama.cpp's chat code does and never throws. The CMake
  project that the Gradle task builds is this shim, with llama.cpp as its subdirectory and
  `LLAMA_BUILD_COMMON` on, which adds `libruna-llama.a`, `libllama-common*.a` and
  `libcpp-httplib.a` to the link and about 3 MB to the executable.
- One Kotlin class, `LlamaModel`, owns the native model and context and frees them in `close()`;
  a turn of the shim, with its sampler, lives for one generation. No raw pointer leaves
  `:engine-llama`.
- llama.cpp is not tied to a thread, it only must not be entered twice at once. So a model has no
  thread of its own: its calls run one at a time on a serial view of `Dispatchers.IO`, which needs
  no shutdown and cannot be closed under a coroutine that still has to resume on it.
- llama.cpp's own log reaches the caller at debug level; the last error line is kept and becomes
  the message of a failed load.
- Token streaming: generation runs on the instance's dispatcher and emits into the
  `Flow<GenerationEvent>` of `LoadedModel.generate`; cancellation checks between decode steps.
- Model loading progress uses llama.cpp's progress callback through `staticCFunction` and a
  `StableRef`.

## 6. Daemon API (what the CLI talks to)

The native API mirrors the shapes of LM Studio's documented v1 REST API
(`https://lmstudio.ai/docs/developer/rest`) so one client type serves both backends.
All endpoints need `Authorization: Bearer <token>` from the discovery file, except
`GET /api/v1/status` on the daemon's own loopback port. On the listener that `server start` opens,
`/api/v1/*` needs that token without exception, and `/v1/*` is open, or takes the API key that
`server start` was given; the key never opens `/api/v1/*`.

| Endpoint | Purpose | Used by |
|---|---|---|
| `GET /api/v1/status` | version, pid, uptime, server bind/port, loaded count | `status`, `daemon status`, discovery probe |
| `GET /api/v1/models` | downloaded models with metadata and `loaded_instances` | `ls`, `ps` |
| `POST /api/v1/models/load` | `{model, context_length, gpu, parallel, ttl, identifier, estimate_only}` → instance; progress as SSE when `stream=true` | `load` |
| `POST /api/v1/models/unload` | `{instance_id}` or `{all: true}` | `unload` |
| `POST /api/v1/models/download` / `GET /api/v1/models/download/status/{job}` | daemon-side downloads (optional; the CLI downloads itself in 1.0) | `get` (later) |
| `POST /api/v1/chat` | native chat `{model, input, system_prompt, previous_response_id, store, stream, …}` → `{model_instance_id, output, stats, response_id}`; with `stream=true` the SSE events `chat.start`, `message.start`, `message.delta`, `message.end`, `chat.end`, or `error` | `chat` |
| `GET /api/v1/logs/stream?source=model\|server\|runtime&filter=input\|output` | SSE log stream from the moment of the request: one `listening` event, then a `log` event per `{time, source, type, text, level, model, endpoint, tool_calls, stats}`; every source without `source` | `log stream` |
| `POST /api/v1/server` | start/stop the public listener `{running, bind, port, cors, api_key}` → `{running, bind, port, cors, api_key_required}`, which `status` carries as `server` | `server start\|stop` |
| `POST /api/v1/shutdown` | graceful stop | `daemon down` |
| `GET /v1/models`, `POST /v1/chat/completions`, `POST /v1/completions`, `POST /v1/embeddings` | OpenAI-compatible, streaming via SSE | third-party clients |

The LM Studio backend (`--backend lmstudio`, phase 8) maps `ls`, `ps`, `load`, `unload`, `chat`
and `get` onto LM Studio's `/api/v1/*` endpoints at `http://localhost:1234` (or `--host/--port`)
with `LM_API_TOKEN` as the bearer token. Commands that LM Studio's REST API cannot serve
(`log stream`, `server start`) print a clear "not supported by this backend" error.

## 7. Files, configuration and environment

| Path | Content |
|---|---|
| `~/.runa/models/<publisher>/<repo>/<file>.gguf` | model store (same layout as LM Studio so `runa import --symlink` can reuse `~/.lmstudio/models`) |
| `~/.runa/models/.index.json` | cache of parsed GGUF headers, keyed by path, size and mtime; safe to delete |
| `~/.runa/config.json` | defaults: models dir, server port/bind/cors, default context, gpu, ttl, backend |
| `~/.runa/cli-pref.json` | last loaded models, one-time warnings |
| `~/.runa/run/daemon.json` | discovery: `{pid, port, token, version}`, mode 0600 |
| `~/.runa/logs/daemon.log` | daemon log; at 10 MB it becomes `daemon.log.1`, and the three files before the current one are kept |

On Linux, `XDG_DATA_HOME`/`XDG_CONFIG_HOME`/`XDG_STATE_HOME` are honoured; `RUNA_HOME`
overrides everything. Environment: `RUNA_HOME`, `RUNA_HOST`, `RUNA_PORT`, `RUNA_TOKEN`,
`RUNA_BACKEND`, `RUNA_API_KEY`, `RUNA_LOG_LEVEL`, `RUNA_LOG_PROMPTS`, `HF_TOKEN`, `HF_ENDPOINT` (a Hub mirror; `https://`, or `http://` on localhost),
`LM_API_TOKEN`. Precedence: flag > env > `config.json` > default.

## 8. Implementation phases

Each phase is one or more pull requests. A phase is done when every acceptance criterion holds
on CI for all enabled targets.

### Phase 0: toolchain and skeleton

Status: done ([#142](https://github.com/pyrlyn/runa/pull/142)). ktlint, not detekt (detekt 1.23 does
not support Kotlin 2.4; detekt 2 is alpha). SonarCloud is on; CodeQL Kotlin is off because CodeQL
only traces Kotlin/JVM compilations. Release binaries: 2.6 MB macosArm64, 11 MB linuxX64.
Not verified on CI: GitHub Actions started no run for the pull request (see section 13). Checked
instead on macOS arm64 and, with the linuxArm64 test binaries, in an Ubuntu 24.04 arm64 container.

Steps:
1. Add the Gradle wrapper, `settings.gradle.kts`, `gradle/libs.versions.toml`, the module list
   from 5.2 as empty modules, and a convention plugin (`build-logic/`) that applies Kotlin
   Multiplatform with targets `macosArm64`, `linuxX64`, `linuxArm64` (and `mingwX64` behind a
   Gradle property).
2. `:app` builds an executable `runa` that prints `runa <version> (<git sha>)`; the version comes
   from `gradle.properties`, the sha from a generated `BuildInfo.kt`.
3. Spike, kept as tests: Ktor client GET over HTTPS on each target; Ktor CIO server on native
   answering a request; Clikt + Mordant rendering a table; Okio reading a file.
4. Add ktlint or detekt (formatting and static checks) and `explicitApi()` in library modules.
5. CI workflow `ci.yml` (section 10); turn on CodeQL for Kotlin and SonarCloud in `.github/infra.yml`
   if they support the Kotlin/Native build, otherwise record why not.

Acceptance: `./gradlew build` is green on macOS arm64 and Linux x64 runners; `runa --version`
runs from `build/bin/<target>/releaseExecutable`; the spike tests pass on every target.

### Phase 1: CLI framework, output and config

Status: done ([#143](https://github.com/pyrlyn/runa/pull/143)). Verified like phase 0: macOS arm64
and an Ubuntu 24.04 arm64 container, not CI.

Steps:
1. Root command with global options `--json`, `--log-level`, `--quiet`, `--verbose`, `--host`,
   `--port`, `--backend`, `--yes`; help layout grouped like lms (Local models, Serve, Runtime).
2. Output layer: one `Renderer` for human output (Mordant tables, colours off when not a TTY or
   `NO_COLOR` is set) and one for `--json` (kotlinx.serialization, stable field names, no colours).
3. Error handling: `UserInputError` → one line on stderr, exit 2; network/daemon errors → exit 3;
   unexpected → stack trace, exit 1; Ctrl-C → exit 130.
4. `:config` with the precedence rule from section 7, `runa config get|set|path` (small, useful
   for tests), shell completion (`runa completion bash|zsh|fish`).

Acceptance: golden-file tests for help output and `--json` output of `version`; exit-code tests;
config precedence tests (flag > env > file > default).

### Phase 2: local model store, `ls`, `import`

Status: done ([#146](https://github.com/pyrlyn/runa/pull/146)). Verified on macOS arm64 and in an
Ubuntu 24.04 arm64 container; warm `ls` on 200 models takes 36 ms with the release binary. Tests use
synthetic headers; the parser was also run on a GGUF file written by llama.cpp in phase 3
(Qwen2.5 0.5B: `qwen2`, 494M parameters, context 32768, Q4_K_M). `import` derives `publisher/repo` as
`<general.organization or "local">/<file name without the quantization>`; a variant is named by its
quantization (`@q4_k_m`), or by its file name when two files of a repo share one.

Steps:
1. `:gguf` parser: header, metadata key/values, tensor infos; reads only the header (never maps
   the whole file); derives architecture, parameter count, quantization, context length,
   chat template, embedding vs LLM.
2. `:store` scans the models dir, caches metadata in `~/.runa/models/.index.json` keyed by path,
   size and mtime.
3. `runa ls [key] [--llm|--embedding] [--variants] [--json]`: groups quantizations of one repo as
   variants, shows size, params, arch, quant, and whether loaded (once the daemon exists).
4. `runa import <file>`: infer `publisher/repo` from the file name or ask (`--user-repo`, `-y`);
   move by default, `--copy`, `--hard-link`, `--symlink`, `--dry-run`; refuse non-GGUF files.

Acceptance: parser tests on synthetic GGUF headers generated in tests (no weights in git) plus a
corrupt/truncated-file test; `ls --json` schema test; import tests for all four modes in a temp
dir; `ls` on 200 models finishes under 200 ms with a warm index.

### Phase 3: Hugging Face download, `get`

Status: done ([#147](https://github.com/pyrlyn/runa/pull/147)). Verified on macOS arm64 and in an
Ubuntu 24.04 arm64 container, with tests on Ktor's `MockEngine`, with the real binaries against a
local stand-in for the Hub (interrupt, resume, split model), and with one real download:
`runa get bartowski/Qwen2.5-0.5B-Instruct-GGUF@Q4_K_M` (398 MB), interrupted once and resumed.
The default variant is the largest quantized one whose files take at most 70 % of physical memory:
`runtime survey` does not exist yet, and the engine's own estimate comes with phase 4. `mmproj`
files are listed but not downloaded; pairing a projector with a vision model belongs to the phase
that loads one. `HF_TOKEN` goes to the Hub's own origin only, never to the CDN host a download
redirects to.

Steps:
1. `:hub`: resolve `owner/repo`, `owner/repo@quant`, or a full `https://huggingface.co/...` URL;
   list `.gguf` files (including split `-00001-of-0000N` shards and `mmproj` files).
2. Variant choice: default to the largest quantization that fits the memory estimate from
   phase 2 metadata and `runtime survey`; `--select` opens a picker; `-y` takes the default.
3. Downloads: `.part` file, HTTP range resume, parallel shard downloads, SHA-256 against the
   Hub's LFS oid, atomic rename into the store, progress bar with speed and ETA, Ctrl-C keeps
   the `.part` for resume. `HF_TOKEN` for gated repos.

Acceptance: tests with Ktor `MockEngine` for resolution, resume after a cut connection, checksum
mismatch, gated repo without token (clear error); one manual end-to-end download of a small
model (for example a 0.5B Q4 GGUF) recorded in the PR.

### Phase 4: inference engine, `runa run`

Status: done (the `Engine` abstraction in [#144](https://github.com/pyrlyn/runa/pull/144), the
llama.cpp engine and `runa run` in [#148](https://github.com/pyrlyn/runa/pull/148)). llama.cpp is pinned to v1.0.0. Verified on macOS
arm64 (Metal) and in an Ubuntu 24.04 arm64 container (CPU) with Qwen2.5 0.5B Q4_K_M: the engine
tests, the leak check (55 load, generate and unload cycles, peak resident size within 5 %) and two
`runa run --temp 0 --seed 1` runs with the same answer. `linuxX64` is linked locally and tested on
CI; `mingwX64` only compiles. The numbers are in the pull request.

What differs from the steps below, and what is left:

- Speed. Eight alternating runs of 256 tokens against `llama-completion` built from the same
  submodule gave a median of 69 tokens/s for `runa run` and 46 for `llama-completion`, on a machine
  that other builds kept busy (load average over 20), so the figures say only that runa is not
  slower. `llama-cli` itself needs the server build, hence `llama-completion`. A measurement on an
  idle machine is still owed.
- The context is 4096 tokens unless `-c/--context-length` or `context_length` says otherwise, and
  never more than the model was trained with; llama.cpp rounds it up to a multiple of 256.
  `runa run` also takes `--seed` and `--gpu`, and a path to a `.gguf` file instead of a model name.
- Chat templates are the ones llama.cpp's C API recognises (it has no Jinja); for any other
  template the engine warns and uses ChatML.
- The estimate is the weights plus the KV cache, as step 3 says. It was checked against llama.cpp
  (48 MiB for 4096 tokens of this model) and leaves out the compute buffers, about 320 MiB here.
  Nothing uses it yet: `--estimate-only` comes with `runa load` in phase 5, and `get` still picks a
  variant by file size.
- Embeddings were tested by pooling the chat model's output; no embedding model has been run.
- Tokens per second count from the first token, so an answer of one token reports 0.
- The first run of a new binary spends about 4 s compiling the embedded Metal shaders; macOS caches
  them, and later runs take 0.02 s for that.
- The CI checks are steps of the build job rather than a job of their own, so llama.cpp is built
  once: the model is fetched with `runa get` into a cache, the engine tests run on it, and the
  same seed must give the same answer twice.

Steps:
1. Submodule llama.cpp, CMake build task per target, `llama.def`, link flags (section 5.4), in a
   new `:engine-llama` module.
2. `LlamaEngine`, the llama.cpp implementation of `Engine`: load with `n_gpu_layers` from `--gpu off|max|0..1`, context length,
   threads; tokenize; apply the model's chat template; sample (temperature, top-p, top-k, min-p,
   repeat penalty, seed); stream tokens as a `Flow`; stop strings; embeddings.
3. Memory estimate (weights + KV cache for the requested context) used by `--estimate-only` and by
   variant selection.
4. `runa run <model> "<prompt>" [--system] [--max-tokens] [--temp] [--json]` loads in-process,
   streams to stdout, prints stats (time to first token, tokens/s) with `--stats`.

Acceptance: a CI job downloads a tiny GGUF (cached between runs) and checks that `runa run` with a
fixed seed and temperature 0 produces deterministic output; leak check (load/unload 50 times,
RSS stable within 5 %); tokens/s on macOS arm64 within 10 % of `llama-cli` from the same
llama.cpp tag on the same model (measured, recorded in the PR). `LlamaEngine` passes the same
contract checks as the fake engine (`EngineContractTest`).

### Phase 5: daemon, `load`, `unload`, `ps`, `status`

Status: done ([#149](https://github.com/pyrlyn/runa/pull/149)). Verified on macOS arm64 and in an Ubuntu 24.04
arm64 container with Qwen2.5 0.5B Q4_K_M: the unit tests (the registry and the HTTP API against
the fake engine, the commands against a fake daemon) and `app/integration/daemon.sh`, which runs
the real executable: `status` starts nothing, `daemon up` twice, load, `ps --json`, unload, a 2 s
TTL, a killed daemon replaced by the next command, SIGTERM and `daemon down`. CI runs the script
on the release executable of each target. `mingwX64` only compiles.

What differs from the steps below, and what is left:

- A daemon is alive when it holds the lock on `run/daemon.lock`, not when its pid exists: a pid
  can be reused, a lock cannot outlive its process. A discovery file whose lock is free is stale
  and is removed by the next command, `status` included.
- The daemon is detached with a double `fork` and `setsid` rather than `posix_spawn`: besides a
  session of its own it must stop being a child of the CLI, so that nobody has to wait for it.
  It does not run on Windows yet: there is no llama.cpp build to test it with (section 13), so
  `load`, `ps` and `daemon up` fail there with a clear error.
- `unload` does not start the daemon, although section 5.1 lists it: a daemon that does not run
  has nothing to unload.
- A daemon named by address (`--host` other than loopback, or `--port` with a token) is never
  started and its discovery file is not read; it needs `--port` and `--token` or `RUNA_TOKEN`.
  The daemon itself still listens on loopback only; another bind address comes with
  `server start` in phase 6.
- Of the global flags only `--port` and `--log-level` are passed on to a daemon started on demand.
  `--port` against a daemon that already runs on another port is an error, not a restart.
- JIT load needs a request that names a model, so it comes with the chat endpoints in phase 6.
  `--parallel` limits how many requests use an instance at once; decoding them in parallel is
  also phase 6.
- Every request the daemon refuses is `400 invalid_request_error`, an unknown model included;
  `404` waits for the OpenAI-compatible endpoints.
- `load` without a model asks only on an interactive terminal; with `-y`, `--json` or a pipe it
  names the choices and exits with 2, unless there is exactly one model.
- The log is one file, `logs/daemon.log`, appended to; rotation is phase 7.
- The first start of a new binary takes about 4 s (the Metal shaders, as in phase 4); later
  starts take 0.2 to 0.5 s.

Steps:
1. `runa daemon up|down|status`: detach, discovery file with a random token, single-instance lock,
   graceful shutdown on SIGTERM/`/api/v1/shutdown`, stale discovery file detection (dead pid).
2. Ktor server on loopback with the endpoints in section 6 needed here (`status`, `models`,
   `load`, `unload`, `shutdown`).
3. Model registry: several instances, `--identifier`, `--parallel` slots, `--ttl` idle eviction,
   JIT load on first request when enabled, unload frees native memory.
4. CLI: `load` (interactive picker when no key, `-y`, progress over SSE), `unload [id] [--all]`,
   `ps [--json]`, `status [--json]`; auto-start rules from 5.1.
5. `--host/--port` reach a remote daemon; a remote daemon needs `--token` or `RUNA_TOKEN`.

Acceptance: integration tests that start the real daemon binary on a random port, load the tiny
model, see it in `ps --json`, unload it, and stop the daemon; TTL test with a 2 s TTL; a killed
daemon leaves a stale file that the next command detects and replaces; `status` never starts the
daemon (asserted).

### Phase 6: OpenAI-compatible server and chat

Status: done, in three pull requests. Landed in
[#150](https://github.com/pyrlyn/runa/pull/150): the `/v1/*` endpoints of step 1 without tools and
`response_format`, and step 2. Verified on macOS arm64 with
Qwen2.5 0.5B Q4_K_M: the unit tests (the endpoints against the fake engine with golden SSE
streams, the public listener over real sockets, the `server` commands against a fake daemon) and
`app/integration/server.sh`, which drives the real executable with `curl`; the unit tests also in
an Ubuntu 24.04 arm64 container. CI runs the script on the release executable of each target.
`mingwX64` only compiles.

Landed in [#151](https://github.com/pyrlyn/runa/pull/151): `/api/v1/chat` of step 1 and step 3.
Verified the same way: the unit tests (the endpoint against the fake engine with a golden stream,
the input state machine, the command against a fake daemon), also in the Ubuntu container;
`app/integration/chat.sh` with the real executable, which CI runs too; and by hand a conversation
in a pseudo-terminal, where the model recalled a number from an earlier turn and forgot it after
`/clear`.

Landed in [#152](https://github.com/pyrlyn/runa/pull/152): tools and `response_format` of step 1,
and [`docs/openai.md`](docs/openai.md) with its examples. Verified on macOS arm64 with the same
model: the unit tests (the endpoint against the fake engine with a golden stream of a tool call,
the JSON that crosses the shim), the engine tests on real weights (a required and a named call
whose arguments fit the schema, a tool's result read back, JSON of a schema), also in the Ubuntu
container; `app/integration/server.sh`, which now asks the real executable for a tool call, whole
and streamed, and for JSON of a schema. `app/integration/openai.sh` runs the example of
`docs/openai.md` with the official `openai` Python client; it installs that package from PyPI, so
it was first run by CI and not on the machine the phase was written on.

Still owed: embeddings are tested against the fake engine only (the test model is a chat model);
requests to one instance are decoded one after another; `runa chat` has no recall of earlier
input on the arrow keys.

What differs from the steps below:

- A request's `model` is the identifier of a loaded instance, or the name of a stored model as
  `load` takes it. A stored model that is not loaded is loaded by the request, with the configured
  defaults and once however many requests arrive together. An unknown model is `404` with
  `code: model_not_found` on `/v1/*`; the native API keeps `400`.
- What does not exist is refused with `400` rather than ignored: `functions`, content that is not
  text, `n` above 1, `echo`, `suffix`, `dimensions`.
- Tools and JSON answers come from llama.cpp's `common` library through a C shim (section 5.4),
  not from grammars written here: each chat template has its own format for a tool call, and
  llama.cpp already has the grammar and the parser for each. Since then a chat is rendered with
  the model's own Jinja template; the note of phase 4 about templates no longer holds, except that
  a template llama.cpp cannot read is still answered with ChatML and a warning.
- `tool_choice` with a function's name is met by offering only that tool and requiring a call:
  llama.cpp can require a call but not name it. `none` keeps the tools from the model, because a
  model that is shown tools it may not call writes the call out as text. `required` holds the
  call to the tool's schema but, as in llama.cpp, lets the model write text before it, and a small
  model can spend its `max_tokens` there.
- `response_format` is `json_object` (any object, or llama.cpp's `schema` beside the type) or
  `json_schema`, on chat completions only; `strict` is not read, the schema always binds.
  Reasoning is not separated: what a model thinks aloud stays in `content`, and no reasoning
  model has been run.
- A tool call gets its `id` from runa (`call_` and 24 characters); `/api/v1/chat` and `runa chat`
  give the model no tools.
- `top_k`, `min_p` and `repeat_penalty` are taken as llama.cpp's server takes them. `max_tokens`
  below zero means no limit; `developer` is read as the system role; message content may be a
  string or a list of text parts.
- A streamed answer gets its status only when the engine has produced its first event, so a prompt
  the model cannot take is still a `400`. A failure after that is a last `data:` event with the
  error and no `[DONE]`.
- `server start` takes `--api-key` or `RUNA_API_KEY` besides the warning the step asks for; the
  access rules are in section 6. With other settings than the running server's it is an error that
  names `server stop`, not a restart. The server cannot be stopped through its own port.
- A port that is taken no longer ends the daemon: Ktor's native engine reports a failed bind as an
  uncaught exception of its accept loop, and both listeners now start through one helper that
  catches it. `server start` then exits with 2 and the reason; a daemon that cannot take its
  `--port` logs the reason and stops, and the command that started it points at the log.
- Requests to one instance are still decoded one after another; `--parallel` bounds how many may
  wait. Decoding them in one batch is left for later.
- `/api/v1/chat` has the shapes of LM Studio's endpoint and less of it: `integrations` and input
  that is not text are refused with `400`; `reasoning` and `context_length` are ignored; the stream
  has no `model_load.*`, `prompt_processing.*`, `reasoning.*` or `tool_call.*` events.
  `stats.stop_reason` (`stop` or `length`) is runa's addition. It needs the daemon's token on both
  listeners.
- The daemon keeps the conversations, in memory only: the last 1024 responses, the least recently
  used forgotten first, all of them gone when it stops. Continuing a response it no longer has is a
  `400`. `store: false` gives an answer without a `response_id`. A `system_prompt` replaces the
  conversation's, an empty one removes it, none keeps it.
- `runa chat` reads its lines in the terminal's cooked mode: "history" is the conversation, which
  the daemon holds; there is no recall of earlier input on the arrow keys and no editing beyond
  the terminal's own. A line that ends with a backslash continues, and `"""` opens and closes a
  block of lines.
- The prompt, the notes and `--stats` go to stderr and only the answer to stdout. Standard input
  that is not a terminal is read whole as one question, like `-p`; `--json` is for one question
  only. One question is not stored by the daemon.
- Without a model, `chat` takes the one that is loaded, or asks which stored one to load. `--ttl`
  applies only when the command loads the model itself. `/model` with a name switches and starts
  a new conversation.

Steps:
1. `/v1/models`, `/v1/chat/completions` (stream and non-stream, tools, `response_format` with JSON
   schema through grammars), `/v1/completions`, `/v1/embeddings`; `/api/v1/chat` with stats.
2. `runa server start [-p] [--bind] [--cors]`, `stop`, `status [--json]`; warn when binding to a
   non-loopback address without a token.
3. `runa chat [model]`: REPL with history, multi-line input, slash commands `/help`, `/exit`,
   `/model`, `/clear`, `/system-prompt`, `/stats`; `-p` for one prompt to stdout; `-s` system
   prompt; `--ttl`. The input handling is a pure state machine with its own unit tests.

Acceptance: the official `openai` Python client and `curl` examples in `docs/` work against
`runa server` (CI smoke test); streaming chunks follow the OpenAI SSE format (golden tests);
chat REPL state-machine tests; `chat -p` usable in pipes (no colours, no spinner when not a TTY).

### Phase 7: logs and observability

Status: done ([#153](https://github.com/pyrlyn/runa/pull/153)). Verified on macOS arm64 with
Qwen2.5 0.5B Q4_K_M: the unit tests (the stream over a real loopback socket against the fake
engine, the bus, the rotation in a temporary directory, the command against a fake daemon with
golden output, the rendering of a prompt on real weights), also in an Ubuntu 24.04 arm64
container; and `app/integration/logs.sh`, which runs the real executable on both: a chat request
is read as an input and an output event from `log stream --json`, the prompt is in `daemon.log`
only with `--log-prompts`, the streams end on `daemon down`, and a log file of 10 MB is put aside.
CI runs the script on the release executable of each target. By hand: a stream that was idle for
75 s still delivered the 1830 lines llama.cpp logs for one load, none lost. `mingwX64` only
compiles, and has no log file to rotate because it has no daemon.

What differs from the steps below:

- `--log-prompts` is a global flag like `--log-level`, with `RUNA_LOG_PROMPTS` and `log_prompts`
  in `config.json` behind it. It is read when the daemon starts; a daemon that runs keeps what it
  was started with.
- The `model` source has the prompt as the model read it, after the chat template: the engine
  interface got `render` for that. A prompt is rendered only while somebody listens or the file
  wants it, and a prompt that cannot be rendered is left out without failing the request.
- The `server` source has, besides the daemon's own lines, one line for each request it answered,
  at the debug level: the log file has them only with `--log-level debug`, like the lines of
  llama.cpp, which are the `runtime` source. The stream carries every level whatever the file's.
- The stream is live only: nothing is kept for a listener that comes later, and a listener that
  falls more than 1024 events behind loses the oldest. Its first event, `listening`, tells a
  client that nothing after it is missed; a comment line every 15 s keeps a quiet stream open and
  is how the daemon notices a client that went away.
- `--filter` belongs to the `model` source; with another source it is a usage error. `--stats`
  adds the statistics of each answer, in text and in `--json` alike. In JSON a field that does not
  apply is left out.
- The log file is the daemon's standard output, so that what llama.cpp or a crash prints lands in
  it too. Rotation moves the file aside and points the output at a new one; it happens only in the
  daemon that holds the lock, and only when its output really is the log file. A daemon started in
  a terminal logs there, and `daemon status` then shows no path. The size is counted by the
  daemon, so what is printed past its log is not counted. The limits (10 MB, three old files) are
  not configurable.

Steps:
1. Log bus in the daemon with sources `model` (prompts and outputs), `server` (requests),
   `runtime` (llama.cpp log callback).
2. `runa log stream [--source] [--filter input|output] [--json] [--stats]` over SSE.
3. Daemon log file with rotation; `runa daemon status` shows its path.

Acceptance: integration test that sends a chat request and sees the input and output events in
`log stream --json`; prompts are never written to the log file unless `--log-prompts` is set.

### Phase 8: LM Studio backend

Status: done ([#154](https://github.com/pyrlyn/runa/pull/154)). Verified on macOS arm64: contract
tests of `LmStudioBackend` against the documented example responses of LM Studio's v1 REST API,
kept verbatim in `client/src/commonTest/fixtures/lmstudio/` with the page each one came from, and
against four error answers recorded from LM Studio 0.4.28; tests of the commands against a fake LM
Studio that is as strict about unknown keys as the real one, with golden output; both also in an
Ubuntu 24.04 arm64 container. By hand against LM Studio 0.4.28 with an MLX model and a GGUF
embedding model: `ls`, `ps`, `load`, `unload`, `chat` with one prompt and as a conversation of two
turns, and the refusals; the transcript is in the pull request. Not run against the real LM
Studio: `get`, because it downloads a model, and `import --from-lmstudio` on real GGUF files,
because that machine's folder had an MLX model only; both are covered by tests. How to use it is
in [`docs/lmstudio.md`](docs/lmstudio.md).

What differs from the steps below:

- There is one interface, `ModelBackend`, for what the model commands need: list, load, chat and
  unload. `DaemonClient` and `LmStudioBackend` both implement it and answer in the types of
  runa's API; what LM Studio does not tell is null there, so the parameters, the GPU share, the
  parallel slots, the idle time, the estimate and the stop reason became nullable. The HTTP
  plumbing of both is `ApiTransport`.
- LM Studio is never started by runa. `--host` and `--port` name it, with port 1234 when none is
  given. Its token is `--token` or `LM_API_TOKEN`; `RUNA_TOKEN` is never sent to it.
- `load` passes `--context-length` and refuses the other settings before anything is sent: the
  REST API of LM Studio 0.4 takes no GPU share, slots, TTL or identifier. `ps` leaves out the GPU,
  IDLE and TTL columns, which LM Studio has nothing for.
- A model can be named without its publisher, as with runa's own models. LM Studio answers such a
  name with an internal error, so the name is looked up in its list first.
- `get` starts LM Studio's own download job and follows it until it ends. `--select` and the other
  options of runa's downloader are refused.
- A model that reasons first begins its answer with empty lines; `chat` does not print them, and
  `--json` keeps the text as it came.
- The commands that need runa's own daemon, engine or store (`run`, `import`, `status`, `daemon`,
  `server`, `log stream`) end with exit code 2 and a line that names `--backend runa`.
- The optional step is done: `runa import --from-lmstudio[=<dir>]` adds every GGUF model of the
  folder under the publisher and name it has there, as symlinks unless `--copy` or `--hard-link`
  is given. A model the store already has is left alone; MLX models and projector files are
  passed over.

Steps:
1. `LmStudioBackend` in `:client` over `http://<host>:1234/api/v1/*` with `LM_API_TOKEN`.
2. `--backend lmstudio` (or `RUNA_BACKEND`) for `ls`, `ps`, `load`, `unload`, `chat`, `get`;
   unsupported commands fail with a clear message.
3. Optional: `runa import --from-lmstudio` symlinks `~/.lmstudio/models` into runa's store.

Acceptance: contract tests against recorded LM Studio responses (fixtures from the documented
examples); a manual run against a real LM Studio 0.4.x recorded in the PR.

### Phase 9: hardware survey and smarter loading

Status: done ([#158](https://github.com/pyrlyn/runa/pull/158)), steps 1 and 2; step 3 is marked
"later" and is still open. Verified on macOS arm64 with `./gradlew build` and by hand with
Qwen2.5 0.5B Q4_K_M (`runtime survey`, `load --auto` alone, with `--estimate-only` and with a
setting given), in an Ubuntu 24.04 arm64 container (the tests of every module, the survey and
`load --auto` on the CPU), and with the integration scripts. Not verified: a Linux machine with a
real NVIDIA or AMD card, where only the parsing is covered, by faked sources. How to use it is in
[`docs/hardware.md`](docs/hardware.md).

What differs from the steps below:

- What the survey reads sits behind `SystemSources` in `:platform`, and the survey is common
  code over it; `computeCoreCount()` is an answer of the same survey.
- Whether a GPU counts comes from the engine (`llama_supports_gpu_offload()`, `gpu_offload` in
  the survey and in `status --json`), not from the list of compiled-in backends: that list has
  `blas` in it, which is no GPU.
- `--auto` is resolved by the daemon, on its machine and with what it has loaded: the whole model
  on the GPU with the longest context that fits (at most 32768), else a split on a card with
  memory of its own, else the CPU. `--gpu` and `--context-length` that are given are kept. A
  fifth of the memory, at least 2 GiB, stays free, and 10 % of a card's own memory.
- Known limits: `nvidia-smi` runs without a timeout; several GPUs are planned as the largest one;
  the Windows survey has the memory and the processors only; a load in progress is not counted.
- Not done, as the step says: `--draft-model`, `--reasoning`, CUDA and Vulkan release variants,
  `runa runtime ls`.

Steps:
1. `runa runtime survey [--json]`: CPU model and cores, RAM, GPU and VRAM (Metal
   `MTLDevice.recommendedMaxWorkingSetSize` on macOS; `nvidia-smi`/sysfs on Linux), compiled-in
   backends.
2. `runa load --auto`: choose GPU offload and context length from the survey and the estimate.
3. Later: speculative decoding with `--draft-model`, `--reasoning` controls, CUDA and Vulkan
   builds as separate release variants (`runa runtime ls` lists what the binary was built with).

Acceptance: survey snapshot tests with faked system sources; `--auto` picks a config whose
estimate fits the reported memory in unit tests.

### Phase 10: packaging and release (section 11)

Acceptance: a tagged release produces signed archives for every enabled target, the install
script and Homebrew formula install a working `runa`, and `runa --version` matches the tag.

### Phase 11: hardening

- Windows (`mingwX64`) support if still wanted; fuzz-style property tests for the GGUF parser and
  the argument parser; performance pass on start-up time (target: `runa --version` < 20 ms,
  `runa ls` < 100 ms warm); documentation in `docs/` (getting started, commands, API).

## 9. Testing strategy

- Unit tests in `commonTest` wherever possible; native-only code in `nativeTest`/`<target>Test`.
- Golden files for help text and every `--json` output, so output changes are explicit in review.
- No model weights in git: tests generate synthetic GGUF headers; the integration job downloads one
  tiny public GGUF into a CI cache.
- The daemon, the CLI and the HTTP API are tested against the fake `Engine` of `:engine-testing`
  (no weights, no llama.cpp). One contract test defines what every `Engine` must do; the llama.cpp
  engine is held to the same rules by its own tests on a real model, which skip when
  `RUNA_TEST_MODEL` does not name a GGUF file.
- Integration tests run the built binary as a subprocess (daemon start/stop, server, chat `-p`).
- Each phase adds its tests in the same pull request; coverage is reported once SonarCloud runs.

## 10. CI

- Keep `pipeline.yml` (pyrlyn/ci reusable `ci.yml`: scans, licence check, actionlint behind
  `pipeline / gate`) and `coderabbit-issues.yml`, `release-ci-alert.yml` from pyrlyn/ci.
- New `ci.yml` in phase 0: matrix `macos-14` (macosArm64) and `ubuntu-24.04` (linuxX64), later
  `ubuntu-24.04-arm` (linuxArm64); steps: checkout with submodules, `mise install`, Gradle cache,
  cache of the llama.cpp CMake build keyed by submodule sha and target, `./gradlew check`,
  `./gradlew :app:linkReleaseExecutable<Target>`, upload the binary as an artifact.
- Integration steps in the build job (phase 4 onward) with the cached test model; draft PRs skip
  CI as today.
- macOS jobs can use pyrlyn/ci's `macos-runner` action (`MACOS_RUNNER` variable) as before.

## 11. Packaging and release

- Archives: `runa-<version>-<target>.tar.gz` (`.zip` for Windows) with the binary, LICENSE and a
  SHA-256 file; macOS binaries codesigned and notarized when the signing identity is available.
- Release workflow on tag `v*`: build per target on native runners (cinterop with C++ needs the
  target toolchain), create the GitHub release, attach archives and checksums.
- Install: `curl -fsSL https://github.com/pyrlyn/runa/releases/latest/download/install.sh | sh`
  and a Homebrew formula (tap or the release's `runa.rb`); re-add `ketch.toml` for the pyrlyn ketch
  registry once releases exist.
- Changelog from conventional commits (git-cliff) once there is a first release.

## 12. Repository reset

This pull request removes the previous implementation and everything tied to it:
`crates/`, `fuzz/`, `tests/`, `docs/` (old guides and translations), `dashboard/`, `web/`,
`scripts/`, `.cargo/`, `.moon/`, `moon.yml`, `.serena/`, `Cargo.toml`, `Cargo.lock`,
`rust-toolchain.toml`, `dist-workspace.toml`, `cliff.toml`, `ketch.toml`,
`sonar-project.properties`, `CLAUDE.md`, `AGENTS.md`, `CHANGELOG.md`, the old planning notes
(`plan.md`, `done.md`, `research.md`, `report.html`, `roadmap.md`, `todo.md`, `ideas.md`,
`toolchain.md`), `.github/build-setup.yml` and the old build workflows (`ci.yml`, `perf.yml`,
`release.yml`, `release-variants.yml`, `release-failure.yml`, `review.yml`, `bump.yml`,
`sync-docs.yml`). The history stays in git.

Kept: `LICENSE` and `PRICING.md` (licensing), `brand/` (logos and colour tokens),
`.coderabbit.yaml` and `renovate.json` (org-wide configuration), `.github/FUNDING.yml`,
`pipeline.yml`, `coderabbit-issues.yml`, `release-ci-alert.yml` with `.github/infra.yml` reduced to
workflow and licence checks, `hk.pkl` and the jactionlint configuration, `.editorconfig`.
Rewritten: `README.md`, `CONTRIBUTING.md`, `mise.toml`, `.gitignore`, `.gitattributes`.

## 13. Risks and open questions

Risks:

- **cinterop with C++ llama.cpp per target.** Static linking of the C++ runtime and Metal differs
  per OS. Phase 4 builds llama.cpp with the Kotlin/Native toolchain, so the Linux targets
  cross-build from macOS; Windows (`mingwX64`) has no llama.cpp build yet, and a GPU backend other
  than Metal needs its own SDK on the build machine.
- **Ktor on native.** Engine coverage differs by OS (TLS in CIO client, Curl availability on
  Linux, WinHttp on Windows). Mitigation: phase 0 spike tests on every target.
- **Kotlin/Native memory and threads.** Native pointers must never be touched from two threads at
  once; every loaded model has a serial dispatcher and all its llama.cpp calls go through it.
- **Binary size and start-up time** with Ktor server + client in one executable. Measure in
  phase 0; split `runa __daemon` into its own executable target if needed.
- **llama.cpp API churn.** Pin a tag, upgrade deliberately, keep `:engine-llama` the only module
  that sees the C API.
- **CI runs only on Linux.** GitHub Actions runs again (it started no run between 2026-10-26 and
  phase 3), and `ubuntu-24.04 (linuxX64)` passes for phases 1 to 3. The `macos-14 (macosArm64)` job
  is cancelled before it starts: GitHub answers "The configuration 'macos-14-arm64-us-default' is
  not supported", because the macOS 14 images were retired on 2026-11-02
  ([actions/runner-images#15226](https://github.com/actions/runner-images/issues/15226), read
  2026-11-22). Until the job gets a runner that exists (the hosted `macos-15` or `macos-26` label,
  or the dedicated Mac through the `MACOS_RUNNER` variable), macOS arm64 is checked locally with
  `./gradlew build`.
- **Linux system dependencies** (found in phases 0 and 1). The binary is dynamically linked to
  glibc and needs `libcrypt.so.1` (Fedora: `libxcrypt-compat`), glibc's gconv modules for Ktor
  (openSUSE minimal images: `glibc-locale-base`) and a system CA bundle for HTTPS (Debian and
  Ubuntu minimal images: `ca-certificates`). Packaging (phase 10) must declare or remove them.

Open questions for Ivan:

1. Product shape: own engine with daemon (this plan) first, or the LM Studio client
   (`--backend lmstudio`, phase 8) first as a quick win? The plan keeps both; the order can swap.
2. Is Windows (`mingwX64`) needed for 1.0, or later?
3. GPU backends beyond Metal and CPU for 1.0 (CUDA, Vulkan)?
4. Should features from the previous version come back after 1.0 (`fit` forecasting, cloud
   OpenAI/Anthropic providers, media/ASR, adaptive memory, MCP tool loop)?
5. Licence stays GPL-3.0 with the commercial option in `PRICING.md`?
