# runa

https://github.com/pyrlyn/runa

A single CLI that runs AI models locally (GGUF via ggml/llama.cpp) or through OpenAI/Anthropic APIs; fit checker, three compute modes, adaptive memory, OpenAI-compatible server.

| # | Status | Priority | Complexity | Readiness | Agent |
| --- | --- | --- | --- | --- | --- |
| P14.2 | in progress | high | S | ready | Muse Spark |

## Tasks

### P14.2. v0.1.0 release red: dist builds miss mise/zig, vulkan variant misses glslc

Plan: fix dist build-setup (mise install incl. zig, like CI) + install
glslc/shaderc in release-variants linux-vulkan; delete + re-push v0.1.0
(same commit, CI green), wait for Release + variants, verify
`ketch install pyrlyn/runa`.

Machine check: `gh release view v0.1.0` lists 3 portable archives +
ketch install runa works.

## Reference

`runa` is a single command-line binary that runs AI models locally (GGUF via ggml/llama.cpp)
or through the OpenAI and Anthropic APIs, with:

- deep-thinking control (on/off, token budget, effort level) that works the same way for local and cloud models;
- audio and video input (native audio/vision models, or ASR → text, or cloud);
- three compute modes — `cpu`, `gpu`, `hybrid` — plus `auto`;
- a fit checker (`runa fit`) that says *whether* a model runs on this machine and *how fast*, before downloading it;
- adaptive memory management: shrink toward a floor when there is no request or job, grow (bounded) when a task is heavy;
- a cooperative agent task protocol: agents claim only `free` tasks, mark `in progress` + agent name + start time, and release on stop/done; taking an `in-progress` task requires asking first;
- config files, profiles, an OpenAI-compatible server.

Rust owns orchestration; the compute core is C (ggml); own kernels in `runa-kernels` are Zig where Zig is better, otherwise C/`.S`, and only where a benchmark proves a win.

Companion documents: `research.md` (analysis, analogs, formulas, fact-check ledger, in English) and `report.html`. Agent coordination lives in `AGENTS.md` with the claim registry in `docs/tasks.md`; project overview in `README.md`; memory/task public-method docs in `docs/memory.md`.

---

## 1. Decisions

| ID | Decision | Why |
|----|----------|-----|
| D1 | **Language split.** Rust for CLI, config, planner, media pipeline, cloud clients, server. C (ggml, whisper.cpp) for the compute core through FFI. Own kernels live in `runa-kernels` behind a **benchmark gate: a kernel is merged only if it beats the ggml path on the same op and hardware by ≥ 5 %** (end-to-end tok/s or ≥ 2× on the isolated op). Prefer Zig for those kernels (D23); C/`.S` only where Zig is worse or missing. | ggml already *is* hand-tuned C/asm per platform (NEON, i8mm, SVE, AVX2/512-VNNI, AMX). Rewriting it is negative value; adding kernels where it is weak is positive value. Stable Rust has no SVE/SME/AMX intrinsics (nightly only, `std::simd` unstable as of 1.98). |
| D2 | **Engine.** llama.cpp/ggml through the `llama-cpp-2` crate (0.1.133, features `cuda`, `metal`, `vulkan`, `openmp`, `native`, `mtmd`) is the primary backend. `mistral.rs` (0.9.3, MIT) is an optional, feature-gated second backend (`--features mistralrs`) for models ggml cannot run (safetensors-only or omni models without mtmd support). candle/burn are not used for LLM inference. | llama.cpp has the widest model, quant and backend coverage (CUDA, Metal, Vulkan, ROCm, SYCL, OpenCL, Hexagon, RPC), auto-fit (`--fit`, `llama_params_fit`), mtmd (audio + image + video since June 2026), speculative decoding, grammars, reasoning budgets. Every serious local runner (Ollama, LM Studio, Jan, koboldcpp, LocalAI, llamafile, Lemonade) is built on it. |
| D3 | **Model formats.** GGUF for local inference. safetensors only through the mistral.rs feature. Model references: local path, `hf:<repo>:<file-or-quant>`, config alias. | One format keeps the fit checker exact (tensor sizes come from the GGUF header). |
| D4 | **Three modes = one placement planner.** `--mode cpu|gpu|hybrid|auto` produces a `Placement { n_gpu_layers, tensor_overrides, kv_device, mmproj_device }`. `hybrid` = partial layer offload and/or MoE expert tensors kept on CPU (`ffn_*_exps` overrides). `auto` = the planner's best plan from the fit checker. | This is exactly how llama.cpp exposes hardware (`-ngl`, `-ot`, `--n-cpu-moe`, `--device`). One planner replaces three code paths. |
| D5 | **Fit checker is a standalone library** (`runa-fit`) with no engine dependency: GGUF header parser (local file or HTTP range request), hardware probe, analytic estimator with an explicit uncertainty band. When the model file is local and the engine is compiled in, an **exact mode** asks the engine allocator for real numbers. | Pre-download checks need the analytic path; post-download checks deserve exact numbers. Nobody in the analog set does the first one from the CLI. |
| D6 | **Speed prediction = bandwidth model + calibration.** `decode tok/s = eff(device, quant) × BW / bytes_per_token`. `eff` starts from a default table and is updated from measured runs stored in a local SQLite calibration DB. Predictions are always printed as a range. | Decode is memory-bound; the "speed of light" model is accurate to a constant factor that depends on device and kernel quality. Measuring that factor once per device beats any formula. |
| D7 | **Thinking primitive.** One `ThinkConfig { Off \| On \| Budget{tokens, grace} \| Effort(Low\|Medium\|High\|Max) }`. Local: budget forcing in our own sampling loop (count reasoning tokens after the model's think-open token, inject the think-close token at the budget, soft logit bias before it). Cloud: OpenAI `reasoning.effort`; Anthropic `thinking: {type: "adaptive"}` + `output_config.effort`. Output always carries `Event::Reasoning` and `Event::Text` separately. | Same semantics as llama.cpp's `--reasoning-budget`, `--reasoning-budget-grace-tokens`, `--reasoning-budget-soft-ratio`, `--reasoning-budget-message` (common/reasoning-budget.cpp, PR #25961) and vLLM's reasoning budget (PR #37112), but usable from a library and mapped onto both cloud APIs. |
| D8 | **Media pipeline in Rust, encoders in C.** Decode with `symphonia`/`hound` → f32 mono 16 kHz (`rubato`); video via `ffmpeg-sidecar` (binary, not linked) → sampled frames + audio track. Three routes: **native** (audio/vision mmproj through mtmd), **asr** (whisper.cpp via `whisper-rs`, or sherpa-onnx Parakeet behind a feature) → text, **cloud** (OpenAI `input_audio` / images; Anthropic images + transcript). Video = ≤ `max_frames` frames at `fps` with scene-change keyframes, plus the audio route. | Neither cloud API takes video natively; Anthropic takes no audio. Frame sampling + transcript is what every VLM pipeline does internally (~1 fps, ≤ 32–64 frames). |
| D9 | **Cloud clients.** One `Backend` trait for local and cloud. OpenAI through `async-openai` (Responses API, streaming, `base_url` override for compatible providers). Anthropic through a thin `reqwest` + SSE client of our own (no official Rust SDK). | Two thin adapters, one interface, mock-tested. |
| D10 | **CLI and config.** `clap` subcommands: `run`, `chat`, `fit`, `pull`, `serve`, `doctor`, `bench`, `models`, `config`. Config layering with `figment`: built-in defaults < `~/.config/runa/config.toml` < `./runa.toml` < env `RUNA_*` < flags. Named profiles. | Standard, predictable, testable. |
| D11 | **Server.** `axum`; OpenAI-compatible `/v1/chat/completions`, `/v1/models`, `/v1/embeddings`, `/v1/audio/transcriptions`; reasoning in `reasoning_content`; request fields `reasoning_effort` and `reasoning_budget_tokens` (the same names llama-server uses, so clients written for it keep working). Anthropic-compatible `/v1/messages` in P6. | Lets existing SDKs and tools use `runa` unchanged. |
| D12 | **No silent fallback.** `on_unfit = error \| cpu \| cloud:<backend>:<model>` is explicit in config. Every run prints one verdict line (placement, memory, predicted speed) before the first token. | Ollama's most-reported pain is silent CPU fallback. |
| D13 | **Platform tiers.** Tier 1: macOS arm64 (Metal), Linux x86_64 (CUDA, Vulkan, CPU). Tier 2: Linux aarch64, Windows x86_64 (CUDA/Vulkan). Tier 3 (manual, P9.4): Hexagon/OpenVINO NPUs as probe-only stubs — no ggml backend, no CI runners, never default. Backends are cargo features; `runa doctor` lists what the binary was built with. | Matches where the hardware table says local inference actually happens; Tier 3 waits on upstream features + SDKs + on-device validation. |
| D14 | **Kernel candidates** (ordered by expected payoff): (1) sampling over 150k-token vocabularies (top-k/top-p/min-p), (2) image preprocessing (resize, normalize, patchify), (3) audio front-end (resample, mel), (4) quantized mat-vec on SME2/AMX only where ggml lacks a path on the target at that time. | Profiling first; these are the ops that live outside ggml's hot loop or where ggml is known to be generic. |
| D15 | **Every task has a machine check.** Benchmarks via `criterion` and `runa bench --json`; fit estimates are golden-tested against llama.cpp's own allocator logs (±5 %). Perf CI fails on > 3 % regression. | The plan is meant to be executed by agents; a check is the definition of done. |
| D16 | **Version pins.** Rust toolchain, `llama-cpp-2`, whisper-rs, async-openai pinned in `Cargo.lock` and `docs/versions.md`; llama.cpp upgraded through the benchmark gate. | Upstream moves ~50 builds/week; drift must be deliberate. |
| D17 | **Adaptive memory.** A `MemoryManager` (`runa-memory`) shrinks toward a configured floor when there is no request or job for `idle_timeout_s` (release prompt cache, encoder buffers, draft model, shrink pools; never unloads the active model), and grows — bounded by `max_growth_mib` and the fit verdict + margin — when a task is heavy (`on_heavy`/`grow_for`). Every transition is logged with before/after RSS. | Idle servers and CLIs should not sit on gigabytes of cache; heavy jobs (large ctx, batch, media) should pre-grow once instead of OOM-ing mid-run. |
| D18 | **Cooperative task claims.** Plan tasks live in `docs/tasks.md` with status `free` \| `in progress` (+ agent name + `started_at` UTC, RFC 3339). An agent takes a task by atomically marking it `in progress`; on stop or done it clears the claim back to `free` (completion itself is tracked by moving the task to `done.md`). Agents take only `free` tasks; taking an `in-progress` task requires asking the owner (or the human) first and proceeding only on explicit approval. Protocol and ask-flow are defined in `AGENTS.md`. | The plan is executed by parallel agents; without claims two agents redo or collide on the same task. Ask-before-steal keeps collaboration explicit. |
| D19 | **Polyglot escape hatch.** Rust owns orchestration (D1); another language may own a component only where a benchmark proves it beats the Rust path on the same hardware (same gate as D1: ≥ 5 % end-to-end or ≥ 2× on the isolated op). Non-Rust code lives behind a Rust-owned interface, ships with equivalence tests, and is re-evaluated at each toolchain upgrade. | ggml's hot loop is already hand-tuned C/asm; media codecs, SIMD front-ends or vendor SDKs (e.g. sherpa-onnx, platform ML APIs) can beat a pure-Rust implementation. The gate keeps polyglot code justified instead of fashionable. |
| D20 | **Parallelism by default.** I/O-bound work is async, CPU-bound data-parallel work uses threads (rayon), builds and moon pipelines use all cores; no manual thread caps or serial fallbacks without a measurement. Parallelism that doesn't move wall-clock time is removed (profiled via D15). | Everything around the serial decode loop parallelizes (media decode, batch ingest, fit checks, CI matrix, task graph). Defaults already do this (cargo jobs = cores, moon concurrent targets); the rule stops hand-rolled serial code. |
| D21 | **mise owns tool installs.** Every developer/CI tool (Rust, moon, ffmpeg, python fixtures, node for SDK smoke tests, cargo-dist, Zig for `runa-kernels`, …) is pinned in `mise.toml` and installed with `mise install`. `rust-toolchain.toml` stays the rustup source of truth; moon's rust plugin mirrors the pin. CI bootstraps with mise; no toolchain installs outside mise in workflows. | One bootstrap command per machine; no dev/CI version drift; D16 pins stay in one visible place. |
| D22 | **moon orchestrates the monorepo.** moon v2 (WASM plugin toolchains; rust plugin for graph/hashing/caching) provides the task graph over the cargo workspace: `.moon/workspace.yml`, `.moon/toolchains.yml`, `.moon/tasks/*.yml`, root `moon.yml`. Cargo remains the build source of truth — moon tasks wrap `cargo build/test/clippy/fmt`; no build logic is duplicated in moon config. | Cargo knows how to build Rust; moon knows how to skip, cache and parallelize workspace-wide work (affected-only runs, remote cache later). Each does what it's best at (D19 applied to our own tooling). |
| D23 | **Zig for own kernels (where better).** Do not rewrite ggml/whisper. New `runa-kernels` code is Zig (C ABI `export fn`, `@Vector` SIMD, slices instead of `malloc`/`qsort`) when Zig is the better tool. Keep C or `.S` only where Zig is worse or missing: wrapping C headers, SME2/AMX assembly, or a measured C path that already wins. Same D1 merge gate. Zig is pinned in `mise.toml` (D21). Protocol for agents: `AGENTS.md` §7. | Zig's `@Vector` is portable SIMD without a `.c` per ISA; sampling/top-k is simpler than C. Intrinsics ggml already owns stay C. SME/AMX still need `.S` until Zig covers them. |

---

## 2. Repository layout

```text
runa/
├── Cargo.toml                 # workspace, [profile.release] lto="fat", codegen-units=1
├── rust-toolchain.toml        # 1.98
├── mise.toml                  # all tool pins (D21): rust 1.98, moon 2.5.4
├── moon.yml                   # workspace-root project `root` (repo-wide checks)
├── .moon/
│   ├── workspace.yml          # projects, versionConstraint, vcs (D22)
│   ├── toolchains.yml         # rust pin mirror (D22)
│   └── tasks/rust.yml         # shared build/test/clippy/fmt (D20, D22)
├── crates/
│   ├── runa/                  # binary: clap CLI, figment config, output, server (axum)
│   ├── runa-core/             # Backend trait, Request/Event types, ThinkConfig, Mode, errors
│   ├── runa-engine/           # llama-cpp-2 wrapper: load, placement, sampling loop, mtmd, state save
│   ├── runa-fit/              # gguf header (local/remote), hw probe, estimator, planner, calibration db
│   ├── runa-media/            # audio/video decode, resample, frame sampling, ASR bridge (whisper-rs)
│   ├── runa-cloud/            # openai (async-openai) + anthropic (reqwest+SSE) adapters, price table
│   ├── runa-kernels/          # own kernels: Zig (C ABI) preferred, C/`.S` where Zig is worse (D23)
│   └── runa-memory/           # adaptive memory manager (idle shrink / heavy grow) + task-claim registry
├── AGENTS.md                  # agent coordination protocol: claim only free tasks, ask for in-progress
├── README.md                  # project overview and doc index
├── docs/                      # adr/, fit.md, thinking.md, media.md, memory.md, tasks.md, config.md, baselines.md, versions.md
├── benches/
└── tests/
    ├── fixtures/              # tiny GGUFs, audio/video clips, recorded API responses
    └── e2e/
```

---

## 3. Target metrics for v1.0

| # | Metric | Target |
|---|--------|--------|
| M1 | `runa fit` on a local GGUF | < 300 ms; memory estimate within ±5 % of llama.cpp's actual allocation (exact mode: ±1 %) |
| M2 | `runa fit` on a remote HF GGUF | < 3 s, no download (header via HTTP range) |
| M3 | Speed prediction error | ≤ ±30 % cold; ≤ ±15 % after 3 calibration runs on the device |
| M4 | Decode throughput vs `llama-cli` | ≥ 95 % on the same build and flags (runa adds no overhead) |
| M5 | Cold start to first token, 8B Q4_K_M on Apple M-series | < 2 s (mmap) |
| M6 | Thinking budget enforcement | reasoning tokens ≤ budget + grace in 100 % of runs (Qwen3, DeepSeek-R1-distill, gpt-oss) |
| M7 | Audio → text, 1 min of speech on CPU | < 5 s (whisper base / parakeet) |
| M8 | Video preprocessing, 30 s clip → ≤ 32 frames + transcript | < 3 s |
| M9 | Server compatibility | OpenAI Python SDK and Anthropic Python SDK smoke tests pass unmodified |
| M10 | Binary | single static-ish binary per platform, CPU build ≤ 40 MB, no telemetry |
| M11 | Idle memory shrink | with no request/job for `idle_timeout_s`, RSS drops to ≤ floor + 10 % (caches/buffers released, model stays loaded) |
| M12 | Task-claim integrity | no task is ever held by two agents; every `in progress` row has agent + `started_at`; stop/done always clears the claim |
| M13 | moon/mise parity | `moon run :test` matches `cargo test --workspace`; `moon run :clippy` matches CI clippy; fresh `mise install` yields pinned rust + moon; `moon run root:lint-tasks` green |

---

## 4. Phases

Phase order is deliberate: **the fit checker (P1) comes before the engine (P2)** because it is the differentiator and it needs no GPU to develop and test.

### P0 — Skeleton, spikes, baselines

### P1 — Fit checker

### P2 — Engine and the three modes

### P3 — Thinking and cloud

### P4 — Audio and video

### P5 — Kernels and speed

### P6 — Server, packaging, release

### P7 — Adaptive memory + agent task protocol

### K — Monorepo, tooling & performance discipline (cross-cutting)

K runs alongside P0–P7 (first slice lands with the P0 skeleton).
Decisions: D19 (polyglot gate), D20 (parallelism default), D21 (mise),
D22 (moon), D23 (Zig for own kernels). Metric: M13. Finished as `K1`–`K5` in `done.md`.

## 5. Core types (sketch)

```rust
// runa-core
pub enum Mode { Cpu, Gpu, Hybrid, Auto }

pub enum ThinkMode {
    Off,
    On,
    Budget { tokens: u32, grace: u32 },
    Effort(Effort),
}
pub enum Effort { Low, Medium, High, Max }
pub struct ThinkConfig { pub mode: ThinkMode, pub show: bool }

pub enum Media { Image(ImageBuf), Audio(Pcm16k), Video { frames: Vec<(f32, ImageBuf)>, audio: Option<Pcm16k> } }

pub struct Request {
    pub messages: Vec<Message>,
    pub media: Vec<Media>,
    pub think: ThinkConfig,
    pub sampling: Sampling,
    pub max_tokens: Option<u32>,
}

pub enum Event {
    Reasoning(String),
    Text(String),
    Usage(Usage),
    Done(StopReason),
}

#[async_trait::async_trait]
pub trait Backend: Send + Sync {
    fn capabilities(&self) -> Caps; // audio_in, video_in, images, thinking_budget, effort
    async fn generate(&self, req: Request) -> anyhow::Result<BoxStream<'static, anyhow::Result<Event>>>;
}

// runa-fit
pub enum Fit { Gpu, Hybrid { gpu_layers: u32, of: u32, experts_on_cpu: bool }, Cpu, No { reason: String } }

pub struct Verdict {
    pub fit: Fit,
    pub bytes: PerDevice,          // weights, kv, compute, mmproj, margin per device
    pub speed: SpeedEstimate,      // decode_lo/hi, prefill_lo/hi, ttft_s, confidence
    pub warnings: Vec<Warning>,
    pub suggestions: Vec<Suggestion>,
}

// runa-memory (D17, D18; full method docs in docs/memory.md)
pub enum LoadState { Idle, Normal, Heavy }

pub struct MemoryPolicy {
    pub idle_timeout_s: u64,
    pub floor_mib: u64,
    pub max_growth_mib: u64,
}

pub struct Usage {
    pub rss_mib: u64,
    pub budget_mib: u64,
    pub state: LoadState,
}

pub trait MemoryManager: Send + Sync {
    fn current_usage(&self) -> Usage;
    fn on_idle(&self);                          // no request/job for idle_timeout_s: shrink toward floor
    fn on_heavy(&self, demand_mib: u64);        // heavy task: pre-grow within ceiling
    fn shrink_to_floor(&self);
    fn grow_for(&self, demand_mib: u64) -> Result<(), MemoryError>;
}

pub struct TaskClaim {
    pub task_id: String,    // e.g. "P2.6"
    pub agent: String,      // agent name
    pub started_at: String, // RFC 3339 UTC
}

pub enum TaskStatus {
    Free,
    InProgress { agent: String, started_at: String },
}

pub trait TaskRegistry: Send + Sync {
    fn list_free(&self) -> Vec<String>;
    fn status(&self, task_id: &str) -> Option<TaskStatus>;
    fn claim(&self, task_id: &str, agent: &str) -> Result<TaskClaim, ClaimError>;
    fn release(&self, task_id: &str, agent: &str) -> Result<(), ClaimError>;
}
```

---

## 6. Config example (`runa.toml`)

```toml
[defaults]
mode = "auto"            # cpu | gpu | hybrid | auto
ctx = 8192
kv = "f16"               # f16 | q8_0 | q4_0
fit_margin_mib = 1024
on_unfit = "error"       # error | cpu | cloud:anthropic:claude-sonnet-5

[think]
mode = "on"              # off | on | budget | effort
budget = 2048
grace = 64
effort = "medium"        # low | medium | high | max
show = true

[models.qwen]
source = "hf:unsloth/Qwen3-8B-GGUF:Q4_K_M"

[models.voxtral]
source = "hf:ggml-org/Voxtral-Mini-3B-2507-GGUF:Q8_0"
mmproj = "auto"

[cloud.openai]
model = "gpt-5"
base_url = "https://api.openai.com/v1"   # or an OpenAI-compatible provider

[cloud.anthropic]
model = "claude-opus-5"
effort = "high"

[media]
audio_route = "auto"     # auto | native | asr
asr_model = "whisper:large-v3-turbo"
video_fps = 1
video_max_frames = 32

[memory]
idle_timeout_s = 300     # no request/job for this long -> shrink toward floor
floor_mib = 512          # shrink target for caches/buffers (model stays loaded)
max_growth_mib = 4096    # max pre-grow beyond current use for a heavy task

[system]
max_load_percent = 80    # max share of total RAM/CPU this app may use (1..=100);
                         # startup prints a warning: with the value to set when over

[agents]
registry = "docs/tasks.md"   # task-claim registry: free | in progress (+ agent, started_at)

[profiles.fast]
think = { mode = "off" }
sampling = { temperature = 0.2 }

[server]
host = "127.0.0.1"
port = 8080
models = ["qwen", "voxtral"]
```

---

## 7. `runa fit` example output

```text
$ runa fit hf:unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M --ctx 16384 --kv q8_0

Model    Qwen3-30B-A3B · Q4_K_M · 30.5B total / 3.3B active · 48 layers · 17.3 GiB weights
Machine  Apple M4 Pro · 48 GiB unified · 273 GB/s (spec) / 221 GB/s (measured) · Metal

Memory   weights 17.3 + KV 0.8 (q8_0, 16k) + compute 0.5 + mmproj 0.0 + margin 1.0 = 19.6 GiB
         usable 36.0 GiB (iogpu.wired_limit) → fits with 16.4 GiB to spare

Verdict  FITS · gpu · all 48 layers on GPU · confidence: high (header exact, speed calibrated ×3)
Speed    decode 48–62 tok/s · prefill 550–800 tok/s · first token for a 2k prompt ≈ 3 s
Note     thinking on, budget 2048 → expect ~40 s of reasoning per answer at 50 tok/s

$ runa fit hf:unsloth/Qwen3-235B-A22B-GGUF:Q4_K_M --ctx 8192
Verdict  NO · needs 133 GiB, machine has 36 GiB usable
Try      UD-Q2_K_XL (~82 GiB) still does not fit · Qwen3-30B-A3B Q4_K_M fits · on_unfit=cloud:anthropic:claude-sonnet-5
```

---

## 8. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| `llama-cpp-2` lags upstream or breaks its API | pin + monthly upgrade through the benchmark gate; fallback is our own `bindgen` over `llama.h`/`mtmd.h` in `runa-engine/sys` |
| Compute-buffer estimate drifts across llama.cpp versions | calibrate per pinned version; exact mode for local files; +15 % safety margin |
| Speed model wrong on new hardware | uncertainty band always shown; calibration after the first run; never claim better than ±25 % cold |
| mtmd lacks a model's audio/video path | mistral.rs feature or ASR route; `runa fit` prints `audio: via ASR` so the user knows |
| Cloud API drift (Anthropic thinking params, OpenAI Responses changes) | adapters versioned per API date; recorded fixtures; live smoke tests behind an env flag |
| Kernel work absorbs time without gains | hard gate ≥ 5 %; P5 time-boxed; scalar reference always shipped |
| Licenses | llama.cpp MIT, whisper.cpp MIT, sherpa-onnx Apache-2.0, ffmpeg as a separate binary (LGPL/GPL, not linked); `runa` MIT OR Apache-2.0 |
| Silent behaviour differences vs llama-cli (templates, samplers) | golden tests at temperature 0 against `llama-cli` output for 5 models |
| Memory thrash (shrink/grow oscillation on bursty load) | hysteresis: grow immediately, shrink only after a full `idle_timeout_s` of silence; transitions logged; soak test in P7.2 |
| Stale task claims (agent dies holding `in progress`) | claims carry agent + `started_at`; takeover requires asking the owner (or human) first; CI lint surfaces claims older than 7 days |


---

## Note 2026-09-17 — testing library candidates

Shared catalog: [`listepo/rust.md`](../../rust.md) → *Testing candidates*.
Catalog only — no blanket dependency adds.

Fits for runa (1–3):

1. `tokio-test` — optional for `runa-cloud` async unit tests beyond
   `#[tokio::test]` + `wiremock`.
2. `fake` — only if synthetic fit/cloud fixtures beat domain builders +
   `proptest` (already on `runa-fit`).
3. Otherwise **none new — covered** by `assert_cmd`, `assert_fs`, `insta`,
   `predicates`, `pretty_assertions`, `trycmd`, `wiremock`, `tempfile`,
   `proptest`, `rstest`, `criterion`.

Skip `testcontainers` unless a Docker-backed engine/cloud e2e is required;
skip extra fuzzers and `mockall` until a trait-heavy seam needs them.

### Runa audit — features

Findings from the 2026-09-20 features-only audit (English). Local tree: `listepo/apps/runa`; remote: `pyrlyn/runa`.

#### Crates today

- `crates/runa` — CLI binary: clap surface in `main.rs`, plus `serve.rs`, `daemon.rs`, `mcp.rs`, `pull.rs`, `bench.rs`, `fit.rs`, `tui.rs`, `pool.rs`.
- `crates/runa-core` — `Backend` trait, `Request`/`Event`, `ThinkConfig`, `Mode`, errors (no engine dependency).
- `crates/runa-engine` — `llama-cpp-2` wrapper: load, placement, sampling, mtmd, state save; features such as `rpc`.
- `crates/runa-fit` — GGUF header (local/HTTP range), hardware probe, estimator, planner, calibration DB (must not depend on the engine, D5).
- `crates/runa-cloud` — OpenAI + Anthropic adapters, price table (`docs/prices.toml`).
- `crates/runa-media` — audio/video decode, resample, frames, whisper-rs ASR.
- `crates/runa-memory` — adaptive memory (idle shrink / heavy grow) + task-claim registry.
- `crates/runa-kernels` — own kernels (Zig preferred, C/`.S` fallback) behind the ≥5% bench gate (D1/D23).

#### Commands today

From `crates/runa/src/main.rs` `Commands`: `run` / `chat`, `fit`, `pull` / `models`, `bench`, `serve`, `daemon`, `doctor`, `media`, `tasks`.

#### Gaps vs llama.cpp / Ollama

- Release broken: active card **P14.2** — `v0.1.0` Release workflow red (mise/zig missing in dist; vulkan needs glslc); no installable artifacts / `ketch install` yet.
- GPU features (Metal/CUDA/Vulkan) stay opt-in (D13) and are not yet forwarded into the default user path (`docs/versions.md`).
- No Modelfile / curated library UX like Ollama.
- No MLX path (ggml/llama.cpp + optional mistral.rs only).
- `--rpc` needs the `rpc` cargo feature — not a first-class distributed runner.
- CLI/TUI only — no desktop GUI.

#### Known limitations

- **P14.2** is the sole active plan row and blocks shipping.
- **D1/D23** kernel gate: own kernels only if they beat ggml by a measured margin.
- Agent **task protocol** (`docs/tasks.md` claims) is easy to confuse with model **MCP** tool-calling (`crates/runa/src/mcp.rs`).
- **Daemon** is Unix-socket oriented; Windows story weaker than launchd/systemd units.

#### Architecture

- **Good split:** `runa-core` Backend trait; `runa-fit` isolated from the engine; cloud adapters in their own crate.
- **Coupling:** most product surface concentrates in fat `crates/runa/src/main.rs` — hard to reuse serve/daemon without the binary.
- **Dual-backend drift risk:** GGUF (`llama-cpp-2`) + optional mistral.rs; pins live in `docs/versions.md`.
- **`runa-memory` mixes** runtime adaptive memory with the plan-file claim registry (different lifetimes/audiences).

### Runa audit — documentation

Findings from the 2026-09-20 documentation audit (English).

#### Adequacy (strong)

- Indexed in `docs/README.md`: `getting-started.md`, `guide.md`, `config.md`, `fit.md`, `thinking.md`, `media.md`, `structured.md`, `profiles.md`, `versions.md`, `baselines.md`, `perf-nightly.md`, `kernels.md`, `release.md`, `release-1.0.md`, `memory.md`, `tasks.md`, man pages `runa.1` / `runa-run.1`, ADRs under `docs/adr/`.
- Site at `https://pyrlyn.github.io/runa/` (`site/`, homepage set on the GitHub repo).
- Research companions: `research.md`, `report.html`.

#### Gaps

- **No Troubleshooting page** for install/release failures (exactly the pain of P14.2: red `v0.1.0` Release, missing mise/zig/glslc), GPU feature flags, daemon socket on Windows, MCP quoting, or HF pull errors.
- **`docs/versions.md`**: GPU accel (Metal/CUDA/Vulkan) “not yet forwarded” into the default path — user-facing docs still undersell how to turn GPU on after a successful install.
- **`docs/release.md` / `getting-started.md` assume a working GitHub Release installer** — today Release is red, so getting-started’s curl/Homebrew path is aspirational until P14.2 lands.
- **Site content is thin** relative to `docs/guide.md` (risk of docs/site drift; only `_index.md`-style landing in `site/content`).
- **Task-claim docs vs MCP**: `docs/tasks.md` / `memory.md` vs `docs/structured.md` MCP loop — easy for agents/users to confuse “tasks” (plan claims) with model tools; needs a one-line cross-link callout.
- **No FAQ** covering cloud keys, `on_unfit=cloud:`, calibration DB location, or `--max-load-percent`.
- **CHANGELOG** exists at repo root but is not linked prominently from `docs/README.md` / getting-started.

#### Suggested docs work

1. Add `docs/troubleshooting.md` (release install, GPU features, daemon, MCP, pull) and link it from `docs/README.md` + getting-started.
2. Update getting-started with a “Release status” note until P14.2 is green (or point at local `cargo build --release`).
3. Expand `versions.md` with a short “enable Metal/CUDA/Vulkan” recipe once features are forwarded.
4. Cross-link tasks vs MCP in `tasks.md` and `structured.md`.
5. Mirror key guide sections onto the Pages site or clearly defer to `docs/guide.md`.

### Runa audit — docs

Evaluation of `README.md`, `docs/`, `plan.md`, and `AGENTS.md` (2026-09-20, English).

#### Solid

- **README.md** — clear product pitch, install paths, first commands, link to the site and deeper docs.
- **docs/** — strong index in `docs/README.md`; user path via `getting-started.md` + `guide.md`; reference depth in `config.md`, `fit.md`, `thinking.md`, `media.md`, `structured.md`, `versions.md`, `memory.md`, `tasks.md`, man pages `runa.1` / `runa-run.1`, ADRs under `docs/adr/`.
- **plan.md** — decisions D1–D23, active-task table, task cards; now also carries features + documentation audit sections.
- **AGENTS.md** — agent operating rules, crate map, claim protocol, kernel language (D23), tool/model routing — usable as the agent runbook.

#### Missing

- **Troubleshooting** — no `docs/troubleshooting.md` (release/install failures, GPU flags, daemon on Windows, MCP quoting, HF pull errors).
- **API reference as one page** — no `docs/api.md` / OpenAPI-style HTTP reference for `runa serve`; behavior is split across `structured.md`, man pages, and clap help.
- **Command cheat-sheet** — no single COMMANDS.md listing every subcommand + flags (contrast with ketch’s `docs/COMMANDS.md`); users rely on `--help` and the guide.
- **FAQ** — cloud keys, `on_unfit=cloud:`, calibration DB location, `--max-load-percent` not gathered in one place.
- **Examples gallery** — examples live inside `guide.md`, not a dedicated `docs/examples/` set of copy-paste scripts.

#### Outdated / drift risks

- **getting-started / release.md** assume a green GitHub Release installer while **P14.2** still has `v0.1.0` Release red — curl/Homebrew paths are aspirational until that lands.
- **versions.md** still says GPU accel is “not yet forwarded” to the default path — docs lag a turnkey GPU story vs Ollama.
- **Site (`site/`)** is thinner than `docs/guide.md` — Pages vs repo docs can drift.
- **tasks.md vs MCP** — plan-claim “tasks” vs model tool loop in `structured.md` need an explicit cross-link to avoid agent confusion.

### Runa audit — tests

Findings from the 2026-09-20 tests-only audit (English).

#### Suites that exist

- **Workspace unit (`--lib`)**: dense coverage across crates — e.g. `crates/runa/src/config.rs`, `pull.rs`, `serve.rs`, `mcp.rs`, `daemon_proto.rs`; `runa-core` (`think.rs`, `reason.rs`, `backend.rs`); `runa-fit` (`planner`, `speed`, `recommend`, `calibration`); `runa-engine` (`placement`, `load`, `lora`, `structured`); `runa-cloud` (`openai`, `anthropic`, `secrets`); `runa-media` (`asr`, `video`); `runa-memory` (`memory.rs`, `registry.rs`); `runa-kernels` (`lib.rs` + reference).
- **Crate integration tests**: `crates/runa-fit/tests/{gguf,remote}.rs`; `crates/runa-engine/tests/{load,generate,rpc}.rs`; `crates/runa-cloud/tests/openai.rs`.
- **Binary / CLI tests (`crates/runa/tests/`)**: `e2e.rs` (run/chat/fit/serve/daemon/MCP/KV/LoRA/RPC/auto-unfit/calibration); `doctor.rs`; `bench.rs`; `media.rs`; `pull.rs`; `cloud.rs`; `secrets.rs`; `security.rs`; `trycmd.rs` + `tests/cmd/{run,chat,serve,daemon}-help.toml` (help stdout snapshots only).
- **Fixtures**: `tests/fixtures/` (GGUF + task lint fixtures); cleanup via `scripts/test-with-fixture-cleanup.sh` / moon `test-with-cleanup`.
- **Benchmarks / perf**: `crates/runa-kernels/benches/softmax.rs`; nightly `.github/workflows/perf.yml` runs `runa bench` + `scripts/perf-regress.py` vs `docs/perf-baseline.json` (>3% drop fails). CI also runs `perf-regress.py --self-test`.

#### Concrete gaps

- **trycmd thin**: only four help fixtures — no snapshots for `fit`/`pull`/`models`/`doctor`/`media`/`tasks`/`bench` help or error exits.
- **`tasks` CLI**: claim/release paths exercised mainly via `runa-memory` unit tests + `scripts/lint-tasks.py`; little/no binary e2e for `runa tasks`.
- **`media`**: probe/transcribe help covered; full ASR/video golden paths need fixtures and are lightly tested vs `run`/`serve`.
- **Cloud live paths**: `cloud.rs` / secrets reject inline keys; real OpenAI/Anthropic calls depend on env and are not a default CI gate.
- **Remote fit**: `crates/runa-fit/tests/remote.rs` has `#[ignore]` cases — network-dependent coverage often skipped.
- **GPU / Metal / CUDA / Vulkan**: CI does not run GPU inference e2e (Metal present on macos runners but tests stay CPU-oriented; CUDA build-only on ubuntu).
- **Windows**: build-only matrix entry — no `cargo test` / e2e / fixture download on windows-2022 (`if: runner.os != 'Windows'` on heavy steps).
- **Daemon**: e2e covers Unix socket (`daemon_serves_run_over_socket`); launchd/systemd install units and Windows daemon behavior largely untested in CI.
- **Mistral.rs optional backend**: doctor feature flags tested; full generate path under `--features mistralrs` not a first-class CI job.
- **Parallel / stress**: `serve_parallel_eight_chat` exists; little stress for concurrent `daemon` + multi-client, or two upgrades of the binary itself.

#### CI coverage

- **`.github/workflows/ci.yml`**: `push` to `main`, non-draft `pull_request` to `main`, `workflow_dispatch`. Draft PRs skipped.
- **Matrix**: macos-14, ubuntu-22.04, windows-2022. Shared: fmt, clippy `-D warnings`, build. **Tests**: `cargo test --workspace --lib` (non-Windows); fixture GGUF download; selected `runa` e2e (`serve_…`); doctor feature tests; rpc+trycmd; `runa-memory`; task lint; moon pipeline on ubuntu; perf-regress self-test.
- **`.github/workflows/review.yml`**: `/review` comment — ubuntu-only fast gate (fmt/clippy/build/lib tests/memory/registry lint).
- **`.github/workflows/perf.yml`**: nightly bench regression (not every PR).
- **Release workflows**: separate; currently the shipping blocker is **P14.2** (dist Release red), not the unit/e2e matrix itself.

#### Quality concerns

- **Fixture / HF dependency**: e2e and perf need `tests/fixtures/qwen2-0_5b-instruct-q4_0.gguf`; anonymous HF curl can flake (“Disconnected”) — CI comments already call this out.
- **Heavy e2e**: `e2e.rs` is large (~39 tests) and model-backed — slow, sensitive to runner CPU; `--test-threads=1` used for some serve tests.
- **trycmd vs assert_cmd split** is intentional (`trycmd.rs` docs) but leaves most CLI error strings without golden files.
- **Ignored network tests** can rot silently (`remote.rs` `#[ignore]`).
- **Windows untested at runtime** — regressions in path/quoting/daemon will only show on contributor machines.

