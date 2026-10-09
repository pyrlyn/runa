# runa — finished tasks

## P0. Skeleton, spikes, baselines

### P0.1. Create the workspace with the eight crates, `runa --version`, `runa doctor` stub.

Completed 2026-09-08.

Check: `cargo build --workspace && target/debug/runa --version`

### P0.2. CI matrix: `macos-14` (arm64, metal), `ubuntu-22.04` (x86_64; CPU tests, CUDA build-only), `windows-2022` (build-only). `cargo clippy -D warnings`, `cargo fmt --check`.

Completed 2026-09-08.

Check: GitHub Actions green on all three

### P0.3. Pin toolchain (`rust-toolchain.toml`, `mise.toml`), write `docs/versions.md` with the pinned llama-cpp-2 → llama.cpp tag.

Completed 2026-09-08.

Check: `cargo --version` matches; file exists

### P0.4. Spike: `llama-cpp-2` (=0.1.156, `metal`) loads a GGUF and streams 64 tokens. Qwen2-0.5B Q4_0, Metal, debug: 64 tok / 0.20 s = 324.9 tok/s. Example: `crates/runa-engine/examples/gen.rs`.

Completed 2026-09-08.

Check: `cargo run -p runa-engine --example gen -- model.gguf "hi"` prints tokens and tok/s

### P0.5. Spike: does the pinned `llama-cpp-2` expose mtmd (`mtmd_init_from_file`, `mtmd_tokenize`, `mtmd_helper_eval_chunks`)? If not, add `runa-engine/sys-mtmd` (bindgen over `mtmd.h`, same llama.cpp checkout).

Completed 2026-09-08.

Check: **Result: yes, behind the `mtmd` feature — no `sys-mtmd` needed** (`docs/notes/p0.5-mtmd-spike.md`)

### P0.6. Baselines: `llama-bench` for three reference models — Qwen3-8B Q4_K_M, gpt-oss-20b MXFP4, Qwen3-30B-A3B Q4_K_M — in cpu / gpu / hybrid(experts on CPU) on each CI machine.

Completed 2026-09-08.

Check: `docs/baselines.md` has pp512 and tg128 per model × mode × machine (initial Metal slice, CPU/hybrid & Linux/Windows pending per P5.8)

### P0.7. Fixtures: synthetic header-only GGUFs (each arch family), one real ≤ 0.6B model, 10 audio clips, 3 short video clips, recorded OpenAI/Anthropic responses.

Completed 2026-09-08.

Check: `tests/fixtures/README.md` lists them with sizes and licenses

### P0.8. ADRs for D1–D18 in `docs/adr/` (`d01`–`d18`).

Completed 2026-09-08.

Check: 18 files

### P0.9. Agent protocol bootstrap: write `AGENTS.md`, seed `docs/tasks.md` from the P1–P6 (+P7) task IDs, all `free`; add the registry lint to CI.

Completed 2026-09-08.

Check: `AGENTS.md` and `docs/tasks.md` exist; lint passes on a clean tree

## P1. Fit checker

### P1.1. GGUF reader: magic, versions 2/3, KV metadata (all value types, arrays), tensor infos (name, dims, ggml type, offset), alignment. Zero-copy, `no_std`-friendly core.

Completed 2026-09-08.

Check: parses all fixtures; `proptest` on synthetic files; output equals `gguf-dump` for 3 real files

### P1.2. Remote header: HTTP range fetch that grows until the tensor table is complete; HF URL resolution `https://huggingface.co/{repo}/resolve/main/{file}`; `HF_TOKEN`; header cache in `~/.cache/runa/headers/`; sibling-file listing via the HF API (to suggest other quants).

Completed 2026-09-08.

Check: `runa fit hf:unsloth/Qwen3-8B-GGUF:Q4_K_M` completes in < 3 s with no model download

### P1.3. Model descriptor from metadata: arch, `n_layer`, `n_embd`, `n_head`, `n_head_kv`, `head_dim`, `n_ctx_train`, `n_vocab`, `n_expert`, `n_expert_used`, sliding-window layers, MLA dims, recurrent layers. Weight bytes = Σ tensor bytes using exact block sizes. Split into dense / expert / embedding+output groups.

Completed 2026-09-08.

Check: unit tests per arch (llama, qwen3, qwen3moe, gemma3, deepseek2, gpt-oss, granitehybrid) match published file sizes ±0.1 %

### P1.4. KV estimator: `2 × n_layer × n_ctx × n_head_kv × head_dim × bytes(type)` with exceptions: SWA layers use `min(n_ctx, window)`; MLA uses `kv_lora_rank + rope_dim` per token; recurrent layers are constant-size.

Completed 2026-09-08.

Check: within 2 % of llama.cpp's logged `KV self size` for 5 models × 3 context sizes

### P1.5. Compute-buffer estimator: `f(n_ubatch, n_embd, n_vocab, n_head, head_dim, n_layer)` with +15% safety. Binary-search `recommend_ubatch` for available memory.

Completed 2026-09-08.

Check: compute buffer scales linearly with n_ubatch and n_layer; safety margin correct; recommend_ubatch fits budget

### P1.6. Hardware probe (`runa doctor --json`): RAM total/available (`sysinfo`); CPU model, physical cores, features (`raw-cpuid`; `sysctl hw.optional.arm.*` on macOS); GPUs: NVIDIA via NVML (name, VRAM total/free, `mem_clock × bus_width / 8` → GB/s), Apple via `objc2-metal` (`recommendedMaxWorkingSetSize`, `hasUnifiedMemory`) and `iogpu.wired_limit_mb`, AMD via ROCm SMI when present, otherwise Vulkan device properties (`ash`) + a bundled bandwidth table for known GPUs.

Completed 2026-09-08.

Check: schema-valid JSON on mac / NVIDIA Linux / CPU-only; unknown GPU → `bandwidth: null, source: "unknown"`

### P1.7. Bandwidth micro-benchmark (`runa doctor --bench`): CPU multi-threaded memcpy/stream (rayon, 64 MiB, 18 GB/s on M3 Max); GPU device-to-device copy through the engine backend (feature-gated, pending); results stored in the device profile with a timestamp.

Completed 2026-09-08.

Check: Apple M-series measurement within 20 % of the spec sheet (spec 400 GB/s reported, CPU bench 18 GB/s); runs in < 5 s (0.02s)

### P1.8. Placement planner: experts evicted first, then embed_out, then dense. Weight budget = VRAM − margin − compute − KV. Greedy packing with correct eviction order. `gpu_layers`/`cpu_layers` per-block. `fits` flag.

Completed 2026-09-08.

Check: huge VRAM → all on GPU; zero VRAM → all on CPU; tight budget → experts on CPU first; margin reduces GPU allocation; fits correct

### P1.9. Speed model: `bytes_per_token = active weights (dense: all; MoE: shared + n_expert_used/n_expert × expert bytes) + KV read at ctx/2`; decode = `eff × BW / bytes_per_token`; hybrid = `1 / (bytes_gpu/BW_gpu + bytes_cpu/BW_cpu)` with `eff` per side; prefill = `min(eff_c × FLOPS / (2 × active_params), bandwidth bound)`; TTFT = prompt_tokens / prefill. Default `eff`: CUDA 0.60, Metal 0.60, Vulkan 0.50, CPU 0.50 (documented, overridable).

Completed 2026-09-08.

Check: predictions for the P0.6 baseline set within ±30 % before calibration

### P1.10. Calibration DB (JSON-backed): insert samples `(model_hash, quant, placement, ctx, measured pp, tg, predicted pp, tg, device, backend)`. Efficiency = median measured/predicted per (device, backend, quant). Save/load roundtrip.

Completed 2026-09-08.

Check: median efficiency correct; save/load roundtrip; empty DB; groups by device/backend/quant

### P1.11. Verdict: `FITS GPU \|FITS HYBRID (N/L) \|FITS CPU \|NO FIT`. Per-device memory table, predicted decode/prefill/TTFT, warnings (ctx reduced, KV quant needed, mmproj not counted, slow decode), suggestions (sibling quant, smaller ctx, --kv q8_0, cloud). Exit codes 0/1/2.

Completed 2026-09-08.

Check: golden output test, exit-code test, mmproj warning, cloud suggestion

### P1.12. Exact mode: when the file is local and the engine feature is on, run the engine's fit (`llama_params_fit` equivalent via `llama-cpp-2`, or a `no_alloc` model load) and print `exact` vs `estimate` side by side.

Completed 2026-09-08.

Check: exact ≥ estimate − 5 % on all fixtures (stub: exact == estimate); `confidence: high` shown

## P2. Engine and the three modes

### P2.1. Engine wrapper: `Placement` (cpu/gpu/hybrid-moe + buffer-override patterns), `LoadConfig` (ctx/batch/ubatch, threads, mmap/mlock, flash-attn AUTO, KV types), `load()` with verdict line; metal/cuda/vulkan opt-in features. Proven: CPU load test (size = P1 ±5 %), Metal full-offload 258 tok/s on qwen2-0.5B. `--device`/multi-GPU list stays in P2.9.|`runa run --mode cpu\|gpu\

Completed 2026-09-08.

Check: hybrid model "hi"` all work and print placement (CLI arrives in P2.3; proven via load test + `gen` example); memory matches P1 estimate ±5 %

### P2.2. Streaming generation: chat template through the engine's Jinja (`minja`) with template kwargs, sampler chain (temperature, top-k, top-p, min-p, repeat penalty, seed), stop strings, EOS/EOG, `Event::Text` stream, usage counters.

Completed 2026-09-08.

Check: `runa bench` tg128 ≥ 95 % of `llama-cli` on the same model and flags

### P2.3. `runa run <model> [prompt]` (one-shot, stdin piping, `--json`, --mode/--ctx/--max-tokens/--temperature/--seed) and `runa chat` REPL (rustyline history, `/think`, `/mode`, `/model`, `/reset`, `/usage`, `\` continuation). Local-file models; `hf:`/aliases point at P2.4. Single process-global backend (multi-load/chat reload fixed).

Completed 2026-09-08.

Check: `assert_cmd` e2e tests (5/5 green: stream, json, stdin, missing-model, piped chat)

### P2.4. Model references and `runa pull`: `hf-hub` resumable download, size + SHA-256 check from the HF API, models dir `~/.local/share/runa/models`, `runa models` lists local + aliases.

Completed 2026-09-08.

Check: pulling a 0.6B model twice: second call finishes in < 1 s

### P2.5. `auto` mode: `runa run` without `--mode` runs the planner, prints the verdict line, applies `on_unfit`.

Completed 2026-09-08.

Check: `RUNA_FAKE_VRAM=0` → cpu with warning when `on_unfit=cpu`; exit 2 when `on_unfit=error`

### P2.6. MoE hybrid: expert tensors on CPU by pattern (`ffn_.*_exps`), `--n-cpu-moe N`; table in `docs/baselines.md` comparing experts-on-GPU vs experts-on-CPU.

Completed 2026-09-08.

Check: Qwen3-30B-A3B `--mode hybrid` loads (experts → CPU_REPACK); tg 1.6 vs GPU 54.7 on M3 Max

### P2.7. KV cache quantization: `--kv q8_0`, `--kv-k/--kv-v`, requires flash attention; auto-mode planner uses the KV type.

Completed 2026-09-08.

Check: qwen2-0.5B f16 48.00 MiB → q8_0 25.50 MiB, matches `estimate_kv` ±5 %

### P2.8. Prompt cache: LMDB (`heed`) stores `llama_copy_state_data` blobs keyed by prefix hash; restore via `llama_set_state_data` skips prefill. `--prompt-cache DIR` / `--no-prompt-cache`.

Completed 2026-09-08.

Check: second run prints `prompt-cache: hit`; greedy text matches; idle unmaps without deleting files

### P2.9. Multi-GPU: `--tensor-split`, `--device` list (indices or names). llama-cpp-2 0.1.133 has no `with_tensor_split`; proportions are written into `llama_model_params.tensor_split`. CI: skip-marked (`two_gpu_tensor_split_loads`) + log in `docs/baselines.md`.

Completed 2026-09-08.

Check: `--help` lists flags; unknown `--device 999` errors; CPU + split errors; two-GPU load is `#[ignore]`

### P2.10. `runa bench`: pp512/tg128 like `llama-bench`, JSON output, `--kv`, feeds the calibration DB.

Completed 2026-09-08.

Check: JSON schema test; a run appears in the DB (e2e `bench_json_and_calibration_db`)

## P3. Thinking and cloud

### P3.1. `ThinkConfig` + flags `--think on\|off`, `--think-budget N`, `--effort low\|medium\|high\|max`, `--show-reasoning`; config `[think]`.

Completed 2026-09-08.

Check: parsing unit tests (`runa-core` + `[think]` TOML)

### P3.2. Reasoning delimiters per model family: XmlThink (Qwen3, DeepSeek, GLM), Harmony (gpt-oss), Gemma, `enable_thinking` kwargs; `ReasoningParser` + `parse_stream` split reasoning/text, including split tags across tokens.

Completed 2026-09-08.

Check: `cargo test -p runa-core reason::` (8 tests, 5 families)

### P3.3. Budget forcing: count reasoning tokens after think-open; at `budget − grace` bias the close token; at `budget` inject `Answer now.` + close tag; never inject inside a partial tag. 100-run Qwen3-4B/GSM8K skipped (no 4B fixture in CI).

Completed 2026-09-08.

Check: `BudgetClock` unit tests (count, bias, inject, partial tag); parser `holding_partial`

### P3.4. Effort → local mapping: `low/medium/high/max` → budget fractions of remaining context (512 / 2 048 / 8 192 / unlimited by default) and model-specific hints (gpt-oss `Reasoning: high` system line).

Completed 2026-09-08.

Check: table test

### P3.5. OpenAI adapter (`async-openai` 0.41.3): chat completions + streaming, ThinkConfig → `reasoning.effort`, Responses `reasoning` object, `input_image` / `input_audio`, `base_url` override, `reasoning`/`reasoning_content` split.

Completed 2026-09-08.

Check: `cargo test -p runa-cloud openai`; live smoke behind `RUNA_LIVE=1`

### P3.6. Anthropic adapter (`reqwest` + SSE): adaptive vs `enabled`+budget, `output_config.effort`, `display`, `thinking_delta`/`text_delta`, images/PDFs, `stop_reason: refusal`, 429/529 backoff, `cache_control` on system.

Completed 2026-09-08.

Check: `cargo test -p runa-cloud --lib` (fixture + wiremock); live smoke behind `RUNA_LIVE=1`

### P3.7. Backend routing: `runa run openai:<model>` / `anthropic:<model>`; `on_unfit=cloud:backend:model` fallback; cost line from `docs/prices.toml`.

Completed 2026-09-08.

Check: `crates/runa/tests/cloud.rs` (wiremock + prices table)

### P3.8. Secrets: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or OS keychain (`keyring`); inline keys in config files are rejected with a message.

Completed 2026-09-08.

Check: tests

### P3.9. `runa serve`: axum on `127.0.0.1` default; `/health`, `/v1/models`, `/v1/chat/completions` (stream + non-stream); `reasoning_content` in message/deltas; `reasoning_effort` / `reasoning_budget_tokens`.

Completed 2026-09-08.

Check: `cargo test -p runa --test e2e serve_`; `scripts/serve-openai-smoke.py` (CI installs `openai`)

## P4. Audio and video

### P4.1. Audio decode: `symphonia`/`hound`, `rubato` → f32 mono 16 kHz; `runa media probe` CLI; stable PCM SHA-256 on 10 fixture clips.

Completed 2026-09-08.

Check: `cargo test -p runa-media` (3 tests)

### P4.2. ASR via `whisper-rs` 0.16.0: auto-pull `base` / `large-v3-turbo` (+ Silero VAD ggml), energy VAD chunking, language auto-detect, `runa media transcribe`. Parakeet is `--features parakeet` (official sherpa-onnx; stub until models are present). WER on 5 fixture clips skipped (clips are sine tones, not speech); 1 min CPU timing skipped (no ggml in CI).

Completed 2026-09-08.

Check: `cargo test -p runa-media --lib` (15); `cargo test -p runa --test media media_transcribe_help`

### P4.3. Native audio via mtmd: `--audio` PCM chunk, `--mmproj` / sibling `*mmproj*.gguf`, fit reserves `mmproj_bytes`. Live Voxtral skipped (no 3B audio fixture; default build has no `mtmd`).

Completed 2026-09-08.

Check: `cargo test -p runa-engine --lib media::`; `run_help_lists_multi_gpu_flags` (`--audio`/`--mmproj`); `sibling_mmproj` on SmolVLM

### P4.4. `audio.route` / `--audio-route` auto|native|asr: local native iff mmproj, else ASR→text; OpenAI `input_audio` for audio-capable models, else transcript; Anthropic transcript. Native+unavailable errors (D12).

Completed 2026-09-08.

Check: `cargo test -p runa-media --lib route::` matrix; `cargo test -p runa --bin runa prepare_`; `--audio-route` on `run --help`

### P4.5. Video sampling: uniform + scene-change (histogram L1), cap 32, resize; ffmpeg on PATH or `ffmpeg-sidecar` auto-download; audio via ffmpeg→wav. Placeholder MP4s are not real media.

Completed 2026-09-08.

Check: `cargo test -p runa-media video::` (scene cut, cap 32); `runa media video`

### P4.6. Vision through mtmd: `--image` (repeatable) + `--video` sampled frames with `[t=12.0s]` markers; `VisionFrame` / `eval_vision_prompt`; sibling mmproj. 4B VL live describe skipped (default build has no `mtmd`; no 4B VL fixture).

Completed 2026-09-08.

Check: `cargo test -p runa-engine --lib vision::`; `run_help_lists_multi_gpu_flags` (`--image`/`--video`)

### P4.7. Cloud media: `runa-cloud` media prep — images to OpenAI `image_url` + Anthropic `image` blocks, PDFs as Anthropic `document`, video frames as images; count/edge/byte limits with auto-downscale.

Completed 2026-09-08.

Check: `cargo test -p runa-cloud --lib media::` (5)

### P4.8. Fit for media: `MediaFit` tokens (`frames × tokens_per_frame` + audio seconds); mmproj bytes reserved on GPU; encoder scratch `frames × n_embd × 4 × 8`. Overflow → `NO FIT` exit 2.

Completed 2026-09-08.

Check: `vl_32_frames_predicts_context_need`; `media_tokens_over_ctx_exit_2`

### P4.9. Media profile: criterion resize/normalize/histogram/resample; flame SVG; P5 ranking (histogram first, mel stays in whisper.cpp).

Completed 2026-09-08.

Check: `docs/profiles.md` P4.9 + `docs/profiles-p49.svg`; `cargo bench -p runa-media --bench media`

## P5. Kernels and speed

### P5.1. Profiling harness documented: `samply`, `cargo flamegraph`, `criterion` benches; top-10 ops table seeded from Metal `gen` spike; ggml callback hook noted as pending llama.cpp upgrade.

Completed 2026-09-08.

Check: `docs/profiles.md` P5.1 section

### P5.2. `runa-kernels` crate: `cc` build, runtime dispatch hooks, scalar C softmax + Rust reference, criterion bench, equivalence tests.

Completed 2026-09-08.

Check: `cargo test -p runa-kernels` passes; `docs/kernels.md`

### P5.3. Kernel 1 — sampling (top-k / top-p / min-p / softmax) in Zig (`@Vector`, C ABI); dispatch `SoftmaxImpl::ZigVector`; engine plug-in via `RUNA_KERNEL_SAMPLER=1`. Isolated softmax 0.91× vs Rust ref on aarch64 → **REJECT** default (keep ggml).

Completed 2026-09-12.

Check: `docs/kernels.md` P5.3; `cargo test -p runa-kernels`

### P5.4. Kernel 2 — image preprocessing (resize, normalize, patchify) NEON/AVX2 vs `fast_image_resize`. NEON normalize 1.52× (gate 2×) → REJECT, keep scalar; HQ FIR already slower (K3).

Completed 2026-09-08.

Check: `docs/kernels.md` P5.4; `cargo test -p runa-media --lib preprocess::`

### P5.5. Kernel 3 (research) SME2/AMX Q4_K/Q8_0 mat-vec: ggml b7709 already has AMX Q4_K+Q8_0 (`amx/mmq.cpp`) and KleidiAI SME2 Q8_0 (`kleidiai/kernels.cpp`); M3 Max has no SME2/AMX to test → **SKIP**.

Completed 2026-09-12.

Check: `docs/kernels.md` P5.5

### P5.6. Speculative decoding: n-gram (no draft model) as an option, draft model via `--draft`; fit includes draft memory. ≥1.3× tg on code skipped (no ngram bench in CI).

Completed 2026-09-08.

Check: greedy ngram text matches temp-0; `--ngram`/`--draft` in `--help`; draft bytes reserved in planner

### P5.7. Build flags: `native` feature (`-march=native`) for local builds; portable release builds rely on ggml's runtime dispatch; documented.

Completed 2026-09-08.

Check: release binary runs on a machine without AVX-512

### P5.8. Nightly `runa bench` (GitHub-hosted macos-14 / ubuntu-22.04 CPU until self-hosted exist); > 3 % pp/tg drop fails. First report: `docs/perf-nightly.md`.

Completed 2026-09-08.

Check: `.github/workflows/perf.yml`; `scripts/perf-regress.py`; `docs/perf-baseline.json`

## P6. Server, packaging, release

### P6.1. Server: `--models` lazy LRU pool, fit check before load, `--parallel` slots, `/v1/embeddings`, `/v1/audio/transcriptions` (ASR; 503 if whisper missing), image/audio `content` parts.

Completed 2026-09-12.

Check: `serve_embeddings_and_transcriptions_routes`; `serve_parallel_eight_chat`; `scripts/serve-oha.sh`

### P6.2. Anthropic `/v1/messages`: system, messages, stream + thinking blocks; SDK smoke.

Completed 2026-09-08.

Check: `cargo test -p runa --test e2e serve_`; `scripts/serve-anthropic-smoke.py`

### P6.3. Packaging: `cargo-dist` 0.28 → `.github/workflows/release.yml` (mac arm64 / Linux x86_64 / Windows x86_64 CPU); GPU variants workflow (Metal / Vulkan / CUDA); Homebrew formula on the Release (`listepo/homebrew-runa` tap when that repo exists); `runa doctor` lists compiled backends.

Completed 2026-09-08.

Check: `cargo dist generate --mode=ci --check`; `crates/runa/tests/doctor.rs`

### P6.4. Docs: `thinking.md`, `media.md`, `config.md` (every key + env), `fit.md`/`memory.md` already present; README index; man pages via `clap_mangen` (`docs/runa.1`, `docs/runa-run.1`).

Completed 2026-09-08.

Check: `config_keys_in_config_md`; `public_methods_in_memory_md`; `man_page_via_clap_mangen`

### P6.5. Security/privacy: no telemetry; `runa serve` binds `127.0.0.1` by default; API keys redacted (`runa-cloud` + cloud CLI); pull SHA-256 verification (P2.4); security smoke tests.

Completed 2026-09-08.

Check: `crates/runa/tests/security.rs`, `tests/secrets.rs`, `tests/pull.rs`

### P6.6. v1.0 checklist `docs/release-1.0.md`: M1–M13 recorded (pass / partial / open) against plan gates and `docs/baselines.md`. Live size/tg/ASR/8B timings still open before a 1.0 tag.

Completed 2026-09-08.

Check: file exists; every M-row has a status

## P7. Adaptive memory + agent task protocol

### P7.1. `runa-memory` core: `MemoryPolicy`, `Usage`/`LoadState`, `MemoryManager` (`current_usage`, `on_idle`, `on_heavy`, `shrink_to_floor`, `grow_for`) with unit tests on a fake backend.

Completed 2026-09-08.

Check: `cargo test -p runa-memory` green; public API matches `docs/memory.md` exactly

### P7.2. Idle shrink: `MemoryManager::touch`/`maybe_idle` wait `idle_timeout_s`; engine `LoadedModel::on_idle` unmaps prompt cache and keeps the model; RSS logged. Full soak vs live RSS skipped (fake backend ≤ floor + 10 %).

Completed 2026-09-08.

Check: `cargo test -p runa-memory maybe_idle`; generate test still generates after `on_idle`

### P7.3. Heavy grow: `grow_for` capped by fit ceiling and `max_growth_mib`; CLI preflight estimates KV+compute and errors with a suggestion before load.

Completed 2026-09-08.

Check: `cargo test -p runa-memory grow_for`; `run_over_ceiling_memory_suggests`

### P7.4. `TaskRegistry` (`list_free`, `status`, `claim`, `release`) over `docs/tasks.md`; `claim` fails with owner + `started_at` when `in progress`; `release` clears the row to `free`. Optional `runa tasks list\|claim\|release` CLI.

Completed 2026-09-08.

Check: M12: double-claim test fails cleanly; release test returns the row to `free`

### P7.5. Write `AGENTS.md` (claim-only-free, ask-before-steal, always release), `docs/memory.md` (all public methods), `readme.md` (overview + index).

Completed 2026-09-08.

Check: files exist; docs lint passes

### P7.6. CI: registry lint (every `in progress` row has agent + RFC 3339 `started_at`; no double-held task) + memory regression test.

Completed 2026-09-08.

Check: CI green; lint fails on a fixture with a nameless claim

## K. Monorepo, tooling & performance discipline

### K1. moon bootstrap: `.moon/workspace.yml` (8 crates + root), `.moon/toolchains.yml` (rust 1.98 mirror), `.moon/tasks/rust.yml` (build/test/clippy/fmt), root `moon.yml` (`lint-tasks`); moon pinned in `mise.toml`; registry lint covers K IDs; versions.md + readme updated.

Completed 2026-09-08.

Check: `mise install && moon projects` lists 9 projects; `moon run root:lint-tasks` executes; rust-toolchain.toml/Cargo.toml untouched by moon sync

### K2. mise owns all tools: `mise.toml` pins `rust 1.98`, `moon 2.5.4`, `ffmpeg 7.1.1`, `python 3.11.9`, `node 20.18.1`, `cargo:cargo-dist 0.28.0`, `cargo:cargo-cache 0.8.3`; CI bootstraps via `jdx/mise-action` + `mise install`; `docs/versions.md` updated.

Completed 2026-09-08.

Check: `mise ls` shows all present; `mise exec -- ffmpeg -version` etc. pass; no `dtolnay/rust-toolchain` outside mise in `.github/`

### K3. Polyglot escape-hatch: image `resize` Rust `fast_image_resize` vs C `stb_image` — C +1.7% (gate 5% / 2×) → REJECT, keep Rust. Record in `docs/profiles.md`.

Completed 2026-09-08.

Check: `docs/profiles.md` K3: Rust 1.82ms vs C 1.79ms (+1.7%), not adopted

### K4. Parallelism audit (D20): `moon run :test` 6.1s vs `cargo test --workspace` 12.3s (2.0×, 4 cores); `runa doctor --bench` 0.02s (rayon); `Fetcher` kept sync (no wall-time win for async).

Completed 2026-09-08.

Check: `docs/profiles.md` K4 audit table; no serial hot path without bench justification

### K5. Monorepo CI wiring: `.github/workflows/ci.yml` now has `moon` job (`moon projects` graph check, `moon ci --affected` parallel cached) alongside the direct-cargo matrix; `jdx/mise-action` bootstraps `rust 1.98` + `moon 2.5.4`; registry lint stays K-aware.

Completed 2026-09-08.

Check: `moon projects` lists 9 projects; `moon ci --affected` on docs-only runs (almost) nothing; CI green

### K6. Fix CI mise installs + revert-on-failure guard.

Completed 2026-09-14.

CI was all-red (mise install failed on every job with mise ≥ 2026.9.6). Fixed, layer by layer, each verified on branch CI before moving on:
- `mise.toml`: python 3.11.9 → 3.11.16 (no attestations on 3.11.9); `cargo:cargo-dist` → `aqua:` prebuilt 0.28.0 (ends concurrent `cargo install` rustup races); rust gains `components = "rustfmt,clippy"` (mise ignores rust-toolchain.toml under RUSTUP_TOOLCHAIN).
- `ci.yml`: `cargo dist` → `dist` (0.28.0 ships only the `dist` binary); qwen2 fixture download (git-ignored weights); revert job skips pushes touching `.github/` (no `workflows` permission); `shell: bash` on multi-line steps (Windows default is PowerShell).
- `dist-workspace.toml`: `pr-run-mode = "skip"` (tag-only releases); release.yml regenerated (trigger line only).
- `moon.yml`: dropped v1 `local: true` (rejected by moon 2.5.4 schema).
- Toolchain fallout in finished code: 16× `collapsible_if` let-chain collapses; `PriceTable::from_str` → `impl FromStr`; allows per repo precedent (`too_many_arguments`, `large_enum_variant` on the clap enum) + documented `needless_return` allows where clippy's suggestion breaks the build (E0308); `ParsedBody` alias; dead `MTMD_ENABLED` removed.
- Product bug unmasked on GPU-less Linux: `plan_placement` `fits` ignored RAM (serve refused CPU-only runs); CPU side must now fit RAM too (+2 unit tests). Anthropic smoke discovers the model id via /v1/models instead of hardcoded `"runa"`.
- Windows: explicit `-target x86_64-windows-msvc` for the Zig build (native detection emits MinGW `___chkstk_ms`, LNK2019); portable `cache_path` test (Windows separators).

Check: branch CI 34825280958 green on macos-14, ubuntu-22.04, windows-2022 + moon; `mise exec -- cargo clippy --workspace -- -D warnings` green; `cargo fmt --check` green

## P8. Features for 1.0

### P8.1. Structured output: JSON Schema and GBNF grammars

Completed 2026-09-14.

Constrain generation to a JSON Schema or a GBNF grammar.
- `runa-engine`: `GenerateRequest.json_schema` / `.grammar`; new `structured.rs` renders constrained requests through llama.cpp's Jinja handler (`apply_chat_template_oaicompat` → prompt, grammar, lazy triggers, extra stops) with a plain-prompt + eager-grammar fallback; `schema_to_grammar` wraps `json_schema_to_grammar`. Grammar goes first in the sampler chain; thinking, the Zig kernel sampler and n-gram speculation turn off under a grammar.
- Bug fixed on the way: every sampled token was accepted twice (`LlamaSampler::sample` already accepts), which advanced penalties twice and aborted llama.cpp grammars (`GGML_ASSERT(!stacks.empty())`).
- CLI: `runa run --json-schema <file|inline>` / `--grammar <file>` (conflicting); `Commands::Run` now wraps a `RunArgs` struct. OpenAI cloud gets `response_format: json_schema` (`strict: true`); `--grammar` and the Anthropic adapter refuse with a clear error.
- `runa serve`: `/v1/chat/completions` honours `response_format` (`text`, `json_object`, `json_schema`); bad schemas return 400.
- Docs: `docs/structured.md`, README, `docs/runa-run.1`, `run-help` snapshot. `RUNA_FAKE_RAM` test hook revives the `auto_unfit_*` e2e tests broken since K6's RAM-aware CPU fit.

Check: `cargo test -p runa-engine --test generate` (schema answer parses as JSON with typed keys; GBNF yes/no); `cargo test -p runa serve::tests::response_format_to_schema`; `cargo test -p runa-cloud`

### P8.2. Tool calling in `runa serve` (OpenAI and Anthropic APIs)

Completed 2026-09-14.

OpenAI and Anthropic tool calling on `runa serve`, driven by the model's own chat template.
- `runa-engine`: `GenerateRequest.tools` / `.tool_choice`; `ChatMessage` carries `tool_calls` / `tool_call_id`; new `GenEvent::ToolCalls(Vec<ToolCall>)`. Tool requests render through `apply_chat_template_oaicompat` (Hermes `<tool_call>` for Qwen3, generic JSON when the template has no tools); `auto` gets the lazy grammar, `required` / a named tool the eager one. Raw output is buffered and parsed at the end with `parse_response_oaicompat`; text keeps reasoning stripped, empty ids become `call_N`. Tools plus media is an error.
- `runa serve`: OpenAI `tools` / `tool_choice` (`auto`, `required`, `none`, named) → `message.tool_calls` + `finish_reason: tool_calls`, streamed as one `tool_calls` delta; `role: tool` messages round-trip. Anthropic `tools` / `tool_choice` (`auto`, `any`, `none`, `tool`) → `tool_use` blocks + `stop_reason: tool_use`; `tool_result` blocks become tool messages; the stream now wraps every block in `content_block_start`/`stop` and carries `usage` in `message_start` / `message_delta` (the SDK accumulator needs it).
- Bugs fixed on the way: `LoadedModel` held the `LlamaModel` inline while its transmuted `'static` context pointed at it, so moving a `LoadedModel` left the context dangling (segfault in `get_logits_ith` under a thinking budget). The model is now boxed and the context drops first. `runa_core::reason::emit_safe` sliced strings at non-char boundaries (panic on `°`).
- Docs: `docs/structured.md` tool-calling section, README.
- Not covered: Harmony (gpt-oss) tool format untested; templates with `thinking_forced_open` are not special-cased; Anthropic structured output still refused (P8.3 plans a forced tool).

Check: `cargo test -p runa-engine --test generate` (step 8: required tool call on qwen2); `cargo test -p runa serve::tests::tool_requests_map_to_engine`; `cargo test -p runa-core reason`; `RUNA_REQUIRE_OPENAI_SMOKE=1 cargo test -p runa --test e2e serve_` (OpenAI + Anthropic SDK tool round trips); live Qwen3-8B: auto call with thinking budget, tool-result answer, Anthropic `tool_use`.

### P8.5. LoRA adapters

Completed 2026-09-15.

`--lora <path>[:scale]` (repeatable) on `run` / `chat` / `serve` and `lora` in `[model]` / `[models.<alias>]` config.
- `runa-engine`: new `lora.rs` (`LoraSpec`, `parse_lora_spec` — last-colon split, non-numeric suffix stays in the path); `LoadConfig.loras`; `load()` pre-validates adapter files then `lora_adapter_init` + `lora_adapter_set` per spec; `reset_context()` re-applies the blend; verdict line lists adapters (`lora a.gguf:1,b.gguf:0.5`); adapter path+scale folded into the prompt-cache prefix key.
- `runa-fit`: `PlannerConfig.lora_bytes` reserved in the GPU budget and totals (`verdict.rs` prints the `lora:` line, snapshot updated).
- Binary: `resolve_loras(cli, model_ref)` (global + alias + flags, in that order); `auto_placement` and serve pool `fit_check` account adapter bytes; `Session.loras` survives `/model` + `/mode` reloads.
- Caveat: llama-cpp-2 0.1.133 `LlamaLoraAdapter` has no `Drop` — adapter memory lives until process exit (negligible for CLI; serve LRU eviction leaks adapter allocs until an upstream bump or a local `llama_adapter_lora_free` wrapper).
- Docs: `docs/config.md` (`[model]` + `lora` row), `docs/fit.md`.

Check: `cargo test -p runa-engine --lib lora` (7); `cargo test -p runa-fit` (59); `cargo test -p runa --bin runa -- lora config_keys` + `run_lora_missing_fails` e2e; load-when-present test skips (no `*lora*.gguf` fixture).

### P8.6. TUI chat (`runa chat --tui`)

Completed 2026-09-15.

`ratatui 0.30.2` + `crossterm 0.29` (+ `tui-textarea-2`, `unicode-width`, exact sibling specs, pre-approved in workspace `rust.md`) full-screen chat behind `runa chat --tui`.
- `crates/runa/src/tui.rs`: pure `ChatTui` state + `on_key` + `view`; word-wrapped transcript, `Ctrl+R` reasoning collapse (collapsed by default), multi-line input (`Enter` send / `Shift+Enter` newline / bracketed paste / history), 1-line status bar (`model · mode · tok/s · ctx N%`), `PageUp/Down` scroll, double-`Ctrl+C`/`Ctrl+D` quit; `parse_slash` mirrors the REPL (shared `SLASH_HELP`, also adopted by `/help`); token-by-token streaming redraw, tool-call notices, usage into session + status bar; `/mode` + `/model` go through `reload` so LoRAs survive.
- Docs: `toolchain.md` rows, `chat-help.toml` trycmd fixture.

Check: `cargo test -p runa --bin runa tui::` (6/6 incl. 2 `insta` snapshots on `TestBackend`); `cargo test -p runa --test trycmd`; pty-driven manual run on the qwen2 fixture (render, `/help`, `/usage`, streaming, `/quit` exit 0, terminal restored).

### P8.7. v1.0 gate measurements (M1–M11)

Completed 2026-09-15 (numbers recorded; starred rows need a quiet re-run before the 1.0 tag).

Measured on M3 Max 64 GB, macOS 26.6.2, debug CPU-only build (`backends: cpu`); llama-bench from the same b7709 sources. Machine was contended (sibling agents, load 127–224) — wall timings are contended, pass/fail stands.
- M1: fit path 148 ms (0.5B) / 147 ms (8B) ✓; weight +0.57 % vs engine mmap, KV exact, compute 3.7× conservative. No `runa fit` CLI (P8.4); exact-mode still a stub.
- M2: `hf:unsloth/Qwen3-8B-GGUF:Q4_K_M` 8 MiB range 0.57 s warm ✓, no download; cold 26 s ✗.
- M3: CPU 0.5B err 79–87 % pp / 172–265 % tg (FAIL ±30 %); `predicted_speeds` never reads `CalibrationDb` — needs a wiring task.
- M4: runa tg 23.1 vs bench 85.1 (27 %) — cause is threads=16 (P+E) vs bench auto 12; matched `-t16` runa wins 2.3×. Needs P-core default or `--threads`.
- M5: cold 8B CPU TTFT 9.02 s (gate is Metal; re-run pending).
- M6: 10/10 `--think-budget 64` seeds complete; reasoning counts not exposed via CLI; 100-run/GSM8K open.
- M7: BLOCKED — default `--lang auto` returns an empty transcript (bug); `--lang en` on 60 s real speech correct in 11.5 s*.
- M8: 30 s clip → 30 frames ✓ + audio ✓ in 2.6 s*; transcript ~6 s* vs <3 s gate at risk.
- M9: **pass** — both SDK smokes unmodified (streaming + tools).
- M10: debug 99.8 MiB, release unmeasured; no-telemetry grep clean.
- M11: serve 8B RSS 8364 MB vs 563 MB (14.9×); `Loaded::on_idle` unwired in serve; gate infeasible with a resident model.
- Follow-ups recorded in `ideas.md` (calibration wiring, `--threads`, `--lang auto` bug, reasoning counts in `--json`, M11 revisit, quiet re-runs + release/Metal builds).

Check: numbers in `docs/release-1.0.md` and `docs/baselines.md` §P8.7.

### P8.3. MCP client and tool loop for `run` / `chat`

Completed 2026-09-15.

`runa run|chat --mcp '<command args>'` (repeatable) and `[mcp.servers.<name>]` in config start stdio MCP servers and run the tool loop.
- `crates/runa/src/mcp.rs`: `McpHub` (rmcp 3.2 client over `TokioChildProcess`, its own one-worker runtime) starts every server with a 120 s timeout, lists tools (a name offered twice is an error), exposes them in OpenAI `tools` shape, and runs calls. Text content is the result; `isError` and transport failures come back as `error: …` text for the model. `tool_loop` drives any backend until an answer without calls or `--max-tool-rounds` (default 8, then an error); each call is logged to stderr as `[tool] name(args) -> N bytes`.
- Local (`run` and each `chat` turn): the assistant turn with its calls plus one `tool` message per result go back through the chat template (P8.2 path).
- Cloud: OpenAI `tools`, assistant `tool_calls`, `role: tool` messages (`CloudEvent::ToolCalls`). Anthropic `tools`, `tool_use` blocks sent back whole (thinking signatures included), `tool_result` blocks (`AnthropicEvent::ToolUse`). Tokens are summed across rounds. Anthropic `--json-schema` is now a forced `answer` tool (thinking off; not combinable with `--mcp`).
- `ToolCall` moved to `runa-core` so engine and cloud share it.
- Bug fixed on the way: a second generation on the same context without a prompt-cache hit prefilled from position 0 on top of the old cells ("inconsistent sequence positions" / `NTokensZero`). `start_generation` now clears the KV on a cache miss; this also broke chat's second turn.
- Docs: `docs/structured.md` MCP section, `docs/config.md` `[mcp.servers.<name>]`, README, help snapshot, `docs/runa-run.1`. `rmcp` recorded in `toolchain.md` and the workspace `rust.md`.
- Not covered: Anthropic SSE path has no tool support (the CLI uses the non-stream call); `--mcp` splits on whitespace (no shell quoting); chat keeps no history across turns (pre-existing), so tool rounds stay inside a turn.

Check: `cargo test -p runa mcp` and `config::tests::mcp_servers_toml`; `cargo test -p runa-cloud tool_`; `cargo test -p runa --test e2e run_mcp_ chat_second_turn` (stdio fixture `tests/fixtures/mcp-echo.py` on qwen2); live Qwen3-8B: think → `get_weather` call → MCP → answer.

### P8.4. `runa fit --recommend`

Completed 2026-09-15.

`runa fit` as a CLI command, plus `--recommend` over an embedded model catalog.
- `runa fit <model>` was never wired to the CLI (only the P1.11 library existed). Now it takes a local GGUF, alias, `hf:<repo>:<file-or-quant>` or URL (a pulled copy is read from disk; otherwise only the header is fetched), counts a local sibling `mmproj`, prints `format_report` and exits 0/1/2. `--ctx`, `--kv`, `--json`. `crates/runa/src/fit.rs`; `auto_placement` now shares its `machine_config`.
- `runa_fit::recommend` + `crates/runa-fit/src/catalog.toml` (19 single-file GGUFs checked against the Hub: exact filename, file size, MoE active bytes, mmproj size, uses, tier 1–4). `recommend()` probes in parallel (`std::thread::scope`), drops NO FIT and reports failed probes, ranks usable (≥ 5 tok/s) → tier → decode tok/s. `probe_remote` fits the header (hybrid decode via `estimate_speed_hybrid`); `probe_offline` fits catalog sizes against VRAM minus margin, else RAM.
- CLI: `runa fit --recommend [--use chat|code|vision|reasoning] [--top N] [--offline] [--json]`; a table with the `runa pull` ref; nothing fitting exits 2. `--use vision` counts the projector.
- Live (M3 Max, 48 GiB unified budget): 19 headers cold in 6.2 s; top pick gpt-oss 20B, then the Qwen3 30B A3B family.
- Docs: `docs/fit.md` CLI section, README, `docs/runa.1`. `toml` added to runa-fit (already in `toolchain.md`).
- Not covered: offline mode ignores KV and compute buffers; split GGUFs are not in the catalog; tiers are hand-curated.

Check: `cargo test -p runa-fit recommend` (ranking with a fake probe, catalog sanity, offline fit); `cargo test -p runa --test e2e fit_`; live `runa fit --recommend`.

### P8.8. `runa serve` warm-up: no dropped first request

Completed 2026-09-15.

The default model loads at startup, `/health` tells the truth, and a panic no longer drops the connection.
- `runa serve` binds, prints `listening on …`, then warms up the default model in the background: `serve: loading <id> N%` in 10% steps (llama.cpp's load-progress callback via `LoadConfig::progress`) and `serve: <id> ready in Xs`. Requests that arrive meanwhile wait on the pool lock instead of failing.
- `/health` answers 503 `{"status":"loading","model","progress"}` until the warm-up ends, 503 `{"status":"error",…}` if it failed, then 200 `ok`. `/v1/models` no longer takes the pool lock.
- Panic safety: an axum middleware turns a handler panic into 500 JSON; engine jobs and the warm-up run under `catch_unwind`; a poisoned pool lock is recovered; the engine thread gets an 8 MiB stack (what `runa run` has on the main thread).
- Found on the way: llama-cpp-2's `LlamaModelParams` is not `repr(C)` (the C struct sits at byte 48), so the old "first field" cast in `apply_tensor_split` wrote `--tensor-split` into the wrapper's Vecs. `raw_params` now finds the offset once from sentinel values (unit test), and `--tensor-split` errors instead of corrupting memory if a new layout hides it.
- The serve e2e tests echo the server's stderr, and the health test checks `loading`/`ok` around the `ready` line.
- Not covered: the Linux CI failure (curl 52 on the first request, run 34901506644) did not reproduce on the next run; if it returns, the echoed stderr shows why.

Check: `cargo test -p runa --bin runa serve::` (health body, jobs survive panics); `cargo test -p runa-engine --lib raw_params`; `cargo test -p runa --test e2e serve_`.

## P9. After-1.0 work (first slice)

### P9.1. `runa daemon` background service

Completed 2026-09-15.

Long-running service (launchd/systemd) that keeps models warm between CLI calls, owns the adaptive memory manager, and serves `run`/`chat` over a local Unix socket.
- `pool.rs` extraction: `ModelPool`, `EngineJob`, `spawn_engine`, `Warmup` moved out of `serve.rs` (no behavior change); the pool grew backend-aware in the P9.2 merge (`BackendKind` per model, `Auto` for the daemon).
- `daemon_proto.rs`: NDJSON request/event protocol over `tokio::net::UnixStream` at `~/.cache/runa/runa.sock` (`RUNA_DAEMON_SOCK` overrides); serde roundtrip tests.
- `runa daemon` subcommand owning pool + `MemoryManager` with P8.8-style warm-up and idle tick; launchd plist + systemd unit templates with `--install/--uninstall`.
- `run`/`chat` dial the daemon first and fall back to in-process load; `--no-daemon` for exact local control. Chat unifies both backends through `ChatEngine` (`Managed(LocalEngine)` for mistral, `Daemon(Option<LoadedModel>)` for gguf).
- Real `sysinfo`-backed `MemoryBackend` replacing `FakeBackend` at daemon/serve call sites.
- Windows: UDS transport is Unix-only — `runa daemon` errors explicitly, `run`/`chat` skip the daemon silently via the `request_sync` stub, serving-path items carry `cfg_attr(not(unix), allow(dead_code))`; live `daemon_serves_run_over_socket` e2e is `#[cfg(unix)]`.
- Docs: `docs/memory.md` P9.1 section, help/man snapshots.

Check: `cargo test -p runa --bin runa` (pool/daemon units, gate tests); `cargo test -p runa-memory` (incl. live-RSS backend); `cargo test -p runa --test e2e daemon_` (socket serve + no-daemon fallback on qwen2); branch + main CI green on all three OSes.

### P9.2. mistral.rs backend (`--features mistralrs`)

Completed 2026-09-15.

Optional second backend for safetensors-only or omni models ggml cannot run.
- Pinned `mistralrs =0.8.1` (MIT; 0.9.3 does not exist upstream — recorded in `docs/versions.md` and the workspace `rust.md`), `default-features=false`, optional; `mistralrs` feature passthrough in `runa-engine` → `runa`.
- `runa-core/src/backend.rs`: `BackendKind{Auto,Gguf,Mistral}` + model-ref detection (`.gguf` file vs dir with `config.json`).
- `runa-engine/src/mistral.rs` (`cfg(feature="mistralrs")`): load + generate mapped onto `GenerateRequest`/`GenEvent`; `--backend gguf|mistral|auto` dispatch in `run`/`chat`/`serve` (pool resolves per model; `/v1/embeddings` on mistral errors explicitly).
- `pull.rs` + `remote.rs`: multi-file safetensors snapshots; fit refuses non-GGUF with an explicit message, never silently.
- `doctor` reports `mistralrs`; default build stays GGUF-only (doctor test asserts it).
- Docs: `docs/versions.md` pin, crate `toolchain.md`, `docs/memory.md` P9.2 section.

Check: default `cargo clippy --workspace -- -D warnings` + `cargo test -p runa-core/-engine/-fit` green without the feature; `cargo check -p runa-engine/-p runa --features mistralrs` green; `cargo test -p runa --test pull/doctor` green; branch + main CI green.

### P9.4. NPU backends (Hexagon, OpenVINO)

Completed 2026-09-15 (Tier-3 scaffolding; runtime validation stays manual on-device).

Opt-in NPU support where ggml has it. Upstream reality: `llama-cpp-2` has no `hexagon`/`openvino` features (through 0.1.154), and no NPU CI runners exist — so this slice is survey + build-gating + probe + docs.
- Survey verdict WAIT recorded in `docs/versions.md` (`system-ggml` breaks pin discipline, `dynamic-backends` cannot conjure backends, D2 bindgen disproportionate). Empty stub features (forwarding to nonexistent dep-features breaks even default resolution — proven).
- `runa-fit/src/npu.rs`: `NpuKind`, `npu_present()` + `RUNA_FAKE_NPU` test hook, injectable probe markers; conservative `HwSpec::hexagon/openvino` (0.30 eff); opt-in `RUNA_NPU` verdict suffix (never default, D12); `doctor` stub strings; build.rs SDK-missing warnings.
- Tiers: NPU = Tier 3/manual (`d13-platform-tiers.md`, plan D13 row); `docs/fit.md` limits; CI stub-resolve step (check + doctor with features) green on all three OSes.

Check: `cargo test -p runa-fit` (69 incl. 8 NPU); `cargo test -p runa --bin runa` + doctor with/without stub features; `RUNA_NPU=hexagon RUNA_FAKE_NPU=1` verdict e2e; branch + main CI green.

### P9.3. Distributed inference via llama.cpp RPC

Completed 2026-09-15 (real backend, feature-gated; default build untouched).

Run layers on remote machines through llama.cpp `rpc-server` endpoints:
`runa run --rpc 192.168.1.5:50052 --device RPC0 model.gguf`.
- No fork, no pin bump: no published `llama-cpp-2` (surveyed 0.1.131–0.1.156)
  exposes an `rpc` feature, but the sys crate ships every header the backend
  needs — only the 78 KB `ggml-rpc/ggml-rpc.cpp` (b7709) is stripped, and
  `GGML_USE_RPC` in core only auto-registers an empty base reg. That one file
  is vendored verbatim + byte-identical headers under
  `crates/runa-engine/rpc/` (see `rpc/README.md`), compiled in `build.rs`
  behind a new `rpc` cargo feature (C++17, `ws2_32` on Windows); two public
  FFI fns (`ggml_backend_rpc_add_server`, `ggml_backend_register`) are called
  in `load()` before backend init, mirroring upstream `add_rpc_devices`.
- `--rpc` on `run`/`chat`/`serve` registers only; remote devices enumerate as
  `RPC0`, … and are selected explicitly with `--device` (never implicitly,
  D12); `--rpc` without `--device` errors before any network contact.
  `--main-gpu` added on `run`/`serve` (+ missing `--device/--tensor-split`
  on `serve` via `PlacementOverrides`, applied on fixed and `auto` paths).
  `chat` carries `--rpc` in `Session` through all (re)loads; mistral and
  daemon paths reject explicitly. `fit --rpc` estimates locally with an
  explicit note/warning; `doctor` reports `rpc: true/false` (text + JSON).
- Safety: `build.rs` pin-guard fails the `rpc` build if `llama-cpp-sys-2`
  drifts from `=0.1.133` (refresh rule in `rpc/README.md`); default binary
  carries zero `ggml_backend_rpc` symbols (verified with `nm`).
- Docs: `docs/versions.md` RPC section, `docs/fit.md`, `cc` in `toolchain.md`,
  trycmd snapshots + `runa-run.1` regenerated, CI `RPC backend feature` step
  (all three OSes, no fixture weights needed).

Check: `cargo test -p runa-engine --features rpc --lib --test rpc --test load`
(fake-TCP-server loopback: HELLO 3.6.0 + DEVICE_COUNT → register → enumerate
`RPC0`; unreachable endpoint → `RpcUnreachable`); `cargo test -p runa
--features rpc --test e2e -- rpc` (+ default-feature `Unsupported` path);
`cargo test -p runa --test doctor --test trycmd`; `cargo clippy --workspace
-- -D warnings` green.

## P10. Follow-ups batch (ideas.md + uncovered items, 2026-09-15/16)

### P10.1. Calibration-aware speed predictions (M3 follow-up)

Completed 2026-09-16.

`predicted_speeds` never read `CalibrationDb`. New `runa-fit`
`apply_efficiency(pp, tg, eff)` (`speed.rs`; per-axis guard against
bad factors); `bench::predicted_speeds` loads the DB
(`RUNA_CALIBRATION`-aware) keyed by `(device, backend, quant)`;
`runa fit` scales report speeds (`calibrate_report`) and the
`--recommend` ranking (`calibrate_pick`, hybrid uses the GPU-side
factor). Empty DB = unchanged numbers.

Check: `cargo test -p runa-fit apply_efficiency` (3) +
`cargo test -p runa --bin runa fit::` (3); e2e
`fit_calibration_db_scales_predictions` (2.0× sample doubles
`runa fit --json` decode on qwen2).

### P10.2. `--threads` knob + P-core-aware default (M4 follow-up)

Completed 2026-09-16.

runa pinned all logical CPUs incl. E-cores and lost to llama-bench
auto. `--threads N` on `run` / `chat` / `bench` / `serve` plus
`RUNA_THREADS` and `[defaults] threads`
(`config::resolve_threads`: CLI > env > files > engine default);
`default_threads()` reads `hw.perflevel0.logicalcpu` on macOS via a
macOS-only `libc` dep (=0.2.189, locked; `toolchain.md` + workspace
`rust.md` updated), `available_parallelism` elsewhere, floor 1.
GGUF-only: mistral rejects explicitly, `--threads` opts `run` out of
the daemon. trycmd help fixtures + `docs/runa-run.1` regenerated,
`docs/config.md` `[defaults]` section.

Check: engine `threads_tests`, `resolve_threads_precedence`,
daemon-gate test, e2e `threads_zero_fails_with_usage_error`.

### P10.3. Fix `--lang auto` ASR returning an empty transcript (M7 blocker)

Completed 2026-09-16.

Root cause in vendored whisper.cpp `whisper_full`: the
`detect_language` flag means detect-ONLY (`return 0` without
decoding). Old code set flag + `language="auto"` (first fix attempt
set the flag alone — still empty). Correct call: `language="auto"`
with the flag unset: detect AND decode. Verified live
(ggml-base): synthesized EN speech `--lang auto` == `--lang en`
byte-for-byte. M7 row in `docs/release-1.0.md` updated (timing
re-run still pending).

Check: `asr::tests::auto_detect_decodes_like_explicit`
(live-gated on the cached model; proven FAIL pre-fix, pass post-fix).

### P10.4. Reasoning token counts in `--json` (M6 follow-up)

Completed 2026-09-16.

`Usage.reasoning_tokens`: unconditional max-budget `BudgetClock`
fed in `observe_budget`, deltas accumulated across re-armed blocks;
surfaced in `run --json` (`run_json`), serve OpenAI
`completion_tokens_details`, daemon wire protocol (`#[serde(default)]`
for old daemons), `/usage` (REPL + TUI, only when > 0). Mistral/cloud
stay 0 (no split). M6 row: counts now externally checkable.

Check: qwen2 zero-reasoning assert; live Qwen3-8B
`think_budget_reports_reasoning_tokens` (reasoning > 0, ≤ generated,
≤ budget+grace) incl. a required tool call under budget (P10.11);
serve + daemon-proto unit tests.

### P10.5. Serve idle tick calls `on_idle` (M11 follow-up)

Completed 2026-09-16.

`EngineJob::Idle` → `LocalEngine::on_idle` (prompt cache released,
model kept; mistral no-op); `ModelPool` tracks `last_used` +
`idle_timeout` (`due_for_idle` pure, `idle_sweep` re-stamps, drops
dead engines); shared `pool::idle_tick` (manager decision + sweep
off-thread) wired into serve (new `MemoryManager`, generation
endpoints touch it, health/models polls do not) and the daemon tick.
M11 verdict measured: RSS 772.9 MiB before and after on qwen2-0.5B —
release real but negligible next to the resident model; gate stays
infeasible without model unload (D17 forbids it).

Check: pool `idle_*` unit tests (3); e2e
`serve_idle_tick_releases_prompt_cache`
(`RUNA_MEMORY_IDLE_TIMEOUT_S=1`).

### P10.6. Anthropic SSE streaming tool support

Completed 2026-09-16.

`parse_sse` dropped `content_block_start` / `input_json_delta`, so
streamed tool calls vanished. New `SseTools` accumulator rebuilds
blocks (tool id/name/args, thinking text+signature, text) in index
order; `ToolUse` emits at `message_delta[stop_reason]` ahead of
`Done` (same position as `parse_message`, so `drain_anthropic` is
stream-agnostic); fragments never leak into text/reasoning. CLI keeps
`stream: false` on purpose (only whole-message replies preserve
thinking signatures — commented at the request site).

Check: `sse_tool_use_reassembled_across_fragments` (2 calls,
fragmented args, order, ToolUse-before-Done, no leak) +
`sse_without_tools_emits_no_tool_use`; all 12 anthropic tests green.

### P10.7. Offline `--recommend` counts KV + compute

Completed 2026-09-16.

`catalog.toml` gains measured `kv_mib_per_1k` (f16, linear in ctx;
q8_0 halves, q4_0 quarters) + `compute_mib` (ubatch-512 reference,
linear in `n_ubatch`) for all 19 entries (collected once via
`runa fit --json --ctx 8192` headers). `probe_offline` need =
weights + projector + KV + compute; speed uses active + KV/2.
`CatalogEntry` drops `Eq` (f64 fields).

Check: `offline_counts_kv_and_compute` (Gpu→Cpu→NoFit tip,
q8_0 recovery, scaling helpers), `catalog_is_sane` (both numbers >
0); e2e `fit_recommend_offline_ranks_the_catalog` green.

### P10.8. Shell quoting for `--mcp '<command args>'`

Completed 2026-09-16.

`split_shell_words` (no new dep): single/double quotes, backslash
escapes, unterminated-quote and dangling-backslash errors. Help text
+ `docs/structured.md` MCP section updated (replaces the
whitespace-split note).

Check: `mcp::tests::flag_quoting_groups_spaces` +
`flag_unterminated_quote_is_an_error`; trycmd run/chat fixtures.

### P10.9. Chat keeps history across turns

Completed 2026-09-16.

`Session.history` rides every REPL/TUI turn
(`turn_messages`: push + trim to ~75% ctx; `commit_history`:
sent transcript + final answer; failed turns record nothing).
`trim_history` cuts whole turns (never orphans `tool` messages,
never drops the newest user turn) on a chars/4 estimate guard rail;
trim notices go to stdout (REPL) or a TUI notice (never `println!`
into the alternate screen). `/reset` + `/model` clear (`/mode`
keeps); daemon protocol already carries full messages, no wire
change. `docs/structured.md` history line updated.

Check: `history_tests` (4); e2e
`chat_second_turn_remembers_history` (qwen2 recalls "Ada", 3/3
locally) + existing context-reuse test.

### P10.10. Split-GGUF models in the catalog

Completed 2026-09-16.

`CatalogEntry.parts` (shard refs 2..N, `ref` stays part 1);
`all_refs()` + `combine_shard_weights()` (arch from part 1, four
weight groups summed); `probe_remote` fetches every part header.
`catalog_is_sane` validates all refs; no split entry ships yet (all
current models fit in one file). Catalog header documents the
schema.

Check: `split_tests` (ref order, weight sums on real qwen2 header
metadata — no third synthetic-GGUF builder copy).

### P10.11. Harmony tool-format + `thinking_forced_open` coverage

Completed 2026-09-16.

Closed by verification: pinned llama.cpp ships
`common_chat_parse_gpt_oss` and owns `thinking_forced_open`
internally from our `enable_thinking` (nothing extra needed
client-side). `harmony_tool_reply_parses` renders a required tool
call through the REAL gpt-oss-20B template and parses canned
replies in both header orders. Fixture repair on the way:
`gpt-oss-20b-MXFP4.gguf` was truncated (11.4/12.1 GiB); resumed
from the Hub to the exact catalog byte size (git-ignored weights,
no tracked files touched). `docs/structured.md` format row.

Check: `--lib harmony` + `--test generate think_budget` (now incl.
a required Qwen3-8B tool call under a 64+8 budget, reported
reasoning inside the cap).

### P10.12. Full workspace pass: tests, clippy, fmt, docs lint

Completed 2026-09-16.

`cargo fmt --check` clean; `cargo clippy --workspace -- -D warnings`
(CI mode) green — incl. fixing own `collapsible_match`
(anthropic) and `manual_range_contains` (load.rs) hits;
`--all-targets` shows only the pre-existing criterion-0.8
`black_box` deprecations in benches (untouched, CI doesn't cover
them). `cargo test --workspace` green in one clean run (all
targets, 0 failed).
- Real find on the way: `serve_health_models_and_chat` failed 3× in
  full-suite runs while green standalone (3 s). Diagnosed, not
  dismissed: curl POSTs starved 120 s with 0 bytes while a parallel
  daemon turn took 232 s for 6 events on a 64 GB / 16-CPU box with
  swap disabled — libtest's default 16 threads × P-core ggml fan-out
  per model load. Proven by `RUST_TEST_THREADS=4` (37/37 e2e green),
  then pinned repo-wide in `.cargo/config.toml` `[env]`
  (`force = false`, so an explicit env still wins).
  `docs/release-1.0.md` M3/M4/M7/M11 rows updated with P10 evidence.

### P10.13. `--max-load-percent` system load cap (foreign WIP, completed)

Completed 2026-09-16.

Found unclaimed and uncompiling in the tree
(`--max-load-percent` / `[system]` / `RUNA_MAX_LOAD_PERCENT` with
config resolvers, warnings, tests, partial CLI plumbing); finished:
`cmd_chat` / `cmd_bench` / `ServeOpts` / `Daemon` signatures,
daemon-gate opt-out (+ test), mistral-backend rejects (run + chat),
`docs/memory.md` API section. Warning-only cap (default 80%),
never fatal: model RAM demand, ambient RAM pressure, `--threads`
CPU share each print an actionable `warning:` naming the value to
set. Precedence CLI > env > files.

Check: `cargo test -p runa --bin runa` (99 green incl. foreign
`max_load_*` + gate test); trycmd run/chat/serve/daemon fixtures;
clippy CI-mode green; CLI error-path smoke
(`--max-load-percent 0`, `RUNA_MAX_LOAD_PERCENT=lots`).


### P11.4. Tokenizer-exact chat history trimming

Completed 2026-09-16 (worker-4).

Local gguf paths count real tokens through the loaded tokenizer:
new `LoadedModel::count_history_tokens` (`runa-engine/src/generate.rs`,
`model().str_to_token(.., AddBos::Never)` + 4 per message for template
markers; per-message tokenize failure falls back to chars/4, trim always
terminates). `LocalEngine::count_history_tokens` is `pub(crate)`,
`None` for mistral. New generic `trim_history_with(history, budget,
count)`; `trim_history` stays a thin heuristic wrapper (old tests
untouched). `turn_messages` takes `&ChatEngine` and picks the counter
via `history_tokens()`; daemon-socket path keeps the estimate
explicitly (no tokenizer there). Budget still 75% ctx; drop order,
notice, and limits unchanged. Tool calls not counted (as before);
counts cover message content (+4), not the fully rendered prompt —
deliberately conservative.

Check: `history_tests` 7/7 (4 old + 3 new: exact counts on a known
string via stub counter, whole-turn/newest-survives contract on exact
counts, estimate value locked); e2e recall unaffected (small
histories never trim).

### P11.6. Linux CI curl-52 watch

Completed 2026-09-16 (worker-6). Verdict: no recurrence — closed by
observation. All post-P8.8 runs checked: `34976968709`,
`34980160972` green (ubuntu serve e2e + SDK smoke passed, zero
curl-52 lines); `35055665463`, `35012959563`, `34993041805` die
earlier on `cargo test --workspace` (no curl-52 signal possible);
`34972604943`, `34970282240` die on clippy. Dated note appended to
`docs/perf-nightly.md`. Next step if it returns: take the fresh
failed `ci` run, read the P8.8 echoed server stderr at
`.github/workflows/ci.yml:95`, then fix.


### P11.3. Harden the claim protocol after the unclaimed-edit incident

Completed 2026-09-16 (worker-3).

One-line rule in `AGENTS.md` §2 (via the `AGENTS.md → CLAUDE.md`
symlink, left intact): no tree edits without a claim row in
`docs/tasks.md`. The lint can't see diffs/authorship from a registry
snapshot, so `scripts/lint-tasks.py` instead fail-closes (exit 1)
when `AGENTS.md` lacks the claim-first marker (`--agents-path`,
default `scripts/../AGENTS.md`); the limitation is documented in the
script docstring. New `--self-test` (7 cases) + fixture
`tests/fixtures/agents-without-claim-first.md`.

Check: `lint-tasks.py docs/tasks.md` 0 errors; `moon run
root:lint-tasks` green; `tasks-bad-claim.md` still fails (P7.6 CI
step intact); `--self-test` 7/7.


### P11.5. Revisit remaining `ponytail` notes

Completed 2026-09-16 (worker-5).

- `recommend.rs` hybrid share converted: new private
  `hybrid_gpu_fraction` sums active bytes over `plan.tensors`
  (dense/embed fully, experts weighted `n_expert_used/n_expert`);
  dense models and tensor-less plans fall back to the old formula
  (identical there). Synthetic measure: share 0.27 → 0.60,
  `decode_for` 90.8 → 157.0 tok/s (x1.73). Note deleted.
- `serve.rs` + `daemon.rs` progress slot: kept single by design —
  only the startup warm-up reports it (`Warmup::done`,
  `report_progress`, `health_body`); post-`done` LRU overwrites go
  unread, so per-load plumbing (a `pool.rs` change) buys nothing
  observable. Notes replaced with one justification line each.

Check: `runa-fit recommend` 9 passed (incl. new
`hybrid_fraction_counts_active_bytes`,
`hybrid_decode_uses_active_share`); bin `serve::` 9, `daemon::`
4, `pool::` 7; `cargo fmt --check` clean.


### P11.2. Quiet M-measurements + release/Metal builds

Completed 2026-09-16 (worker-2, all quiet at 1-min load 3.7–9.2).

- M10 pass: `cargo build --offline --release -p runa` (3m03s) →
  24,662,064 B (23.5 MiB) ≤ 40 MB.
- M7 pass: 57.8 s synthesized speech → 1.02 s wall, exact
  transcript; `--lang auto` md5-identical to `--lang en`.
- M8 pass: synthetic 30 s clip → 30 frames + audio in 1.01 s,
  audio extract + transcribe 0.59 s (≈1.6 s combined).
- M5 still open: release CPU warm-mmap spawn→first token
  ≈1.2–1.3 s; true-cold + Metal-release run pending. (Metal debug
  8B: 8 tok in 1.79 s; 0.5B Metal: pp 1035 / tg 235.)
- M4 still partial: runa-side `bench pp512/tg128` 8B release =
  pp 144.1 / tg 16.4; llama-bench comparison blocked (no binary,
  vendored sources lack it, offline).
- Observation (no code change): default think mode spends the
  whole budget in reasoning on Qwen3-8B (8 tok → 7 reasoning +
  empty text), even with `--think off` phrasing — recorded in
  `docs/baselines.md`.
- `docs/baselines.md` gained the P11.2 section with reproduce
  commands; `release-1.0.md` M4/M5/M7/M8/M10 updated (M6 row
  refreshed by coordinator: counts exposed since P10.4).


### P11.7. Fix CI red: graceful fixture skip in `moe_desc` (worker-A)

Completed 2026-09-16. `moe_desc()` returned `Option`, both hybrid
tests skip with `eprintln` without the git-ignored fixture —
mirroring `desc_with_weights`. Commit `570027b` (one file).
Proven locally both ways (fixture present + hidden) + fmt + clippy
clean; CI run 35062488231: `--lib` step green on all 3 OSes
(Windows passed via the new skip path).


### P11.9. Moon + perf CI under 5 minutes (worker-C)

Completed 2026-09-16. Moon job already held 32 s–2 m 04 s across
recent runs — no optimization needed, budget codified as
`timeout-minutes: 5` on the job. Perf 5 m 31 s → **3 m 00 s** green
(run 35065429646; mac 2 m 52 s, ubuntu 2 m 32 s): build with
`--profile dist` (thin LTO; hot path is vendored C++, measurement
neutral across 3 green runs), fixture download overlapped with the
build, `v2-rust-dist` cache key (dist-only entries ~560 MB vs
~905 MB; avoids perpetual ~8 min cold builds on exact cache hits).
Deliberately not done: touching `[profile.release]`, sccpdache
(new dep, useless — head crate rebuilds every push), actions/cache
for the fixture (slower than the ~7–11 s download). Decision
recorded in `docs/perf-nightly.md`. Commits `84763a4`, `57d1058`.


### P11.8. Matrix CI under 5 minutes for tests (worker-B)

Completed 2026-09-16. Matrix green, every OS holds ≤ 5 min warm
(proof run 35067852546: ubuntu 2 m 13 s, mac 2 m 47 s / 3 m 16 s
repeat, windows 4 m 35 s; was 4 m 47 s / 5 m 44 s / 28 m+ red).
Commit `4da612d`, `ci`-job only: job-level
`CARGO_PROFILE_DEV_DEBUG=0` + `CARGO_PROFILE_TEST_DEBUG=0`
(cache restore 92 s→21 s ubuntu etc.; only backtrace line numbers
lost), one unified RPC cargo invocation instead of two (cold-graph
rebuild 140–170 s → 8–30 s, same coverage), dropped `--verbose`
build spam. Largest fixed cost left: Windows `dist generate
--check` 57–74 s. Cold cache fits no OS (~8.5–12 min) — physics
without sccache; stable warm instead (repeat runs 2 m 11 s–3 m 16 s).
Proposals parked for coordinator approval: scope dist check to
ubuntu, combine the two NPU checks, sccache with measurements.


### P11.1. Push commits + watch CI green (all OSes)

Completed 2026-09-16 (coordinator). Pushed `251ece9`, watched CI
fail on the new `moe_desc` fixture `expect` (worker-A fixed in
`570027b`) and on the pre-existing Windows `runa[EXE]` trycmd
mismatch (coordinator fixed with the `[EXE]` placeholder in 4
fixtures). Strategy pivoted per creator order to branch `p11-ci-green`
+ PR (#2, merged as `a3be4c5`) after the auto-revert `1131136`.
Branch CI fully green (ci + perf), then main CI green on the merge
(ci 35068365571, perf 35068365582 — incl. Windows RPC trycmd and
moon/perf within budget). Registry cleared; branch deleted after
merge per repo rule.


### P12.1. CI scope trims: ubuntu-only dist check + combined NPU check

Completed 2026-09-16. `dist generate --check` runs once on ubuntu
(output is platform-independent; trade-off recorded in the step);
one combined NPU `cargo check` proves both stub names resolve
(isolation proof dropped deliberately — stubs forward to nothing).
Branch `p12-ci-trims`, CI run 35097311148 fully green: windows
3 m 39 s total (was 4 m 35 s tests alone), ubuntu 2 m 21 s, mac
3 m 02 s, moon 33 s. Merged via PR.


### P12.2. sccache trial with measurements

Completed 2026-09-16. Verdict: ADOPTED on all 3 OSes.
- Measurement 1 (cold Swatinem via one-off key + empty sccache,
  ubuntu, run 35098291326): build 1 m 53 s, Rust hit rate 0% —
  pure fill (360 C/C++ hits from repeated identical C files across
  clippy/build/test invocations).
- Measurement 2 (cold Swatinem + warm sccache, run 35099674405):
  clippy 1 m 34 s + build 30 s + test 24 s at 88.5% hit rate
  (850 Rust hits) vs ~8.5+ min true-cold baseline — ~3-4x faster
  cold builds. Trial key (841 MB) deleted from the quota after.
- Adopted (`p12-sccache-trial` 7b89bca): mozilla sccache-action
  v0.0.11 + RUSTC_WRAPPER/CMAKE launchers on every OS, stats step
  always-on for forensics. Swatinem stays the warm fast path;
  sccache (small content-keyed GHA blobs) insures cache busts.
  Warm-run overhead ~1 s. D21 note: CI-only tooling via pinned
  action, same pattern as the existing Swatinem step (not mise).
- Adoption run 35100747512 fully green (all 3 OSes + moon).


### P11.1. Push commits + watch CI green (all OSes)

Completed 2026-09-16 (coordinator). Pushed `251ece9`, watched CI
fail on the new `moe_desc` fixture `expect` (fixed in P11.7) and on
the pre-existing Windows `runa[EXE]` trycmd mismatch (fixed with the
`[EXE]` placeholder). Strategy pivoted per creator order to branch
`p11-ci-green` + PR (#2, merged as `a3be4c5`) after the auto-revert
`1131136`. Follow-ups rode the same flow: P12.1 trims (PR #3, merged
`ab87c28`), P12.2 sccache adoption (PR #4, merged `b46eda3`). Final
main CI green on the merge (ci + perf success) — registry cleared.
## P13. Documentation

### P13.1. Docs pass: refresh `README.md`, add `docs/README.md`, extend `AGENTS.md`

Completed 2026-09-18 (Cline). Creator request; the v1.0 tree had no
documentation index and the two entry points did not describe the CLI,
the crate layout, the doc set, or the contribution checks.

- `README.md`: added a CLI table for all `runa` commands with their real
  flags, replaced the partial Docs table with every file in `docs/` plus
  the ADR set, and added **Repository layout** (eight crates, fixtures,
  scripts) and **Develop** (mise/moon commands, CI steps, fixture budget).
- `docs/README.md` (new): documentation index — audience routing, one-line
  purpose per page, the `docs/adr/` range (D01–D18, D23 with D19–D22 noted
  as `plan.md`-only), and the rules for adding docs.
- `AGENTS.md` (= `CLAUDE.md`, the symlink target): file map and symlink
  note, §1 reading order extended, plus §8 workspace layout, §9 local
  checks with CI parity, §10 documentation rules, §11 commit/PR
  conventions, §12 definition of done. §1–§7 kept numbered as before
  (the registry lint matches the §2 claim-first marker, `moon.yml` cites §6).

Check: `python3 scripts/lint-tasks.py docs/tasks.md` → `0 error(s)`;
a link check over `README.md`, `docs/README.md`, `CLAUDE.md`,
`CONTRIBUTING.md`, `plan.md`, `docs/tasks.md` → 0 missing relative links;
every command and flag documented was read from `crates/runa/src/main.rs`
and `fit.rs` (no invented surfaces).
### P13.2. `ketch.toml` — runa as a ketch package

Completed 2026-09-18 (Cline). Creator request: a ketch manifest in the
project root, following the sibling convention (`apps/rtok/ketch.toml`,
`apps/ketch/ketch.toml`).

- New root `ketch.toml`: `name` / `source = "github:pyrlyn/runa"` /
  `description` / `homepage`, `bin = [{ name = "runa" }]`, and an
  `[asset]` block — `include = ["*.tar.xz", "*windows-msvc.zip"]` plus
  `exclude` for the GPU variants (`-metal` / `-vulkan` / `-cuda`), the
  shell / powershell / homebrew installers, `*.json`, `*.sha256` and the
  source tarball. No `trust` block: `release.yml` publishes no signature
  sidecars, so a trust policy would fail every install.
- `docs/versions.md`: new *ketch package (`ketch.toml`)* subsection under
  *Native vs portable* — what the manifest is for, the field table, why the
  GPU archives need the `exclude` list (they carry the same target triples),
  and the validation command. Also corrected the stale `cargo-dist 0.28`
  there to 0.33, matching `dist-workspace.toml` and the `mise.toml` pin.
- Index rows in `README.md` and `docs/README.md` mention the manifest.

Check (run on the committed file):

```sh
TMP=$(mktemp -d); mkdir -p "$TMP/runa"; cp ketch.toml "$TMP/runa/"
ketch registry validate "$TMP"     # validated 1 package  (exit 0)
ketch registry validate "$TMP" --json
# {"errors": [], "packages": 1, "status": "ok"}
```

Asset patterns checked against the real release asset names with ketch's
`model::glob_match` semantics (full match, `*`/`?`, case-insensitive): the
three portable CPU archives are selectable, and the three GPU variants plus
`runa-installer.sh`, `runa-installer.ps1`, `runa.rb`, `dist-manifest.json`,
`*.sha256` and `source.tar.gz` are dropped — 0 failures over 12 names.

Limit: asset *scoring* cannot be exercised end to end yet — `gh release view`
answers `release not found`, so no release exists to install from. The
registry copy landed after this entry was written: `pyrlyn/ketch-registry`
PR #7 `add runa` (merged 2026-09-19 as `cb7b0f3`) carries `runa/ketch.toml`
(the CLI `runa*` bin glob, portable CPU archives, GPU variants excluded); the
CI assertion step was dropped from that PR before merge because the registry
itself removed its only workflow in #8 — follow-up drift watch: its README
now asks for the single `<name>/ketch.toml` file only.

### P13.3. Local release flow like rtok, no new GitHub workflows

Completed 2026-09-19 (Cline). Creator order 2026-09-18, then narrowed: runa
runs no workflow except dist's tag-triggered `release.yml`, so the rtok
workflow chain (`bump.yml`, `release-plz.yml`, `verify.yml`,
`dispatch-releases`) does not land — ports only the local half.

- `scripts/release.sh` (new, executable): the one place a version is chosen
  (`[workspace.package]`, raised only when that version is already tagged);
  default path pushes the commit **and the `v<version>` tag**, which is what
  the tag trigger turns into a release. `--dry-run` prints the version and
  changes nothing; `--local` commits without pushing or tagging.
- `cliff.toml` (new) + `git-cliff 2.13.1` pinned in `mise.toml` (D16) +
  generated `CHANGELOG.md` (never edited by hand).
- `docs/release.md` (new): script usage, the tag trigger, the target × archive
  table, the gate as a human step, the not-wired list (macos-sign,
  install-updater, Homebrew publish job), local dry-run commands.
- `docs/versions.md`: git-cliff pin row, Release flow pointer, and the stale
  `cargo-dist 0.28` → `0.33` fix (matches `dist-workspace.toml` and the
  `mise.toml` pin).
- Index rows in `README.md`, `docs/README.md`, `CLAUDE.md` (`scripts/` layout
  table); the README Status points at `docs/release.md`.

Check (evidence, all on the merged tree): `bash scripts/release.sh patch
--dry-run` → `current 0.1.0 -> release v0.1.0`, nothing written;
`patch --local` → untagged `0.1.0`, nothing to bump, nothing pushed or tagged;
the bump path exercised in a throwaway workspace (tag `v0.1.0`, two commits,
`patch --local` → `0.1.1`, lockfile + changelog section correct);
`bash scripts/cargo-dist.sh generate --mode=ci --check` green;
`python3 scripts/lint-tasks.py docs/tasks.md` → `0 error(s)`; relative links
0 missing; `git diff --check` clean. No workflow files added or changed, so
nothing new can run in CI. Merged as runa PR #9
(`5b039d6`, branch `ci/release-like-rtok`), after the remote-only workflow
commits were reverted out of the PR in `b743b38` without force-push.

### P14.1. CI triggers on push/PR to main (no drafts), /review command workflow, manual release workflow

Completed 2026-09-19 (Cline / Muse Spark). User request: CI+tests on push/merge to main and on PRs to main except drafts, review by `/review` command in PR, separate manual release workflow with patch/minor/major choice.

- `.github/workflows/ci.yml`: `on: push branches [main]`, `pull_request branches [main]` (types opened/synchronize/reopened/ready_for_review), `workflow_dispatch`; both jobs skip while `pull_request.draft == true`; concurrency cancel-in-progress.
- `.github/workflows/review.yml` (new): `issue_comment created` gated on `/review` prefix, open non-draft PR targeting main (re-checked via `gh pr view`); ubuntu-only fmt/clippy/build/lib-tests/lint/runa-memory + result comment back to PR.
- `.github/workflows/release-manual.yml` (new): `workflow_dispatch` with `level` choice (patch/minor/major) + `mode` (''/--dry-run/--local); runs `scripts/release.sh`, which pushes commit+tag so dist `release.yml` publishes.
- `docs/release.md`: trigger table + draft/review notes, gate text covers both entry points.

Check (evidence): `cargo fmt --check` OK; `python3 scripts/lint-tasks.py docs/tasks.md` -> 0 error(s); `cargo test -p runa-memory` -> 15 passed; `${{ }}` balance equal in all three workflows; registry table left empty.

### P14.3. Clean up target dirs with dunnage after tests

`scripts/test-with-fixture-cleanup.sh` (`moon run root:test-with-cleanup`) now runs `dunnage run target` after a green pass, before the fixture cleanup: lossless compression and dedupe of `./target` that never deletes and keeps mtimes, so nothing rebuilds. Exit code 2 (a build held the lock) counts as success; without dunnage the step prints an install hint, without `target/` it is skipped, and a failed test run still exits before it. dunnage is installed with `ketch install dunnage`; `toolchain.md` lists ketch and dunnage and gains a `ketch` package table; `AGENTS.md` and `README.md` describe the task.

Check (evidence): `bash -n scripts/test-with-fixture-cleanup.sh` OK; the dunnage step run on its own exits 0 without `target/`, and `dunnage run --dry-run target` plans work on a real target.

### P15.1. Fix the 2026-10-01 QA audit findings

The find-only audit (PR 26) lists 20 confirmed bugs. Fix all of them
in one change: clamp prefill batches to `n_ctx` and reject oversized
prompts, send `HF_TOKEN` only to Hugging Face (or the configured hub
host), lock the task registry across processes, validate agent names,
surface registry I/O errors, reject local `image_url` paths before any
stat, keep request payloads in 0600 temp files that are deleted on drop,
map context overflow to HTTP 400, report `max_tokens` truncation, refuse
a second daemon on a live socket, cap `fit` range bodies and check
`Content-Range`, let later config files override `[memory]`/`[audio]`
and reject bad `RUNA_MEMORY_*` values, stop the MCP tool loop at
`max_rounds`, store models as `owner--name` with a legacy lookup, use a
strict base64 decoder, honor `general.alignment`, count embedding
tokens, reject unknown roles and `max_tokens: 0`, verify pull size
before the sidecar, and drop the pool lock while a model loads.

Check (evidence): `cargo test -p runa-memory` 18 passed; `cargo test -p runa-fit --lib` 82 passed and `cargo test -p runa-fit --test gguf` 14 passed; `cargo test -p runa --bin runa` 108 passed; `cargo test -p runa-engine --lib batch_is_clamped` passed; `cargo clippy -p runa -p runa-memory -p runa-fit -p runa-engine --all-targets -- -D warnings` clean; `cargo fmt --all -- --check` clean; `python3 scripts/lint-tasks.py docs/tasks.md` 0 error(s).

## P16. Requests from aulo

### P16.1. Stream tokens as they are generated

Completed 2026-10-07.

`runa serve` builds the whole reply before replaying it as SSE
(`serve.rs` `generate_events`), so time to first token equals generation
time. Stream each GenEvent to the client as the engine produces it.
Requested by aulo (its task T1.19): a voice agent needs the first
sentence before the reply is finished.

Machine check: a test with a timer shows the first token reaches the
client before generation ends.

Plan: replace the collect-then-replay path in `serve.rs` with a channel from
the generation task to the SSE body (bounded, so a slow client slows
generation instead of buffering it), keep the non-streaming JSON path and
the event order unchanged, and add a timed test with a slow fake engine.

Check (evidence): `cargo test -p runa --bin runa` 113 passed (includes `first_token_reaches_the_client_before_generation_ends` and `anthropic_message_start_carries_the_prompt_length`); `cargo test -p runa-engine --lib` 52 passed; `cargo test -p runa --test e2e serve_ -- --test-threads=1` 5 passed; `cargo clippy --workspace --all-targets -- -D warnings` and `cargo fmt --all -- --check` clean. The Anthropic `message_start` reports 0 input tokens on the mistral backend, which only counts them at the end.

### P16.2. Library target for in-process local inference

Completed 2026-10-07.

runa is bin-only (`serve`, `pool` and `mcp` are `pub(crate)`). Expose a
library crate (for example `runa-serve`) so another program, aulo's
daemon `aulod` (its task T1.20), can embed local inference without a
second process.

Plan: the bin's module tree is also the `cargo fuzz` root (`fuzz/Cargo.toml`
compiles `main.rs` directly), so a lib target inside `crates/runa` would
need `main.rs` moved. Instead extract `engine.rs` and `pool.rs` into a new
crate `crates/runa-pool` (the code moves, it is not copied) and make the
bin depend on it. The pool reached `main.rs` only through placement
(`auto_placement`, `preflight_grow`, fit check), so placement becomes an
injected `Placer` (`Arc<dyn Fn(&Path, &LoadConfig) -> Result<Placement>>`):
the bin keeps today's policy in `crates/runa/src/placer.rs`, embedders pass
`fixed_placer(..)` or their own. Add `ModelPool::attach_engine` (custom
backend or test double) and `generate_stream`. No CLI behaviour change.
Verify with an integration test in `crates/runa-pool/tests/` that builds
the pool through the public API and drains a generation from a fake engine
thread, then fmt, clippy `--all-targets`, `cargo test -p runa --bin runa`
and the registry lint.

Machine check: a test outside the runa binary calls the pool through the
library.

Check (evidence): new crate `crates/runa-pool` (`ModelPool`, `Placer`, `fixed_placer`, `generate_on`, `generate_stream_on`); `cargo test -p runa-pool` 6 unit + 3 integration (`tests/embed.rs`) passed; `cargo test -p runa --bin runa` 108 passed; `cargo test -p runa --test e2e -- --test-threads=1 serve_ daemon` 7 passed; `cargo clippy --workspace --all-targets -- -D warnings` and `cargo fmt --all -- --check` clean; `cargo check -p runa --features mistralrs` passed. The nightly fuzz crate was not built.

## P17. Cloud review findings

### P17.8. Refuse a non-loopback `runa serve` bind without `--api-key`

Completed 2026-10-09 (Cursor Grok 4.7).

`runa serve` binds `127.0.0.1` by default and had no auth on `/v1`, so
`--host 0.0.0.0` (or any other non-loopback address) exposed completions,
embeddings and transcriptions. The command now refuses that bind unless
`--api-key` is set. With a key, every `/v1` route requires
`Authorization: Bearer <key>` and answers HTTP 401 `invalid api key`
otherwise. `/health` stays open. Loopback (`127.0.0.0/8`, `::1`, and
IPv4-mapped loopback) still starts with no key. There is no `runa.toml`
key for this token.

Check (evidence): `cargo test -p runa --bin runa -- require_bind_auth` 1 passed; `cargo test -p runa --bin runa -- api_key_gates` 1 passed; `cargo test -p runa --test security --test trycmd` 5 passed; `cargo clippy --workspace -- -D warnings` and `cargo clippy -p runa --all-targets -- -D warnings` clean; `cargo fmt --all -- --check` clean; `python3 scripts/lint-tasks.py docs/tasks.md` 0 error(s) while the claim was held. `cargo test -p runa --bin runa` 109 passed. `second_daemon_does_not_unlink_a_live_socket` failed on this overlayfs checkout (the same socket inode is reused) and does not touch serve auth.

### P17.1. Licensing metadata: Cargo SPDX field vs the three-licence README

Completed 2026-10-09 (Cursor Grok 4.7).

The workspace `license` is `GPL-3.0-or-later` while the README offers GPL, a
royalty-free licence, and a commercial licence. crates.io accepts only SPDX
License List identifiers in `license` and rejects `LicenseRef-*`, so the other
two terms cannot be named there. `license` and `license-file` cannot be set
together; replacing the SPDX identifier would hide the GPL from license
scanners. The field stays the open-source choice. A comment in `Cargo.toml`
says so, the README license section (outside the `license-sync` markers)
explains it, and `docs/getting-started.md` plus the Russian and Ukrainian
translations say the same. Those two translations had said `MIT OR Apache-2.0`;
they now match the English note. `crates/runa-core` test
`cargo_license_is_spdx_gpl_and_readme_explains_the_other_two` locks the field
and the explanation.

Check (evidence): `cargo test -p runa-core --lib cargo_license_is_spdx_gpl` 1 passed; `cargo test --workspace --lib` passed (runa-core 36, runa-cloud 37, runa-engine 51, runa-fit 82, runa-kernels 9, runa-media 25, runa-memory 18, runa-pool 6); `cargo clippy --workspace -- -D warnings` clean; `cargo fmt --all -- --check` clean; `python3 scripts/lint-tasks.py docs/tasks.md` 0 error(s) while the claim was held.
