# runa — research: how to write a program for running AI models

Date: 8 September 2026. Plan (`plan.md`), code and commands are in English.

Method: six parallel Haiku agents collected facts from primary sources (GitHub, crates.io, Hugging Face, official documentation), three adversarial Haiku agents re-checked 130+ claims, synthesis and decisions — Fable. Everything unconfirmed is marked in §13.

---

## 1. Summary

**Verdict.** The program should be built as a thin but smart Rust layer over ggml/llama.cpp (C), not as a new engine. All twenty-odd analogues that people actually use (Ollama, LM Studio, Jan, koboldcpp, LocalAI, llamafile, Lemonade) are wrappers around llama.cpp. The heaviest parts (matrix kernels for NEON/SVE/SME, AVX2/AVX-512/AMX, CUDA/Metal/Vulkan) are already written there in C/asm per platform and updated ~50 builds per week. Custom C/asm kernels only make sense where the profiler shows ggml is weak, and only if they give ≥ 5% on the same hardware.

**Where the niche is.** No analogue closes the loop "forecast → launch → measure → correct". A "will the model fit" estimate exists (llama.cpp `--fit`, LM Studio `lms load --estimate-only`, gguf-parser-go, llmfit), but it either lives separately from launch, or does not predict speed, or does not learn from real measurements. Ollama even silently falls back to CPU (issue #14258 open). Nobody has unified "deep thinking" control for local models and OpenAI/Anthropic clouds either. Audio + video + thinking + cloud in a single Rust binary — nobody has that.

**What runa does.**
1. `runa fit <model>` — before downloading, reads the GGUF header via an HTTP range request, probes the hardware and prints a verdict: whether it fits (GPU / hybrid / CPU / no), how much memory on each device, speed forecast with a confidence interval, what to change if it does not fit. After each run the forecast is calibrated against measurements.
2. Three modes `cpu | gpu | hybrid` + `auto` — a single tensor placement planner (layers on GPU, MoE experts on CPU, KV cache and its quantization).
3. `ThinkConfig` — off / on / token budget / effort level — identical for local models (forced closing of the reasoning block in our sampling loop) and for OpenAI (`reasoning.effort`) and Anthropic (`thinking: adaptive` + `output_config.effort`).
4. Audio and video — decoding in Rust, encoders in C (mtmd in llama.cpp, whisper.cpp), three routes: native audio/visual model, ASR → text, cloud (accounting for OpenAI not accepting video, and Anthropic accepting neither audio nor video).
5. CLI + TOML configs + profiles + OpenAI-compatible server; explicit `on_unfit = error | cpu | cloud:<model>` policy instead of silent fallback.

Step-by-step plan (P0–P7, ~70 tasks with machine-checkable verification for each) — in `plan.md`.

---

## 2. Requirements and how they are covered

| # | Requirement | Solution | Plan ref |
|---|-----------|---------|-------------|
| R1 | Deep thinking | `ThinkConfig`, reasoning-block parser per model family, forced budget, mapping to clouds | D7, P3 |
| R2 | Audio and video | `runa-media` (symphonia/rubato/ffmpeg-sidecar) + mtmd/whisper.cpp + cloud adapters | D8, P4 |
| R3 | Rust + C/asm in hot spots | ggml (C) as the core; `runa-kernels` (C/`.S`) behind a ≥ 5% benchmark gate | D1, D14, P5 |
| R4 | Ready-made libraries | llama-cpp-2, whisper-rs, sherpa-onnx, async-openai, hf-hub, sysinfo, nvml-wrapper, objc2-metal, axum, clap, figment | D2, D9, D10 |
| R5 | Three modes CPU / GPU / CPU+GPU | `Placement` planner, `--mode` | D4, P1.8, P2 |
| R6 | Command line + configs | clap, figment (defaults < user < project < env < flags), profiles | D10 |
| R7 | OpenAI / Anthropic API | `Backend` trait; async-openai (Responses API); own thin Anthropic client (reqwest + SSE) | D9, P3.5–P3.7 |
| R8 | Check whether it will run, warn and predict how it will perform | `runa-fit`: GGUF header (local/remote), hardware probe, analytical estimate + exact mode, speed model + calibration, verdict with exit codes | D5, D6, D12, P1 |

---

## 3. Analogues

### 3.1. Who is who (September 2026)

| Project | Language / engine | Version | Modes | Fit check | Speed forecast | Thinking | Audio input | Video input | OpenAI/Anthropic as backend | License |
|--------|---------------|--------|--------|----------------------|------------------|----------|-----------|-----------|------------------------------|----------|
| **llama.cpp** | C/C++, ggml | v0.4.0 / b10853 (Sep 2026) | cpu / gpu / hybrid (`-ngl`, `-ot`, `--n-cpu-moe`, `--device`) | auto-fit at load: `--fit on` (default), `--fit-margin`, `--fit-target`, `llama_params_fit`, `llama-fit-params` utility | no | `--reasoning-budget`, `--reasoning-budget-{message,soft-ratio,grace-tokens}`, `--reasoning-format`, per-request `reasoning_budget_tokens` (PR #25961, July 2026) | yes (mtmd: Ultravox, Voxtral, Qwen2-Audio, Qwen3-ASR) | yes (PR #24269, June 2026, via ffmpeg) | no | MIT |
| **Ollama** | Go + ggml (MLX backend in preview since 0.19) | 0.33.2 (27 Aug 2026) | auto layer offload | internal estimate (`llm/memory.go`), but **silent fallback to CPU** (#14258 open) | no | `think: true/false/low/medium/high/max` | no | no | own cloud models, not OpenAI/Anthropic | MIT |
| **LM Studio** | TS/Electron + llama.cpp, MLX | 0.4.x | GPU-offload slider | `lms load --estimate-only`, GUI indicators | no | on/off, block parsing | no | no | no | proprietary (free) |
| **mistral.rs** | Rust (candle) | 0.9.3 (7 Sep 2026) | cpu / cuda / metal, auto device-map | auto device-map | no | no budget | claimed in README for some models, not confirmed in 0.9.3 releases | Qwen-VL | server serves OpenAI **and** Anthropic-compatible APIs (does not consume them) | MIT |
| **Jan** | TS + llama.cpp | 0.8.x | GPU-offload | memory warnings in GUI | no | on/off | no | no | yes (OpenAI, Anthropic and other providers) | open |
| **koboldcpp** | C++/Python + llama.cpp | — | cpu / gpu / hybrid | no | no | partial | no (TTS/whisper separate) | no | no | AGPL-3.0 |
| **LocalAI** | Go + llama.cpp and others | 4.x | cpu / gpu | no | no | partial | transcription (whisper) | no | no | MIT |
| **llamafile** | C (cosmopolitan) + llama.cpp | 0.10.x | cpu / gpu | no | no | no | no | no | no | Apache-2.0 |
| **vLLM** | Python/CUDA | 0.21.x | gpu (CPU backend limited) | "does not fit" = error at startup | no | `reasoning_effort`, thinking budget (PR #37112) | yes (audio models) | yes (VLM) | no | Apache-2.0 |
| **SGLang** | Python/CUDA | — | gpu | error at startup | no | `--reasoning-parser`, `separate_reasoning` | yes | yes | no | Apache-2.0 |
| **MLX-LM** | Python (Apple) | 0.31.x | Apple GPU | no | no | partial | no | no | no | MIT |
| **AMD Lemonade** | Python + llama.cpp / ONNX / NPU | 10.8.0 (Jun 2026) | cpu / gpu / npu (Ryzen AI) | no | no | partial | no | no | no | Apache-2.0 |
| **Nexa SDK** | C++/Rust, own runtime | — | cpu / gpu / npu (Qualcomm, Apple) | no | no | partial | yes (Qwen3-Omni) | yes | no | unconfirmed |
| **gguf-parser-go** (GPUStack) | Go, utility | — | — | RAM/VRAM per device **without downloading** | yes, "MAX TPS" via `--device-metric` | — | — | — | — | MIT |
| **llmfit** | Rust, CLI + TUI | — | — | pre-load estimate from model table and memory (Ollama, llama.cpp, MLX, LM Studio) | yes, bandwidth model | — | — | — | — | open |
| **Kalosm / Crane** | Rust (candle) | — | cpu / gpu | no | no | no | Kalosm: whisper | no | no | MIT/Apache |

The rest of the reviewed projects (exo, GPT4All, text-generation-webui, RamaLama, llama-swap, Docker Model Runner, Foundry Local) add nothing new against our eight requirements: they are either wrappers around llama.cpp/vLLM or model-switching orchestrators.

### 3.2. Requirements matrix

| Requirement | llama.cpp | Ollama | LM Studio | mistral.rs | Jan | vLLM | llmfit | **runa (goal)** |
|-----------|-----------|--------|-----------|------------|-----|------|--------|-----------------|
| Thinking: on/off | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | — | ✓ |
| Thinking: token budget | ✓ | ✗ (levels) | ✗ | ✗ | ✗ | ✓ | — | ✓ |
| Thinking: same for local and cloud | ✗ | partial | ✗ | ✗ | ✗ | ✗ | — | ✓ |
| Audio input | ✓ | ✗ | ✗ | ? | ✗ | ✓ | — | ✓ (native / ASR / cloud) |
| Video input | ✓ (Jun 2026) | ✗ | ✗ | partial | ✗ | ✓ | — | ✓ (frames + audio track) |
| Rust | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✓ | ✓ |
| C/asm kernels per platform | ✓ (ggml) | via ggml | via ggml | candle (Rust/CUDA) | via ggml | CUDA | — | ggml + own behind gate |
| cpu / gpu / hybrid | ✓ | ✓ | ✓ | ✓ | ✓ | gpu | — | ✓ + `auto` |
| CLI + configs | ✓ | Modelfile | GUI + `lms` | ✓ | GUI | ✓ | ✓ | ✓ TOML + profiles |
| OpenAI / Anthropic as backend | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | — | ✓ + `on_unfit=cloud:` |
| Fit check before download | at load | internal, silent | ✓ | ✗ | GUI | ✗ | ✓ | ✓ before **download** |
| Speed forecast | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ with interval |
| Forecast calibration from measurements | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| Accounting for mmproj / frames / thinking budget in forecast | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| Explicit policy on "does not fit" | error/fit | silent CPU | error | error | error | error | — | `error \| cpu \| cloud` |

### 3.3. Confirmed user pains of analogues

- **Ollama, silent fallback to CPU.** The model "started" but runs 10x slower; there is no warning (issue #14258, open; proposal — raise the log from debug to warn). This is the main reason runa always prints a one-line verdict before the first token.
- **llama.cpp, context-checkpoint memory.** `--ctx-checkpoints` (default 32) accumulated RAM; fixed in 2026 (issue #24055, PR #22929). Implication for the fit checker: count not only weights and KV but everything the engine allocates, and calibrate against logs of the pinned version.
- **llama.cpp, `--fit` does not count mmproj.** The `fit-params` documentation does not confirm accounting for the multimodal projector; when loading a VLM/audio model a "fitted" configuration may not fit. runa counts mmproj and encoder buffers separately (P4.8).
- **Ollama MLX.** The MLX backend since 0.19 (March 2026) is preview, only ≥ 32 GB unified memory and a single model; betting on ggml remains right for Apple.

### 3.4. What nobody does (runa's niche)

1. **Closed fit loop.** Pre-download forecast → launch with the same planner → `runa bench` measurement → efficiency-coefficient correction for that (device, quant) pair. gguf-parser-go and llmfit compute but do not launch and do not learn; llama.cpp fits at load time but does not predict speed and does not work before download.
2. **One thinking knob for everything.** Budget/effort applies identically to Qwen3, gpt-oss, DeepSeek locally and to OpenAI/Anthropic in the cloud; the response-time forecast accounts for the reasoning budget ("2048 thinking tokens at 50 tok/s ≈ 40 s").
3. **Audio + video + cloud with an honest capability table.** If a backend cannot do audio (Anthropic) — the ASR → text route; if it cannot do video (both clouds) — frames + transcript; `runa fit` prints which route the media will take.
4. **No silent decisions.** `on_unfit` is set explicitly; the verdict is always printed; exit codes 0/1/2.

---

## 4. Engine and languages

### 4.1. Why ggml/llama.cpp, not a custom engine and not candle/burn

| Criterion | ggml / llama.cpp | mistral.rs (candle) | candle / burn directly |
|----------|------------------|---------------------|------------------------|
| Backends | CUDA, Metal, Vulkan, ROCm, SYCL, OpenCL/Adreno, Hexagon NPU, CANN, MUSA, OpenVINO, WebGPU, RPC | CPU (MKL/Accelerate), CUDA, Metal | candle: CPU/CUDA/Metal; burn: + Vulkan/ROCm/WebGPU via CubeCL |
| Quants | all GGUF formats (Q2_K…Q8_0, IQ*, MXFP4), quantized KV | ISQ, GGUF, GPTQ, AWQ, FP8, MXFP4 | GGUF loader exists, fewer kernels |
| CPU kernels | NEON/i8mm/SVE, AVX2/AVX-512-VNNI/AMX, KleidiAI, ZenDNN — in C/asm | Rust + `gemm` | Rust |
| Multimodality | mtmd: images, audio, video (Jun 2026) | vision (Qwen-VL), audio claimed | limited |
| Memory fitting | `--fit`, `llama_params_fit` | auto device-map | manual |
| Speculative decoding | draft, EAGLE-3, n-gram without draft model | partial | no |
| Thinking | `--reasoning-budget*`, `--reasoning-format` | no | no |
| Rust binding | `llama-cpp-2` 0.1.133 (Aug 2026): features `cuda`, `metal`, `vulkan`, `openmp`, `native`, `dynamic-link`, `mtmd`; `rig-llama-cpp` on top | native | native |
| License | MIT | MIT | Apache-2.0 |

Decision (D2): ggml via `llama-cpp-2` is the primary backend; mistral.rs is optional (feature) for models missing from ggml (safetensors, omni models without mtmd). The `llama-cpp-2` risk (lags upstream, unstable API) is covered by a version pin and our own `bindgen` over `llama.h`/`mtmd.h` as a fallback path.

P9.4 note (2026-09-15): the "Hexagon NPU / OpenVINO" cells above describe ggml/upstream scope, not what our binding exposes — `llama-cpp-2` has **no** `hexagon`/`openvino` cargo features through 0.1.154 (survey in `docs/versions.md`), so runa ships them as probe-only Tier-3 stubs (`hexagon`/`openvino` cargo features, `runa doctor` `*-stub` strings, `RUNA_FAKE_NPU` probe, conservative `HwSpec` defaults) with placement pinned to CPU. Real offload waits on upstream features + vendor SDKs + manual on-device validation.

### 4.2. Rust and "the heaviest parts in C/asm"

Stable Rust 1.98.1 state (20 Aug 2026):

| Capability | Status | Implication |
|-------------|--------|-----------|
| `asm!`, `global_asm!` | stable since 1.59 | assembly kernels can be embedded directly from Rust |
| `#[unsafe(naked)]` | stable since 1.88 | bare functions for manual prologue |
| `std::simd` (portable SIMD) | nightly only | portable SIMD in Rust is unavailable on stable |
| SVE/SVE2 intrinsics | nightly (PR to stdarch Apr 2026) | ARM servers (Graviton, Ampere) — C/asm only |
| SME/SME2 intrinsics (Apple M4+) | design stage (2026 target) | matrix kernels for Apple M4/M5/M6 — C/asm only |
| AMX intrinsics (x86) | nightly, incomplete (#126622) | Sapphire Rapids+ — C/asm only |
| `cc` crate | stable | compiling `.c`/`.S` with per-file flags |

Hence rule D1: ggml already contains these paths (including SME via KleidiAI and AMX), and "rewrite the hot parts in C/asm" means not rewriting ggml but (a) building it with the right per-platform flags and (b) adding our own kernels only where profiling shows a gap: vocabulary sampling over a 150k-token vocabulary, image preprocessing (resize/normalize/patchify), audio frontend (resampling, mel spectrogram), and — as a research task — quantized mat-vec on SME2/AMX where ggml has no path for a specific platform yet. Each kernel: scalar reference implementation, equivalence fuzz test, `criterion` bench, runtime dispatcher (`is_x86_feature_detected!`, `is_aarch64_feature_detected!`, `sysctl hw.optional.arm.FEAT_SME2`). Merge gate — ≥ 5% on end-to-end tok/s or ≥ 2x on an isolated operation.

---

## 5. Three modes: CPU, GPU, CPU + GPU

In llama.cpp these are not three modes but one set of tensor-placement knobs:

| Knob | What it does | Use in runa |
|-------|-----------|----------------------|
| `-ngl N` / `--gpu-layers auto` | how many layers on GPU | `gpu` = all, `cpu` = 0, `hybrid` = N from planner |
| `-ot <regex>=CPU` / `--n-cpu-moe N` | MoE expert tensors on CPU | `hybrid` for MoE: attention and shared layers on GPU, experts on CPU |
| `--device`, `--tensor-split` | device list, share per device | multi-GPU (P2.9) |
| `--cache-type-k/v q8_0` | quantized KV cache (requires flash attention) | `--kv q8_0` in fit and at launch |
| `--fit`, `--fit-margin` | auto-fit at load | fit checker's "exact mode" for a local file |
| `ggml_backend_sched` | graph distribution across backends | inside the engine |

Why hybrid matters specifically for MoE: in Qwen3-30B-A3B or Qwen3.6-35B-A3B only ~3B parameters per token are active, but expert weights occupy 90% of the file. If experts sit in RAM while attention, normalizations and the router are on GPU, the model runs on 8 GB VRAM at acceptable speed, because only the selected experts are read per token. The runa planner (P1.8) puts experts on CPU first and only then reduces the number of layers on GPU.

Hybrid speed is computed harmonically: `t_token = bytes_gpu / (eff_gpu × BW_gpu) + bytes_cpu / (eff_cpu × BW_cpu)`; the bottleneck is almost always the CPU part.

---

## 6. Fit checker: "will it run and how fast"

### 6.1. How analogues do it

| Tool | Method | Before download | Speed | Learns |
|-----------|-------|---------------|----------|-----------|
| llama.cpp `--fit` / `llama_params_fit` | trial graph build with the allocator, fitting `-ngl`, `-c`, `-ot` to free memory minus `--fit-margin` (default 1024 MiB) | no | no | no |
| Ollama `llm/memory.go` | per-layer estimate: weights + KV + graph buffers by architecture family | no | no | no |
| LM Studio `lms load --estimate-only` | memory estimate before loading | no (file is local) | no | no |
| gguf-parser-go | parses the GGUF header by URL, computes RAM/VRAM per device, `--device-metric` → MAX TPS | **yes** | yes | no |
| llmfit | model table + machine memory + bandwidth model | yes (from table) | yes | no |
| **runa fit** | GGUF header (local or HTTP range), hardware probe, analytical estimate with interval, exact mode via the engine allocator, calibration | yes | yes, with interval | yes |

### 6.2. Memory

`total = weights + kv + compute + mmproj + margin`

**Weights** — exact sum of tensor sizes from the GGUF header: `Σ n_elements(tensor) × bytes_per_block(type) / elements_per_block(type)`. Bits per weight per `ggml-common.h`:

| Type | bytes / block | elements / block | bits per weight |
|-----|-------------|------------------|-----------|
| Q4_0 | 18 | 32 | 4.50 |
| Q4_1 | 20 | 32 | 5.00 |
| Q5_0 | 22 | 32 | 5.50 |
| Q8_0 | 34 | 32 | 8.50 |
| Q2_K | 84 | 256 | 2.625 |
| Q3_K | 110 | 256 | 3.4375 |
| Q4_K | 144 | 256 | 4.50 |
| Q5_K | 176 | 256 | 5.50 |
| Q6_K | 210 | 256 | 6.5625 |
| IQ4_XS | 136 | 256 | 4.25 |
| IQ4_NL | 18 | 32 | 4.50 |
| MXFP4 | 17 | 32 | 4.25 |
| F16 / BF16 | 2 | 1 | 16 |

"Q4_K_M" is not a type but a mixture of types across tensors (some in Q6_K), so effectively ≈ 4.8 bits/weight; that is why one must sum over tensors rather than multiply parameters by an "average bpw" (in one of the agents' reports bpw was confused with file gigabytes — see §13).

**KV cache** — `2 × n_layer × n_ctx × n_head_kv × head_dim × bytes(type)` (f16 = 2, q8_0 = 1.0625, q4_0 = 0.5625). Exceptions that break the formula and must be read from metadata: sliding-window layers (Gemma 3/4, gpt-oss: `min(n_ctx, window)`), MLA (DeepSeek V3/V4: `kv_lora_rank + rope_dim` per token), recurrent/Mamba layers (Granite 4.x, Nemotron 3: constant size).

**Compute buffers** — depend on `n_ubatch`, `n_embd`, `n_vocab`, `n_head`, backend; in practice from hundreds of MiB to 1–2 GiB for large vocabularies and batches. The "20–50 MB" estimate from one of the reports is wrong; runa calibrates the formula against ≥ 20 llama.cpp logs on the pinned version and adds 15% (P1.5), and for a local file takes the exact number from the allocator.

**mmproj** — a separate encoder file (vision/audio), plus the encoder buffer; not accounted for in llama.cpp `--fit` (not confirmed by documentation), runa accounts for it.

**Available memory** — not "total" but: on NVIDIA `nvmlDeviceGetMemoryInfo().free`; on Apple `recommendedMaxWorkingSetSize` (≈ 75% of unified memory by default, raised via `sysctl iogpu.wired_limit_mb`); on CPU — `available` from `sysinfo` minus a system reserve.

### 6.3. Speed

Single-token decoding is memory-bandwidth bound (Kapoulkine, "LLM inference speed of light", 2024): each token reads all active weights and the KV cache.

```
bytes_per_token = active_weights_bytes + kv_bytes_at(ctx/2)
decode_tok_s    = eff(device, quant) × BW_bytes_s / bytes_per_token
prefill_tok_s   = min(eff_c × FLOPS / (2 × active_params), BW-bound)
ttft_s          = prompt_tokens / prefill_tok_s
hybrid          = 1 / (bytes_gpu / (eff_gpu × BW_gpu) + bytes_cpu / (eff_cpu × BW_cpu))
```

For MoE `active_weights = shared + n_expert_used / n_expert × expert_bytes`. Real efficiency `eff`: per Kapoulkine's measurements llama.cpp reaches 58–82% of the theoretical RTX 4090 bandwidth depending on weight format, a specialized engine — ~90%. runa starts at 0.60 (CUDA, Metal), 0.50 (Vulkan, CPU) and after each run updates the median `measured / predicted` for the (device, quant) pair — this is how the forecast converges to ±15% after three runs (M3 in the plan).

### 6.4. Hardware: memory bandwidth (verified against specifications)

| Device | Memory | Bandwidth, GB/s | Forecast tok/s for 8B Q4_K_M (≈ 4.9 GiB), eff 0.6 |
|-----------|--------|--------------|-----------------------------------------------|
| Apple M4 | up to 32 GB | 120 | ~14 |
| Apple M4 Pro | up to 64 GB | 273 | ~31 |
| Apple M4 Max | up to 128 GB | 546 | ~62 |
| Apple M5 | up to 32 GB | 153 | ~17 |
| Apple M6 (Aug 2026) | up to 32 GB | 153 / 170 | ~19 |
| Apple M5 Ultra (Aug 2026) | up to 512 GB | 1 200 | ~137 |
| AMD Ryzen AI Max+ 395 | up to 128 GB unified | 256 | ~29 |
| NVIDIA DGX Spark | 128 GB unified | 273 | ~31 |
| Snapdragon X2 Elite / Extreme | — | 152 / 228 | ~17 / ~26 |
| NVIDIA RTX 5070 | 12 GB | 672 | ~77 |
| NVIDIA RTX 5070 Ti | 16 GB | 896 | ~102 |
| NVIDIA RTX 5080 | 16 GB | 960 | ~110 |
| NVIDIA RTX 4090 | 24 GB | 1 008 | ~115 |
| NVIDIA RTX 5090 | 32 GB | 1 792 | ~205 |
| AMD RX 9070 XT | 16 GB | 640 | ~73 |
| Intel Arc B580 | 12 GB | 456 | ~52 |
| Desktop DDR5-6000, 2 channels | — | ~96 | ~11 |
| EPYC server, 12 DDR5 channels | — | ~460 | ~50 |

The tok/s column illustrates the formula, not a promise; runa always prints an interval and refines it with measurements. Memory bandwidth matters more than core count and TFLOPS for decoding; for prefill (prompt processing) it is the other way around.

### 6.5. Verdict

```
Verdict  FITS · hybrid · 36/48 layers on GPU, experts on CPU · confidence: medium (estimate)
Memory   GPU 11.2 / 12.0 GiB · CPU 15.9 / 32.0 GiB
Speed    decode 9–14 tok/s · prefill 220–350 tok/s · first token (2k prompt) ≈ 7 s
Warn     decode below 15 tok/s: thinking with budget 2048 will take ~3 min per answer
Try      Q4_K_M → IQ4_XS saves 1.1 GiB (all layers on GPU, ≈ 2× faster) · --kv q8_0 saves 0.6 GiB
```

Exit codes: 0 — fits, 1 — fits with warnings, 2 — does not fit. `--json` for scripts.

---

## 7. Deep thinking

### 7.1. Reasoning models that actually run locally (verified)

| Model | Size | Date | License | Thinking control |
|--------|--------|------|----------|----------------------|
| LFM2.5-1.2B-Thinking / LFM2.5-2.6B | 1.2B / 2.6B | 2026 | LFM Open | always thinks |
| SmolLM3-3B | 3B | Jul 2025 | Apache-2.0 | `/think` / `/no_think` |
| Gemma 4 E2B / E4B / 12B / 26B-MoE / 31B | 2–31B | Apr–Jun 2026 | Apache-2.0 | `<\|think\|>` token in system prompt; no explicit off switch |
| gpt-oss-20b / 120b | 21B-A3.6B / 117B-A5.1B | Aug 2025 | Apache-2.0 | `Reasoning: low/medium/high` in system prompt (Harmony) |
| Qwen3 4B–32B, Qwen3-30B-A3B | 4–32B | 2025 | Apache-2.0 | `enable_thinking` in template, `/no_think` |
| Qwen3.6-27B (dense), Qwen3.6-35B-A3B (MoE) | 27B / 35B-A3B | 2026 | Apache-2.0 | same as Qwen3 |
| Phi-4-reasoning-vision-15B | 15B | Mar 2026 | MIT | always thinks; vision |
| Olmo 3.1 Think 7B / 32B | 7B / 32B | 2026 | Apache-2.0 | separate Think variants |
| Granite 4.2 3B / 8B / 30B | Mamba-2 hybrid | 2026 | Apache-2.0 | `thinking` in template |
| Nemotron 3 Nano-Omni | Mamba-2 + MoE hybrid | Jun 2026 | NVIDIA Open | omni input |
| DeepSeek-V4-Flash | 284B-A13B | Jul 2026 | MIT | `thinking: {type: enabled}` (API); locally ≥ 160 GB in Q4 |

Server-side (not for a laptop): Qwen3.8-Max 2.4T-A95B (Aug 2026, own license), DeepSeek-V4-Pro 1.6T-A49B (Aug 2026, MIT), Kimi K2-Thinking / K2.5–K2.7 1T-A32B, GLM-5 / 5.1 / 5.2, Nemotron 3 Ultra 550B-A55B. Qwen3.7 and GLM-5.3 do not exist (see §13).

### 7.2. How engines and APIs control thinking

| Layer | Enable / disable | Budget | Level | Where reasoning appears in the response |
|------|----------------------|--------|---------|--------------------------|
| llama.cpp | `--chat-template-kwargs '{"enable_thinking":false}'`, `--reasoning-budget 0` | `--reasoning-budget N`, `--reasoning-budget-grace-tokens`, `--reasoning-budget-soft-ratio`, `--reasoning-budget-message`; per-request `reasoning_budget_tokens` | — | `reasoning_content` (`--reasoning-format deepseek`), `auto` detects format from template |
| Ollama | `think: false` | — | `think: low/medium/high/max` (gpt-oss: no max) | `message.thinking` |
| vLLM | reasoning parser | thinking budget (PR #37112) | `reasoning_effort` | `reasoning` / `reasoning_content` (depends on version) |
| SGLang | `--reasoning-parser` | — | — | `separate_reasoning` |
| OpenAI (GPT-6 Astra, GPT-5.6) | `reasoning.effort: none` (on Astra → 400) | — | `minimal/low/medium/high/xhigh/max` | Responses API: reasoning summary |
| Anthropic (Claude 5) | do not pass `thinking` | `budget_tokens` only for models < 4.6 (on the 5th family → 400) | `thinking: {type: adaptive}` + `output_config.effort: low/medium/high/xhigh/max`, `display: summarized/omitted/updates` | `thinking` blocks in stream (`thinking_delta`) |
| DeepSeek API | `thinking: {type: enabled}` | — | — | `reasoning_content` |
| Mistral API | — | — | `reasoning_effort` | — |
| OpenRouter | `reasoning.exclude` | `reasoning.max_tokens` | `reasoning.effort` | `reasoning`, `reasoning_details` |
| Gemini 3.x | — | — | `thinking_level: minimal/low/medium/high` | — |

### 7.3. Unification in runa

```rust
pub enum ThinkMode { Off, On, Budget { tokens: u32, grace: u32 }, Effort(Effort) }
```

| ThinkMode | Locally (ggml) | OpenAI | Anthropic |
|-----------|-----------------|--------|-----------|
| `Off` | `enable_thinking=false` in template or closing the block on the first token | `reasoning.effort = minimal` (or `none` where allowed) | no `thinking` |
| `On` | as trained | `medium` | `adaptive` + `effort: medium` |
| `Budget{n}` | forced closing: count tokens after the opening tag, at `n − grace` shift the closing-tag logit, at `n` insert it plus the phrase "Answer now." | nearest level per table | `adaptive` + nearest `effort` (budgets rejected on the 5th family) |
| `Effort(x)` | context shares: low 512, medium 2048, high 8192, max ∞ + model hint (gpt-oss `Reasoning: high`) | `x` | `x` |

Output is always split into `Event::Reasoning` and `Event::Text`; the server returns `reasoning_content`, like llama-server and DeepSeek — the most common field name among clients. The fit checker accounts for the thinking budget in response time.

Note on Gemma 4: reasoning is set by the `<|think|>` token in the system prompt, there is no clean off switch — for it `Off` is implemented as budget 0 (immediate block closing).

---

## 8. Audio and video

### 8.1. What engines can do

| Engine | Audio input | Video input | How |
|--------|-----------|-----------|-----|
| llama.cpp mtmd | Ultravox 0.5, Voxtral Mini 3B, Qwen2-Audio, Qwen3-ASR 0.6B/1.7B (per `docs/multimodal.md`) | yes, since June 2026 (PR #24269): `mtmd-cli --video file.mp4`, frames via ffmpeg subprocess | encoder mmproj file + main GGUF |
| llama.cpp mtmd, vision | — | frames as images | Qwen3-VL, Gemma 3/4, InternVL, MiniCPM-V, Pixtral, LLaVA |
| whisper.cpp 1.9.3 / whisper-rs 0.16 | ASR (99 languages, large-v3-turbo) | — | C, ggml; Metal/CUDA/CoreML |
| sherpa-onnx (official Rust binding; `sherpa-rs` archived 6 June 2026) | Parakeet-TDT 0.6B v3 (25 European languages, including Russian, CC-BY-4.0), Canary, SenseVoice, Moonshine | — | ONNX Runtime |
| mistral.rs | claimed (Phi-4-multimodal and others), not confirmed in 0.9.3 release | Qwen-VL | candle |
| Nexa SDK | Qwen3-Omni (omni: audio + video + text) | yes | own runtime, NPU |
| voxtral-mini-realtime-rs | Voxtral Mini 4B Realtime (ASR + TTS) | — | Burn, Vulkan/Metal/WASM |

Not supported in llama.cpp (as of September 2026): Qwen3-Omni audio, Qwen3.5-Omni audio (no weights — API only), Gemma 4 E2B/E4B audio, LFM2-Audio. For them — the ASR → text route or mistral.rs/Nexa as an optional backend.

### 8.2. Models (verified)

| Model | Input | Size | License | Date |
|--------|------|--------|----------|------|
| Voxtral Mini 3B 2507 / Small 24B 2507 | audio → text, dialogue | 3B / 24B | Apache-2.0 | 2025 |
| Voxtral Mini 4B Realtime 2602 | streaming ASR | 4B | Apache-2.0 | Feb 2026 |
| Qwen3-ASR 0.6B / 1.7B | ASR | — | Apache-2.0 | 2026 |
| Parakeet-TDT 0.6B v3 | ASR, 25 languages (RU included) | 0.6B | CC-BY-4.0 | Aug 2025 |
| whisper large-v3-turbo | ASR, 99 languages | 0.8B | MIT | 2024 |
| Qwen3-Omni-30B-A3B (Instruct / Thinking) | audio + video + text → text/speech | 30B-A3B | Apache-2.0 | Sep 2025 |
| Gemma 4 E2B / E4B | audio + images + video | 2B / 4B | Apache-2.0 | Apr 2026 |
| Gemma 4 12B / 26B-MoE / 31B | images + video (no audio) | — | Apache-2.0 | Apr–Jun 2026 |
| Qwen3-VL | images + video | 2B–235B | Apache-2.0 | 2025 |
| MiniCPM-o 4.5 | omni | — | open | Feb 2026 |
| LLaVA-OneVision-2 | images + video | — | unspecified | Apr 2026 |
| Nemotron 3 Nano-Omni | omni | hybrid | NVIDIA Open | Jun 2026 |
| Phi-4-multimodal | audio + images | 5.6B | MIT | 2025 |

### 8.3. Routes in runa

```
audio ──▶ decode (symphonia/hound) ──▶ 16 kHz mono f32 (rubato)
   ├─ native : model has audio mmproj ─▶ mtmd audio chunk ─▶ LLM
   ├─ asr    : whisper.cpp | parakeet ─▶ transcript ─▶ LLM (any model, any cloud)
   └─ cloud  : OpenAI input_audio (gpt-audio) | Anthropic: transcript only

video ──▶ ffmpeg-sidecar ──▶ frames @1 fps, ≤ 32, scene-change keyframes ──▶ VLM (mtmd) or cloud images
      └─▶ audio track ──▶ audio route above
```

Cloud limitations (verified against documentation as of September 2026): OpenAI — images, PDF, `input_audio` for audio models, **no video**; Anthropic — images and PDF, **neither audio nor video**; the Anthropic OpenAI-compatible layer does not support PDF and audio and truncates thinking — use only the native Messages API. Gemini via the OpenAI-compatible endpoint — images and audio, no video.

Rust crates (current versions): `cpal` 0.18.2 (capture), `symphonia` 0.5.5, `hound`, `rubato` 3.0.0, `ffmpeg-sidecar` 2.5.2 (ffmpeg binary, not linked — no GPL issues), `rsmpeg` 0.18 (if linking is needed), `image`, `fast_image_resize`, `whisper-rs` 0.16.0.

---

## 9. Cloud APIs from Rust

| Provider | Current models and prices ($/1M in / out) | Thinking | Multimodality | Crate |
|----------|-------------------------------------------|----------|-------------------|-------|
| Anthropic | `claude-fable-5-1` 10 / 50 (1M context), `claude-opus-5` 5 / 25, `claude-sonnet-5` 2 / 10, `claude-haiku-4-5` 1 / 5 | `thinking: {type: adaptive}` + `output_config.effort`; `budget_tokens` only < 4.6 | images, PDF, Files API; SSE streaming; `stop_reason: refusal`; batches −50%; prompt caching | no official Rust SDK → thin `reqwest` + `eventsource-stream` (D9); alternatives: `adk-anthropic` 2.2.0 (adaptive thinking, effort, Files), `misanthropic`, `anthropic-sdk-rust` |
| OpenAI | GPT-6 Astra 10 / 50; GPT-5.6 Sol / Terra / Luna; Responses API (recommended), Chat Completions supported, Assistants closed 26 Aug 2026 | `reasoning.effort: none…max` | images, PDF, `input_audio` (gpt-audio); Realtime API; no video | `async-openai` 0.41.3 (Jul 2026): streaming, audio, Realtime |
| OpenAI-compatible (OpenRouter, DeepSeek, Groq, Gemini, llama-server, runa serve) | — | `reasoning`, `reasoning_effort`, `thinking` — normalized by adapter | depends | same `async-openai` with `base_url` |
| Multi-provider wrappers | — | partial | partial | `genai` 0.6 (May 2026), `rig-core` 0.42 (Aug 2026) — no full control over thinking, hence not used as the base |

Secrets — only from `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` or the system keychain; a config containing a key is rejected.

---

## 10. runa architecture (brief)

```
runa (CLI, config, server)
 ├─ runa-core     Backend trait · Request/Event · ThinkConfig · Mode
 ├─ runa-fit      gguf header (local/HTTP range) · hw probe · estimator · planner · calibration db
 ├─ runa-engine   llama-cpp-2: load(Placement) · sampling loop · budget forcing · mtmd · state cache
 ├─ runa-media    symphonia/rubato/ffmpeg-sidecar · frame sampling · whisper-rs / sherpa-onnx
 ├─ runa-cloud    openai (async-openai, Responses) · anthropic (reqwest + SSE) · price table
 └─ runa-kernels  C/.S kernels · cc build · runtime dispatch · reference impls · criterion gate
```

`runa run` flow: config → model reference → `runa-fit` (verdict, `Placement`) → `on_unfit` policy → backend (local or cloud) → `Reasoning`/`Text` event stream → output; on completion `runa-fit` writes the measurement to the calibration DB.

Details: `plan.md` (D1–D16, seven crates, metrics M1–M10, phases P0–P7).

---

## 11. Work order and why

1. **P0 (1 wk)** — skeleton, CI on three OSes, spikes verifying risky assumptions (`llama-cpp-2` + Metal/CUDA, `mtmd`), baseline `llama-bench` measurements for three reference models in three modes.
2. **P1 (2–3 wks) — fit checker before the engine**, because it is the differentiator from analogues, and it can be developed without a GPU: GGUF parser, HTTP range, model descriptor, KV estimate with exceptions, buffers, hardware probe, planner, speed model, calibration, verdict.
3. **P2 (2–3 wks)** — engine, three modes, `auto`, MoE hybrid, quantized KV, prompt cache, `runa bench`.
4. **P3 (2 wks)** — thinking (parser, budget, levels) and clouds (OpenAI, Anthropic, routing, server v1).
5. **P4 (2–3 wks)** — audio (ASR, native), video (frames), cloud limitations, media accounting in fit.
6. **P5 (2–3 wks, with a hard time cap)** — profiling, `runa-kernels` with the ≥ 5% gate, speculative decoding, perf CI.
7. **P6 (2 wks)** — server (multi-model, Anthropic-compatible endpoint), `cargo-dist` packaging, Homebrew, documentation, 1.0 release.
8. **P7** — mistral.rs as a feature, RPC distribution, tool calling, NPU, per-machine model recommendations.

Total ≈ 14–18 weeks to 1.0 for one developer with agents; each plan task has machine-runnable verification.

---

## 12. Target comparison: runa vs analogues

| Feature | Ollama | LM Studio | llama.cpp | llmfit | **runa** |
|-----|--------|-----------|-----------|--------|----------|
| Pre-download check | no | no | no | from table | from GGUF header |
| Speed forecast | no | no | no | yes | yes, interval + calibration |
| What to do if it does not fit | silent CPU | error | fit | — | list: different quant, ctx, KV, cloud |
| Thinking local + cloud | levels / own clouds | on/off | budget | — | one knob for everything |
| Audio / video | no | no | yes | — | yes + routes + fit accounting |
| OpenAI / Anthropic as backend | no | no | no | — | yes + `on_unfit=cloud` |
| Language | Go | TS | C++ | Rust | Rust + C/asm |

---

## 13. Fact-check: what was not confirmed and what was fixed

| # | Claim from agent reports | Verdict | Fact |
|---|-------------------------------|---------|------|
| 1 | Bits per weight: Q4_0 4.34, Q8_0 8.0, Q4_K_M 4.58 | refuted | agent confused file GB with bpw; exact values per `ggml-common.h` in §6.2 (Q4_0 4.5, Q8_0 8.5, Q4_K 4.5) |
| 2 | Compute buffers "20–50 MB" | refuted | hundreds of MiB – GiB; calibrated against logs |
| 3 | Ryzen AI Max+ 395 ≈ 96 GB/s | refuted | 256 GB/s (LPDDR5X-8000, 256-bit) |
| 4 | RTX 5070 Ti and RTX 5070 ≈ 576 GB/s | refuted | 896 and 672 GB/s |
| 5 | llama-server field `thinking_budget_tokens` | refuted | `reasoning_budget_tokens` (PR #25961) |
| 6 | llama.cpp "v0.27+" | refuted | semantic tags exist, but the latest is v0.4.0 (4 Sep 2026); builds b10853 |
| 7 | mtmd has no video | refuted | video added in PR #24269 (8 Jun 2026), via ffmpeg |
| 8 | mistral.rs 0.8.2, Apache-2.0 | refuted | 0.9.3 (7 Sep 2026), MIT |
| 9 | mistral.rs supports audio/video natively | unconfirmed | absent from 0.9.3 release notes; claimed in README — treat as "partial" |
| 10 | Ollama MLX — primary backend on Apple | refuted | preview since 0.19 (30 Mar 2026), only ≥ 32 GB, one model |
| 11 | Qwen3.6-35B — dense, text-only | refuted | Qwen3.6-35B-A3B is MoE; there is a dense Qwen3.6-27B |
| 12 | Kimi K2 ≈ 13B | refuted | 1T-A32B |
| 13 | Qwen3.7, GLM-5.3-Flash | not found | lines: Qwen3.5 / 3.6 / 3.8; GLM-5 / 5.1 / 5.2 |
| 14 | Qwen3.8-Max — CC-BY-NC | refuted | own "qwen3.8-max" license |
| 15 | DeepSeek V4 "Think High/Max" | not found | modes not documented |
| 16 | Gemma 4 12B in the April release | partial | 12B added 3 Jun 2026; April: E2B, E4B, 26B-MoE, 31B |
| 17 | Gemma 4: audio in all sizes; thinking off switch | refuted | audio only on E2B/E4B; thinking via `<\|think\|>` token, no off switch |
| 18 | Qwen3-Omni — 2026, proprietary; ggml-org published GGUFs with audio | refuted | Qwen3-Omni — Sep 2025, Apache-2.0; Qwen3.5-Omni (Mar 2026) — API only; no official GGUFs with audio encoder, llama.cpp unsupported |
| 19 | sherpa-rs 0.6.8 — current binding | refuted | archived 6 Jun 2026; use the official sherpa-onnx Rust binding |
| 20 | rubato 0.19/0.20, symphonia 0.17 | refuted | rubato 3.0.0 (May 2026), symphonia 0.5.5 (Oct 2025) |
| 21 | Nexa AI acquired by Qualcomm | unconfirmed | main repo NexaAI/nexa-sdk, Qualcomm has a fork |
| 22 | Lemonade: 55 tok/s Qwen3.5-35B-A3B on Ryzen AI Max+ | unconfirmed | AMD publishes ~24.5 tok/s for Qwen3.8-27B; 61 tok/s on iGPU for another model |
| 23 | OpenAI: current are o1/o3/o4-mini | refuted | GPT-6 Astra, GPT-5.6 Sol/Terra/Luna |
| 24 | OpenAI accepts video | refuted | images/PDF/audio only |
| 25 | Anthropic OpenAI-compatible layer is full-featured | refuted | no PDF, no audio, thinking truncated — native API only |
| 26 | Mistral: `prompt_mode` parameter | refuted | `reasoning_effort` |
| 27 | reqwest-eventsource abandoned | refuted | 0.6.0 (30 Aug 2026); suitable, like eventsource-stream 0.2.3 |
| 28 | `adk-anthropic` does not exist | refuted (my doubt) | 2.2.0 (1 Sep 2026): adaptive thinking, effort, Files API |
| 29 | Canary-1B-v2 = same languages and license as Parakeet | refuted | separate model, coverage and license do not match |
| 30 | mmproj accounted for in `--fit` | unconfirmed | fit-params documentation does not mention it; runa counts it itself |
| 31 | Exact K-quant block sizes | unconfirmed by web agent | taken from `ggml-common.h` (block_q4_K = 144 bytes etc.); verify in P1.3 with tests on real files |
| 32 | vLLM renamed `reasoning_content` → `reasoning` | partial | version-dependent; adapter normalizes both |

Confirmed (sample): llama.cpp `--fit`/`--fit-margin`/`--fit-target`/`llama_params_fit`/`llama-fit-params`, `--fit on` by default; all `--reasoning-budget*` flags; Ollama 0.33.2 and issue #14258; `llama-cpp-2` feature `mtmd`; `rig-llama-cpp`; llmfit; gguf-parser-go MAX TPS; `lms load --estimate-only`; M5 Ultra 1.2 TB/s and 512 GB; M6 153/170 GB/s; DGX Spark 273 GB/s; Voxtral 3B/4B/24B Apache-2.0; Parakeet v3 with Russian; whisper-rs 0.16; async-openai 0.41.3; Anthropic prices and parameters (from current API docs).

---

## 14. Sources

Engines and bindings
- llama.cpp: https://github.com/ggml-org/llama.cpp — `tools/fit-params`, `tools/server/README.md`, `docs/multimodal.md`, `docs/speculative.md`, `docs/build.md`, `common/reasoning-budget.cpp`; PR #25961 (reasoning budget), PR #24269 (video), PR #11607 (reasoning-format), issue #24055 / PR #22929 (ctx-checkpoints)
- llama-cpp-2: https://crates.io/crates/llama-cpp-2 · https://github.com/utilityai/llama-cpp-rs
- rig-llama-cpp: https://crates.io/crates/rig-llama-cpp
- mistral.rs: https://github.com/EricLBuehler/mistral.rs/releases/tag/v0.9.3
- candle: https://github.com/huggingface/candle · burn: https://github.com/tracel-ai/burn
- whisper.cpp / whisper-rs: https://github.com/ggml-org/whisper.cpp · https://crates.io/crates/whisper-rs
- sherpa-onnx: https://github.com/k2-fsa/sherpa-onnx · sherpa-rs (archive): https://github.com/thewh1teagle/sherpa-rs
- Nexa SDK: https://github.com/NexaAI/nexa-sdk
- voxtral-mini-realtime-rs: https://github.com/TrevorS/voxtral-mini-realtime-rs

Runners and estimators
- Ollama: https://github.com/ollama/ollama/releases/tag/v0.33.2 · https://github.com/ollama/ollama/issues/14258 · https://ollama.com/blog/mlx · https://docs.ollama.com/capabilities/thinking
- LM Studio: https://lmstudio.ai/docs/cli/local-models/load
- gguf-parser-go: https://github.com/gpustack/gguf-parser-go
- llmfit: https://github.com/AlexsJones/llmfit
- AMD Lemonade: https://www.phoronix.com/news/Lemonade-SDK-10.2-Released · https://www.amd.com/en/blogs/2026/run-qwen-3-8-27b-on-amd-ryzen-ai-max-and-radeon-graphics-cards-day-0.html
- vLLM: https://docs.vllm.ai/en/latest/features/reasoning_outputs/ · https://github.com/vllm-project/vllm/pull/37112

Models
- Qwen: https://huggingface.co/Qwen/Qwen3.6-35B-A3B · https://huggingface.co/Qwen/Qwen3.6-27B · https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B · https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct
- Gemma 4: https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/ · https://blog.google/innovation-and-ai/technology/developers-tools/introducing-gemma-4-12b/
- gpt-oss: https://huggingface.co/openai/gpt-oss-20b
- DeepSeek V4: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro · https://api-docs.deepseek.com/guides/thinking_mode/
- Kimi: https://huggingface.co/moonshotai/Kimi-K2-Thinking · GLM: https://huggingface.co/collections/zai-org/glm-52
- Nemotron 3: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 · https://huggingface.co/blog/nvidia/nemotron-3-nano-omni-multimodal-intelligence
- Liquid: https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking · Granite 4.2: https://www.ibm.com/granite/docs/models/granite4-2 · Olmo 3.1: https://huggingface.co/allenai/Olmo-3.1-32B-Think · Phi-4-reasoning-vision: https://huggingface.co/microsoft/Phi-4-reasoning-vision-15B · SmolLM3: https://huggingface.co/HuggingFaceTB/SmolLM3-3B
- Voxtral: https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602 · Parakeet: https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3 · MiniCPM-o: https://github.com/OpenBMB/MiniCPM-V · LLaVA-OneVision-2: https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-2

Cloud APIs
- Anthropic: https://platform.claude.com/docs/en/build-with-claude/thinking · https://platform.claude.com/docs/en/build-with-claude/effort · https://platform.claude.com/docs/en/api/openai-sdk · https://platform.claude.com/docs/en/build-with-claude/working-with-messages
- OpenAI: https://developers.openai.com/api/docs/models · https://developers.openai.com/api/docs/guides/reasoning · https://developers.openai.com/api/docs/guides/audio · https://developers.openai.com/api/docs/guides/migrate-to-responses
- Mistral: https://docs.mistral.ai/capabilities/reasoning · OpenRouter: https://openrouter.ai/docs/docs/best-practices/reasoning-tokens · Gemini: https://ai.google.dev/gemini-api/docs/generate-content/thinking · https://ai.google.dev/gemini-api/docs/openai
- Crates: https://docs.rs/crate/async-openai/latest · https://docs.rs/adk-anthropic · https://docs.rs/reqwest-eventsource · https://docs.rs/eventsource-stream · https://github.com/jeremychone/rust-genai · https://docs.rs/crate/rig-core/latest

Hardware and speed physics
- Kapoulkine, "LLM inference speed of light": https://zeux.io/2024/03/15/llm-inference-sol/
- Apple M6 / M5 Ultra: https://www.apple.com/newsroom/2026/08/apple-introduces-m6-and-m5-ultra-for-a-big-leap-in-performance-and-ai-compute/
- AMD Ryzen AI Max+ 395: https://www.amd.com/en/products/processors/laptop/ryzen-pro/ai-max-pro-300-series/amd-ryzen-ai-max-plus-pro-395.html
- Intel Arc B580: https://www.intel.com/content/www/us/en/products/sku/241598/intel-arc-b580-graphics/specifications.html
- RTX 50: https://www.techspot.com/news/106565-nvidia-reveals-complete-geforce-rtx-5070-rtx-5070.html · RX 9070 XT: https://www.techspot.com/review/2961-amd-radeon-9070-xt/ · Snapdragon X2: https://www.cnx-software.com/2025/10/02/snapdragon-x2-elite-extreme-and-x2-elite-processors-target-high-end-windows-pcs/

Rust
- Rust 1.98: https://blog.rust-lang.org/2026/08/20/Rust-1.98.0/ · SVE/SME 2026 goal: https://rust-lang.github.io/rust-project-goals/2026/scalable-vectors.html · AMX: https://github.com/rust-lang/rust/issues/126622 · portable SIMD: https://github.com/rust-lang/portable-simd
- GPU: https://crates.io/crates/cudarc · https://docs.rs/objc2-metal · https://crates.io/crates/ash · https://wgpu.rs/
- Media: https://crates.io/crates/symphonia · https://crates.io/crates/rubato · https://crates.io/crates/ffmpeg-sidecar · https://crates.io/crates/cpal · https://crates.io/crates/rsmpeg

---

## 15. Model sources and loading (P18)

Checked 2026-10-14. Feeds plan P18 (`runa-models` crate, `roadmap.md`). A fact backed only by secondary sources is marked **unverified**.

### 15.1 What runa already ships

- `runa pull hf:<repo>:<file|quant>` downloads through `hf-hub` into `<data dir>/models/<owner>--<name>/`, checks size and SHA-256 against the Hub's `x-linked-size` / `x-linked-etag`, and records a `.verified` sidecar (`crates/runa/src/pull.rs`).
- `runa-fit` parses `hf:` / `https://` / local refs and reads GGUF headers over HTTP ranges with a header cache (`crates/runa-fit/src/remote.rs`).
- API keys come from the environment or the OS keychain and are rejected in config files (`crates/runa-cloud/src/secrets.rs`, P3.8).

### 15.2 Hugging Face sign-in

| Fact | Source |
| --- | --- |
| The Hub supports the OAuth device code flow: `POST https://huggingface.co/oauth/device`, then `POST https://huggingface.co/oauth/token` with `grant_type=urn:ietf:params:oauth:grant-type:device_code` | https://huggingface.co/docs/hub/oauth (2026-10-14) |
| A public OAuth app (no client secret) authenticates with the client id only, in device code or PKCE flows; an app must be registered in the user's settings | https://huggingface.co/docs/hub/oauth (2026-10-14) |
| Scopes include `read-repos` (personal repos) and `gated-repos` (public gated repos the user was granted, not private repos) | https://huggingface.co/docs/hub/oauth (2026-10-14) |
| `hf auth login` defaults to the browser device flow and stores the token under `~/.cache/huggingface/`; `--token $HF_TOKEN` is the non-interactive path | https://huggingface.co/docs/cli (2026-10-14); storage path **unverified** (from a mirror of the docs) |

### 15.3 Alternatives to Hugging Face

Free first.

| Service | What it hosts | Programmatic access | Sign-in | Source |
| --- | --- | --- | --- | --- |
| ModelScope (modelscope.cn) | Git-backed model repos, strongest in the Chinese ecosystem | Python SDK `HubApi` / `snapshot_download`, CLI `modelscope download` | SDK token, `modelscope login --token` | **unverified**: https://adg.csdn.net/694cf62e5b9f5f31781aaeee.html, https://tessl.io/registry/tessl/pypi-modelscope/files/docs/hub.md (no official REST reference found) |
| Docker Hub `ai/` namespace (Docker Model Runner) | GGUF, safetensors and DDUF packaged as OCI artifacts; works with GHCR and any OCI registry | `docker model package --gguf <file> --push <ref>`; `--push` needs a registry-qualified reference | Registry token | https://docs.docker.com/reference/cli/docker/model/package.md, https://www.docker.com/blog/publish-ai-models-on-docker-hub/ (2026-10-14) |
| KitOps ModelKit / CNCF ModelPack | Models, datasets and code as OCI artifacts with SHA-256 digests and Cosign signing | `kit` CLI, any OCI registry | Registry token | https://www.cncf.io/blog/2025/01/31/kitops-1-0-release-proven-in-production-and-looking-to-cncf/ (2026-10-14); feature details **unverified** |
| Ollama registry (registry.ollama.ai) | GGUF as manifest + `sha256-` blobs | Manifest, then blobs by digest | Public models need none | **unverified**: no official registry API doc; https://medium.com/@dewasheesh.rana/inside-ollamas-model-storage-understanding-blobs-and-manifests-06f1620dd0b2 |
| Kaggle Models | TF, JAX, PyTorch and other models | `kagglehub.model_download(handle)` | Public models need none; else `KAGGLE_USERNAME` + `KAGGLE_KEY` or `~/.kaggle/kaggle.json` | https://pypi.org/project/kagglehub/0.2.5/ (2026-10-14) |
| Civitai | Image-generation models (SD, LoRA) | `GET /api/v1/model-versions/{id}` returns `downloadUrl` | `Authorization: Bearer <key>`; a creator may require sign-in for downloads | https://developer.civitai.com/docs/api/public-rest (2026-10-14) |
| GitHub Releases | Any file by direct URL | HTTPS | Token for private repos | Per-file limit under 2 GiB: https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github (2026-10-14); a 2 GB vs 2 GiB mismatch is reported in https://github.com/orgs/community/discussions/61733 (**unverified**) |
| WiseModel (wisemodel.cn) | Chinese model hub | Python `wisemodel-hub` | Git token | https://pypi.org/project/wisemodel-hub/0.5.1/ (2026-10-14) |
| Cloudflare R2 (self-hosting) | Own files over S3 API or public URL | S3 / HTTPS | Account keys | Zero egress fees: https://www.cloudflare.com/en-gb/developer-platform/products/r2/ (2026-10-14); 10 GB-month free tier **unverified** (https://flarecalc.com/calculators/r2) |
| HF mirrors (e.g. hf-mirror.com) | Same API on another host | Endpoint switch | Same token | **unverified** |

Paid or with paid tiers, all with an API.

| Service | What it hosts | Programmatic access | Source |
| --- | --- | --- | --- |
| NVIDIA NGC | NVIDIA model catalog | `ngc registry model download-version org/[team/]model:version`, key in `NGC_API_KEY` | https://docs.nvidia.com/brev/guides/inference-deployment/ngc-developer-guide (2026-10-14) |
| Weights & Biases Registry | Model artifacts | `wandb.Api().artifact("wandb-registry-<registry>/<collection>:<alias>").download()` | https://docs.wandb.ai/guides/registry/download_use_artifact (2026-10-14) |
| RunPod Serverless model caching | Hugging Face models pre-placed on worker hosts under `/runpod-volume/huggingface-cache/hub/` | Endpoint setting; Hugging Face models only | https://docs.runpod.io/serverless/endpoints/model-caching (2026-10-14) |
| Replicate, AWS SageMaker JumpStart, Azure AI Foundry, Vertex AI Model Garden, JFrog Artifactory (HF proxy) | Cloud model catalogs and enterprise proxies | Vendor APIs | **unverified**: not checked in this pass |

### 15.4 Libraries

| Need | Crate or tool | Source |
| --- | --- | --- |
| OCI registry client (Ollama, Docker `ai/`, GHCR) | `oci-client` (formerly `oci-distribution`, ORAS project): manifests, blobs, digest validation, token caching | https://docs.rs/oci-distribution/latest/oci_distribution/struct.Client.html (2026-10-14) |
| Wasm inference interface | `wasmtime-wasi-nn`: backends `onnx`, `openvino` (64-bit), `winml` (Windows), `pytorch`, each behind a Cargo feature; the README calls the crate experimental | https://docs.wasmtime.dev/api/wasmtime_wasi_nn/backend/index.html (2026-10-14) |
| Wasm + llama.cpp | WasmEdge `wasi_nn-ggml` plugin; LlamaEdge apps load GGUF through it | https://wasmedge.org/docs/contribute/source/plugin/wasi_nn/, https://github.com/second-state/LlamaEdge (2026-10-14) |
| Model signatures | OpenSSF `model-signing` v1.0 (Sigstore), Python only; signs a manifest of every file in the model directory | https://blog.sigstore.dev/model-transparency-v1.0, https://pypi.org/project/model-signing/1.0.0/ (2026-10-14) |
