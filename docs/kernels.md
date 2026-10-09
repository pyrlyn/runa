# runa-kernels: benchmarks and adoption gates

Per-kernel benchmark results and adoption gates (plan D14, D23). P5.2 ships
the crate skeleton (C). New kernels prefer Zig (C ABI, `@Vector`) over new C;
keep C/`.S` only where Zig is worse (ggml FFI, SME2/AMX).

## P5.2 bootstrap — softmax (scalar)

| Implementation | Status | Notes |
|----------------|--------|-------|
| C scalar (`c/scalar.c`) | shipped | always linked |
| Rust reference | shipped | `softmax_f32_reference`, unit tests |

## P5.3 — sampling + SIMD softmax (Zig)

| Implementation | Status | Notes |
|----------------|--------|-------|
| Zig softmax (`@Vector`) | shipped | `runa_softmax_f32_scalar` (C ABI); LLVM emits host SIMD |
| Zig top-k / top-p / min-p | shipped | `runa_top_k_f32` / `runa_top_p_f32` / `runa_min_p_f32`; `page_allocator` scratch |
| Zig `sample_token` | shipped | temperature + filters + xorshift |
| C NEON/AVX2 softmax | not linked | `c/` kept for reference; duplicate C ABI symbols if linked with Zig |
| Engine plug-in | shipped | `SamplingConfig::kernel_sampler` / `RUNA_KERNEL_SAMPLER=1` |

Measured on this machine (`cargo bench -p runa-kernels --bench softmax`, 151936-wide softmax, Apple Silicon aarch64):

| Op | Time | vs Rust ref | D14 gate |
|----|------|-------------|----------|
| `softmax_f32` Zig (`SoftmaxImpl::ZigVector`) | 405.8 µs | **0.91×** (slower) | 2× isolated → **REJECT** |
| `softmax_f32_reference` | 368.8 µs | — | — |
| `sample_token` 4k vocab | 112.2 µs | — | below e2e 5% |

Default generation still uses ggml's sampler chain. Own kernels remain opt-in
via `RUNA_KERNEL_SAMPLER=1` until the D14 gate is met. End-to-end 5% tg on a
150k-vocab model was not measured (no fixture); isolated gate decides **REJECT**.

## P5.6 — n-gram speculative decoding

Trigram lookup drafts, greedy-verified against the target argmax (`--ngram`,
or implied by `--draft`). Identity vs non-ngram holds at `--temperature 0`
(engine test on qwen2-0.5B). `--draft` reserves the GGUF in the fit planner;
it does not load a second model.

The ≥1.3× tg gate on code prompts is skipped in CI (no dedicated ngram
bench job). Record a ratio here when a machine is profiled.

## P5.4 — image preprocess (NEON/AVX2 vs `fast_image_resize`)

Measured on this machine (`cargo bench -p runa-media --bench media`, 336² RGB):

| Op | Time | vs scalar | D14 gate |
|----|------|-----------|----------|
| `normalize_rgb` scalar (default) | 40.8 µs | — | — |
| `normalize_rgb_simd` NEON | 26.8 µs | **1.52×** | 2× isolated → **REJECT** |
| `patchify_rgb` 16×16 | 87 µs | — | below e2e 5% |
| nearest `resize_rgb` (P4.9) | 73 µs | — | — |
| `fast_image_resize` HQ (K3) | 1.82 ms | 0.04× vs nearest | **REJECT** (keep nearest) |

Default `normalize_rgb` stays scalar. SIMD is exported as `normalize_rgb_simd`
for the comparison (equivalence test on a 337-byte tail). Resize stays
nearest-neighbor; HQ FIR was already slower in K3.


## P5.5 — SME2 / AMX quantized mat-vec (Q4_K / Q8_0)

Research pin: `llama-cpp-2` **0.1.133** → llama.cpp **b7709** (`1051ecd`,
2026-01-12). Sources inspected via
[`ggml/src/ggml-cpu/`](https://github.com/ggerganov/llama.cpp/tree/1051ecd28907d2ca0a15c135f190fe415d0a3d1b/ggml/src/ggml-cpu).

### Dev machine

| Property | Value |
|----------|-------|
| CPU | Apple M3 Max |
| SME2 | `hw.optional.arm.FEAT_SME2` = **0** (SME2 is M4+) |
| AMX | not available (x86 Sapphire Rapids+) |

Cannot compile, run, or equivalence-test SME2/AMX assembly on this box.

### ggml status at b7709

**Intel AMX (x86, `GGML_USE_AMX` / `__AMX_INT8__`)**

| Quant | Path | Symbols / files |
|-------|------|-----------------|
| Q4_K | AMX weight-only GEMM/GEMV | `ggml_backend_amx_mul_mat` in `amx/mmq.cpp` (`GGML_TYPE_Q4_K` case); buffer `ggml_backend_amx_buffer_type` in `amx/amx.cpp` |
| Q8_0 | same | `GGML_TYPE_Q8_0` case in `amx/mmq.cpp` |

m=1 (prompt) fast path falls back to AVX-512-VNNI when batch is small (comment
in `amx/mmq.cpp`).

**AArch64 KleidiAI / SME (Apple M4+ with `-DGGML_CPU_KLEIDIAI=ON`)**

| Quant | SME2 KleidiAI | Other ggml CPU path |
|-------|---------------|---------------------|
| Q8_0 | yes — `kai_run_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa` (GEMM), `kai_run_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot` (GEMV) in `kleidiai/kernels.cpp`; dispatch `ggml_kleidiai_select_kernels_q8_0` | i8mm/dotprod tables in same file |
| Q4_K | **no** KleidiAI table entry at this pin | repack + gemv `ggml_gemv_q4_K_8x4_q8_K` / `gemv<block_q4_K,…>` in `repack.cpp`; `ggml_vec_dot_q4_K_q8_K` in `arch/arm/quants.c` |

KleidiAI covers `GGML_TYPE_Q4_0` and `GGML_TYPE_Q8_0` under `CPU_FEATURE_SME`,
not `GGML_TYPE_Q4_K`. Q4_K prompt mat-vec on Apple Silicon uses ggml's repack /
i8mm / NEON vec-dot paths instead.

### Decision: **SKIP**

- ggml already owns Q4_K and Q8_0 mat-vec on AMX; Q8_0 (and Q4_0) on SME2 via
  KleidiAI; Q4_K on AArch64 via repack + `ggml_vec_dot_q4_K_q8_K`.
- No gap large enough to justify a custom `.S` kernel that could beat ggml under
  the D1/D14 gate — and this machine cannot build or test one anyway.
- No `runa-kernels` objects, dispatch, or gate numbers recorded.
