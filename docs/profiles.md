# Profiling and polyglot evaluation

## K3 — Polyglot escape-hatch evaluation (D19, gate D1)

**Candidate:** image preprocessing `resize` + `normalize` + `patchify` (P5.4) via
`fast_image_resize` (Rust, pure) vs C `stb_image_resize` (single-header, `cc`).

**Interface:** `runa-media::image::resize()` is Rust-owned (`ImageBuf` → `ImageBuf`);
the C path would be `runa-media/src/image_c.c` behind `#[cfg(feature = "c-resize")]`
with identical `ImageBuf` in/out and `#[cfg]` dispatch.

**Benchmark (M3 Max, 512×512 RGB → 336×336, 100 iterations, `criterion`):**

| Impl | mean | ± | vs Rust |
|------|------|---|---------|
| Rust `fast_image_resize` (default) | 1.82 ms | 0.04 | — |
| C `stb_image_resize` (`cc -O3`) | 1.79 ms | 0.05 | **+1.7%** |

*Equivalence:* `cargo test -p runa-media --test image` pixel-wise `max_diff ≤ 1` (u8) on
`tests/fixtures/shapes.png` and 10 synthetic gradients.

**Gate D1:** C must beat Rust by **≥5% end-to-end or ≥2× on the isolated op**.
Result **+1.7%** does **not** meet the gate.

**Decision:** **REJECT** — keep the Rust path. No `image_c.c` merged; the Rust
interface stays. Re-evaluate at each toolchain bump (D19).

**Other candidates considered (no benchmark yet):**
- Audio resample `rubato` (Rust) vs `libsamplerate` (C): `rubato` is already
  2–3× faster than `libsamplerate` on x86_64 in prior P4.1 spikes; no C path needed.
- Whisper `sherpa-onnx` (Rust binding over C++ ONNX) vs `whisper-rs` (C): Parakeet
  is the highest-payoff polyglot candidate for P4.2, but it requires the `sherpa-onnx`
  C++ runtime and model download (25 languages). Evaluation deferred to P4.2.

## K4 — Parallelism audit (D20)

**Audit 2026-09-08 (M3 Max, 4 cores, `moon run :test` vs `cargo test --workspace`):**

| Path | Before | After | Wall-time | Decision |
|------|--------|-------|-----------|----------|
| `cargo test --workspace` (8 crates, serial) | `cargo test -p runa-fit && cargo test -p runa ...` | `moon run :test` (parallel, cached, 4 cores) | 12.3s → 6.1s (2.0×) | Keep moon; cargo remains source of truth |
| `runa doctor --bench` (CPU memcpy, 64 MiB) | single-threaded `copy_from_slice` | `rayon` 4 chunks `par_chunks_mut` | 0.04s → 0.02s (2×) | Keep rayon; gate D1 met |
| `runa-fit::Fetcher::fetch_header` | sync `reqwest::blocking` (current) | `reqwest::async` + `tokio` | no wall-time win (header fetch is latency-bound, one range per round-trip) | Keep sync (D20: parallelism that doesn't move wall-clock is removed) |
| `runa-media` audio decode (10 clips) | serial `symphonia` per file | `rayon` per file | 1.2s → 0.7s (1.7×, 4 cores) | Keep serial for now (gate not met 2×); re-evaluate with 32 clips |

*No serial hot path remains without a bench justification. I/O-bound work stays sync where async doesn't help; CPU-bound work is data-parallel where measured ≥1.5×.*

Audit table lives here per K4 check; `moon run :test` vs `cargo test` wall-time recorded above.

## P5.1 — Profiling harness (top-10 ops per mode)

**Tools (local + nightly P5.8):**

| Tool | Install | Command |
|------|---------|---------|
| `samply` | `cargo install samply` | `samply record cargo run -p runa-engine --example gen -- tests/fixtures/qwen2-0_5b-instruct-q4_0.gguf "hi"` |
| `cargo flamegraph` | `cargo install flamegraph` | `cargo flamegraph -p runa -- bench tests/fixtures/qwen2-0_5b-instruct-q4_0.gguf` |
| `criterion` (kernels) | dev-dep in `runa-kernels` | `cargo bench -p runa-kernels --bench softmax` |

**Procedure:** profile pp512 (prefill) and tg128 (decode) separately with `runa bench --json`;
for ggml internals use `samply` on the `gen` example or `runa bench` with Metal/CUDA enabled.
Record the top-10 symbols by sample count per mode in this file.

**Initial spike (M3 Max Metal, `gen` example, qwen2-0.5B Q4_0):** decode is dominated by
`ggml_metal` kernels (`kernel_mul_mv_q4_0`, `kernel_flash_attn`); Rust sampling is <1% of
tokens/sec wall time — aligns with D14 ordering (sampling kernel is P5.3, not hot yet).

**ggml scheduler callbacks:** not wired in the pinned `llama-cpp-2` 0.1.133 wrapper;
revisit when upgrading llama.cpp exposes stable per-op timing hooks.

| Mode | Machine | Top ops (sample %) | Notes |
|------|---------|-------------------|-------|
| tg128 decode | M3 Max Metal | `kernel_mul_mv_q4_0` ~45%, `kernel_flash_attn` ~30% | from `samply` on `gen` |
| pp512 prefill | — | pending | run `runa bench` + samply |
| cpu decode | — | nightly | `.github/workflows/perf.yml` (P5.8; GitHub-hosted until self-hosted) |

## P4.9 — Media pipeline profile (candidates for P5)

**Machine:** M3 Max, `cargo bench -p runa-media --bench media` (release, criterion, 20 samples).
**Reproduce:** same command; HTML under `target/criterion/`.
**Flame:** [profiles-p49.svg](profiles-p49.svg) — exclusive Rust time for a 5 s clip (32 VL frames + 5 s of 48 kHz audio resampled to 16 kHz). `samply` / `cargo flamegraph` were not on PATH; the SVG is built from the criterion means below, not a sampled stack.

| Op | Size | mean | 32-frame / 5 s clip |
|----|------|------|---------------------|
| `resize_rgb` nearest 512→336 | one frame | **73 µs** | 2.3 ms |
| `normalize_rgb` → [−1, 1] | 336² RGB | **42 µs** | 1.3 ms |
| `histogram_l1` (scene cut) | 336² pair | **642 µs** | **20.5 ms** |
| `resample_mono` rubato 48 k→16 k | 1 s PCM | **1.38 ms** | 6.9 ms |

**Mel:** not implemented in `runa-media`. Whisper.cpp (`whisper-rs` 0.16) builds the log-mel inside the C++ runtime after we hand it 16 kHz PCM. A Rust/C mel kernel would only pay off if we leave whisper; that is not on the P5 list.

**P5 ranking (gate ≥5% e2e or ≥2× isolated):**

1. **`histogram_l1`** — ~65% of this Rust slice, 9× a resize. Best isolated kernel (NEON/AVX histogram). Does **not** move VL/ASR e2e (ffmpeg + whisper dominate). Hold unless scene-detect shows up in a full `runa media video` samply.
2. **`resize_rgb`** — nearest is already 73 µs; K3 high-quality `fast_image_resize` was **1.82 ms** (25× slower). P5.4 is a quality swap, not a speed win over nearest.
3. **`normalize_rgb`** — 42 µs scalar; NEON SIMD 26.8 µs (**1.52×**, P5.4). D14 gate is 2× → REJECT, keep scalar.
4. **`resample_mono`** — K3 already rejected `libsamplerate`; rubato stays.
5. **whisper mel** — lives in ggml/whisper.cpp; not a `runa-kernels` task unless we replace ASR.

**ffmpeg / whisper** (not in the bench, order-of-magnitude from P4.1/P4.2/P4.5): decode + ASR for 5 s of audio is hundreds of ms to seconds. The Rust preprocess table is a few tens of ms. P5 time should stay on ggml sampling (P5.3) and, if ever, scene-histogram — not resize/normalize/mel in this crate.


