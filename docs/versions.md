# docs/versions.md — version pins

Plan D16: the Rust toolchain, `llama-cpp-2`, `whisper-rs` and `async-openai`
are pinned. `Cargo.lock` enforces the crate pins (once the workspace exists,
P0.1); this file records **what each pin maps to upstream**, why, and how it
was verified. Researched 2026-09-08 via the crates.io API, docs.rs source
tarballs (`.cargo_vcs_info.json`), upstream submodule pins and tag lists.

## Pins

| Component | Pin | Maps to upstream | Published | Notes |
|-----------|-----|------------------|-----------|-------|
| Rust | `1.98` (`rust-toolchain.toml`, `mise.toml`) | rustc 1.98.1 (built 2026-08-05) at the time of writing | — | channel `1.98` tracks the latest 1.98.x patch. `mise.toml` also pins `components = "rustfmt,clippy"` (mirror of rust-toolchain.toml: mise ignores that file under RUSTUP_TOOLCHAIN, and without this CI has no cargo-fmt/clippy) |
| llama-cpp-2 | `=0.1.133` | llama-cpp-sys-2 0.1.133 → **llama.cpp `b7709`** (commit `1051ecd`, 2026-01-12) | 2026-02-03 | exact pin — the crate does not follow semver |
| whisper-rs | `=0.16.0` | whisper-rs-sys 0.15.0 → **whisper.cpp `v1.8.3`** (commit `2eeeba5`, 2026-01-15) | 2026-03-12 | latest whisper-rs release |
| async-openai | `=0.42.0` | — | 2026-09-15 | Responses API behind the `responses` feature; MSRV 1.75 |
| mistral.rs | `=0.8.1` | mistralrs 0.8.1 → mistralrs-core 0.8.1 (see below) | 2026-04-02 | optional second backend, `--features mistralrs` (D2, P9.2). Exact pin like the other engine crates; `default-features = false` keeps GPU accel opt-in (D13) |
| moon | `2.5.4` (`mise.toml` `aqua:moonrepo/moon`, `.moon/workspace.yml` `versionConstraint`) | moonrepo/moon `v2.5.4` (2026-09-03) | 2026-09-08 | monorepo task graph over the cargo workspace (D22, K1) |
| reqwest | `=0.13.5` (default-features off; `blocking` + `rustls`) | — | 2026-09-15 | P1.2 remote header fetch; future Anthropic adapter client (D9). Pure-Rust TLS, no system libs on any CI target |
| serde | `=1.0.229` (`derive`) | — | 2026-09-08 | P1.10 calibration DB persistence |
| serde_json | `=1.0.151` (no-derive `Value` walk) | — | 2026-09-08 | P1.2 Hub API sibling listing |
| rustyline | `=18.0.1` | — | 2026-09-08 | P2.3 chat REPL (line editing + file history) |
| assert_cmd | `=2.2.2` | — | 2026-09-08 | P2.3 CLI e2e tests |
| ffmpeg | `9.0.1` (`mise.toml`) | ffmpeg `9.0.1` | 2026-09-15 | P4.5 video via ffmpeg-sidecar (binary, not linked) |
| python | `3.13.15` (`mise.toml`) | CPython `3.13.15` (newest 3.13.x) | 2026-09-15 | P0.7 fixtures, P3.9/P6.2 SDK smoke tests (stdlib only; 3.11.9 lacked attestations) |
| node | `24.21.0` (`mise.toml`) | Node `24.21.0` LTS | 2026-09-15 | P3.9/P6.2 SDK smoke tests (20.x EOL 2026-04) |
| cargo-dist | `0.33.0` (`mise.toml` `aqua:axodotdev/cargo-dist`) | cargo-dist `0.33.0` | 2026-09-15 | P6.3 packaging. Prebuilt binary via aqua since 2026-09-14 (was `cargo:` backend, which compiled from source and raced rustup downloads in CI) |
| cargo-cache | `0.8.3` (`mise.toml` `cargo:cargo-cache`) | cargo-cache `0.8.3` | 2023-09-01 | developer utility: `moon run root:cache` / `cache-dry-run` / `cache-autoclean`; not a crate dependency |
| zig | `0.16.0` (`mise.toml`) | zig `0.16.0` | 2026-09-15 | D23 own kernels (`runa-kernels`) |
| keyring | `4.2.0` | — | 2026-09-15 | P3.8 OS keychain (`service = runa`) |
| symphonia | `=0.6.1` (mp3/aac/flac/ogg/pcm/wav/isomp4) | — | 2026-09-15 | P4.1 audio decode |
| hound | `=3.5.1` | — | 2026-09-08 | P4.1 WAV read/write |
| rubato | `=5.0.0` | — | 2026-09-15 | P4.1 resample to 16 kHz |
| git-cliff | `2.13.1` (`mise.toml`) | git-cliff `2.13.1` | 2026-03-01 | P13.3 `CHANGELOG.md` generation from commit subjects (`cliff.toml`); the same pin rtok uses, so a release commit's notes are reproducible |

Use the `=` exact-pin operator for the three engine/API crates in
`Cargo.toml`; `llama-cpp-2` explicitly does not follow semver, and `whisper-rs`
reserves the right to break in patch releases when `raw-api` is involved.

## llama-cpp-2 → llama.cpp mapping (verified chain)

1. `llama-cpp-2 0.1.133` was published 2026-02-03 from
   `utilityai/llama-cpp-rs` tag `0.1.133` = commit `349ae33`
   (crate `.cargo_vcs_info.json`).
2. The repo tree at that commit pins the submodule
   `llama-cpp-sys-2/llama.cpp` at `1051ecd28907d2ca0a15c135f190fe415d0a3d1b`.
3. That commit is exactly llama.cpp release tag **`b7709`** (2026-01-12,
   "vulkan: Disable large coopmat matmul configuration on proprietary AMD
   driver", #18763). Cross-check: tag `b8400` is 691 commits ahead of it and
   8400 − 7709 = 691 — llama.cpp tag numbers count commits, so the mapping is
   exact.
4. `llama-cpp-2 0.1.133` depends on `llama-cpp-sys-2 ^0.1.133` (features
   forwarded); on macOS arm64 the crate auto-enables `metal` through the sys
   crate.

Feature flags available on 0.1.133: `cuda`, `cuda-no-vmm`, `metal`,
`vulkan`, `openmp`, `mtmd`, `sampler`, `dynamic-link`, `system-ggml`,
`android-shared-stdcxx`.

> **Deviation from plan D2:** the `native` feature (listed in D2, used by
> P5.7) **does not exist in llama-cpp-2 0.1.133**. It appears in later
> llama-cpp-2 releases (present by 0.1.156). Portable release builds must
> rely on ggml's runtime dispatch (D13) until we upgrade past this pin;
> `runa-kernels` dispatch (P5.2) is unaffected.

### NPU pin survey (P9.4, 2026-09-15)

Question: does any newer `llama-cpp-2` expose `hexagon`/`openvino` cargo
features for the P9.4 NPU slice?

- Checked `llama-cpp-2` **0.1.154** (crate tarball
  `llama-cpp-2-0.1.154.crate`, git `bed81ad4`, plus the docs.rs features
  page): full feature list is `android-shared-stdcxx`,
  `android-static-stdcxx`, `common`, `cuda`, `cuda-no-vmm`, `default`,
  `dynamic-backends`, `dynamic-link`, `llguidance`, `metal`, `mkl`,
  `mtmd`, `opencl`, `openmp`, `rocm`, `sampler`, `static-openmp`,
  `static-stdcxx`, `system-ggml`, `system-ggml-static`, `vulkan`.
  **No `hexagon`, no `openvino`.** (0.1.154 adds `opencl`, `rocm`, `mkl`,
  `llguidance`, `dynamic-backends` over our 0.1.133 pin; none is an NPU
  path.) The matching docs.rs page for `llama-cpp-sys-2` 0.1.154 shows the
  same set.
- Locally verified that referencing a nonexistent dep-feature
  (`hexagon = ["llama-cpp-2/hexagon"]`) breaks **even the default**
  resolution (`cargo build` fails at resolve time), so the P9.4 features
  cannot forward to llama-cpp-2 until upstream lands them.

Decision: **wait** (keep the `=0.1.133` pin; P9.4 ships probe +
build-gating + docs only):

- `system-ggml` / `system-ggml-static` would fork the build off the pinned
  llama.cpp (b7709) onto whatever ggml the host provides — breaks the D16
  pin discipline and build reproducibility for a Tier-3 path.
- `dynamic-backends` only loads ggml backends built into that same pinned
  llama.cpp at compile time; it cannot conjure Hexagon/OpenVINO backends
  the pin was built without.
- The D2 `bindgen` fallback (our own bindings over `llama.h`) is
  disproportionate for Tier-3/manual hardware with no CI runners.

Unblocks for real NPU offload, in order: (1) upstream `llama-cpp-2`
`hexagon`/`openvino` features (or a ggml release with those backends worth
a deliberate pin upgrade through the D1/D15 gates); (2) vendor SDKs
(`HEXAGON_SDK_ROOT`, `INTEL_OPENVINO_DIR`); (3) manual on-device
validation + calibration of the `HwSpec::{hexagon,openvino}` stubs and the
`npu_present()` markers. Until then the `hexagon`/`openvino` cargo
features are probe-only stubs: enabling them changes no placement (tensors
stay on CPU — plan D12).

## Build flags (P5.7)

| Build | Flags | Notes |
|-------|-------|-------|
| Local dev (macOS arm64) | `cargo build -p runa-engine --features metal` | Metal enabled by default on Apple Silicon via `llama-cpp-sys-2` |
| Local dev (Linux CUDA) | `--features cuda` on `runa-engine` / workspace | CI ubuntu job is CPU-only; CUDA is build-only elsewhere |
| Portable release | no `-march=native`; ggml runtime dispatch (D13) | `native` on `llama-cpp-2` arrives after the 0.1.133 pin — use a newer pin before enabling |
| `runa-kernels` (P5.2+) | Zig (C ABI) preferred for new kernels (D23); C/`.S` only where Zig is worse | SIMD via Zig `@Vector` or C runtime dispatch |

Release binaries must run on hosts **without** AVX-512; do not ship `-C target-cpu=native` in release profiles (`Cargo.toml` `[profile.release]` has no `target-cpu`).

Release highlights of 0.1.133 relevant to runa: mtmd bindings
(audio/image/video through `LlamaContext`/mtmd context), function calling +
OpenAI-style conversion helpers, sampler chain. Note that llama.cpp removed
its own OpenAI-compat server upstream later (llama-cpp-2 0.1.147 sync) —
irrelevant to us: `runa serve` is our own axum server (D11).

### RPC backend: vendored, feature-gated (P9.3, 2026-09-15)

llama.cpp b7709 supports distributed inference via the RPC backend
(`--rpc host:port`, `ggml_backend_rpc_add_server`), and no published
`llama-cpp-2` (surveyed 0.1.131–0.1.156 via the sparse index — none exposes
an `rpc` feature) can build it, but no fork or pin bump was needed. The
published `llama-cpp-sys-2 0.1.133` crate ships every header the backend
needs (`ggml-rpc.h`, plus the internal `ggml-impl.h`,
`ggml-backend-impl.h`, `ggml-common.h`); only the 78 KB `ggml-rpc/`
implementation directory is stripped, so `-DGGML_RPC=ON` fails at CMake
configure time (`add_subdirectory(ggml-rpc)` → missing directory) and
`wrapper.h` never parses `ggml-rpc.h` (no `ggml_backend_rpc_*` bindings).
`GGML_USE_RPC` in ggml core only auto-registers an *empty* base reg — the
real flow (`ggml_backend_rpc_add_server` → `ggml_backend_register`) goes
through the public C API, so `runa-engine` compiles the verbatim b7709
`ggml-rpc.cpp` (vendored under `crates/runa-engine/rpc/` with byte-identical
headers, see `rpc/README.md`) in `build.rs` behind a new `rpc` cargo feature
(C++17, `ws2_32` on Windows) and registers each `--rpc` endpoint in `load()`
before backend init — mirroring upstream `common/arg.cpp:add_rpc_devices`.
A `#[link]`-style shim was unnecessary: the symbols come from our own
compiled object, ABI-matched to the same b7709 sources the sys crate builds.
A `build.rs` pin-guard fails the `rpc` build if `llama-cpp-sys-2` drifts
from `=0.1.133` (D16; refresh rule in `rpc/README.md`).

Usage: `runa run --rpc 192.168.1.5:50052 --device RPC0 model.gguf`
(`--rpc` registers only; remote devices enumerate as `RPC0`, `RPC1`, …
and are selected explicitly with `--device` — never implicitly, D12).
`runa doctor` reports `rpc: true/false`; `runa fit --rpc` estimates the
local placement with an explicit note. Loopback-tested against a fake TCP
server answering `HELLO` 3.6.0 + `DEVICE_COUNT` (`runa-engine/tests/rpc.rs`).

### Engine upgrade survey: `llama-cpp-2` 0.1.156 (2026-09-15, verdict: WAIT)

Bumping `=0.1.133` → `=0.1.156` was attempted and **reverted**. Five compile
errors, all in the structured-output / tool-calling path (P8.1/P8.2):
upstream removed the entire oaicompat layer those features are built on
(`apply_chat_template_oaicompat`, `OpenAIChatTemplateParams`,
`ChatTemplateResult`, `GrammarTrigger*`, `wrapper_oai.*`). Its replacement
(`common/chat.{h,cpp}` + peg/auto parsers, built by default via the new
`common` cargo feature) is a C++-typed API (`std::string`, `std::vector`,
`nlohmann::ordered_json`) with no Rust bindings — restoring P8.1/P8.2 on it
means a C shim plus a full re-validation of grammars, lazy/eager triggers
and both SDK tool round trips. Per plan D16 (deliberate engine upgrades
through the benchmark gate) the pin stays at `=0.1.133` until that port is
scheduled as its own task; the RPC vendoring above (protocol 3.6.0) stays
matched to b7709. Note a future bump also moves the RPC protocol to 5.0.0
(submodule `e79e4bf`) — refresh `rpc/` together with it.

## whisper-rs → whisper.cpp mapping (verified chain)

1. `whisper-rs 0.16.0` (latest, 2026-03-12, codeberg `tazz4843/whisper-rs`,
   commit `7558e1b7`) requires `whisper-rs-sys ^0.15`; the only 0.15.x is
   0.15.0, published the same day from the same commit.
2. The sys crate ships whisper.cpp as the submodule `sys/whisper.cpp`,
   pinned at `2eeeba56e9edd762b4b38467bab96c2517163158`.
3. That commit is whisper.cpp **`v1.8.3`** — the literal
   `release : v1.8.3` bump commit (2026-01-15, CMake 1.8.2 → 1.8.3).

Feature flags on 0.16.0: `cuda`, `hipblas`, `metal`, `vulkan`, `openblas`,
`raw-api`, `log_backend`, `tracing_backend`.

Upstream whisper.cpp has moved on (v1.8.4 … v1.8.7, then v1.9.x; latest
surveyed: **v1.9.3**, 2026-09-08). Our ASR pin is therefore ~2 minor lines
behind; upgrade through the M7 benchmark gate (1 min of speech < 5 s on CPU)
when whisper-rs tracks a newer whisper.cpp.

## mistral.rs

0.8.1 (2026-04-02), MIT, MSRV 1.88. The `mistralrs` facade re-exports
`mistralrs-core 0.8.x` builders and types; runa uses the high-level
`ModelBuilder` (auto-detecting, local-directory capable) plus
`blocking::BlockingModel` / `BlockingStream` (own tokio runtime, sync token
iterator — runa-engine stays runtime-free). No `default` feature exists on
0.8.1, so `default-features = false` is a no-op pin for future-proofing;
GPU accel (`metal`, `cuda`, …) stays opt-in per D13 and is not yet forwarded
(the backend auto-maps devices). Upgrade with the 0.9.x line if/when it
lands; note the plan/roadmap text that anticipated `0.9.3` was written
before that version existed upstream (latest on crates.io at the time of
writing is 0.8.1).

## moon

2.5.4 (2026-09-03), installed via mise (`aqua:moonrepo/moon` — there is no
core mise plugin for moon, the aqua backend serves the official release
asset `moon_cli-<arch>.tar.xz`). `.moon/workspace.yml` enforces
`versionConstraint >= 2.5.4`. Verified 2026-09-08 with `mise install` +
`moon projects` (9 projects: 8 crates + root). The rust toolchain plugin
(1.0.9, bundled WASM) does **not** download Rust toolchains
(`download_prebuilt` unimplemented) — installs come from mise (D21); moon
records the pin (`version: '1.98'`) for graph/hashing/caching, but BOTH sync
flags stay **off**: either flag rewrites the pins as semver requirements
(`~1.98` — invalid in Cargo's `rust-version` and rustup's `channel`) and
strips file comments, breaking every cargo command (observed 2026-09-08).
Upgrade
through the M13 parity check (`moon run :test` == `cargo test
--workspace`).

## async-openai

0.42.0 (2026-09-15), MIT. Enable `responses` for the Responses API
with streaming (P3.5); `byot` (bring your own types) plus `base_url` override
covers OpenAI-compatible providers (OpenRouter, DeepSeek, Groq, llama-server).
Rate-limit retries with exponential backoff are built in. The Anthropic
adapter is our own `reqwest` + SSE client (D9) — no SDK pin involved.

## Not yet pinned — pin on first use (in Cargo.lock + a row here)

| Crate | Plan task |
|-------|-----------|
| `sherpa-onnx` (Parakeet; official Rust binding — **not** `sherpa-rs`, archived 2026-06). Empty `parakeet` cargo feature removed; `transcribe_parakeet` returns "not available yet" until a later task wires the crate. | P4.2 |
| `ffmpeg-sidecar` | P4.5 |
| `hf-hub` | P2.4 |
| `axum` | P3.9 |
| `figment`, `clap` | P0.1/P2.3 |
| `sysinfo`, `raw-cpuid`, `objc2-metal`, `ash`, NVML bindings | P1.6 |
| `rusqlite` (bundled) | P1.10 |
| `criterion` | D15 |


## Native vs portable (P5.7)

Default **release** builds are portable: `llama-cpp-sys-2` sets `GGML_NATIVE=OFF`
and ggml runtime-dispatches AVX2 / AVX-512 / NEON / SME. That binary runs on a
machine without AVX-512.

Host-tuned local builds:

```sh
RUSTFLAGS='-C target-cpu=native' cargo build --release --features native
```

`--features native` alone does **not** pass `-march=native` into llama.cpp
(the crate has no `native` cargo feature). `llama-cpp-sys-2` only turns
`GGML_NATIVE=ON` when it sees `-C target-cpu=native` in `RUSTFLAGS`.
`runa-kernels` follows the same rule: Zig builds with `-mcpu=baseline`, and
with `-mcpu=native` only under `-C target-cpu=native`. `whisper-rs-sys` builds
its own ggml, whose CMake default is `GGML_NATIVE=ON`; `.cargo/config.toml`
sets `GGML_NATIVE=OFF` (an exported `GGML_NATIVE=ON` still wins). Otherwise a
library built on an AVX-512 host (a cached CI build, a release runner) raises
SIGILL on an older CPU.
`runa doctor --json` reports `native_build` (true iff the binary was compiled
with `--features native`) and `backends` (`cpu` plus any of `metal` / `cuda` /
`vulkan` / `mtmd` compiled in). Tag `vX.Y.Z` runs `.github/workflows/release.yml`
(cargo-dist 0.33 CPU archives + shell/powershell/homebrew installers). GPU
variants: `.github/workflows/release-variants.yml`. Homebrew formula is on the
GitHub Release; `brew install pyrlyn/tap/runa` needs the formula in the shared
`pyrlyn/homebrew-tap` repo, which does not carry it yet. Local: `bash scripts/cargo-dist.sh generate --mode=ci --check`.

### ketch package (`ketch.toml`)

The root `ketch.toml` is the [ketch](https://github.com/pyrlyn/ketch) manifest
for this repo — the file `ketch registry push` offers to
`pyrlyn/ketch-registry` as `runa/ketch.toml`, and what `ketch info runa` shows
once it is there. Inference from `github:pyrlyn/runa` already works without
it; the manifest is what pins the asset choice.

| Field | Value |
|-------|-------|
| `source` | `github:pyrlyn/runa` |
| `bin` | `runa` (the root of the unwrapped cargo-dist archive) |
| `[asset] include` | `*.tar.xz`, `*windows-msvc.zip` |
| `[asset] exclude` | the GPU variants (`-metal`, `-vulkan`, `-cuda`), the shell / powershell / homebrew installers, `dist-manifest.json`, `*.sha256`, the source tarball |

The GPU archives matter here: `release-variants.yml` uploads them under the
same target triples as the portable CPU archives (`runa-aarch64-apple-darwin`
vs `runa-aarch64-apple-darwin-metal`), so without the `exclude` list asset
scoring could reach a variant that needs the GPU features compiled in. No
`trust` block: the release publishes no signature sidecars, and a `trust`
policy would fail every install.

Check a change before pushing it (a registry tree is `<package>/ketch.toml`
folders):

```sh
TMP=$(mktemp -d); mkdir -p "$TMP/runa"; cp ketch.toml "$TMP/runa/"
ketch registry validate "$TMP"    # validated 1 package
```

### Release flow

A release is a version commit made by `scripts/release.sh`, landed on `main`
through a pull request that `bump.yml` merges by rebase once the required
checks are green; bump then tags the landed commit and dispatches the
dist-generated `release.yml` (details, modes and the not-wired list:
[`release.md`](release.md)).

## Upgrade policy (D16)

- llama.cpp moves fast: ~8–12 tags/day, and **tag numbers count commits**
  (b7709 → b8400 = 691 commits). llama-cpp-2 releases ~weekly and does not
  follow semver — always pin exact and upgrade deliberately, at most monthly.
- Every upgrade passes the benchmark gate (D1: kernels must beat the ggml
  path by ≥ 5 % end-to-end or ≥ 2× isolated; D15: perf CI fails on > 3 %
  regression) and updates `Cargo.lock` and this file in the same change.
- Latest surveyed 2026-09-08: llama-cpp-2 **0.1.156** (llama.cpp ≈ b10405+),
  whisper-rs 0.16.0 (unchanged, but whisper.cpp upstream at v1.9.3),
  async-openai 0.42.0.

## Verification

```sh
cargo --version   # 1.98.x — rust-toolchain.toml; a RUSTUP_TOOLCHAIN env var
                  # overrides the file locally, unset it to test the pin
mise ls            # rust 1.98 (mise.toml)
mise install && moon --version   # moon 2.5.4 (K1); moon projects → 9 projects
```

Files that carry the pins: `rust-toolchain.toml`, `mise.toml`, this file,
and `Cargo.lock` once the workspace lands (P0.1).
