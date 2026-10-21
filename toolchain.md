# Toolchain

Project programs and direct packages from manifests.

## Programs

| Program | How to install | Why here | Source |
| --- | --- | --- | --- |
| mise | brew / curl, then `mise install` | Pinned tool versions | https://github.com/jdx/mise |
| rust | mise | Compiler and std | https://github.com/rust-lang/rust |
| rustc | mise (pin rust) | Rust compiler | https://github.com/rust-lang/rust |
| cargo | mise (pin rust) | Rust build and dependencies | https://github.com/rust-lang/cargo |
| moon | mise | Monorepo tasks | https://github.com/moonrepo/moon |
| ketch | see its README | Installs swarfr | https://github.com/pyrlyn/ketch |
| swarfr | ketch | `root:test-with-cleanup` compacts `target/` after a green pass | https://github.com/listepo/swarfr |
| ffmpeg | mise | Audio/video fixtures | https://github.com/FFmpeg/FFmpeg |
| python | mise | Scripts | https://github.com/python/cpython |
| node | mise | JS runtime | https://github.com/nodejs/node |
| cargo-dist | mise | Release artifacts | https://github.com/axodotdev/cargo-dist |
| cargo-cache | mise | Clean cargo home | https://github.com/matthiaskrgr/cargo-cache |
| zig | mise | Custom kernels / native | https://github.com/ziglang/zig |
| jactionlint | mise (`mise.toml` pins `github:jdx/jactionlint` 2.0.2) | Lint the workflows (the pre-push hook and `mise x -- jactionlint`; CI lints through pyrlyn/ci) | https://github.com/jdx/jactionlint |
| hk | mise (`mise.toml` pins 2.5.0) | Runs jactionlint as a pre-push hook (`hk.pkl`) | https://github.com/jdx/hk |
| shellcheck | mise (`mise.toml` pins 0.11.0) | Checks `run:` scripts inside jactionlint | https://github.com/koalaman/shellcheck |

## ketch

| Package | Where | Source | Why here |
| --- | --- | --- | --- |
| swarfr | global | https://github.com/listepo/swarfr | Lossless `target/` cleanup after tests |

## cargo

| Package | Where | Source | Why here |
| --- | --- | --- | --- |
| anyhow | local | https://crates.io/crates/anyhow | CLI errors |
| assert_cmd | local | https://crates.io/crates/assert_cmd | P2.3 e2e CLI tests. |
| assert_fs | local | https://crates.io/crates/assert_fs | e2e temp dirs with auto-cleanup (bench calibration DB). |
| async-openai | local | https://crates.io/crates/async-openai | Rust dependency |
| axum | local | https://crates.io/crates/axum | Rust dependency |
| base64 | local | https://crates.io/crates/base64 | Strict `data:` / `input_audio` decoding |
| chrono | local | https://crates.io/crates/chrono | Rust dependency |
| clap | local | https://crates.io/crates/clap | CLI |
| config-schema | local (dev) | https://github.com/pyrlyn/crates-packages | Stale-schema check for `[store]` (`runa-models`), pinned by git rev |
| clap_mangen | local | https://crates.io/crates/clap_mangen | Rust dependency |
| cc | local | https://crates.io/crates/cc | P9.3: compiles the vendored ggml RPC backend behind `--features rpc`. |
| criterion | local | https://crates.io/crates/criterion | Statistical benchmarks with HTML reports (kernels/media gates) |
| encoding_rs | local | https://crates.io/crates/encoding_rs | Rust dependency |
| ffmpeg | mise | https://ffmpeg.org/ | P4.5: binary on PATH or `FFMPEG_PATH`. runa does not download it. |
| futures | local | https://crates.io/crates/futures | Rust dependency |
| heed | local | https://crates.io/crates/heed | P2.8 prompt cache: LMDB via heed (mmap KV, fastest local store). |
| hf-hub | local | https://crates.io/crates/hf-hub | P2.4 HF downloads (blocking API: sync CLI, runtime on bg thread). |
| hound | local | https://crates.io/crates/hound | Rust dependency |
| insta | local | https://crates.io/crates/insta | Snapshot tests for stable library renderings (fit verdict report). |
| keychain-secret | local | https://github.com/pyrlyn/crates | P17.19 keychain read and inline-secret scan. Git rev pin; not on crates.io. |
| keyring | local | https://crates.io/crates/keyring | P3.8 OS keychain, transitive through `keychain-secret`. Env vars still win. |
| libc | local | https://crates.io/crates/libc | P10.2 macOS sysctl for the Apple P-core default (`hw.perflevel0.logicalcpu`, macOS-only dep of `runa-engine`). |
| llama-cpp-2 | local | https://crates.io/crates/llama-cpp-2 | Local GGUF / llama.cpp |
| llama-cpp-sys-2 | local | https://crates.io/crates/llama-cpp-sys-2 | Rust dependency |
| mistralrs | local | https://crates.io/crates/mistralrs | P9.2 safetensors backend (optional, `--features mistralrs`). |
| predicates | local | https://crates.io/crates/predicates | Composable output matchers for assert_cmd e2e tests. |
| pretty_assertions | local | https://crates.io/crates/pretty_assertions | Diff output for rich assert_eq (doctor JSON). |
| proptest | local | https://crates.io/crates/proptest | Rust dependency |
| raw-cpuid | local | https://crates.io/crates/raw-cpuid | Rust dependency |
| rayon | local | https://crates.io/crates/rayon | Rust dependency |
| reqwest | local | https://crates.io/crates/reqwest | HTTP |
| rstest | local | https://crates.io/crates/rstest | Parametrized case matrices (ggml block table). |
| rubato | local | https://crates.io/crates/rubato | Rust dependency |
| rmcp | local | https://crates.io/crates/rmcp | P8.3 MCP client (stdio servers) for the run/chat tool loop. |
| rustyline | local | https://crates.io/crates/rustyline | P2.3 REPL line editing + file history. |
| schemars | local (dev) | https://crates.io/crates/schemars | JSON Schema of `[store]` (`runa-models`) |
| serde | local | https://crates.io/crates/serde | Serialization |
| serde_json | local | https://crates.io/crates/serde_json | JSON |
| sha2 | local | https://crates.io/crates/sha2 | Rust dependency |
| symphonia | local | https://crates.io/crates/symphonia | P4.1 audio decode. |
| sysinfo | local | https://crates.io/crates/sysinfo | Rust dependency |
| tempfile | local | https://crates.io/crates/tempfile | Rust dependency |
| thiserror | local | https://crates.io/crates/thiserror | Errors |
| tokio | local | https://crates.io/crates/tokio | Async runtime |
| toml | local | https://crates.io/crates/toml | Config, `runa fit --recommend` catalog |
| trycmd | local | https://crates.io/crates/trycmd | Full CLI output fixtures (help screens). |
| ratatui | local | https://crates.io/crates/ratatui | P8.6 `chat --tui` transcript/status frames. |
| crossterm | local | https://crates.io/crates/crossterm | P8.6 TUI events + alternate screen. |
| tui-textarea-2 | local | https://crates.io/crates/tui-textarea-2 | P8.6 TUI multi-line input box. |
| unicode-width | local | https://crates.io/crates/unicode-width | P8.6 TUI transcript word wrap. |
| whisper-rs | local | https://crates.io/crates/whisper-rs | Rust dependency |
| wiremock | local | https://crates.io/crates/wiremock | Rust dependency |

## Linkers

- **Windows (MSVC):** `rust-lld` through `.cargo/config.toml` (`[target.*-pc-windows-msvc] linker`). It ships with rustup, nothing to install.
- **Linux:** `mold`, pinned in `mise.toml` for Linux only and driven by `clang`. Opt-in, off by default: `cargo --config .cargo/mold.toml <cmd>` (or `just mold <cmd>` where the repo has a justfile). CI and the default build keep the stock linker.
- **macOS:** Apple's native linker (mold has no macOS support).
- **Cranelift:** not used. It needs a nightly toolchain and this repository is on stable.
