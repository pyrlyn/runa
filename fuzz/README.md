# fuzz/ — cargo-fuzz (libFuzzer) targets

Coverage-guided fuzzing of runa's input parsers. This crate is **not** part
of the root workspace (`exclude = ["fuzz"]`), so `cargo build`, `clippy`,
tests and CI never compile it. It needs a nightly toolchain; the repo pin in
`rust-toolchain.toml` is left alone — always pass `+nightly`.

```sh
rustup toolchain install nightly
cargo install cargo-fuzz
cd fuzz
cargo +nightly fuzz list
```

## Targets

Default build (pure Rust, no llama.cpp; first build ~1 min):

- `gguf-header` — raw GGUF header bytes (what `runa fit` reads from disk or
  HTTP ranges) through `runa_fit::Reader::parse` → `Descriptor::from_reader`
  → `estimate_kv`, `estimate_compute`, `estimate_speed_single`, `check_fit`,
  `format_report`.
- `reasoning-stream` — model text in arbitrary chunks through
  `runa_core::ReasoningParser` (`push` / `flush` / `finish`), `parse_stream`
  and `ReasonFamily::from_template`: `<think>`, Harmony and Gemma tags.
  Asserts the parser never emits more bytes than it was given.
- `cloud-responses` — Anthropic SSE streams and message bodies
  (`parse_sse`, `parse_message`), OpenAI assistant fields
  (`split_assistant_fields`), `tools_from_openai`.
- `model-refs` — `parse_model_ref`, `pick_quant`, `is_safetensors_tag`,
  `parse_cloud_ref`, `BackendKind` / `NpuKind` / `Effort` / `--think` /
  budget parsers and `ThinkConfig::apply`.

With `--features cli` (compiles the real `runa` binary crate,
`crates/runa/src/main.rs`, with `--cfg fuzzing`; that pulls llama.cpp and
whisper.cpp, first build ~3 min):

- `cli-argv` — NUL-separated argv as raw bytes (non-UTF-8 included) through
  `Cli::try_parse_from` (clap, with help / error rendering), then the pure
  flag validators each command runs before loading anything: `--mode`,
  `--backend`, `--kv*`, `--device`, `--tensor-split`, `--rpc`, `--lora`,
  `--on-unfit`, `--think`, `--effort`, `--audio-route`, `--mcp` quoting,
  model and cloud refs.
- `cli-config` — `runa.toml` text through every reader in `config.rs`
  (`[defaults]`, `[system]`, `on_unfit`, `[think]`, `[memory]`, `[audio]`,
  `[mcp.servers]`, `[models]`, `lora`), inline-secret rejection, REPL / TUI
  `/slash` commands and `--mcp` word splitting.
- `serve-body` — `runa serve` HTTP bodies: OpenAI `/v1/chat/completions` and
  Anthropic `/v1/messages` → messages, tools / `tool_choice`,
  `response_format` (incl. llama.cpp's JSON Schema → GBNF converter),
  thinking knobs, base64.
- `daemon-proto` — one daemon socket line: `DaemonRequest` / `DaemonEvent`
  decode → `into_generate` → re-encode, `parse_stop`.

The `cli` targets pick their surface by binary name in
`crates/runa/src/fuzz_hooks.rs`; the only other tree changes are
`#[cfg(fuzzing)]` helpers in `config.rs` and `serve.rs`. Nothing loads a
model, opens a socket, spawns a process or writes files (`serve-body` skips
bodies with `image_url` / `input_audio` parts: those write temp files).

Not covered: tool-call extraction from model output
(`ToolReply::parse`) needs a `ChatTemplateResult` from a loaded model's
template (llama.cpp); media decoding (`runa-media`) and prompt caching touch
files.

## Run

```sh
cd fuzz
# pure-Rust targets
cargo +nightly fuzz run gguf-header corpus/gguf-header seeds/gguf-header -- -max_total_time=120

# CLI-crate targets (macOS: see the ASan note below)
export RUSTFLAGS="-Cllvm-args=-asan-globals=0"
cargo +nightly fuzz run --features cli --target-dir target/cli \
  cli-argv corpus/cli-argv seeds/cli-argv -- -max_total_time=120
```

The first corpus dir (`corpus/<target>`, git-ignored) receives new inputs;
`seeds/<target>` holds small committed seeds (regenerate with
`python3 seeds/make_seeds.py`). Useful libFuzzer flags: `-rss_limit_mb=4096`,
`-timeout=10`, and `-fork=4 -ignore_crashes=1 -ignore_ooms=1` to keep going
past known crashes.

- Crashes land in `artifacts/<target>/`. Reproduce with
  `cargo +nightly fuzz run <target> artifacts/<target>/crash-…`, shrink with
  `cargo +nightly fuzz tmin <target> <file>`.
- Known reproducers are committed under `regressions/<target>/`; replay all
  of them with `cargo +nightly fuzz run <target> regressions/<target>/* -- -runs=0`
  (a fixed bug exits 0).
- `--target-dir target/cli` keeps the heavy `cli` build apart from the
  default one, so switching between them does not rebuild everything.

ASan on macOS: `ctor` (via `hf-hub` → `xet-runtime`) fails to link under
ASan's global instrumentation (`ld: initializer pointer has no target`);
`-Cllvm-args=-asan-globals=0` keeps heap/stack checks and fixes the link.
Alternatively build with `--sanitizer none`. Linux does not need it.

## CI

No CI job runs these (no blocking gate). A future optional job can run each
target for a fixed time on `workflow_dispatch` / schedule with
`-max_total_time`.
