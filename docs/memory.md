# docs/memory.md — public methods: `MemoryManager` and `TaskRegistry`

Crate: `runa-memory` (plan D17/D18, phase P7, metrics M11/M12).
Targets: M11 (idle RSS ≤ floor + 10 %), M12 (no double-held task).
Config: `[memory] idle_timeout_s, floor_mib, max_growth_mib`;
`[agents] registry = "docs/tasks.md"`. Sketches live in `plan.md` §5.

Conventions: `demand_mib`/`rss_mib` are MiB (`u64`). Timestamps are UTC
RFC 3339 strings. Errors are documented per method; all methods are
synchronous and thread-safe (`Send + Sync`).

---

## `MemoryManager`

Adaptive process memory. Idle (no request/job for `idle_timeout_s`) →
release prompt cache, encoder buffers, draft model, shrink pools toward
`floor_mib`. Heavy task → pre-grow arenas up to fit verdict + margin and
at most `max_growth_mib` beyond current use. Never unloads the active
model. Every transition logs before/after RSS.

### `fn current_usage(&self) -> Usage`

- Returns: `Usage { rss_mib, budget_mib, state }` where `state` is
  `Idle` | `Normal` | `Heavy` and `budget_mib` is the current ceiling
  (fit verdict + margin + granted growth).
- Example: `let u = mm.current_usage(); assert!(u.rss_mib <= u.budget_mib);`
- Notes: pure observation, no side effects; `state` derives from recent
  load vs `idle_timeout_s` and pending demand.

### `fn touch(&self)`

- Effect: record that a request/job is in flight. Resets the idle timer.
- Call when: every generate / server request starts (`grow_for` already
  does this).
- Notes: does not change `LoadState`.

### `fn maybe_idle(&self)`

- Effect: if `last_activity` is older than `idle_timeout_s`, calls
  `on_idle`. No-op when work happened recently (hysteresis).
- Call when: poll after a request finishes, or on a server idle tick.
- Check: `idle_timeout_s = 0` shrinks immediately; a 3600 s timeout does
  not shrink right after `touch`.

### `fn on_idle(&self)`

- Effect: shrink toward `floor_mib` (release caches/buffers/pools as
  above); no-op if already at/below floor. Logs before/after RSS.
- Call when: no request or job for a full `idle_timeout_s` (or from
  `maybe_idle`).
- Check: M11 soak test — idle RSS ≤ floor + 10 %.
- Notes: hysteresis — shrink only after full silence, never mid-burst
  (see Risks in `plan.md` §8).

### `fn on_heavy(&self, demand_mib: u64)`

- Params: `demand_mib` — expected extra memory for the incoming heavy
  task (ctx growth, batch, media).
- Effect: convenience wrapper — grants what fits via `grow_for` and
  otherwise keeps current placement (caller falls back to smaller
  ctx/quant or cloud per the fit suggestion). Never exceeds ceiling.
- Example: `mm.on_heavy(2048); // make room for a 2 GiB ctx bump`

### `fn shrink_to_floor(&self)`

- Effect: unconditional shrink to `floor_mib`, same release set as
  `on_idle` but immediate (used by tests and explicit
  `runa doctor --shrink`-style maintenance paths).
- Notes: still never unloads the active model.

### `fn grow_for(&self, demand_mib: u64) -> Result<(), MemoryError>`

- Params: `demand_mib` — bytes (MiB) the task needs beyond current use.
- Returns: `Ok(())` after pre-growing; `Err(MemoryError::OverCeiling {
  demand_mib, ceiling_mib, suggestion })` when demand exceeds
  fit verdict + margin (`suggestion`: smaller ctx, other quant,
  `--kv q8_0`, or cloud — same vocabulary as `runa fit`).
- Check: heavy-ctx test passes without mid-run OOM; over-ceiling test
  asserts the `Err` carries a suggestion.

---

## `TaskRegistry`

Cooperative claims over `docs/tasks.md` (file-backed). Row states:
`free` | `in progress` (+ `agent`, `started_at`). Completion is tracked
by checking the box in `plan.md`; the registry only tracks live claims.

### `fn list_free(&self) -> Result<Vec<String>, ClaimError>`

- Returns: IDs of all `free` tasks (e.g. `["P0.1", "P1.4", …]`), sorted
  by phase order.
- Errors: `Io` when the registry file cannot be read. A missing file is
  not an empty list.
- Example: `for id in reg.list_free()? { println!("{id}"); }`

### `fn status(&self, task_id: &str) -> Result<Option<TaskStatus>, ClaimError>`

- Params: `task_id` — e.g. `"P2.6"`.
- Returns: `Ok(None)` for unknown IDs; `Ok(Some(Free))` or
  `Ok(Some(InProgress { agent, started_at }))`.
- Errors: `Io` when the registry file cannot be read.
- Notes: read-only; use before asking about an `in-progress` task.

### `fn claim(&self, task_id: &str, agent: &str) -> Result<TaskClaim, ClaimError>`

- Params: `task_id`, `agent` (claiming agent's name; `started_at` is
  stamped by the registry at claim time).
- Returns: `Ok(TaskClaim { task_id, agent, started_at })` and the row
  becomes `in progress`.
- Errors: `NotFound` (unknown ID); `AlreadyClaimed { agent, started_at }`
  when `in progress` — the caller must follow the ask-flow in
  `AGENTS.md` §3–4 and retry only on explicit approval; `Io` when the
  file cannot be read or written; `InvalidAgent` when `agent` is empty
  or contains `|` or a control character. Claims take an OS file lock
  and replace the file by rename, so two processes cannot both win.
- Example: `reg.claim("P1.4", "fable")?; // do work …; reg.release("P1.4", "fable")?;`
- Check: double-claim test — second `claim` fails with the owner's
  name + time, first holder unaffected.

### `fn release(&self, task_id: &str, agent: &str) -> Result<(), ClaimError>`

- Effect: clears the row to `free` (agent/started emptied). Call on
  **every** stop or done, including failures and interrupts.
- Errors: `NotFound` (unknown ID); `NotOwner { agent }` when another
  agent holds it — ask, don't force; `Io` and `InvalidAgent` as on
  `claim`.
- Check: release test — row returns to `free` and re-claimable.

---

## Registry file format (`docs/tasks.md`)

Markdown table: `| Task | Status | Agent | Started (UTC) |`.
`Status` ∈ `free` | `in progress`. `in progress` rows MUST carry agent +
RFC 3339 `started_at`; `free` rows MUST have both empty. Lint (P7.6)
enforces this plus no double-held task and flags claims older than 7 days.

---

## P9.3 — RPC placement (distributed inference, blocked)

Crate: `runa-engine` (`placement.rs`, `load.rs`, `prompt_cache.rs`).
The pinned `llama-cpp-sys-2 0.1.133` strips the ggml RPC backend, so
full distributed inference is blocked (spike verdict + unblock path in
`docs/versions.md`, "`GGML_RPC` unavailable on this pin"). Until a
sys-crate fork (or a pin bump restoring the sources) lands, the API
carries the intent and fails loudly — never silently runs locally when
distribution was requested.

### `fn parse_rpc_list(s: &str) -> Result<Vec<String>, String>`

- Params: `s` — comma-separated `host:port` endpoints of `rpc-server`
  instances, e.g. `"127.0.0.1:50052,10.0.0.2:50052"`.
- Returns: trimmed non-empty entries; `Err("--rpc: empty list")` when
  nothing remains. Shape-light like `parse_device_list` — hostnames,
  IPv4 and bracketed IPv6 literals all pass through.
- Example: `parse_rpc_list("node1:50052")? // ["node1:50052"]`
- Check: `parse_rpc_list_csv` unit test (`placement.rs`).

### `Placement::rpc_servers: Vec<String>`

- Field on `Placement` (default empty = local inference on every
  constructor: `cpu`, `gpu`, `hybrid_moe`).
- Notes: `prefix_key` hashes the list, so an RPC-enabled build later
  never shares prompt-cache state with local runs.

### `fn with_rpc_servers(self, rpc_servers: Vec<String>) -> Placement`

- Effect: builder recording llama.cpp RPC endpoints on the placement.
- Example: `Placement::gpu().with_rpc_servers(parse_rpc_list(s)?)`
- Notes: `load` prints the endpoints in the verdict line (`rpc=…`
  suffix) and then rejects a non-empty list with
  `EngineError::Unsupported("--rpc …")` before backend init — no
  global state is touched.
- Check: `with_rpc_servers_preserves_mode`,
  `verdict_line_lists_rpc_servers` (`load.rs`), and the integration
  test `rpc_servers_fail_unsupported_before_backend_init`
  (`crates/runa-engine/tests/load.rs`).

## P9.4 — NPU probe and speed stubs (Tier 3)

Crate: `runa-fit` (`npu`, `speed`); CLI surface in the `runa` binary
(`RUNA_NPU`, `runa auto` verdict suffix, `runa doctor` stub strings).
Tier-3 scaffolding only: no ggml NPU backend exists in `llama-cpp-2`
through 0.1.154, so nothing here moves tensors (see `docs/versions.md`
for the pin survey).

### `runa_fit::npu::NpuKind`

- Variants: `Hexagon` (Qualcomm Hexagon DSP/NPU), `OpenVino` (Intel NPU
  via OpenVINO).
- `fn as_str(self) -> &'static str` — canonical name
  (`hexagon` / `openvino`).
- `fn parse(s: &str) -> Option<NpuKind>` — case-insensitive names
  (`1`/`hexagon`/`qcom`/`snapdragon`, `openvino`/`ov`/`intel-npu`);
  `None` for anything else.
- `fn hw_spec(self) -> HwSpec` — the conservative stub for the family
  (`HwSpec::hexagon` / `HwSpec::openvino`).
- `Display` prints `as_str`.

### `fn runa_fit::npu_present() -> Option<NpuKind>`

- Returns: `Some(kind)` when an NPU is present (real or faked), `None`
  otherwise. Pure hardware path is Linux-only; other OSes report absent
  until an on-device owner validates a marker there.
- Test hook `RUNA_FAKE_NPU`: `1`/`hexagon` → `Hexagon`,
  `openvino` → `OpenVino`, `0`/`no`/`off`/`none` → force absent; unset
  (or unrecognized) → hardware heuristic.
- Heuristic: `/proc/device-tree/compatible` contains `qcom` → Hexagon;
  else `/dev/accel` exists → OpenVINO. Conservative and unvalidated
  on-device (Tier 3).

### `fn runa_fit::probe_markers(device_tree_compatible: &Path, accel_dir: &Path) -> Option<NpuKind>`

- Effect: the `npu_present` heuristic with injectable marker paths, so
  unit tests never touch the real filesystem or environment.
- Check: `qcom` in the compat file wins over `/dev/accel`; no markers →
  `None`.

### `HwSpec::hexagon() / HwSpec::openvino()`

- Returns: conservative uncalibrated stubs — Hexagon: 60 GB/s, 45 TOPS,
  efficiency 0.30; OpenVINO: 40 GB/s, 13 TOPS (floor SKU), efficiency
  0.30. Both under-predict CUDA on the same model by construction.
- Limits (see `docs/fit.md`): Q4_0-centric, text-only OpenVINO, Hexagon
  ~3.5 GiB DSP window split. Recalibrate from on-device `runa bench`
  before quoting NPU speeds.

### Binary surface (`runa`)

- `RUNA_NPU=hexagon|openvino` opts the `runa auto` verdict line into an
  NPU suffix: `NPU <kind> present (opt-in stub): ~X tok/s decode
  (uncalibrated …); placement stays CPU` on a match, or an explicit
  `requested but …; staying on CPU (explicit, no silent fallback)` when
  the probe disagrees. Without `RUNA_NPU` the verdict never mentions NPU;
  an unrecognized value is an explicit error.
- `runa doctor` lists `hexagon-stub` / `openvino-stub` only in binaries
  built with `--features hexagon` / `openvino`; default binaries list
  neither. The `runa-engine` build script warns when a stub feature is on
  (always: stub status; plus SDK-missing: `HEXAGON_SDK_ROOT` /
  `INTEL_OPENVINO_DIR`).

## P9.2

mistral.rs backend for safetensors / omni models ggml cannot run.
`--backend gguf|mistral|auto` on `run` / `chat` / `serve` (default `auto`).
Opt-in cargo feature `mistralrs` (`runa-engine`, forwarded by `runa`);
`runa doctor` reports `mistralrs` when compiled in.

### `runa_core::BackendKind`

- Variants: `Auto` (default) | `Gguf` | `Mistral`.
- `fn parse(s: &str) -> Option<BackendKind>` — `auto` | `gguf` | `mistral`,
  case-insensitive.
- `fn as_str(self) -> &'static str` (+ `Display`).
- `fn detect_backend(path: &Path) -> Result<BackendKind, String>` — `.gguf`
  file → `Gguf`; directory holding `config.json` → `Mistral`; anything else
  is an explicit error (never a silent fallback).
- `fn resolve_backend(requested: BackendKind, path: &Path) -> Result<BackendKind, String>` —
  `Auto` detects; an explicit kind is checked against the path (mismatch
  fails fast, e.g. `--backend mistral` on a `.gguf` file).
- `fn is_gguf_file(path: &Path) -> bool`,
  `fn is_mistral_dir(path: &Path) -> bool` — the two predicates above.

### `runa_engine::MistralModel` (feature `mistralrs`)

- `fn load(path: &Path) -> Result<MistralModel, EngineError>` — validates
  the directory (`config.json`) before any mistral.rs call (fast offline
  failure), then loads via `ModelBuilder` + `blocking::BlockingModel`
  (own tokio runtime; must not run inside an existing runtime — `run` /
  `chat` are sync, `serve` uses plain std engine threads).
- `fn path(&self) -> &Path`.
- `fn generate(&mut self, req: GenerateRequest) -> Result<MistralGeneration, EngineError>` —
  `MistralGeneration: Iterator<Item = Result<GenEvent, EngineError>>` with
  the ggml terminal order (`Text…`, `Usage`, `Done`).
- `fn ensure_backend_available(kind: BackendKind) -> Result<(), EngineError>` —
  always compiled; `Mistral` without the feature errors with the rebuild
  pointer (`--features mistralrs`).
- `EngineError::Mistral(String)` — load/generate failures and GGUF-only
  options used with `--backend mistral`.

### Request mapping (`GenerateRequest` → mistral.rs)

| runa field | mistral.rs |
|---|---|
| messages (`system`/`user`/`assistant`) | `RequestBuilder::add_message` (`tool` roles rejected) |
| `think.mode != Off` | `enable_thinking(bool)`; `think.show` gates `Reasoning` events |
| temperature ≤ 0 | `set_deterministic_sampler()` |
| temperature / top_k (> 0) / top_p / min_p | `set_sampler_temperature/topk/topp/minp` |
| `max_tokens` | `set_sampler_max_len` |
| `stop` | `StopTokens::Seqs` |
| finish `length` / other | `Done(MaxTokens)` / `Done(Eos)` |
| usage | prompt/completion tokens + pp/tg tok/s |

Not mapped, rejected explicitly (never silent): `tools` (`--mcp`),
`json_schema`/`grammar`, `audio_pcm`/`images` (mtmd), `speculative`
(ngram/draft). Not mapped, accepted as ggml-only (see `--backend` help):
`seed` (no mistral.rs equivalent), `--mode`/`--ctx` (mistral auto-maps
devices/context). Requested tools are refused, so surfaced `ToolCalls`
cannot occur; volunteered tool calls (unprompted by any request) surface as
JSON text, mirroring how the ggml backend surfaces unrequested tool markup.

### Pull / fit / serve / doctor

- `runa pull hf:<repo>:safetensors` downloads the snapshot
  (`config.json`, tokenizer `*.json`, `*.index.json`, `*.safetensors`,
  flat layout only) into the model store with per-file size + SHA-256
  verification and `.verified` sidecars, like GGUF pulls.
- `runa_fit::is_safetensors_tag(tag)` (case-insensitive);
  `Fetcher::siblings_all(repo)` (all rfilenames; `siblings` is the
  gguf-filtered view); `RemoteError::Safetensors` refuses `fetch_header`
  for safetensors refs, and `runa fit` refuses mistral directories —
  both with an explicit message.
- `serve` resolves the backend per model (`Auto` detects); the pool thread
  holds `LocalEngine` (`run`/`chat` share the enum in `runa-pool/src/engine.rs`);
  `/v1/embeddings` on a mistral model errors explicitly (gguf only).

## P9.1 (`runa daemon` background service)

The daemon keeps models warm between CLI calls, owns one
`MemoryManager` over a real RSS backend, and serves `run` / `chat` over a
Unix socket (`~/.cache/runa/runa.sock`, `RUNA_DAEMON_SOCK` overrides).
Wire: one NDJSON [`DaemonRequest`] line per request, a stream of
[`DaemonEvent`] lines ending in `done` / `error` per reply
(`crates/runa/src/daemon_proto.rs`). `run` / `chat` dial first and fall
back to in-process load on refusal (`--no-daemon` skips the dial);
media, MCP, speculation, and load-shaping flags always stay local.
Per-request preflight is `touch()` + `on_heavy(0)` — admission is
pool/LRU bound. The idle tick calls `maybe_idle()` at least every
`MAX_IDLE_TICK_SECS`.

### `fn SysinfoBackend::new() -> SysinfoBackend`

- Returns: a real RSS backend (P9.1). `rss_mib()` reads this process's
  resident set via `sysinfo`; `shrink_to` / `grow` are advisory no-ops
  returning current RSS (the OS owns the pages — release happens through
  `LoadedModel::on_idle` and pool LRU eviction).
- Example: `MemoryManager::new(policy, ceiling, Box::new(SysinfoBackend::new()))`
- Notes: replaces `FakeBackend` at the `run` / `serve` / daemon call
  sites; unit tests keep using `FakeBackend`.

### `fn SysinfoBackend::process_rss_mib() -> u64`

- Returns: current process RSS in MiB, `0` when the process table is
  unreadable. Pure observation, no side effects.

### `fn default_socket_path() -> PathBuf`

- Returns: the daemon socket path — `RUNA_DAEMON_SOCK` when set, else
  `$XDG_CACHE_HOME/runa/runa.sock` or `~/.cache/runa/runa.sock`.

### `fn ModelPool::insert_spec(&mut self, path: &Path) -> Result<String, String>`

- Params: `path` — model file to serve on demand.
- Returns: the pool id (`Ok`): the existing id when the path is known,
  else the file stem (`stem-2`, … on collision). `Err` when the file is
  missing. Never unloads models.

### `fn resolve_or_insert(pool: &Mutex<ModelPool>, model: Option<&str>) -> Result<String, String>`

- Effect: `resolve_id` first; when the model is an on-disk path the pool
  does not know yet, `insert_spec` it and return the new id.
- Errors: `model <id> not found` when the model is neither a known id
  nor an existing file.

### `fn generate(pool: &Arc<Mutex<ModelPool>>, model_id: &str, req: GenerateRequest) -> Result<Vec<GenEvent>, String>`

- Effect: resolve + (blocking) load the engine, run one generation on
  its thread, collect the events. Async wrapper — never blocks the
  executor. `serve` keeps its own status-mapped variant.

### `fn request_sync(socket: &Path, req: &DaemonRequest, timeout: Duration) -> Result<Vec<DaemonEvent>, String>`

- Returns: daemon events up to and including `done` / `error`.
- Errors: transport failures mean "no daemon" (the caller falls back);
  a daemon-side failure arrives as `DaemonEvent::Error` inside `Ok`.
  Non-unix stub always errs (unix sockets only).

### `fn install_daemon(home: &Path, exe: &Path, argv: &[String]) -> Result<Vec<PathBuf>, String>`

- Effect: write the launchd plist
  (`~/Library/LaunchAgents/ai.runa.daemon.plist`) and the systemd user
  unit (`~/.config/systemd/user/runa-daemon.service`) for `exe argv…`;
  returns both paths. Overwrite is idempotent.

### `fn uninstall_daemon(home: &Path) -> Result<Vec<PathBuf>, String>`

- Effect: remove both units; returns the paths removed (empty when
  nothing was installed). Missing files are not errors.

### `fn launchd_plist(exe: &Path, argv: &[String]) -> String`

- Returns: the launchd plist text (`ai.runa.daemon`, `RunAtLoad` +
  `KeepAlive`, logs to `~/.cache/runa/daemon.{out,err}.log`).

### `fn systemd_unit(exe: &Path, argv: &[String]) -> String`

- Returns: the systemd user-unit text (`Restart=on-failure`,
  `WantedBy=default.target`).

## P10.1 — Calibration-aware speed predictions (M3 follow-up)

Crate: `runa-fit` (`speed::apply_efficiency`); wired in the `runa`
binary (`bench::predicted_speeds`, `fit::calibrate_report`,
`fit::calibrate_pick`). `runa bench` records measured/predicted pairs;
predictions now scale by the median measured/predicted ratio for the
exact `(device, backend, quant)` triple. An empty or missing DB is a
no-op (raw model output, M3-cold behaviour).

### `fn runa_fit::apply_efficiency(pp: f64, tg: f64, eff: Option<&Efficiency>) -> (f64, f64)`

- Params: raw prefill/decode predictions + `CalibrationDb::get_efficiency`
  result for the run's `(device, backend, quant)`.
- Returns: `(pp × pp_efficiency, tg × tg_efficiency)`. `None` returns the
  inputs unchanged; a non-positive or non-finite factor is treated as
  missing for that axis only (one bad bench run never zeroes a line).
- Check: `apply_efficiency_scales_both_axes`,
  `apply_efficiency_missing_db_keeps_raw`,
  `apply_efficiency_ignores_bad_factors_per_axis` (`speed.rs`).

### Binary surface (`runa`)

- `bench::predicted_speeds` loads `default_calibration_path()`
  (`RUNA_CALIBRATION`-aware) and keys `(device_backend(placement),
  quant_from_name(path))` before printing or recording predictions.
  Note: the recorded `predicted_*` values are post-calibration, so the
  next sample's ratio measures residual error, not the raw model.
- `fit::calibrate_report` scales the `speed_gpu` line by the GPU-side
  (`metal:0`/`metal` on macOS, else `cuda:0`/`cuda`) efficiency and the
  `speed_cpu` line by `("cpu", "cpu", quant)`; `quant` comes from
  `calibration_quant` (local filename, HF file/quant, or URL tail).
  `DecodeSlow` warnings still use the uncalibrated check inside
  `check_fit` (order-of-magnitude gate).
- `fit::calibrate_pick` scales each `--recommend` pick for the side its
  verdict chose (hybrid uses the GPU-side factor — the same
  approximation `decode_for` documents).
- Check: `fit::tests::{entry_quant_…, calibration_quant_…,
  calibrate_pick_…}`; e2e `fit_calibration_db_scales_predictions`
  (2.0× sample doubles `runa fit --json` decode on the qwen2 fixture).

## P10.2 — `--threads` knob + P-core-aware default (M4 follow-up)

Crate: `runa-engine` (`load::default_threads`); CLI surface in the
`runa` binary (`--threads` on `run` / `chat` / `bench` / `serve`,
`RUNA_THREADS`, `[defaults] threads`). Previously every load used all
logical CPUs incl. E-cores and lost to `llama-bench` auto-threading
(M4); the default now matches llama.cpp's own (`cpu_get_num_math`):
P-cores on Apple Silicon, logical CPUs elsewhere.

### `fn default_threads() -> i32` (`runa-engine/src/load.rs`)

- Returns: `hw.perflevel0.logicalcpu` on macOS (P-cores via
  `sysctlbyname`, `libc` macOS-only dep), else
  `available_parallelism`; always `>= 1` (fallback 4 when unknowable).
- `fn apple_pcore_threads() -> Option<i32>` — the raw sysctl read;
  `None` on any failure (missing key, short read, absurd value; macOS
  is Apple Silicon only) so the caller falls back silently to logical CPUs.
- Check: `threads_tests::{default_threads_is_sane,
  apple_pcore_reading_is_plausible}`.

### `fn config::resolve_threads(cli: Option<i32>) -> Result<Option<i32>, String>`

- Precedence: CLI `--threads` > `RUNA_THREADS` > `[defaults] threads`
  in config files (later files win) > `None` (engine default).
- Errors: any value `< 1` (`--threads 0`, `RUNA_THREADS=lots`, or a
  bad `[defaults] threads`) fails before any load.
- Notes: gguf loads only — `run`/`chat` on `--backend mistral` reject
  `--threads` explicitly (mistral.rs owns its threads); `--threads`
  also opts a `run` out of the daemon (load-shaping flag, P9.1), like
  `--device`/`--kv*`.
- Check: `config::tests::{defaults_threads_toml,
  resolve_threads_precedence}` (`RUNA_THREADS` save/restore),
  `daemon_gate_tests` (`threads` stays local), e2e
  `threads_zero_fails_with_usage_error`; trycmd `run`/`chat`/`serve`
  help fixtures + `docs/runa-run.1` regenerated.

## P10.3 — `--lang auto` ASR fix (M7 blocker)

Crate: `runa-media` (`asr::AsrEngine::transcribe`). Default
`--lang auto` reported a correctly detected language with an empty
transcript; `--lang en` on the same audio was correct.

- Root cause (verified in the vendored `whisper.cpp`
  `whisper_full`): the `detect_language` flag means *detect only* —
  the function returns 0 right after detection without decoding.
  Setting both the flag and `language = "auto"` (the old code) or
  the flag alone (first fix attempt) always yields zero segments.
  The correct call is `language = "auto"` with the flag unset:
  whisper.cpp then auto-detects *and* decodes.
- Fix: on `None | Some("auto")` only `set_language(Some("auto"))`,
  never `set_detect_language(true)`.
- Verified live (cached `ggml-base.bin`, Apple Silicon CPU):
  `say`-synthesized EN speech transcribes byte-identical under
  `--lang auto` and `--lang en`; sine fixture decodes identically too.
- Check: `asr::tests::auto_detect_decodes_like_explicit` —
  live-gated on the cached model (skips in CI like the other
  model-dependent tests); proven to FAIL on the pre-fix code and
  pass after. M7 row in `docs/release-1.0.md` updated (timing
  re-run still pending).

## P10.5 — Serve/daemon idle tick calls `on_idle` (M11 follow-up)

`LoadedModel::on_idle` existed but nothing called it outside unit
tests: the daemon's `maybe_idle` only drove the (advisory no-op)
`SysinfoBackend`, and serve had no manager at all. Now the pool owns
per-engine activity and both long-lived processes sweep it.

### `EngineJob::Idle` (`pool.rs`)

- Fire-and-forget job: the engine thread runs `LocalEngine::on_idle`
  (ggml unmaps the LMDB prompt cache, model kept; mistral no-op) and
  never fails the tick. Queued behind any in-flight request.
- `fn LocalEngine::on_idle(&mut self)` (`engine.rs`) — the backend
  passthrough.

### `ModelPool` idle tracking (`pool.rs`)

- Fields `last_used: HashMap<String, Instant>` (stamped by
  `touch_lru` on every hit and fresh load, cleared on LRU evict) and
  `idle_timeout: Duration` (`Duration::MAX` = disabled; serve/daemon
  set it from the memory policy via `with_idle_timeout`).
- `fn due_for_idle(&self, now: Instant) -> Vec<String>` — pure
  decision (engines unused ≥ timeout), unit-testable without models.
- `fn idle_sweep(&mut self) -> Vec<String>` — sends `Idle` to every
  due engine, re-stamps it (one sweep per timeout, no tick spam),
  forgets dead engine threads. May block briefly behind a busy engine
  — call from a blocking thread only.
- Check: `idle_sweep_fires_once_per_timeout`,
  `idle_sweep_disabled_by_default`, `idle_sweep_drops_dead_engines`
  (channel ends stand in for engine threads).

### `async fn idle_tick(pool, mm, tag)` (`pool.rs`)

- One tick for both loops: `mm.maybe_idle()` (manager state/logs)
  plus the pool sweep off-thread (`spawn_blocking`), one
  `<tag>: idle <id>: prompt cache released (model kept)` line per
  swept engine.
- Serve builds its own `MemoryManager` in `listen` (policy from
  config, tick clamped to `MAX_IDLE_TICK_SECS`); the daemon reuses
  its existing manager and its tick now calls `idle_tick` instead of
  bare `maybe_idle`. Generation endpoints (`chat/completions`,
  `messages`, `embeddings`, `transcriptions`, daemon `serve_request`)
  `touch` the manager; `/health` and `/v1/models` deliberately do
  not, so monitoring polls cannot hold engines awake.
- Check: e2e `serve_idle_tick_releases_prompt_cache`
  (`RUNA_MEMORY_IDLE_TIMEOUT_S=1`, asserts the sweep log line).
- M11 verdict (measured, `docs/release-1.0.md`): the release is real
  but RSS-negligible next to the resident model (772.9 MiB before
  and after on qwen2-0.5B) — the gate stays infeasible without model
  unload, which D17 forbids.

## P10.6 — Anthropic SSE streaming tool support

Crate: `runa-cloud` (`anthropic::parse_sse`); the CLI tool loop
(`drain_anthropic`) already drains `ToolUse` — only the streaming
parser dropped the blocks, so a `stream: true` tool call vanished.

### `struct SseTools` + `SseBlock` (`anthropic.rs`)

- Accumulates `content_block_start` (tool_use id/name, text/thinking
  block kinds) plus `input_json_delta` / `text_delta` /
  `thinking_delta` / `signature_delta` fragments, keyed by block
  index (`BTreeMap` keeps wire order).
- `fn flush(&mut self) -> Option<AnthropicEvent>` rebuilds the
  content-block array and the `ToolCall`s (bad-fragment JSON becomes
  `{}`); thinking blocks keep their streamed signature when present.
  Emitted once: at `message_delta` carrying a `stop_reason` (ahead of
  `Done`, the `parse_message` position), or at end-of-transcript for
  a cut stream. Plain-text streams stay quiet.
- `push_sse_event` additionally swallows tool/signature fragments so
  partial JSON never leaks into `Text`/`Reasoning` events.
- The CLI keeps `stream: false` on purpose: only the whole-message
  reply preserves thinking signatures for the next tool round
  (comment on the request site in `cloud.rs`).
- Check: `sse_tool_use_reassembled_across_fragments` (two calls,
  fragmented args, block order, ToolUse-before-Done, no text leak),
  `sse_without_tools_emits_no_tool_use`; existing
  `thinking_delta_and_refusal` + wiremock tests unchanged.

## P10.7 — Offline `--recommend` counts KV + compute

`probe_offline` used to fit on file sizes alone, so a long context on
a big model fit offline but not online. The catalog now carries the
two measured numbers the header used to be the only source for.

### `CatalogEntry::{kv_mib_per_1k, compute_mib}` (`recommend.rs`)

- `kv_mib_per_1k`: KV MiB per 1024 ctx tokens at f16, measured once
  per entry (`runa fit --json --ctx 8192` over the header; KV is
  exactly linear in ctx). `q8_0` halves it, `q4_0` quarters it;
  unknown `kv_type` strings keep the f16 number (never silently
  shrink the need).
- `compute_mib`: compute-buffer MiB at ubatch 512, linear in
  `n_ubatch` (mirrors `estimate_compute`'s scaling).
- `fn kv_bytes_for(ctx_len, kv_type) -> u64` and
  `fn compute_bytes_for(n_ubatch) -> u64` do the scaling (ceiled,
  floored at zero).
- `probe_offline` need = weights + projector + KV(ctx) +
  compute(ubatch); speed uses the online bytes-per-token shape
  (active + KV/2). `catalog_is_sane` requires both numbers > 0 on
  every entry.
- Check: `offline_counts_kv_and_compute` (KV tips Gpu→Cpu→NoFit,
  `q8_0` recovers Gpu, scaling helpers); e2e
  `fit_recommend_offline_ranks_the_catalog` unchanged-green.

## P10.9 — Chat keeps history across turns

`Session.history: Vec<ChatMessage>` rides every REPL/TUI turn
(P10.9). Previously each turn sent one user message; tool rounds
already stayed inside a turn, but nothing carried answers forward.

- `fn turn_messages(session, input) -> Vec<ChatMessage>` — pushes the
  new user message, trims to ~75% of `session.ctx`, returns the full
  transcript for the request. Trimming prints `(history trimmed to
  fit ctx)`.
- `fn commit_history(session, sent, answer)` — after an answered
  turn, history becomes the sent messages (they rode the whole tool
  loop, so every round is in there) plus the final assistant answer.
  Failed turns record nothing (retry sends the same history).
- `fn trim_history(history, budget) -> usize` — drops oldest *turns*
  (up to the next `user` message, so `tool` messages never orphan
  from their `assistant` call); the newest message always survives.
  `fn estimate_history_tokens` is a chars/4 heuristic guard rail —
  the engine still fails loudly past real ctx.
- `/reset` (REPL + TUI) and `/model` (new weights, stale transcript)
  clear history; `/mode` keeps it. The daemon protocol already
  carries full `messages` (`ProtoMessage` incl. tool calls), so no
  wire change was needed.
- Check: `history_tests` (trim order/newest-survives/no orphaned
  tool messages/commit shape); e2e `chat_second_turn_remembers_history`
  (qwen2 fixture recalls "Ada", 3/3 locally) next to the existing
  context-reuse test.

### P11.4 — Tokenizer-exact history trimming

Local gguf turns count real tokens: `LoadedModel::
count_history_tokens(messages)` tokenizes every message's content
with the model's own tokenizer (`AddBos::Never`) plus the same +4
per-message template overhead the estimate uses; the ~75%-of-ctx
guard in `turn_messages` is re-checked against that exact count.
The daemon socket path has no tokenizer (and the mistral backend
exposes none), so those keep `estimate_history_tokens` (chars/4).
`trim_history` stays as the estimate-backed wrapper over the generic
`trim_history_with`; `count_history_tokens_with` pins the exact path
against a stub counter in unit tests.
- Check: `history_tests::{exact_counter_counts_known_string,
  trim_with_exact_counter_drops_oldest_turn_first,
  fallback_estimate_locked_on_known_string}`.

## P10.10 — Split-GGUF models in the catalog

`CatalogEntry.parts: Vec<String>` holds shard refs 2..N (`ref` stays
part 1); `size` is the parts' byte sum for such entries.

### `fn CatalogEntry::all_refs(&self) -> Vec<&str>` + `fn combine_shard_weights(first, rest)` (`recommend.rs`)

- `probe_remote` fetches every part header, builds the descriptor
  from part 1 (shards repeat the full metadata) and sums the four
  weight groups across parts before `check_fit`. Single-file entries
  take the identical path with one ref — no behaviour change there.
- `catalog_is_sane` now also requires every ref (part 1 + parts) to
  be a `hf:….gguf` ref with no duplicates. No split entry ships yet:
  every current model fits in one file, so `parts` is schema +
  probe support with coverage, waiting for the first sharded model
  that fits Tier-1/2 hardware.
- Check: `split_tests::{all_refs_orders_part_one_first,
  split_parts_combine_weights}` (real qwen2 header metadata with
  overridden groups — no third synthetic-GGUF builder copy).

## P10.11 — Harmony tool-format + `thinking_forced_open` coverage

Both P8.2 leftovers closed by verification, not new branches:
llama.cpp's own handler already owns both formats.

- Harmony: the pinned llama.cpp ships `common_chat_parse_gpt_oss`
  (commentary/analysis channels, `to=functions.<name>` recipient in
  either header order). `harmony_tool_reply_parses`
  (`structured.rs`) renders a required tool call through the REAL
  gpt-oss-20B template and parses canned replies in both header
  orders — 1 call, `get_weather`, no markup leak. Fixture-gated
  (11 GiB model, dev-only, skips in CI).
- `thinking_forced_open`: a llama.cpp-internal flag (set from the
  template + our `enable_thinking`, consumed by its parser/grammar);
  our render path already feeds it everything it needs. Proven by
  the extended `think_budget_reports_reasoning_tokens` (`generate.rs`
  integration): Qwen3-8B completes a *required* tool call under a
  64+8 budget with the reported reasoning count inside budget+grace.
- Fixture repair on the way: `tests/fixtures/gpt-oss-20b-MXFP4.gguf`
  was truncated (11.4 of 12.1 GiB — tensors out of file bounds);
  resumed from the Hub to the exact catalog byte size. Fixture files
  are git-ignored weights, so this touches no tracked files.
- Check: `cargo test -p runa-engine --lib harmony` +
  `--test generate think_budget`; format row in `docs/structured.md`.

## P10.13 — `--max-load-percent` system load cap (foreign WIP, completed)

Found in the tree unclaimed and uncompiling (CLI fields without
matching signatures); finished here: missing signature/field
plumbing, daemon-gate opt-out, mistral-backend rejects, docs.
Caps the share of total system resources a run may use, percent
1..=100. Warning-only by design: the run always proceeds, the user
decides (plan D12 — loud, never blocking).

### `const DEFAULT_MAX_LOAD_PERCENT: u8` (= 80)

### `fn config::resolve_max_load_percent(cli: Option<u8>) -> Result<u8, String>`

- Precedence: CLI `--max-load-percent` > `RUNA_MAX_LOAD_PERCENT` >
  `[system] max_load_percent` (later files win) > 80.
- Errors: any value outside 1..=100, at any layer.

### `struct config::SystemSnapshot` + `fn config::read_system_snapshot() -> SystemSnapshot`

- Point-in-time `{total_ram_bytes, avail_ram_bytes, cpu_count}` via
  `sysinfo`. `RUNA_FAKE_TOTAL_RAM_MIB` / `RUNA_FAKE_AVAIL_RAM_MIB` /
  `RUNA_FAKE_CPU_COUNT` override all three for tests.

### `fn config::system_load_warnings(snap, demand_bytes, threads, limit) -> Vec<String>`

- Pure math, one `warning:` line per breached resource: model RAM
  demand over the cap, ambient system RAM pressure over the cap,
  `--threads` CPU share over the cap. Every line names the value to
  set so the run fits.
- `fn config::warn_if_over_system_limit(demand, threads, cli)` —
  resolves the cap and prints the lines (startup path: `run`,
  `chat`, `bench`, `serve`, `daemon`); returns the limit.
- `fn config::warn_if_demand_over_limit(demand, cli)` — model-need
  line only (`preflight_grow`, after the header is read).
- Check: `config::tests::max_load_*` (TOML/env/CLI precedence,
  fake-snapshot warnings); e2e-grade paths smoke-tested via CLI
  (`--max-load-percent 0` and `RUNA_MAX_LOAD_PERCENT=lots` fail
  before any load).
- Notes: `--max-load-percent` opts `run` out of the daemon (the
  daemon owns placement and knows no per-request cap) and is
  rejected on `--backend mistral` (never warned there, so never
  silently accepted).

## P14.4 — serve dashboard model residency

### `fn ModelPool::loaded_weight_bytes(&self) -> Vec<(String, u64)>`

- Returns: one `(id, bytes)` pair per engine currently resident, sorted by
  id. `bytes` is the spec file length, or for a directory spec the sum of
  the regular files directly inside that directory.
- Example: `lock(&pool).loaded_weight_bytes()`.
- Notes: approximate on-disk weight size, not process RSS. The serve
  dashboard (`docs/dashboard.md`) publishes this list. Empty when nothing
  is loaded.
