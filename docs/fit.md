# Fit checker internals

Formulas, estimators and calibration for `runa fit` (plan P1, decisions
D5/D6). Estimator sections land with their tasks (P1.5 compute buffer,
P1.9 speed model); this file starts with the CLI and the remote header
path (P1.2).

## CLI

```sh
runa fit hf:unsloth/Qwen3-8B-GGUF:Q4_K_M --ctx 16384 --kv q8_0
runa fit ./model.gguf --json
runa fit --recommend --use code --top 3
runa fit --recommend --offline
```

`runa fit <model>` takes a local GGUF, an alias, `hf:<repo>:<file-or-quant>`,
or an http(s) URL. A pulled copy is read from disk; anything else fetches
only the header. It prints the verdict report (memory table, decode,
prefill, TTFT, warnings, suggestions) and exits `0` (fits), `1` (fits with
warnings) or `2` (no fit). `--json` prints the same numbers as one object.
A local model's sibling `mmproj-*.gguf` is counted. `--rpc host:port,…`
(P9.3) estimates the local placement only: without the `rpc` cargo feature
an explicit warning is printed, with it a note that remote memory is not
counted; `runa run --rpc … --device RPC0` contacts the servers at load.

`--recommend` fits every model in the embedded catalog
(`crates/runa-fit/src/catalog.toml`, `runa_fit::recommend`) in parallel and
lists the top `--top` (default 5), each with the `runa pull` ref. NO FIT
entries drop out. The rest rank by:

1. usable speed first (predicted decode ≥ 5 tok/s),
2. then quality tier (1 small … 4 best in the catalog),
3. then predicted decode tok/s.

`--use chat|code|vision|reasoning` keeps the models tagged for that use;
with `vision` the projector size counts. Headers are cached after the first
run (see below). `--offline` needs no network: it fits the catalog file
size (+ projector) against VRAM minus the margin, else RAM, and predicts
decode from bandwidth over the active bytes (MoE entries store them). KV
and compute buffers are not counted offline, and speeds print with `~`.
A failed probe prints `skip <name>: <error>` on stderr. Nothing fitting
exits `2`.

The catalog holds single-file GGUFs only (the header probe reads the first
file, so split models would hide most of their tensors). Update sizes from
`https://huggingface.co/api/models/<repo>?blobs=true`.

## Remote header fetch (P1.2)

`runa fit` answers *before* downloading, so the GGUF header (magic +
metadata + tensor-info table) is fetched with HTTP `Range` requests that
grow until [`Reader::parse`](../crates/runa-fit/src/gguf.rs) succeeds.
Implementation: `crates/runa-fit/src/remote.rs` (`Fetcher`).

Model references (D3):

| Ref | Resolution |
|-----|------------|
| `hf:<repo>:<file.gguf>` | `https://huggingface.co/<repo>/resolve/main/<file.gguf>` |
| `hf:<repo>:<quant>` | Hub API sibling listing → best `<quant>` match → as above |
| `https://…` | direct URL (must serve byte ranges) |
| `<local path>` | same grow-until-parse loop over the file (seek, no network) |

Quant matching (`pick_quant`): exact filename → `-`/`_`-suffixed
(`…-Q4_K_M.gguf`) → case-insensitive substring; ties break by shortest
name, then lexicographic. Deterministic; unit-tested.

Fetch loop: ask `bytes=have..want-1` starting at 8 MiB, doubling to a
256 MiB cap. Truncation-shaped parse errors (`Truncated`,
`UnexpectedEof`, `TensorTableTruncated`, `StringTooLong`) mean "fetch
more"; any other parse error means the remote file is corrupt. Servers
that ignore ranges (200 instead of 206) are accepted only when the whole
body fits the header budget. Auth: `HF_TOKEN` bearer (redirects to
signed S3 URLs need none — reqwest strips auth cross-domain).

Cache: `<cache>/runa/headers/` (`XDG_CACHE_HOME`, `%LOCALAPPDATA%`,
`~/.cache`, else temp dir), one file per URL plus a JSON sidecar
(`total_len`, `etag`). Reuse requires the server to report the same
total length and ETag on a 1-byte probe.

Measured 2026-09-08 (live, cold):

| Model | Header bytes | Tensors | File size | Elapsed |
|-------|--------------|---------|-----------|---------|
| `hf:unsloth/Qwen3-8B-GGUF:Q4_K_M` | 8 MiB (1 round trip) | 399 | 5.0 GiB | 4.3 s |
| local `qwen2-0_5b-instruct-q4_0.gguf` (test server) | 8 MiB (1 round trip) | 290 | 353 MiB | 0.2 s |

The 4.3 s cold fetch is TLS + redirect chain + 8 MiB transfer; warm
(cached) fetches do a single 1-byte probe. The `< 3 s` plan target is
for the `runa fit` CLI path (P1.11) with a warm connection.

## Placement (P1.8)

`plan_placement` (`crates/runa-fit/src/planner.rs`) maximizes GPU-resident
weights under `VRAM − margin` (default margin 1 GiB, `--fit-margin` later).
Compute buffer + KV + mmproj + LoRA adapters are reserved on GPU first;
the remaining weight budget is filled back-to-front by eviction priority:
MoE experts (`ffn_*_exps`) spill first, then embedding/output tables, dense
attention/FFN last. `mmproj_bytes` (default 0, set for VL/audio models —
see P4.8) and `lora_bytes` (default 0, sum of the `--lora` adapter file
sizes — see P8.5) participate in the reservation and the per-device byte
table (`gpu_weight_bytes`, `cpu_weight_bytes`, `compute_buffer_bytes`,
`kv_bytes`, `mmproj_bytes`, `lora_bytes`, `gpu_total_bytes`).

Verified on the real Qwen3-30B-A3B header: 8 GiB VRAM parks experts on
CPU while dense stays on GPU; ample VRAM puts all layers on GPU.
(`real_moe_fixture_small_vram_parks_experts_on_cpu`).

## Media / VL context (P4.8)

`FitConfig.media` (`MediaFit`) adds image/video/audio tokens to the context
check:

```text
media_tokens = frames × tokens_per_frame
             + ceil(audio_seconds) × tokens_per_audio_second
```

Defaults: 256 tokens/frame, 25 tokens/s of audio. If `media_tokens > n_ctx`
the verdict is `NO FIT` (exit 2) with `Warning::MediaExceedsContext`.

VRAM: `PlannerConfig.mmproj_bytes` is reserved on GPU (same as compute/KV).
`--draft PATH` (P5.6) adds the draft GGUF file size to `mmproj_bytes` so
fit accounts for a second model even though decode still uses n-gram.
`has_mmproj && mmproj_bytes == 0` still warns that the projector was not
sized. Encoder scratch is `frames × n_embd × 4 × 8` via
`estimate_encoder_compute` and is added to the GPU reservation.

## NPU limits (P9.4, Tier 3)

NPU support is opt-in scaffolding, not offload: `llama-cpp-2` has no
`hexagon`/`openvino` backends through 0.1.154 (survey in
`docs/versions.md`), so every NPU estimate below is a conservative stub
and placement always stays on CPU/GPU. `runa doctor` reports
`hexagon-stub` / `openvino-stub` only when built with the matching cargo
feature; `RUNA_NPU=hexagon|openvino` opts the `runa auto` verdict line
into naming the probe outcome (without the variable the verdict never
mentions NPU); `RUNA_FAKE_NPU=1|hexagon|openvino` fakes presence for
tests, `RUNA_FAKE_NPU=0` forces absent.

- `HwSpec::hexagon()` — 60 GB/s, 45 TOPS, efficiency 0.30 (Snapdragon
  X-class NPU, uncalibrated). Hexagon DSP kernels page weights through a
  ~3.5 GiB window: treat any model (or offloaded slice) above ~3.5 GiB as
  a split across the window, not one resident blob.
- `HwSpec::openvino()` — 40 GB/s, 13 TOPS (floor SKU), efficiency 0.30
  (uncalibrated). Text-only models: vision/audio encoder paths have no
  OpenVINO validation, so multimodal estimates are CPU/GPU-only.
- Both stubs are Q4_0-centric: higher-precision weights on an NPU have no
  measured efficiency, so the 0.30 factor assumes Q4-class quants and
  under-predicts anything larger (safe direction for a fit verdict).
- Recalibrate (`efficiency`, bandwidth, TOPS) from on-device `runa bench`
  runs before quoting NPU speeds to users; until then the verdict labels
  them `(uncalibrated Tier-3 estimate)`.
