// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! Model loading with [`Placement`] (plan P2.1).
//!
//! [`load`] initializes the backend, translates [`Placement`] +
//! [`LoadConfig`] into `llama-cpp-2` params, loads the model and prints the
//! verdict line (placement + memory — plan D12) before returning. Speed
//! forecasts join the verdict line with the `runa run` CLI (P2.5), which
//! owns the fit verdict.

use std::ffi::CString;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::{KvCacheType, LlamaContextParams};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::LlamaLoraAdapter;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::{LlamaModelParams, LlamaSplitMode};

use crate::lora::LoraSpec;
use crate::placement::Placement;

/// KV-cache quantization (P2.7: `--kv` / `--kv-k` / `--kv-v`).
/// Quantized types require flash attention (enforced in [`load`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KvKind {
    /// Full precision (llama.cpp default).
    #[default]
    F16,
    /// 8-bit KV.
    Q8_0,
    /// 4-bit KV.
    Q4_0,
}

impl KvKind {
    fn as_llama(self) -> KvCacheType {
        match self {
            KvKind::F16 => KvCacheType::F16,
            KvKind::Q8_0 => KvCacheType::Q8_0,
            KvKind::Q4_0 => KvCacheType::Q4_0,
        }
    }

    /// Canonical CLI / planner name (`f16`, `q8_0`, `q4_0`).
    pub fn as_str(self) -> &'static str {
        match self {
            KvKind::F16 => "f16",
            KvKind::Q8_0 => "q8_0",
            KvKind::Q4_0 => "q4_0",
        }
    }

    /// True for types that llama.cpp only supports with flash attention.
    pub fn is_quantized(self) -> bool {
        !matches!(self, KvKind::F16)
    }
}
impl std::str::FromStr for KvKind {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "f16" | "fp16" => Ok(KvKind::F16),
            "q8_0" | "q8" => Ok(KvKind::Q8_0),
            "q4_0" | "q4" => Ok(KvKind::Q4_0),
            other => Err(format!(
                "unknown kv type {other:?}; expected f16, q8_0, or q4_0"
            )),
        }
    }
}

/// Planner uses one KV type string. Mixed K/V picks the larger (bytes) type
/// so auto-mode never under-estimates VRAM.
pub fn planner_kv_type(k: Option<KvKind>, v: Option<KvKind>) -> &'static str {
    fn rank(kind: KvKind) -> u8 {
        match kind {
            KvKind::F16 => 2,
            KvKind::Q8_0 => 1,
            KvKind::Q4_0 => 0,
        }
    }
    let kk = k.unwrap_or(KvKind::F16);
    let vv = v.unwrap_or(KvKind::F16);
    if rank(kk) >= rank(vv) {
        kk.as_str()
    } else {
        vv.as_str()
    }
}

/// Load-time configuration: context sizes, threads, mapping flags.
#[derive(Debug, Clone)]
pub struct LoadConfig {
    /// Context length (tokens).
    pub n_ctx: u32,
    /// Batch size (prompt processing).
    pub n_batch: u32,
    /// Micro-batch size.
    pub n_ubatch: u32,
    /// Worker threads; default = logical CPUs (`available_parallelism`).
    /// Physical-core topology arrives with the P1.6 hardware probe.
    pub threads: Option<i32>,
    /// Memory-map the model file (default on).
    pub mmap: bool,
    /// Lock pages in RAM (default off).
    pub mlock: bool,
    /// Flash attention `AUTO` (lets llama.cpp pick per model/hardware).
    pub flash_attn: bool,
    /// KV-cache types; `None` = engine default (f16).
    pub kv_k: Option<KvKind>,
    /// Optional mtmd projector (P4.3). Loaded when the `mtmd` feature is on.
    pub mmproj: Option<PathBuf>,
    /// See [`LoadConfig::kv_k`].
    pub kv_v: Option<KvKind>,
    /// LoRA adapters (P8.5): each is loaded with `lora_adapter_init` and
    /// attached to the context with `lora_adapter_set` at its scale.
    pub loras: Vec<LoraSpec>,
    /// Weight-load progress in per mille (0–1000), updated by llama.cpp
    /// while [`load`] runs (P8.8 serve warm-up).
    pub progress: Option<Arc<AtomicU32>>,
}

impl Default for LoadConfig {
    fn default() -> Self {
        LoadConfig {
            n_ctx: 4096,
            n_batch: 512,
            n_ubatch: 512,
            threads: None,
            mmap: true,
            mlock: false,
            flash_attn: true,
            kv_k: None,
            kv_v: None,
            mmproj: None,
            loras: Vec::new(),
            progress: None,
        }
    }
}

/// Errors from [`load`].
#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    /// The model file does not exist.
    #[error("model file not found: {0}")]
    ModelNotFound(PathBuf),
    /// llama.cpp refused the load (bad file, OOM, backend mismatch…).
    #[error("llama.cpp load failed for {path}: {msg}")]
    LoadFailed { path: PathBuf, msg: String },
    /// Global backend init failed (llama.cpp backends, NUMA, …).
    #[error("llama.cpp backend init failed: {0}")]
    BackendInit(String),
    /// Context creation failed.
    #[error("llama.cpp context failed (n_ctx={n_ctx}): {msg}")]
    ContextFailed { n_ctx: u32, msg: String },
    /// A CPU-buffer override pattern is not valid for the FFI boundary.
    #[error("bad cpu pattern {0:?}: interior nul byte")]
    BadPattern(String),
    /// Requested option unsupported by the pinned llama-cpp-2 version.
    #[error("unsupported on llama-cpp-2 0.1.133: {0}")]
    Unsupported(&'static str),
    /// An `--rpc` endpoint refused the connection (P9.3). The servers are
    /// contacted during [`load`] before any tensor is allocated, so this
    /// never degrades into a silent local run.
    #[error("rpc server unreachable: {0} (is rpc-server running there?)")]
    RpcUnreachable(String),
    /// Chat-template rendering failed (P2.2).
    #[error("template error: {0}")]
    Template(String),
    /// Prompt tokenization failed (P2.2).
    #[error("tokenize error: {0}")]
    Tokenize(String),
    /// Batch decode failed (P2.2).
    #[error("decode error: {0}")]
    Decode(String),
    /// `--device` index/name is unknown (P2.9).
    #[error("{0}")]
    BadDevices(String),
    /// mtmd / mmproj / native audio (P4.3).
    #[error("{0}")]
    Media(String),
    /// `/v1/embeddings` (P6.1).
    #[error("embedding error: {0}")]
    Embed(String),
    /// JSON Schema / GBNF grammar rejected (P8.1).
    #[error("grammar error: {0}")]
    Grammar(String),
    /// LoRA adapter failed to load or attach (P8.5).
    #[error("lora {path}: {msg}")]
    Lora { path: PathBuf, msg: String },
    /// mistral.rs backend (P9.2): load/generate failures and GGUF-only
    /// options used with `--backend mistral` (tools, grammar, media, …).
    #[error("mistral backend: {0}")]
    Mistral(String),
}

/// llama.cpp prints `llama_kv_cache: size = N.NN MiB` at context create.
/// `llama_state_get_size` is the session blob (often empty-KV) and does not
/// track quantization, so P2.7 scrapes this line instead.
fn parse_kv_cache_bytes(log: &str) -> Option<u64> {
    let mut last = None;
    for line in log.lines() {
        let rest = line
            .split("llama_kv_cache: size =")
            .nth(1)
            .or_else(|| line.split("KV buffer size =").nth(1));
        let Some(rest) = rest else {
            continue;
        };
        let num = rest.split_whitespace().next()?;
        let mib: f64 = num.parse().ok()?;
        last = Some((mib * 1024.0 * 1024.0).round() as u64);
    }
    last
}

static KV_LOG: std::sync::Mutex<String> = std::sync::Mutex::new(String::new());

unsafe extern "C" fn kv_log_callback(
    _level: llama_cpp_sys_2::ggml_log_level,
    text: *const std::os::raw::c_char,
    _user: *mut std::os::raw::c_void,
) {
    if text.is_null() {
        return;
    }
    // SAFETY: llama.cpp passes a NUL-terminated C string for the log line.
    let s = unsafe { std::ffi::CStr::from_ptr(text) }.to_string_lossy();
    eprint!("{s}");
    if let Ok(mut buf) = KV_LOG.lock() {
        buf.push_str(&s);
    }
}

fn capture_kv_log<T>(f: impl FnOnce() -> T) -> (T, u64) {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    KV_LOG.lock().unwrap_or_else(|e| e.into_inner()).clear();
    // SAFETY: process-global C callback; `kv_log_callback` only reads `text`
    // as a C string and appends to `KV_LOG`.
    unsafe {
        llama_cpp_sys_2::llama_log_set(Some(kv_log_callback), std::ptr::null_mut());
    }
    let out = f();
    let bytes = parse_kv_cache_bytes(&KV_LOG.lock().unwrap_or_else(|e| e.into_inner()));
    (out, bytes.unwrap_or(0))
}

/// A loaded model: model + context + the config it was loaded with.
///
/// The llama.cpp backend is process-global (second `init` fails, and
/// dropping a backend frees global state) — see [`global_backend`]. Models
/// share it, which is also what multi-model serving (P6.1) needs.
///
/// Owns the `CString` override patterns for the model's lifetime: the FFI
/// layer keeps raw pointers to them (see `add_cpu_buft_override`), so they
/// must not move or drop before the model.
///
/// `context` holds a `&LlamaModel`, so the model is boxed: moving a
/// `LoadedModel` must not move the model it points at. Fields drop in
/// declaration order, so `context` is freed before `model`.
pub struct LoadedModel {
    context: LlamaContext<'static>,
    model: Box<LlamaModel>,
    placement: Placement,
    config: LoadConfig,
    path: PathBuf,
    /// Owned override patterns (lifetime anchor for FFI pointers).
    _cpu_patterns: Vec<CString>,
    /// Loaded LoRA adapters (P8.5). llama-cpp-2 0.1.133 never frees these
    /// (`LlamaLoraAdapter` has no `Drop`), so they live as long as the
    /// model; the field order keeps them alive while the context borrows
    /// them and drops them before the model.
    _loras: Vec<LlamaLoraAdapter>,
    /// LMDB prompt-prefix cache (P2.8). `None` until attached.
    prompt_cache: Option<crate::prompt_cache::PromptCache>,
    /// Whether the last [`LoadedModel::generate`] restored from LMDB.
    last_cache_hit: bool,
    /// Allocated KV-cache bytes from llama.cpp (`llama_kv_cache: size = …`).
    kv_cache_bytes: u64,
    /// Audio/vision projector (P4.3). None when no `--mmproj`.
    #[cfg(feature = "mtmd")]
    pub(crate) mtmd: Option<llama_cpp_2::mtmd::MtmdContext>,
}

/// Process-global backend, initialized once and leaked: `LlamaBackend::drop`
/// calls `llama_backend_free`, so per-model ownership would let one model's
/// drop kill every other model's backend.
pub(crate) fn global_backend() -> Result<&'static LlamaBackend, EngineError> {
    use std::sync::OnceLock;
    static BACKEND: OnceLock<Result<LlamaBackend, String>> = OnceLock::new();
    BACKEND
        .get_or_init(|| LlamaBackend::init().map_err(|e| format!("{e:?}")))
        .as_ref()
        .map_err(|msg| EngineError::BackendInit(msg.clone()))
}

impl LoadedModel {
    /// Total size of all tensors in bytes (`llama_model_size`).
    /// P2.1 check: matches the P1 weight estimate within ±5 %.
    pub fn size_bytes(&self) -> u64 {
        self.model.size()
    }

    /// llama.cpp session-blob size (`llama_state_get_size`). Empty KV does
    /// not shrink this when cache type changes; use [`Self::kv_cache_bytes`].
    pub fn state_size(&self) -> usize {
        self.context.get_state_size()
    }

    /// Allocated KV-cache bytes reported by llama.cpp at context create.
    pub fn kv_cache_bytes(&self) -> u64 {
        self.kv_cache_bytes
    }

    /// Parameter count (`llama_model_n_params`).
    pub fn n_params(&self) -> u64 {
        self.model.n_params()
    }

    /// Layer count (`llama_model_n_layer`).
    pub fn n_layer(&self) -> u32 {
        self.model.n_layer()
    }

    /// Borrow the underlying model (sampling loop lands in P2.2).
    pub fn model(&self) -> &LlamaModel {
        &self.model
    }

    /// Borrow the context (sampling loop lands in P2.2).
    pub fn context(&self) -> &LlamaContext<'static> {
        &self.context
    }

    /// Mutably borrow the context for prefill/decode (P2.2 generation).
    pub fn context_mut(&mut self) -> &mut LlamaContext<'static> {
        &mut self.context
    }

    /// Drop KV cells so the next [`generate`](Self::generate) is independent.
    /// Used by `runa serve` between HTTP requests (P3.9).
    pub fn clear_kv(&mut self) {
        self.context.clear_kv_cache();
    }

    /// Drop and recreate the context (fresh KV cache) with the same config.
    /// Needed between generations and chat turns (P2.3); prompt-cache
    /// save/restore builds on this in P2.8.
    pub fn reset_context(&mut self) -> Result<(), EngineError> {
        let params = context_params(&self.config);
        let context = self
            .model
            .new_context(global_backend()?, params)
            .map_err(|e| EngineError::ContextFailed {
                n_ctx: self.config.n_ctx,
                msg: format!("{e:?}"),
            })?;
        // A fresh context carries no adapters: re-attach each one at its
        // spec scale so chat turns and retries keep the LoRA blend (P8.5).
        for (adapter, spec) in self._loras.iter_mut().zip(self.config.loras.iter()) {
            context
                .lora_adapter_set(adapter, spec.scale)
                .map_err(|e| EngineError::Lora {
                    path: spec.path.clone(),
                    msg: format!("re-apply after context reset: {e:?}"),
                })?;
        }
        // SAFETY: as in `load` — `model` outlives `context` inside `LoadedModel`.
        self.context =
            unsafe { std::mem::transmute::<LlamaContext<'_>, LlamaContext<'static>>(context) };
        Ok(())
    }

    /// The placement this model was loaded with.
    pub fn placement(&self) -> &Placement {
        &self.placement
    }

    /// The load config (context sizes, threads, mapping flags).
    pub fn config(&self) -> &LoadConfig {
        &self.config
    }

    /// Model file path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Attach an LMDB prompt cache. Subsequent [`generate`](crate::LoadedModel::generate)
    /// calls save/restore KV state keyed by the prompt prefix.
    pub fn attach_prompt_cache(&mut self, cache: crate::prompt_cache::PromptCache) {
        self.prompt_cache = Some(cache);
    }

    /// Whether the most recent [`generate`](Self::generate) restored from LMDB.
    pub fn last_prompt_cache_hit(&self) -> bool {
        self.last_cache_hit
    }

    /// Unmap the LMDB environment (D17 / P7.2). Files stay on disk.
    pub fn on_idle(&mut self) {
        // Never unload the active model (P7.2). Encoder/draft pools land in P4/P6.
        eprintln!("memory: on_idle: releasing prompt cache (model kept)");
        if let Some(cache) = self.prompt_cache.as_mut() {
            cache.on_idle();
        }
    }

    /// Restore KV state for `tokens` when a cache entry exists.
    pub(crate) fn try_restore_prompt(&mut self, tokens: &[llama_cpp_2::token::LlamaToken]) -> bool {
        self.last_cache_hit = false;
        let Some(cache) = self.prompt_cache.as_mut() else {
            return false;
        };
        let key =
            crate::prompt_cache::prefix_key(&self.path, &self.placement, &self.config, tokens);
        let bytes = match cache.get(&key) {
            Ok(Some(b)) => b,
            Ok(None) => return false,
            Err(e) => {
                eprintln!("prompt-cache: lookup failed ({e})");
                return false;
            }
        };
        if let Err(e) = self.reset_context() {
            eprintln!("prompt-cache: restore reset failed ({e})");
            return false;
        }
        // SAFETY: `bytes` is a blob previously produced by `copy_state_data`
        // for this model/context size; llama.cpp reads it as an opaque state.
        let n = unsafe { self.context.set_state_data(&bytes) };
        if n == 0 {
            eprintln!("prompt-cache: restore read 0 bytes");
            return false;
        }
        eprintln!("prompt-cache: hit");
        self.last_cache_hit = true;
        true
    }

    /// Persist current context state under the prompt-prefix key.
    pub(crate) fn store_prompt(&mut self, tokens: &[llama_cpp_2::token::LlamaToken]) {
        if self.prompt_cache.is_none() {
            return;
        }
        let key =
            crate::prompt_cache::prefix_key(&self.path, &self.placement, &self.config, tokens);
        let size = self.context.get_state_size();
        if size == 0 {
            return;
        }
        let mut buf = vec![0u8; size];
        // SAFETY: `buf` is `get_state_size()` bytes, as required by llama.cpp.
        let n = unsafe { self.context.copy_state_data(buf.as_mut_ptr()) };
        buf.truncate(n);
        if buf.is_empty() {
            return;
        }
        if let Some(cache) = self.prompt_cache.as_mut() {
            match cache.put(&key, &buf) {
                Ok(()) => eprintln!("prompt-cache: store"),
                Err(e) => eprintln!("prompt-cache: store failed ({e})"),
            }
        }
    }
}

/// Default worker threads (P10.2, M4 follow-up).
///
/// llama.cpp's own default (`cpu_get_num_math`) counts performance cores on
/// Apple Silicon and all logical CPUs elsewhere; matching it keeps `runa`
/// within noise of `llama-bench` auto-threading instead of spreading decode
/// over E-cores. macOS reads `hw.perflevel0.logicalcpu` (P-cores) via
/// sysctl; anything unreadable falls back to logical CPUs, floor 1.
pub(crate) fn default_threads() -> i32 {
    let logical = std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(4);
    #[cfg(target_os = "macos")]
    {
        apple_pcore_threads().unwrap_or(logical).max(1)
    }
    #[cfg(not(target_os = "macos"))]
    {
        logical.max(1)
    }
}

/// `hw.perflevel0.logicalcpu` via `sysctlbyname(3)`. `None` on any failure
/// (missing key on Intel Macs without perf levels, short read, absurd
/// value) — the caller falls back to logical CPUs.
#[cfg(target_os = "macos")]
fn apple_pcore_threads() -> Option<i32> {
    let name = c"hw.perflevel0.logicalcpu";
    let mut count: i32 = 0;
    let mut len = std::mem::size_of::<i32>() as libc::size_t;
    // SAFETY: `name` is a NUL-terminated static; `count`/`len` are valid
    // writable out-params for exactly `len` bytes; the call writes at most
    // `size_of::<i32>` bytes on success.
    let rc = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            (&raw mut count).cast::<libc::c_void>(),
            &raw mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc != 0 || len != std::mem::size_of::<i32>() as libc::size_t {
        return None;
    }
    (1..=1024).contains(&count).then_some(count)
}

/// One-line verdict printed before load (plan D12): placement + memory.
fn verdict_line(path: &Path, placement: &Placement, config: &LoadConfig) -> String {
    let gpu = if placement.n_gpu_layers == 0 {
        "cpu (0 layers on GPU)".to_owned()
    } else if placement.n_gpu_layers == u32::MAX {
        "gpu (all layers)".to_owned()
    } else {
        format!("hybrid ({} layers on GPU)", placement.n_gpu_layers)
    };
    let experts = if placement.cpu_patterns.is_empty() {
        String::new()
    } else {
        format!(" experts-on-cpu:{}", placement.cpu_patterns.join(","))
    };
    let kv = match (config.kv_k, config.kv_v) {
        (None, None) => String::new(),
        (k, v) => format!(
            " kv k={} v={}",
            k.unwrap_or(KvKind::F16).as_str(),
            v.unwrap_or(KvKind::F16).as_str(),
        ),
    };
    let devices = if placement.devices.is_empty() {
        String::new()
    } else {
        format!(" devices={}", placement.devices.join(","))
    };
    let split = if placement.tensor_split.is_empty() {
        String::new()
    } else {
        let parts: Vec<String> = placement
            .tensor_split
            .iter()
            .map(|v| format!("{v}"))
            .collect();
        format!(" tensor-split={}", parts.join(","))
    };
    let rpc = if placement.rpc_servers.is_empty() {
        String::new()
    } else {
        format!(" rpc={}", placement.rpc_servers.join(","))
    };
    let threads = config.threads.unwrap_or_else(default_threads);
    let loras = if config.loras.is_empty() {
        String::new()
    } else {
        let specs: Vec<String> = config.loras.iter().map(|s| s.to_string()).collect();
        format!(" lora {}", specs.join(","))
    };
    format!(
        "runa load {} · {gpu}{experts}{kv}{devices}{split}{rpc}{loras} · ctx {} batch {}/{} threads {threads} mmap:{} mlock:{}",
        path.display(),
        config.n_ctx,
        config.n_batch,
        config.n_ubatch,
        config.mmap,
        config.mlock,
    )
}

fn format_backend_devices(devs: &[llama_cpp_2::LlamaBackendDevice]) -> String {
    devs.iter()
        .map(|d| format!("{}:{} ({})", d.index, d.name, d.backend))
        .collect::<Vec<_>>()
        .join(", ")
}

fn resolve_device_spec(spec: &str) -> Result<usize, EngineError> {
    let devs = llama_cpp_2::list_llama_ggml_backend_devices();
    if let Ok(i) = spec.parse::<usize>() {
        if i < devs.len() {
            return Ok(i);
        }
        return Err(EngineError::BadDevices(format!(
            "device index {i} out of range; known: {}",
            format_backend_devices(&devs)
        )));
    }
    let lower = spec.to_ascii_lowercase();
    if let Some(d) = devs
        .iter()
        .find(|d| d.name.eq_ignore_ascii_case(spec) || d.description.to_ascii_lowercase() == lower)
    {
        return Ok(d.index);
    }
    Err(EngineError::BadDevices(format!(
        "unknown device {spec:?}; known: {}",
        format_backend_devices(&devs)
    )))
}

/// The C `llama_model_params` inside llama-cpp-2's wrapper, for fields
/// 0.1.133 has no setter for (`tensor_split`, the progress callback).
/// `LlamaModelParams` is not `repr(C)`: the C struct is not its first field
/// (byte 48 on rustc 1.98), so the offset is found once from two sentinel
/// values set through the safe setters. `None` if a new layout hides them.
fn raw_params(
    params: std::pin::Pin<&mut LlamaModelParams>,
) -> Option<*mut llama_cpp_sys_2::llama_model_params> {
    use llama_cpp_sys_2::llama_model_params as C;
    use std::mem::{align_of, offset_of, size_of};
    static OFFSET: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    let off = (*OFFSET.get_or_init(|| {
        const GPU: i32 = 0x3C3C_5A5A;
        const MAIN: i32 = 0x1234_5678;
        let probe = LlamaModelParams::default()
            .with_n_gpu_layers(GPU as u32)
            .with_main_gpu(MAIN);
        let base = (&raw const probe).cast::<u8>();
        // SAFETY: every read is an in-bounds, unaligned i32 of `probe`.
        let at = |o: usize| unsafe { base.add(o).cast::<i32>().read_unaligned() };
        (0..=size_of::<LlamaModelParams>() - size_of::<C>())
            .step_by(align_of::<C>())
            .find(|&o| {
                at(o + offset_of!(C, n_gpu_layers)) == GPU
                    && at(o + offset_of!(C, main_gpu)) == MAIN
            })
    }))?;
    // SAFETY: one type, one layout: the C struct sits at `off` in every
    // `LlamaModelParams`; callers write plain fields and move nothing.
    Some(unsafe {
        std::ptr::from_mut(params.get_unchecked_mut())
            .cast::<u8>()
            .add(off)
            .cast()
    })
}

fn apply_tensor_split(
    params: std::pin::Pin<&mut LlamaModelParams>,
    split: &[f32],
) -> Result<(), EngineError> {
    let raw = raw_params(params).ok_or(EngineError::Unsupported(
        "--tensor-split (unknown llama-cpp-2 params layout)",
    ))?;
    // SAFETY: llama.cpp copies `tensor_split` during load, so `split` only
    // needs to live until `load_from_file` returns.
    unsafe { (*raw).tensor_split = split.as_ptr() };
    Ok(())
}

/// Progress is best effort: an unknown layout just loads without it.
fn apply_progress(params: std::pin::Pin<&mut LlamaModelParams>, progress: &Arc<AtomicU32>) {
    unsafe extern "C" fn on_progress(p: f32, user: *mut std::ffi::c_void) -> bool {
        // SAFETY: `user` is the `AtomicU32` set below, alive for the load.
        let slot = unsafe { &*(user as *const AtomicU32) };
        slot.store((p.clamp(0.0, 1.0) * 1000.0) as u32, Ordering::Relaxed);
        true
    }
    let Some(raw) = raw_params(params) else {
        return;
    };
    // SAFETY: the callback only runs inside `load_from_file`, while `config`
    // (holding the `Arc`) is borrowed by `load`.
    unsafe {
        (*raw).progress_callback = Some(on_progress);
        (*raw).progress_callback_user_data = Arc::as_ptr(progress) as *mut std::ffi::c_void;
    }
}

/// Load a GGUF with a [`Placement`]: mmap/mlock, GPU layers, CPU-buffer
/// overrides, context params (threads, flash-attn AUTO, KV types).
///
/// Backend scope: `LlamaBackend::init` is process-global in the pinned
/// llama-cpp-2 (a second init fails), so one process holds one backend;
/// multi-model serving shares it (P6.1).
pub fn load(
    path: &Path,
    placement: &Placement,
    config: &LoadConfig,
) -> Result<LoadedModel, EngineError> {
    if !path.is_file() {
        return Err(EngineError::ModelNotFound(path.to_owned()));
    }
    // Fail fast on a missing adapter: model load takes seconds, and a
    // typo in `--lora` should not pay that cost (P8.5).
    for spec in &config.loras {
        if !spec.path.is_file() {
            return Err(EngineError::Lora {
                path: spec.path.clone(),
                msg: "adapter file not found".into(),
            });
        }
    }
    let kv_quant = config.kv_k.is_some_and(KvKind::is_quantized)
        || config.kv_v.is_some_and(KvKind::is_quantized);
    if kv_quant && !config.flash_attn {
        return Err(EngineError::Unsupported(
            "quantized KV requires flash attention",
        ));
    }
    if placement.n_gpu_layers == 0
        && (!placement.tensor_split.is_empty() || !placement.devices.is_empty())
    {
        return Err(EngineError::BadDevices(
            "--device/--tensor-split require GPU offload (--mode gpu|hybrid|auto)".into(),
        ));
    }
    eprintln!("{}", verdict_line(path, placement, config));

    // P9.3: RPC endpoints are registered before backend init (mirroring
    // upstream `add_rpc_devices`), so the `RPC0`, … devices enumerate and
    // `--device` can select them. Registration is explicit at every step:
    // without `--device` the remote servers would sit idle while tensors
    // stay local — that silent non-use is an error, not a default.
    #[cfg(feature = "rpc")]
    if !placement.rpc_servers.is_empty() {
        if placement.devices.is_empty() {
            return Err(EngineError::BadDevices(
                "--rpc registers remote servers only; select them with --device RPC0 \
                 (see runa doctor: rpc)"
                    .into(),
            ));
        }
        crate::rpc::register_servers(&placement.rpc_servers)?;
        for endpoint in &placement.rpc_servers {
            match crate::rpc::device_name_for(endpoint) {
                Some(name) => eprintln!("rpc: {endpoint} available as --device {name}"),
                None => eprintln!("rpc: {endpoint} registered (device not enumerated yet)"),
            }
        }
    }
    #[cfg(not(feature = "rpc"))]
    if !placement.rpc_servers.is_empty() {
        return Err(EngineError::Unsupported(
            "--rpc (rebuild with --features rpc; see docs/versions.md)",
        ));
    }

    let backend = global_backend()?;

    // Own the pattern strings first: FFI keeps raw pointers into them.
    let mut owned: Vec<CString> = Vec::with_capacity(placement.cpu_patterns.len());
    for pat in &placement.cpu_patterns {
        owned.push(CString::new(pat.as_str()).map_err(|_| EngineError::BadPattern(pat.clone()))?);
    }

    // 0.1.133 `LlamaModelParams` has no mmap builder (mmap defaults on in
    // llama.cpp); disabling mmap is an explicit error, not silent drift.
    if !config.mmap {
        return Err(EngineError::Unsupported("mmap=false"));
    }

    let mut params = LlamaModelParams::default()
        .with_n_gpu_layers(placement.n_gpu_layers)
        .with_use_mlock(config.mlock)
        .with_main_gpu(placement.main_gpu);
    let resolved: Vec<usize> = placement
        .devices
        .iter()
        .map(|s| resolve_device_spec(s))
        .collect::<Result<_, _>>()?;
    if resolved.len() == 1 {
        params = params.with_split_mode(LlamaSplitMode::None);
    } else if resolved.len() > 1 || placement.tensor_split.len() > 1 {
        params = params.with_split_mode(LlamaSplitMode::Layer);
    }
    if !resolved.is_empty() {
        params = params
            .with_devices(&resolved)
            .map_err(|e| EngineError::BadDevices(e.to_string()))?;
    }
    let max_dev = llama_cpp_2::max_devices().max(1);
    let mut split_owned = vec![0.0f32; max_dev];
    if !placement.tensor_split.is_empty() {
        if placement.tensor_split.len() > max_dev {
            return Err(EngineError::BadDevices(format!(
                "--tensor-split: {} values > max devices {max_dev}",
                placement.tensor_split.len()
            )));
        }
        split_owned[..placement.tensor_split.len()].copy_from_slice(&placement.tensor_split);
    }
    let mut params = Box::pin(params);
    for pat in &owned {
        // SAFETY: `owned` outlives the model (moved into `LoadedModel`).
        params.as_mut().add_cpu_buft_override(pat.as_c_str());
    }
    if !placement.tensor_split.is_empty() {
        apply_tensor_split(params.as_mut(), &split_owned)?;
    }
    if let Some(progress) = &config.progress {
        apply_progress(params.as_mut(), progress);
    }

    let model = Box::new(
        LlamaModel::load_from_file(backend, path, &params).map_err(|e| {
            EngineError::LoadFailed {
                path: path.to_owned(),
                msg: format!("{e:?}"),
            }
        })?,
    );

    let (context, kv_cache_bytes) = capture_kv_log(|| {
        model
            .new_context(backend, context_params(config))
            .map_err(|e| EngineError::ContextFailed {
                n_ctx: config.n_ctx,
                msg: format!("{e:?}"),
            })
    });
    let context = context?;

    // LoRA adapters (P8.5): init each file against the model, then attach
    // to the context at its scale. The context only borrows the adapters,
    // so they move into `LoadedModel` alongside it.
    let mut adapters = Vec::with_capacity(config.loras.len());
    for spec in &config.loras {
        let mut adapter = model
            .lora_adapter_init(&spec.path)
            .map_err(|e| EngineError::Lora {
                path: spec.path.clone(),
                msg: format!("{e:?}"),
            })?;
        context
            .lora_adapter_set(&mut adapter, spec.scale)
            .map_err(|e| EngineError::Lora {
                path: spec.path.clone(),
                msg: format!("set scale {}: {e:?}", spec.scale),
            })?;
        adapters.push(adapter);
    }

    #[cfg(feature = "mtmd")]
    let mtmd =
        crate::media::load_mtmd(config.mmproj.as_deref(), &model, placement.n_gpu_layers > 0)?;
    #[cfg(not(feature = "mtmd"))]
    crate::media::load_mtmd(config.mmproj.as_deref(), &model, false)?;

    // Transmute the context lifetime: it borrows the boxed model, whose heap
    // address survives every move of `LoadedModel`.
    // SAFETY: the box is never replaced, and `context` drops first.
    let context: LlamaContext<'static> =
        unsafe { std::mem::transmute::<LlamaContext<'_>, LlamaContext<'static>>(context) };

    Ok(LoadedModel {
        context,
        model,
        placement: placement.clone(),
        config: config.clone(),
        path: path.to_owned(),
        _cpu_patterns: owned,
        _loras: adapters,
        prompt_cache: None,
        last_cache_hit: false,
        kv_cache_bytes,
        #[cfg(feature = "mtmd")]
        mtmd,
    })
}

/// Build context params from a [`LoadConfig`] (shared by `load` and
/// [`LoadedModel::reset_context`]).
fn context_params(config: &LoadConfig) -> LlamaContextParams {
    let threads = config.threads.unwrap_or_else(default_threads);
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(config.n_ctx.try_into().ok())
        .with_n_batch(config.n_batch)
        .with_n_ubatch(config.n_ubatch)
        .with_n_threads(threads)
        .with_n_threads_batch(threads)
        .with_flash_attention_policy(if config.flash_attn {
            llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_AUTO
        } else {
            llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_DISABLED
        });
    match (config.kv_k, config.kv_v) {
        (None, None) => ctx_params,
        (k, v) => {
            let mut p = ctx_params;
            if let Some(k) = k {
                p = p.with_type_k(k.as_llama());
            }
            if let Some(v) = v {
                p = p.with_type_v(v.as_llama());
            }
            p
        }
    }
}

#[cfg(test)]
mod threads_tests {
    use super::default_threads;

    #[test]
    fn default_threads_is_sane() {
        let t = default_threads();
        let logical = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(4);
        assert!(t >= 1, "at least one worker: {t}");
        // P-cores never exceed logical CPUs; elsewhere it equals them.
        assert!(t <= logical.max(1), "{t} vs logical {logical}");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn apple_pcore_reading_is_plausible() {
        // Apple Silicon exposes perf levels; Intel Macs return None and
        // fall back to logical CPUs — both are acceptable here.
        if let Some(p) = super::apple_pcore_threads() {
            let logical = std::thread::available_parallelism()
                .map(|n| n.get() as i32)
                .unwrap_or(4);
            assert!((1..=logical).contains(&p), "{p} vs logical {logical}");
        }
    }
}

#[cfg(test)]
mod raw_params_tests {
    use super::{LlamaModelParams, raw_params};

    /// The raw pointer sees what the safe setters wrote (the old
    /// "first field" cast wrote `tensor_split` into the wrapper's Vecs).
    #[test]
    fn raw_params_points_at_the_c_struct() {
        let mut p = Box::pin(
            LlamaModelParams::default()
                .with_n_gpu_layers(7)
                .with_main_gpu(3),
        );
        let raw = raw_params(p.as_mut()).expect("layout found");
        // SAFETY: `raw` points into `p`, alive and pinned here.
        let (layers, main) = unsafe { ((*raw).n_gpu_layers, (*raw).main_gpu) };
        assert_eq!((layers, main), (7, 3));
    }
}

#[cfg(test)]
mod kv_kind_tests {
    use super::{KvKind, planner_kv_type};

    #[test]
    fn parse_aliases() {
        assert_eq!("q8_0".parse::<KvKind>().unwrap(), KvKind::Q8_0);
        assert_eq!("Q8".parse::<KvKind>().unwrap(), KvKind::Q8_0);
        assert_eq!("fp16".parse::<KvKind>().unwrap(), KvKind::F16);
        assert_eq!("q4_0".parse::<KvKind>().unwrap(), KvKind::Q4_0);
        assert!("int8".parse::<KvKind>().is_err());
    }

    #[test]
    fn planner_picks_larger_of_mixed() {
        assert_eq!(planner_kv_type(None, None), "f16");
        assert_eq!(
            planner_kv_type(Some(KvKind::Q8_0), Some(KvKind::Q8_0)),
            "q8_0"
        );
        assert_eq!(
            planner_kv_type(Some(KvKind::Q4_0), Some(KvKind::F16)),
            "f16"
        );
        assert_eq!(
            planner_kv_type(Some(KvKind::Q4_0), Some(KvKind::Q8_0)),
            "q8_0"
        );
    }

    #[test]
    fn parse_kv_cache_size_line() {
        let log = "llama_kv_cache: size =   25.50 MiB (  4096 cells,  24 layers,  1/1 seqs), K (q8_0):   12.75 MiB, V (q8_0):   12.75 MiB\n";
        assert_eq!(
            super::parse_kv_cache_bytes(log),
            Some((25.5_f64 * 1024.0 * 1024.0).round() as u64)
        );
        assert_eq!(super::parse_kv_cache_bytes("nope"), None);
    }

    #[test]
    fn verdict_line_lists_lora_adapters() {
        use super::{LoadConfig, Placement, verdict_line};
        use crate::lora::LoraSpec;
        use std::path::PathBuf;

        let plain = verdict_line(
            PathBuf::from("m.gguf").as_path(),
            &Placement::cpu(),
            &LoadConfig::default(),
        );
        assert!(!plain.contains("lora"), "{plain}");

        let with_lora = verdict_line(
            PathBuf::from("m.gguf").as_path(),
            &Placement::cpu(),
            &LoadConfig {
                loras: vec![
                    LoraSpec {
                        path: PathBuf::from("a.gguf"),
                        scale: 1.0,
                    },
                    LoraSpec {
                        path: PathBuf::from("b.gguf"),
                        scale: 0.5,
                    },
                ],
                ..LoadConfig::default()
            },
        );
        assert!(
            with_lora.contains("lora a.gguf:1,b.gguf:0.5"),
            "{with_lora}"
        );
    }

    #[test]
    fn verdict_line_lists_rpc_servers() {
        use super::{LoadConfig, Placement, verdict_line};
        use std::path::PathBuf;

        let plain = verdict_line(
            PathBuf::from("m.gguf").as_path(),
            &Placement::cpu(),
            &LoadConfig::default(),
        );
        assert!(!plain.contains("rpc="), "{plain}");

        let with_rpc = verdict_line(
            PathBuf::from("m.gguf").as_path(),
            &Placement::gpu().with_rpc_servers(vec!["127.0.0.1:50052".into()]),
            &LoadConfig::default(),
        );
        assert!(with_rpc.contains("rpc=127.0.0.1:50052"), "{with_rpc}");
    }
}
