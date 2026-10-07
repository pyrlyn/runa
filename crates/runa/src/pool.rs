// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! Shared engine pool (P9.1): `ModelPool`, `EngineJob`, `spawn_engine`,
//! and the P8.8 `Warmup`, moved verbatim from `serve.rs` so `serve` and
//! `daemon` share one loader. No behavior change.

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use runa_core::BackendKind;
use runa_engine::{GenEvent, GenerateRequest, LoadConfig, Placement};
use runa_fit::{
    Descriptor, FitConfig, HwSpec, PlannerConfig, Reader, check_fit, read_local_prefix,
};
use tokio::sync::oneshot;

/// Startup load of the default model, read by `/health` (P8.8) and by the
/// daemon's ready line.
pub(crate) struct Warmup {
    pub model: String,
    /// Per mille, written by llama.cpp's load callback.
    pub progress: Arc<AtomicU32>,
    /// `None` while loading.
    pub state: Mutex<Option<Result<(), String>>>,
}

impl Warmup {
    pub fn new(model: String, progress: Arc<AtomicU32>) -> Self {
        Warmup {
            model,
            progress,
            state: Mutex::new(None),
        }
    }

    pub fn done(&self) -> Option<Result<(), String>> {
        lock(&self.state).clone()
    }
}

pub(crate) fn warm_up(pool: &Mutex<ModelPool>, warm: &Warmup, tag: &str) {
    let t = Instant::now();
    let r = catch_job(|| ensure_engine(pool, &warm.model).map(drop));
    match &r {
        Ok(()) => eprintln!(
            "{tag}: {} ready in {:.1}s",
            warm.model,
            t.elapsed().as_secs_f64()
        ),
        Err(e) => eprintln!("{tag}: loading {} failed: {e}", warm.model),
    }
    *lock(&warm.state) = Some(r);
}

/// `<tag>: loading <id> N%` in 10% steps while the warm-up runs.
pub(crate) async fn report_progress(warm: Arc<Warmup>, tag: &'static str) {
    let mut shown = 0;
    while warm.done().is_none() {
        tokio::time::sleep(Duration::from_millis(250)).await;
        let step = warm.progress.load(Ordering::Relaxed) / 100;
        if step > shown && step < 10 {
            shown = step;
            eprintln!("{tag}: loading {} {}%", warm.model, step * 10);
        }
    }
}

/// A poisoned lock still holds consistent pool state (every mutation is a
/// single insert/remove), so recover it instead of failing every request.
pub(crate) fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

pub(crate) fn panic_text(p: &(dyn std::any::Any + Send)) -> String {
    p.downcast_ref::<&str>()
        .map(|s| (*s).to_owned())
        .or_else(|| p.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "panic".into())
}

/// Run one engine/pool step; a panic becomes an error, not a dead thread.
pub(crate) fn catch_job<T>(f: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f))
        .unwrap_or_else(|p| Err(format!("internal error: {}", panic_text(&*p))))
}

/// One item on a streaming generation channel.
#[derive(Debug)]
pub(crate) enum Streamed {
    /// Prompt length in tokens, sent once before the first event by
    /// backends that know it up front.
    Prompt(u32),
    Event(GenEvent),
}

pub(crate) enum EngineJob {
    Generate {
        req: Box<GenerateRequest>,
        resp: oneshot::Sender<Result<Vec<GenEvent>, String>>,
    },
    /// P16.1: the same generation, delivered event by event. The channel is
    /// bounded, so a slow reader slows the engine; a dropped receiver
    /// cancels the generation. An `Err` item ends the stream.
    GenerateStream {
        req: Box<GenerateRequest>,
        tx: tokio::sync::mpsc::Sender<Result<Streamed, String>>,
    },
    Embed {
        input: String,
        resp: oneshot::Sender<Result<(Vec<f32>, u32), String>>,
    },
    /// Idle tick (P10.5): the engine releases its prompt cache and keeps
    /// the model (`LoadedModel::on_idle`). Fire-and-forget: queued behind
    /// any in-flight request, never fails the tick.
    Idle,
}

pub(crate) struct ModelPool {
    specs: HashMap<String, PathBuf>,
    order: Vec<String>,
    engines: HashMap<String, Arc<std::sync::mpsc::Sender<EngineJob>>>,
    lru: VecDeque<String>,
    max_loaded: usize,
    /// Requested `--backend` (possibly `Auto`; resolved per model path).
    backend: BackendKind,
    placement_base: Placement,
    /// CLI placement overrides (`--device/--tensor-split/--main-gpu/--rpc`,
    /// P2.9/P9.3): applied on top of both fixed and `auto` placements so an
    /// explicit flag is never dropped silently (plan D12).
    overrides: PlacementOverrides,
    mode: String,
    config: LoadConfig,
    /// Last request per loaded engine, for the idle sweep (P10.5).
    last_used: HashMap<String, Instant>,
    /// Models whose load is in flight. Waiters block on the condvar, not
    /// on the pool mutex, so a multi-GB load does not stall other models.
    loading: HashMap<String, Arc<LoadWait>>,
    /// Engines unused for this long get an `EngineJob::Idle`
    /// (prompt cache released, model kept). `Duration::MAX` disables.
    idle_timeout: Duration,
    /// CLI `--max-load-percent` for the demand-aware startup cap check
    /// (warn-only; `None` = resolve from env/config, default 80).
    max_load_percent: Option<u8>,
}

/// Parsed `--device/--tensor-split/--main-gpu/--rpc` overrides for
/// `serve` (and the daemon later): empty = no overrides.
#[derive(Debug, Clone, Default)]
pub(crate) struct PlacementOverrides {
    pub devices: Vec<String>,
    pub tensor_split: Vec<f32>,
    pub main_gpu: Option<i32>,
    pub rpc_servers: Vec<String>,
}

impl PlacementOverrides {
    pub fn apply(&self, mut placement: Placement) -> Placement {
        if !self.devices.is_empty() {
            placement = placement.with_devices(self.devices.clone());
        }
        if !self.tensor_split.is_empty() {
            placement = placement.with_tensor_split(self.tensor_split.clone());
        }
        if let Some(n) = self.main_gpu {
            placement = placement.with_main_gpu(n);
        }
        if !self.rpc_servers.is_empty() {
            placement = placement.with_rpc_servers(self.rpc_servers.clone());
        }
        placement
    }
}

impl ModelPool {
    pub(crate) fn new(
        models: Vec<(String, PathBuf)>,
        backend: BackendKind,
        placement_base: Placement,
        mode: String,
        config: LoadConfig,
        max_loaded: usize,
    ) -> Result<Self, String> {
        let mut specs = HashMap::new();
        let mut order = Vec::new();
        for (id, path) in models {
            if !path.is_file() && !path.is_dir() {
                return Err(format!(
                    "no such model file or directory: {}",
                    path.display()
                ));
            }
            specs.insert(id.clone(), path);
            order.push(id);
        }
        Ok(ModelPool {
            specs,
            order,
            engines: HashMap::new(),
            lru: VecDeque::new(),
            max_loaded: max_loaded.max(1),
            backend,
            placement_base,
            overrides: PlacementOverrides::default(),
            mode,
            config,
            last_used: HashMap::new(),
            loading: HashMap::new(),
            idle_timeout: Duration::MAX,
            max_load_percent: None,
        })
    }

    /// Idle-sweep timeout from the memory policy (serve + daemon set it;
    /// unit tests keep the disabled default).
    pub(crate) fn with_idle_timeout(mut self, timeout: Duration) -> Self {
        self.idle_timeout = timeout;
        self
    }

    /// Attach CLI placement overrides (serve `--device/…/--rpc`); the daemon
    /// keeps the default (empty).
    pub(crate) fn with_overrides(mut self, overrides: PlacementOverrides) -> Self {
        self.overrides = overrides;
        self
    }

    /// Carry CLI `--max-load-percent` into the per-load demand check so a
    /// flag is never silently replaced by the config default there.
    pub(crate) fn with_max_load_percent(mut self, max_load_percent: Option<u8>) -> Self {
        self.max_load_percent = max_load_percent;
        self
    }

    pub(crate) fn model_ids(&self) -> &[String] {
        &self.order
    }

    pub(crate) fn resolve_id(&self, model: Option<&str>) -> Result<String, String> {
        let id = model
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or(self.order.first().ok_or("no models configured")?);
        if self.specs.contains_key(id) {
            Ok(id.to_owned())
        } else {
            Err(format!("model {id} not found"))
        }
    }

    /// Register one more model (file or directory) under a stem id
    /// (daemon on-demand serving of paths it was not started with).
    /// No-op when the id or an equal path is already known. Returns the id.
    pub(crate) fn insert_spec(&mut self, path: &Path) -> Result<String, String> {
        if !path.is_file() && !path.is_dir() {
            return Err(format!(
                "no such model file or directory: {}",
                path.display()
            ));
        }
        if let Some((id, _)) = self.specs.iter().find(|(_, p)| *p == path) {
            return Ok(id.clone());
        }
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("runa")
            .to_owned();
        let mut id = stem.clone();
        let mut n = 1;
        while self.specs.contains_key(&id) {
            n += 1;
            id = format!("{stem}-{n}");
        }
        self.specs.insert(id.clone(), path.to_owned());
        self.order.push(id.clone());
        Ok(id)
    }

    fn touch_lru(&mut self, id: &str) {
        self.lru.retain(|x| x != id);
        self.lru.push_back(id.to_owned());
        self.last_used.insert(id.to_owned(), Instant::now());
    }

    fn evict_if_needed(&mut self) {
        while self.engines.len() > self.max_loaded {
            let victim = self
                .lru
                .front()
                .cloned()
                .filter(|id| self.engines.contains_key(id));
            let Some(id) = victim else {
                break;
            };
            self.lru.pop_front();
            self.last_used.remove(&id);
            if let Some(tx) = self.engines.remove(&id) {
                drop(tx);
                eprintln!("serve: unloaded model {id} (LRU)");
            }
        }
    }

    /// Ids whose engines have seen no request for `idle_timeout` (P10.5).
    /// Pure decision: unit-testable without spawning engines.
    pub(crate) fn due_for_idle(&self, now: Instant) -> Vec<String> {
        if self.idle_timeout == Duration::MAX {
            return Vec::new();
        }
        self.engines
            .keys()
            .filter(|id| {
                self.last_used
                    .get(*id)
                    .is_none_or(|t| now.duration_since(*t) >= self.idle_timeout)
            })
            .cloned()
            .collect()
    }

    /// Send `EngineJob::Idle` to every due engine and re-stamp it, so the
    /// next tick leaves it alone for another full timeout. Returns the
    /// swept ids for logging. May block briefly while an engine finishes
    /// its current request (the job queues behind it) — call from a
    /// blocking thread, never the async hot path.
    pub(crate) fn idle_sweep(&mut self) -> Vec<String> {
        let now = Instant::now();
        let due = self.due_for_idle(now);
        let mut swept = Vec::new();
        for id in due {
            let Some(tx) = self.engines.get(&id) else {
                continue;
            };
            // A dead engine thread drops the receiver: forget the id.
            if tx.send(EngineJob::Idle).is_err() {
                self.engines.remove(&id);
                self.last_used.remove(&id);
                continue;
            }
            self.last_used.insert(id.clone(), now);
            swept.push(id);
        }
        swept
    }

    fn placement_for(&self, path: &Path) -> Result<Placement, String> {
        match crate::parse_mode_choice(&self.mode)? {
            crate::ModeChoice::Auto => match crate::auto_placement(
                path,
                self.config.n_ctx,
                &crate::config::OnUnfit::Error,
                runa_engine::planner_kv_type(self.config.kv_k, self.config.kv_v),
                None,
                None,
                &self.config.loras,
            )? {
                crate::AutoPlacement::Local(p) => Ok(self.overrides.apply(p)),
                crate::AutoPlacement::Cloud(_) => {
                    Err("unfit: auto mode chose cloud fallback; serve is local-only".into())
                }
            },
            crate::ModeChoice::Fixed(_) => {
                crate::preflight_grow(
                    path,
                    self.config.n_ctx,
                    runa_engine::planner_kv_type(self.config.kv_k, self.config.kv_v),
                    0,
                    self.max_load_percent,
                )?;
                Ok(self.overrides.apply(self.placement_base.clone()))
            }
        }
    }

    fn fit_check_no_fit(path: &Path, ctx: u32, lora_bytes: u64) -> Result<(), String> {
        let header = read_local_prefix(path).map_err(|e| e.to_string())?;
        let reader = Reader::parse(&header.bytes).map_err(|e| e.to_string())?;
        let desc = Descriptor::from_reader(&reader).map_err(|e| e.to_string())?;
        let (vram, _) = crate::vram_bytes()?;
        let planner = PlannerConfig {
            vram_bytes: vram,
            ram_bytes: crate::ram_bytes(),
            ctx_len: u64::from(ctx),
            kv_type: runa_engine::planner_kv_type(None, None).to_owned(),
            lora_bytes,
            ..PlannerConfig::default()
        };
        let report = check_fit(
            &desc,
            &FitConfig {
                planner,
                gpu_hw: None,
                cpu_hw: HwSpec::cpu(),
                has_mmproj: false,
                media: runa_fit::MediaFit::default(),
            },
        );
        if matches!(report.verdict, runa_fit::Verdict::NoFit) {
            return Err(format!("unfit: model does not fit (ctx={ctx})"));
        }
        Ok(())
    }

    /// Adapter bytes summed from the configured `--lora` files (P8.5).
    fn lora_bytes(config: &LoadConfig) -> u64 {
        config
            .loras
            .iter()
            .map(|s| runa_fit::mmproj_file_bytes(&s.path))
            .sum()
    }

    fn begin_load(&mut self, id: &str) -> Result<LoadPoll, String> {
        if let Some(tx) = self.engines.get(id).cloned() {
            self.touch_lru(id);
            return Ok(LoadPoll::Ready(tx));
        }
        if let Some(wait) = self.loading.get(id) {
            return Ok(LoadPoll::Wait(Arc::clone(wait)));
        }
        let path = self
            .specs
            .get(id)
            .ok_or_else(|| format!("model {id} not found"))?
            .clone();
        let kind = crate::engine::resolve_requested(self.backend, &path)?;
        // Fit, placement and LoRA are ggml concepts; the mistral backend
        // manages devices and KV itself.
        let (placement, kind) = match kind {
            BackendKind::Gguf => {
                Self::fit_check_no_fit(&path, self.config.n_ctx, Self::lora_bytes(&self.config))?;
                (self.placement_for(&path)?, BackendKind::Gguf)
            }
            BackendKind::Mistral => {
                if !self.config.loras.is_empty() {
                    return Err("--lora needs the gguf backend".into());
                }
                (Placement::cpu(), BackendKind::Mistral)
            }
            BackendKind::Auto => {
                return Err("internal error: backend was not resolved".into());
            }
        };
        self.loading
            .insert(id.to_owned(), Arc::new(LoadWait::new()));
        Ok(LoadPoll::Start(Box::new(LoadStart {
            id: id.to_owned(),
            path,
            kind,
            placement,
            config: self.config.clone(),
        })))
    }

    fn complete_load(
        &mut self,
        id: &str,
        outcome: Result<std::sync::mpsc::Sender<EngineJob>, String>,
    ) -> Result<Arc<std::sync::mpsc::Sender<EngineJob>>, String> {
        let wait = self.loading.remove(id);
        let result = outcome.map(Arc::new);
        if let Ok(tx) = &result {
            self.engines.insert(id.to_owned(), Arc::clone(tx));
            self.touch_lru(id);
            self.evict_if_needed();
        }
        if let Some(wait) = wait {
            wait.finish(result.clone());
        }
        result
    }
}

struct LoadWait {
    state: Mutex<Option<Result<Arc<std::sync::mpsc::Sender<EngineJob>>, String>>>,
    cv: Condvar,
}

impl LoadWait {
    fn new() -> Self {
        Self {
            state: Mutex::new(None),
            cv: Condvar::new(),
        }
    }

    fn finish(&self, result: Result<Arc<std::sync::mpsc::Sender<EngineJob>>, String>) {
        *lock(&self.state) = Some(result);
        self.cv.notify_all();
    }

    fn wait(&self) -> Result<Arc<std::sync::mpsc::Sender<EngineJob>>, String> {
        let mut guard = lock(&self.state);
        loop {
            if let Some(result) = guard.clone() {
                return result;
            }
            guard = self.cv.wait(guard).unwrap_or_else(PoisonError::into_inner);
        }
    }
}

struct LoadStart {
    id: String,
    path: PathBuf,
    kind: BackendKind,
    placement: Placement,
    config: LoadConfig,
}

enum LoadPoll {
    Ready(Arc<std::sync::mpsc::Sender<EngineJob>>),
    Wait(Arc<LoadWait>),
    Start(Box<LoadStart>),
}

/// Resolve or load `id`. The pool mutex is not held across `spawn_engine`,
/// so a load of one model does not block requests to models already loaded.
pub(crate) fn ensure_engine(
    pool: &Mutex<ModelPool>,
    id: &str,
) -> Result<Arc<std::sync::mpsc::Sender<EngineJob>>, String> {
    let start = {
        let mut guard = lock(pool);
        match guard.begin_load(id)? {
            LoadPoll::Ready(tx) => return Ok(tx),
            LoadPoll::Wait(wait) => {
                drop(guard);
                return wait.wait();
            }
            LoadPoll::Start(start) => *start,
        }
    };
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        spawn_engine(start.path, start.kind, start.placement, start.config)
    }))
    .unwrap_or_else(|p| Err(format!("internal error: {}", panic_text(&*p))));
    lock(pool).complete_load(&start.id, outcome)
}

/// Drive one generation, handing each item to `emit`; `emit` returns
/// `false` to stop early (the client is gone), and an `Err` item ends it.
fn run_generation(
    engine: &mut crate::engine::LocalEngine,
    used: &mut bool,
    req: GenerateRequest,
    mut emit: impl FnMut(Result<Streamed, String>) -> bool,
) -> Result<(), String> {
    // The ggml backend reuses one context per thread: drop KV cells between
    // requests (P3.9). The mistral backend is stateless (no-op there).
    if *used {
        engine.clear_kv();
    }
    *used = true;
    let (prompt_len, events) = engine.generate_counted(req)?;
    if let Some(n) = prompt_len
        && !emit(Ok(Streamed::Prompt(n)))
    {
        return Ok(());
    }
    for item in events {
        let item = item.map(Streamed::Event).map_err(|e| e.to_string());
        let failed = item.is_err();
        if !emit(item) || failed {
            break;
        }
    }
    Ok(())
}

pub(crate) fn spawn_engine(
    path: PathBuf,
    kind: BackendKind,
    placement: Placement,
    config: LoadConfig,
) -> Result<std::sync::mpsc::Sender<EngineJob>, String> {
    let (tx, rx) = std::sync::mpsc::channel::<EngineJob>();
    let (ready_tx, ready_rx) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name(format!(
            "runa-engine-{}",
            path.file_stem().and_then(|s| s.to_str()).unwrap_or("m")
        ))
        // `runa run` drives the engine on the 8 MiB main thread; match it
        // (the 2 MiB default is tight for llama.cpp's Jinja templates).
        .stack_size(8 << 20)
        .spawn(move || {
            let mut engine =
                match crate::engine::LocalEngine::load(kind, &path, &placement, &config) {
                    Ok(m) => {
                        let _ = ready_tx.send(Ok(()));
                        m
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
            let mut used = false;
            while let Ok(job) = rx.recv() {
                match job {
                    EngineJob::Generate { req, resp } => {
                        let mut events = Vec::new();
                        let mut failed = None;
                        let out = catch_job(|| {
                            run_generation(&mut engine, &mut used, *req, |item| match item {
                                Ok(Streamed::Event(ev)) => {
                                    events.push(ev);
                                    true
                                }
                                Ok(Streamed::Prompt(_)) => true,
                                Err(e) => {
                                    failed = Some(e);
                                    false
                                }
                            })
                        });
                        let _ = resp.send(out.and_then(|()| failed.map_or(Ok(events), Err)));
                    }
                    EngineJob::GenerateStream { req, tx } => {
                        let out = catch_job(|| {
                            run_generation(&mut engine, &mut used, *req, |item| {
                                tx.blocking_send(item).is_ok()
                            })
                        });
                        if let Err(e) = out {
                            let _ = tx.blocking_send(Err(e));
                        }
                    }
                    EngineJob::Embed { input, resp } => {
                        let out = catch_job(|| match &mut engine {
                            crate::engine::LocalEngine::Gguf(loaded) => {
                                loaded.embed(&input).map_err(|e| e.to_string())
                            }
                            #[cfg(feature = "mistralrs")]
                            crate::engine::LocalEngine::Mistral(_) => {
                                Err("mistral backend does not serve /v1/embeddings (gguf only)"
                                    .into())
                            }
                        });
                        let _ = resp.send(out);
                    }
                    EngineJob::Idle => {
                        // Prompt cache released, model kept (P7.2); a
                        // failure here must not kill the engine thread.
                        let _ = catch_job(|| {
                            engine.on_idle();
                            Ok(())
                        });
                    }
                }
            }
        })
        .map_err(|e| e.to_string())?;
    ready_rx
        .recv()
        .map_err(|_| "engine thread stopped during load".to_string())??;
    Ok(tx)
}

/// One idle tick for the serve/daemon loops (P10.5): the manager's shrink
/// decision plus the pool sweep. The blocking sweep runs off-thread;
/// `/health` and `/v1/models` never touch activity, so monitoring polls
/// cannot hold engines awake.
pub(crate) async fn idle_tick(
    pool: &Arc<Mutex<ModelPool>>,
    mm: &Arc<runa_memory::MemoryManager>,
    tag: &'static str,
) {
    mm.maybe_idle();
    let pool = Arc::clone(pool);
    let swept = tokio::task::spawn_blocking(move || lock(&pool).idle_sweep())
        .await
        .unwrap_or_default();
    for id in swept {
        eprintln!("{tag}: idle {id}: prompt cache released (model kept)");
    }
}

/// Resolve `model_id` against the pool and run one generation on its
/// engine thread (blocking load happens on a blocking thread). Shared by
/// the daemon; `serve` keeps its own status-mapped variant.
pub(crate) async fn generate(
    pool: &Arc<Mutex<ModelPool>>,
    model_id: &str,
    req: GenerateRequest,
) -> Result<Vec<GenEvent>, String> {
    let jobs = {
        let pool = Arc::clone(pool);
        let id = model_id.to_owned();
        tokio::task::spawn_blocking(move || ensure_engine(&pool, &id))
            .await
            .map_err(|e| e.to_string())??
    };
    let (resp, rx) = oneshot::channel();
    jobs.send(EngineJob::Generate {
        req: Box::new(req),
        resp,
    })
    .map_err(|_| "engine thread stopped".to_string())?;
    rx.await.map_err(|e| e.to_string())?
}

/// Look up `model` (pool id, or an on-disk path to serve on demand) and
/// return the id to generate with. Inserting a path never unloads models.
pub(crate) fn resolve_or_insert(
    pool: &Mutex<ModelPool>,
    model: Option<&str>,
) -> Result<String, String> {
    let mut pool = lock(pool);
    match pool.resolve_id(model) {
        Ok(id) => Ok(id),
        Err(_) => {
            let raw = model.unwrap_or("").trim();
            let path = Path::new(raw);
            if !raw.is_empty() && path.is_file() {
                pool.insert_spec(path)
            } else {
                Err(format!("model {} not found", model.unwrap_or("")))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tiny_gguf(dir: &std::path::Path, name: &str) -> PathBuf {
        let path = dir.join(name);
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(b"GGUF").unwrap();
        path
    }

    #[test]
    fn insert_spec_reuses_path_and_suffixes_collisions() {
        let dir = std::env::temp_dir().join(format!(
            "runa-pool-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        // Same stem in two dirs → `stem`, `stem-2`.
        let sub = dir.join("sub");
        std::fs::create_dir_all(&sub).unwrap();
        let a = tiny_gguf(&dir, "m.gguf");
        let b = tiny_gguf(&sub, "m.gguf");
        let pool = Mutex::new(
            ModelPool::new(
                vec![],
                BackendKind::Gguf,
                Placement::cpu(),
                "cpu".into(),
                LoadConfig {
                    n_ctx: 512,
                    ..LoadConfig::default()
                },
                2,
            )
            .unwrap(),
        );
        let id_a = lock(&pool).insert_spec(&a).unwrap();
        assert_eq!(id_a, "m");
        assert_eq!(lock(&pool).insert_spec(&a).unwrap(), "m");
        assert_eq!(lock(&pool).insert_spec(&b).unwrap(), "m-2");
        assert_eq!(resolve_or_insert(&pool, Some("m")).unwrap(), "m");
        assert!(resolve_or_insert(&pool, Some("nope")).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn jobs_survive_panics() {
        let r: Result<(), String> = catch_job(|| panic!("boom"));
        assert_eq!(r, Err("internal error: boom".into()));
    }

    #[test]
    fn placement_overrides_apply_on_top_of_any_mode() {
        // P2.9/P9.3: explicit serve flags survive both fixed and `auto`
        // placements (plan D12 — never dropped silently).
        let base = Placement::gpu();
        let full = PlacementOverrides {
            devices: vec!["0".into()],
            tensor_split: vec![3.0, 1.0],
            main_gpu: Some(1),
            rpc_servers: vec!["127.0.0.1:50052".into()],
        }
        .apply(base);
        assert_eq!(full.devices, vec!["0"]);
        assert_eq!(full.tensor_split, vec![3.0, 1.0]);
        assert_eq!(full.main_gpu, 1);
        assert_eq!(full.rpc_servers, vec!["127.0.0.1:50052"]);
        assert_eq!(full.n_gpu_layers, u32::MAX);

        let empty = PlacementOverrides::default().apply(Placement::cpu());
        assert_eq!(empty, Placement::cpu());
    }

    #[test]
    fn pool_rejects_missing_files() {
        let err = ModelPool::new(
            vec![("m".into(), PathBuf::from("/no/such/model.gguf"))],
            BackendKind::Gguf,
            Placement::cpu(),
            "cpu".into(),
            LoadConfig::default(),
            1,
        )
        .err()
        .expect("missing file errors");
        assert!(err.contains("no such model file"), "{err}");
    }

    fn idle_test_pool(timeout: Duration) -> ModelPool {
        ModelPool::new(
            vec![],
            BackendKind::Gguf,
            Placement::cpu(),
            "cpu".into(),
            LoadConfig::default(),
            2,
        )
        .unwrap()
        .with_idle_timeout(timeout)
    }

    /// Register a fake engine thread end (no thread behind it): the
    /// receiver end proves which jobs the sweep sends.
    fn fake_engine(pool: &mut ModelPool, id: &str) -> std::sync::mpsc::Receiver<EngineJob> {
        let (tx, rx) = std::sync::mpsc::channel::<EngineJob>();
        pool.engines.insert(id.to_owned(), Arc::new(tx));
        rx
    }

    #[test]
    fn idle_sweep_fires_once_per_timeout() {
        // P10.5: an unused engine gets exactly one Idle per timeout.
        let mut pool = idle_test_pool(Duration::from_secs(60));
        let rx = fake_engine(&mut pool, "m");
        pool.last_used
            .insert("m".into(), Instant::now() - Duration::from_secs(61));
        assert_eq!(pool.due_for_idle(Instant::now()), ["m"]);
        assert_eq!(pool.idle_sweep(), ["m"]);
        assert!(
            matches!(
                rx.recv_timeout(Duration::from_secs(1)).unwrap(),
                EngineJob::Idle
            ),
            "sweep sends Idle"
        );
        // Re-stamped by the sweep: quiet for another full timeout.
        assert!(pool.due_for_idle(Instant::now()).is_empty());
        assert!(pool.idle_sweep().is_empty());
    }

    #[test]
    fn idle_sweep_disabled_by_default() {
        let mut pool = idle_test_pool(Duration::MAX);
        let _rx = fake_engine(&mut pool, "m");
        assert!(pool.due_for_idle(Instant::now()).is_empty());
        assert!(pool.idle_sweep().is_empty());
    }

    #[test]
    fn idle_sweep_drops_dead_engines() {
        let mut pool = idle_test_pool(Duration::from_secs(0));
        let rx = fake_engine(&mut pool, "m");
        drop(rx);
        assert_eq!(pool.idle_sweep(), Vec::<String>::new());
        assert!(!pool.engines.contains_key("m"), "dead engine forgotten");
    }
}
