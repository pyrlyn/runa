// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! OpenAI-compatible HTTP server (plan P3.9 / P6.1 / D11).
//!
//! Multi-model LRU pool, `--parallel` in-flight cap, `/v1/embeddings`,
//! `/v1/audio/transcriptions`, and multimodal chat `content` parts.
//! The default model loads at startup (P8.8): `/health` answers 503
//! `loading` until it is ready, and a panic answers 500 instead of dropping
//! the connection.

use std::collections::HashMap;
use std::io::Write;
use std::net::SocketAddr;
use std::panic::AssertUnwindSafe;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use axum::Router;
use axum::extract::{DefaultBodyLimit, Multipart, Request, State};
use axum::http::StatusCode;
use axum::middleware::{self, Next};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Json, Response};
use axum::routing::{get, post};
use base64::Engine;
use futures::{FutureExt, Stream, stream};
use runa_core::{BackendKind, Effort, ThinkConfig, ThinkOverrides};
use runa_engine::{
    ChatMessage, GenEvent, GenerateRequest, LoadConfig, Mode, Placement, SamplingConfig,
    StopReason, ToolCall, VisionFrame, VisionSource,
};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, mpsc, oneshot};

use crate::pool::{EngineJob, ModelPool, Streamed, Warmup, lock, panic_text};
use runa_memory::MemoryManager;

/// CLI bundle for `runa serve` (P6.1).
pub(crate) struct ServeOpts {
    pub models: Vec<(String, PathBuf)>,
    /// Requested `--backend` (possibly `Auto`; resolved per model path).
    pub backend: BackendKind,
    pub host: String,
    pub port: u16,
    pub mode: String,
    pub ctx: u32,
    pub parallel: usize,
    pub max_loaded: Option<usize>,
    /// LoRA adapters applied to every served model (P8.5).
    pub loras: Vec<runa_engine::LoraSpec>,
    /// CLI placement overrides (`--device/--tensor-split/--main-gpu/--rpc`).
    pub overrides: crate::pool::PlacementOverrides,
    /// Worker threads for every gguf load (P10.2).
    pub threads: Option<i32>,
    /// CLI `--max-load-percent` for the startup cap check (warn-only).
    pub max_load_percent: Option<u8>,
}

pub(crate) fn cmd_serve(opts: ServeOpts) -> Result<(), String> {
    crate::config::warn_if_over_system_limit(None, opts.threads, opts.max_load_percent)?;
    if opts.models.is_empty() {
        return Err("serve: need at least one model (positional or --models)".into());
    }
    if opts.parallel == 0 {
        return Err("serve: --parallel must be >= 1".into());
    }
    let placement_base = match crate::parse_mode_choice(&opts.mode)? {
        crate::ModeChoice::Fixed(m) => Placement::from_mode(m),
        crate::ModeChoice::Auto => Placement::from_mode(Mode::Cpu),
    };
    let config = LoadConfig {
        n_ctx: opts.ctx,
        loras: opts.loras,
        threads: opts.threads,
        ..LoadConfig::default()
    };
    let max_loaded = opts
        .max_loaded
        .unwrap_or_else(|| opts.models.len().min(opts.parallel).max(1));
    let default_id = opts.models[0].0.clone();
    let backend = opts.backend;
    let overrides = opts.overrides.clone();
    let rt = tokio::runtime::Runtime::new().map_err(|e| e.to_string())?;
    rt.block_on(listen(
        &opts.host,
        opts.port,
        opts.models,
        backend,
        placement_base,
        overrides,
        opts.mode,
        config,
        default_id,
        opts.parallel,
        max_loaded,
        opts.max_load_percent,
    ))
}

#[allow(clippy::too_many_arguments)]
async fn listen(
    host: &str,
    port: u16,
    models: Vec<(String, PathBuf)>,
    backend: BackendKind,
    placement_base: Placement,
    overrides: crate::pool::PlacementOverrides,
    mode: String,
    config: LoadConfig,
    default_id: String,
    parallel: usize,
    max_loaded: usize,
    max_load_percent: Option<u8>,
) -> Result<(), String> {
    let progress = Arc::new(AtomicU32::new(0));
    // Single slot by design: only the startup warm-up reports it, so later LRU reload overwrites go unread.
    let config = LoadConfig {
        progress: Some(Arc::clone(&progress)),
        ..config
    };
    let pool = ModelPool::new(models, backend, placement_base, mode, config, max_loaded)?
        .with_overrides(overrides)
        .with_max_load_percent(max_load_percent);
    let models: Arc<[String]> = pool.model_ids().into();
    let policy = crate::config::resolve_memory_policy()?;
    let tick_secs = policy
        .idle_timeout_s
        .clamp(1, crate::daemon_proto::MAX_IDLE_TICK_SECS);
    let idle_timeout = std::time::Duration::from_secs(policy.idle_timeout_s.max(1));
    let mm = Arc::new(MemoryManager::new(
        policy,
        crate::memory_ceiling_mib(),
        Box::new(runa_memory::SysinfoBackend::new()),
    ));
    let pool = Arc::new(Mutex::new(pool.with_idle_timeout(idle_timeout)));
    let state = AppState {
        pool: Arc::clone(&pool),
        models,
        warm: Arc::new(Warmup::new(default_id.clone(), Arc::clone(&progress))),
        default_id,
        parallel: Arc::new(Semaphore::new(parallel)),
        mm: Arc::clone(&mm),
    };
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/messages", post(anthropic_messages))
        .route("/v1/embeddings", post(embeddings))
        .route("/v1/audio/transcriptions", post(audio_transcriptions))
        .layer(DefaultBodyLimit::max(32 * 1024 * 1024))
        .layer(middleware::from_fn(catch_panic))
        .with_state(state.clone());
    let addr: SocketAddr = format!("{host}:{port}")
        .parse()
        .map_err(|e| format!("bind {host}:{port}: {e}"))?;
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|e| format!("bind {addr}: {e}"))?;
    let bound = listener.local_addr().map_err(|e| e.to_string())?;
    eprintln!("listening on http://{bound}");
    // Requests that arrive meanwhile queue on the pool lock the warm-up holds.
    let (pool, warm) = (Arc::clone(&state.pool), Arc::clone(&state.warm));
    tokio::task::spawn_blocking(move || crate::pool::warm_up(&pool, &warm, "serve"));
    tokio::spawn(crate::pool::report_progress(
        Arc::clone(&state.warm),
        "serve",
    ));
    // P10.5: idle shrink (manager decision + pool sweep) every tick.
    // Generation endpoints touch `mm`; health/models polls do not.
    let (tick_pool, tick_mm) = (Arc::clone(&state.pool), Arc::clone(&state.mm));
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(std::time::Duration::from_secs(tick_secs));
        loop {
            tick.tick().await;
            crate::pool::idle_tick(&tick_pool, &tick_mm, "serve").await;
        }
    });
    axum::serve(listener, app).await.map_err(|e| e.to_string())
}

/// A handler panic answers 500 JSON instead of closing the socket.
async fn catch_panic(req: Request, next: Next) -> Response {
    match AssertUnwindSafe(next.run(req)).catch_unwind().await {
        Ok(resp) => resp,
        Err(p) => {
            let msg = format!("internal error: {}", panic_text(&*p));
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": {"message": msg, "type": "server_error"}})),
            )
                .into_response()
        }
    }
}

#[derive(Clone)]
struct AppState {
    pool: Arc<Mutex<ModelPool>>,
    /// Configured ids (fixed at startup; `/v1/models` must not wait on a load).
    models: Arc<[String]>,
    warm: Arc<Warmup>,
    default_id: String,
    parallel: Arc<Semaphore>,
    /// Adaptive memory (P10.5): touched by generation-bearing endpoints
    /// only — `/health` and `/v1/models` polls must not hold engines awake.
    mm: Arc<MemoryManager>,
}

/// Owned so a streaming reply can hold the slot until its last event.
async fn acquire_parallel(st: &AppState) -> Result<OwnedSemaphorePermit, (StatusCode, String)> {
    Arc::clone(&st.parallel).acquire_owned().await.map_err(|_| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "server shutting down".into(),
        )
    })
}

async fn with_engine(
    pool: &Arc<Mutex<ModelPool>>,
    model_id: &str,
) -> Result<Arc<std::sync::mpsc::Sender<EngineJob>>, (StatusCode, String)> {
    let pool = pool.clone();
    let id = model_id.to_owned();
    tokio::task::spawn_blocking(move || {
        let resolved = lock(&pool).resolve_id(Some(&id))?;
        crate::pool::ensure_engine(&pool, &resolved)
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .map_err(|e| {
        if e.starts_with("unfit:") || e.contains("does not fit") {
            (StatusCode::SERVICE_UNAVAILABLE, e)
        } else if e.contains("not found") {
            (StatusCode::NOT_FOUND, e)
        } else {
            (StatusCode::BAD_REQUEST, e)
        }
    })
}

async fn health(State(st): State<AppState>) -> (StatusCode, Json<Value>) {
    health_body(&st.warm)
}

/// 200 `ok` once the default model is loaded, else 503 `loading` / `error`.
fn health_body(warm: &Warmup) -> (StatusCode, Json<Value>) {
    match warm.done() {
        Some(Ok(())) => (StatusCode::OK, Json(json!({"status": "ok"}))),
        Some(Err(e)) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"status": "error", "model": warm.model, "error": e})),
        ),
        None => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "status": "loading",
                "model": warm.model,
                "progress": f64::from(warm.progress.load(Ordering::Relaxed)) / 1000.0,
            })),
        ),
    }
}

async fn list_models(State(st): State<AppState>) -> Json<Value> {
    let data = st
        .models
        .iter()
        .map(|id| {
            json!({
                "id": id,
                "object": "model",
                "owned_by": "runa"
            })
        })
        .collect::<Vec<_>>();
    Json(json!({"object": "list", "data": data}))
}

async fn chat_completions(
    State(st): State<AppState>,
    Json(body): Json<ChatCompletionBody>,
) -> Result<axum::response::Response, (StatusCode, String)> {
    let permit = acquire_parallel(&st).await?;
    // P10.5: real work arrived — hold off the idle sweep.
    st.mm.touch();
    let think = think_from_request(
        body.reasoning_effort.as_deref(),
        body.reasoning_budget_tokens,
    )
    .map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    let max_tokens = checked_max_tokens(body.max_completion_tokens.or(body.max_tokens))
        .map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    let parsed = messages_from_body(&body.messages).map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    let temps = parsed.temps;
    let messages = parsed.messages;
    let images = parsed.images;
    let audio_pcm = parsed.audio_pcm;
    if messages.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "messages must be non-empty".into()));
    }
    let sampling = SamplingConfig {
        temperature: body.temperature.unwrap_or(0.8),
        ..SamplingConfig::default()
    };
    let json_schema = schema_from_response_format(body.response_format.as_ref())
        .map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    let (tools, tool_choice) = engine_tools(body.tools.unwrap_or_default(), body.tool_choice)
        .map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    let req = GenerateRequest {
        messages,
        sampling,
        max_tokens,
        think,
        audio_pcm,
        images,
        json_schema,
        tools,
        tool_choice,
        ..GenerateRequest::default()
    };
    let model_id = body
        .model
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| st.default_id.clone());
    let jobs = with_engine(&st.pool, &model_id).await?;
    if body.stream.unwrap_or(false) {
        let enc = ChatEncoder::new(&model_id);
        return stream_reply(&jobs, req, enc, (permit, temps)).await;
    }
    let events = generate_events(&jobs, req).await.map_err(engine_status)?;
    let _keep_temps = temps;
    Ok(Json(non_stream_body(&model_id, &events)).into_response())
}

async fn anthropic_messages(
    State(st): State<AppState>,
    Json(body): Json<MessagesBody>,
) -> Result<axum::response::Response, (StatusCode, String)> {
    let permit = acquire_parallel(&st).await?;
    // P10.5: real work arrived — hold off the idle sweep.
    st.mm.touch();
    let think =
        think_from_anthropic(body.thinking.as_ref()).map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    let max_tokens =
        checked_max_tokens(body.max_tokens).map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    let mut temps = Vec::new();
    let mut messages = Vec::new();
    if let Some(sys) = body.system.as_ref() {
        let text = content_text(Some(sys), &mut Vec::new(), &mut None, &mut temps)
            .map_err(|e| (StatusCode::BAD_REQUEST, e))?;
        if !text.is_empty() {
            messages.push(ChatMessage {
                role: "system".into(),
                content: text,
                ..ChatMessage::default()
            });
        }
    }
    let parsed = messages_from_body(&body.messages).map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    temps.extend(parsed.temps);
    messages.extend(parsed.messages);
    if messages.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "messages must be non-empty".into()));
    }
    if !parsed.images.is_empty() || parsed.audio_pcm.is_some() {
        return Err((
            StatusCode::BAD_REQUEST,
            "multimodal content on /v1/messages is not supported; use /v1/chat/completions".into(),
        ));
    }
    let sampling = SamplingConfig {
        temperature: body.temperature.unwrap_or(0.8),
        ..SamplingConfig::default()
    };
    let (tools, tool_choice) =
        anthropic_tools(body.tools.unwrap_or_default(), body.tool_choice.as_ref())
            .map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    let req = GenerateRequest {
        messages,
        sampling,
        max_tokens,
        think,
        tools,
        tool_choice,
        ..GenerateRequest::default()
    };
    let model_id = body
        .model
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| st.default_id.clone());
    let jobs = with_engine(&st.pool, &model_id).await?;
    if body.stream.unwrap_or(false) {
        let enc = AnthropicEncoder::new(&model_id);
        return stream_reply(&jobs, req, enc, (permit, temps)).await;
    }
    let events = generate_events(&jobs, req).await.map_err(engine_status)?;
    let _keep_temps = temps;
    Ok(Json(anthropic_message_body(&model_id, &events)).into_response())
}

#[derive(Debug, Deserialize)]
struct EmbeddingsBody {
    model: Option<String>,
    input: EmbeddingsInput,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum EmbeddingsInput {
    One(String),
    Many(Vec<String>),
}

async fn embeddings(
    State(st): State<AppState>,
    Json(body): Json<EmbeddingsBody>,
) -> Result<Json<Value>, (StatusCode, String)> {
    let _permit = acquire_parallel(&st).await?;
    // P10.5: real work arrived — hold off the idle sweep.
    st.mm.touch();
    let inputs = match body.input {
        EmbeddingsInput::One(s) => vec![s],
        EmbeddingsInput::Many(v) => v,
    };
    if inputs.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "input must be non-empty".into()));
    }
    let model_id = body
        .model
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| st.default_id.clone());
    let jobs = with_engine(&st.pool, &model_id).await?;
    let mut data = Vec::new();
    let mut total_tokens = 0u32;
    for (index, text) in inputs.iter().enumerate() {
        let (vec, n_tokens) = embed_vector(&jobs, text).await.map_err(engine_status)?;
        total_tokens += n_tokens;
        data.push(json!({
            "object": "embedding",
            "embedding": vec,
            "index": index
        }));
    }
    Ok(Json(json!({
        "object": "list",
        "data": data,
        "model": model_id,
        "usage": {
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens
        }
    })))
}

async fn audio_transcriptions(
    State(st): State<AppState>,
    mut multipart: Multipart,
) -> Result<Json<Value>, (StatusCode, String)> {
    let _permit = acquire_parallel(&st).await?;
    // P10.5: real work arrived — hold off the idle sweep.
    st.mm.touch();
    let mut file_bytes: Option<Vec<u8>> = None;
    let mut whisper_model = "base".to_owned();
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?
    {
        match field.name() {
            Some("file") => {
                file_bytes = Some(
                    field
                        .bytes()
                        .await
                        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?
                        .to_vec(),
                );
            }
            Some("model") => {
                whisper_model = field
                    .text()
                    .await
                    .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
            }
            _ => {}
        }
    }
    let bytes = file_bytes.ok_or((
        StatusCode::BAD_REQUEST,
        "multipart field `file` is required".into(),
    ))?;
    let kind = runa_media::WhisperKind::parse(&whisper_model)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    if runa_media::ensure_whisper_model(kind, false).is_err() {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            format!(
                "whisper weights missing for {whisper_model}; run `runa media transcribe` once to pull, or set RUNA_WHISPER=1 in dev"
            ),
        ));
    }
    let mut tmp = tempfile::Builder::new()
        .prefix("runa-audio-")
        .tempfile()
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    tmp.write_all(&bytes)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let transcript = runa_media::transcribe_file(
        tmp.path(),
        &runa_media::AsrOptions {
            kind,
            pull: false,
            ..runa_media::AsrOptions::default()
        },
    )
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(json!({"text": transcript.text})))
}

fn engine_status(err: String) -> (StatusCode, String) {
    if err.contains("context_length_exceeded") {
        (StatusCode::BAD_REQUEST, err)
    } else {
        (StatusCode::INTERNAL_SERVER_ERROR, err)
    }
}

fn checked_max_tokens(explicit: Option<u32>) -> Result<u32, String> {
    match explicit {
        Some(0) => Err("max_tokens must be >= 1".into()),
        Some(n) => Ok(n),
        None => Ok(512),
    }
}

async fn generate_events(
    jobs: &Arc<std::sync::mpsc::Sender<EngineJob>>,
    req: GenerateRequest,
) -> Result<Vec<GenEvent>, String> {
    let (resp, rx) = oneshot::channel();
    jobs.send(EngineJob::Generate {
        req: Box::new(req),
        resp,
    })
    .map_err(|_| "engine thread stopped".to_string())?;
    rx.await.map_err(|e| e.to_string())?
}

async fn embed_vector(
    jobs: &Arc<std::sync::mpsc::Sender<EngineJob>>,
    input: &str,
) -> Result<(Vec<f32>, u32), String> {
    let (resp, rx) = oneshot::channel();
    jobs.send(EngineJob::Embed {
        input: input.to_owned(),
        resp,
    })
    .map_err(|_| "engine thread stopped".to_string())?;
    rx.await.map_err(|e| e.to_string())?
}

/// OpenAI `tool_calls` entries; `index` is only set on stream deltas.
fn openai_tool_calls(calls: &[ToolCall], indexed: bool) -> Value {
    calls
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let mut v = json!({
                "id": c.id,
                "type": "function",
                "function": {"name": c.name, "arguments": c.arguments}
            });
            if indexed {
                v["index"] = json!(i);
            }
            v
        })
        .collect()
}

fn openai_finish(events: &[GenEvent]) -> &'static str {
    openai_finish_reason(has_tool_calls(events), last_done(events))
}

fn openai_finish_reason(tool_calls: bool, stop: Option<&StopReason>) -> &'static str {
    if tool_calls {
        return "tool_calls";
    }
    match stop {
        Some(StopReason::MaxTokens) => "length",
        _ => "stop",
    }
}

fn last_done(events: &[GenEvent]) -> Option<&StopReason> {
    events.iter().rev().find_map(|e| match e {
        GenEvent::Done(reason) => Some(reason),
        _ => None,
    })
}

fn has_tool_calls(events: &[GenEvent]) -> bool {
    events
        .iter()
        .any(|e| matches!(e, GenEvent::ToolCalls(c) if !c.is_empty()))
}

/// Slots in the channel between the engine thread and the SSE body. Small
/// on purpose: when the client reads slowly the engine blocks on a full
/// channel instead of buffering the reply.
const STREAM_BACKLOG: usize = 8;

/// Turns engine events into the SSE events of one endpoint, one at a time.
trait SseEncoder: Send + 'static {
    /// `input_tokens` is the prompt length when the backend reports it
    /// before the first token.
    fn start(&mut self, input_tokens: Option<u32>) -> Vec<Event>;
    fn push(&mut self, ev: GenEvent) -> Vec<Event>;
    fn finish(&mut self) -> Vec<Event>;
    /// The engine failed after the response had started; the status line is
    /// already sent, so the failure travels as an event.
    fn error(&mut self, msg: &str) -> Vec<Event>;
}

/// Everything a streamed reply keeps alive until its last event: the
/// `--parallel` slot and the `data:` payload temp files.
type StreamGuards = (OwnedSemaphorePermit, Vec<tempfile::NamedTempFile>);

/// Stream one generation as SSE. Engine start-up failures (for example an
/// oversized prompt) still answer with an HTTP status, because the first
/// engine event is awaited before the response is built. Dropping the body
/// (client disconnect) drops the receiver, which cancels the generation.
async fn stream_reply<E: SseEncoder>(
    jobs: &Arc<std::sync::mpsc::Sender<EngineJob>>,
    req: GenerateRequest,
    mut enc: E,
    guards: StreamGuards,
) -> Result<Response, (StatusCode, String)> {
    let (tx, mut rx) = mpsc::channel(STREAM_BACKLOG);
    jobs.send(EngineJob::GenerateStream {
        req: Box::new(req),
        tx,
    })
    .map_err(|_| engine_status("engine thread stopped".into()))?;
    // The prompt length comes first, but errors such as an oversized prompt
    // surface after it, so wait for the first real event before answering.
    let mut input_tokens = None;
    let first = loop {
        match rx.recv().await {
            Some(Ok(Streamed::Prompt(n))) => input_tokens = Some(n),
            other => break other,
        }
    };
    let mut queue: std::collections::VecDeque<Event> = enc.start(input_tokens).into();
    let mut open = true;
    match first {
        Some(Ok(Streamed::Event(ev))) => queue.extend(enc.push(ev)),
        Some(Ok(Streamed::Prompt(_))) => {}
        Some(Err(e)) => return Err(engine_status(e)),
        None => open = false,
    }
    let body = stream::unfold(
        (rx, enc, queue, open, guards),
        |(mut rx, mut enc, mut queue, mut open, guards)| async move {
            loop {
                if let Some(ev) = queue.pop_front() {
                    return Some((
                        Ok::<_, std::convert::Infallible>(ev),
                        (rx, enc, queue, open, guards),
                    ));
                }
                if !open {
                    return None;
                }
                match rx.recv().await {
                    Some(Ok(Streamed::Event(ev))) => queue.extend(enc.push(ev)),
                    Some(Ok(Streamed::Prompt(_))) => {}
                    Some(Err(e)) => {
                        queue.extend(enc.error(&e));
                        open = false;
                    }
                    None => {
                        queue.extend(enc.finish());
                        open = false;
                    }
                }
            }
        },
    );
    Ok(sse_response(body))
}

fn sse_response(
    body: impl Stream<Item = Result<Event, std::convert::Infallible>> + Send + 'static,
) -> Response {
    Sse::new(body)
        .keep_alive(KeepAlive::default())
        .into_response()
}

/// OpenAI `chat.completion.chunk` stream.
struct ChatEncoder {
    id: String,
    model: String,
    tool_calls: bool,
    stop: Option<StopReason>,
}

impl ChatEncoder {
    fn new(model: &str) -> Self {
        ChatEncoder {
            id: completion_id(),
            model: model.to_owned(),
            tool_calls: false,
            stop: None,
        }
    }

    fn chunk(&self, delta: Value, finish: Option<&str>) -> Event {
        let body = json!({
            "id": self.id,
            "object": "chat.completion.chunk",
            "model": self.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]
        });
        Event::default().data(body.to_string())
    }
}

impl SseEncoder for ChatEncoder {
    fn start(&mut self, _input_tokens: Option<u32>) -> Vec<Event> {
        Vec::new()
    }

    fn push(&mut self, ev: GenEvent) -> Vec<Event> {
        let delta = match ev {
            GenEvent::Text(t) if !t.is_empty() => json!({"content": t}),
            GenEvent::Reasoning(t) if !t.is_empty() => json!({"reasoning_content": t}),
            GenEvent::ToolCalls(c) if !c.is_empty() => {
                self.tool_calls = true;
                json!({"tool_calls": openai_tool_calls(&c, true)})
            }
            GenEvent::Done(reason) => {
                self.stop = Some(reason);
                return Vec::new();
            }
            _ => return Vec::new(),
        };
        vec![self.chunk(delta, None)]
    }

    fn finish(&mut self) -> Vec<Event> {
        let finish = openai_finish_reason(self.tool_calls, self.stop.as_ref());
        vec![
            self.chunk(json!({}), Some(finish)),
            Event::default().data("[DONE]"),
        ]
    }

    fn error(&mut self, msg: &str) -> Vec<Event> {
        let body = json!({"error": {"message": msg, "type": "server_error"}});
        vec![Event::default().data(body.to_string())]
    }
}

/// The whole reply as SSE events, in order (unit tests).
#[cfg(test)]
fn stream_chunks(model: &str, events: &[GenEvent]) -> Vec<Event> {
    encode_all(ChatEncoder::new(model), events)
}

#[cfg(test)]
fn encode_all<E: SseEncoder>(mut enc: E, events: &[GenEvent]) -> Vec<Event> {
    let mut out = enc.start(None);
    for ev in events {
        out.extend(enc.push(ev.clone()));
    }
    out.extend(enc.finish());
    out
}

fn non_stream_body(model: &str, events: &[GenEvent]) -> Value {
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut calls: &[ToolCall] = &[];
    let mut usage = None;
    for ev in events {
        match ev {
            GenEvent::Text(t) => content.push_str(t),
            GenEvent::Reasoning(t) => reasoning.push_str(t),
            GenEvent::ToolCalls(c) => calls = c,
            GenEvent::Usage(u) => usage = Some(u),
            GenEvent::Done(_) => {}
        }
    }
    let mut message = json!({"role": "assistant", "content": content});
    if !reasoning.is_empty() {
        message["reasoning_content"] = json!(reasoning);
    }
    if !calls.is_empty() {
        message["tool_calls"] = openai_tool_calls(calls, false);
        if content.is_empty() {
            message["content"] = Value::Null;
        }
    }
    let usage = usage.map(|u| {
        json!({
            "prompt_tokens": u.prompt_tokens,
            "completion_tokens": u.generated_tokens,
            "total_tokens": u.prompt_tokens + u.generated_tokens,
            // P10.4 (M6): reasoning split of the completion tokens.
            "completion_tokens_details": {"reasoning_tokens": u.reasoning_tokens}
        })
    });
    json!({
        "id": completion_id(),
        "object": "chat.completion",
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": openai_finish(events)
        }],
        "usage": usage
    })
}

fn completion_id() -> String {
    format!(
        "chatcmpl-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0)
    )
}

#[derive(Debug, Deserialize)]
struct ChatCompletionBody {
    model: Option<String>,
    messages: Vec<IncomingMessage>,
    stream: Option<bool>,
    max_tokens: Option<u32>,
    max_completion_tokens: Option<u32>,
    temperature: Option<f32>,
    reasoning_effort: Option<String>,
    reasoning_budget_tokens: Option<u32>,
    response_format: Option<ResponseFormatBody>,
    tools: Option<Vec<Value>>,
    tool_choice: Option<Value>,
}

/// OpenAI `tools` + `tool_choice` → the engine's JSON text + choice word
/// (P8.2). A named function narrows `tools` to that one and requires it;
/// `none` drops the tools.
fn engine_tools(
    mut tools: Vec<Value>,
    choice: Option<Value>,
) -> Result<(Option<String>, Option<String>), String> {
    let choice = match choice {
        None => None,
        Some(Value::String(s)) => match s.as_str() {
            "none" => return Ok((None, None)),
            "auto" | "required" => Some(s),
            other => return Err(format!("unsupported tool_choice {other:?}")),
        },
        Some(v) => {
            let name = v["function"]["name"]
                .as_str()
                .ok_or("tool_choice object needs function.name")?;
            tools.retain(|t| t["function"]["name"] == name);
            if tools.is_empty() {
                return Err(format!("tool_choice names unknown tool {name:?}"));
            }
            Some("required".to_owned())
        }
    };
    if tools.is_empty() {
        return Ok((None, None));
    }
    Ok((Some(Value::Array(tools).to_string()), choice))
}

/// Anthropic `tools` / `tool_choice` → OpenAI shape → [`engine_tools`].
fn anthropic_tools(
    tools: Vec<AnthropicTool>,
    choice: Option<&AnthropicToolChoice>,
) -> Result<(Option<String>, Option<String>), String> {
    let tools = tools
        .into_iter()
        .map(|t| {
            json!({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description.unwrap_or_default(),
                    "parameters": t.input_schema
                }
            })
        })
        .collect();
    let choice = match choice {
        None => None,
        Some(c) => Some(match c.kind.as_str() {
            "auto" => json!("auto"),
            "any" => json!("required"),
            "none" => json!("none"),
            "tool" => json!({"type": "function", "function": {"name": c.name}}),
            other => return Err(format!("unsupported tool_choice type {other:?}")),
        }),
    };
    engine_tools(tools, choice)
}

#[derive(Debug, Deserialize)]
struct AnthropicTool {
    name: String,
    description: Option<String>,
    input_schema: Value,
}

#[derive(Debug, Deserialize)]
struct AnthropicToolChoice {
    #[serde(rename = "type")]
    kind: String,
    name: Option<String>,
}

/// OpenAI `response_format` (P8.1).
#[derive(Debug, Deserialize)]
struct ResponseFormatBody {
    #[serde(rename = "type")]
    kind: String,
    json_schema: Option<JsonSchemaBody>,
}

#[derive(Debug, Deserialize)]
struct JsonSchemaBody {
    schema: Option<Value>,
}

/// `response_format` → JSON Schema text for the engine grammar;
/// `json_object` means "any JSON object".
fn schema_from_response_format(rf: Option<&ResponseFormatBody>) -> Result<Option<String>, String> {
    let Some(rf) = rf else { return Ok(None) };
    let schema = match rf.kind.as_str() {
        "text" => return Ok(None),
        "json_object" => json!({"type": "object"}),
        "json_schema" => rf
            .json_schema
            .as_ref()
            .and_then(|j| j.schema.clone())
            .ok_or("response_format json_schema needs json_schema.schema")?,
        other => return Err(format!("unsupported response_format type {other:?}")),
    };
    let text = schema.to_string();
    runa_engine::schema_to_grammar(&text).map_err(|e| e.to_string())?;
    Ok(Some(text))
}

#[derive(Debug, Deserialize)]
struct MessagesBody {
    model: Option<String>,
    messages: Vec<IncomingMessage>,
    stream: Option<bool>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    system: Option<IncomingContent>,
    thinking: Option<ThinkingBody>,
    tools: Option<Vec<AnthropicTool>>,
    tool_choice: Option<AnthropicToolChoice>,
}

#[derive(Debug, Deserialize)]
struct ThinkingBody {
    #[serde(rename = "type")]
    kind: Option<String>,
    budget_tokens: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct IncomingMessage {
    role: String,
    #[serde(default)]
    content: Option<IncomingContent>,
    /// OpenAI assistant turn that called tools.
    #[serde(default)]
    tool_calls: Vec<IncomingToolCall>,
    /// OpenAI `tool` message: the call it answers.
    tool_call_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct IncomingToolCall {
    id: Option<String>,
    function: IncomingFunction,
}

#[derive(Debug, Deserialize)]
struct IncomingFunction {
    name: String,
    /// JSON text per the spec; an object is accepted too.
    arguments: Option<Value>,
}

/// Arguments as JSON text, whatever shape the client sent.
fn arguments_text(v: Option<&Value>) -> String {
    match v {
        Some(Value::String(s)) => s.clone(),
        Some(v) => v.to_string(),
        None => "{}".to_owned(),
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum IncomingContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Deserialize)]
struct ContentPart {
    #[serde(rename = "type")]
    kind: Option<String>,
    text: Option<String>,
    image_url: Option<ImageUrlPart>,
    input_audio: Option<InputAudioPart>,
    /// Anthropic `tool_use` block: call id, tool name, arguments.
    id: Option<String>,
    name: Option<String>,
    input: Option<Value>,
    /// Anthropic `tool_result` block: the call it answers and its output.
    tool_use_id: Option<String>,
    content: Option<IncomingContent>,
}

#[derive(Debug, Deserialize)]
struct ImageUrlPart {
    url: String,
}

#[derive(Debug, Deserialize)]
struct InputAudioPart {
    data: String,
    format: Option<String>,
}

pub(crate) fn think_from_request(
    reasoning_effort: Option<&str>,
    reasoning_budget_tokens: Option<u32>,
) -> Result<ThinkConfig, String> {
    let mut o = ThinkOverrides {
        show: Some(true),
        ..Default::default()
    };
    if let Some(n) = reasoning_budget_tokens {
        if n == 0 {
            o.think = Some(false);
        } else {
            o.budget = Some(n);
        }
    } else if let Some(e) = reasoning_effort {
        o.effort = Some(Effort::parse(e)?);
    } else {
        o.think = Some(true);
    }
    ThinkConfig::default().apply(&o)
}

fn think_from_anthropic(thinking: Option<&ThinkingBody>) -> Result<ThinkConfig, String> {
    let Some(t) = thinking else {
        return think_from_request(None, None);
    };
    match t.kind.as_deref() {
        Some("disabled") => think_from_request(None, Some(0)),
        Some("enabled") => think_from_request(None, t.budget_tokens.or(Some(1024))),
        Some("adaptive") => think_from_request(Some("high"), None),
        _ => think_from_request(None, t.budget_tokens),
    }
}

fn anthropic_message_body(model: &str, events: &[GenEvent]) -> Value {
    let mut text = String::new();
    let mut thinking = String::new();
    let mut calls: &[ToolCall] = &[];
    let mut usage = None;
    for ev in events {
        match ev {
            GenEvent::Text(t) => text.push_str(t),
            GenEvent::Reasoning(t) => thinking.push_str(t),
            GenEvent::ToolCalls(c) => calls = c,
            GenEvent::Usage(u) => usage = Some(u),
            GenEvent::Done(_) => {}
        }
    }
    let mut content = Vec::new();
    if !thinking.is_empty() {
        content.push(json!({"type": "thinking", "thinking": thinking}));
    }
    if !text.is_empty() || calls.is_empty() {
        content.push(json!({"type": "text", "text": text}));
    }
    content.extend(calls.iter().map(tool_use_block));
    let usage = usage.map(|u| {
        json!({
            "input_tokens": u.prompt_tokens,
            "output_tokens": u.generated_tokens
        })
    });
    json!({
        "id": format!("msg-{}", completion_id().trim_start_matches("chatcmpl-")),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": anthropic_stop(events, calls),
        "usage": usage
    })
}

fn anthropic_stop(events: &[GenEvent], calls: &[ToolCall]) -> &'static str {
    anthropic_stop_reason(!calls.is_empty(), last_done(events))
}

fn anthropic_stop_reason(tool_calls: bool, stop: Option<&StopReason>) -> &'static str {
    if tool_calls {
        return "tool_use";
    }
    match stop {
        Some(StopReason::MaxTokens) => "max_tokens",
        _ => "end_turn",
    }
}

/// Anthropic `tool_use` block; arguments that are not JSON become `{}`.
fn tool_use_block(c: &ToolCall) -> Value {
    json!({
        "type": "tool_use",
        "id": c.id,
        "name": c.name,
        "input": serde_json::from_str::<Value>(&c.arguments).unwrap_or_else(|_| json!({}))
    })
}

fn sse(name: &str, data: Value) -> Event {
    Event::default().event(name).data(data.to_string())
}

/// Anthropic message stream. `message_start` carries the prompt length the
/// engine reports up front (0 for a backend that only counts at the end);
/// the final counts arrive in the closing `message_delta`.
struct AnthropicEncoder {
    id: String,
    model: String,
    /// Index of the content block being written.
    idx: u32,
    /// Kind of the open content block; each run of thinking / text deltas
    /// is one block, every tool call is its own.
    open: Option<&'static str>,
    tool_calls: bool,
    stop: Option<StopReason>,
    usage: (u32, u32),
}

impl AnthropicEncoder {
    fn new(model: &str) -> Self {
        AnthropicEncoder {
            id: format!("msg-{}", completion_id().trim_start_matches("chatcmpl-")),
            model: model.to_owned(),
            idx: 0,
            open: None,
            tool_calls: false,
            stop: None,
            usage: (0, 0),
        }
    }

    fn close_block(&mut self, out: &mut Vec<Event>) {
        if self.open.take().is_some() {
            out.push(sse(
                "content_block_stop",
                json!({"type": "content_block_stop", "index": self.idx}),
            ));
            self.idx += 1;
        }
    }
}

impl SseEncoder for AnthropicEncoder {
    fn start(&mut self, input_tokens: Option<u32>) -> Vec<Event> {
        let start = json!({
            "type": "message_start",
            "message": {
                "id": self.id,
                "type": "message",
                "role": "assistant",
                "model": self.model,
                "content": [],
                "stop_reason": null,
                "usage": {"input_tokens": input_tokens.unwrap_or(0), "output_tokens": 0}
            }
        });
        vec![sse("message_start", start)]
    }

    fn push(&mut self, ev: GenEvent) -> Vec<Event> {
        let mut out = Vec::new();
        let (kind, block, delta) = match ev {
            GenEvent::Reasoning(t) if !t.is_empty() => (
                "thinking",
                json!({"type": "thinking", "thinking": ""}),
                json!({"type": "thinking_delta", "thinking": t}),
            ),
            GenEvent::Text(t) if !t.is_empty() => (
                "text",
                json!({"type": "text", "text": ""}),
                json!({"type": "text_delta", "text": t}),
            ),
            GenEvent::ToolCalls(c) => {
                self.tool_calls = !c.is_empty();
                self.close_block(&mut out);
                for call in &c {
                    let mut block = tool_use_block(call);
                    block["input"] = json!({});
                    out.push(sse(
                        "content_block_start",
                        json!({"type": "content_block_start", "index": self.idx, "content_block": block}),
                    ));
                    out.push(sse(
                        "content_block_delta",
                        json!({
                            "type": "content_block_delta",
                            "index": self.idx,
                            "delta": {"type": "input_json_delta", "partial_json": call.arguments}
                        }),
                    ));
                    self.open = Some("tool_use");
                    self.close_block(&mut out);
                }
                return out;
            }
            GenEvent::Usage(u) => {
                self.usage = (u.prompt_tokens, u.generated_tokens);
                return out;
            }
            GenEvent::Done(reason) => {
                self.stop = Some(reason);
                return out;
            }
            _ => return out,
        };
        if self.open != Some(kind) {
            self.close_block(&mut out);
            out.push(sse(
                "content_block_start",
                json!({"type": "content_block_start", "index": self.idx, "content_block": block}),
            ));
            self.open = Some(kind);
        }
        out.push(sse(
            "content_block_delta",
            json!({"type": "content_block_delta", "index": self.idx, "delta": delta}),
        ));
        out
    }

    fn finish(&mut self) -> Vec<Event> {
        let mut out = Vec::new();
        self.close_block(&mut out);
        out.push(sse(
            "message_delta",
            json!({
                "type": "message_delta",
                "delta": {"stop_reason": anthropic_stop_reason(self.tool_calls, self.stop.as_ref())},
                "usage": {"input_tokens": self.usage.0, "output_tokens": self.usage.1}
            }),
        ));
        out.push(sse("message_stop", json!({"type": "message_stop"})));
        out
    }

    fn error(&mut self, msg: &str) -> Vec<Event> {
        let body = json!({"type": "error", "error": {"type": "api_error", "message": msg}});
        vec![sse("error", body)]
    }
}

/// Parsed request body: chat messages plus optional media (vision frames,
/// PCM audio for the ASR route). `temps` keeps `data:` payloads alive
/// until generation finishes, then deletes them.
#[derive(Debug)]
struct ParsedBody {
    messages: Vec<ChatMessage>,
    images: Vec<VisionFrame>,
    audio_pcm: Option<Vec<f32>>,
    temps: Vec<tempfile::NamedTempFile>,
}

fn messages_from_body(messages: &[IncomingMessage]) -> Result<ParsedBody, String> {
    let mut out = Vec::new();
    let mut images = Vec::new();
    let mut audio_pcm: Option<Vec<f32>> = None;
    let mut temps = Vec::new();
    for m in messages {
        validate_role(&m.role)?;
        let text = content_text(m.content.as_ref(), &mut images, &mut audio_pcm, &mut temps)?;
        let mut tool_calls: Vec<ToolCall> = m
            .tool_calls
            .iter()
            .map(|c| ToolCall {
                id: c.id.clone().unwrap_or_default(),
                name: c.function.name.clone(),
                arguments: arguments_text(c.function.arguments.as_ref()),
            })
            .collect();
        // Anthropic carries calls and results as content blocks; results
        // become `tool` messages ahead of the turn's own text.
        if let Some(IncomingContent::Parts(parts)) = &m.content {
            for p in parts {
                match p.kind.as_deref() {
                    Some("tool_use") => tool_calls.push(ToolCall {
                        id: p.id.clone().unwrap_or_default(),
                        name: p.name.clone().ok_or("tool_use block needs a name")?,
                        arguments: arguments_text(p.input.as_ref()),
                    }),
                    Some("tool_result") => out.push(ChatMessage {
                        role: "tool".into(),
                        content: content_text(
                            p.content.as_ref(),
                            &mut Vec::new(),
                            &mut None,
                            &mut temps,
                        )?,
                        tool_call_id: p.tool_use_id.clone(),
                        ..ChatMessage::default()
                    }),
                    _ => {}
                }
            }
        }
        if !text.is_empty() || m.role == "assistant" || m.role == "tool" || !tool_calls.is_empty() {
            out.push(ChatMessage {
                role: m.role.clone(),
                content: text,
                tool_calls,
                tool_call_id: m.tool_call_id.clone(),
            });
        }
    }
    if !images.is_empty() && audio_pcm.is_some() {
        return Err("cannot mix image and audio in one request".into());
    }
    if !images.is_empty() && !cfg!(feature = "mtmd") {
        return Err(
            "image_url / vision content requires rebuilding runa with --features mtmd".into(),
        );
    }
    Ok(ParsedBody {
        messages: out,
        images,
        audio_pcm,
        temps,
    })
}

fn validate_role(role: &str) -> Result<(), String> {
    match role {
        "system" | "developer" | "user" | "assistant" | "tool" => Ok(()),
        other => Err(format!("unsupported message role: {other}")),
    }
}

fn content_text(
    c: Option<&IncomingContent>,
    images: &mut Vec<VisionFrame>,
    audio_pcm: &mut Option<Vec<f32>>,
    temps: &mut Vec<tempfile::NamedTempFile>,
) -> Result<String, String> {
    match c {
        Some(IncomingContent::Text(s)) => Ok(s.clone()),
        Some(IncomingContent::Parts(parts)) => {
            let mut text = String::new();
            for p in parts {
                let kind = p.kind.as_deref().unwrap_or("text");
                match kind {
                    "text" => {
                        if let Some(t) = p.text.as_deref() {
                            text.push_str(t);
                        }
                    }
                    "image_url" => {
                        let url = p
                            .image_url
                            .as_ref()
                            .ok_or("image_url part missing image_url field")?
                            .url
                            .clone();
                        if !url.starts_with("data:") {
                            return Err("image_url must be a data: URI".into());
                        }
                        if !cfg!(feature = "mtmd") {
                            return Err(
                                "image_url / vision content requires rebuilding runa with --features mtmd"
                                    .into(),
                            );
                        }
                        let path = image_data_to_temp(&url, temps)?;
                        images.push(VisionFrame {
                            t_sec: None,
                            source: VisionSource::Path(path),
                        });
                    }
                    "input_audio" => {
                        let part = p
                            .input_audio
                            .as_ref()
                            .ok_or("input_audio part missing input_audio field")?;
                        let pcm = decode_input_audio(part)?;
                        if audio_pcm.is_some() {
                            return Err("only one input_audio part per request".into());
                        }
                        *audio_pcm = Some(pcm);
                    }
                    // Mapped to tool calls / tool messages by the caller.
                    "tool_use" | "tool_result" => {}
                    other => return Err(format!("unsupported content part type: {other}")),
                }
            }
            Ok(text)
        }
        None => Ok(String::new()),
    }
}

fn image_data_to_temp(
    url: &str,
    temps: &mut Vec<tempfile::NamedTempFile>,
) -> Result<PathBuf, String> {
    let rest = url.strip_prefix("data:").ok_or("malformed data URI")?;
    let (_meta, b64) = rest
        .split_once(',')
        .ok_or("malformed data URI: missing comma")?;
    let bytes = b64_decode(b64.trim())?;
    let mut tmp = tempfile::Builder::new()
        .prefix("runa-img-")
        .tempfile()
        .map_err(|e| e.to_string())?;
    tmp.write_all(&bytes).map_err(|e| e.to_string())?;
    let path = tmp.path().to_owned();
    temps.push(tmp);
    Ok(path)
}

fn decode_input_audio(part: &InputAudioPart) -> Result<Vec<f32>, String> {
    let bytes = b64_decode(part.data.trim())?;
    let ext = match part.format.as_deref() {
        Some("wav") | None => "wav",
        Some("mp3") => "mp3",
        other => return Err(format!("unsupported input_audio format: {:?}", other)),
    };
    let mut tmp = tempfile::Builder::new()
        .prefix("runa-in-audio-")
        .suffix(&format!(".{ext}"))
        .tempfile()
        .map_err(|e| e.to_string())?;
    tmp.write_all(&bytes).map_err(|e| e.to_string())?;
    let decoded = runa_media::decode_audio(tmp.path()).map_err(|e| e.to_string())?;
    Ok(decoded.samples)
}

fn b64_decode(input: &str) -> Result<Vec<u8>, String> {
    let cleaned: String = input.chars().filter(|c| !c.is_ascii_whitespace()).collect();
    base64::engine::general_purpose::STANDARD
        .decode(cleaned)
        .map_err(|_| "invalid base64".to_owned())
}

/// Build model id + path list from CLI paths (stem ids; suffix on collision).
pub(crate) fn model_specs_from_paths(
    paths: Vec<PathBuf>,
) -> Result<Vec<(String, PathBuf)>, String> {
    let mut out = Vec::new();
    let mut counts: HashMap<String, usize> = HashMap::new();
    for path in paths {
        let resolved = if path.is_file() {
            path
        } else {
            crate::resolve_model(path.to_str().unwrap_or(""))?
        };
        let stem = resolved
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("runa")
            .to_owned();
        let n = counts.entry(stem.clone()).or_insert(0);
        let id = if *n == 0 {
            stem.clone()
        } else {
            format!("{stem}-{}", *n)
        };
        *n += 1;
        out.push((id, resolved));
    }
    Ok(out)
}

/// Fuzzing entry (`cargo fuzz`, fuzz/README.md): an HTTP request body
/// through the same parsing the `/v1/chat/completions` and `/v1/messages`
/// handlers do before touching a model. Bodies with `image_url` or
/// `input_audio` parts are skipped: those write temp files and decode media.
#[cfg(fuzzing)]
pub(crate) fn fuzz_request_body(body: &[u8]) {
    fn has_media(messages: &[IncomingMessage]) -> bool {
        messages.iter().any(|m| match &m.content {
            Some(IncomingContent::Parts(parts)) => parts
                .iter()
                .any(|p| p.image_url.is_some() || p.input_audio.is_some()),
            _ => false,
        })
    }
    if let Ok(b) = serde_json::from_slice::<ChatCompletionBody>(body) {
        let _ = think_from_request(b.reasoning_effort.as_deref(), b.reasoning_budget_tokens);
        if !has_media(&b.messages) {
            let _ = messages_from_body(&b.messages);
        }
        let _ = schema_from_response_format(b.response_format.as_ref());
        let _ = engine_tools(b.tools.unwrap_or_default(), b.tool_choice);
    }
    if let Ok(b) = serde_json::from_slice::<MessagesBody>(body) {
        let _ = think_from_anthropic(b.thinking.as_ref());
        if !has_media(&b.messages) {
            let _ = messages_from_body(&b.messages);
        }
        if let Some(IncomingContent::Text(_)) = b.system.as_ref() {
            let _ = content_text(
                b.system.as_ref(),
                &mut Vec::new(),
                &mut None,
                &mut Vec::new(),
            );
        }
        let _ = anthropic_tools(b.tools.unwrap_or_default(), b.tool_choice.as_ref());
    }
    if let Ok(text) = std::str::from_utf8(body) {
        let _ = b64_decode(text);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use runa_core::ThinkMode;

    #[test]
    fn effort_and_budget_fields() {
        let t = think_from_request(Some("high"), None).unwrap();
        assert!(matches!(t.mode, ThinkMode::Effort(Effort::High)));
        assert!(t.show);
        let t = think_from_request(None, Some(256)).unwrap();
        assert!(matches!(t.mode, ThinkMode::Budget { tokens: 256, .. }));
        let t = think_from_request(None, Some(0)).unwrap();
        assert!(matches!(t.mode, ThinkMode::Off));
    }

    #[test]
    fn response_format_to_schema() {
        let rf = |raw: &str| serde_json::from_str::<ResponseFormatBody>(raw).unwrap();
        assert_eq!(schema_from_response_format(None).unwrap(), None);
        assert_eq!(
            schema_from_response_format(Some(&rf(r#"{"type":"text"}"#))).unwrap(),
            None
        );
        assert_eq!(
            schema_from_response_format(Some(&rf(r#"{"type":"json_object"}"#))).unwrap(),
            Some(r#"{"type":"object"}"#.into())
        );
        let s = schema_from_response_format(Some(&rf(
            r#"{"type":"json_schema","json_schema":{"name":"a","schema":{"type":"string"}}}"#,
        )))
        .unwrap()
        .unwrap();
        assert_eq!(s, r#"{"type":"string"}"#);
        assert!(schema_from_response_format(Some(&rf(r#"{"type":"json_schema"}"#))).is_err());
        assert!(schema_from_response_format(Some(&rf(r#"{"type":"xml"}"#))).is_err());
    }

    #[test]
    fn tool_requests_map_to_engine() {
        let tool = |name: &str| json!({"type": "function", "function": {"name": name}});
        let two = vec![tool("a"), tool("b")];
        let (tools, choice) =
            engine_tools(two.clone(), Some(json!({"function": {"name": "b"}}))).unwrap();
        assert_eq!(tools.unwrap(), Value::Array(vec![tool("b")]).to_string());
        assert_eq!(choice.as_deref(), Some("required"));
        assert_eq!(
            engine_tools(two.clone(), Some(json!("none"))).unwrap(),
            (None, None)
        );
        assert!(engine_tools(two, Some(json!({"function": {"name": "z"}}))).is_err());

        let tools: Vec<AnthropicTool> =
            serde_json::from_str(r#"[{"name":"a","input_schema":{"type":"object"}}]"#).unwrap();
        let any: AnthropicToolChoice = serde_json::from_str(r#"{"type":"any"}"#).unwrap();
        let (json, choice) = anthropic_tools(tools, Some(&any)).unwrap();
        assert!(json.unwrap().contains(r#""name":"a""#));
        assert_eq!(choice.as_deref(), Some("required"));

        let raw = r#"[
            {"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"a","input":{"x":1}}]},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"ok"}]}
        ]"#;
        let msgs: Vec<IncomingMessage> = serde_json::from_str(raw).unwrap();
        let out = messages_from_body(&msgs).unwrap().messages;
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].tool_calls[0].name, "a");
        assert_eq!(out[0].tool_calls[0].arguments, r#"{"x":1}"#);
        assert_eq!(out[1].role, "tool");
        assert_eq!(out[1].tool_call_id.as_deref(), Some("t1"));
        assert_eq!(out[1].content, "ok");
    }

    #[test]
    fn messages_join_text_parts() {
        let raw = r#"[{"role":"user","content":[{"type":"text","text":"hi "},{"type":"text","text":"there"}]}]"#;
        let msgs: Vec<IncomingMessage> = serde_json::from_str(raw).unwrap();
        let parsed = messages_from_body(&msgs).unwrap();
        assert_eq!(parsed.messages[0].content, "hi there");
        assert!(parsed.images.is_empty());
        assert!(parsed.audio_pcm.is_none());
    }

    #[test]
    fn non_stream_includes_reasoning_content() {
        let events = vec![
            GenEvent::Reasoning("plan".into()),
            GenEvent::Text("ok".into()),
            GenEvent::Usage(runa_engine::Usage {
                prompt_tokens: 5,
                generated_tokens: 7,
                reasoning_tokens: 3,
                pp_toks_per_s: 1.0,
                tg_toks_per_s: 2.0,
            }),
        ];
        let v = non_stream_body("m", &events);
        assert_eq!(v["choices"][0]["message"]["content"], "ok");
        assert_eq!(v["choices"][0]["message"]["reasoning_content"], "plan");
        // P10.4: reasoning split of the completion tokens (M6).
        assert_eq!(v["usage"]["completion_tokens"], 7);
        assert_eq!(
            v["usage"]["completion_tokens_details"]["reasoning_tokens"],
            3
        );
        assert_eq!(v["choices"][0]["finish_reason"], "stop");
        let truncated = vec![
            GenEvent::Text("Once upon a".into()),
            GenEvent::Done(runa_engine::StopReason::MaxTokens),
        ];
        let v = non_stream_body("m", &truncated);
        assert_eq!(v["choices"][0]["finish_reason"], "length");
        let v = anthropic_message_body("m", &truncated);
        assert_eq!(v["stop_reason"], "max_tokens");
    }

    #[test]
    fn image_url_rejects_local_paths_without_statting() {
        let raw = r#"[{"role":"user","content":[{"type":"image_url","image_url":{"url":"/etc/hosts"}}]}]"#;
        let msgs: Vec<IncomingMessage> = serde_json::from_str(raw).unwrap();
        let err = messages_from_body(&msgs).unwrap_err();
        assert!(err.contains("data: URI"), "{err}");
        let raw = r#"[{"role":"user","content":[{"type":"image_url","image_url":{"url":"/no/such/file"}}]}]"#;
        let msgs: Vec<IncomingMessage> = serde_json::from_str(raw).unwrap();
        let err2 = messages_from_body(&msgs).unwrap_err();
        assert_eq!(err, err2);
    }

    #[test]
    fn base64_rejects_trailing_junk_and_max_tokens_zero() {
        assert!(b64_decode("UklGRg==!!!not-base64###").is_err());
        assert_eq!(b64_decode("AA==").unwrap(), vec![0]);
        assert!(checked_max_tokens(Some(0)).is_err());
        assert_eq!(checked_max_tokens(None).unwrap(), 512);
        let raw = r#"[{"role":"wizard","content":"hi"}]"#;
        let msgs: Vec<IncomingMessage> = serde_json::from_str(raw).unwrap();
        assert!(messages_from_body(&msgs).unwrap_err().contains("role"));
    }

    #[test]
    fn anthropic_message_includes_thinking_block() {
        let events = vec![
            GenEvent::Reasoning("plan".into()),
            GenEvent::Text("ok".into()),
        ];
        let v = anthropic_message_body("m", &events);
        assert_eq!(v["type"], "message");
        assert_eq!(v["content"][0]["type"], "thinking");
        assert_eq!(v["content"][0]["thinking"], "plan");
        assert_eq!(v["content"][1]["text"], "ok");
        let t = think_from_anthropic(Some(&ThinkingBody {
            kind: Some("enabled".into()),
            budget_tokens: Some(256),
        }))
        .unwrap();
        assert!(matches!(t.mode, ThinkMode::Budget { tokens: 256, .. }));
    }

    #[test]
    fn stream_chunks_include_reasoning_and_done() {
        let events = vec![
            GenEvent::Reasoning("plan".into()),
            GenEvent::Text("ok".into()),
        ];
        let chunks = stream_chunks("m", &events);
        assert_eq!(chunks.len(), 4, "reasoning, text, finish, [DONE]");
    }

    #[test]
    fn health_reports_warm_up() {
        let warm = Warmup::new("m".into(), Arc::new(AtomicU32::new(420)));
        let (code, Json(body)) = health_body(&warm);
        assert_eq!(code, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body["status"], "loading");
        assert_eq!(body["progress"], 0.42);
        *warm.state.lock().unwrap() = Some(Err("unfit".into()));
        let (code, Json(body)) = health_body(&warm);
        assert_eq!(
            (code, body["error"].as_str()),
            (StatusCode::SERVICE_UNAVAILABLE, Some("unfit"))
        );
        *warm.state.lock().unwrap() = Some(Ok(()));
        assert_eq!(health_body(&warm).0, StatusCode::OK);
        // Later LRU reloads overwrite the shared slot, but nothing reads
        // it once warm-up is done.
        warm.progress.store(1000, Ordering::Relaxed);
        assert_eq!(health_body(&warm).0, StatusCode::OK);
    }

    #[test]
    fn b64_roundtrip() {
        let raw = b"hello";
        let enc = runa_media::wav_base64(raw);
        assert_eq!(b64_decode(&enc).unwrap(), raw);
    }

    /// A scripted engine thread behind the same job protocol as the real
    /// one: `script` runs on the engine side with the stream sender.
    fn fake_engine(
        script: impl Fn(&mpsc::Sender<Result<Streamed, String>>) + Send + 'static,
    ) -> Arc<std::sync::mpsc::Sender<EngineJob>> {
        let (jobs, rx) = std::sync::mpsc::channel::<EngineJob>();
        std::thread::spawn(move || {
            while let Ok(job) = rx.recv() {
                if let EngineJob::GenerateStream { tx, .. } = job {
                    script(&tx);
                }
            }
        });
        Arc::new(jobs)
    }

    fn ev(e: GenEvent) -> Result<Streamed, String> {
        Ok(Streamed::Event(e))
    }

    async fn start_stream(
        jobs: &Arc<std::sync::mpsc::Sender<EngineJob>>,
        enc: impl SseEncoder,
    ) -> Result<Response, (StatusCode, String)> {
        let permit = Arc::new(Semaphore::new(1)).acquire_owned().await.unwrap();
        stream_reply(jobs, GenerateRequest::default(), enc, (permit, Vec::new())).await
    }

    async fn start_chat_stream(
        jobs: &Arc<std::sync::mpsc::Sender<EngineJob>>,
    ) -> Result<Response, (StatusCode, String)> {
        start_stream(jobs, ChatEncoder::new("m")).await
    }

    async fn next_text(
        body: &mut (impl Stream<Item = Result<axum::body::Bytes, axum::Error>> + Unpin),
    ) -> String {
        let chunk = body
            .next()
            .await
            .expect("stream ended")
            .expect("body error");
        String::from_utf8(chunk.to_vec()).unwrap()
    }

    #[tokio::test]
    async fn first_token_reaches_the_client_before_generation_ends() {
        let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = Arc::clone(&done);
        let jobs = fake_engine(move |tx| {
            tx.blocking_send(ev(GenEvent::Text("first".into())))
                .unwrap();
            std::thread::sleep(std::time::Duration::from_millis(600));
            tx.blocking_send(ev(GenEvent::Text("second".into())))
                .unwrap();
            tx.blocking_send(ev(GenEvent::Done(StopReason::Eos)))
                .unwrap();
            flag.store(true, Ordering::SeqCst);
        });
        let started = std::time::Instant::now();
        let resp = start_chat_stream(&jobs).await.unwrap();
        let mut body = resp.into_body().into_data_stream();
        let first = next_text(&mut body).await;
        assert!(first.contains(r#""content":"first""#), "{first}");
        assert!(started.elapsed() < std::time::Duration::from_millis(400));
        assert!(
            !done.load(Ordering::SeqCst),
            "generation ended before the first token arrived"
        );
        let mut rest = String::new();
        while let Some(chunk) = body.next().await {
            rest.push_str(std::str::from_utf8(&chunk.unwrap()).unwrap());
        }
        assert!(started.elapsed() >= std::time::Duration::from_millis(600));
        assert!(rest.contains(r#""content":"second""#), "{rest}");
        assert!(rest.contains(r#""finish_reason":"stop""#) && rest.ends_with("data: [DONE]\n\n"));
    }

    #[tokio::test]
    async fn dropping_the_body_cancels_generation_and_a_slow_reader_throttles_it() {
        let sent = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let cancelled = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let (count, stopped) = (Arc::clone(&sent), Arc::clone(&cancelled));
        let jobs = fake_engine(move |tx| {
            for _ in 0..1000 {
                if tx.blocking_send(ev(GenEvent::Text("t".into()))).is_err() {
                    stopped.store(true, Ordering::SeqCst);
                    return;
                }
                count.fetch_add(1, Ordering::SeqCst);
            }
        });
        let resp = start_chat_stream(&jobs).await.unwrap();
        let mut body = resp.into_body().into_data_stream();
        next_text(&mut body).await;
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        let ahead = sent.load(Ordering::SeqCst);
        assert!(
            ahead <= STREAM_BACKLOG + 4,
            "engine ran {ahead} events ahead of the reader"
        );
        drop(body);
        for _ in 0..100 {
            if cancelled.load(Ordering::SeqCst) {
                return;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        panic!("generation was not cancelled after the client left");
    }

    #[tokio::test]
    async fn engine_errors_keep_their_status_before_the_stream_and_become_events_after() {
        let early = fake_engine(|tx| {
            tx.blocking_send(Err("context_length_exceeded".into()))
                .unwrap();
        });
        let err = start_chat_stream(&early).await.unwrap_err();
        assert_eq!(err.0, StatusCode::BAD_REQUEST);

        let late = fake_engine(|tx| {
            tx.blocking_send(ev(GenEvent::Text("ok".into()))).unwrap();
            tx.blocking_send(Err("boom".into())).unwrap();
        });
        let resp = start_chat_stream(&late).await.unwrap();
        let mut body = resp.into_body().into_data_stream();
        let mut all = String::new();
        while let Some(chunk) = body.next().await {
            all.push_str(std::str::from_utf8(&chunk.unwrap()).unwrap());
        }
        assert!(
            all.contains(r#""content":"ok""#) && all.contains("boom"),
            "{all}"
        );
        assert!(!all.contains("[DONE]"), "{all}");
    }

    #[tokio::test]
    async fn anthropic_message_start_carries_the_prompt_length() {
        let jobs = fake_engine(|tx| {
            tx.blocking_send(Ok(Streamed::Prompt(42))).unwrap();
            tx.blocking_send(ev(GenEvent::Text("hi".into()))).unwrap();
        });
        let resp = start_stream(&jobs, AnthropicEncoder::new("m"))
            .await
            .unwrap();
        let mut body = resp.into_body().into_data_stream();
        let first = next_text(&mut body).await;
        assert!(first.starts_with("event: message_start"), "{first}");
        assert!(first.contains(r#""input_tokens":42"#), "{first}");
    }

    #[test]
    fn anthropic_stream_keeps_block_order() {
        let events = vec![
            GenEvent::Reasoning("plan".into()),
            GenEvent::Text("ok".into()),
            GenEvent::Done(StopReason::Eos),
        ];
        let out = encode_all(AnthropicEncoder::new("m"), &events);
        // message_start, (block start, delta, stop) x2, message_delta, message_stop
        assert_eq!(out.len(), 9);
    }
}
