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
use futures::{FutureExt, stream};
use runa_core::{BackendKind, Effort, ThinkConfig, ThinkOverrides};
use runa_engine::{
    ChatMessage, GenEvent, GenerateRequest, LoadConfig, Mode, Placement, SamplingConfig, ToolCall,
    VisionFrame, VisionSource,
};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::SemaphorePermit;
use tokio::sync::{Semaphore, oneshot};

use crate::pool::{EngineJob, ModelPool, Warmup, lock, panic_text};
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

async fn acquire_parallel(st: &AppState) -> Result<SemaphorePermit<'_>, (StatusCode, String)> {
    st.parallel.acquire().await.map_err(|_| {
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
        let mut p = lock(&pool);
        match p.resolve_id(Some(&id)) {
            Ok(resolved) => p.ensure_engine(&resolved),
            Err(e) if e.contains("not found") => Err(e),
            Err(e) => Err(e),
        }
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
    let _permit = acquire_parallel(&st).await?;
    // P10.5: real work arrived — hold off the idle sweep.
    st.mm.touch();
    let think = think_from_request(
        body.reasoning_effort.as_deref(),
        body.reasoning_budget_tokens,
    )
    .map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    let (messages, images, audio_pcm) =
        messages_from_body(&body.messages).map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    if messages.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "messages must be non-empty".into()));
    }
    let max_tokens = body
        .max_completion_tokens
        .or(body.max_tokens)
        .unwrap_or(512);
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
        let events = generate_events(&jobs, req)
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
        let sse = stream_chunks(&model_id, &events);
        return Ok(Sse::new(stream::iter(
            sse.into_iter().map(Ok::<_, std::convert::Infallible>),
        ))
        .keep_alive(KeepAlive::default())
        .into_response());
    }
    let events = generate_events(&jobs, req)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(non_stream_body(&model_id, &events)).into_response())
}

async fn anthropic_messages(
    State(st): State<AppState>,
    Json(body): Json<MessagesBody>,
) -> Result<axum::response::Response, (StatusCode, String)> {
    let _permit = acquire_parallel(&st).await?;
    // P10.5: real work arrived — hold off the idle sweep.
    st.mm.touch();
    let think =
        think_from_anthropic(body.thinking.as_ref()).map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    let mut messages = Vec::new();
    if let Some(sys) = body.system.as_ref() {
        let text = content_text(Some(sys), &mut Vec::new(), &mut None)
            .map_err(|e| (StatusCode::BAD_REQUEST, e))?;
        if !text.is_empty() {
            messages.push(ChatMessage {
                role: "system".into(),
                content: text,
                ..ChatMessage::default()
            });
        }
    }
    let (msgs, images, audio_pcm) =
        messages_from_body(&body.messages).map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    messages.extend(msgs);
    if messages.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "messages must be non-empty".into()));
    }
    if !images.is_empty() || audio_pcm.is_some() {
        return Err((
            StatusCode::BAD_REQUEST,
            "multimodal content on /v1/messages is not supported; use /v1/chat/completions".into(),
        ));
    }
    let max_tokens = body.max_tokens.unwrap_or(512);
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
        let events = generate_events(&jobs, req)
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
        let sse = anthropic_stream_chunks(&model_id, &events);
        return Ok(Sse::new(stream::iter(
            sse.into_iter().map(Ok::<_, std::convert::Infallible>),
        ))
        .keep_alive(KeepAlive::default())
        .into_response());
    }
    let events = generate_events(&jobs, req)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
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
        let vec = embed_vector(&jobs, text)
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
        total_tokens += text.split_whitespace().count().max(1) as u32;
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
    let path = write_temp_file(&bytes, "audio")?;
    let transcript = runa_media::transcribe_file(
        &path,
        &runa_media::AsrOptions {
            kind,
            pull: false,
            ..runa_media::AsrOptions::default()
        },
    )
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let _ = std::fs::remove_file(&path);
    Ok(Json(json!({"text": transcript.text})))
}

fn write_temp_file(bytes: &[u8], prefix: &str) -> Result<PathBuf, (StatusCode, String)> {
    let path = std::env::temp_dir().join(format!(
        "runa-{prefix}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    std::fs::write(&path, bytes).map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(path)
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
) -> Result<Vec<f32>, String> {
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

fn has_tool_calls(events: &[GenEvent]) -> bool {
    events
        .iter()
        .any(|e| matches!(e, GenEvent::ToolCalls(c) if !c.is_empty()))
}

fn stream_chunks(model: &str, events: &[GenEvent]) -> Vec<Event> {
    let id = completion_id();
    let mut out = Vec::new();
    for ev in events {
        let delta = match ev {
            GenEvent::Text(t) if !t.is_empty() => json!({"content": t}),
            GenEvent::Reasoning(t) if !t.is_empty() => json!({"reasoning_content": t}),
            GenEvent::ToolCalls(c) if !c.is_empty() => {
                json!({"tool_calls": openai_tool_calls(c, true)})
            }
            _ => continue,
        };
        let body = json!({
            "id": id,
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": null}]
        });
        out.push(Event::default().data(body.to_string()));
    }
    let finish = if has_tool_calls(events) {
        "tool_calls"
    } else {
        "stop"
    };
    let done = json!({
        "id": id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]
    });
    out.push(Event::default().data(done.to_string()));
    out.push(Event::default().data("[DONE]"));
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
            "finish_reason": if calls.is_empty() { "stop" } else { "tool_calls" }
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
        "stop_reason": anthropic_stop(calls),
        "usage": usage
    })
}

fn anthropic_stop(calls: &[ToolCall]) -> &'static str {
    if calls.is_empty() {
        "end_turn"
    } else {
        "tool_use"
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

fn anthropic_stream_chunks(model: &str, events: &[GenEvent]) -> Vec<Event> {
    let id = format!("msg-{}", completion_id().trim_start_matches("chatcmpl-"));
    let mut out = Vec::new();
    let (input_tokens, output_tokens) = events
        .iter()
        .find_map(|e| match e {
            GenEvent::Usage(u) => Some((u.prompt_tokens, u.generated_tokens)),
            _ => None,
        })
        .unwrap_or_default();
    let start = json!({
        "type": "message_start",
        "message": {
            "id": id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": null,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0}
        }
    });
    out.push(sse("message_start", start));
    // Each run of thinking / text deltas is one content block; every tool
    // call is its own block with the arguments in one `input_json_delta`.
    let mut idx = 0u32;
    let mut open: Option<&str> = None;
    let mut calls: &[ToolCall] = &[];
    let stop = |out: &mut Vec<Event>, idx: &mut u32, open: &mut Option<&str>| {
        if open.take().is_some() {
            out.push(sse(
                "content_block_stop",
                json!({"type": "content_block_stop", "index": *idx}),
            ));
            *idx += 1;
        }
    };
    for ev in events {
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
                calls = c;
                stop(&mut out, &mut idx, &mut open);
                for call in c {
                    let mut block = tool_use_block(call);
                    block["input"] = json!({});
                    out.push(sse(
                        "content_block_start",
                        json!({"type": "content_block_start", "index": idx, "content_block": block}),
                    ));
                    out.push(sse(
                        "content_block_delta",
                        json!({
                            "type": "content_block_delta",
                            "index": idx,
                            "delta": {"type": "input_json_delta", "partial_json": call.arguments}
                        }),
                    ));
                    open = Some("tool_use");
                    stop(&mut out, &mut idx, &mut open);
                }
                continue;
            }
            _ => continue,
        };
        if open != Some(kind) {
            stop(&mut out, &mut idx, &mut open);
            out.push(sse(
                "content_block_start",
                json!({"type": "content_block_start", "index": idx, "content_block": block}),
            ));
            open = Some(kind);
        }
        out.push(sse(
            "content_block_delta",
            json!({"type": "content_block_delta", "index": idx, "delta": delta}),
        ));
    }
    stop(&mut out, &mut idx, &mut open);
    out.push(sse(
        "message_delta",
        json!({
            "type": "message_delta",
            "delta": {"stop_reason": anthropic_stop(calls)},
            "usage": {"output_tokens": output_tokens}
        }),
    ));
    out.push(
        Event::default()
            .event("message_stop")
            .data(json!({"type": "message_stop"}).to_string()),
    );
    out
}

/// Parsed request body: chat messages plus optional media (vision frames,
/// PCM audio for the ASR route).
type ParsedBody = (Vec<ChatMessage>, Vec<VisionFrame>, Option<Vec<f32>>);

fn messages_from_body(messages: &[IncomingMessage]) -> Result<ParsedBody, String> {
    let mut out = Vec::new();
    let mut images = Vec::new();
    let mut audio_pcm: Option<Vec<f32>> = None;
    for m in messages {
        let text = content_text(m.content.as_ref(), &mut images, &mut audio_pcm)?;
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
                        content: content_text(p.content.as_ref(), &mut Vec::new(), &mut None)?,
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
    Ok((out, images, audio_pcm))
}

fn content_text(
    c: Option<&IncomingContent>,
    images: &mut Vec<VisionFrame>,
    audio_pcm: &mut Option<Vec<f32>>,
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
                        let path = image_url_to_path(&url)?;
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

fn image_url_to_path(url: &str) -> Result<PathBuf, String> {
    if url.starts_with("data:") {
        let rest = url.strip_prefix("data:").ok_or("malformed data URI")?;
        let (_meta, b64) = rest
            .split_once(',')
            .ok_or("malformed data URI: missing comma")?;
        let bytes = b64_decode(b64.trim())?;
        return write_temp_file(&bytes, "img").map_err(|(_, e)| e);
    }
    if url.starts_with("http://") || url.starts_with("https://") {
        return Err("remote image_url is not supported; use a data: URI".into());
    }
    let p = PathBuf::from(url);
    if p.is_file() {
        Ok(p)
    } else {
        Err(format!("image_url path not found: {url}"))
    }
}

fn decode_input_audio(part: &InputAudioPart) -> Result<Vec<f32>, String> {
    let bytes = b64_decode(part.data.trim())?;
    let ext = match part.format.as_deref() {
        Some("wav") | None => "wav",
        Some("mp3") => "mp3",
        other => return Err(format!("unsupported input_audio format: {:?}", other)),
    };
    let path = write_temp_file(&bytes, "in-audio").map_err(|(_, e)| e)?;
    let path = {
        let named = path.with_extension(ext);
        std::fs::rename(&path, &named).map_err(|e| e.to_string())?;
        named
    };
    let decoded = runa_media::decode_audio(&path).map_err(|e| e.to_string())?;
    let _ = std::fs::remove_file(&path);
    Ok(decoded.samples)
}

fn b64_decode(input: &str) -> Result<Vec<u8>, String> {
    const T: &[u8; 256] = &{
        let mut t = [255u8; 256];
        let chars = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut i = 0;
        while i < 64 {
            t[chars[i] as usize] = i as u8;
            i += 1;
        }
        t
    };
    let mut out = Vec::new();
    let mut buf = [0u8; 4];
    let mut n = 0usize;
    for &b in input.as_bytes() {
        if b == b'=' {
            break;
        }
        if b.is_ascii_whitespace() {
            continue;
        }
        let v = T[b as usize];
        if v == 255 {
            return Err("invalid base64".into());
        }
        buf[n] = v;
        n += 1;
        if n == 4 {
            out.push((buf[0] << 2) | (buf[1] >> 4));
            out.push((buf[1] << 4) | (buf[2] >> 2));
            out.push((buf[2] << 6) | buf[3]);
            n = 0;
        }
    }
    if n == 2 {
        out.push((buf[0] << 2) | (buf[1] >> 4));
    } else if n == 3 {
        out.push((buf[0] << 2) | (buf[1] >> 4));
        out.push((buf[1] << 4) | (buf[2] >> 2));
    }
    Ok(out)
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
            let _ = content_text(b.system.as_ref(), &mut Vec::new(), &mut None);
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
        let (out, _, _) = messages_from_body(&msgs).unwrap();
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
        let (out, imgs, aud) = messages_from_body(&msgs).unwrap();
        assert_eq!(out[0].content, "hi there");
        assert!(imgs.is_empty());
        assert!(aud.is_none());
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
}
