// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! Chat session: state shared by the REPL and the full-screen TUI loop.

mod format;
mod history;
mod repl;
mod screen;
mod stream;

use std::path::{Path, PathBuf};

use runa_core::{BackendKind, ThinkConfig};
use runa_engine::{ChatMessage, LoadConfig, LoraSpec, Mode, Placement, Usage, load};
use runa_pool::engine;

use crate::cli::input::{attach_prompt_cache, attach_prompt_cache_local};
use crate::mcp;
use crate::usage_log;

pub(crate) use format::{json_escape, run_json};
pub(crate) use history::push_tool_results;
pub(crate) use repl::cmd_chat;
pub(crate) use stream::{LocalTurn, drain_local, push_tool_calls};

/// REPL session: model path + pending mode + stored think args (P3).
struct Session {
    path: PathBuf,
    /// Requested `--backend` (possibly `Auto`; re-resolved on `/model`).
    backend: BackendKind,
    mode: Mode,
    ctx: u32,
    /// Worker threads re-applied on every `/model` + `/mode` reload (P10.2).
    threads: Option<i32>,
    /// Conversation so far, sent with every turn (P10.9). `/reset` and
    /// `/model` clear it; `/mode` keeps it (same weights).
    history: Vec<ChatMessage>,
    think: ThinkConfig,
    last_usage: Option<Usage>,
    /// MCP servers for the tool loop (P8.3).
    hub: Option<mcp::McpHub>,
    max_tool_rounds: u32,
    /// LoRA adapters re-applied on every `/model` + `/mode` reload (P8.5).
    loras: Vec<LoraSpec>,
    /// llama.cpp RPC endpoints (`--rpc`, P9.3): carried into every
    /// (re)load so the intent can never be dropped silently.
    rpc_servers: Vec<String>,
}

impl Session {
    /// Placement for every chat (re)load: REPL mode plus the startup `--rpc`
    /// intent. A non-empty list is rejected explicitly by `load`.
    fn placement(&self) -> Placement {
        Placement::from_mode(self.mode).with_rpc_servers(self.rpc_servers.clone())
    }
}

/// Chat-time engine (P9.1 + P9.2): mistral requests always run a managed
/// local backend; gguf requests prefer the daemon and fall back to a
/// local load (`Daemon(None)` = the daemon serves the session).
enum ChatEngine {
    Managed(engine::LocalEngine),
    Daemon(Option<runa_engine::LoadedModel>),
}

fn reload(session: &Session, engine: &mut engine::LocalEngine, path: &Path) -> Result<(), String> {
    let kind = engine::resolve_requested(session.backend, path)?;
    let fresh = engine::LocalEngine::load(
        kind,
        path,
        &session.placement(),
        &LoadConfig {
            n_ctx: session.ctx,
            loras: session.loras.clone(),
            threads: session.threads,
            ..LoadConfig::default()
        },
    )?;
    *engine = fresh;
    attach_prompt_cache(engine, false, None)?;
    Ok(())
}

/// `reload` for a bare ggml load (P9.1 daemon path): the caller holds a
/// `LoadedModel`, not a `LocalEngine`.
fn reload_local(
    session: &Session,
    loaded: &mut runa_engine::LoadedModel,
    path: &Path,
) -> Result<(), String> {
    let fresh = load(
        path,
        &session.placement(),
        &LoadConfig {
            n_ctx: session.ctx,
            loras: session.loras.clone(),
            threads: session.threads,
            ..LoadConfig::default()
        },
    )
    .map_err(|e| e.to_string())?;
    *loaded = fresh;
    attach_prompt_cache_local(loaded, false, None)?;
    Ok(())
}

fn model_label(path: &Path) -> String {
    path.file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| path.display().to_string())
}

fn store_chat_usage(session: &mut Session, usage: Option<Usage>) -> Option<Usage> {
    if let Some(u) = &usage {
        usage_log::record_usage("chat", &model_label(&session.path), u);
        session.last_usage = Some(u.clone());
    }
    usage
}
