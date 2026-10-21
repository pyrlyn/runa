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

#[cfg(test)]
mod backend_tests {
    use super::*;
    use crate::daemon::client::drain_daemon;
    use crate::daemon_proto;
    use runa_core::{Backend, Request, ScriptedBackend};
    use runa_engine::{GenEvent, StopReason, ToolCall};

    #[test]
    fn tool_loop_runs_on_a_scripted_backend_without_a_model() {
        let weather = ToolCall {
            id: "t1".into(),
            name: "weather".into(),
            arguments: "{}".into(),
        };
        let mut backend = ScriptedBackend::new([
            vec![
                GenEvent::ToolCalls(vec![weather.clone()]),
                GenEvent::Done(StopReason::Eos),
            ],
            vec![
                GenEvent::Text("sunny".into()),
                GenEvent::Usage(Usage {
                    prompt_tokens: 9,
                    generated_tokens: 1,
                    ..Usage::default()
                }),
                GenEvent::Done(StopReason::Eos),
            ],
        ]);
        let mut req = Request {
            messages: vec![ChatMessage::user("weather?")],
            ..Request::default()
        };
        let mut answer = String::new();
        let mut usage = None;
        mcp::tool_loop(
            4,
            |c| format!("{}: sunny", c.name),
            |results| {
                push_tool_results(&mut req.messages, results);
                let stream = backend.generate(&req).map_err(|e| e.to_string())?;
                let turn = drain_local(stream, false)?;
                if turn.usage.is_some() {
                    usage = turn.usage.clone();
                }
                answer = turn.text.clone();
                Ok(push_tool_calls(&mut req.messages, &turn, false))
            },
        )
        .unwrap();
        assert_eq!(answer, "sunny");
        assert_eq!(usage.unwrap().prompt_tokens, 9);
        let roles: Vec<&str> = req.messages.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(roles, ["user", "assistant", "tool"]);
        assert_eq!(req.messages[1].tool_calls, vec![weather]);
        assert_eq!(req.messages[2].tool_call_id.as_deref(), Some("t1"));
        assert_eq!(req.messages[2].content, "weather: sunny");
        // The second round sent the tool answer back to the backend.
        assert_eq!(backend.requests.len(), 2);
        assert_eq!(backend.requests[1].messages, req.messages);
    }

    #[test]
    fn daemon_events_drain_through_the_core_event_path() {
        use daemon_proto::DaemonEvent;
        let events = [
            DaemonEvent::Text { text: "hi".into() },
            DaemonEvent::Usage {
                prompt_tokens: 2,
                generated_tokens: 1,
                reasoning_tokens: 0,
            },
            DaemonEvent::Done {
                stop: "max_tokens".into(),
            },
        ];
        let turn = drain_daemon(&events, false).unwrap();
        assert_eq!(turn.text, "hi");
        assert_eq!(turn.reason, StopReason::MaxTokens);
        assert_eq!(turn.usage.unwrap().generated_tokens, 1);
        let failed = [DaemonEvent::Error {
            message: "boom".into(),
        }];
        assert_eq!(drain_daemon(&failed, false).err().as_deref(), Some("boom"));
    }
}
