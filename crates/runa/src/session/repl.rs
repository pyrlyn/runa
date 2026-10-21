// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! Line-mode chat REPL (`runa chat`).

use std::path::PathBuf;

use runa_core::{BackendKind, ThinkOverrides};
use runa_engine::{
    GenerateRequest, LoadConfig, Mode, Placement, SamplingConfig, load, parse_rpc_list,
};
use runa_pool::engine;

use super::format::format_usage;
use super::history::{commit_history, push_tool_results, turn_messages};
use super::screen::run_tui;
use super::stream::drain_daemon;
use super::stream::{drain_local, push_tool_calls};
use super::{ChatEngine, Session, reload, reload_local, store_chat_usage};
use crate::cli::ToolArgs;
use crate::cli::input::{attach_prompt_cache, attach_prompt_cache_local, cli_think};
use crate::daemon::{daemon_available, daemon_generate};
use crate::placement::{parse_mode, resolve_model};
use crate::{config, mcp, tui};

#[allow(clippy::too_many_arguments)]
pub(crate) fn cmd_chat(
    model: Option<&str>,
    backend: &str,
    mode: &str,
    ctx: u32,
    threads: Option<i32>,
    max_load_percent: Option<u8>,
    lora: Vec<String>,
    think: Option<String>,
    think_budget: Option<u32>,
    effort: Option<String>,
    show_reasoning: bool,
    no_show_reasoning: bool,
    tui: bool,
    no_daemon: bool,
    rpc: Option<String>,
    tools: &ToolArgs,
) -> Result<(), String> {
    use rustyline::error::ReadlineError;
    use rustyline::history::FileHistory;
    use rustyline::{Config, Editor};

    let requested = BackendKind::parse(backend)
        .ok_or_else(|| format!("{backend}: --backend must be gguf | mistral | auto"))?;
    // P9.3: the intent is carried from here into every load path below and
    // rejected explicitly by `load` (no RPC backend in the pinned sys
    // crate) — never dropped silently, never sent to the daemon.
    let rpc_servers = rpc
        .as_deref()
        .map(parse_rpc_list)
        .transpose()?
        .unwrap_or_default();
    let mut mode = parse_mode(mode)?;
    let mut path = match model {
        Some(m) => resolve_model(m)?,
        None => {
            return Err(
                "chat needs a model path or alias (local file, hf: ref, or [models] alias)".into(),
            );
        }
    };
    let kind = engine::resolve_requested(requested, &path)?;
    let loras = config::resolve_loras(&lora, model.unwrap_or(""))?;
    let threads = config::resolve_threads(threads)?;
    // Startup cap check (warn-only): RAM pressure + thread share against
    // `[system] max_load_percent` (default 80).
    config::warn_if_over_system_limit(None, threads, max_load_percent)?;
    // Mistral requests always load a managed local backend (P9.2); gguf
    // requests prefer the daemon and fall back to a local load (P9.1).
    let mut engine = if kind == BackendKind::Mistral {
        if !loras.is_empty() {
            return Err("--lora needs the gguf backend".into());
        }
        if threads.is_some() {
            return Err("--threads needs the gguf backend".into());
        }
        if max_load_percent.is_some() {
            return Err("--max-load-percent needs the gguf backend".into());
        }
        if !rpc_servers.is_empty() {
            return Err("--rpc needs the gguf backend".into());
        }
        if !tools.mcp.is_empty() || !config::load_mcp_servers()?.is_empty() {
            return Err("--mcp tools need the gguf backend".into());
        }
        let mut managed = engine::LocalEngine::load(
            kind,
            &path,
            &Placement::from_mode(mode),
            &LoadConfig {
                n_ctx: ctx,
                loras: loras.clone(),
                ..LoadConfig::default()
            },
        )?;
        attach_prompt_cache(&mut managed, false, None)?;
        ChatEngine::Managed(managed)
    } else if !no_daemon && rpc_servers.is_empty() && tools.mcp.is_empty() && daemon_available() {
        // Daemon first: with no MCP servers the daemon serves every turn;
        // any dial failure below loads locally instead. `--no-daemon`
        // skips the probe. `--rpc` forces a local load: the daemon owns
        // placement and knows no RPC endpoints, so routing there would
        // silently drop the request (P9.3); the local `load` below errors
        // explicitly instead.
        ChatEngine::Daemon(None)
    } else {
        let mut local = load(
            &path,
            &Placement::from_mode(mode).with_rpc_servers(rpc_servers.clone()),
            &LoadConfig {
                n_ctx: ctx,
                loras: loras.clone(),
                threads,
                ..LoadConfig::default()
            },
        )
        .map_err(|e| e.to_string())?;
        attach_prompt_cache_local(&mut local, false, None)?;
        ChatEngine::Daemon(Some(local))
    };

    let config = Config::builder().auto_add_history(true).build();
    let mut rl: Editor<(), FileHistory> =
        Editor::with_config(config).map_err(|e| format!("readline: {e}"))?;
    if let Some(home) = config::home_dir() {
        let hist = home.join(".runa_history");
        let _ = rl.load_history(&hist);
    }
    let mut session = Session {
        path: path.clone(),
        backend: requested,
        mode,
        ctx,
        threads,
        history: Vec::new(),
        think: cli_think(
            think.as_deref(),
            think_budget,
            effort.as_deref(),
            show_reasoning,
            no_show_reasoning,
        )?,
        last_usage: None,
        hub: mcp::start(&tools.mcp)?,
        max_tool_rounds: tools.max_tool_rounds,
        loras,
        rpc_servers,
    };
    println!("runa chat ({}). /help for commands.", path.display());

    if tui {
        return run_tui(&mut session, &mut engine, &mut path, &mut mode);
    }

    let mut pending_line = String::new();
    loop {
        let prompt = if pending_line.is_empty() {
            "> "
        } else {
            "... "
        };
        match rl.readline(prompt) {
            Ok(line) => {
                // `\` continues on the next line (multi-line paste).
                if line.ends_with('\\') {
                    pending_line.push_str(&line[..line.len() - 1]);
                    pending_line.push('\n');
                    continue;
                }
                pending_line.push_str(&line);
                let input = std::mem::take(&mut pending_line);
                if input.trim().is_empty() {
                    continue;
                }
                if input.starts_with('/') {
                    if chat_command(&input, &mut session, &mut engine, &mut path, &mut mode)? {
                        break;
                    }
                    continue;
                }
                chat_turn(&mut engine, &input, &mut session);
            }
            Err(ReadlineError::Eof) | Err(ReadlineError::Interrupted) => break,
            Err(e) => {
                eprintln!("input: {e}");
                break;
            }
        }
    }
    if let Some(home) = config::home_dir() {
        let hist = home.join(".runa_history");
        let _ = rl.save_history(&hist);
    }
    Ok(())
}

/// Returns true when the session should exit. `loaded` is `None` while
/// the daemon serves the session: `/mode` is daemon-owned there and
/// `/model` only re-points the next request.
fn chat_command(
    input: &str,
    session: &mut Session,
    engine: &mut ChatEngine,
    path: &mut PathBuf,
    mode: &mut Mode,
) -> Result<bool, String> {
    let mut parts = input[1..].split_whitespace();
    match parts.next().unwrap_or("") {
        "quit" | "exit" | "q" => Ok(true),
        "help" => {
            for line in tui::SLASH_HELP {
                println!("{line}");
            }
            println!("end a line with \\ to continue it (multi-line paste)");
            Ok(false)
        }
        "mode" => {
            let m = parts.next().ok_or("usage: /mode <cpu|gpu|hybrid>")?;
            match engine {
                ChatEngine::Managed(managed) => {
                    if !managed.is_gguf() {
                        println!("(the mistral backend manages devices itself; mode is gguf-only)");
                        return Ok(false);
                    }
                    *mode = parse_mode(m)?;
                    session.mode = *mode;
                    reload(session, managed, path)?;
                }
                ChatEngine::Daemon(loaded) => {
                    *mode = parse_mode(m)?;
                    session.mode = *mode;
                    match loaded {
                        Some(local) => reload_local(session, local, path)?,
                        None => println!(
                            "/mode is owned by `runa daemon` in this session; restart the daemon with --mode {m} to change it"
                        ),
                    }
                }
            }
            Ok(false)
        }
        "model" => {
            let m = parts.next().ok_or("usage: /model <local-path>")?;
            *path = resolve_model(m)?;
            session.path = path.clone();
            if !session.history.is_empty() {
                session.history.clear();
                println!("(history cleared: new model)");
            }
            match engine {
                ChatEngine::Managed(managed) => reload(session, managed, path)?,
                ChatEngine::Daemon(loaded) => match loaded {
                    Some(local) => reload_local(session, local, path)?,
                    // The daemon loads the new model on the next turn.
                    None => println!("model: {} (daemon loads on next turn)", path.display()),
                },
            }
            Ok(false)
        }
        "think" => {
            let rest: String = parts.collect::<Vec<_>>().join(" ");
            if rest.is_empty() {
                println!("think: {}", session.think);
                return Ok(false);
            }
            session.think = session
                .think
                .apply(&ThinkOverrides::from_slash_args(&rest)?)?;
            println!("think: {}", session.think);
            Ok(false)
        }
        "reset" => {
            match engine {
                ChatEngine::Managed(managed) => {
                    managed.reset_context().map_err(|e| e.to_string())?;
                }
                ChatEngine::Daemon(loaded) => {
                    if let Some(local) = loaded {
                        local.reset_context().map_err(|e| e.to_string())?;
                    }
                    // Daemon requests are independent: nothing to clear.
                }
            }
            session.history.clear();
            println!("(context cleared)");
            Ok(false)
        }
        "usage" => {
            match &session.last_usage {
                Some(u) => println!("{}", format_usage(u)),
                None => println!("(no turn yet)"),
            }
            Ok(false)
        }
        other => {
            println!("unknown command /{other} — /help lists commands");
            Ok(false)
        }
    }
}

fn chat_turn(engine: &mut ChatEngine, input: &str, session: &mut Session) {
    let (messages, trimmed) = turn_messages(session, input, engine);
    if trimmed {
        println!("(history trimmed to fit ctx)");
    }
    let mut req = GenerateRequest {
        messages,
        sampling: SamplingConfig::default(),
        max_tokens: 512,
        think: session.think,
        tools: session.hub.as_ref().map(|h| h.tools_json().to_string()),
        ..GenerateRequest::default()
    };
    match engine {
        ChatEngine::Managed(managed) => {
            let mut usage = None;
            let mut answer = String::new();
            let done = mcp::tool_loop(
                session.max_tool_rounds,
                mcp::call_with(session.hub.as_ref()),
                |results| {
                    push_tool_results(&mut req.messages, results);
                    let stream = managed.generate(req.clone()).map_err(|e| e.to_string())?;
                    let turn = drain_local(stream, true)?;
                    if turn.usage.is_some() {
                        usage = turn.usage.clone();
                    }
                    answer = turn.text.clone();
                    Ok(push_tool_calls(&mut req.messages, &turn, true))
                },
            );
            store_chat_usage(session, usage);
            match done {
                Ok(()) => {
                    commit_history(session, &req.messages, &answer);
                    println!();
                }
                Err(e) => eprintln!("\ngenerate: {e}"),
            }
        }
        ChatEngine::Daemon(loaded) => {
            // The daemon serves the whole turn (no MCP hub exists in daemon
            // sessions, so no tool loop runs there).
            if loaded.is_none() {
                match daemon_generate(&session.path.display().to_string(), &req)
                    .map_err(|e| {
                        format!(
                            "daemon: {e}\n(hint: restart `runa daemon`, or chat with --no-daemon)"
                        )
                    })
                    .and_then(|events| drain_daemon(&events, true))
                {
                    Ok(turn) => {
                        commit_history(session, &req.messages, &turn.text);
                        println!();
                        store_chat_usage(session, turn.usage);
                    }
                    Err(e) => eprintln!("\ngenerate: {e}"),
                }
                return;
            }
            let loaded = loaded.as_mut().expect("daemon branch returned above");
            let mut usage = None;
            let mut answer = String::new();
            let done = mcp::tool_loop(
                session.max_tool_rounds,
                mcp::call_with(session.hub.as_ref()),
                |results| {
                    push_tool_results(&mut req.messages, results);
                    let stream = loaded.generate(req.clone()).map_err(|e| e.to_string())?;
                    let turn = drain_local(stream, true)?;
                    if turn.usage.is_some() {
                        usage = turn.usage.clone();
                    }
                    answer = turn.text.clone();
                    Ok(push_tool_calls(&mut req.messages, &turn, true))
                },
            );
            store_chat_usage(session, usage);
            match done {
                Ok(()) => {
                    commit_history(session, &req.messages, &answer);
                    println!();
                }
                Err(e) => eprintln!("\ngenerate: {e}"),
            }
        }
    }
}
