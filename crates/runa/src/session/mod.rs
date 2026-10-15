// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! Chat session: history, the REPL, and the full-screen TUI loop.

use std::io::{self, Write};
use std::path::{Path, PathBuf};

use runa_core::{BackendKind, ThinkConfig, ThinkOverrides};
use runa_engine::{
    ChatMessage, GenEvent, GenerateRequest, LoadConfig, LoraSpec, Mode, Placement, SamplingConfig,
    StopReason, ToolCall, Usage, load, parse_rpc_list,
};

use crate::cli::{
    ToolArgs, attach_prompt_cache, attach_prompt_cache_local, cli_think, daemon_generate,
    drain_daemon, json_escape, parse_mode, resolve_model,
};
use crate::config;
use crate::daemon_proto;
use crate::mcp;
use crate::tui;
use runa_pool::engine;

pub(crate) struct LocalTurn {
    pub(crate) text: String,
    pub(crate) calls: Vec<ToolCall>,
    pub(crate) usage: Option<Usage>,
    pub(crate) reason: StopReason,
}

/// Drain a generation; with `print`, answer text streams to stdout and
/// reasoning to stderr. Generic over both backends: ggml `Generation` and
/// `MistralGeneration` both yield `Result<GenEvent, EngineError>`.
pub(crate) fn drain_local(
    stream: impl Iterator<Item = Result<GenEvent, runa_engine::EngineError>>,
    print: bool,
) -> Result<LocalTurn, String> {
    let mut turn = LocalTurn {
        text: String::new(),
        calls: Vec::new(),
        usage: None,
        reason: StopReason::MaxTokens,
    };
    for ev in stream {
        match ev.map_err(|e| e.to_string())? {
            GenEvent::Text(piece) => {
                if print {
                    print!("{piece}");
                    io::stdout().flush().map_err(|e| format!("stdout: {e}"))?;
                }
                turn.text.push_str(&piece);
            }
            GenEvent::Reasoning(piece) => {
                if print {
                    eprint!("{piece}");
                    io::stderr().flush().map_err(|e| format!("stderr: {e}"))?;
                }
            }
            GenEvent::Usage(u) => turn.usage = Some(u),
            GenEvent::ToolCalls(calls) => turn.calls = calls,
            GenEvent::Done(r) => turn.reason = r,
        }
    }
    Ok(turn)
}

/// Answer the previous round's calls with `tool` messages (P8.3).
pub(crate) fn push_tool_results(messages: &mut Vec<ChatMessage>, results: &[(ToolCall, String)]) {
    messages.extend(results.iter().map(|(call, out)| ChatMessage {
        role: "tool".into(),
        content: out.clone(),
        tool_call_id: Some(call.id.clone()),
        ..ChatMessage::default()
    }));
}

/// Record the assistant turn that made tool calls; returns the calls.
pub(crate) fn push_tool_calls(
    messages: &mut Vec<ChatMessage>,
    turn: &LocalTurn,
    print: bool,
) -> Vec<ToolCall> {
    if !turn.calls.is_empty() {
        if print && !turn.text.is_empty() {
            println!();
        }
        messages.push(ChatMessage {
            role: "assistant".into(),
            content: turn.text.clone(),
            tool_calls: turn.calls.clone(),
            ..ChatMessage::default()
        });
    }
    turn.calls.clone()
}

/// Rough token estimate for history trimming (P10.9): ~4 chars per
/// token plus per-message overhead. Still the counter on paths without
/// a tokenizer (daemon socket, mistral backend) — a guard rail, not a
/// bill; the engine still fails loudly past real ctx.
fn estimate_history_tokens(messages: &[ChatMessage]) -> u64 {
    messages
        .iter()
        .map(|m| m.content.len() as u64 / 4 + 4)
        .sum()
}

/// Exact history size via an injected per-message token counter
/// (P11.4): local gguf turns pass the loaded tokenizer here, so unit
/// tests can pin the trim against a stub. The +4 per-message overhead
/// matches the estimate (template markers stay covered on every path).
#[cfg(test)]
fn count_history_tokens_with(messages: &[ChatMessage], count_one: &dyn Fn(&str) -> u64) -> u64 {
    messages.iter().map(|m| count_one(&m.content) + 4).sum()
}

/// Drop oldest turns while `count` exceeds `budget` (P10.9).
/// Cuts whole turns (up to the next `user` message) so no `tool`
/// message is orphaned from its `assistant` call; the newest message
/// (the current user turn) is never dropped. Returns dropped count.
fn trim_history_with(
    history: &mut Vec<ChatMessage>,
    budget: u64,
    count: &dyn Fn(&[ChatMessage]) -> u64,
) -> usize {
    let mut dropped = 0;
    while history.len() > 1 && count(history) > budget {
        let cut = history
            .iter()
            .skip(1)
            .position(|m| m.role == "user")
            .map_or(history.len() - 1, |i| i + 1)
            .max(1);
        history.drain(..cut);
        dropped += cut;
    }
    dropped
}

/// History size for this turn (P11.4): local gguf loads count real
/// tokens through the model's own tokenizer; the daemon socket path
/// has no tokenizer (and the mistral backend exposes none), so those
/// keep the chars/4 estimate.
fn history_tokens(engine: &ChatEngine, messages: &[ChatMessage]) -> u64 {
    match engine {
        ChatEngine::Managed(managed) => managed
            .count_history_tokens(messages)
            .unwrap_or_else(|| estimate_history_tokens(messages)),
        ChatEngine::Daemon(Some(loaded)) => loaded.count_history_tokens(messages),
        ChatEngine::Daemon(None) => estimate_history_tokens(messages),
    }
}

/// Start the next turn's messages: history + new user input, trimmed to
/// ~75% of ctx. The guard is re-checked against exact tokenizer counts
/// on local gguf paths and against the estimate over the daemon socket
/// (P11.4). Returns the messages plus whether a trim happened — the
/// caller reports it (stdout in the REPL, a notice in the TUI, where
/// `println!` would corrupt the alternate screen).
fn turn_messages(
    session: &mut Session,
    input: &str,
    engine: &ChatEngine,
) -> (Vec<ChatMessage>, bool) {
    session.history.push(ChatMessage::user(input));
    let budget = u64::from(session.ctx) * 3 / 4;
    let trimmed = trim_history_with(&mut session.history, budget, &|msgs| {
        history_tokens(engine, msgs)
    }) > 0;
    (session.history.clone(), trimmed)
}

/// Record a finished turn: the sent messages rode the whole tool loop,
/// so they already hold every tool round — just add the final answer.
fn commit_history(session: &mut Session, sent: &[ChatMessage], answer: &str) {
    session.history = sent.to_vec();
    if !answer.trim().is_empty() {
        session.history.push(ChatMessage {
            role: "assistant".into(),
            content: answer.to_owned(),
            ..ChatMessage::default()
        });
    }
}

/// One-line usage for `/usage` (REPL + TUI): reasoning appears only when
/// the turn actually thought (P10.4).
fn format_usage(u: &Usage) -> String {
    let thinking = if u.reasoning_tokens > 0 {
        format!(" (reasoning {})", u.reasoning_tokens)
    } else {
        String::new()
    };
    format!(
        "prompt {} / generated {}{thinking} · {:.1} pp tok/s · {:.1} tg tok/s",
        u.prompt_tokens, u.generated_tokens, u.pp_toks_per_s, u.tg_toks_per_s
    )
}

pub(crate) fn run_json(text: &str, usage: &Usage, reason: &StopReason) -> String {
    let stop = match reason {
        StopReason::Eos => "eos".to_owned(),
        StopReason::MaxTokens => "max_tokens".to_owned(),
        StopReason::StopString(s) => format!("stop:{}", json_escape(s)),
    };
    format!(
        "{{\"text\":\"{}\",\"usage\":{{\"prompt_tokens\":{},\"generated_tokens\":{},\"reasoning_tokens\":{},\"pp_toks_per_s\":{:.1},\"tg_toks_per_s\":{:.1}}},\"stop\":\"{stop}\"}}",
        json_escape(text),
        usage.prompt_tokens,
        usage.generated_tokens,
        usage.reasoning_tokens,
        usage.pp_toks_per_s,
        usage.tg_toks_per_s,
    )
}

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
            if usage.is_some() {
                session.last_usage = usage;
            }
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
                        session.last_usage = turn.usage;
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
            if usage.is_some() {
                session.last_usage = usage;
            }
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

/// One cheap probe: is a daemon listening? The probe connection is
/// dropped immediately; real turns dial per request.
fn daemon_available() -> bool {
    #[cfg(unix)]
    {
        std::os::unix::net::UnixStream::connect(daemon_proto::default_socket_path()).is_ok()
    }
    #[cfg(not(unix))]
    {
        false
    }
}

/// Full-screen chat event loop (`runa chat --tui`, P8.6). Draws `tui::view`
/// every key and every streamed token; slash lines run through
/// `execute_tui_slash`, the same commands as `chat_command`.
type TuiTerm = ratatui::Terminal<ratatui::backend::CrosstermBackend<std::io::Stdout>>;

fn tui_mode_name(mode: Mode) -> &'static str {
    match mode {
        Mode::Cpu => "cpu",
        Mode::Gpu => "gpu",
        Mode::Hybrid => "hybrid",
    }
}

fn tui_draw(term: &mut TuiTerm, tui: &tui::ChatTui) -> Result<(), String> {
    term.draw(|frame| {
        if let Some(pos) = tui::view(tui, frame.area(), frame.buffer_mut()) {
            frame.set_cursor_position(pos);
        }
    })
    .map_err(|e| format!("tui: {e}"))?;
    Ok(())
}

fn run_tui(
    session: &mut Session,
    engine: &mut ChatEngine,
    path: &mut PathBuf,
    mode: &mut Mode,
) -> Result<(), String> {
    use crossterm::{
        event::{self, DisableBracketedPaste, EnableBracketedPaste, Event},
        execute,
        terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
    };
    use ratatui::{Terminal, backend::CrosstermBackend};

    enable_raw_mode().map_err(|e| format!("tui: {e}"))?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableBracketedPaste)
        .map_err(|e| format!("tui: {e}"))?;
    let mut term = Terminal::new(CrosstermBackend::new(stdout)).map_err(|e| format!("tui: {e}"))?;
    let mut tui = tui::ChatTui::new(
        &path.display().to_string(),
        tui_mode_name(*mode),
        session.ctx,
    );
    tui.push(tui::Message::notice(
        "runa chat — Enter sends, Shift+Enter breaks the line, /help lists commands.",
    ));

    loop {
        tui_draw(&mut term, &tui)?;
        match event::read().map_err(|e| format!("input: {e}"))? {
            Event::Key(key) => match tui::on_key(&mut tui, key) {
                tui::KeyAction::Nothing => {}
                tui::KeyAction::Submit(text) => {
                    tui.push(tui::Message::user(&text));
                    tui_turn(&mut term, &mut tui, engine, session, &text);
                }
                tui::KeyAction::Slash(cmd) => {
                    if execute_tui_slash(&mut tui, session, engine, path, mode, cmd)? {
                        break;
                    }
                }
                tui::KeyAction::Quit => break,
            },
            Event::Paste(text) => tui.insert_text(&text),
            _ => {}
        }
    }
    let _ = disable_raw_mode();
    let _ = execute!(
        term.backend_mut(),
        LeaveAlternateScreen,
        DisableBracketedPaste
    );
    let _ = term.show_cursor();
    Ok(())
}

/// The REPL's slash commands against TUI state: notices land in the
/// transcript instead of stdout, and bad arguments stay in the session
/// instead of exiting it. Returns true when the session should exit.
fn execute_tui_slash(
    tui: &mut tui::ChatTui,
    session: &mut Session,
    engine: &mut ChatEngine,
    path: &mut PathBuf,
    mode: &mut Mode,
    cmd: tui::SlashCmd,
) -> Result<bool, String> {
    match cmd {
        tui::SlashCmd::Quit => Ok(true),
        tui::SlashCmd::Help => {
            let mut help = tui::SLASH_HELP.join("\n");
            help.push_str("\nEnter sends · Shift+Enter breaks the line · Ctrl+R toggles reasoning");
            tui.push(tui::Message::notice(&help));
            Ok(false)
        }
        tui::SlashCmd::Mode(raw) => match parse_mode(&raw) {
            Ok(next) => {
                match engine {
                    ChatEngine::Managed(managed) => {
                        if !managed.is_gguf() {
                            tui.push(tui::Message::notice(
                                "the mistral backend manages devices itself; mode is gguf-only",
                            ));
                            return Ok(false);
                        }
                        *mode = next;
                        session.mode = next;
                        reload(session, managed, path)?;
                        tui.set_mode(tui_mode_name(next));
                        tui.push(tui::Message::notice(&format!("mode: {raw}")));
                    }
                    ChatEngine::Daemon(loaded) => {
                        *mode = next;
                        session.mode = next;
                        match loaded {
                            Some(local) => {
                                reload_local(session, local, path)?;
                                tui.set_mode(tui_mode_name(next));
                                tui.push(tui::Message::notice(&format!("mode: {raw}")));
                            }
                            None => tui.push(tui::Message::notice(
                                "/mode is owned by `runa daemon`; restart it with --mode to change",
                            )),
                        }
                    }
                }
                Ok(false)
            }
            Err(e) => {
                tui.push(tui::Message::notice(&e));
                Ok(false)
            }
        },
        tui::SlashCmd::Model(raw) => match resolve_model(&raw) {
            Ok(next) => {
                *path = next;
                session.path = path.clone();
                if !session.history.is_empty() {
                    session.history.clear();
                    tui.push(tui::Message::notice("(history cleared: new model)"));
                }
                match engine {
                    ChatEngine::Managed(managed) => reload(session, managed, path)?,
                    ChatEngine::Daemon(loaded) => {
                        if let Some(local) = loaded {
                            reload_local(session, local, path)?;
                        }
                        // Without a local model the daemon loads the new
                        // model on the next turn.
                    }
                }
                tui.set_model(&path.display().to_string());
                tui.push(tui::Message::notice(&format!("model: {}", path.display())));
                Ok(false)
            }
            Err(e) => {
                tui.push(tui::Message::notice(&e));
                Ok(false)
            }
        },
        tui::SlashCmd::Think(rest) => {
            if rest.is_empty() {
                tui.push(tui::Message::notice(&format!("think: {}", session.think)));
                return Ok(false);
            }
            match ThinkOverrides::from_slash_args(&rest).and_then(|o| session.think.apply(&o)) {
                Ok(think) => {
                    session.think = think;
                    tui.push(tui::Message::notice(&format!("think: {}", session.think)));
                }
                Err(e) => tui.push(tui::Message::notice(&e)),
            }
            Ok(false)
        }
        tui::SlashCmd::Reset => {
            match engine {
                ChatEngine::Managed(managed) => {
                    managed.reset_context().map_err(|e| e.to_string())?;
                }
                ChatEngine::Daemon(loaded) => {
                    if let Some(local) = loaded {
                        local.reset_context().map_err(|e| e.to_string())?;
                    }
                }
            }
            session.history.clear();
            tui.push(tui::Message::notice("(context cleared)"));
            Ok(false)
        }
        tui::SlashCmd::Usage => {
            match &session.last_usage {
                Some(u) => tui.push(tui::Message::notice(&format_usage(u))),
                None => tui.push(tui::Message::notice("(no turn yet)")),
            }
            Ok(false)
        }
        tui::SlashCmd::Unknown(msg) => {
            tui.push(tui::Message::notice(&msg));
            Ok(false)
        }
    }
}

/// One TUI turn: like `chat_turn`, but answer and reasoning stream token by
/// token into the transcript (the TUI redraws after every event) and tool
/// calls surface as notices. Usage lands in the session and the status bar.
fn tui_turn(
    term: &mut TuiTerm,
    tui: &mut tui::ChatTui,
    engine: &mut ChatEngine,
    session: &mut Session,
    input: &str,
) {
    let (messages, trimmed) = turn_messages(session, input, engine);
    if trimmed {
        tui.push(tui::Message::notice("(history trimmed to fit ctx)"));
    }
    let mut req = GenerateRequest {
        messages,
        sampling: SamplingConfig::default(),
        max_tokens: 512,
        think: session.think,
        tools: session.hub.as_ref().map(|h| h.tools_json().to_string()),
        ..GenerateRequest::default()
    };
    tui.set_busy(true);
    match engine {
        ChatEngine::Managed(engine) => {
            let mut usage = None;
            let mut answer = String::new();
            let done = mcp::tool_loop(
                session.max_tool_rounds,
                mcp::call_with(session.hub.as_ref()),
                |results| {
                    push_tool_results(&mut req.messages, results);
                    let stream = engine.generate(req.clone()).map_err(|e| e.to_string())?;
                    let mut calls = Vec::new();
                    for ev in stream {
                        match ev.map_err(|e| e.to_string())? {
                            GenEvent::Text(piece) => {
                                answer.push_str(&piece);
                                tui.extend_last(tui::Role::Assistant, &piece);
                            }
                            GenEvent::Reasoning(piece) => {
                                tui.extend_last(tui::Role::Reasoning, &piece);
                            }
                            GenEvent::Usage(u) => usage = Some(u),
                            GenEvent::ToolCalls(made) => calls = made,
                            GenEvent::Done(_) => {}
                        }
                        if tui_draw(term, tui).is_err() {
                            break;
                        }
                    }
                    for call in &calls {
                        tui.push(tui::Message::notice(&format!("tool call: {}", call.name)));
                    }
                    if !calls.is_empty() {
                        req.messages.push(ChatMessage {
                            role: "assistant".into(),
                            content: String::new(),
                            tool_calls: calls.clone(),
                            ..ChatMessage::default()
                        });
                    }
                    Ok(calls)
                },
            );
            tui.set_busy(false);
            if usage.is_some() {
                session.last_usage = usage.clone();
            }
            if done.is_ok() {
                commit_history(session, &req.messages, &answer);
            }
            if let Some(u) = usage {
                tui.set_status(
                    u.tg_toks_per_s,
                    u.prompt_tokens.saturating_add(u.generated_tokens),
                );
            }
            match done {
                Ok(()) => {}
                Err(e) => tui.push(tui::Message::notice(&format!("generate: {e}"))),
            }
            let _ = tui_draw(term, tui);
        }
        ChatEngine::Daemon(loaded) => {
            // Daemon sessions have no MCP hub, so the whole turn is one request.
            if loaded.is_none() {
                match daemon_generate(&session.path.display().to_string(), &req) {
                    Ok(events) => {
                        let mut usage = None;
                        let mut answer = String::new();
                        for ev in &events {
                            match ev {
                                daemon_proto::DaemonEvent::Text { text } => {
                                    answer.push_str(text);
                                    tui.extend_last(tui::Role::Assistant, text);
                                }
                                daemon_proto::DaemonEvent::Reasoning { text } => {
                                    tui.extend_last(tui::Role::Reasoning, text);
                                }
                                daemon_proto::DaemonEvent::ToolCalls { calls } => {
                                    for call in calls {
                                        tui.push(tui::Message::notice(&format!(
                                            "tool call: {}",
                                            call.name
                                        )));
                                    }
                                }
                                daemon_proto::DaemonEvent::Usage {
                                    prompt_tokens,
                                    generated_tokens,
                                    reasoning_tokens,
                                } => {
                                    usage = Some(Usage {
                                        prompt_tokens: *prompt_tokens,
                                        generated_tokens: *generated_tokens,
                                        reasoning_tokens: *reasoning_tokens,
                                        pp_toks_per_s: 0.0,
                                        tg_toks_per_s: 0.0,
                                    });
                                }
                                daemon_proto::DaemonEvent::Done { .. } => {}
                                daemon_proto::DaemonEvent::Error { message } => {
                                    tui.push(tui::Message::notice(&format!("generate: {message}")));
                                }
                            }
                            if tui_draw(term, tui).is_err() {
                                break;
                            }
                        }
                        if usage.is_some() {
                            session.last_usage = usage;
                        }
                        commit_history(session, &req.messages, &answer);
                    }
                    Err(e) => tui.push(tui::Message::notice(&format!("generate: daemon: {e}"))),
                }
                tui.set_busy(false);
                let _ = tui_draw(term, tui);
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
                    let mut calls = Vec::new();
                    for ev in stream {
                        match ev.map_err(|e| e.to_string())? {
                            GenEvent::Text(piece) => {
                                answer.push_str(&piece);
                                tui.extend_last(tui::Role::Assistant, &piece);
                            }
                            GenEvent::Reasoning(piece) => {
                                tui.extend_last(tui::Role::Reasoning, &piece);
                            }
                            GenEvent::Usage(u) => usage = Some(u),
                            GenEvent::ToolCalls(made) => calls = made,
                            GenEvent::Done(_) => {}
                        }
                        if tui_draw(term, tui).is_err() {
                            break;
                        }
                    }
                    for call in &calls {
                        tui.push(tui::Message::notice(&format!("tool call: {}", call.name)));
                    }
                    if !calls.is_empty() {
                        req.messages.push(ChatMessage {
                            role: "assistant".into(),
                            content: String::new(),
                            tool_calls: calls.clone(),
                            ..ChatMessage::default()
                        });
                    }
                    Ok(calls)
                },
            );
            tui.set_busy(false);
            if usage.is_some() {
                session.last_usage = usage.clone();
            }
            if done.is_ok() {
                commit_history(session, &req.messages, &answer);
            }
            if let Some(u) = usage {
                tui.set_status(
                    u.tg_toks_per_s,
                    u.prompt_tokens.saturating_add(u.generated_tokens),
                );
            }
            match done {
                Ok(()) => {}
                Err(e) => tui.push(tui::Message::notice(&format!("generate: {e}"))),
            }
            let _ = tui_draw(term, tui);
        }
    }
}

#[cfg(test)]
mod history_tests {
    use super::*;

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: role.into(),
            content: content.into(),
            ..ChatMessage::default()
        }
    }

    #[test]
    fn trim_keeps_everything_under_budget() {
        let mut h = vec![msg("user", "hi"), msg("assistant", "hello")];
        assert_eq!(
            trim_history_with(&mut h, 10_000, &estimate_history_tokens),
            0
        );
        assert_eq!(h.len(), 2);
    }

    #[test]
    fn trim_drops_oldest_turns_first_and_never_the_newest() {
        // P10.9: whole turns go, the current user turn stays.
        let mut h = vec![
            msg("user", &"old question ".repeat(50)),
            msg("assistant", &"old answer ".repeat(50)),
            msg("user", &"mid question ".repeat(50)),
            msg("assistant", &"mid answer ".repeat(50)),
            msg("user", "new"),
        ];
        let dropped = trim_history_with(&mut h, 200, &estimate_history_tokens);
        assert!(dropped >= 2, "oldest turn goes first");
        assert_eq!(h.last().unwrap().content, "new");
        assert!(estimate_history_tokens(&h) <= 200, "{h:?}");
        // Roles still alternate user-first: no orphaned tool/assistant tail.
        assert_eq!(h[0].role, "user");
    }

    #[test]
    fn trim_never_orphans_tool_messages() {
        let mut h = vec![
            msg("user", &"q ".repeat(200)),
            ChatMessage {
                role: "assistant".into(),
                tool_calls: vec![runa_engine::ToolCall {
                    id: "t1".into(),
                    name: "a".into(),
                    arguments: "{}".into(),
                }],
                ..ChatMessage::default()
            },
            ChatMessage {
                role: "tool".into(),
                content: "out".into(),
                tool_call_id: Some("t1".into()),
                ..ChatMessage::default()
            },
            msg("user", "new"),
        ];
        trim_history_with(&mut h, 50, &estimate_history_tokens);
        // The whole first turn (user+assistant+tool) drops together.
        assert_eq!(h, vec![msg("user", "new")]);
    }

    #[test]
    fn exact_counter_counts_known_string() {
        // P11.4: the exact path sums real per-message counts (+4 template
        // overhead each). The stub stands in for the loaded tokenizer at
        // one token per char on a known string.
        let h = vec![msg("user", "abcd"), msg("assistant", "ef")];
        let count_one = |s: &str| s.len() as u64;
        assert_eq!(count_history_tokens_with(&h, &count_one), (4 + 4) + (2 + 4));
    }

    #[test]
    fn trim_with_exact_counter_drops_oldest_turn_first() {
        // Same whole-turn / newest-survives contract as the estimate
        // path, re-checked against exact counts.
        let count_one = |s: &str| s.len() as u64;
        let count = |msgs: &[ChatMessage]| count_history_tokens_with(msgs, &count_one);
        let mut h = vec![
            msg("user", &"old question ".repeat(20)),
            msg("assistant", &"old answer ".repeat(20)),
            msg("user", "new"),
        ];
        let dropped = trim_history_with(&mut h, 50, &count);
        assert!(dropped >= 2, "oldest turn goes first");
        assert_eq!(h.last().unwrap().content, "new");
        assert!(count(&h) <= 50, "{h:?}");
        assert_eq!(h[0].role, "user");
    }

    #[test]
    fn fallback_estimate_locked_on_known_string() {
        // P11.4: paths without a tokenizer (daemon socket, mistral
        // backend) keep the chars/4 estimate — lock its value on a
        // known string so the fallback cannot drift silently.
        assert_eq!(estimate_history_tokens(&[msg("user", "abcd")]), 1 + 4);
        assert_eq!(
            estimate_history_tokens(&[msg("user", "abcd"), msg("assistant", "ef")]),
            (1 + 4) + 4
        );
    }

    #[test]
    fn commit_appends_answer_after_sent_rounds() {
        let mut session_history = vec![msg("user", "first"), msg("assistant", "one")];
        let sent = vec![
            msg("user", "first"),
            msg("assistant", "one"),
            msg("user", "second"),
        ];
        // Simulate commit: reuse the helper through a scratch session.
        let mut s = Session {
            path: PathBuf::from("m.gguf"),
            backend: BackendKind::Gguf,
            mode: runa_engine::Mode::Cpu,
            ctx: 8192,
            threads: None,
            history: std::mem::take(&mut session_history),
            think: ThinkConfig::default(),
            last_usage: None,
            hub: None,
            max_tool_rounds: 8,
            loras: Vec::new(),
            rpc_servers: Vec::new(),
        };
        commit_history(&mut s, &sent, "two");
        assert_eq!(
            s.history,
            vec![
                msg("user", "first"),
                msg("assistant", "one"),
                msg("user", "second"),
                msg("assistant", "two")
            ]
        );
        // Empty answers leave the sent transcript as-is.
        commit_history(&mut s, &sent, "  ");
        assert_eq!(s.history, sent);
    }
}
