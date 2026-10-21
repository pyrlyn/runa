// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! Full-screen chat loop (`runa chat --tui`, P8.6).

use std::io;
use std::path::PathBuf;

use runa_core::ThinkOverrides;
use runa_engine::{ChatMessage, GenEvent, GenerateRequest, Mode, SamplingConfig, Usage};

use super::format::format_usage;
use super::history::{commit_history, push_tool_results, turn_messages};
use super::{ChatEngine, Session, reload, reload_local, store_chat_usage};
use crate::daemon::client::daemon_generate;
use crate::placement::{parse_mode, resolve_model};
use crate::{daemon_proto, mcp, tui};

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

pub(super) fn run_tui(
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
            let usage = store_chat_usage(session, usage);
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
                        store_chat_usage(session, usage);
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
            let usage = store_chat_usage(session, usage);
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
