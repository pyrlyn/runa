// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! Draining a generation to the terminal: answer to stdout, reasoning to stderr.

use std::io::{self, Write};

use runa_engine::{ChatMessage, GenEvent, StopReason, ToolCall, Usage};

use super::history;

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

/// Record the assistant turn that made tool calls; returns the calls.
/// With `print`, the streamed answer line is ended first.
pub(crate) fn push_tool_calls(
    messages: &mut Vec<ChatMessage>,
    turn: &LocalTurn,
    print: bool,
) -> Vec<ToolCall> {
    if print && !turn.calls.is_empty() && !turn.text.is_empty() {
        println!();
    }
    history::push_tool_calls(messages, &turn.text, &turn.calls);
    turn.calls.clone()
}

/// Drain daemon events like [`drain_local`]: answer text to stdout,
/// reasoning to stderr. A daemon-side `error` fails the request.
pub(crate) fn drain_daemon(
    events: &[crate::daemon_proto::DaemonEvent],
    print: bool,
) -> Result<LocalTurn, String> {
    use crate::daemon_proto::DaemonEvent;
    let mut turn = LocalTurn {
        text: String::new(),
        calls: Vec::new(),
        usage: None,
        reason: StopReason::MaxTokens,
    };
    for ev in events {
        match ev {
            DaemonEvent::Text { text } => {
                if print {
                    print!("{text}");
                    io::stdout().flush().map_err(|e| format!("stdout: {e}"))?;
                }
                turn.text.push_str(text);
            }
            DaemonEvent::Reasoning { text } => {
                if print {
                    eprint!("{text}");
                    io::stderr().flush().map_err(|e| format!("stderr: {e}"))?;
                }
            }
            DaemonEvent::ToolCalls { calls } => {
                turn.calls = calls.iter().cloned().map(ToolCall::from).collect();
            }
            DaemonEvent::Usage {
                prompt_tokens,
                generated_tokens,
                reasoning_tokens,
            } => {
                turn.usage = Some(Usage {
                    prompt_tokens: *prompt_tokens,
                    generated_tokens: *generated_tokens,
                    reasoning_tokens: *reasoning_tokens,
                    pp_toks_per_s: 0.0,
                    tg_toks_per_s: 0.0,
                });
            }
            DaemonEvent::Done { stop } => turn.reason = crate::daemon_proto::parse_stop(stop),
            DaemonEvent::Error { message } => return Err(message.clone()),
        }
    }
    Ok(turn)
}
