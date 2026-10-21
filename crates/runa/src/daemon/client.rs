// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! Daemon client for `run` and `chat`: probe, one request, drain.

use std::io::{self, Write};

use runa_engine::{GenerateRequest, StopReason, ToolCall, Usage};

use crate::daemon;
use crate::daemon_proto;

/// Send one request to the daemon; transport errors mean "no daemon".
pub(crate) fn daemon_generate(
    model: &str,
    req: &GenerateRequest,
) -> Result<Vec<daemon_proto::DaemonEvent>, String> {
    let dreq = daemon_proto::DaemonRequest::new(model.to_owned(), req);
    daemon_proto::request_sync(
        &daemon_proto::default_socket_path(),
        &dreq,
        daemon::REQUEST_TIMEOUT,
    )
}

/// Drain daemon events like [`drain_local`]: answer text to stdout,
/// reasoning to stderr. A daemon-side `error` fails the request.
pub(crate) fn drain_daemon(
    events: &[daemon_proto::DaemonEvent],
    print: bool,
) -> Result<crate::session::LocalTurn, String> {
    use daemon_proto::DaemonEvent;
    let mut turn = crate::session::LocalTurn {
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
            DaemonEvent::Done { stop } => turn.reason = daemon_proto::parse_stop(stop),
            DaemonEvent::Error { message } => return Err(message.clone()),
        }
    }
    Ok(turn)
}

/// One cheap probe: is a daemon listening? The probe connection is
/// dropped immediately; real turns dial per request.
pub(crate) fn daemon_available() -> bool {
    #[cfg(unix)]
    {
        std::os::unix::net::UnixStream::connect(daemon_proto::default_socket_path()).is_ok()
    }
    #[cfg(not(unix))]
    {
        false
    }
}
