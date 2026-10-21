// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! Chat history: tool rounds, token counting, trimming and commit.
//! Prints nothing; callers report trims and errors.

use runa_engine::{ChatMessage, ToolCall};

use super::{ChatEngine, Session};

/// Answer the previous round's calls with `tool` messages (P8.3).
pub(crate) fn push_tool_results(messages: &mut Vec<ChatMessage>, results: &[(ToolCall, String)]) {
    messages.extend(results.iter().map(|(call, out)| ChatMessage {
        role: "tool".into(),
        content: out.clone(),
        tool_call_id: Some(call.id.clone()),
        ..ChatMessage::default()
    }));
}

/// Record the assistant message that made tool calls; a turn without
/// calls adds nothing.
pub(super) fn push_tool_calls(messages: &mut Vec<ChatMessage>, text: &str, calls: &[ToolCall]) {
    if calls.is_empty() {
        return;
    }
    messages.push(ChatMessage {
        role: "assistant".into(),
        content: text.to_owned(),
        tool_calls: calls.to_vec(),
        ..ChatMessage::default()
    });
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
pub(super) fn turn_messages(
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
pub(super) fn commit_history(session: &mut Session, sent: &[ChatMessage], answer: &str) {
    session.history = sent.to_vec();
    if !answer.trim().is_empty() {
        session.history.push(ChatMessage {
            role: "assistant".into(),
            content: answer.to_owned(),
            ..ChatMessage::default()
        });
    }
}

#[cfg(test)]
mod history_tests {
    use std::path::PathBuf;

    use runa_core::{BackendKind, ThinkConfig};

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
    fn tool_rounds_record_calls_then_results() {
        let call = ToolCall {
            id: "t1".into(),
            name: "a".into(),
            arguments: "{}".into(),
        };
        let mut h = vec![msg("user", "q")];
        // A turn without calls is the final answer: nothing to record here.
        push_tool_calls(&mut h, "plain", &[]);
        assert_eq!(h, vec![msg("user", "q")]);
        push_tool_calls(&mut h, "let me check", std::slice::from_ref(&call));
        push_tool_results(&mut h, &[(call.clone(), "out".into())]);
        assert_eq!(h.len(), 3);
        assert_eq!(h[1].role, "assistant");
        assert_eq!(h[1].content, "let me check");
        assert_eq!(h[1].tool_calls, vec![call]);
        assert_eq!(h[2].role, "tool");
        assert_eq!(h[2].content, "out");
        assert_eq!(h[2].tool_call_id.as_deref(), Some("t1"));
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
