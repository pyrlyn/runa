// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! MCP client and tool loop for `run` / `chat` (P8.3).
//!
//! Each server is a stdio child process (rmcp). Its tools are offered to the
//! model in OpenAI `tools` shape; a call goes to the server that listed it,
//! and the result goes back as a `tool` message until the model answers.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Duration;

use rmcp::ServiceExt;
use rmcp::model::{CallToolRequestParams, Tool};
use rmcp::service::{RoleClient, RunningService};
use rmcp::transport::TokioChildProcess;
use runa_core::ToolCall;
use serde_json::{Value, json};

/// Handshake, `tools/list` and each `tools/call` get this long.
const TIMEOUT: Duration = Duration::from_secs(120);

/// A stdio MCP server: `--mcp '<command args>'` or `[mcp.servers.<name>]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct McpServer {
    pub name: String,
    pub command: String,
    pub args: Vec<String>,
    pub env: BTreeMap<String, String>,
}

impl McpServer {
    /// `--mcp` value: command and arguments with shell-like quoting
    /// (P10.8): single/double quotes group spaces, backslash escapes the
    /// next character. Anything fancier belongs in `[mcp.servers]`.
    pub(crate) fn from_flag(spec: &str) -> Result<Self, String> {
        let words = split_shell_words(spec)?;
        let mut words = words.into_iter();
        let command = words.next().ok_or("--mcp needs a command")?;
        let name = Path::new(&command)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("mcp")
            .to_owned();
        Ok(McpServer {
            name,
            command,
            args: words.collect(),
            env: BTreeMap::new(),
        })
    }
}

/// Split like a shell (no expansions): whitespace separates, `'...'` is
/// literal, `"..."` allows backslash escapes, and a backslash outside
/// quotes escapes the next character (including whitespace).
fn split_shell_words(spec: &str) -> Result<Vec<String>, String> {
    let mut words = Vec::new();
    let mut cur = String::new();
    let mut in_word = false;
    let mut quote = None;
    let mut chars = spec.chars();
    while let Some(c) = chars.next() {
        match quote {
            Some('\'') => {
                if c == '\'' {
                    quote = None;
                } else {
                    cur.push(c);
                }
            }
            Some('"') => {
                if c == '"' {
                    quote = None;
                } else if c == '\\' {
                    match chars.next() {
                        Some(next) => cur.push(next),
                        None => {
                            return Err(format!("--mcp: dangling backslash in {spec:?}"));
                        }
                    }
                } else {
                    cur.push(c);
                }
            }
            Some(_) => unreachable!("quotes are only ' and \""),
            None => {
                if c == '\'' || c == '"' {
                    quote = Some(c);
                    in_word = true;
                } else if c == '\\' {
                    match chars.next() {
                        Some(next) => {
                            cur.push(next);
                            in_word = true;
                        }
                        None => {
                            return Err(format!("--mcp: dangling backslash in {spec:?}"));
                        }
                    }
                } else if c.is_whitespace() {
                    if in_word {
                        words.push(std::mem::take(&mut cur));
                        in_word = false;
                    }
                } else {
                    cur.push(c);
                    in_word = true;
                }
            }
        }
    }
    if quote.is_some() {
        return Err(format!("--mcp: unterminated quote in {spec:?}"));
    }
    if in_word {
        words.push(cur);
    }
    Ok(words)
}

/// Connected servers and the tools they offer.
pub(crate) struct McpHub {
    rt: tokio::runtime::Runtime,
    sessions: Vec<RunningService<RoleClient, ()>>,
    /// Each tool with the index of its session.
    tools: Vec<(Tool, usize)>,
}

impl McpHub {
    /// Start every server and list its tools. A tool name offered twice is
    /// an error: the model could not tell the two apart.
    pub(crate) fn start(servers: &[McpServer]) -> Result<Self, String> {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .map_err(|e| format!("mcp runtime: {e}"))?;
        let mut hub = McpHub {
            rt,
            sessions: Vec::new(),
            tools: Vec::new(),
        };
        for s in servers {
            let (session, listed) = hub
                .rt
                .block_on(async { tokio::time::timeout(TIMEOUT, connect(s)).await })
                .map_err(|_| format!("mcp {}: no handshake within {}s", s.name, TIMEOUT.as_secs()))?
                .map_err(|e| format!("mcp {}: {e}", s.name))?;
            hub.sessions.push(session);
            for t in listed {
                if hub.tools.iter().any(|(x, _)| x.name == t.name) {
                    return Err(format!("mcp {}: tool {} is offered twice", s.name, t.name));
                }
                hub.tools.push((t, hub.sessions.len() - 1));
            }
        }
        eprintln!(
            "mcp: {} tools from {} servers",
            hub.tools.len(),
            hub.sessions.len()
        );
        Ok(hub)
    }

    /// All tools as an OpenAI `tools` array.
    pub(crate) fn tools_json(&self) -> Value {
        self.tools
            .iter()
            .map(|(t, _)| {
                json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description.as_deref().unwrap_or_default(),
                        "parameters": Value::Object((*t.input_schema).clone()),
                    }
                })
            })
            .collect()
    }

    /// Run one call. Failures come back as text the model can read.
    pub(crate) fn call(&self, call: &ToolCall) -> String {
        let Some((_, idx)) = self.tools.iter().find(|(t, _)| t.name == call.name) else {
            return format!("error: unknown tool {}", call.name);
        };
        let args = match serde_json::from_str::<Value>(&call.arguments) {
            Ok(Value::Object(m)) => m,
            Ok(Value::Null) => serde_json::Map::new(),
            _ => return format!("error: arguments are not a JSON object: {}", call.arguments),
        };
        let params = CallToolRequestParams::new(call.name.clone()).with_arguments(args);
        let session = &self.sessions[*idx];
        let result = self
            .rt
            .block_on(async { tokio::time::timeout(TIMEOUT, session.call_tool(params)).await });
        match result {
            Err(_) => format!(
                "error: {} timed out after {}s",
                call.name,
                TIMEOUT.as_secs()
            ),
            Ok(Err(e)) => format!("error: {e}"),
            Ok(Ok(r)) => {
                let mut text = r
                    .content
                    .iter()
                    .filter_map(|c| c.as_text().map(|t| t.text.as_str()))
                    .collect::<Vec<_>>()
                    .join("\n");
                if text.is_empty()
                    && let Some(v) = r.structured_content
                {
                    text = v.to_string();
                }
                if r.is_error == Some(true) {
                    format!("error: {text}")
                } else {
                    text
                }
            }
        }
    }
}

impl Drop for McpHub {
    fn drop(&mut self) {
        for s in self.sessions.drain(..) {
            let _ = self.rt.block_on(s.cancel());
        }
    }
}

async fn connect(s: &McpServer) -> Result<(RunningService<RoleClient, ()>, Vec<Tool>), String> {
    let mut cmd = tokio::process::Command::new(&s.command);
    cmd.args(&s.args).envs(&s.env);
    let transport = TokioChildProcess::new(cmd).map_err(|e| format!("spawn {}: {e}", s.command))?;
    let session = ().serve(transport).await.map_err(|e| e.to_string())?;
    let tools = session.list_all_tools().await.map_err(|e| e.to_string())?;
    Ok((session, tools))
}

/// Start the servers named on the command line plus those in config.
pub(crate) fn start(flags: &[String]) -> Result<Option<McpHub>, String> {
    let mut servers = crate::config::load_mcp_servers()?;
    for spec in flags {
        servers.push(McpServer::from_flag(spec)?);
    }
    if servers.is_empty() {
        return Ok(None);
    }
    McpHub::start(&servers).map(Some)
}

/// Drive `step` until it answers without tool calls. `step` receives the
/// previous round's calls with their outputs (empty on the first round),
/// appends them to its own history, generates, and returns the new calls.
pub(crate) fn tool_loop(
    max_rounds: u32,
    mut call: impl FnMut(&ToolCall) -> String,
    mut step: impl FnMut(&[(ToolCall, String)]) -> Result<Vec<ToolCall>, String>,
) -> Result<(), String> {
    let mut results = Vec::new();
    for round in 0..=max_rounds {
        let calls = step(&results)?;
        if calls.is_empty() {
            return Ok(());
        }
        // The budget is executions, not model steps. Stop before running
        // the calls that would be the round past `max_rounds`.
        if round == max_rounds {
            break;
        }
        results = calls
            .into_iter()
            .map(|c| {
                let out = call(&c);
                eprintln!("[tool] {}({}) -> {} bytes", c.name, c.arguments, out.len());
                (c, out)
            })
            .collect();
    }
    Err(format!(
        "no answer after {max_rounds} tool rounds (raise --max-tool-rounds)"
    ))
}

/// `call` for [`tool_loop`] when no hub is running.
pub(crate) fn call_with(hub: Option<&McpHub>) -> impl Fn(&ToolCall) -> String + '_ {
    move |c| match hub {
        Some(h) => h.call(c),
        None => format!("error: no MCP server offers {}", c.name),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(name: &str) -> ToolCall {
        ToolCall {
            id: format!("id-{name}"),
            name: name.into(),
            arguments: "{}".into(),
        }
    }

    #[test]
    fn flag_splits_command_and_args() {
        let s = McpServer::from_flag("  /usr/bin/python3 server.py --x 1 ").unwrap();
        assert_eq!(s.name, "python3");
        assert_eq!(s.command, "/usr/bin/python3");
        assert_eq!(s.args, ["server.py", "--x", "1"]);
        assert!(McpServer::from_flag("   ").is_err());
    }

    #[test]
    fn flag_quoting_groups_spaces() {
        // P10.8: single/double quotes and backslash escapes.
        let s = McpServer::from_flag(
            r#"python3 "my server.py" --root '/tmp/my dir' --q 'a"b' --e a\ b"#,
        )
        .unwrap();
        assert_eq!(
            s.args,
            [
                "my server.py",
                "--root",
                "/tmp/my dir",
                "--q",
                "a\"b",
                "--e",
                "a b"
            ]
        );
        // Empty quoted strings are empty args, like a shell.
        let s = McpServer::from_flag("cmd ''").unwrap();
        assert_eq!(s.args, [""]);
        // Mixed quoting inside one word.
        let s = McpServer::from_flag("cmd a'b c'd\"e f\"").unwrap();
        assert_eq!(s.args, ["ab cde f"]);
    }

    #[test]
    fn flag_unterminated_quote_is_an_error() {
        for bad in ["cmd 'oops", "cmd \"oops", "cmd \\"] {
            let err = McpServer::from_flag(bad).unwrap_err();
            assert!(
                err.contains("unterminated quote") || err.contains("dangling backslash"),
                "{bad}: {err}"
            );
        }
    }

    #[test]
    fn loop_feeds_results_back_until_an_answer() {
        let mut seen = Vec::new();
        let mut round = 0;
        tool_loop(
            4,
            |c| format!("out-{}", c.name),
            |results| {
                seen.push(results.to_vec());
                round += 1;
                Ok(if round < 3 { vec![call("a")] } else { vec![] })
            },
        )
        .unwrap();
        assert_eq!(seen.len(), 3);
        assert!(seen[0].is_empty());
        assert_eq!(seen[1], [(call("a"), "out-a".to_owned())]);
    }

    #[test]
    fn loop_stops_at_max_rounds() {
        let err = tool_loop(2, |_| String::new(), |_| Ok(vec![call("a")])).unwrap_err();
        assert!(err.contains("--max-tool-rounds"), "{err}");
    }

    #[test]
    fn loop_does_not_run_tools_past_the_budget() {
        let mut ran = 0u32;
        let err = tool_loop(
            2,
            |_| {
                ran += 1;
                String::new()
            },
            |_| Ok(vec![call("a")]),
        )
        .unwrap_err();
        assert!(err.contains("--max-tool-rounds"), "{err}");
        assert_eq!(ran, 2);
        ran = 0;
        let err = tool_loop(
            0,
            |_| {
                ran += 1;
                String::new()
            },
            |_| Ok(vec![call("a")]),
        )
        .unwrap_err();
        assert_eq!(ran, 0, "{err}");
    }
}
