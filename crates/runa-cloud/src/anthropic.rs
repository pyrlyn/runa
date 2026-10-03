// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! Anthropic Messages API: `reqwest` + SSE (P3.6).

use std::time::Duration;

use runa_core::{Effort, ThinkConfig, ThinkMode, ToolCall};
use serde_json::{Value, json};

/// One streamed or completed piece.
#[derive(Debug, Clone, PartialEq)]
pub enum AnthropicEvent {
    Reasoning(String),
    Text(String),
    /// `tool_use` blocks of a non-streamed reply (P8.3). `content` is the
    /// reply's whole block array: sending it back as the assistant turn
    /// keeps the thinking signatures the API requires next to tool use.
    ToolUse {
        calls: Vec<ToolCall>,
        content: Value,
    },
    Usage {
        input_tokens: u32,
        output_tokens: u32,
    },
    Done {
        stop_reason: String,
    },
    Refusal(String),
}

/// Image source already encoded as base64.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageBlock {
    pub media_type: String,
    pub data: String,
}

/// PDF source already encoded as base64.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PdfBlock {
    pub data: String,
}

/// One user/assistant turn (`system` is separate for `cache_control`).
#[derive(Debug, Clone, PartialEq)]
pub struct ChatTurn {
    pub role: String,
    pub text: String,
    /// Raw content blocks sent instead of `text` (tool turns, P8.3).
    pub blocks: Option<Value>,
}

impl ChatTurn {
    pub fn text(role: &str, text: impl Into<String>) -> Self {
        ChatTurn {
            role: role.into(),
            text: text.into(),
            blocks: None,
        }
    }

    /// A user turn answering tool calls: one `tool_result` block per
    /// `(tool_use_id, output)`.
    pub fn tool_results(results: &[(String, String)]) -> Self {
        let blocks = results
            .iter()
            .map(|(id, out)| json!({"type": "tool_result", "tool_use_id": id, "content": out}))
            .collect();
        ChatTurn {
            role: "user".into(),
            text: String::new(),
            blocks: Some(Value::Array(blocks)),
        }
    }
}

/// OpenAI-shape `tools` (`{type: function, function: {name, description,
/// parameters}}`) → Anthropic `{name, description, input_schema}`.
pub fn tools_from_openai(tools: &Value) -> Vec<Value> {
    tools
        .as_array()
        .into_iter()
        .flatten()
        .map(|t| {
            let f = &t["function"];
            json!({
                "name": f["name"],
                "description": f["description"].as_str().unwrap_or_default(),
                "input_schema": if f["parameters"].is_object() {
                    f["parameters"].clone()
                } else {
                    json!({"type": "object"})
                },
            })
        })
        .collect()
}

/// Outgoing Messages request.
#[derive(Debug, Clone)]
pub struct AnthropicRequest {
    pub model: String,
    pub system: Option<String>,
    pub messages: Vec<ChatTurn>,
    pub think: ThinkConfig,
    pub max_tokens: u32,
    pub images: Vec<ImageBlock>,
    pub pdfs: Vec<PdfBlock>,
    pub stream: bool,
    /// Anthropic-shape tools (P8.3); see [`tools_from_openai`].
    pub tools: Vec<Value>,
    /// `{"type": "auto" | "any" | "tool", ...}`; `None` = API default.
    pub tool_choice: Option<Value>,
}

/// HTTP client for `https://api.anthropic.com/v1/messages`.
pub struct AnthropicClient {
    api_key: String,
    base_url: String,
    http: reqwest::Client,
    retry_base: Duration,
    max_retries: u32,
}

impl AnthropicClient {
    pub fn new(api_key: impl Into<String>) -> Self {
        AnthropicClient {
            api_key: api_key.into(),
            base_url: "https://api.anthropic.com".into(),
            http: reqwest::Client::new(),
            retry_base: Duration::from_millis(200),
            max_retries: 3,
        }
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into().trim_end_matches('/').to_string();
        self
    }

    #[cfg(test)]
    fn with_retry_base(mut self, d: Duration) -> Self {
        self.retry_base = d;
        self
    }

    /// POST `/v1/messages`. Streaming bodies are parsed as SSE.
    pub async fn generate(&self, req: &AnthropicRequest) -> Result<Vec<AnthropicEvent>, String> {
        let body = build_body(req);
        let url = format!("{}/v1/messages", self.base_url);
        let mut attempt = 0;
        loop {
            let resp = self
                .http
                .post(&url)
                .header("x-api-key", &self.api_key)
                .header("anthropic-version", "2023-06-01")
                .header("content-type", "application/json")
                .json(&body)
                .send()
                .await
                .map_err(|e| e.to_string())?;
            let status = resp.status().as_u16();
            if (status == 429 || status == 529) && attempt < self.max_retries {
                let wait = self.retry_base * 2u32.pow(attempt);
                tokio::time::sleep(wait).await;
                attempt += 1;
                continue;
            }
            if !resp.status().is_success() {
                let t = resp.text().await.unwrap_or_default();
                return Err(format!("anthropic HTTP {status}: {t}"));
            }
            let text = resp.text().await.map_err(|e| e.to_string())?;
            return if req.stream {
                parse_sse(&text)
            } else {
                parse_message(&text)
            };
        }
    }
}

/// Claude 4.6+ / 5-family use adaptive thinking; older models take a budget.
pub fn adaptive_model(model: &str) -> bool {
    let m = model.to_ascii_lowercase();
    m.contains("claude-5")
        || m.contains("opus-5")
        || m.contains("sonnet-5")
        || m.contains("haiku-5")
        || m.contains("4-6")
        || m.contains("4.6")
        || m.contains("4-7")
        || m.contains("4.7")
}

pub fn thinking_body(model: &str, think: &ThinkConfig) -> Option<Value> {
    match think.mode {
        ThinkMode::Off => None,
        ThinkMode::On | ThinkMode::Effort(_) | ThinkMode::Budget { .. } => {
            if adaptive_model(model) {
                Some(json!({"type": "adaptive"}))
            } else {
                let tokens = think.reasoning_budget(8192).unwrap_or(1024);
                Some(json!({"type": "enabled", "budget_tokens": tokens}))
            }
        }
    }
}

fn effort_name(think: &ThinkConfig) -> Option<&'static str> {
    match think.mode {
        ThinkMode::Effort(e) => Some(match e {
            Effort::Low => "low",
            Effort::Medium => "medium",
            Effort::High | Effort::Max => "high",
        }),
        _ => None,
    }
}

pub fn build_body(req: &AnthropicRequest) -> Value {
    let mut user_content: Vec<Value> = Vec::new();
    for img in &req.images {
        user_content.push(json!({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": img.media_type,
                "data": img.data,
            }
        }));
    }
    for pdf in &req.pdfs {
        user_content.push(json!({
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": pdf.data,
            }
        }));
    }

    let mut messages: Vec<Value> = Vec::new();
    for (i, turn) in req.messages.iter().enumerate() {
        if turn.role == "user" && i + 1 == req.messages.len() && !user_content.is_empty() {
            let mut parts = user_content.clone();
            if !turn.text.is_empty() {
                parts.push(json!({"type": "text", "text": turn.text}));
            }
            messages.push(json!({"role": "user", "content": parts}));
        } else {
            messages.push(json!({
                "role": turn.role,
                "content": turn.blocks.clone().unwrap_or_else(|| json!(turn.text)),
            }));
        }
    }

    let mut body = json!({
        "model": req.model,
        "max_tokens": req.max_tokens,
        "messages": messages,
        "stream": req.stream,
    });
    if let Some(sys) = &req.system {
        body["system"] = json!([{
            "type": "text",
            "text": sys,
            "cache_control": {"type": "ephemeral"}
        }]);
    }
    if let Some(t) = thinking_body(&req.model, &req.think) {
        body["thinking"] = t;
    }
    if let Some(effort) = effort_name(&req.think) {
        body["output_config"] = json!({"effort": effort});
    }
    if req.think.show {
        body["display"] = json!("thinking");
    }
    if !req.tools.is_empty() {
        body["tools"] = json!(req.tools);
    }
    if let Some(choice) = &req.tool_choice {
        body["tool_choice"] = choice.clone();
    }
    body
}

pub fn parse_message(text: &str) -> Result<Vec<AnthropicEvent>, String> {
    let v: Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    if let Some(blocks) = v.get("content").and_then(|c| c.as_array()) {
        let mut calls = Vec::new();
        for b in blocks {
            match b.get("type").and_then(|t| t.as_str()) {
                Some("thinking") => {
                    if let Some(s) = b.get("thinking").and_then(|t| t.as_str())
                        && !s.is_empty()
                    {
                        out.push(AnthropicEvent::Reasoning(s.to_string()));
                    }
                }
                Some("text") => {
                    if let Some(s) = b.get("text").and_then(|t| t.as_str())
                        && !s.is_empty()
                    {
                        out.push(AnthropicEvent::Text(s.to_string()));
                    }
                }
                Some("tool_use") => calls.push(ToolCall {
                    id: b["id"].as_str().unwrap_or_default().to_owned(),
                    name: b["name"].as_str().unwrap_or_default().to_owned(),
                    arguments: b.get("input").map_or("{}".into(), Value::to_string),
                }),
                _ => {}
            }
        }
        if !calls.is_empty() {
            out.push(AnthropicEvent::ToolUse {
                calls,
                content: Value::Array(blocks.clone()),
            });
        }
    }
    if let Some(u) = v.get("usage") {
        out.push(AnthropicEvent::Usage {
            input_tokens: u32_field(u, "input_tokens"),
            output_tokens: u32_field(u, "output_tokens"),
        });
    }
    let stop = v
        .get("stop_reason")
        .and_then(|s| s.as_str())
        .unwrap_or("end_turn");
    if stop == "refusal" {
        let msg = out
            .iter()
            .find_map(|e| match e {
                AnthropicEvent::Text(s) => Some(s.clone()),
                _ => None,
            })
            .unwrap_or_default();
        out.push(AnthropicEvent::Refusal(msg));
    }
    out.push(AnthropicEvent::Done {
        stop_reason: stop.to_string(),
    });
    Ok(out)
}

pub fn parse_sse(text: &str) -> Result<Vec<AnthropicEvent>, String> {
    let mut out = Vec::new();
    let mut data = String::new();
    // P10.6: tool_use blocks stream as `content_block_start` (id/name)
    // plus `input_json_delta` fragments — accumulate them here so a
    // streamed tool call is not silently dropped (the CLI still prefers
    // the non-stream call: only it preserves thinking signatures for
    // the next round).
    let mut acc = SseTools::default();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("data:") {
            if !data.is_empty() {
                data.push('\n');
            }
            data.push_str(rest.trim_start());
            continue;
        }
        if line.is_empty() && !data.is_empty() {
            push_sse_event(&data, &mut out, &mut acc)?;
            data.clear();
        }
    }
    if !data.is_empty() {
        push_sse_event(&data, &mut out, &mut acc)?;
    }
    // A transcript cut before `message_delta` still yields its calls.
    if let Some(ev) = acc.flush() {
        out.push(ev);
    }
    Ok(out)
}

/// Accumulates streamed content blocks (`content_block_start` /
/// `*_delta` / `content_block_stop`) so `parse_sse` can rebuild the
/// `tool_use` calls — and the surrounding blocks — that the non-stream
/// `parse_message` sees whole (P10.6).
#[derive(Debug, Default)]
struct SseTools {
    blocks: std::collections::BTreeMap<u64, SseBlock>,
    flushed: bool,
}

#[derive(Debug, Default)]
struct SseBlock {
    kind: String,
    id: String,
    name: String,
    text: String,
    signature: String,
    json: String,
}

impl SseTools {
    fn start(&mut self, index: u64, block: &Value) {
        let kind = block
            .get("type")
            .and_then(|t| t.as_str())
            .unwrap_or_default();
        let mut b = SseBlock {
            kind: kind.to_owned(),
            ..SseBlock::default()
        };
        if kind == "tool_use" {
            b.id = block
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .into();
            b.name = block
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .into();
            // Some servers inline the whole input up front.
            b.json = block.get("input").map(Value::to_string).unwrap_or_default();
            if b.json == "null" {
                b.json.clear();
            }
        }
        self.blocks.insert(index, b);
    }

    fn delta(&mut self, index: u64, delta: &Value) {
        let Some(b) = self.blocks.get_mut(&index) else {
            return;
        };
        match delta.get("type").and_then(|t| t.as_str()) {
            Some("input_json_delta") => {
                if let Some(s) = delta.get("partial_json").and_then(|v| v.as_str()) {
                    b.json.push_str(s);
                }
            }
            Some("thinking_delta") => {
                if let Some(s) = delta.get("thinking").and_then(|v| v.as_str()) {
                    b.text.push_str(s);
                }
            }
            Some("signature_delta") => {
                if let Some(s) = delta.get("signature").and_then(|v| v.as_str()) {
                    b.signature.push_str(s);
                }
            }
            Some("text_delta") => {
                if let Some(s) = delta.get("text").and_then(|v| v.as_str()) {
                    b.text.push_str(s);
                }
            }
            _ => {}
        }
    }

    /// Rebuild the content-block array (index order) plus the tool calls.
    /// `None` when no block was ever started.
    fn flush(&mut self) -> Option<AnthropicEvent> {
        if self.flushed || self.blocks.is_empty() {
            return None;
        }
        self.flushed = true;
        let mut content = Vec::new();
        let mut calls = Vec::new();
        for b in self.blocks.values() {
            match b.kind.as_str() {
                "tool_use" => {
                    let input: Value = serde_json::from_str(&b.json).unwrap_or_else(|_| json!({}));
                    calls.push(ToolCall {
                        id: b.id.clone(),
                        name: b.name.clone(),
                        arguments: input.to_string(),
                    });
                    content.push(json!({
                        "type": "tool_use",
                        "id": b.id,
                        "name": b.name,
                        "input": input,
                    }));
                }
                "thinking" => {
                    let mut block = json!({"type": "thinking", "thinking": b.text});
                    if !b.signature.is_empty() {
                        block["signature"] = json!(b.signature);
                    }
                    content.push(block);
                }
                "text" if !b.text.is_empty() => {
                    content.push(json!({"type": "text", "text": b.text}));
                }
                _ => {}
            }
        }
        if calls.is_empty() {
            return None;
        }
        Some(AnthropicEvent::ToolUse {
            calls,
            content: Value::Array(content),
        })
    }
}

fn push_sse_event(
    data: &str,
    out: &mut Vec<AnthropicEvent>,
    acc: &mut SseTools,
) -> Result<(), String> {
    if data.trim().is_empty() || data.trim() == "[DONE]" {
        return Ok(());
    }
    let v: Value = serde_json::from_str(data).map_err(|e| e.to_string())?;
    let index = v.get("index").and_then(|i| i.as_u64());
    match v.get("type").and_then(|t| t.as_str()) {
        Some("content_block_start") => {
            if let (Some(i), Some(block)) = (index, v.get("content_block")) {
                acc.start(i, block);
            }
        }
        Some("content_block_delta") => {
            if let (Some(i), Some(delta)) = (index, v.get("delta")) {
                acc.delta(i, delta);
                // P10.6: tool fragments must not leak into text/reasoning.
                if delta
                    .get("type")
                    .and_then(|t| t.as_str())
                    .is_some_and(|t| t == "input_json_delta" || t == "signature_delta")
                {
                    return Ok(());
                }
            }
            if let Some(delta) = v.get("delta") {
                match delta.get("type").and_then(|t| t.as_str()) {
                    Some("thinking_delta") => {
                        if let Some(s) = delta.get("thinking").and_then(|t| t.as_str())
                            && !s.is_empty()
                        {
                            out.push(AnthropicEvent::Reasoning(s.to_string()));
                        }
                    }
                    Some("text_delta") => {
                        if let Some(s) = delta.get("text").and_then(|t| t.as_str())
                            && !s.is_empty()
                        {
                            out.push(AnthropicEvent::Text(s.to_string()));
                        }
                    }
                    _ => {}
                }
            }
        }
        Some("message_delta") => {
            if let Some(u) = v.get("usage") {
                out.push(AnthropicEvent::Usage {
                    input_tokens: u32_field(u, "input_tokens"),
                    output_tokens: u32_field(u, "output_tokens"),
                });
            }
            if let Some(stop) = v
                .get("delta")
                .and_then(|d| d.get("stop_reason"))
                .and_then(|s| s.as_str())
            {
                if stop == "refusal" {
                    out.push(AnthropicEvent::Refusal(String::new()));
                }
                // P10.6: streamed tool calls surface here, ahead of Done —
                // the same position `parse_message` uses, so `drain_anthropic`
                // needs no stream/non-stream split.
                if let Some(ev) = acc.flush() {
                    out.push(ev);
                }
                out.push(AnthropicEvent::Done {
                    stop_reason: stop.to_string(),
                });
            }
        }
        Some("message_stop") if !out.iter().any(|e| matches!(e, AnthropicEvent::Done { .. })) => {
            out.push(AnthropicEvent::Done {
                stop_reason: "end_turn".into(),
            });
        }
        _ => {}
    }
    Ok(())
}

fn u32_field(v: &Value, key: &str) -> u32 {
    v.get(key).and_then(|n| n.as_u64()).unwrap_or(0) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn fixture(name: &str) -> String {
        let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/api")
            .join(name);
        std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
    }

    fn req_text(model: &str) -> AnthropicRequest {
        AnthropicRequest {
            model: model.into(),
            system: Some("sys".into()),
            messages: vec![ChatTurn::text("user", "hi")],
            think: ThinkConfig {
                mode: ThinkMode::Effort(Effort::High),
                show: true,
            },
            max_tokens: 64,
            images: vec![ImageBlock {
                media_type: "image/png".into(),
                data: "aaa".into(),
            }],
            pdfs: vec![PdfBlock { data: "bbb".into() }],
            stream: false,
            tools: Vec::new(),
            tool_choice: None,
        }
    }

    #[test]
    fn tool_use_round_trip() {
        let tools = tools_from_openai(&json!([{"type": "function", "function": {
            "name": "get_weather", "parameters": {"type": "object"}}}]));
        assert_eq!(tools[0]["name"], "get_weather");
        assert_eq!(tools[0]["input_schema"]["type"], "object");
        let reply = r#"{"content":[
            {"type":"thinking","thinking":"t","signature":"s"},
            {"type":"tool_use","id":"tu1","name":"get_weather","input":{"city":"Paris"}}
        ],"stop_reason":"tool_use","usage":{"input_tokens":3,"output_tokens":4}}"#;
        let ev = parse_message(reply).unwrap();
        let (calls, content) = ev
            .iter()
            .find_map(|e| match e {
                AnthropicEvent::ToolUse { calls, content } => Some((calls, content)),
                _ => None,
            })
            .unwrap();
        assert_eq!(calls[0].id, "tu1");
        assert_eq!(calls[0].arguments, r#"{"city":"Paris"}"#);
        assert_eq!(content[0]["signature"], "s");
        let mut req = req_text("claude-sonnet-5");
        req.images.clear();
        req.pdfs.clear();
        req.tools = tools;
        req.messages.push(ChatTurn {
            role: "assistant".into(),
            text: String::new(),
            blocks: Some(content.clone()),
        });
        req.messages
            .push(ChatTurn::tool_results(&[("tu1".into(), "sunny".into())]));
        let body = build_body(&req);
        assert_eq!(body["tools"][0]["name"], "get_weather");
        assert_eq!(body["messages"][1]["content"][1]["type"], "tool_use");
        assert_eq!(body["messages"][2]["content"][0]["tool_use_id"], "tu1");
    }

    #[test]
    fn fixture_message_parses() {
        let ev = parse_message(&fixture("anthropic-message.json")).unwrap();
        assert!(
            ev.iter().any(
                |e| matches!(e, AnthropicEvent::Text(t) if t.contains("Hello from Anthropic"))
            )
        );
        assert!(ev.iter().any(|e| matches!(
            e,
            AnthropicEvent::Usage {
                input_tokens: 5,
                output_tokens: 6
            }
        )));
        assert!(ev.iter().any(
            |e| matches!(e, AnthropicEvent::Done { stop_reason } if stop_reason == "end_turn")
        ));
    }

    #[test]
    fn fixture_sse_parses_text_delta() {
        let ev = parse_sse(&fixture("anthropic-message-stream.sse")).unwrap();
        assert!(
            ev.iter()
                .any(|e| matches!(e, AnthropicEvent::Text(t) if t == "Hello"))
        );
        assert!(ev.iter().any(|e| matches!(e, AnthropicEvent::Done { .. })));
    }

    #[test]
    fn thinking_delta_and_refusal() {
        let sse = "\
event: content_block_delta
data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"hmm\"}}

event: content_block_delta
data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"no\"}}

event: message_delta
data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"refusal\"},\"usage\":{\"output_tokens\":1}}

";
        let ev = parse_sse(sse).unwrap();
        assert!(
            ev.iter()
                .any(|e| matches!(e, AnthropicEvent::Reasoning(s) if s == "hmm"))
        );
        assert!(ev.iter().any(|e| matches!(e, AnthropicEvent::Refusal(_))));
        assert!(ev.iter().any(
            |e| matches!(e, AnthropicEvent::Done { stop_reason } if stop_reason == "refusal")
        ));
    }

    #[test]
    fn sse_tool_use_reassembled_across_fragments() {
        // P10.6: streamed `tool_use` (start + input_json_delta fragments)
        // must surface as one ToolUse ahead of Done — the shape the CLI
        // tool loop already drains from non-stream replies.
        let sse = "\
event: content_block_start
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}

event: content_block_delta
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Checking \"}}

event: content_block_start
data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"tu1\",\"name\":\"get_weather\"}}

event: content_block_delta
data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"city\\\":\\\"Par\"}}

event: content_block_delta
data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"is\\\"}\"}}

event: content_block_start
data: {\"type\":\"content_block_start\",\"index\":2,\"content_block\":{\"type\":\"tool_use\",\"id\":\"tu2\",\"name\":\"get_time\"}}

event: content_block_delta
data: {\"type\":\"content_block_delta\",\"index\":2,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{}\"}}

event: message_delta
data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":9}}

event: message_stop
data: {\"type\":\"message_stop\"}

";
        let ev = parse_sse(sse).unwrap();
        // Text still streams; tool fragments never leak into it.
        assert!(
            ev.iter()
                .any(|e| matches!(e, AnthropicEvent::Text(t) if t == "Checking "))
        );
        assert!(
            !ev.iter()
                .any(|e| matches!(e, AnthropicEvent::Text(t) if t.contains("city"))),
            "{ev:?}"
        );
        let (calls, content) = ev
            .iter()
            .find_map(|e| match e {
                AnthropicEvent::ToolUse { calls, content } => Some((calls, content)),
                _ => None,
            })
            .expect("ToolUse event");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].id, "tu1");
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments, r#"{"city":"Paris"}"#);
        assert_eq!(calls[1].name, "get_time");
        assert_eq!(calls[1].arguments, "{}");
        // Rebuilt blocks keep index order (text, tu1, tu2).
        let kinds: Vec<&str> = content
            .as_array()
            .unwrap()
            .iter()
            .map(|b| b["type"].as_str().unwrap())
            .collect();
        assert_eq!(kinds, ["text", "tool_use", "tool_use"]);
        // ToolUse lands ahead of Done, like parse_message.
        let tool_pos = ev
            .iter()
            .position(|e| matches!(e, AnthropicEvent::ToolUse { .. }))
            .unwrap();
        let done_pos = ev
            .iter()
            .position(|e| {
                matches!(
                    e, AnthropicEvent::Done { stop_reason } if stop_reason == "tool_use"
                )
            })
            .unwrap();
        assert!(tool_pos < done_pos);
    }

    #[test]
    fn sse_without_tools_emits_no_tool_use() {
        // The flush-at-end path must stay quiet on plain text streams.
        let ev = parse_sse(&fixture("anthropic-message-stream.sse")).unwrap();
        assert!(
            !ev.iter()
                .any(|e| matches!(e, AnthropicEvent::ToolUse { .. })),
            "{ev:?}"
        );
    }

    #[test]
    fn thinking_adaptive_vs_enabled() {
        let on = ThinkConfig {
            mode: ThinkMode::On,
            show: false,
        };
        let a = thinking_body("claude-sonnet-5", &on).unwrap();
        assert_eq!(a["type"], "adaptive");
        let b = thinking_body("claude-3-5-sonnet-20241022", &on).unwrap();
        assert_eq!(b["type"], "enabled");
        assert!(b["budget_tokens"].as_u64().unwrap() >= 1024);
        let budgeted = ThinkConfig {
            mode: ThinkMode::Budget {
                tokens: 256,
                grace: 16,
            },
            show: false,
        };
        let c = thinking_body("claude-3-opus", &budgeted).unwrap();
        assert_eq!(c["budget_tokens"], 256);
        assert!(thinking_body("claude-3-opus", &ThinkConfig::default()).is_none());
    }

    #[test]
    fn body_has_cache_control_media_effort() {
        let body = build_body(&req_text("claude-opus-5"));
        assert_eq!(body["thinking"]["type"], "adaptive");
        assert_eq!(body["output_config"]["effort"], "high");
        assert_eq!(body["display"], "thinking");
        assert_eq!(body["system"][0]["cache_control"]["type"], "ephemeral");
        let content = body["messages"][0]["content"].as_array().unwrap();
        assert!(
            content
                .iter()
                .any(|c| c["type"] == "image" && c["source"]["data"] == "aaa")
        );
        assert!(
            content
                .iter()
                .any(|c| c["type"] == "document" && c["source"]["media_type"] == "application/pdf")
        );
    }

    #[tokio::test]
    async fn wiremock_fixture_message() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string(fixture("anthropic-message.json"))
                    .insert_header("content-type", "application/json"),
            )
            .mount(&server)
            .await;
        let client = AnthropicClient::new("sk-ant-test").with_base_url(server.uri());
        let ev = client
            .generate(&req_text("claude-3-5-sonnet"))
            .await
            .unwrap();
        assert!(
            ev.iter()
                .any(|e| matches!(e, AnthropicEvent::Text(t) if t.contains("Anthropic fixture")))
        );
    }

    #[tokio::test]
    async fn wiremock_fixture_stream() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string(fixture("anthropic-message-stream.sse"))
                    .insert_header("content-type", "text/event-stream"),
            )
            .mount(&server)
            .await;
        let mut req = req_text("claude-3-5-sonnet");
        req.stream = true;
        let client = AnthropicClient::new("sk-ant-test").with_base_url(server.uri());
        let ev = client.generate(&req).await.unwrap();
        assert!(
            ev.iter()
                .any(|e| matches!(e, AnthropicEvent::Text(t) if t == "Hello"))
        );
    }

    #[tokio::test]
    async fn retries_429_then_ok() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(429).set_body_string("slow"))
            .up_to_n_times(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(
                ResponseTemplate::new(200).set_body_string(fixture("anthropic-message.json")),
            )
            .mount(&server)
            .await;
        let client = AnthropicClient::new("sk-ant-test")
            .with_base_url(server.uri())
            .with_retry_base(Duration::from_millis(1));
        let ev = client
            .generate(&req_text("claude-3-5-sonnet"))
            .await
            .unwrap();
        assert!(ev.iter().any(|e| matches!(e, AnthropicEvent::Text(_))));
    }

    #[tokio::test]
    async fn live_smoke() {
        if std::env::var("RUNA_LIVE").ok().as_deref() != Some("1") {
            return;
        }
        let key = crate::resolve_api_key(crate::Provider::Anthropic)
            .expect("ANTHROPIC_API_KEY for RUNA_LIVE=1");
        let client = AnthropicClient::new(key.value);
        let req = AnthropicRequest {
            model: "claude-3-5-haiku-latest".into(),
            system: None,
            messages: vec![ChatTurn::text("user", "Reply with the single word pong.")],
            think: ThinkConfig::default(),
            max_tokens: 16,
            images: Vec::new(),
            pdfs: Vec::new(),
            stream: false,
            tools: Vec::new(),
            tool_choice: None,
        };
        let ev = client.generate(&req).await.expect("live anthropic");
        assert!(ev.iter().any(|e| matches!(e, AnthropicEvent::Text(_))));
    }
}
