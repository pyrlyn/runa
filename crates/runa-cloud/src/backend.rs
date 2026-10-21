// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! [`Backend`] over the OpenAI and Anthropic clients (P21.1).
//!
//! Both adapters send one non-streamed request per `generate` and replay
//! the reply as [`Event`]s: only a whole Anthropic message carries the
//! thinking signatures a later tool round must send back, and OpenAI keeps
//! the same shape. Each backend owns a current-thread runtime, so call
//! `generate` from a plain thread or `spawn_blocking`, never from inside an
//! async task.

use runa_core::{
    Backend, BackendError, ChatMessage, Event, EventStream, Request, StopReason, ThinkMode, Usage,
};
use serde_json::{Value, json};
use tokio::runtime::Runtime;

use crate::anthropic::{
    ANSWER_TOOL, AnthropicClient, AnthropicEvent, AnthropicRequest, ChatTurn, answer_tool,
    thinking_body, tools_from_openai,
};
use crate::openai::{self, ChatRequest, CloudEvent, OpenAiClient};

fn runtime() -> Result<Runtime, BackendError> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| BackendError(format!("tokio runtime: {e}")))
}

/// Neither client sends stop strings or `tool_choice` yet: refuse them
/// instead of dropping them silently.
fn reject_unsupported(provider: &str, req: &Request) -> Result<(), BackendError> {
    if !req.stop.is_empty() {
        return Err(BackendError(format!(
            "{provider}: stop strings are not supported"
        )));
    }
    if req.tool_choice.is_some() {
        return Err(BackendError(format!(
            "{provider}: tool_choice is not supported"
        )));
    }
    Ok(())
}

fn parse_json(field: &str, text: Option<&str>) -> Result<Option<Value>, BackendError> {
    text.map(serde_json::from_str)
        .transpose()
        .map_err(|e| BackendError(format!("{field}: {e}")))
}

fn cloud_usage(prompt_tokens: u32, generated_tokens: u32) -> Usage {
    Usage {
        prompt_tokens,
        generated_tokens,
        ..Usage::default()
    }
}

/// Close a reply the way the local engines do: `Usage`, then `Done`.
fn finish(mut events: Vec<Event>, usage: Option<Usage>, stop: StopReason) -> EventStream<'static> {
    events.extend(usage.map(Event::Usage));
    events.push(Event::Done(stop));
    Box::new(events.into_iter().map(Ok))
}

/// OpenAI Chat Completions (or any `base_url`-compatible server) as a [`Backend`].
pub struct OpenAiBackend {
    client: OpenAiClient,
    model: String,
    rt: Runtime,
}

impl OpenAiBackend {
    pub fn new(client: OpenAiClient, model: impl Into<String>) -> Result<Self, BackendError> {
        Ok(OpenAiBackend {
            client,
            model: model.into(),
            rt: runtime()?,
        })
    }
}

impl Backend for OpenAiBackend {
    fn generate(&mut self, req: &Request) -> Result<EventStream<'_>, BackendError> {
        reject_unsupported("openai", req)?;
        let chat = ChatRequest {
            model: self.model.clone(),
            messages: req
                .messages
                .iter()
                .map(|m| openai::ChatMessage {
                    role: m.role.clone(),
                    content: m.content.clone(),
                    tool_calls: m.tool_calls.clone(),
                    tool_call_id: m.tool_call_id.clone(),
                    ..openai::ChatMessage::default()
                })
                .collect(),
            think: req.think,
            max_tokens: Some(req.max_tokens),
            json_schema: parse_json("json_schema", req.json_schema.as_deref())?,
            tools: parse_json("tools", req.tools.as_deref())?,
        };
        let reply = self
            .rt
            .block_on(self.client.complete(chat))
            .map_err(|e| BackendError(e.to_string()))?;
        let mut events = Vec::new();
        let mut usage = None;
        for ev in reply {
            match ev {
                CloudEvent::Text(t) => events.push(Event::Text(t)),
                CloudEvent::Reasoning(t) if req.think.show => events.push(Event::Reasoning(t)),
                CloudEvent::Reasoning(_) => {}
                CloudEvent::ToolCalls(calls) => events.push(Event::ToolCalls(calls)),
                CloudEvent::Usage {
                    prompt_tokens,
                    completion_tokens,
                } => usage = Some(cloud_usage(prompt_tokens, completion_tokens)),
            }
        }
        // The client does not surface `finish_reason`.
        Ok(finish(events, usage, StopReason::Eos))
    }
}

/// Anthropic Messages as a [`Backend`].
pub struct AnthropicBackend {
    client: AnthropicClient,
    model: String,
    rt: Runtime,
}

impl AnthropicBackend {
    pub fn new(client: AnthropicClient, model: impl Into<String>) -> Result<Self, BackendError> {
        Ok(AnthropicBackend {
            client,
            model: model.into(),
            rt: runtime()?,
        })
    }
}

impl Backend for AnthropicBackend {
    fn generate(&mut self, req: &Request) -> Result<EventStream<'_>, BackendError> {
        reject_unsupported("anthropic", req)?;
        let structured = req.json_schema.is_some();
        let body = anthropic_request(&self.model, req)?;
        let reply = self
            .rt
            .block_on(self.client.generate(&body))
            .map_err(BackendError)?;
        let mut events = Vec::new();
        let mut usage = None;
        let mut stop = StopReason::Eos;
        for ev in reply {
            match ev {
                AnthropicEvent::Text(t) => events.push(Event::Text(t)),
                AnthropicEvent::Reasoning(t) if req.think.show => events.push(Event::Reasoning(t)),
                AnthropicEvent::Reasoning(_) => {}
                // Structured output: the forced tool's input is the answer.
                AnthropicEvent::ToolUse { calls, .. } if structured => events.extend(
                    calls
                        .into_iter()
                        .filter(|c| c.name == ANSWER_TOOL)
                        .map(|c| Event::Text(c.arguments)),
                ),
                AnthropicEvent::ToolUse { calls, .. } => events.push(Event::ToolCalls(calls)),
                AnthropicEvent::Usage {
                    input_tokens,
                    output_tokens,
                } => usage = Some(cloud_usage(input_tokens, output_tokens)),
                AnthropicEvent::Done { stop_reason } if stop_reason == "max_tokens" => {
                    stop = StopReason::MaxTokens;
                }
                AnthropicEvent::Done { .. } => {}
                AnthropicEvent::Refusal(s) => {
                    return Err(BackendError(format!("anthropic: refusal: {s}")));
                }
            }
        }
        Ok(finish(events, usage, stop))
    }
}

/// Map the portable request onto the Messages API: `system` messages join
/// into the top-level `system`, a run of `tool` messages becomes one user
/// turn of `tool_result` blocks, and assistant tool calls become `tool_use`
/// blocks.
pub(crate) fn anthropic_request(
    model: &str,
    req: &Request,
) -> Result<AnthropicRequest, BackendError> {
    let mut think = req.think;
    let (tools, tool_choice) = match parse_json("json_schema", req.json_schema.as_deref())? {
        Some(schema) => {
            if req.tools.is_some() {
                return Err("anthropic: json_schema with tools is not supported".into());
            }
            think.mode = ThinkMode::Off;
            let (tools, choice) = answer_tool(schema);
            (tools, Some(choice))
        }
        None => (
            parse_json("tools", req.tools.as_deref())?
                .as_ref()
                .map(tools_from_openai)
                .unwrap_or_default(),
            None,
        ),
    };
    let mut system = Vec::new();
    let mut turns = Vec::new();
    let mut results: Vec<(String, String)> = Vec::new();
    for m in &req.messages {
        if m.role == "tool" {
            let id = m
                .tool_call_id
                .clone()
                .ok_or("anthropic: a tool message needs tool_call_id")?;
            results.push((id, m.content.clone()));
            continue;
        }
        if !results.is_empty() {
            turns.push(ChatTurn::tool_results(&std::mem::take(&mut results)));
        }
        if m.role == "system" {
            system.push(m.content.as_str());
        } else if m.tool_calls.is_empty() {
            turns.push(ChatTurn::text(&m.role, m.content.clone()));
        } else if thinking_body(model, &think).is_some() {
            // The API wants the turn's signed thinking blocks back next to
            // its tool_use, and a portable message does not carry them.
            return Err(
                "anthropic: replaying tool calls with thinking on is not supported; turn thinking off"
                    .into(),
            );
        } else {
            turns.push(ChatTurn {
                role: m.role.clone(),
                text: String::new(),
                blocks: Some(tool_use_blocks(m)?),
            });
        }
    }
    if !results.is_empty() {
        turns.push(ChatTurn::tool_results(&results));
    }
    Ok(AnthropicRequest {
        model: model.to_owned(),
        system: (!system.is_empty()).then(|| system.join("\n\n")),
        messages: turns,
        think,
        max_tokens: req.max_tokens,
        images: Vec::new(),
        pdfs: Vec::new(),
        stream: false,
        tools,
        tool_choice,
    })
}

fn tool_use_blocks(m: &ChatMessage) -> Result<Value, BackendError> {
    let mut blocks = Vec::new();
    if !m.content.is_empty() {
        blocks.push(json!({"type": "text", "text": m.content}));
    }
    for call in &m.tool_calls {
        let input: Value = serde_json::from_str(&call.arguments)
            .map_err(|e| BackendError(format!("tool call {}: arguments: {e}", call.name)))?;
        blocks.push(json!({"type": "tool_use", "id": call.id, "name": call.name, "input": input}));
    }
    Ok(Value::Array(blocks))
}

#[cfg(test)]
mod tests {
    use super::*;
    use runa_core::{ThinkConfig, ToolCall};

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: role.into(),
            content: content.into(),
            ..ChatMessage::default()
        }
    }

    fn tool_round() -> Vec<ChatMessage> {
        vec![
            msg("system", "be brief"),
            msg("user", "weather?"),
            ChatMessage {
                role: "assistant".into(),
                content: "checking".into(),
                tool_calls: vec![ToolCall {
                    id: "t1".into(),
                    name: "weather".into(),
                    arguments: r#"{"city":"Kyiv"}"#.into(),
                }],
                ..ChatMessage::default()
            },
            ChatMessage {
                tool_call_id: Some("t1".into()),
                ..msg("tool", "sunny")
            },
        ]
    }

    #[test]
    fn anthropic_request_maps_system_tool_use_and_tool_results() {
        let req = Request {
            messages: tool_round(),
            max_tokens: 64,
            ..Request::default()
        };
        let body = anthropic_request("claude-3-5-sonnet", &req).unwrap();
        assert_eq!(body.system.as_deref(), Some("be brief"));
        assert_eq!(body.messages.len(), 3);
        assert_eq!(body.messages[0], ChatTurn::text("user", "weather?"));
        let use_blocks = body.messages[1].blocks.as_ref().unwrap();
        assert_eq!(use_blocks[0], json!({"type": "text", "text": "checking"}));
        assert_eq!(use_blocks[1]["type"], "tool_use");
        assert_eq!(use_blocks[1]["input"], json!({"city": "Kyiv"}));
        assert_eq!(
            body.messages[2],
            ChatTurn::tool_results(&[("t1".into(), "sunny".into())])
        );
        assert_eq!(body.max_tokens, 64);
        assert!(!body.stream);
    }

    #[test]
    fn anthropic_request_refuses_tool_replay_with_thinking_on() {
        let req = Request {
            messages: tool_round(),
            think: ThinkConfig {
                mode: ThinkMode::On,
                ..ThinkConfig::default()
            },
            ..Request::default()
        };
        let err = anthropic_request("claude-3-5-sonnet", &req).unwrap_err();
        assert!(err.0.contains("thinking"), "{err}");
    }

    #[test]
    fn anthropic_structured_output_forces_the_answer_tool_and_drops_thinking() {
        let req = Request {
            messages: vec![ChatMessage::user("hi")],
            think: ThinkConfig {
                mode: ThinkMode::On,
                ..ThinkConfig::default()
            },
            json_schema: Some(r#"{"type":"object"}"#.into()),
            ..Request::default()
        };
        let body = anthropic_request("claude-3-5-sonnet", &req).unwrap();
        assert_eq!(body.think.mode, ThinkMode::Off);
        assert_eq!(body.tools[0]["name"], ANSWER_TOOL);
        assert_eq!(body.tool_choice.unwrap()["name"], ANSWER_TOOL);
    }

    #[test]
    fn unsupported_fields_fail_instead_of_being_dropped() {
        let stop = Request {
            stop: vec!["END".into()],
            ..Request::default()
        };
        assert!(reject_unsupported("openai", &stop).is_err());
        let choice = Request {
            tool_choice: Some("required".into()),
            ..Request::default()
        };
        assert!(reject_unsupported("anthropic", &choice).is_err());
        let bad_json = Request {
            tools: Some("[not json".into()),
            ..Request::default()
        };
        assert!(
            anthropic_request("m", &bad_json)
                .unwrap_err()
                .0
                .starts_with("tools:")
        );
    }
}
