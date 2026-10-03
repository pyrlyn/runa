// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! OpenAI adapter (plan P3.5 / D9).
//!
//! Chat Completions via `async-openai`, with `base_url` override, ThinkConfig →
//! `reasoning.effort`, image (`image_url`) and audio (`input_audio`) parts, and
//! `reasoning` / `reasoning_content` split into [`CloudEvent::Reasoning`].

use async_openai::Client;
use async_openai::config::OpenAIConfig;
use async_openai::error::OpenAIError;
use async_openai::types::chat::{
    ChatCompletionRequestAssistantMessage, ChatCompletionRequestAssistantMessageContent,
    ChatCompletionRequestMessage, ChatCompletionRequestMessageContentPartAudio,
    ChatCompletionRequestMessageContentPartImage, ChatCompletionRequestMessageContentPartText,
    ChatCompletionRequestSystemMessage, ChatCompletionRequestSystemMessageContent,
    ChatCompletionRequestUserMessage, ChatCompletionRequestUserMessageContent,
    ChatCompletionRequestUserMessageContentPart, CreateChatCompletionRequest, ImageUrl, InputAudio,
    InputAudioFormat, ReasoningEffort, ResponseFormat, ResponseFormatJsonSchema,
};
use async_openai::types::responses::Reasoning;
use futures::StreamExt;
use runa_core::{Effort, ThinkConfig, ThinkMode, ToolCall};
use serde_json::{Value, json};

/// One turn sent to the cloud model.
#[derive(Debug, Clone, Default)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    pub images: Vec<ImageInput>,
    pub audio: Vec<AudioInput>,
    /// Calls an `assistant` turn made (P8.3).
    pub tool_calls: Vec<ToolCall>,
    /// The call a `tool` turn answers (P8.3).
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    pub fn user(content: impl Into<String>) -> Self {
        ChatMessage {
            role: "user".into(),
            content: content.into(),
            ..ChatMessage::default()
        }
    }
}

/// `image_url` part: https URL or `data:image/...;base64,...`.
#[derive(Debug, Clone)]
pub struct ImageInput {
    pub url: String,
}

/// `input_audio` part.
#[derive(Debug, Clone)]
pub struct AudioInput {
    pub data_b64: String,
    pub format: AudioFormat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioFormat {
    Wav,
    Mp3,
}

/// One cloud generation request.
#[derive(Debug, Clone)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub think: ThinkConfig,
    pub max_tokens: Option<u32>,
    /// Structured output (P8.1): strict `response_format` JSON Schema.
    pub json_schema: Option<Value>,
    /// OpenAI-shape `tools` array (P8.3).
    pub tools: Option<Value>,
}

/// Stream items. Reasoning is split from answer text (D7).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CloudEvent {
    Reasoning(String),
    Text(String),
    /// Function calls from a non-streamed reply (P8.3).
    ToolCalls(Vec<ToolCall>),
    Usage {
        prompt_tokens: u32,
        completion_tokens: u32,
    },
}

#[derive(Debug)]
pub enum CloudError {
    OpenAi(String),
    Empty,
}

impl std::fmt::Display for CloudError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CloudError::OpenAi(s) => write!(f, "openai: {s}"),
            CloudError::Empty => write!(f, "openai: empty choices"),
        }
    }
}

impl std::error::Error for CloudError {}

impl From<OpenAIError> for CloudError {
    fn from(e: OpenAIError) -> Self {
        CloudError::OpenAi(e.to_string())
    }
}

/// OpenAI-compatible chat client (`base_url` for OpenRouter / Groq / llama-server).
pub struct OpenAiClient {
    inner: Client<OpenAIConfig>,
}

impl OpenAiClient {
    pub fn new(api_key: impl Into<String>, base_url: Option<&str>) -> Self {
        let mut cfg = OpenAIConfig::new().with_api_key(api_key.into());
        if let Some(base) = base_url {
            cfg = cfg.with_api_base(base);
        }
        OpenAiClient {
            inner: Client::with_config(cfg),
        }
    }

    pub async fn complete(&self, req: ChatRequest) -> Result<Vec<CloudEvent>, CloudError> {
        let body = build_chat_request(&req, false)?;
        let resp = self.inner.chat().create(body).await?;
        let choice = resp.choices.into_iter().next().ok_or(CloudError::Empty)?;
        let raw = serde_json::to_value(&choice.message).unwrap_or(Value::Null);
        let mut events = events_from_assistant(&raw, choice.message.content.as_deref());
        let calls = tool_calls_from_message(&raw);
        if !calls.is_empty() {
            events.push(CloudEvent::ToolCalls(calls));
        }
        if let Some(u) = resp.usage {
            events.push(CloudEvent::Usage {
                prompt_tokens: u.prompt_tokens,
                completion_tokens: u.completion_tokens,
            });
        }
        Ok(events)
    }

    pub async fn stream(&self, req: ChatRequest) -> Result<Vec<CloudEvent>, CloudError> {
        let body = build_chat_request(&req, true)?;
        let mut s = self.inner.chat().create_stream(body).await?;
        let mut events = Vec::new();
        while let Some(chunk) = s.next().await {
            let chunk = chunk?;
            if let Some(u) = chunk.usage {
                events.push(CloudEvent::Usage {
                    prompt_tokens: u.prompt_tokens,
                    completion_tokens: u.completion_tokens,
                });
            }
            let Some(choice) = chunk.choices.into_iter().next() else {
                continue;
            };
            let raw = serde_json::to_value(&choice.delta).unwrap_or(Value::Null);
            events.extend(events_from_assistant(&raw, choice.delta.content.as_deref()));
        }
        Ok(events)
    }
}

/// Map [`ThinkConfig`] onto OpenAI `reasoning.effort`.
pub fn openai_reasoning_effort(think: ThinkConfig) -> Option<ReasoningEffort> {
    match think.mode {
        ThinkMode::Off => None,
        ThinkMode::On => Some(ReasoningEffort::Medium),
        ThinkMode::Budget { tokens, .. } => Some(if tokens <= 512 {
            ReasoningEffort::Low
        } else if tokens <= 2048 {
            ReasoningEffort::Medium
        } else {
            ReasoningEffort::High
        }),
        ThinkMode::Effort(Effort::Low) => Some(ReasoningEffort::Low),
        ThinkMode::Effort(Effort::Medium) => Some(ReasoningEffort::Medium),
        ThinkMode::Effort(Effort::High) => Some(ReasoningEffort::High),
        ThinkMode::Effort(Effort::Max) => Some(ReasoningEffort::Xhigh),
    }
}

/// Responses API `reasoning` object (effort only).
pub fn responses_reasoning(think: ThinkConfig) -> Option<Reasoning> {
    openai_reasoning_effort(think).map(|effort| Reasoning {
        effort: Some(effort),
        summary: None,
        // 0.42 additions ( Responses `reasoning` object): unset = same wire
        // shape as before.
        mode: None,
        context: None,
    })
}

/// Split `reasoning` / `reasoning_content` from answer `content`.
pub fn split_assistant_fields(v: &Value) -> (Option<String>, Option<String>) {
    let reasoning = v
        .get("reasoning")
        .and_then(Value::as_str)
        .or_else(|| v.get("reasoning_content").and_then(Value::as_str))
        .map(str::to_string)
        .filter(|s| !s.is_empty());
    let text = v
        .get("content")
        .and_then(Value::as_str)
        .map(str::to_string)
        .filter(|s| !s.is_empty());
    (reasoning, text)
}

fn events_from_assistant(raw: &Value, content_fallback: Option<&str>) -> Vec<CloudEvent> {
    let (reasoning, text) = split_assistant_fields(raw);
    let mut out = Vec::new();
    if let Some(r) = reasoning {
        out.push(CloudEvent::Reasoning(r));
    }
    if let Some(t) = text.or_else(|| {
        content_fallback
            .map(str::to_string)
            .filter(|s| !s.is_empty())
    }) {
        out.push(CloudEvent::Text(t));
    }
    out
}

/// `message.tool_calls` → [`ToolCall`]s (function calls only).
fn tool_calls_from_message(raw: &Value) -> Vec<ToolCall> {
    raw["tool_calls"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|c| {
            Some(ToolCall {
                id: c["id"].as_str().unwrap_or_default().to_owned(),
                name: c["function"]["name"].as_str()?.to_owned(),
                arguments: c["function"]["arguments"]
                    .as_str()
                    .unwrap_or("{}")
                    .to_owned(),
            })
        })
        .collect()
}

fn build_chat_request(
    req: &ChatRequest,
    stream: bool,
) -> Result<CreateChatCompletionRequest, CloudError> {
    let mut messages = Vec::with_capacity(req.messages.len());
    for m in &req.messages {
        messages.push(to_openai_message(m)?);
    }
    if messages.is_empty() {
        return Err(CloudError::Empty);
    }
    Ok(CreateChatCompletionRequest {
        model: req.model.clone(),
        messages,
        reasoning_effort: openai_reasoning_effort(req.think),
        max_completion_tokens: req.max_tokens,
        stream: stream.then_some(true),
        response_format: req
            .json_schema
            .clone()
            .map(|schema| ResponseFormat::JsonSchema {
                json_schema: ResponseFormatJsonSchema {
                    description: None,
                    name: "answer".into(),
                    schema,
                    strict: Some(true),
                },
            }),
        tools: req
            .tools
            .clone()
            .map(serde_json::from_value)
            .transpose()
            .map_err(|e| CloudError::OpenAi(format!("tools: {e}")))?,
        ..Default::default()
    })
}

fn to_openai_message(m: &ChatMessage) -> Result<ChatCompletionRequestMessage, CloudError> {
    let role = m.role.to_ascii_lowercase();
    // Tool turns go through serde: the builder types for them are verbose.
    let tool_turn = match role.as_str() {
        "tool" => Some(json!({
            "role": "tool",
            "content": m.content,
            "tool_call_id": m.tool_call_id.clone().unwrap_or_default(),
        })),
        "assistant" if !m.tool_calls.is_empty() => Some(json!({
            "role": "assistant",
            "content": (!m.content.is_empty()).then_some(&m.content),
            "tool_calls": m.tool_calls.iter().map(|c| json!({
                "id": c.id,
                "type": "function",
                "function": {"name": c.name, "arguments": c.arguments},
            })).collect::<Vec<_>>(),
        })),
        _ => None,
    };
    if let Some(v) = tool_turn {
        return serde_json::from_value(v).map_err(|e| CloudError::OpenAi(e.to_string()));
    }
    match role.as_str() {
        "system" | "developer" => Ok(ChatCompletionRequestMessage::System(
            ChatCompletionRequestSystemMessage {
                content: ChatCompletionRequestSystemMessageContent::Text(m.content.clone()),
                name: None,
            },
        )),
        "assistant" => Ok(ChatCompletionRequestMessage::Assistant(
            ChatCompletionRequestAssistantMessage {
                content: Some(ChatCompletionRequestAssistantMessageContent::Text(
                    m.content.clone(),
                )),
                ..Default::default()
            },
        )),
        _ => Ok(ChatCompletionRequestMessage::User(
            ChatCompletionRequestUserMessage {
                content: user_content(m),
                name: None,
            },
        )),
    }
}

fn user_content(m: &ChatMessage) -> ChatCompletionRequestUserMessageContent {
    if m.images.is_empty() && m.audio.is_empty() {
        return ChatCompletionRequestUserMessageContent::Text(m.content.clone());
    }
    let mut parts = Vec::new();
    if !m.content.is_empty() {
        parts.push(ChatCompletionRequestUserMessageContentPart::Text(
            ChatCompletionRequestMessageContentPartText {
                text: m.content.clone(),
                prompt_cache_breakpoint: None,
            },
        ));
    }
    for img in &m.images {
        parts.push(ChatCompletionRequestUserMessageContentPart::ImageUrl(
            ChatCompletionRequestMessageContentPartImage {
                image_url: ImageUrl {
                    url: img.url.clone(),
                    detail: None,
                },
                prompt_cache_breakpoint: None,
            },
        ));
    }
    for a in &m.audio {
        parts.push(ChatCompletionRequestUserMessageContentPart::InputAudio(
            ChatCompletionRequestMessageContentPartAudio {
                input_audio: InputAudio {
                    data: a.data_b64.clone(),
                    format: match a.format {
                        AudioFormat::Wav => InputAudioFormat::Wav,
                        AudioFormat::Mp3 => InputAudioFormat::Mp3,
                    },
                },
                prompt_cache_breakpoint: None,
            },
        ));
    }
    ChatCompletionRequestUserMessageContent::Array(parts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effort_table() {
        let off = ThinkConfig::default();
        assert!(openai_reasoning_effort(off).is_none());
        let high = ThinkConfig {
            mode: ThinkMode::Effort(Effort::High),
            show: false,
        };
        assert_eq!(openai_reasoning_effort(high), Some(ReasoningEffort::High));
        let budget = ThinkConfig {
            mode: ThinkMode::Budget {
                tokens: 256,
                grace: 64,
            },
            show: false,
        };
        assert_eq!(openai_reasoning_effort(budget), Some(ReasoningEffort::Low));
        let max = ThinkConfig {
            mode: ThinkMode::Effort(Effort::Max),
            show: false,
        };
        assert_eq!(openai_reasoning_effort(max), Some(ReasoningEffort::Xhigh));
    }

    #[test]
    fn normalize_reasoning_content() {
        let v: Value = serde_json::json!({
            "role": "assistant",
            "content": "Paris",
            "reasoning_content": "the capital"
        });
        let (r, t) = split_assistant_fields(&v);
        assert_eq!(r.as_deref(), Some("the capital"));
        assert_eq!(t.as_deref(), Some("Paris"));
    }

    #[test]
    fn normalize_reasoning_field() {
        let v: Value = serde_json::json!({
            "content": "42",
            "reasoning": "count"
        });
        let (r, t) = split_assistant_fields(&v);
        assert_eq!(r.as_deref(), Some("count"));
        assert_eq!(t.as_deref(), Some("42"));
    }

    #[test]
    fn chat_request_serializes_image_and_audio() {
        let req = ChatRequest {
            model: "gpt-4o".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: "what".into(),
                images: vec![ImageInput {
                    url: "data:image/png;base64,AAA".into(),
                }],
                audio: vec![AudioInput {
                    data_b64: "YmE=".into(),
                    format: AudioFormat::Wav,
                }],
                ..ChatMessage::default()
            }],
            think: ThinkConfig {
                mode: ThinkMode::Effort(Effort::Low),
                show: false,
            },
            max_tokens: Some(32),
            json_schema: Some(serde_json::json!({"type": "object"})),
            tools: None,
        };
        let body = build_chat_request(&req, false).unwrap();
        let v = serde_json::to_value(&body).unwrap();
        assert_eq!(v["reasoning_effort"], "low");
        assert_eq!(v["response_format"]["type"], "json_schema");
        assert_eq!(v["response_format"]["json_schema"]["strict"], true);
        assert_eq!(
            v["response_format"]["json_schema"]["schema"]["type"],
            "object"
        );
        assert_eq!(v["max_completion_tokens"], 32);
        let parts = &v["messages"][0]["content"];
        assert!(
            parts
                .as_array()
                .unwrap()
                .iter()
                .any(|p| p["type"] == "image_url")
        );
        assert!(
            parts
                .as_array()
                .unwrap()
                .iter()
                .any(|p| p["type"] == "input_audio")
        );
    }

    #[test]
    fn tool_turns_round_trip() {
        let call = ToolCall {
            id: "c1".into(),
            name: "get_weather".into(),
            arguments: r#"{"city":"Paris"}"#.into(),
        };
        let req = ChatRequest {
            model: "gpt-5".into(),
            messages: vec![
                ChatMessage::user("weather?"),
                ChatMessage {
                    role: "assistant".into(),
                    tool_calls: vec![call.clone()],
                    ..ChatMessage::default()
                },
                ChatMessage {
                    role: "tool".into(),
                    content: "sunny".into(),
                    tool_call_id: Some("c1".into()),
                    ..ChatMessage::default()
                },
            ],
            think: ThinkConfig::default(),
            max_tokens: None,
            json_schema: None,
            tools: Some(json!([{"type": "function", "function": {
                "name": "get_weather", "parameters": {"type": "object"}}}])),
        };
        let v = serde_json::to_value(build_chat_request(&req, false).unwrap()).unwrap();
        assert_eq!(v["tools"][0]["function"]["name"], "get_weather");
        assert_eq!(v["messages"][1]["tool_calls"][0]["id"], "c1");
        assert_eq!(v["messages"][2]["role"], "tool");
        assert_eq!(v["messages"][2]["tool_call_id"], "c1");
        let reply = json!({"tool_calls": [{"id": "c1", "type": "function",
            "function": {"name": "get_weather", "arguments": r#"{"city":"Paris"}"#}}]});
        assert_eq!(tool_calls_from_message(&reply), vec![call]);
    }

    #[test]
    fn responses_reasoning_serializes() {
        let think = ThinkConfig {
            mode: ThinkMode::Effort(Effort::Medium),
            show: false,
        };
        let r = responses_reasoning(think).unwrap();
        let v = serde_json::to_value(&r).unwrap();
        assert_eq!(v["effort"], "medium");
    }
}
