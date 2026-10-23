// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! The data every backend speaks (P21.1): [`ChatMessage`] and [`Request`]
//! in, [`Event`] out. Local engines and cloud adapters map these onto
//! their own wire shapes; nothing here touches a model or the network.

use crate::ToolCall;
use crate::think::ThinkConfig;

/// One chat message.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChatMessage {
    /// `system` / `user` / `assistant` / `tool` (passed through to the template).
    pub role: String,
    /// Message text.
    pub content: String,
    /// Tool calls an `assistant` turn made (P8.2).
    pub tool_calls: Vec<ToolCall>,
    /// The call a `tool` message answers (P8.2).
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    /// Shorthand for a user message.
    pub fn user(content: impl Into<String>) -> ChatMessage {
        ChatMessage {
            role: "user".to_owned(),
            content: content.into(),
            ..ChatMessage::default()
        }
    }
}

/// What to generate, in the shape every backend accepts. Engine-only knobs
/// (sampler chain, media, speculation, grammar) stay on the engine's own
/// request type.
#[derive(Debug, Clone, PartialEq)]
pub struct Request {
    /// Conversation so far.
    pub messages: Vec<ChatMessage>,
    /// Hard stop after this many new tokens.
    pub max_tokens: u32,
    /// Stop strings (the match itself is not emitted).
    pub stop: Vec<String>,
    /// Thinking mode, and whether [`Event::Reasoning`] is emitted.
    pub think: ThinkConfig,
    /// The answer must match this JSON Schema (P8.1).
    pub json_schema: Option<String>,
    /// OpenAI-shape `tools` JSON array the model may call (P8.2).
    pub tools: Option<String>,
    /// `auto` (default) / `required` / `none` (P8.2).
    pub tool_choice: Option<String>,
}

impl Default for Request {
    fn default() -> Self {
        Request {
            messages: Vec::new(),
            max_tokens: 512,
            stop: Vec::new(),
            think: ThinkConfig::default(),
            json_schema: None,
            tools: None,
            tool_choice: None,
        }
    }
}

/// Token/speed counters for one generation.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Usage {
    /// Prompt tokens decoded.
    pub prompt_tokens: u32,
    /// Tokens generated (excluding the stop token).
    pub generated_tokens: u32,
    /// Reasoning-body tokens among the generated ones (P10.4, M6
    /// follow-up): counted by the same transition rule as
    /// `BudgetClock` (token observed while the think block is open,
    /// open tag excluded, ±1 at block edges from piece batching).
    /// 0 when thinking is off or the backend does not split
    /// reasoning (mistral, cloud, daemon<VERSION without the field>).
    pub reasoning_tokens: u32,
    /// Prompt-processing speed (tok/s); 0 when the backend does not time it.
    pub pp_toks_per_s: f64,
    /// Decode speed (tok/s); 0 when the backend does not time it.
    pub tg_toks_per_s: f64,
}

/// Why generation stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopReason {
    /// The model finished its turn (EOS, or a cloud `end_turn`).
    Eos,
    /// Hit `max_tokens`.
    MaxTokens,
    /// A stop string matched (the match itself is not emitted).
    StopString(String),
}

/// Stream items. `Text` carries answer pieces (already stop-filtered).
/// The stream ends with `Done`; `Usage`, when the backend counts tokens,
/// comes right before it.
#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    Text(String),
    /// Thinking / chain-of-thought (P3.2). Omitted when `think.show` is false.
    Reasoning(String),
    /// Calls parsed from the reply of a request with `tools` (P8.2). Such a
    /// request sends its text as one `Text` at the end, markup stripped.
    ToolCalls(Vec<ToolCall>),
    Usage(Usage),
    Done(StopReason),
}
