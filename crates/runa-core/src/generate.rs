// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! The [`Backend`] trait every generator implements (P21.1): the ggml and
//! mistral.rs engines, the OpenAI and Anthropic adapters, and
//! [`ScriptedBackend`] for tests that must not load a model.

use std::collections::VecDeque;

use crate::chat::{Event, Request};

/// A failed generation, already rendered for the user.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendError(pub String);

impl std::fmt::Display for BackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for BackendError {}

impl From<String> for BackendError {
    fn from(s: String) -> Self {
        BackendError(s)
    }
}

impl From<&str> for BackendError {
    fn from(s: &str) -> Self {
        BackendError(s.to_owned())
    }
}

/// Events of one generation. Local engines decode lazily while the caller
/// pulls, so the stream may borrow the backend.
pub type EventStream<'a> = Box<dyn Iterator<Item = Result<Event, BackendError>> + 'a>;

/// Anything that turns a [`Request`] into a stream of [`Event`]s.
///
/// `&mut self` because local engines own one KV cache and decode one
/// request at a time.
pub trait Backend {
    /// Start one generation. Setup errors (bad template, unsupported
    /// field, unreachable API) fail here; decode errors arrive in the stream.
    fn generate(&mut self, req: &Request) -> Result<EventStream<'_>, BackendError>;
}

/// A fake backend that replays scripted turns, one per `generate` call,
/// and records every request it got. Lets CLI and server paths run in
/// tests without weights or network.
#[derive(Debug, Default)]
pub struct ScriptedBackend {
    turns: VecDeque<Vec<Event>>,
    /// Requests seen so far, oldest first.
    pub requests: Vec<Request>,
}

impl ScriptedBackend {
    /// Replay `turns` in order; a call past the last turn fails.
    pub fn new(turns: impl IntoIterator<Item = Vec<Event>>) -> ScriptedBackend {
        ScriptedBackend {
            turns: turns.into_iter().collect(),
            requests: Vec::new(),
        }
    }
}

impl Backend for ScriptedBackend {
    fn generate(&mut self, req: &Request) -> Result<EventStream<'_>, BackendError> {
        self.requests.push(req.clone());
        let turn = self
            .turns
            .pop_front()
            .ok_or_else(|| BackendError::from("scripted backend: no turn left"))?;
        Ok(Box::new(turn.into_iter().map(Ok)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::{ChatMessage, StopReason, Usage};

    fn drain(backend: &mut dyn Backend, req: &Request) -> Vec<Event> {
        backend
            .generate(req)
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap()
    }

    #[test]
    fn scripted_backend_replays_turns_in_order_and_records_requests() {
        let first = vec![
            Event::Text("hi".into()),
            Event::Usage(Usage {
                prompt_tokens: 3,
                generated_tokens: 1,
                ..Usage::default()
            }),
            Event::Done(StopReason::Eos),
        ];
        let second = vec![Event::Done(StopReason::MaxTokens)];
        let mut backend = ScriptedBackend::new([first.clone(), second.clone()]);
        let req = Request {
            messages: vec![ChatMessage::user("hello")],
            ..Request::default()
        };
        let boxed: &mut dyn Backend = &mut backend;
        assert_eq!(drain(boxed, &req), first);
        assert_eq!(drain(boxed, &Request::default()), second);
        let err = backend.generate(&req).err().unwrap();
        assert!(err.to_string().contains("no turn left"), "{err}");
        assert_eq!(backend.requests.len(), 3);
        assert_eq!(backend.requests[0], req);
    }

    #[test]
    fn request_default_matches_the_engine_default_budget() {
        let req = Request::default();
        assert_eq!(req.max_tokens, 512);
        assert!(req.messages.is_empty() && req.stop.is_empty());
        assert!(req.json_schema.is_none() && req.tools.is_none() && req.tool_choice.is_none());
    }
}
