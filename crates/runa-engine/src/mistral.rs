// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! Safetensors backend over mistral.rs (plan P9.2).
//!
//! [`MistralModel::load`] takes a model **directory** (Hugging Face snapshot
//! layout: `config.json` + tokenizer + `*.safetensors`) and [`generate`]
//! streams [`GenEvent`]s with the same terminal order as the ggml backend
//! (`Text…`, `Usage`, `Done`). Request mapping covers sampling, thinking,
//! stop sequences and usage; everything else is an explicit
//! [`EngineError::Mistral`] — GGUF-only options (`--mcp` tools,
//! `--json-schema`/`--grammar`, `--audio`/`--image`/`--video`,
//! `--ngram`/`--draft`) never silently change meaning here.
//!
//! Sync façade: mistral.rs ships `blocking::{BlockingModel, BlockingStream}`
//! (own tokio runtime, token iterator), so runa-engine needs no async of its
//! own. `BlockingModel` must not run inside an existing tokio runtime —
//! `runa run`/`chat` are sync and `serve` drives engines from plain std
//! threads, so this holds.

#![cfg(feature = "mistralrs")]

use std::collections::VecDeque;
use std::path::{Path, PathBuf};

use mistralrs::{
    ChatCompletionChunkResponse, ChatCompletionResponse, RequestBuilder, Response, StopTokens,
    TextMessageRole, Usage as MistralUsage,
    blocking::{BlockingModel, BlockingStream},
};

use crate::generate::{ChatMessage, GenEvent, GenerateRequest, StopReason, Usage};
use crate::load::EngineError;

/// A mistral.rs model loaded from a local safetensors directory.
pub struct MistralModel {
    model: BlockingModel,
    path: PathBuf,
}

impl MistralModel {
    /// Load a safetensors model directory (must hold `config.json`).
    ///
    /// The directory check runs before any mistral.rs call so a missing path
    /// fails fast without network traffic; an existing directory then loads
    /// fully offline through mistral.rs local-path support.
    pub fn load(path: &Path) -> Result<MistralModel, EngineError> {
        if !runa_core::is_mistral_dir(path) {
            return Err(EngineError::Mistral(format!(
                "{}: --backend mistral needs a model directory holding config.json \
                 (`runa pull hf:<repo>:safetensors` lays one out)",
                path.display()
            )));
        }
        eprintln!("runa load {} · mistral (safetensors)", path.display());
        let id = path.to_string_lossy().into_owned();
        let builder = mistralrs::ModelBuilder::new(id);
        let model = BlockingModel::from_auto_builder(builder).map_err(|e| {
            EngineError::Mistral(format!(
                "mistral.rs load failed for {}: {e}",
                path.display()
            ))
        })?;
        Ok(MistralModel {
            model,
            path: path.to_owned(),
        })
    }

    /// Model directory this was loaded from.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Stream one generation. GGUF-only request fields are rejected here
    /// with an explicit error (see the module docs).
    pub fn generate(&mut self, req: GenerateRequest) -> Result<MistralGeneration, EngineError> {
        let show_reasoning = req.think.show;
        let builder = request_builder(&req)?;
        let stream = self
            .model
            .stream_chat_request(builder)
            .map_err(|e| EngineError::Mistral(format!("mistral.rs request failed: {e}")))?;
        Ok(MistralGeneration {
            stream,
            fold: EventFold::new(show_reasoning),
        })
    }
}

/// Lazily-driven mistral.rs generation. Terminal order matches the ggml
/// backend: queued `Text`/`Reasoning` pieces, then `Usage`, then `Done`.
pub struct MistralGeneration {
    stream: BlockingStream,
    fold: EventFold,
}

impl Iterator for MistralGeneration {
    type Item = Result<GenEvent, EngineError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(ev) = self.fold.pop() {
                return Some(Ok(ev));
            }
            if self.fold.is_finished() {
                return None;
            }
            match self.fold.step(self.stream.next()) {
                Ok(()) => continue, // queued more (or terminal); serve above.
                Err(e) => return Some(Err(e)),
            }
        }
    }
}

/// Accumulates stream items into [`GenEvent`]s. Pure state machine over
/// mistral.rs [`Response`]s, unit-tested without a model.
#[derive(Debug, Default)]
struct EventFold {
    ready: VecDeque<GenEvent>,
    usage: Option<Usage>,
    stop: Option<StopReason>,
    show_reasoning: bool,
    finished: bool,
    /// True once any text was emitted; guards against emitting a terminal
    /// `Done` message twice (streamed chunks + summary).
    emitted_text: bool,
}

impl EventFold {
    fn new(show_reasoning: bool) -> EventFold {
        EventFold {
            show_reasoning,
            ..EventFold::default()
        }
    }

    fn pop(&mut self) -> Option<GenEvent> {
        self.ready.pop_front()
    }

    fn is_finished(&self) -> bool {
        self.finished
    }

    fn close(&mut self) {
        if self.finished {
            return;
        }
        let usage = self.usage.take().unwrap_or(Usage {
            prompt_tokens: 0,
            generated_tokens: 0,
            // P10.4: mistral.rs reports one completion count with no
            // reasoning split — stays 0 (documented on the field).
            reasoning_tokens: 0,
            pp_toks_per_s: 0.0,
            tg_toks_per_s: 0.0,
        });
        self.ready.push_back(GenEvent::Usage(usage));
        self.ready
            .push_back(GenEvent::Done(self.stop.take().unwrap_or(StopReason::Eos)));
        self.finished = true;
    }

    /// Fold one stream item (`None` = exhausted). Queues events, closing the
    /// stream (with `Usage` + `Done`) on terminal items.
    fn step(&mut self, item: Option<Response>) -> Result<(), EngineError> {
        let Some(resp) = item else {
            // Exhausted without a terminal state: close with what we have.
            self.close();
            return Ok(());
        };
        match resp {
            Response::Chunk(c) => {
                self.absorb_chunk(&c);
                if c.choices
                    .first()
                    .and_then(|ch| ch.finish_reason.as_deref())
                    .is_some()
                {
                    self.close();
                }
                Ok(())
            }
            Response::Done(d) => {
                self.absorb_done(&d);
                self.close();
                Ok(())
            }
            Response::ModelError(msg, _) => Err(EngineError::Mistral(format!(
                "mistral.rs model error: {msg}"
            ))),
            Response::InternalError(e) => {
                Err(EngineError::Mistral(format!("mistral.rs error: {e}")))
            }
            Response::ValidationError(e) => Err(EngineError::Mistral(format!(
                "mistral.rs rejected the request: {e}"
            ))),
            other => Err(EngineError::Mistral(format!(
                "mistral.rs returned an unexpected response: {}",
                response_kind(&other)
            ))),
        }
    }

    fn absorb_chunk(&mut self, c: &ChatCompletionChunkResponse) {
        if let Some(u) = c.usage.as_ref() {
            self.usage = Some(map_usage(u));
        }
        let Some(choice) = c.choices.first() else {
            return;
        };
        if let Some(stop) = choice.finish_reason.as_deref() {
            self.stop = Some(map_stop(stop));
        }
        let delta = &choice.delta;
        if let Some(t) = delta.content.as_deref()
            && !t.is_empty()
        {
            self.emitted_text = true;
            self.ready.push_back(GenEvent::Text(t.to_owned()));
        }
        if self.show_reasoning
            && let Some(r) = delta.reasoning_content.as_deref()
            && !r.is_empty()
        {
            self.ready.push_back(GenEvent::Reasoning(r.to_owned()));
        }
        // Requested tools are refused up front, so any calls here are
        // volunteered: surface them as text (the ggml backend likewise
        // surfaces unrequested tool markup as text instead of events).
        if let Some(calls) = delta.tool_calls.as_deref() {
            for call in calls {
                self.emitted_text = true;
                self.ready.push_back(GenEvent::Text(format!(
                    "{{\"name\":{},\"arguments\":{}}}",
                    quote(&call.function.name),
                    quote(&call.function.arguments)
                )));
            }
        }
    }

    fn absorb_done(&mut self, d: &ChatCompletionResponse) {
        self.usage = Some(map_usage(&d.usage));
        if let Some(choice) = d.choices.first() {
            self.stop = Some(map_stop(&choice.finish_reason));
            // Chunks already carried the text; only fall back to the summary
            // message when the stream produced none.
            if !self.emitted_text {
                if let Some(t) = choice.message.content.as_deref()
                    && !t.is_empty()
                {
                    self.emitted_text = true;
                    self.ready.push_back(GenEvent::Text(t.to_owned()));
                }
                if self.show_reasoning
                    && let Some(r) = choice.message.reasoning_content.as_deref()
                    && !r.is_empty()
                {
                    self.ready.push_back(GenEvent::Reasoning(r.to_owned()));
                }
            }
        }
    }
}

/// Map a [`GenerateRequest`] onto mistral.rs. Sampling, thinking, stop
/// sequences and `max_tokens` carry over; GGUF-only fields fail explicitly.
fn request_builder(req: &GenerateRequest) -> Result<RequestBuilder, EngineError> {
    reject_unsupported(req)?;
    let mut b = RequestBuilder::new();
    for m in &req.messages {
        b = b.add_message(map_role(m)?, m.content.clone());
    }
    b = b.enable_thinking(!matches!(req.think.mode, runa_core::ThinkMode::Off));
    let s = &req.sampling;
    if s.temperature <= 0.0 {
        b = b.set_deterministic_sampler();
    } else {
        b = b.set_sampler_temperature(f64::from(s.temperature));
        if s.top_k > 0 {
            b = b.set_sampler_topk(s.top_k as usize);
        }
        b = b
            .set_sampler_topp(f64::from(s.top_p))
            .set_sampler_minp(f64::from(s.min_p));
    }
    b = b.set_sampler_max_len(req.max_tokens as usize);
    if !req.stop.is_empty() {
        b = b.set_sampler_stop_toks(StopTokens::Seqs(req.stop.clone()));
    }
    Ok(b)
}

fn reject_unsupported(req: &GenerateRequest) -> Result<(), EngineError> {
    if req.tools.is_some() {
        return Err(EngineError::Mistral(
            "--mcp tools need the gguf backend; the mistral backend does not take tool specs yet"
                .into(),
        ));
    }
    if req.json_schema.is_some() || req.grammar.is_some() {
        return Err(EngineError::Mistral(
            "--json-schema/--grammar need the gguf backend; the mistral backend is unconstrained text"
                .into(),
        ));
    }
    if req.audio_pcm.is_some() || !req.images.is_empty() {
        return Err(EngineError::Mistral(
            "--audio/--image/--video need the gguf backend (native mtmd)".into(),
        ));
    }
    if req.speculative.ngram || req.speculative.draft.is_some() {
        return Err(EngineError::Mistral(
            "--ngram/--draft need the gguf backend; the mistral backend runs its own scheduler"
                .into(),
        ));
    }
    Ok(())
}

fn map_role(m: &ChatMessage) -> Result<TextMessageRole, EngineError> {
    match m.role.as_str() {
        "system" => Ok(TextMessageRole::System),
        "user" => Ok(TextMessageRole::User),
        "assistant" => Ok(TextMessageRole::Assistant),
        other => Err(EngineError::Mistral(format!(
            "mistral backend: message role {other:?} needs the gguf backend (tool rounds)"
        ))),
    }
}

/// mistral.rs finish reasons: `length` exhausted `max_len`, everything else
/// (notably `stop`) ends the turn like a ggml EOS.
fn map_stop(reason: &str) -> StopReason {
    match reason {
        "length" => StopReason::MaxTokens,
        _ => StopReason::Eos,
    }
}

fn map_usage(u: &MistralUsage) -> Usage {
    Usage {
        prompt_tokens: u.prompt_tokens.min(u32::MAX as usize) as u32,
        generated_tokens: u.completion_tokens.min(u32::MAX as usize) as u32,
        // P10.4: no reasoning split from mistral.rs — stays 0.
        reasoning_tokens: 0,
        pp_toks_per_s: f64::from(u.avg_prompt_tok_per_sec),
        tg_toks_per_s: f64::from(u.avg_compl_tok_per_sec),
    }
}

fn response_kind(r: &Response) -> &'static str {
    match r {
        Response::Chunk(_) => "chunk",
        Response::Done(_) => "done",
        Response::ModelError(_, _) => "model-error",
        Response::InternalError(_) => "internal-error",
        Response::ValidationError(_) => "validation-error",
        Response::CompletionDone(_)
        | Response::CompletionChunk(_)
        | Response::CompletionModelError(_, _) => "completion",
        Response::ImageGeneration(_) => "image-generation",
        Response::Speech { .. } => "speech",
        Response::Raw { .. } => "raw",
        Response::Embeddings { .. } => "embeddings",
    }
}

/// Minimal JSON string quoting for volunteered tool calls surfaced as text.
fn quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use mistralrs::{CalledFunction, Choice, ChunkChoice, Delta, ResponseMessage};
    use mistralrs::{ToolCallResponse, ToolCallType};

    fn chunk(
        content: Option<&str>,
        reasoning: Option<&str>,
        finish: Option<&str>,
        usage: Option<MistralUsage>,
    ) -> Response {
        Response::Chunk(ChatCompletionChunkResponse {
            id: "test".into(),
            choices: vec![ChunkChoice {
                finish_reason: finish.map(str::to_owned),
                index: 0,
                delta: Delta {
                    content: content.map(str::to_owned),
                    role: "assistant".into(),
                    tool_calls: None,
                    reasoning_content: reasoning.map(str::to_owned),
                },
                logprobs: None,
            }],
            created: 0,
            model: "test".into(),
            system_fingerprint: String::new(),
            object: "chat.completion.chunk".into(),
            usage,
        })
    }

    fn usage() -> MistralUsage {
        MistralUsage {
            completion_tokens: 7,
            prompt_tokens: 13,
            total_tokens: 20,
            avg_tok_per_sec: 1.0,
            avg_prompt_tok_per_sec: 100.0,
            avg_compl_tok_per_sec: 50.0,
            total_time_sec: 0.2,
            total_prompt_time_sec: 0.13,
            total_completion_time_sec: 0.14,
        }
    }

    fn done(text: &str, reason: &str) -> Response {
        Response::Done(done_resp(text, reason))
    }

    fn done_resp(text: &str, reason: &str) -> ChatCompletionResponse {
        ChatCompletionResponse {
            id: "test".into(),
            choices: vec![Choice {
                finish_reason: reason.into(),
                index: 0,
                message: ResponseMessage {
                    content: Some(text.into()),
                    role: "assistant".into(),
                    tool_calls: None,
                    reasoning_content: None,
                },
                logprobs: None,
            }],
            created: 0,
            model: "test".into(),
            system_fingerprint: String::new(),
            object: "chat.completion".into(),
            usage: usage(),
        }
    }

    fn plain_req() -> GenerateRequest {
        GenerateRequest {
            messages: vec![ChatMessage::user("hi")],
            max_tokens: 16,
            ..GenerateRequest::default()
        }
    }

    fn drain(fold: &mut EventFold) -> Vec<GenEvent> {
        fold.ready.drain(..).collect()
    }

    /// `Result::unwrap_err` needs `T: Debug`, which mistral.rs types lack.
    fn err_msg<T>(r: Result<T, EngineError>) -> String {
        match r {
            Ok(_) => panic!("expected an error"),
            Err(e) => e.to_string(),
        }
    }

    #[test]
    fn load_missing_dir_fails_without_network() {
        let err = err_msg(MistralModel::load(Path::new("/no/such/runa-mistral-dir")));
        assert!(err.contains("config.json"), "{err}");
    }

    #[test]
    fn load_plain_file_fails_fast() {
        let dir = std::env::temp_dir().join(format!("runa-mistral-neg-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("model.gguf");
        std::fs::write(&f, b"gguf").unwrap();
        let err = err_msg(MistralModel::load(&f));
        assert!(err.contains("--backend mistral"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn gguf_only_fields_are_rejected() {
        let mut req = plain_req();
        req.tools = Some("[]".into());
        assert!(err_msg(request_builder(&req)).contains("--mcp"));
        let mut req = plain_req();
        req.json_schema = Some(r#"{"type":"object"}"#.into());
        assert!(err_msg(request_builder(&req)).contains("--json-schema"));
        let mut req = plain_req();
        req.images = vec![crate::vision::VisionFrame {
            t_sec: None,
            source: crate::vision::VisionSource::Path(PathBuf::from("x.png")),
        }];
        assert!(err_msg(request_builder(&req)).contains("--image"));
        let mut req = plain_req();
        req.speculative.ngram = true;
        assert!(err_msg(request_builder(&req)).contains("--ngram"));
    }

    #[test]
    fn tool_role_is_rejected() {
        let mut req = plain_req();
        req.messages.push(ChatMessage {
            role: "tool".into(),
            content: "out".into(),
            ..ChatMessage::default()
        });
        assert!(err_msg(request_builder(&req)).contains("tool"));
    }

    #[test]
    fn plain_request_builds() {
        // Must not error: messages + sampling + think + stop + max_len map.
        let mut req = plain_req();
        req.stop = vec!["</s>".into()];
        assert!(request_builder(&req).is_ok());
    }

    #[test]
    fn stream_chunks_then_done_yield_text_usage_done() {
        let mut fold = EventFold::new(true);
        fold.step(Some(chunk(Some("Hel"), None, None, None)))
            .unwrap();
        fold.step(Some(chunk(Some("lo"), Some("think"), None, None)))
            .unwrap();
        fold.step(Some(chunk(None, None, Some("stop"), Some(usage()))))
            .unwrap();
        assert!(fold.is_finished());
        assert_eq!(
            drain(&mut fold),
            vec![
                GenEvent::Text("Hel".into()),
                GenEvent::Text("lo".into()),
                GenEvent::Reasoning("think".into()),
                GenEvent::Usage(Usage {
                    prompt_tokens: 13,
                    generated_tokens: 7,
                    reasoning_tokens: 0,
                    pp_toks_per_s: 100.0,
                    tg_toks_per_s: 50.0,
                }),
                GenEvent::Done(StopReason::Eos),
            ]
        );
    }

    #[test]
    fn reasoning_hidden_when_show_is_false() {
        let mut fold = EventFold::new(false);
        fold.step(Some(chunk(Some("x"), Some("hidden"), None, None)))
            .unwrap();
        assert!(
            !fold
                .ready
                .iter()
                .any(|e| matches!(e, GenEvent::Reasoning(_)))
        );
    }

    #[test]
    fn length_finish_maps_to_max_tokens() {
        let mut fold = EventFold::new(true);
        fold.step(Some(chunk(None, None, Some("length"), Some(usage()))))
            .unwrap();
        assert!(fold.ready.contains(&GenEvent::Done(StopReason::MaxTokens)));
    }

    #[test]
    fn done_without_chunks_uses_summary_text() {
        let mut fold = EventFold::new(true);
        fold.step(Some(done("full answer", "stop"))).unwrap();
        let texts: Vec<&str> = fold
            .ready
            .iter()
            .filter_map(|e| match e {
                GenEvent::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(texts, ["full answer"]);
    }

    #[test]
    fn done_after_chunks_does_not_repeat_text() {
        let mut fold = EventFold::new(true);
        fold.step(Some(chunk(Some("part"), None, None, None)))
            .unwrap();
        fold.step(Some(done("part (full)", "stop"))).unwrap();
        let texts: Vec<&str> = fold
            .ready
            .iter()
            .filter_map(|e| match e {
                GenEvent::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(texts, ["part"]);
    }

    #[test]
    fn volunteered_tool_calls_surface_as_text() {
        let mut item = chunk(None, None, None, None);
        if let Response::Chunk(c) = &mut item {
            c.choices[0].delta.tool_calls = Some(vec![ToolCallResponse {
                index: 0,
                id: "call-1".into(),
                tp: ToolCallType::Function,
                function: CalledFunction {
                    name: "get_weather".into(),
                    arguments: "{}".into(),
                },
            }]);
        }
        let mut fold = EventFold::new(true);
        fold.step(Some(item)).unwrap();
        let texts: Vec<&str> = fold
            .ready
            .iter()
            .filter_map(|e| match e {
                GenEvent::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(texts, [r#"{"name":"get_weather","arguments":"{}"}"#]);
    }

    #[test]
    fn exhausted_stream_still_closes() {
        let mut fold = EventFold::new(true);
        fold.step(None).unwrap();
        assert!(fold.ready.contains(&GenEvent::Done(StopReason::Eos)));
    }

    #[test]
    fn model_error_is_an_error() {
        let mut fold = EventFold::new(true);
        let err = fold.step(Some(Response::ModelError(
            "boom".into(),
            done_resp("part", "stop"),
        )));
        assert!(err_msg(err).contains("boom"));
    }

    /// Ignored e2e: needs a real safetensors snapshot
    /// (`RUNA_MISTRAL_TEST_DIR` pointing at a dir with `config.json`).
    /// No such fixture ships with the repo, so this never runs in CI.
    #[test]
    #[ignore]
    fn load_safetensors_dir_e2e() {
        let Ok(dir) = std::env::var("RUNA_MISTRAL_TEST_DIR") else {
            eprintln!("skipped: set RUNA_MISTRAL_TEST_DIR to a safetensors snapshot dir");
            return;
        };
        let mut model = MistralModel::load(Path::new(&dir)).unwrap();
        let mut generation = model.generate(plain_req()).unwrap();
        let mut text = String::new();
        for ev in &mut generation {
            if let GenEvent::Text(t) = ev.unwrap() {
                text.push_str(&t);
            }
        }
        assert!(!text.is_empty(), "expected some generated text");
    }
}
