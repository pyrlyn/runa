// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! Streaming generation (plan P2.2).
//!
//! [`LoadedModel::generate`] renders chat messages through the model's own
//! Jinja template (llama.cpp side, with generation prompt), falls back to a
//! plain `role: content` prompt when the model ships no template, tokenizes,
//! prefills in `n_batch` chunks, then streams tokens through the
//! [`SamplingConfig`] chain as [`GenEvent::Text`] — with stop-string
//! filtering, EOS / max-token termination and [`Usage`] counters.
//!
//! Template kwargs such as `enable_thinking` arrive with thinking budgets
//! (P3.2); reasoning/text separation happens in the stream parser there.

use std::collections::VecDeque;
use std::time::Instant;

use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::{AddBos, LlamaChatMessage};
use llama_cpp_2::token::LlamaToken;

use crate::load::{EngineError, LoadedModel};
use crate::sampling::SamplingConfig;

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
    pub fn user(content: &str) -> ChatMessage {
        ChatMessage {
            role: "user".to_owned(),
            content: content.to_owned(),
            ..ChatMessage::default()
        }
    }
}

pub use runa_core::ToolCall;

/// What to generate.
#[derive(Debug, Clone)]
pub struct GenerateRequest {
    /// Conversation (template input).
    pub messages: Vec<ChatMessage>,
    /// Sampler chain.
    pub sampling: SamplingConfig,
    /// Hard stop after this many new tokens.
    pub max_tokens: u32,
    /// Stop strings (checked against the decoded text stream).
    pub stop: Vec<String>,
    /// Append the assistant turn opener (default true).
    pub add_generation_prompt: bool,
    /// Thinking mode + whether to emit [`GenEvent::Reasoning`].
    pub think: runa_core::ThinkConfig,
    /// 16 kHz mono PCM for native mtmd audio (P4.3).
    pub audio_pcm: Option<Vec<f32>>,
    /// Still images / video frames for native mtmd vision (P4.6).
    pub images: Vec<crate::vision::VisionFrame>,
    /// N-gram / draft-model speculation (P5.6).
    pub speculative: crate::ngram::Speculative,
    /// The answer must match this JSON Schema (P8.1).
    pub json_schema: Option<String>,
    /// The answer must match this GBNF grammar (P8.1).
    pub grammar: Option<String>,
    /// OpenAI-shape `tools` JSON array the model may call (P8.2).
    pub tools: Option<String>,
    /// `auto` (default) / `required` / `none` (P8.2).
    pub tool_choice: Option<String>,
}

impl Default for GenerateRequest {
    fn default() -> Self {
        GenerateRequest {
            messages: Vec::new(),
            sampling: SamplingConfig::default(),
            max_tokens: 512,
            stop: Vec::new(),
            add_generation_prompt: true,
            think: runa_core::ThinkConfig::default(),
            audio_pcm: None,
            images: Vec::new(),
            speculative: crate::ngram::Speculative::default(),
            json_schema: None,
            grammar: None,
            tools: None,
            tool_choice: None,
        }
    }
}

/// Token/speed counters for one generation.
#[derive(Debug, Clone, PartialEq)]
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
    /// Prompt-processing speed (tok/s).
    pub pp_toks_per_s: f64,
    /// Decode speed (tok/s).
    pub tg_toks_per_s: f64,
}

/// Why generation stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopReason {
    /// Model emitted EOS.
    Eos,
    /// Hit `max_tokens`.
    MaxTokens,
    /// A stop string matched (the match itself is not emitted).
    StopString(String),
}

/// Stream items. `Text` carries decoded pieces (already stop-filtered);
/// `Usage` + `Done` close the stream, in that order.
#[derive(Debug, Clone, PartialEq)]
pub enum GenEvent {
    Text(String),
    /// Thinking / chain-of-thought (P3.2). Omitted when `think.show` is false.
    Reasoning(String),
    /// Calls parsed from the reply of a request with `tools` (P8.2). Such a
    /// request sends its text as one `Text` at the end, markup stripped.
    ToolCalls(Vec<ToolCall>),
    Usage(Usage),
    Done(StopReason),
}

impl LoadedModel {
    pub fn generate(&mut self, mut req: GenerateRequest) -> Result<Generation<'_>, EngineError> {
        let json_schema = req.json_schema.take();
        let grammar = req.grammar.take();
        let tools = req.tools.take();
        let tool_choice = req.tool_choice.take();
        let constrained = json_schema.is_some() || grammar.is_some();
        if constrained {
            // An eager grammar leaves no room for a reasoning block.
            req.think.mode = runa_core::ThinkMode::Off;
        }
        let media = !req.images.is_empty() || req.audio_pcm.is_some();
        if media && tools.is_some() {
            return Err(EngineError::Media(
                "tools cannot be combined with image or audio input".into(),
            ));
        }
        if media {
            let grammar =
                crate::structured::eager_grammar(json_schema.as_deref(), grammar.as_deref())?;
            let n_past = if !req.images.is_empty() {
                if req.audio_pcm.is_some() {
                    return Err(EngineError::Media(
                        "cannot mix native --audio with --image/--video".into(),
                    ));
                }
                self.eval_vision_prompt(&req.messages, &req.images, req.add_generation_prompt)?
            } else {
                let pcm = req.audio_pcm.as_deref().unwrap_or_default();
                if pcm.is_empty() {
                    return Err(EngineError::Media("empty --audio PCM".into()));
                }
                self.eval_audio_prompt(&req.messages, pcm, req.add_generation_prompt)?
            };
            let n_tok = n_past.max(0) as usize;
            let prompt_tokens = vec![LlamaToken::new(0); n_tok.max(1)];
            let mut generation = self.start_generation(
                prompt_tokens,
                req.max_tokens,
                req.stop,
                req.sampling,
                false,
                false,
                req.think,
                Some(n_past),
                req.speculative.ngram,
                req.speculative.draft_n,
            )?;
            if let Some(g) = grammar {
                generation.constrain(&g)?;
            }
            return Ok(generation);
        }
        let (prompt, templated, grammar, tool_reply) = if constrained || tools.is_some() {
            let c = self.render_oaicompat(
                &req.messages,
                req.add_generation_prompt,
                &crate::structured::TemplateInputs {
                    json_schema: json_schema.as_deref(),
                    grammar: grammar.as_deref(),
                    tools: tools.as_deref(),
                    tool_choice: tool_choice.as_deref(),
                    think: req.think,
                },
            )?;
            req.stop.extend(c.stops);
            (c.prompt, c.templated, c.grammar, c.tool_reply)
        } else {
            let prompt = self.render_prompt(&req.messages, req.add_generation_prompt)?;
            (prompt, self.has_template(), None, None)
        };
        let prompt_tokens = self
            .model()
            .str_to_token(
                &prompt,
                if templated {
                    AddBos::Never
                } else {
                    AddBos::Always
                },
            )
            .map_err(|e| EngineError::Tokenize(format!("{e:?}")))?;
        if prompt_tokens.is_empty() {
            return Err(EngineError::Tokenize("empty prompt".into()));
        }
        let mut generation = self.start_generation(
            prompt_tokens,
            req.max_tokens,
            req.stop,
            req.sampling,
            false,
            true,
            req.think,
            None,
            req.speculative.ngram,
            req.speculative.draft_n,
        )?;
        if let Some(g) = grammar {
            generation.constrain(&g)?;
        }
        generation.tool_reply = tool_reply;
        Ok(generation)
    }

    /// llama-bench-style pp/tg: `n_prompt` dummy tokens, then `n_gen` decode
    /// steps. Ignores EOS so the generate count is exact. Does not touch the
    /// prompt cache.
    pub fn bench(&mut self, n_prompt: u32, n_gen: u32) -> Result<Usage, EngineError> {
        if n_prompt == 0 {
            return Err(EngineError::Tokenize("bench n_prompt must be > 0".into()));
        }
        let need = n_prompt.saturating_add(n_gen).saturating_add(1);
        if need > self.config().n_ctx {
            return Err(EngineError::ContextFailed {
                n_ctx: self.config().n_ctx,
                msg: format!("bench pp{n_prompt}+tg{n_gen} needs n_ctx >= {need}"),
            });
        }
        self.reset_context()?;
        let bos = self.model().token_bos();
        let tok = if bos.0 == 0 { LlamaToken::new(1) } else { bos };
        let tokens = vec![tok; n_prompt as usize];
        let sampling = SamplingConfig {
            temperature: 0.0,
            ..SamplingConfig::default()
        };
        let generation = self.start_generation(
            tokens,
            n_gen,
            Vec::new(),
            sampling,
            true,
            false,
            runa_core::ThinkConfig::default(),
            None,
            false,
            0,
        )?;
        let (_, usage, _) = generation.collect_text()?;
        Ok(usage)
    }

    #[allow(clippy::too_many_arguments)]
    fn start_generation(
        &mut self,
        prompt_tokens: Vec<LlamaToken>,
        max_tokens: u32,
        stop: Vec<String>,
        sampling: SamplingConfig,
        ignore_eos: bool,
        use_cache: bool,
        think: runa_core::ThinkConfig,
        prefilled: Option<i32>,
        ngram: bool,
        draft_n: usize,
    ) -> Result<Generation<'_>, EngineError> {
        let n_prompt = u32::try_from(prompt_tokens.len()).unwrap_or(u32::MAX);
        let n_ctx = self.config().n_ctx;
        if n_prompt >= n_ctx {
            return Err(EngineError::ContextExceeded {
                prompt_tokens: n_prompt,
                n_ctx,
            });
        }
        let cache_hit =
            prefilled.is_some() || (use_cache && self.try_restore_prompt(&prompt_tokens));
        if !cache_hit {
            // The prompt prefills from position 0: a previous generation's
            // cells (chat turns, tool rounds) must go first.
            self.clear_kv();
        }
        let n_past = prefilled.unwrap_or(if cache_hit {
            prompt_tokens.len() as i32
        } else {
            0
        });
        let last_idx = if prefilled.is_some() {
            -1
        } else if cache_hit {
            last_prefill_idx(prompt_tokens.len(), self.config().n_batch)
        } else {
            0
        };
        let sampler = sampling.build(self.model().n_vocab());
        let ngram = if ngram && sampling.temperature <= 0.0 {
            let mut cache = crate::ngram::NgramCache::new();
            let hist: Vec<i32> = prompt_tokens.iter().map(|t| t.0).collect();
            cache.ingest(&hist);
            Some((cache, hist))
        } else {
            None
        };
        let remaining = self
            .config()
            .n_ctx
            .saturating_sub(prompt_tokens.len() as u32);
        let budget = runa_core::BudgetClock::from_think(think, remaining);
        let family = self
            .model()
            .chat_template(None)
            .ok()
            .and_then(|t| t.to_str().ok().map(runa_core::ReasonFamily::from_template))
            .unwrap_or(runa_core::ReasonFamily::Auto);
        Ok(Generation {
            loaded: self,
            prompt_tokens,
            sampler: Some(sampler),
            kernel_seed: sampling.seed,
            sampling,
            max_tokens,
            stop,
            prefill_chunk: 0,
            pending: String::new(),
            generated: 0,
            pp_start: None,
            tg_start: if cache_hit {
                Some(Instant::now())
            } else {
                None
            },
            pp_elapsed: 0.0,
            tg_elapsed: 0.0,
            n_past,
            decoder: encoding_rs::UTF_8.new_decoder(),
            last_idx,
            terminal: Vec::new(),
            cache_hit,
            ignore_eos,
            store_cache: use_cache && prefilled.is_none(),
            reason: runa_core::ReasoningParser::new(family),
            show_reasoning: think.show,
            ready: Vec::new(),
            budget,
            reason_counter: runa_core::BudgetClock::new(u32::MAX, 0),
            reasoning_tokens: 0,
            close_tokens: Vec::new(),
            inject: VecDeque::new(),
            injected: false,
            ngram,
            draft_n,
            tool_reply: None,
            raw: String::new(),
        })
    }

    /// Render messages with the model's chat template, or fall back to a
    /// plain prompt when the model ships none.
    pub(crate) fn render_prompt(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
    ) -> Result<String, EngineError> {
        let chat: Result<Vec<LlamaChatMessage>, _> = messages
            .iter()
            .map(|m| LlamaChatMessage::new(m.role.clone(), m.content.clone()))
            .collect();
        let chat = chat.map_err(|e| EngineError::Template(format!("{e:?}")))?;
        match self.model().chat_template(None) {
            Ok(tmpl) => self
                .model()
                .apply_chat_template(&tmpl, &chat, add_generation_prompt)
                .map_err(|e| EngineError::Template(format!("{e:?}"))),
            Err(_) => Ok(messages
                .iter()
                .map(|m| format!("{}: {}\n", m.role, m.content))
                .collect::<String>()),
        }
    }

    fn has_template(&self) -> bool {
        self.model().chat_template(None).is_ok()
    }

    /// Exact token count for chat-history trimming (P11.4): every
    /// message's content through the model's own tokenizer
    /// (`AddBos::Never`) plus a fixed per-message overhead for
    /// chat-template markers (same +4 the chars/4 estimate uses, so
    /// the trim stays conservative on every path). A message that
    /// fails to tokenize falls back to the chars/4 estimate, so
    /// trimming still terminates.
    pub fn count_history_tokens(&self, messages: &[ChatMessage]) -> u64 {
        messages
            .iter()
            .map(|m| {
                self.model()
                    .str_to_token(&m.content, AddBos::Never)
                    .map(|t| t.len() as u64)
                    .unwrap_or_else(|_| m.content.len() as u64 / 4)
                    + 4
            })
            .sum()
    }
}

/// Lazily-driven generation over `&mut LoadedModel`.
pub struct Generation<'m> {
    loaded: &'m mut LoadedModel,
    prompt_tokens: Vec<LlamaToken>,
    sampler: Option<llama_cpp_2::sampling::LlamaSampler>,
    sampling: SamplingConfig,
    kernel_seed: u32,
    max_tokens: u32,
    stop: Vec<String>,
    /// Next prompt chunk to prefill (`None` once prefill is done).
    prefill_chunk: usize,
    /// Decoded tail withheld for stop-string matching.
    pending: String,
    generated: u32,
    pp_start: Option<Instant>,
    tg_start: Option<Instant>,
    pp_elapsed: f64,
    tg_elapsed: f64,
    n_past: i32,
    decoder: encoding_rs::Decoder,
    /// Token count of the last decoded batch minus one: the logits index.
    last_idx: i32,
    /// Queued terminal events (remaining text, Usage, Done).
    terminal: Vec<GenEvent>,
    /// Prefill was skipped because LMDB restored the prompt prefix (P2.8).
    cache_hit: bool,
    /// llama-bench-style: keep decoding past EOS until `max_tokens`.
    ignore_eos: bool,
    /// Write the prompt KV into the LMDB cache when prefill finishes.
    store_cache: bool,
    reason: runa_core::ReasoningParser,
    show_reasoning: bool,
    ready: Vec<GenEvent>,
    budget: Option<runa_core::BudgetClock>,
    /// Unconditional reasoning-token counter (P10.4): a max-budget
    /// `BudgetClock` fed on every observed token, so `Usage` carries
    /// reasoning counts even when thinking is unlimited (`On` /
    /// `Effort::Max`) or off (stays 0).
    reason_counter: runa_core::BudgetClock,
    /// Accumulated `reason_counter.counted()` deltas: unlike the clock
    /// itself (which re-arms per think block), this totals every block.
    reasoning_tokens: u32,
    close_tokens: Vec<LlamaToken>,
    inject: VecDeque<LlamaToken>,
    injected: bool,
    ngram: Option<(crate::ngram::NgramCache, Vec<i32>)>,
    draft_n: usize,
    /// Set for requests with `tools`: text is buffered in `raw` and parsed
    /// once at the end (P8.2).
    tool_reply: Option<crate::structured::ToolReply>,
    raw: String,
}

impl Generation<'_> {
    /// Prompt length in tokens, known before the first event; `Usage` reports
    /// the same number at the end.
    pub fn prompt_len(&self) -> u32 {
        u32::try_from(self.prompt_tokens.len()).unwrap_or(u32::MAX)
    }

    /// Convenience: run to completion, concatenating text.
    pub fn collect_text(mut self) -> Result<(String, Usage, StopReason), EngineError> {
        let mut text = String::new();
        let mut usage = None;
        let mut reason = StopReason::MaxTokens;
        for ev in &mut self {
            match ev? {
                GenEvent::Text(piece) => text.push_str(&piece),
                GenEvent::Reasoning(_) | GenEvent::ToolCalls(_) => {}
                GenEvent::Usage(u) => usage = Some(u),
                GenEvent::Done(r) => reason = r,
            }
        }
        Ok((
            text,
            usage.unwrap_or(Usage {
                prompt_tokens: 0,
                generated_tokens: 0,
                reasoning_tokens: 0,
                pp_toks_per_s: 0.0,
                tg_toks_per_s: 0.0,
            }),
            reason,
        ))
    }

    /// Put a grammar in front of the sampler chain (P8.1). The kernel
    /// sampler and n-gram drafts bypass the chain, so both are turned off.
    fn constrain(&mut self, grammar: &crate::structured::Grammar) -> Result<(), EngineError> {
        let constraint = grammar.sampler(self.loaded)?;
        let base = self.sampler.take().expect("fresh generation has a sampler");
        self.sampler = Some(llama_cpp_2::sampling::LlamaSampler::chain_simple([
            constraint, base,
        ]));
        self.sampling.kernel_sampler = false;
        self.ngram = None;
        Ok(())
    }

    fn batch_size(&self) -> usize {
        let cfg = self.loaded.config();
        let ctx = cfg.n_ctx.max(1);
        cfg.n_batch.max(1).min(ctx) as usize
    }

    fn prefill_done(&self) -> bool {
        self.cache_hit || self.prefill_chunk * self.batch_size() >= self.prompt_tokens.len()
    }

    /// Decode one prompt chunk. Returns true while prefill continues.
    fn prefill_step(&mut self) -> Result<bool, EngineError> {
        let size = self.batch_size();
        let start = self.prefill_chunk * size;
        let end = (start + size).min(self.prompt_tokens.len());
        if self.pp_start.is_none() {
            self.pp_start = Some(Instant::now());
        }
        let mut batch = LlamaBatch::new(end - start, 1);
        for (i, &tok) in self.prompt_tokens[start..end].iter().enumerate() {
            let is_last = i + 1 == end - start;
            batch
                .add(tok, self.n_past + i as i32, &[0], is_last)
                .map_err(|e| EngineError::Decode(format!("{e:?}")))?;
        }
        self.loaded
            .context_mut()
            .decode(&mut batch)
            .map_err(|e| EngineError::Decode(format!("{e:?}")))?;
        self.n_past += (end - start) as i32;
        self.last_idx = (end - start) as i32 - 1;
        self.pp_elapsed = self
            .pp_start
            .map(|t| t.elapsed().as_secs_f64())
            .unwrap_or(0.0);
        self.prefill_chunk += 1;
        if self.prefill_done() {
            if !self.cache_hit && self.store_cache {
                self.loaded.store_prompt(&self.prompt_tokens);
            }
            self.tg_start = Some(Instant::now());
            Ok(false)
        } else {
            Ok(true)
        }
    }

    fn decode_step(&mut self) -> Result<Option<GenEvent>, EngineError> {
        if self.generated >= self.max_tokens {
            self.queue_terminal(StopReason::MaxTokens);
            return Ok(None);
        }
        let in_reason = self.reason.in_reason();
        let holding = self.reason.holding_partial();
        let kind = self
            .budget
            .as_ref()
            .map(|b| b.kind(in_reason, holding))
            .unwrap_or(runa_core::ForceKind::Sample);

        if kind == runa_core::ForceKind::Force {
            self.fill_inject()?;
        }

        let tok = {
            let forced = if kind == runa_core::ForceKind::Force {
                self.inject.pop_front()
            } else {
                None
            };
            if let Some(tok) = forced {
                if let Some(sampler) = self.sampler.as_mut() {
                    sampler.accept(tok);
                }
                tok
            } else {
                let ctx = self.loaded.context_mut();
                if kind == runa_core::ForceKind::Bias
                    && let Some(&t) = self.close_tokens.first()
                {
                    add_close_bias(ctx, self.last_idx, t);
                }
                if self.sampling.kernel_sampler {
                    let logits = ctx.get_logits_ith(self.last_idx);
                    let id = self.sampling.sample_logits(logits, &mut self.kernel_seed);
                    llama_cpp_2::token::LlamaToken::new(id)
                } else {
                    let sampler = self.sampler.as_mut().expect("sampler alive until done");
                    // `sample` accepts the token itself; a second accept
                    // would advance grammars and penalties twice.
                    sampler.sample(ctx, self.last_idx)
                }
            }
        };
        if tok == self.loaded.model().token_eos() && !self.ignore_eos {
            self.queue_terminal(StopReason::Eos);
            return Ok(None);
        }
        self.commit_tokens(tok)
    }

    /// Decode `tok` plus n-gram drafts in one batch; verify drafts at temp 0.
    fn commit_tokens(&mut self, tok: LlamaToken) -> Result<Option<GenEvent>, EngineError> {
        let mut seq = vec![tok];
        if let Some((cache, hist)) = self.ngram.as_ref() {
            let mut h = hist.clone();
            h.push(tok.0);
            let room = self.max_tokens.saturating_sub(self.generated + 1) as usize;
            let cap = self.batch_size().saturating_sub(1);
            for id in cache.draft(&h, room.min(self.draft_n).min(cap)) {
                seq.push(LlamaToken::new(id));
            }
        }
        let mut batch = LlamaBatch::new(seq.len(), 1);
        for (i, &t) in seq.iter().enumerate() {
            batch
                .add(t, self.n_past + i as i32, &[0], true)
                .map_err(|e| EngineError::Decode(format!("{e:?}")))?;
        }
        self.loaded
            .context_mut()
            .decode(&mut batch)
            .map_err(|e| EngineError::Decode(format!("{e:?}")))?;

        let mut accepted = 1usize;
        if seq.len() > 1 {
            let ctx = self.loaded.context();
            for i in 0..seq.len() - 1 {
                let logits = ctx.get_logits_ith(i as i32);
                if crate::ngram::argmax_i32(logits) != seq[i + 1].0 {
                    break;
                }
                accepted += 1;
            }
        }
        if accepted < seq.len() {
            let keep = (self.n_past + accepted as i32) as u32;
            let _ = self
                .loaded
                .context_mut()
                .clear_kv_cache_seq(Some(0), Some(keep), None);
        }
        self.n_past += accepted as i32;
        self.last_idx = (accepted as i32) - 1;
        self.tg_elapsed = self
            .tg_start
            .map(|t| t.elapsed().as_secs_f64())
            .unwrap_or(0.0);

        for &t in &seq[..accepted] {
            if t == self.loaded.model().token_eos() && !self.ignore_eos {
                self.queue_terminal(StopReason::Eos);
                return Ok(None);
            }
            if self.generated >= self.max_tokens {
                self.queue_terminal(StopReason::MaxTokens);
                return Ok(None);
            }
            let piece = self
                .loaded
                .model()
                .token_to_piece(t, &mut self.decoder, true, None)
                .map_err(|e| EngineError::Decode(format!("{e:?}")))?;
            self.generated += 1;
            if let Some((cache, hist)) = self.ngram.as_mut() {
                cache.learn(hist, t.0);
                hist.push(t.0);
            }
            if t != tok
                && let Some(sampler) = self.sampler.as_mut()
            {
                sampler.accept(t);
            }
            self.pending.push_str(&piece);
            match split_emit(&self.pending, &self.stop) {
                Split::Emit { emit, keep } => {
                    self.pending = keep;
                    self.enqueue_parsed(&emit);
                    self.observe_budget();
                }
                Split::Stop { before, matched } => {
                    self.pending = before;
                    self.queue_terminal(StopReason::StopString(matched));
                    return Ok(None);
                }
            }
        }
        Ok(None)
    }

    fn queue_terminal(&mut self, reason: StopReason) {
        let text = std::mem::take(&mut self.pending);
        self.enqueue_parsed(&text);
        for piece in self.reason.flush() {
            self.enqueue_piece(piece);
        }
        if let Some(reply) = self.tool_reply.take() {
            let (text, calls) = reply.parse(&std::mem::take(&mut self.raw));
            if !text.is_empty() {
                self.ready.push(GenEvent::Text(text));
            }
            if !calls.is_empty() {
                self.ready.push(GenEvent::ToolCalls(calls));
            }
        }
        self.ready.append(&mut self.terminal);
        std::mem::swap(&mut self.ready, &mut self.terminal);
        // Drop the sampler: no more sampling after a terminal state.
        self.sampler = None;
        let usage = Usage {
            prompt_tokens: self.prompt_tokens.len() as u32,
            generated_tokens: self.generated,
            reasoning_tokens: self.reasoning_tokens,
            pp_toks_per_s: rate(self.prompt_tokens.len() as u32, self.pp_elapsed),
            tg_toks_per_s: rate(self.generated, self.tg_elapsed),
        };
        self.terminal.push(GenEvent::Usage(usage));
        self.terminal.push(GenEvent::Done(reason));
    }

    fn enqueue_parsed(&mut self, s: &str) {
        if s.is_empty() {
            return;
        }
        if self.tool_reply.is_some() {
            self.raw.push_str(s);
        }
        let pieces = self.reason.push(s);
        for piece in pieces {
            self.enqueue_piece(piece);
        }
    }

    fn enqueue_piece(&mut self, piece: runa_core::ReasonPiece) {
        match piece {
            // Tool replies send their text once, parsed, at the end.
            runa_core::ReasonPiece::Text(s) if !s.is_empty() && self.tool_reply.is_none() => {
                self.ready.push(GenEvent::Text(s));
            }
            runa_core::ReasonPiece::Reasoning(s) if !s.is_empty() && self.show_reasoning => {
                self.ready.push(GenEvent::Reasoning(s));
            }
            _ => {}
        }
    }

    fn observe_budget(&mut self) {
        self.refresh_close_tokens();
        let in_reason = self.reason.in_reason();
        let holding = self.reason.holding_partial();
        if let Some(b) = self.budget.as_mut() {
            b.observe(in_reason, holding);
        }
        // P10.4: the unconditional counter uses the same rule; accumulate
        // deltas so a re-armed clock (new think block) never loses the
        // earlier blocks' total.
        let before = self.reason_counter.counted();
        self.reason_counter.observe(in_reason, holding);
        self.reasoning_tokens = self
            .reasoning_tokens
            .saturating_add(self.reason_counter.counted().saturating_sub(before));
    }

    fn refresh_close_tokens(&mut self) {
        if !self.close_tokens.is_empty() {
            return;
        }
        let Some(close) = self.reason.close_tag() else {
            return;
        };
        if let Ok(toks) = self.loaded.model().str_to_token(close, AddBos::Never) {
            self.close_tokens = toks;
        }
    }

    fn fill_inject(&mut self) -> Result<(), EngineError> {
        if self.injected {
            return Ok(());
        }
        self.refresh_close_tokens();
        let Some(close) = self.reason.close_tag() else {
            return Ok(());
        };
        let mut q = VecDeque::new();
        if let Ok(msg) = self
            .loaded
            .model()
            .str_to_token(runa_core::BUDGET_MESSAGE, AddBos::Never)
        {
            q.extend(msg);
        }
        let close_toks = self
            .loaded
            .model()
            .str_to_token(close, AddBos::Never)
            .map_err(|e| EngineError::Tokenize(format!("{e:?}")))?;
        if close_toks.is_empty() {
            return Err(EngineError::Tokenize(format!(
                "empty tokenize for think-close {close:?}"
            )));
        }
        q.extend(close_toks);
        self.inject = q;
        self.injected = true;
        Ok(())
    }
}

/// Outcome of scanning the pending text against stop strings.
enum Split {
    /// Safe prefix to emit + tail to hold back.
    Emit { emit: String, keep: String },
    /// A stop string matched: text before it + the match (not emitted).
    Stop { before: String, matched: String },
}

fn add_close_bias(ctx: &llama_cpp_2::context::LlamaContext<'_>, idx: i32, token: LlamaToken) {
    let logits = ctx.get_logits_ith(idx);
    let i = token.0 as usize;
    if i < logits.len() {
        // SAFETY: llama_get_logits_ith is a mutable buffer; llama-cpp-2 only
        // exposes `&[f32]`.
        unsafe {
            *logits.as_ptr().cast_mut().add(i) += runa_core::CLOSE_LOGIT_BIAS;
        }
    }
}

fn split_emit(pending: &str, stops: &[String]) -> Split {
    // Earliest full match wins.
    let mut best: Option<(usize, &str)> = None;
    for stop in stops {
        if stop.is_empty() {
            continue;
        }
        if let Some(pos) = pending.find(stop.as_str())
            && best.is_none_or(|(bp, _)| pos < bp)
        {
            best = Some((pos, stop.as_str()));
        }
    }
    if let Some((pos, matched)) = best {
        return Split::Stop {
            before: pending[..pos].to_owned(),
            matched: matched.to_owned(),
        };
    }
    // No full match: hold the longest tail that is a strict prefix of a stop.
    let mut hold = 0;
    for stop in stops {
        if stop.is_empty() {
            continue;
        }
        let max = pending.len().min(stop.len().saturating_sub(1));
        for len in (1..=max).rev() {
            if stop.starts_with(&pending[pending.len() - len..]) {
                hold = hold.max(len);
                break;
            }
        }
    }
    if hold == 0 {
        Split::Emit {
            emit: pending.to_owned(),
            keep: String::new(),
        }
    } else if hold >= pending.len() {
        Split::Emit {
            emit: String::new(),
            keep: pending.to_owned(),
        }
    } else {
        Split::Emit {
            emit: pending[..pending.len() - hold].to_owned(),
            keep: pending[pending.len() - hold..].to_owned(),
        }
    }
}

fn last_prefill_idx(n_tokens: usize, n_batch: u32) -> i32 {
    let size = n_batch.max(1) as usize;
    if n_tokens == 0 {
        return 0;
    }
    let rem = n_tokens % size;
    let chunk = if rem == 0 { size.min(n_tokens) } else { rem };
    chunk as i32 - 1
}

fn rate(tokens: u32, secs: f64) -> f64 {
    if secs > 0.0 {
        tokens as f64 / secs
    } else {
        0.0
    }
}

impl Iterator for Generation<'_> {
    type Item = Result<GenEvent, EngineError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if !self.ready.is_empty() {
                return Some(Ok(self.ready.remove(0)));
            }
            // Queued terminal events (remaining text, Usage, Done) first.
            if !self.terminal.is_empty() {
                return Some(Ok(self.terminal.remove(0)));
            }
            // Sampler dropped by `queue_terminal` + queue drained: over.
            self.sampler.as_ref()?;
            if !self.prefill_done() {
                match self.prefill_step() {
                    Ok(_) => continue,
                    Err(e) => return Some(Err(e)),
                }
            }
            match self.decode_step() {
                Ok(Some(ev)) => return Some(Ok(ev)),
                // Withheld for stop matching: loop back and pull more; a
                // just-queued terminal state is served at the loop top.
                Ok(None) => continue,
                Err(e) => return Some(Err(e)),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_split_full_match_cuts() {
        match split_emit("Paris is great</s> and more", &["</s>".to_owned()]) {
            Split::Stop { before, matched } => {
                assert_eq!(before, "Paris is great");
                assert_eq!(matched, "</s>");
            }
            Split::Emit { .. } => panic!("expected a stop match"),
        }
    }

    #[test]
    fn stop_split_holds_partial_tail() {
        match split_emit("hello wor", &["world".to_owned()]) {
            Split::Emit { emit, keep } => {
                assert_eq!(emit, "hello ");
                assert_eq!(keep, "wor");
            }
            Split::Stop { .. } => panic!("no full match expected"),
        }
    }

    #[test]
    fn stop_split_emits_clean_text() {
        match split_emit("plain text", &["zzz".to_owned()]) {
            Split::Emit { emit, keep } => {
                assert_eq!(emit, "plain text");
                assert!(keep.is_empty());
            }
            Split::Stop { .. } => panic!("no match expected"),
        }
    }

    #[test]
    fn earliest_stop_wins() {
        match split_emit("a STOP1 b STOP2", &["STOP2".to_owned(), "STOP1".to_owned()]) {
            Split::Stop { before, matched } => {
                assert_eq!(before, "a ");
                assert_eq!(matched, "STOP1");
            }
            Split::Emit { .. } => panic!("expected a stop match"),
        }
    }
}
