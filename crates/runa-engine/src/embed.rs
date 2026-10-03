// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! Text embeddings via llama.cpp pooling (plan P6.1 `/v1/embeddings`).

use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::AddBos;

use crate::load::{EngineError, LoadedModel, default_threads, global_backend};

impl LoadedModel {
    /// Embed `text` with mean pooling. Uses a short-lived context with
    /// `embeddings=true` so chat contexts stay unchanged.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, EngineError> {
        let text = text.trim();
        if text.is_empty() {
            return Err(EngineError::Embed("input must be non-empty".into()));
        }
        let backend = global_backend()?;
        let cfg = self.config();
        let threads = cfg.threads.unwrap_or_else(default_threads);
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(cfg.n_ctx.try_into().ok())
            .with_n_batch(cfg.n_batch)
            .with_n_ubatch(cfg.n_ubatch)
            .with_embeddings(true)
            .with_pooling_type(LlamaPoolingType::Mean)
            .with_n_threads(threads)
            .with_n_threads_batch(threads);
        let mut ctx = self.model().new_context(backend, ctx_params).map_err(|e| {
            EngineError::ContextFailed {
                n_ctx: cfg.n_ctx,
                msg: format!("embed context: {e:?}"),
            }
        })?;
        let tokens = self
            .model()
            .str_to_token(text, AddBos::Always)
            .map_err(|e| EngineError::Tokenize(format!("{e:?}")))?;
        if tokens.is_empty() {
            return Err(EngineError::Embed("tokenize produced no tokens".into()));
        }
        let last = tokens.len() as i32 - 1;
        let mut batch = LlamaBatch::new(tokens.len(), 1);
        for (pos, tok) in (0_i32..).zip(tokens) {
            batch
                .add(tok, pos, &[0], pos == last)
                .map_err(|e| EngineError::Decode(format!("batch: {e:?}")))?;
        }
        ctx.decode(&mut batch)
            .map_err(|e| EngineError::Decode(format!("embed decode: {e:?}")))?;
        let emb = ctx
            .embeddings_seq_ith(0)
            .map_err(|e| EngineError::Embed(format!("{e:?}")))?;
        Ok(emb.to_vec())
    }
}
