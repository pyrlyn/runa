// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! Local engine dispatch over `--backend` (P9.2).
//!
//! [`LocalEngine`] is either a ggml `LoadedModel` or (with the `mistralrs`
//! cargo feature) a mistral.rs `MistralModel`. Both stream the same
//! [`GenEvent`]s, so `run` / `chat` / `serve` drain them through one path.
//! Backend-specific setup (prompt cache, placement, fit) stays on the ggml
//! side; the mistral side is stateless per request.

use std::path::Path;

use runa_core::BackendKind;
use runa_engine::{
    ChatMessage, EngineError, GenEvent, GenerateRequest, LoadConfig, LoadedModel, Placement, load,
};

/// A loaded local model behind either backend.
pub(crate) enum LocalEngine {
    Gguf(LoadedModel),
    #[cfg(feature = "mistralrs")]
    Mistral(runa_engine::MistralModel),
}

impl LocalEngine {
    /// Load `path` through `kind` (already resolved: never `Auto`).
    pub(crate) fn load(
        kind: BackendKind,
        path: &Path,
        placement: &Placement,
        config: &LoadConfig,
    ) -> Result<LocalEngine, String> {
        runa_engine::ensure_backend_available(kind).map_err(|e| e.to_string())?;
        match kind {
            BackendKind::Gguf => load(path, placement, config)
                .map(LocalEngine::Gguf)
                .map_err(|e| e.to_string()),
            #[cfg(feature = "mistralrs")]
            BackendKind::Mistral => runa_engine::MistralModel::load(path)
                .map(LocalEngine::Mistral)
                .map_err(|e| e.to_string()),
            // Checked by ensure_backend_available above; kept for exhaustiveness.
            #[cfg(not(feature = "mistralrs"))]
            BackendKind::Mistral => Err("mistral backend needs --features mistralrs".into()),
            BackendKind::Auto => Err("internal error: backend was not resolved".into()),
        }
    }

    /// Stream one generation from either backend.
    pub(crate) fn generate(
        &mut self,
        req: GenerateRequest,
    ) -> Result<Box<dyn Iterator<Item = Result<GenEvent, EngineError>> + '_>, String> {
        match self {
            LocalEngine::Gguf(loaded) => loaded
                .generate(req)
                .map(|g| Box::new(g) as Box<dyn Iterator<Item = _> + '_>)
                .map_err(|e| e.to_string()),
            #[cfg(feature = "mistralrs")]
            LocalEngine::Mistral(model) => model
                .generate(req)
                .map(|g| Box::new(g) as Box<dyn Iterator<Item = _> + '_>)
                .map_err(|e| e.to_string()),
        }
    }

    /// Fresh context for the next turn. The mistral backend is stateless per
    /// request, so this is a no-op there.
    pub(crate) fn reset_context(&mut self) -> Result<(), String> {
        match self {
            LocalEngine::Gguf(loaded) => loaded.reset_context().map_err(|e| e.to_string()),
            #[cfg(feature = "mistralrs")]
            LocalEngine::Mistral(_) => Ok(()),
        }
    }

    /// Attach an LMDB prompt cache (ggml only; mistral manages its own KV).
    pub(crate) fn attach_prompt_cache(&mut self, cache: runa_engine::PromptCache) {
        match self {
            LocalEngine::Gguf(loaded) => loaded.attach_prompt_cache(cache),
            #[cfg(feature = "mistralrs")]
            LocalEngine::Mistral(_) => {}
        }
    }

    /// Whether the loaded mmproj accepts PCM audio (ggml only).
    pub(crate) fn supports_native_audio(&self) -> bool {
        match self {
            LocalEngine::Gguf(loaded) => loaded.supports_native_audio(),
            #[cfg(feature = "mistralrs")]
            LocalEngine::Mistral(_) => false,
        }
    }

    /// Drop KV cells between reused-thread requests (P3.9). The mistral
    /// backend is stateless per request, so this is a no-op there.
    pub(crate) fn clear_kv(&mut self) {
        match self {
            LocalEngine::Gguf(loaded) => loaded.clear_kv(),
            #[cfg(feature = "mistralrs")]
            LocalEngine::Mistral(_) => {}
        }
    }

    /// Release the prompt cache, keep the model (D17 / P7.2). The mistral
    /// backend manages its own KV, so this is a no-op there.
    pub(crate) fn on_idle(&mut self) {
        match self {
            LocalEngine::Gguf(loaded) => loaded.on_idle(),
            #[cfg(feature = "mistralrs")]
            LocalEngine::Mistral(_) => {}
        }
    }

    /// Exact history size when a gguf tokenizer is loaded (P11.4).
    /// `None` on the mistral backend (no tokenizer exposed there) —
    /// callers fall back to the chars/4 estimate.
    pub(crate) fn count_history_tokens(&self, messages: &[ChatMessage]) -> Option<u64> {
        match self {
            LocalEngine::Gguf(loaded) => Some(loaded.count_history_tokens(messages)),
            #[cfg(feature = "mistralrs")]
            LocalEngine::Mistral(_) => None,
        }
    }

    /// True for the ggml backend (placement / fit / LoRA apply there only).
    pub(crate) fn is_gguf(&self) -> bool {
        matches!(self, LocalEngine::Gguf(_))
    }
}

/// Resolve an already-parsed `--backend` value against `path`: `Auto`
/// detects, an explicit kind is checked, and a backend the binary lacks
/// fails here with the rebuild pointer.
pub(crate) fn resolve_requested(
    requested: BackendKind,
    path: &Path,
) -> Result<BackendKind, String> {
    let kind = runa_core::resolve_backend(requested, path)?;
    runa_engine::ensure_backend_available(kind).map_err(|e| e.to_string())?;
    Ok(kind)
}
