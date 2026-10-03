// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! runa-engine — llama-cpp-2 wrapper: model load with `Placement`,
//! context params, streaming sampling loop, mtmd (audio/image/video),
//! prompt-cache state save (plan §2).
//!
//! P2.1 delivers [`placement`] (cpu/gpu/hybrid) + [`load`] (params,
//! verdict line, loaded model). Sampling loop (P2.2), prompt cache (P2.8)
//! and multi-GPU devices (P2.9) extend this skeleton.
//! Version pin for llama-cpp-2: docs/versions.md (0.1.133 → llama.cpp b7709).

pub mod embed;
pub mod generate;
pub mod load;
pub mod lora;
mod media;
#[cfg(feature = "mistralrs")]
pub mod mistral;
mod ngram;
pub mod placement;
pub mod prompt_cache;
#[cfg(feature = "rpc")]
pub mod rpc;
pub mod sampling;
mod structured;
mod vision;

pub use generate::{
    ChatMessage, GenEvent, GenerateRequest, Generation, StopReason, ToolCall, Usage,
};
pub use load::{EngineError, KvKind, LoadConfig, LoadedModel, load, planner_kv_type};
pub use lora::{DEFAULT_LORA_SCALE, LoraSpec, parse_lora_spec};
#[cfg(feature = "mistralrs")]
pub use mistral::MistralModel;
pub use ngram::{NgramCache, Speculative, argmax_i32};
pub use placement::{
    FFN_EXPS_REGEX, Mode, Placement, cpu_moe_patterns, parse_device_list, parse_rpc_list,
    parse_tensor_split,
};
pub use prompt_cache::PromptCache;
pub use sampling::SamplingConfig;
pub use structured::schema_to_grammar;
pub use vision::{VisionFrame, VisionSource, format_vision_user_text};

/// Fail when `kind` needs a backend this binary was built without (P9.2).
/// The ggml backend is always compiled in; mistral needs `--features
/// mistralrs`. Call after [`runa_core::resolve_backend`] so `--backend auto`
/// has already become concrete.
pub fn ensure_backend_available(
    kind: runa_core::BackendKind,
) -> Result<(), crate::load::EngineError> {
    match kind {
        runa_core::BackendKind::Gguf | runa_core::BackendKind::Auto => Ok(()),
        runa_core::BackendKind::Mistral => {
            #[cfg(feature = "mistralrs")]
            return Ok(());
            #[cfg(not(feature = "mistralrs"))]
            return Err(crate::load::EngineError::Mistral(
                "this runa binary has no mistral backend; rebuild with --features mistralrs".into(),
            ));
        }
    }
}

#[cfg(test)]
mod native_feature_tests {
    #[test]
    fn portable_default_build_is_not_native() {
        assert!(
            !cfg!(feature = "native"),
            "CI / release must use the default feature set so ggml runtime-dispatches AVX-512"
        );
    }
}
