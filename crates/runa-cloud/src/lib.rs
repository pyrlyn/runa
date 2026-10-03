// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! runa-cloud — cloud adapters (plan D9): OpenAI through `async-openai`
//! (Responses API, streaming, base_url override), Anthropic through our own
//! `reqwest` + SSE client, price table.
//!
//! Anthropic: P3.6 (`reqwest` + SSE). OpenAI: P3.5 (`async-openai` 0.41.3).

mod anthropic;
mod media;
pub mod openai;
mod prices;
mod routing;
mod secrets;

pub use anthropic::{
    AnthropicClient, AnthropicEvent, AnthropicRequest, ChatTurn, ImageBlock, PdfBlock,
    adaptive_model, build_body, parse_message, parse_sse, thinking_body, tools_from_openai,
};
pub use media::{
    MediaError, MediaLimits, PreparedImage, RawImage, anthropic_image_blocks, anthropic_pdf_block,
    downscale, frames_as_images, openai_image_parts, prepare_images,
};
pub use prices::PriceTable;
pub use routing::{CloudRef, parse_cloud_ref};
pub use secrets::{
    KEYRING_SERVICE, Provider, ResolvedKey, SecretError, SecretSource, reject_inline_secrets,
    resolve_api_key, resolve_api_key_from,
};
