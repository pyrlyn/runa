// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! runa-core — shared types: [`BackendKind`], [`ThinkConfig`], reasoning
//! parse, [`ToolCall`]. Local vs cloud generation uses concrete engine
//! and cloud types, not a `Backend` trait.

mod backend;
mod reason;
mod think;

pub use backend::{BackendKind, detect_backend, is_gguf_file, is_mistral_dir, resolve_backend};

pub use reason::{ReasonFamily, ReasonPiece, ReasoningParser, parse_stream};
pub use think::{
    BUDGET_MESSAGE, BudgetClock, CLOSE_LOGIT_BIAS, DEFAULT_GRACE, EFFORT_BUDGET_HIGH,
    EFFORT_BUDGET_LOW, EFFORT_BUDGET_MEDIUM, Effort, ForceKind, ThinkConfig, ThinkMode,
    ThinkOverrides, effort_system_hint, parse_budget,
};

/// One function call from a model's reply, local or cloud (P8.2 / P8.3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    /// Call id, echoed back by the `tool` message that answers it.
    pub id: String,
    /// Function name from the request's `tools`.
    pub name: String,
    /// Arguments as JSON text.
    pub arguments: String,
}

/// P17.1: Cargo `license` stays the SPDX GPL id; the README explains the rest.
#[cfg(test)]
mod license_meta;
