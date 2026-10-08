// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! In-process local inference (P16.2): the engine wrapper and the model pool
//! behind `runa serve` and `runa daemon`, usable without the `runa` binary.
//!
//! Build a [`ModelPool`] over model paths with a [`Placer`] (use
//! [`fixed_placer`] to skip fit checks), then call [`generate`] or
//! [`generate_stream`]. Each loaded model lives on its own engine thread
//! and speaks [`EngineJob`]; [`ModelPool::attach_engine`] swaps in a custom
//! thread.
//!
//! The request and event types are re-exported from `runa-engine` so a host
//! needs no second dependency to build a request.

pub mod engine;
pub mod pool;

pub use pool::{
    EngineJob, ModelPool, Placer, Streamed, Warmup, fixed_placer, generate, generate_on,
    generate_stream, generate_stream_on,
};
pub use runa_core::BackendKind;
pub use runa_engine::{ChatMessage, GenEvent, GenerateRequest, LoadConfig, Placement, StopReason};
