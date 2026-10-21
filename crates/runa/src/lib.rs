// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! `runa` — run AI models locally (GGUF via ggml/llama.cpp) or through the
//! OpenAI and Anthropic APIs.
//!
//! The binary is a thin `main` over [`run`]. Serve and daemon are
//! [`serve::cmd_serve`] and [`daemon::cmd_daemon`], so another program can
//! link this crate without spawning the CLI.
//!
//! `cargo fuzz` builds the fuzz bins with `--cfg fuzzing` and calls
//! `fuzz_hooks::run`.

#![cfg_attr(fuzzing, allow(dead_code, unused_imports))]

mod bench;
pub(crate) mod cli;
mod cloud;
mod cmd;
pub mod daemon;
mod dashboard;
mod fit;
#[cfg(fuzzing)]
pub mod fuzz_hooks;
mod mcp;
mod placement;
pub mod placer;
mod pull;
pub mod serve;
mod session;
mod stats_web;
mod tui;
mod usage_log;

pub(crate) use cli::config;
pub use cmd::run;
pub(crate) use daemon::proto as daemon_proto;
pub(crate) use placement::{
    AutoPlacement, ModeChoice, auto_placement, memory_ceiling_mib, parse_mode_choice,
    preflight_grow, ram_bytes, resolve_model, vram_bytes,
};
pub(crate) use session::json_escape;
