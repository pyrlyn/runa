// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! runa-models — where model files live (plan P18).
//!
//! [`StoreConfig`] is the `[store]` table of `runa.toml`; [`ModelStore`] turns
//! it into the download store plus the extra folders a user keeps models in,
//! and finds or lists files across them. Like `runa-fit` (D5) it does not
//! depend on the engine: it hands out paths, the engine loads them.
//!
//! Public-method docs: `docs/memory.md`.

mod config;
mod format;
mod paths;
mod store;

pub use config::{ConfigError, DEFAULT_MAX_BYTES, StoreConfig};
pub use format::{FormatError, MAX_SAFETENSORS_HEADER, ModelFormat, detect};
pub use paths::{data_dir, default_cache_dir, home_dir};
pub use store::ModelStore;
