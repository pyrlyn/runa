// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! Per-user directories. Read from the environment on every call, so tests
//! can point them at a temp dir.

use std::path::PathBuf;

/// `$HOME`, or `$USERPROFILE` on Windows when `HOME` is unset.
pub fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .filter(|s| !s.is_empty())
        .or_else(|| std::env::var_os("USERPROFILE").filter(|s| !s.is_empty()))
        .map(PathBuf::from)
}

/// Data root: `$XDG_DATA_HOME/runa`, else `%LOCALAPPDATA%\runa` on Windows,
/// else `~/.local/share/runa`.
pub fn data_dir() -> PathBuf {
    if let Ok(xdg) = std::env::var("XDG_DATA_HOME")
        && !xdg.is_empty()
    {
        return PathBuf::from(xdg).join("runa");
    }
    #[cfg(windows)]
    if let Ok(local) = std::env::var("LOCALAPPDATA")
        && !local.is_empty()
    {
        return PathBuf::from(local).join("runa");
    }
    home_dir()
        .map(|h| h.join(".local").join("share").join("runa"))
        .unwrap_or_else(|| PathBuf::from(".").join("runa"))
}

/// Download store used when `[store] cache_dir` is not set.
pub fn default_cache_dir() -> PathBuf {
    data_dir().join("models")
}
