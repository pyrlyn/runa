// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! Local inference backend selection (P9.2).
//!
//! [`BackendKind`] is the `--backend` flag value shared by `run`, `chat`
//! and `serve`. `Auto` inspects the resolved model path: a `.gguf` file
//! loads through the ggml backend (`runa-engine::load`), a directory holding
//! `config.json` loads through mistral.rs (`runa-engine::mistral`, behind
//! the `mistralrs` cargo feature). Anything else is an explicit error, never
//! a silent fallback — a safetensors repo that is not laid out as a model
//! directory must say so.

use std::path::Path;

/// Which local engine loads the model (plan P9.2 `--backend`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BackendKind {
    /// Detect from the model path ([`detect_backend`]).
    #[default]
    Auto,
    /// GGUF file via llama.cpp (`runa-engine::load`).
    Gguf,
    /// Safetensors directory via mistral.rs (`runa-engine::mistral`).
    Mistral,
}

impl BackendKind {
    /// Parse a `--backend` value: `auto` | `gguf` | `mistral`
    /// (case-insensitive, surrounding whitespace ignored).
    pub fn parse(s: &str) -> Option<BackendKind> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" => Some(BackendKind::Auto),
            "gguf" => Some(BackendKind::Gguf),
            "mistral" => Some(BackendKind::Mistral),
            _ => None,
        }
    }

    /// Canonical flag spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            BackendKind::Auto => "auto",
            BackendKind::Gguf => "gguf",
            BackendKind::Mistral => "mistral",
        }
    }
}

impl std::fmt::Display for BackendKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// True for a GGUF model file (extension `.gguf`, case-insensitive).
pub fn is_gguf_file(path: &Path) -> bool {
    path.is_file()
        && path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
}

/// True for a mistral.rs model directory: a directory holding `config.json`
/// (the Hugging Face snapshot layout mistral.rs loads locally).
pub fn is_mistral_dir(path: &Path) -> bool {
    path.is_dir() && path.join("config.json").is_file()
}

/// Detect the backend for an already-resolved model path.
pub fn detect_backend(path: &Path) -> Result<BackendKind, String> {
    if is_gguf_file(path) {
        return Ok(BackendKind::Gguf);
    }
    if is_mistral_dir(path) {
        return Ok(BackendKind::Mistral);
    }
    if !path.exists() {
        return Err(format!(
            "{}: no such model file or directory (pass a .gguf file or a mistral model directory)",
            path.display()
        ));
    }
    if path.is_dir() {
        return Err(format!(
            "{}: directory without config.json is not a mistral model directory \
             (`runa pull hf:<repo>:safetensors` lays one out; pass --backend to override)",
            path.display()
        ));
    }
    Err(format!(
        "{}: cannot detect backend (not a .gguf file); pass --backend gguf|mistral",
        path.display()
    ))
}

/// Resolve a `--backend` flag against a model path: `Auto` detects, an
/// explicit kind is checked against the path so a mismatch fails fast with
/// the reason instead of a loader stack trace.
pub fn resolve_backend(requested: BackendKind, path: &Path) -> Result<BackendKind, String> {
    let kind = match requested {
        BackendKind::Auto => detect_backend(path)?,
        kind => kind,
    };
    match kind {
        BackendKind::Auto => unreachable!("detection always returns a concrete backend"),
        BackendKind::Gguf if is_gguf_file(path) => Ok(BackendKind::Gguf),
        BackendKind::Gguf => Err(format!(
            "{}: --backend gguf needs a .gguf file (this path is not one)",
            path.display()
        )),
        BackendKind::Mistral if is_mistral_dir(path) => Ok(BackendKind::Mistral),
        BackendKind::Mistral => Err(format!(
            "{}: --backend mistral needs a model directory holding config.json \
             (`runa pull hf:<repo>:safetensors` lays one out)",
            path.display()
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("runa-backend-test-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn parse_accepts_all_spellings() {
        assert_eq!(BackendKind::parse("auto"), Some(BackendKind::Auto));
        assert_eq!(BackendKind::parse("GGUF"), Some(BackendKind::Gguf));
        assert_eq!(BackendKind::parse("  mistral "), Some(BackendKind::Mistral));
        assert_eq!(BackendKind::parse("ggml"), None);
        assert_eq!(BackendKind::parse(""), None);
    }

    #[test]
    fn default_is_auto() {
        assert_eq!(BackendKind::default(), BackendKind::Auto);
        assert_eq!(BackendKind::Mistral.as_str(), "mistral");
    }

    #[test]
    fn gguf_file_detects_gguf() {
        let dir = scratch("gguf");
        let f = dir.join("model.GGUF");
        std::fs::write(&f, b"gguf").unwrap();
        assert_eq!(detect_backend(&f).unwrap(), BackendKind::Gguf);
        assert!(is_gguf_file(&f));
        assert!(!is_mistral_dir(&f));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn dir_with_config_detects_mistral() {
        let dir = scratch("mistral").join("weights");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), b"{}").unwrap();
        assert_eq!(detect_backend(&dir).unwrap(), BackendKind::Mistral);
        assert!(is_mistral_dir(&dir));
        let _ = std::fs::remove_dir_all(dir.parent().unwrap());
    }

    #[test]
    fn bare_dir_names_config_json() {
        let dir = scratch("baredir").join("empty");
        std::fs::create_dir_all(&dir).unwrap();
        let err = detect_backend(&dir).unwrap_err();
        assert!(err.contains("config.json"), "{err}");
        let _ = std::fs::remove_dir_all(dir.parent().unwrap());
    }

    #[test]
    fn missing_path_errors() {
        let err = detect_backend(std::path::Path::new("/no/such/runa-model.gguf")).unwrap_err();
        assert!(err.contains("no such model"), "{err}");
    }

    #[test]
    fn explicit_mismatch_fails_fast() {
        let dir = scratch("mismatch");
        let f = dir.join("model.gguf");
        std::fs::write(&f, b"gguf").unwrap();
        let err = resolve_backend(BackendKind::Mistral, &f).unwrap_err();
        assert!(
            err.contains("--backend mistral needs a model directory"),
            "{err}"
        );
        let err = resolve_backend(BackendKind::Gguf, &dir).unwrap_err();
        assert!(err.contains("--backend gguf needs a .gguf file"), "{err}");
        assert_eq!(
            resolve_backend(BackendKind::Gguf, &f).unwrap(),
            BackendKind::Gguf
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
