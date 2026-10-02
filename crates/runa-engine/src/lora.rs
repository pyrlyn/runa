// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! LoRA adapter specs: `--lora <path>[:scale]` (plan P8.5).
//!
//! One spec names a LoRA GGUF plus the scale [`load`](crate::load) applies
//! with `lora_adapter_set`. The flag is repeatable; every entry is loaded
//! with `lora_adapter_init` and attached to the context in order.

use std::path::PathBuf;

/// Default adapter scale when the flag carries no `:scale` suffix.
pub const DEFAULT_LORA_SCALE: f32 = 1.0;

/// A LoRA adapter file plus the scale it is applied with.
#[derive(Debug, Clone, PartialEq)]
pub struct LoraSpec {
    /// Adapter GGUF path.
    pub path: PathBuf,
    /// `lora_adapter_set` scale.
    pub scale: f32,
}

impl std::fmt::Display for LoraSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.path.display(), self.scale)
    }
}

/// Parse `--lora <path>[:scale]`.
///
/// The split is at the last `:`: when the suffix parses as a finite `f32`
/// it is the scale, otherwise the whole string is the path (scale 1.0).
/// This keeps Windows paths (`C:\…`) and colon-bearing filenames working;
/// a mistyped scale surfaces as a missing-file error naming the full
/// string. Empty input is an error.
pub fn parse_lora_spec(raw: &str) -> Result<LoraSpec, String> {
    let s = raw.trim();
    if s.is_empty() {
        return Err("--lora needs a path".into());
    }
    if let Some((head, tail)) = s.rsplit_once(':')
        && !head.is_empty()
        && let Ok(scale) = tail.parse::<f32>()
        && scale.is_finite()
    {
        return Ok(LoraSpec {
            path: PathBuf::from(head),
            scale,
        });
    }
    Ok(LoraSpec {
        path: PathBuf::from(s),
        scale: DEFAULT_LORA_SCALE,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_path_defaults_to_scale_one() {
        let spec = parse_lora_spec("adapter.gguf").unwrap();
        assert_eq!(spec.path, PathBuf::from("adapter.gguf"));
        assert_eq!(spec.scale, 1.0);
    }

    #[test]
    fn path_with_scale() {
        let spec = parse_lora_spec("adapter.gguf:0.5").unwrap();
        assert_eq!(spec.path, PathBuf::from("adapter.gguf"));
        assert_eq!(spec.scale, 0.5);
    }

    #[test]
    fn non_numeric_suffix_stays_a_path() {
        let spec = parse_lora_spec("my:adapter.gguf").unwrap();
        assert_eq!(spec.path, PathBuf::from("my:adapter.gguf"));
        assert_eq!(spec.scale, 1.0);
    }

    #[test]
    fn non_finite_scale_stays_a_path() {
        for raw in ["adapter.gguf:nan", "adapter.gguf:inf", "adapter.gguf:-inf"] {
            let spec = parse_lora_spec(raw).unwrap();
            assert_eq!(spec.path, PathBuf::from(raw), "{raw}");
            assert_eq!(spec.scale, 1.0, "{raw}");
        }
    }

    #[test]
    fn empty_is_an_error() {
        assert!(parse_lora_spec("").is_err());
        assert!(parse_lora_spec("   ").is_err());
    }

    #[test]
    fn display_round_trips_through_parse() {
        let spec = LoraSpec {
            path: PathBuf::from("a.gguf"),
            scale: 0.25,
        };
        assert_eq!(parse_lora_spec(&spec.to_string()).unwrap(), spec);
    }
}
