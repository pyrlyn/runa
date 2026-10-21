// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! The `[store]` table of `runa.toml`.
//!
//! The JSON Schema in `store.schema.json` is generated from [`StoreConfig`]
//! by a test; run `RUNA_BLESS_SCHEMA=1 cargo test -p runa-models` after a
//! change to the type.

use std::num::NonZeroU64;
use std::path::{Path, PathBuf};

use serde::Deserialize;

/// Fresh downloads above this are refused unless `[store] max_bytes` or
/// `RUNA_MAX_MODEL_BYTES` raise it: bigger files usually mean a wrong quant
/// tag, and they do not fit the 3 GiB test-fixture budget.
pub const DEFAULT_MAX_BYTES: u64 = 3 * 1024 * 1024 * 1024;

/// Where models are stored and looked up.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[cfg_attr(test, derive(schemars::JsonSchema))]
#[serde(default, deny_unknown_fields)]
pub struct StoreConfig {
    /// Extra folders searched for model files after the download store, in
    /// order. Each is absolute or starts with `~/`.
    pub dirs: Vec<PathBuf>,
    /// Download store. Absolute or `~/`. Default: `<data dir>/models`.
    pub cache_dir: Option<PathBuf>,
    /// Largest fresh download, in bytes. Default: 3 GiB.
    pub max_bytes: Option<NonZeroU64>,
}

/// A `[store]` value that cannot be used.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ConfigError {
    /// Relative paths are refused: two config files live in different
    /// folders, so "relative to what" has no single answer.
    #[error("[store] {key}: {path:?} must be absolute or start with ~/")]
    Relative { key: &'static str, path: PathBuf },
    /// `~/` was used but neither HOME nor USERPROFILE is set.
    #[error("[store] {key}: {path:?} uses ~/ but HOME is not set")]
    NoHome { key: &'static str, path: PathBuf },
}

/// Expand a leading `~/` and refuse relative paths.
pub(crate) fn resolve_path(
    key: &'static str,
    path: &Path,
    home: Option<&Path>,
) -> Result<PathBuf, ConfigError> {
    if let Ok(rest) = path.strip_prefix("~") {
        let Some(home) = home else {
            return Err(ConfigError::NoHome {
                key,
                path: path.to_path_buf(),
            });
        };
        return Ok(home.join(rest));
    }
    if path.is_absolute() {
        return Ok(path.to_path_buf());
    }
    Err(ConfigError::Relative {
        key,
        path: path.to_path_buf(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(text: &str) -> Result<StoreConfig, toml::de::Error> {
        toml::from_str(text)
    }

    #[test]
    fn empty_table_is_the_default() {
        assert_eq!(parse("").unwrap(), StoreConfig::default());
    }

    #[test]
    fn all_keys_parse() {
        let cfg =
            parse("dirs = [\"/models\", \"~/llm\"]\ncache_dir = \"/var/runa\"\nmax_bytes = 1024\n")
                .unwrap();
        assert_eq!(
            cfg.dirs,
            vec![PathBuf::from("/models"), PathBuf::from("~/llm")]
        );
        assert_eq!(cfg.cache_dir, Some(PathBuf::from("/var/runa")));
        assert_eq!(cfg.max_bytes.map(NonZeroU64::get), Some(1024));
    }

    #[test]
    fn unknown_key_is_refused() {
        let err = parse("dir = \"/models\"\n").unwrap_err().to_string();
        assert!(err.contains("unknown field"), "{err}");
    }

    #[test]
    fn zero_max_bytes_is_refused() {
        assert!(parse("max_bytes = 0\n").is_err());
    }

    #[test]
    fn tilde_expands_against_home() {
        let got = resolve_path("dirs", Path::new("~/llm"), Some(Path::new("/home/u"))).unwrap();
        assert_eq!(got, PathBuf::from("/home/u/llm"));
    }

    #[test]
    fn tilde_without_home_is_an_error() {
        let err = resolve_path("dirs", Path::new("~/llm"), None).unwrap_err();
        assert!(matches!(err, ConfigError::NoHome { .. }));
    }

    #[test]
    fn relative_path_is_refused() {
        let err = resolve_path("cache_dir", Path::new("models"), None).unwrap_err();
        assert_eq!(
            err,
            ConfigError::Relative {
                key: "cache_dir",
                path: PathBuf::from("models")
            }
        );
    }

    #[test]
    fn committed_schema_is_current() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("store.schema.json");
        config_schema::check_schema::<StoreConfig>(
            &path,
            Some("Generated from runa_models::StoreConfig; regenerate with RUNA_BLESS_SCHEMA=1 cargo test -p runa-models"),
            "RUNA_BLESS_SCHEMA",
        )
        .unwrap();
    }
}
