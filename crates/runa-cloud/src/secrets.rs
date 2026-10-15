// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! API keys: environment, OS keychain, never inline in config files (P3.8).
//!
//! The keychain read and the TOML scan come from `keychain-secret` (P17.19).
//! This module keeps runa's provider names, error text, and the
//! `RUNA_NO_KEYRING` switch.

use std::fmt;

use keychain_secret::SecretStore;

/// Cloud provider whose API key we resolve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    OpenAi,
    Anthropic,
}

impl Provider {
    /// `openai` / `anthropic` (case-insensitive).
    pub fn parse(s: &str) -> Result<Self, SecretError> {
        match s.trim().to_ascii_lowercase().as_str() {
            "openai" => Ok(Provider::OpenAi),
            "anthropic" => Ok(Provider::Anthropic),
            other => Err(SecretError::UnknownProvider(other.to_string())),
        }
    }

    /// Environment variable that wins over the keychain.
    pub fn env_var(self) -> &'static str {
        match self {
            Provider::OpenAi => "OPENAI_API_KEY",
            Provider::Anthropic => "ANTHROPIC_API_KEY",
        }
    }

    /// `keyring` user name under service [`KEYRING_SERVICE`].
    pub fn keyring_user(self) -> &'static str {
        match self {
            Provider::OpenAi => "openai",
            Provider::Anthropic => "anthropic",
        }
    }
}

impl fmt::Display for Provider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Provider::OpenAi => "openai",
            Provider::Anthropic => "anthropic",
        })
    }
}

/// OS keychain service id (`keyring::Entry::new`).
pub const KEYRING_SERVICE: &str = "runa";

/// Where a resolved key came from (never log [`ResolvedKey::value`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecretSource {
    Env,
    Keyring,
}

/// API key plus origin. Callers must not print [`Self::value`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedKey {
    pub value: String,
    pub source: SecretSource,
}

impl ResolvedKey {
    /// Last four characters for diagnostics (`sk-…abcd`). Empty → `(empty)`.
    pub fn redacted(&self) -> String {
        keychain_secret::Secret::from(self.value.as_str()).redacted()
    }
}

/// Failures while loading or rejecting secrets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SecretError {
    Missing(Provider),
    UnknownProvider(String),
    InlineConfig { origin: String, key: String },
    InvalidToml { origin: String, message: String },
    Keyring(String),
}

impl fmt::Display for SecretError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SecretError::Missing(p) => write!(
                f,
                "missing API key for {p}; set {} or store it in the OS keychain (service {KEYRING_SERVICE}, user {})",
                p.env_var(),
                p.keyring_user()
            ),
            SecretError::UnknownProvider(s) => {
                write!(f, "{s}: provider must be openai or anthropic")
            }
            SecretError::InlineConfig { origin, key } => write!(
                f,
                "{origin}: inline API key at `{key}` is rejected; set OPENAI_API_KEY / ANTHROPIC_API_KEY or store the key in the OS keychain (service {KEYRING_SERVICE}), not in config files"
            ),
            SecretError::InvalidToml { origin, message } => {
                write!(f, "{origin}: invalid TOML: {message}")
            }
            SecretError::Keyring(s) => write!(f, "keychain: {s}"),
        }
    }
}

impl std::error::Error for SecretError {}

/// Resolve `provider`'s key: non-empty env var, else OS keychain.
///
/// Set `RUNA_NO_KEYRING` to any non-empty value to skip the keychain
/// (tests / headless CI).
pub fn resolve_api_key(provider: Provider) -> Result<ResolvedKey, SecretError> {
    resolve_api_key_from(provider, |k| {
        std::env::var(k).ok().filter(|s| !s.is_empty())
    })
}

/// Same as [`resolve_api_key`] with an injectable env lookup (unit tests).
pub fn resolve_api_key_from(
    provider: Provider,
    getenv: impl Fn(&str) -> Option<String>,
) -> Result<ResolvedKey, SecretError> {
    if let Some(value) = getenv(provider.env_var()) {
        let value = value.trim().to_string();
        if !value.is_empty() {
            return Ok(ResolvedKey {
                value,
                source: SecretSource::Env,
            });
        }
    }
    // Any non-empty value skips the keychain. `keychain_secret::keychain_enabled`
    // treats only off/0/false/no as off, so `RUNA_NO_KEYRING=1` would leave it on.
    if getenv("RUNA_NO_KEYRING").is_some() {
        return Err(SecretError::Missing(provider));
    }
    match keyring_get(provider) {
        Ok(Some(value)) if !value.trim().is_empty() => Ok(ResolvedKey {
            value: value.trim().to_string(),
            source: SecretSource::Keyring,
        }),
        Ok(_) => Err(SecretError::Missing(provider)),
        Err(e) => Err(e),
    }
}

fn keyring_get(provider: Provider) -> Result<Option<String>, SecretError> {
    let store = keychain_secret::Keychain::new(KEYRING_SERVICE);
    match store.get(provider.keyring_user()) {
        Ok(Some(secret)) => Ok(Some(secret.reveal().to_string())),
        Ok(None) => Ok(None),
        Err(keychain_secret::Error::Keychain(message))
            if message.to_ascii_lowercase().contains("no entry") =>
        {
            Ok(None)
        }
        Err(err) => Err(store_error(err)),
    }
}

/// Walk a `runa.toml` / `config.toml` and reject inline API-key fields.
pub fn reject_inline_secrets(text: &str, origin: &str) -> Result<(), SecretError> {
    keychain_secret::reject_inline_secrets(text, origin).map_err(store_error)?;
    // The shared scanner dropped these two names when it grew the suffix
    // rule. They stay rejected here.
    reject_runa_key_names(text, origin)
}

fn reject_runa_key_names(text: &str, origin: &str) -> Result<(), SecretError> {
    let value: toml::Value = toml::from_str(text).map_err(|e| SecretError::InvalidToml {
        origin: origin.to_string(),
        message: e.to_string(),
    })?;
    walk_runa_keys(origin, "", &value)
}

fn walk_runa_keys(origin: &str, path: &str, value: &toml::Value) -> Result<(), SecretError> {
    match value {
        toml::Value::Table(table) => {
            for (k, v) in table {
                let child = if path.is_empty() {
                    k.clone()
                } else {
                    format!("{path}.{k}")
                };
                if is_runa_key_name(k) {
                    return Err(SecretError::InlineConfig {
                        origin: origin.to_string(),
                        key: child,
                    });
                }
                walk_runa_keys(origin, &child, v)?;
            }
            Ok(())
        }
        toml::Value::Array(items) => {
            for (i, v) in items.iter().enumerate() {
                walk_runa_keys(origin, &format!("{path}[{i}]"), v)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn is_runa_key_name(name: &str) -> bool {
    let n = name.to_ascii_lowercase().replace('-', "_");
    matches!(n.as_str(), "openai_key" | "anthropic_key")
}

fn store_error(err: keychain_secret::Error) -> SecretError {
    match err {
        keychain_secret::Error::Inline { origin, key } => SecretError::InlineConfig { origin, key },
        keychain_secret::Error::InvalidToml { origin, message } => {
            SecretError::InvalidToml { origin, message }
        }
        keychain_secret::Error::Keychain(message) => SecretError::Keyring(message),
        keychain_secret::Error::Disabled => {
            SecretError::Keyring("the OS keychain is switched off".to_string())
        }
        other => SecretError::Keyring(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn env(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let map: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect();
        move |k| map.get(k).cloned()
    }

    #[test]
    fn env_openai_wins() {
        let k = resolve_api_key_from(
            Provider::OpenAi,
            env(&[("OPENAI_API_KEY", " sk-test-openai ")]),
        )
        .unwrap();
        assert_eq!(k.value, "sk-test-openai");
        assert_eq!(k.source, SecretSource::Env);
        assert_eq!(k.redacted(), "…enai");
    }

    #[test]
    fn env_anthropic() {
        let k = resolve_api_key_from(
            Provider::Anthropic,
            env(&[("ANTHROPIC_API_KEY", "sk-ant-test")]),
        )
        .unwrap();
        assert_eq!(k.source, SecretSource::Env);
        assert_eq!(k.value, "sk-ant-test");
    }

    #[test]
    fn missing_when_no_env_and_no_keyring() {
        let err =
            resolve_api_key_from(Provider::OpenAi, env(&[("RUNA_NO_KEYRING", "1")])).unwrap_err();
        assert!(matches!(err, SecretError::Missing(Provider::OpenAi)));
        let msg = err.to_string();
        assert!(msg.contains("OPENAI_API_KEY"));
        assert!(msg.contains("keychain"));
    }

    #[test]
    fn empty_env_is_missing() {
        let err = resolve_api_key_from(
            Provider::Anthropic,
            env(&[("ANTHROPIC_API_KEY", "  "), ("RUNA_NO_KEYRING", "1")]),
        )
        .unwrap_err();
        assert!(matches!(err, SecretError::Missing(Provider::Anthropic)));
    }

    #[test]
    fn reject_top_level_openai_key() {
        let err = reject_inline_secrets("openai_api_key = \"sk-live\"\n", "runa.toml").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("runa.toml"));
        assert!(msg.contains("openai_api_key"));
        assert!(msg.contains("rejected"));
        assert!(!msg.contains("sk-live"));
    }

    #[test]
    fn reject_nested_api_key() {
        let err = reject_inline_secrets("[openai]\napi_key = \"sk-live\"\n", "cfg").unwrap_err();
        match err {
            SecretError::InlineConfig { key, .. } => assert_eq!(key, "openai.api_key"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn reject_hyphenated_key() {
        assert!(reject_inline_secrets("anthropic-api-key = \"x\"\n", "x").is_err());
    }

    #[test]
    fn reject_password_and_provider_key_alias() {
        let err = reject_inline_secrets("password = \"x\"\n", "runa.toml").unwrap_err();
        match err {
            SecretError::InlineConfig { key, .. } => assert_eq!(key, "password"),
            other => panic!("{other:?}"),
        }
        let err = reject_inline_secrets("openai-key = \"x\"\n", "runa.toml").unwrap_err();
        match err {
            SecretError::InlineConfig { key, .. } => assert_eq!(key, "openai-key"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn allow_config_without_keys() {
        reject_inline_secrets(
            "[models.qwen]\nsource = \"hf:unsloth/Qwen3-8B-GGUF:Q4_K_M\"\n[think]\nmode = \"on\"\n",
            "runa.toml",
        )
        .unwrap();
    }

    #[test]
    fn provider_parse() {
        assert_eq!(Provider::parse("OpenAI").unwrap(), Provider::OpenAi);
        assert!(Provider::parse("google").is_err());
    }
}
