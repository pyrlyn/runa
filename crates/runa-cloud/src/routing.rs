// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! Cloud model references: `openai:<model>`, `anthropic:<model>`.

use crate::secrets::Provider;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CloudRef {
    pub provider: Provider,
    pub model: String,
}

impl CloudRef {
    pub fn provider_name(&self) -> &'static str {
        match self.provider {
            Provider::OpenAi => "openai",
            Provider::Anthropic => "anthropic",
        }
    }
}

/// Parse `openai:gpt-4o` or `anthropic:claude-sonnet-5`.
pub fn parse_cloud_ref(s: &str) -> Option<CloudRef> {
    if let Some(model) = s.strip_prefix("openai:") {
        if model.is_empty() {
            return None;
        }
        return Some(CloudRef {
            provider: Provider::OpenAi,
            model: model.to_string(),
        });
    }
    if let Some(model) = s.strip_prefix("anthropic:") {
        if model.is_empty() {
            return None;
        }
        return Some(CloudRef {
            provider: Provider::Anthropic,
            model: model.to_string(),
        });
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_openai_and_anthropic() {
        let o = parse_cloud_ref("openai:gpt-4o-mini").unwrap();
        assert_eq!(o.provider, Provider::OpenAi);
        assert_eq!(o.model, "gpt-4o-mini");
        let a = parse_cloud_ref("anthropic:claude-opus-5").unwrap();
        assert_eq!(a.provider, Provider::Anthropic);
    }

    #[test]
    fn rejects_bare_and_empty() {
        assert!(parse_cloud_ref("gpt-4o").is_none());
        assert!(parse_cloud_ref("openai:").is_none());
    }
}
