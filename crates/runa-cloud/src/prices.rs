// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! Cloud price table (`docs/prices.toml`, overridable in `~/.config/runa/prices.toml`).

use std::collections::HashMap;
use std::path::PathBuf;
use std::str::FromStr;

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
struct PriceEntry {
    input_per_million: f64,
    output_per_million: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct PriceFile {
    updated: String,
    openai: Option<HashMap<String, PriceEntry>>,
    anthropic: Option<HashMap<String, PriceEntry>>,
}

#[derive(Debug, Clone)]
pub struct PriceTable {
    pub updated: String,
    openai: HashMap<String, PriceEntry>,
    anthropic: HashMap<String, PriceEntry>,
}

impl FromStr for PriceTable {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, String> {
        let file = toml::from_str(text).map_err(|e| e.to_string())?;
        Ok(Self::from_file(file))
    }
}

impl PriceTable {
    pub fn load() -> Self {
        load_from_paths(&candidate_paths())
    }

    fn from_file(file: PriceFile) -> Self {
        PriceTable {
            updated: file.updated,
            openai: file.openai.unwrap_or_default(),
            anthropic: file.anthropic.unwrap_or_default(),
        }
    }

    fn default_embedded() -> Self {
        Self::from_str(
            r#"
updated = "2026-09-08"

[openai.gpt-4o-mini]
input_per_million = 0.15
output_per_million = 0.60

[anthropic.claude-sonnet-5]
input_per_million = 3.00
output_per_million = 15.00
"#,
        )
        .expect("embedded prices")
    }

    pub fn estimate_usd(
        &self,
        provider: &str,
        model: &str,
        input_tokens: u32,
        output_tokens: u32,
    ) -> Option<f64> {
        let entry = match provider {
            "openai" => self.openai.get(model),
            "anthropic" => self.anthropic.get(model),
            _ => None,
        }?;
        let in_m = input_tokens as f64 / 1_000_000.0;
        let out_m = output_tokens as f64 / 1_000_000.0;
        Some(in_m * entry.input_per_million + out_m * entry.output_per_million)
    }

    pub fn cost_line(
        &self,
        provider: &str,
        model: &str,
        input_tokens: u32,
        output_tokens: u32,
    ) -> Option<String> {
        self.estimate_usd(provider, model, input_tokens, output_tokens)
            .map(|usd| {
                format!(
                    "cost: ${usd:.4} ({input_tokens} in + {output_tokens} out tok · prices.toml {updated})",
                    usd = usd,
                    input_tokens = input_tokens,
                    output_tokens = output_tokens,
                    updated = self.updated,
                )
            })
    }
}

/// Later paths win. Shipped `docs/prices.toml` is only a fallback when the
/// process cwd is the repo; `~/.config/runa/prices.toml` overrides it.
fn candidate_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        paths.push(cwd.join("docs/prices.toml"));
    }
    if let Some(config) = dirs_config() {
        paths.push(config.join("prices.toml"));
    }
    paths
}

fn load_from_paths(paths: &[PathBuf]) -> PriceTable {
    let mut table = PriceTable::default_embedded();
    for path in paths {
        if path.is_file()
            && let Ok(text) = std::fs::read_to_string(path)
            && let Ok(file) = toml::from_str::<PriceFile>(&text)
        {
            overlay_prices(&mut table, file);
        }
    }
    table
}

fn overlay_prices(table: &mut PriceTable, file: PriceFile) {
    table.updated = file.updated;
    if let Some(openai) = file.openai {
        table.openai.extend(openai);
    }
    if let Some(anthropic) = file.anthropic {
        table.anthropic.extend(anthropic);
    }
}

fn dirs_config() -> Option<PathBuf> {
    std::env::var_os("XDG_CONFIG_HOME")
        .filter(|s| !s.is_empty())
        .map(|p| PathBuf::from(p).join("runa"))
        .or_else(|| {
            std::env::var_os("HOME")
                .filter(|s| !s.is_empty())
                .or_else(|| std::env::var_os("USERPROFILE").filter(|s| !s.is_empty()))
                .map(|h| PathBuf::from(h).join(".config").join("runa"))
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn embedded_table_matches_fixture_model() {
        let t = PriceTable::default_embedded();
        let usd = t
            .estimate_usd("openai", "gpt-4o-mini", 1_000_000, 1_000_000)
            .unwrap();
        assert!((usd - 0.75).abs() < 1e-9);
    }

    #[test]
    fn cost_line_format() {
        let t = PriceTable::default_embedded();
        let line = t
            .cost_line("anthropic", "claude-sonnet-5", 1000, 500)
            .unwrap();
        assert!(line.contains("cost: $"));
        assert!(line.contains("prices.toml"));
    }

    #[test]
    fn loads_repo_fixture_when_present() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/prices.toml");
        if !root.is_file() {
            return;
        }
        let text = std::fs::read_to_string(root).unwrap();
        let t = PriceTable::from_str(&text).unwrap();
        assert!(t.estimate_usd("openai", "gpt-4o", 1_000_000, 0).is_some());
    }

    #[test]
    fn later_path_overrides_earlier_prices() {
        let dir = std::env::temp_dir().join(format!(
            "runa-prices-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let shipped = dir.join("docs-prices.toml");
        let user = dir.join("user-prices.toml");
        std::fs::write(
            &shipped,
            r#"
updated = "shipped"
[openai.gpt-4o-mini]
input_per_million = 9.0
output_per_million = 9.0
"#,
        )
        .unwrap();
        std::fs::write(
            &user,
            r#"
updated = "user"
[openai.gpt-4o-mini]
input_per_million = 0.15
output_per_million = 0.60
"#,
        )
        .unwrap();
        let t = load_from_paths(&[shipped, user]);
        assert_eq!(t.updated, "user");
        let usd = t
            .estimate_usd("openai", "gpt-4o-mini", 1_000_000, 1_000_000)
            .unwrap();
        assert!((usd - 0.75).abs() < 1e-9);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn later_path_keeps_models_it_does_not_name() {
        let dir = std::env::temp_dir().join(format!(
            "runa-prices-keep-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let shipped = dir.join("docs-prices.toml");
        let user = dir.join("user-prices.toml");
        std::fs::write(
            &shipped,
            r#"
updated = "shipped"
[openai.gpt-4o]
input_per_million = 2.5
output_per_million = 10.0
[openai.gpt-4o-mini]
input_per_million = 9.0
output_per_million = 9.0
"#,
        )
        .unwrap();
        std::fs::write(
            &user,
            r#"
updated = "user"
[openai.gpt-4o-mini]
input_per_million = 0.15
output_per_million = 0.60
"#,
        )
        .unwrap();
        let t = load_from_paths(&[shipped, user]);
        assert_eq!(t.updated, "user");
        assert!(t.estimate_usd("openai", "gpt-4o", 1_000_000, 0).is_some());
        let usd = t
            .estimate_usd("openai", "gpt-4o-mini", 1_000_000, 1_000_000)
            .unwrap();
        assert!((usd - 0.75).abs() < 1e-9);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
