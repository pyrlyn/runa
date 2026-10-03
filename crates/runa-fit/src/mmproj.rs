// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! Locate a sibling mmproj GGUF next to a language model (P4.3 / P4.8).

use std::path::{Path, PathBuf};

/// First `*mmproj*.gguf` in the same directory as `model`, sorted by name.
pub fn sibling_mmproj(model: &Path) -> Option<PathBuf> {
    let dir = model.parent()?;
    let mut found: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension().and_then(|e| e.to_str()) == Some("gguf")
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.to_ascii_lowercase().contains("mmproj"))
        })
        .collect();
    found.sort();
    found.into_iter().next()
}

/// File size, or 0 if missing.
pub fn mmproj_file_bytes(path: &Path) -> u64 {
    std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixture_smolvlm_has_sibling_mmproj() {
        let model = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/SmolVLM-500M-Instruct-Q8_0.gguf");
        if !model.is_file() {
            return;
        }
        let p = sibling_mmproj(&model).expect("mmproj next to SmolVLM");
        assert!(
            p.file_name()
                .unwrap()
                .to_string_lossy()
                .to_ascii_lowercase()
                .contains("mmproj")
        );
        assert!(mmproj_file_bytes(&p) > 1_000);
    }
}
