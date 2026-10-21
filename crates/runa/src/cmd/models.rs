// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! `runa pull` and `runa models`.

use crate::{config, pull};

pub(super) fn cmd_pull(model_ref: &str) -> Result<(), String> {
    let pulled = pull::pull(model_ref)?;
    if pulled.fresh {
        println!(
            "{} ({}/{}; {} bytes)",
            pulled.path.display(),
            pulled.repo,
            pulled.file,
            pulled.size
        );
    } else {
        println!(
            "{} ({}/{}; already present, verified)",
            pulled.path.display(),
            pulled.repo,
            pulled.file
        );
    }
    Ok(())
}

pub(super) fn cmd_models() -> Result<(), String> {
    let aliases = config::load_aliases()?;
    println!("models in {}:", pull::models_dir().display());
    for dir in pull::extra_dirs() {
        println!("  and {}", dir.display());
    }
    for p in pull::list_models() {
        let size = std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
        println!("  {} ({size} bytes)", p.display());
    }
    if !aliases.models.is_empty() {
        println!("aliases:");
        let mut names: Vec<&String> = aliases.models.keys().collect();
        names.sort();
        for n in names {
            println!("  {n} -> {}", aliases.models[n].source);
        }
    }
    Ok(())
}
