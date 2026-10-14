// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! The synthetic GGUF headers are a few hundred bytes and are committed as
//! plain blobs so CI reads them without Git LFS. The repo-wide `*.gguf` LFS
//! rule must not cover them: under it the clean filter turns each into a
//! pointer and every fresh checkout shows them as modified (P11.10).

use std::path::Path;
use std::process::Command;

#[test]
fn synthetic_fixtures_are_not_lfs_filtered() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut files: Vec<String> = std::fs::read_dir(root.join("tests/fixtures"))
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with("synthetic-") && n.ends_with(".gguf"))
        .map(|n| format!("tests/fixtures/{n}"))
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no synthetic fixtures found");
    let out = match Command::new("git")
        .arg("-C")
        .arg(&root)
        .args(["check-attr", "filter", "--"])
        .args(&files)
        .output()
    {
        Ok(out) if out.status.success() => out,
        // Outside a git checkout (a source tarball) there are no attributes.
        _ => return eprintln!("skip: git check-attr unavailable"),
    };
    let text = String::from_utf8_lossy(&out.stdout);
    for line in text.lines() {
        assert!(line.ends_with(": filter: unset"), "{line}");
    }
    assert_eq!(text.lines().count(), files.len());
}
