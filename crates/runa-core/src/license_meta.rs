// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! The workspace `license` field stays `GPL-3.0-or-later`. The royalty-free
//! and commercial choices have no SPDX identifier, and crates.io rejects
//! `LicenseRef` names.

use std::fs;
use std::path::Path;

fn repo_root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read(rel: &str) -> String {
    let path = repo_root().join(rel);
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// Value of `license` inside `[workspace.package]`, comments ignored.
fn workspace_package_license(cargo: &str) -> String {
    let pkg = cargo
        .split_once("[workspace.package]")
        .expect("[workspace.package]")
        .1;
    let body = pkg.split("\n[").next().expect("package body");
    for line in body.lines() {
        let line = line.trim();
        if line.starts_with('#') {
            continue;
        }
        let Some(rest) = line.strip_prefix("license") else {
            continue;
        };
        let rest = rest.trim();
        let rest = rest
            .strip_prefix('=')
            .expect("license assignment")
            .trim()
            .trim_matches('"');
        return rest.to_string();
    }
    panic!("[workspace.package] has no license key");
}

#[test]
fn cargo_license_is_spdx_gpl_and_readme_explains_the_other_two() {
    let cargo = read("Cargo.toml");
    assert_eq!(
        workspace_package_license(&cargo),
        "GPL-3.0-or-later",
        "crates.io cannot name the royalty-free or commercial terms"
    );
    let package = cargo
        .split_once("[workspace.package]")
        .expect("package section")
        .1;
    let package = package.split("\n[").next().expect("package body");
    assert!(
        package.contains("SPDX License List") && package.contains("LicenseRef"),
        "Cargo.toml must say why the other two licences are absent"
    );

    let readme = read("README.md");
    assert!(readme.contains("GNU GPLv3"), "{readme}");
    assert!(readme.contains("Royalty-free License"), "{readme}");
    assert!(readme.contains("[Commercial license]"), "{readme}");
    assert!(
        readme.contains("SPDX License List") && readme.contains("LicenseRef-*"),
        "README must explain the crates.io SPDX limit"
    );
    // The synced blurb stays inside its markers; the explanation follows it.
    let sync_end = readme
        .find("<!-- license-sync:end -->")
        .expect("license-sync end");
    let after = &readme[sync_end..];
    assert!(
        after.contains("GPL-3.0-or-later") && after.contains("license-file"),
        "explanation must sit outside the license-sync block"
    );

    for rel in [
        "docs/getting-started.md",
        "docs/ru/getting-started.md",
        "docs/uk/getting-started.md",
    ] {
        let doc = read(rel);
        assert!(
            doc.contains("GPL-3.0-or-later") && doc.contains("SPDX") && doc.contains("LicenseRef"),
            "{rel} must match the English licence note"
        );
        assert!(
            !doc.contains("MIT OR Apache-2.0"),
            "{rel} still names the wrong licence"
        );
    }
}
