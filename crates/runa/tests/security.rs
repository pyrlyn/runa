// Copyright (c) 2026 Ivan Tugay
// SPDX-License-Identifier: GPL-3.0-or-later
// Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

//! P6.5 — security and privacy smoke tests.

use std::process::Command;

use assert_cmd::cargo::cargo_bin;

#[test]
fn serve_help_defaults_to_loopback() {
    let out = Command::new(cargo_bin("runa"))
        .args(["serve", "--help"])
        .output()
        .expect("serve --help");
    let help = String::from_utf8_lossy(&out.stdout);
    assert!(
        help.contains("127.0.0.1"),
        "expected loopback default in help"
    );
}

#[test]
fn serve_refuses_non_loopback_without_api_key() {
    let out = Command::new(cargo_bin("runa"))
        .args(["serve", "--host", "0.0.0.0", "--port", "9"])
        .output()
        .expect("serve");
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("--api-key") && err.contains("0.0.0.0"),
        "{err}"
    );
}

#[test]
fn serve_non_loopback_with_api_key_still_needs_a_model() {
    let out = Command::new(cargo_bin("runa"))
        .args(["serve", "--host", "0.0.0.0", "--api-key", "test-key"])
        .output()
        .expect("serve");
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("need at least one model"),
        "key should pass the bind gate: {err}"
    );
    assert!(!err.contains("test-key"), "stderr leaked the key: {err}");
}

#[test]
fn doctor_has_no_telemetry_flag() {
    let out = Command::new(cargo_bin("runa"))
        .args(["doctor", "--help"])
        .output()
        .expect("doctor --help");
    let help = String::from_utf8_lossy(&out.stdout);
    assert!(
        !help.to_ascii_lowercase().contains("telemetry"),
        "doctor should not expose telemetry controls"
    );
}
